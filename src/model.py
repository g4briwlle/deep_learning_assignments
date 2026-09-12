from typing import Any

from scipy.ndimage import label
from skimage.segmentation import watershed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x 2 of decoder"""
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # indicates how many channels it needs to normalize
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # now both paramethers are out_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResNetEncoder(nn.Module):
    """
    Reusable Residual encoder to attach to different decoders.
    """
    def __init__(self, replace_stride_with_dilation: bool = True):
        super().__init__()

        # pre trained encoder
        # ResNet34 pre trained from imagenet
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, False, True] if replace_stride_with_dilation else None
        )

        # stem: entry Conv, BatchNorm e ReLU (reduces into H/2)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        # first layer: includes max pool and first block (reduces into H/4)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)

        # following resnet layers (H/8, H/16, H/32)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 # Bottleneck

class ResUNet34(ResNetEncoder):
    """
    Uses the skip connections to implement the UNet model in a Residual Network,
    creating a ResUNet.
    """

    def __init__(self, out_channels: int = 1):
        """
        Initializes the ResUNet model, using the ResNet encoder and adding
        the skip connections.

        Args:
            out_channels (int): Number of output channels for prediction. Default is 1.
        """

        # uses the ResNetEconder __init__ implemented earlier
        super().__init__()

        # separating and getting the different parts of the net to use as skip connections:

        # decoder
        # Up 1: from H/32 (512 channels) to H/16
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # in_channels of 512 = 256 that went up + 256 from skip connection 4
        self.conv1 = DoubleConv(256 + 256, 256)

        # Up 2: from H/16 to H/8
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128 + 128, 128)

        # Up 3: from H/8 to H/4
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64 + 64, 64)

        # Up 4: from H/4 to H/2
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)

        # Up 5: last up to go back to standard resolution(H, W)
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(32, 32)

        # final layer to do binary segmentation
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER
        skip1 = self.stem(x)            # channels  64, size  64x64 (for entry 128x128)
        skip2 = self.layer1(skip1)      # channels  64  size  32x32
        skip3 = self.layer2(skip2)      # channels 128, size  16x16
        skip4 = self.layer3(skip3)      # channels 256, size  8x8
        bottleneck = self.layer4(skip4) # channels 512, size  4x4

        # Attaching "UNet" decoder, using skip connections
        # DECODER WITH SKIP CONNECTIONS
        x = self.up1(bottleneck)
        x = torch.cat([x, skip4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.conv5(x)

        logits = self.out_conv(x)
        return logits

# --- Implenting DeepLab with ASPP over the same encoder ---------------
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # 1. 1x1 Convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 2. Atrous Convolutions (Rates 6, 12, 18 as per DeepLabv3)
        self.conv_r6 = self._build_atrous_conv(in_channels, out_channels, rate=6)
        self.conv_r12 = self._build_atrous_conv(in_channels, out_channels, rate=12)
        self.conv_r18 = self._build_atrous_conv(in_channels, out_channels, rate=18)

        # 3. Image Pooling (Global Average Pooling)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 4. Final Projection
        # 5 branches * 256 channels = 1280 input channels
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _build_atrous_conv(self, in_channels, out_channels, rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Pass through parallel branches
        feat1 = self.conv1x1(x)
        feat2 = self.conv_r6(x)
        feat3 = self.conv_r12(x)
        feat4 = self.conv_r18(x)

        # Image pooling needs upsampling back to feature map size
        pool_feat = self.image_pool(x)
        pool_feat = F.interpolate(pool_feat, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate and project
        out = torch.cat([feat1, feat2, feat3, feat4, pool_feat], dim=1)
        return self.project(out)

class DeepLabV3_ResNet34(ResNetEncoder):
    def __init__(self, out_channels: int = 1):
        super().__init__(replace_stride_with_dilation=True)

        # ASPP Module that uses different rates
        self.aspp = ASPPModule(in_channels=512, out_channels=256)
        self.classifier = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Keeping the ResNet encoders
        skip1 = self.stem(x)
        skip2 = self.layer1(skip1)
        skip3 = self.layer2(skip2)
        skip4 = self.layer3(skip3)
        bottleneck = self.layer4(skip4) 
        
        # Using aspp in the upsampling
        aspp_out = self.aspp(bottleneck)
        logits = self.classifier(aspp_out)
        
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)