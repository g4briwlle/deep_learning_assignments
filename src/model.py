from typing import Any

from scipy.ndimage import label
from skimage.segmentation import watershed

import torch
import torch.nn as nn
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
    def __init__(self):
        super().__init__()

        # pre trained encoder
        # ResNet34 pre trained from imagenet
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

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

    

