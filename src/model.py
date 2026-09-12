"""Modulo of model classes
"""

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


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Downsampling operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder's convolutional layers
        self.encoder_1 = DoubleConv(in_channels, 64)
        self.encoder_2 = DoubleConv(64, 128)
        self.encoder_3 = DoubleConv(128, 256)
        self.encoder_4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

class UNet(Encoder):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__(in_channels)

        # Decoder's layers
        self.upsampling_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # 512 from up-conv + 512 from skip connection (encoder_4) = 1024
        self.decoder_1 = DoubleConv(512 + 512, 512)
        
        self.upsampling_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_2 = DoubleConv(256 + 256, 256)
        
        self.upsampling_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_3 = DoubleConv(128 + 128, 128)
        
        self.upsampling_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_4 = DoubleConv(64 + 64, 64)
        
        # Output prediction
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        skip1 = self.encoder_1(x)
        x = self.pool(skip1)
        
        skip2 = self.encoder_2(x)
        x = self.pool(skip2)
        
        skip3 = self.encoder_3(x)
        x = self.pool(skip3)
        
        skip4 = self.encoder_4(x)
        x = self.pool(skip4)
        
        # --- BOTTLENECK ---
        x = self.bottleneck(x)
        
        # --- DECODER ---
        x = self.upsampling_1(x)
        x = torch.cat([x, skip4], dim=1) # Concatenate along channel dimension
        x = self.decoder_1(x)
        
        x = self.upsampling_2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder_2(x)
        
        x = self.upsampling_3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder_3(x)
        
        x = self.upsampling_4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder_4(x)
        
        logits = self.out_conv(x)
        return logits
