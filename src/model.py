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
    def __init__(self, in_channels: int = 1):
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
    """
    Inherits the encoder and implements the UNet decoder, using skip connections
    in the decoding transpose convolutions.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
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

# --- Implementing the max unpooling of SegNet -------------------------
class SegNetAblation(Encoder):
    """
    Keeps the same encoder as the UNet and decodes using max unpooling.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__(in_channels)

        # Override the inherited pool to return indices
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # Unpooling layer
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        # Channel reduction convolutions (1x1) to match UNet concatenations
        self.decoder_1 = DoubleConv(512, 256)
        self.decoder_2 = DoubleConv(256, 128)
        self.decoder_3 = DoubleConv(128, 64)
        self.decoder_4 = DoubleConv(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        x = self.encoder_1(x)
        x, idx1 = self.pool(x)
        
        x = self.encoder_2(x)
        x, idx2 = self.pool(x)
        
        x = self.encoder_3(x)
        x, idx3 = self.pool(x)
        
        x = self.encoder_4(x)
        x, idx4 = self.pool(x)
        
        # --- DECODER ---
        x = self.unpool(x, idx4)
        x = self.decoder_1(x)
        
        x = self.unpool(x, idx3)
        x = self.decoder_2(x)
        
        x = self.unpool(x, idx2)
        x = self.decoder_3(x)
        
        x = self.unpool(x, idx1)
        x = self.decoder_4(x)
        
        return self.out_conv(x)

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn

    from .dataset import get_train_test_dataloaders
    from .train import train_model_track_a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting up SegNet ablation model

    # Model with 3 channels for the classes we need to predict:
    # Class 0 - background
    # Class 1 - inside
    # Class 2 - border
    model_part3_axis1 = SegNetAblation(in_channels=3, out_channels=3).to(device)

    # Keeping the same optimizer, criterion and data as part 2
    class_weights = torch.tensor([0.1, 0.3, 0.9], dtype=torch.float32).to(device)

    criterion_part3_axis1 = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_part3_axis1 = optim.Adam(model_part3_axis1.parameters(), lr=1e-3)

    # Data prepping
    # Using new class for 3-classes data
    # train_dataset_part3_axis1 = SyntheticEllipseDatasetTrackA(n_samples=400, size=128)
    # val_dataset_part3_axis1 = SyntheticEllipseDatasetTrackA(n_samples=100, size=128)

    # train_loader_part3_axis1 = DataLoader(train_dataset_part3_axis1, batch_size=16, shuffle=True)
    # val_loader_part3_axis1 = DataLoader(val_dataset_part3_axis1, batch_size=16, shuffle=False)

    train_loader_part3_axis1, test_loader_part3_axis1 = get_train_test_dataloaders(128)

    epochs = 5

    train_model_track_a(
        model_part3_axis1,
        optimizer_part3_axis1,
        criterion_part3_axis1,
        train_loader_part3_axis1,
        test_loader_part3_axis1,
        device,
        epochs,
        'segnet_parte3_eixo3'
    )