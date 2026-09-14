import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2 # uv add opencv-python
from skimage.segmentation import watershed
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import models

from typing import Literal
from pathlib import Path

from src.model import (
    UNet,
    SegNetAblation
)
from src.train import (
    train_part_1_1,
    train_model_track_a
)

# --- Initializing device ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device initialized: {device}")

saved_models_base_path = Path(__file__).resolve().parent


class ModelSelector:
    model_names_list = [
        'unet_part1',
        'unet_part2',
        'resnet_part3',
        'unet_final',
    ]
    model_name_literal = Literal[
        'unet_part1',
        'unet_part2',
        'resnet_part3',
        'unet_final',
    ]

    def _set_model(self):
        self.model_state_dict_path = None
        self.model = None
        self.training_function = None
        self.criterion = None
        self.optimizer = None

        print("=" * 50)
        print(f"Instantiating model {self.model_name}")

        if self.model_name == 'unet_part1':
            # path to state_dict
            self.model_state_dict_path = saved_models_base_path / 'unet_parte1.pth'

            # create empty model with the same architecture
            self.model = UNet(out_channels=1).to(device)

            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            self.training_function

            print("Model architecture:")
            print("|- UNet")
            print("|- 3 input channels")
            print("|- 1 output channel")
            print("|- Loss function: BCEWithLogitsLoss")
            print("|- Optimizer: Adam, Learning Rate: 1e-3")

        elif self.model_name == 'unet_part2':
            self.model_state_dict_path = saved_models_base_path / 'unet_trilhaA_parte2.pth'

            self.model = UNet(in_channels=3, out_channels=3).to(device)

            # Wheighted loss function in training
            class_weights = torch.tensor([0.1, 0.3, 0.9], dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            print("Model architecture:")
            print("|- UNet")
            print("|- 3 input channels")
            print("|- 3 output channels")
            print("|- Loss function: Weighted CE, weights [0.1, 0.3, 0.9]")
            print("|- Optimizer: Adam, Learning Rate: 1e-3")
            print("|- Instance segmentation strategy: Watershed")
        
        elif self.model_name == 'resnet_part3':
            self.model_state_dict_path = saved_models_base_path / 'segnet_parte3_eixo3.pth'

            self.model = SegNetAblation(in_channels=3, out_channels=3).to(device)

            # Keeping the same optimizer, criterion and data as part 2
            class_weights = torch.tensor([0.1, 0.3, 0.9], dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            print("Model architecture:")
            print("|- SegNet Ablation: SegNet with the same encoder as the UNet")
            print("|- 3 input channels")
            print("|- 3 output channels")
            print("|- Loss function: Weighted CE, weights [0.1, 0.3, 0.9]")
            print("|- Optimizer: Adam, Learning Rate: 1e-3")
            print("|- Instance segmentation strategy: Watershed")

        elif self.model_name == 'unet_final':
            raise NotImplementedError("Model of part 5 not implemented yet")

        print(f"Loading weights from file: {self.model_state_dict_path.name}")

        # input memory of saved weigths
        # self.model.load_state_dict(torch.load(self.model_state_dict_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load(self.model_state_dict_path, weights_only=True))
        print(f"Succesfully recovered model {self.model_name}")

    def __init__(self, model_name: model_name_literal = 'unet_final'):
        if model_name not in self.model_names_list:
            raise ValueError(f"Invalid 'model_name'. Expected one of {self.model_names_list}. Got: {model_name}")

        self.model_name = model_name
        print(f"Model selected: {self.model_name}")

        self._set_model()

    
    def get_model(self) -> nn.Module:
        return self.model

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1]

    ModelSelector(model_name)