import numpy as np
from scipy.ndimage import label
import cv2  # uv add opencv-python
from skimage.segmentation import watershed

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from pathlib import Path
from typing import Tuple


class SyntheticEllipseDataset(Dataset):
    def __init__(self, n_samples=500, size=128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # colorful canvas base with random background (3d)
        bg_color = np.random.uniform(low=10, high=60, size=(3,))
        img = np.full(shape=(self.size, self.size, 3), fill_value=bg_color, dtype=np.float32)

        # initiate instance mask (2d)
        instance_mask = np.zeros(shape=(self.size, self.size), dtype=np.float32)

        n_ellipses = np.random.randint(5, 21)
        centers = []

        for instance_id in range(1, n_ellipses + 1):
            # see if this ellipse will touch others and set center
            if centers and np.random.rand() > 0.5:
                # chooses one random ellipse center to touch
                center_to_touch = centers[np.random.randint(len(centers))]
                cx = int(np.clip(center_to_touch[0] + np.random.randint(-20, 20), 10, self.size - 10))
                cy = int(np.clip(center_to_touch[0] + np.random.randint(-20, 20), 10, self.size - 10))
            else:
                cx = np.random.randint(10, self.size - 10)
                cy = np.random.randint(10, self.size - 10)

            # add center to center list
            centers.append((cx, cy))

            # set ellipse major and minor axes
            axes = (np.random.randint(5, 25), np.random.randint(5, 25))
            # set angle of ellipse turn
            angle = np.random.randint(0, 180)

            # set random ellipse color
            fg_color = tuple(np.random.uniform(100, 255, size=(3,)).tolist())

            # draw colorful ellipse and set its instance in instance matrix
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color=fg_color, thickness=-1)  # thickness=-1 fills inside of ellipse
            cv2.ellipse(instance_mask, (cx, cy), axes, angle, 0, 360, color=int(instance_id), thickness=-1)

        # adds gaussian noise in (H, W, 3)
        noise = np.random.normal(0, np.random.uniform(5, 20), (self.size, self.size, 3))
        img = np.clip(img + noise, 0, 255) / 255.0

        # converts into pytorch tensors
        # OpenCV/NumPy (H, W, C) -> PyTorch precisa de (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Shape final: (3, 128, 128)
        mask_tensor = torch.tensor(instance_mask, dtype=torch.long)           # Shape final: (128, 128)

        return img_tensor, mask_tensor


class SyntheticEllipseDatasetTrackA(Dataset):
    def __init__(self, n_samples=500, size=128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # background
        bg_color = np.random.uniform(10, 60, size=(3,))
        img = np.full((self.size, self.size, 3), bg_color, dtype=np.float32)

        # instance mask
        instance_mask = np.zeros((self.size, self.size), dtype=np.int32)

        # 3 class semantic mask
        # 0 = bg, 1 = inside, 2 = border
        semantic_mask = np.zeros((self.size, self.size), dtype=np.int32)

        n_ellipses = np.random.randint(5, 21)
        centers = []

        for inst_id in range(1, n_ellipses + 1):
            # forces ellipses to touch
            if centers and np.random.rand() > 0.5:
                ref_center = centers[np.random.randint(len(centers))]
                cx = int(np.clip(ref_center[0] + np.random.randint(-20, 20), 10, self.size - 10))
                cy = int(np.clip(ref_center[1] + np.random.randint(-20, 20), 10, self.size - 10))
            else:
                cx = np.random.randint(15, self.size - 15)
                cy = np.random.randint(15, self.size - 15)

            centers.append((cx, cy))

            axes = (np.random.randint(6, 22), np.random.randint(6, 22))
            angle = np.random.randint(0, 180)
            fg_color = tuple(np.random.uniform(100, 255, size=(3,)).tolist())

            # draw colorful ellipse
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color=fg_color, thickness=-1)

            # draw id in instance mask
            cv2.ellipse(instance_mask, (cx, cy), axes, angle, 0, 360, color=int(inst_id), thickness=-1)

            # draw 3 class semantic mask
            # fill inside with 1
            cv2.ellipse(semantic_mask, (cx, cy), axes, angle, 0, 360, color=1, thickness=-1)
            # draw border line with 2
            cv2.ellipse(semantic_mask, (cx, cy), axes, angle, 0, 360, color=2, thickness=2)

        # add noises and convert to tensor
        noise = np.random.normal(0, np.random.uniform(5, 20), (self.size, self.size, 3))
        img = np.clip(img + noise, 0, 255) / 255.0

        # final shapes
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # [3, 128, 128]
        semantic_tensor = torch.tensor(semantic_mask, dtype=torch.long)       # [128, 128]
        instance_tensor = torch.tensor(instance_mask, dtype=torch.int32)      # [128, 128]

        return img_tensor, semantic_tensor, instance_tensor


DATA_ROOT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'stage1_train'


class DSB2018DatasetTrackA(Dataset):
    def __init__(self, root_dir: Path = DATA_ROOT_DIR, size: int = 128, border_thickness: int = 2, apply_resize: bool = True):
        self.root_dir = root_dir
        self.size = size
        self.border_thickness = border_thickness
        self.apply_resize = apply_resize

        # Gather all subdirectories (each represents one sample)
        self.sample_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # 1. Load and format the RGB Image
        img_path = list((sample_dir / "images").glob("*.png"))[0]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # type: ignore
        
        if self.apply_resize:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0

        # 2. Initialize the masks
        instance_mask = np.zeros((self.size, self.size), dtype=np.int32)
        semantic_mask = np.zeros((self.size, self.size), dtype=np.int32)

        # 3. Process individual ground truth masks
        mask_paths = list((sample_dir / "masks").glob("*.png"))

        for inst_id, mask_path in enumerate(mask_paths, start=1):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if self.apply_resize:
                mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST) # type: ignore

            # Binarize to ensure strict 0 or 255 values
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # type: ignore

            if binary_mask.max() == 0:
                continue

            # Write instance ID
            instance_mask[binary_mask > 0] = inst_id

            # Extract borders using morphological erosion
            kernel = np.ones((self.border_thickness, self.border_thickness), np.uint8)
            erosion = cv2.erode(binary_mask, kernel, iterations=1)
            border = binary_mask - erosion

            # Write to 3-class semantic mask
            semantic_mask[binary_mask > 0] = 1  # 1 = Inside
            semantic_mask[border > 0] = 2       # 2 = Border (overwrites edges)

        # 4. Convert to PyTorch Tensors
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        semantic_tensor = torch.tensor(semantic_mask, dtype=torch.long)
        instance_tensor = torch.tensor(instance_mask, dtype=torch.int32)

        return img_tensor, semantic_tensor, instance_tensor


def get_train_test_dataloaders(size: int | None = 128) -> Tuple[DataLoader, DataLoader]:
    if not size:
        full_dataset = DSB2018DatasetTrackA(apply_resize=False)
    else:
        full_dataset = DSB2018DatasetTrackA(size=size)

    # Calculating split size
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    # Performing a reproducible random split
    train_dataset, test_dataset = random_split(
        dataset=full_dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader
