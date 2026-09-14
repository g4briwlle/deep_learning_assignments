import numpy as np
from scipy.ndimage import label
import cv2  # uv add opencv-python
from skimage.segmentation import watershed

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def resize_and_pad(img, size, is_mask=False):
    """
    Resize keeping aspect ratio so the longer side == size, then pad with zeros
    to a (size, size) square. Returns the padded array and a bool valid_mask
    (True where the original image lives, False on the pad).

    img: HWC (image) or HW (mask). Works for both.
    """
    H, W = img.shape[:2]
    scale = size / max(H, W)
    new_w, new_h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if img.ndim == 3:
        out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        out[:new_h, :new_w] = resized
    else:
        out = np.zeros((size, size), dtype=img.dtype)
        out[:new_h, :new_w] = resized

    valid = np.zeros((size, size), dtype=bool)
    valid[:new_h, :new_w] = True

    return out, valid

DATA_ROOT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'stage1_train'


class DSB2018DatasetTrackA(Dataset):
    transform = A.Compose([
        A.ToGray(p=1.0), 
        A.InvertImg(p=0.5), 
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    ])
    
    def __init__(self, root_dir: Path = DATA_ROOT_DIR, size: int = 128, border_thickness: int = 2, transform_images: bool = False):
        self.root_dir = root_dir
        self.size = size
        self.border_thickness = border_thickness
        self.transform_images = transform_images

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
        
        # Apply the modality shift augmentations
        if self.transform_images:
            augmented = self.transform(image=img)
            img = augmented["image"]
        
        img, _ = resize_and_pad(img, self.size, is_mask=False)
        H, W = img.shape[:2]

        img = img.astype(np.float32) / 255.0

        # 2. Initialize the masks
        instance_mask = np.zeros((H, W), dtype=np.int32)
        semantic_mask = np.zeros((H, W), dtype=np.int32)

        # 3. Process individual ground truth masks
        mask_paths = list((sample_dir / "masks").glob("*.png"))

        for inst_id, mask_path in enumerate(mask_paths, start=1):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            mask, _ = resize_and_pad(mask, self.size, is_mask=True)

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


# --- Otimizing data loading --------------------------------------------
# Function to run once and create three .npy files with everything needed
def build_cache(
    root_dir: Path = DATA_ROOT_DIR,
    size: int = 128,
    border_thickness: int = 2,
    out_dir="cache",
    transform_images: bool = False,
):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    base = DSB2018DatasetTrackA(
        root_dir,
        size=size,
        border_thickness=border_thickness,
        transform_images=transform_images
    )

    N = len(base)
    x0, _, _ = base[0]
    _, H, W = x0.shape # probe
    print(f"caching {N} samples at {H}x{W}")

    imgs = np.empty((N, H, W, 3), dtype=np.uint8)
    sem  = np.empty((N, H, W),    dtype=np.uint8)
    inst = np.empty((N, H, W),    dtype=np.int32)

    for i in range(N):
        x, s, m = base[i]
        imgs[i] = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sem[i]  = s.numpy().astype(np.uint8)
        inst[i] = m.numpy()

    np.save(out_dir / "images.npy",   imgs)
    np.save(out_dir / "semantic.npy", sem)
    np.save(out_dir / "instance.npy", inst)
    print(f"{(imgs.nbytes + sem.nbytes + inst.nbytes) / 1e6:.1f} MB written to {out_dir}")

class DSB2018Cached(Dataset):
    def __init__(self, cache_dir="cache"):
        self.images = np.load(Path(cache_dir) / "images.npy", mmap_mode="r")
        self.semantics = np.load(Path(cache_dir) / "semantic.npy", mmap_mode="r")
        self.instances = np.load(Path(cache_dir) / "instance.npy", mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float().div_(255.0) # HWC uint8 -> CHW float
        sem  = torch.from_numpy(self.semantics[idx].astype(np.int64))
        inst = torch.from_numpy(self.instances[idx])
        return img, sem, inst

def get_train_test_dataloaders(
    cache_dir: Path,
    data_images_size: int = 256,
    batch_size: int = 16,
    use_cache: bool = False,
    transform_images: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds (if asked to) the cache with the asked images size, loads it into
    memory with mmap and returns the optimized dataloaders.
    
    Args:
        data_images_size (int): Size of the images images. Get's passed to the `size` of the dataset, that resizes the images with cv2. If you want to use the original image, do not set. Default is None.
        batch_size (int): Batch size of the dataloaders.
        use_cache (bool): If True, do not build the cache, using the existing one. Use this if you're running this function more than once with the `data_images_size`.
    
    Returns:
        Tuple[DataLoader, DataLoader]: The optimized train and test dataloaders with numpy's mmap, respectively.
    """

    import warnings
    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable",
        category=UserWarning,
    )    
    
    if not use_cache:
        build_cache(size=data_images_size, out_dir=cache_dir, transform_images=transform_images)
        
    full_dataset = DSB2018Cached()

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    get_train_test_dataloaders('cache_128', 128)
    get_train_test_dataloaders('cache_256', 256)
    get_train_test_dataloaders('cache_256_transform', 256, transform_images=True)