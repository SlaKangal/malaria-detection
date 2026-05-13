"""
dataset.py
----------
PyTorch Dataset sınıfı ve DataLoader fabrika fonksiyonu.
Veri augmentasyon pipeline'larını da içerir.
"""

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, SEED


# ── Augmentasyon / Normalizasyon pipeline'ları ────────────────

# ImageNet istatistikleri (transfer learning için standart)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(batch_size: int = BATCH_SIZE,
                    num_workers: int = 2) -> tuple:
    """
    Train / Val / Test DataLoader'larını döndürür.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORMS)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=EVAL_TRANSFORMS)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=EVAL_TRANSFORMS)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, worker_init_fn=lambda _: torch.manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"  Train: {len(train_dataset):,} görüntü | {len(train_loader)} batch")
    print(f"  Val  : {len(val_dataset):,} görüntü | {len(val_loader)} batch")
    print(f"  Test : {len(test_dataset):,} görüntü | {len(test_loader)} batch")
    print(f"  Sınıflar: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes


def get_single_image_tensor(img_path: str) -> torch.Tensor:
    """
    Tek bir görüntüyü model girdisine uygun tensöre çevirir.
    Grad-CAM ve tahmin scriptleri için kullanılır.
    """
    img = Image.open(img_path).convert("RGB")
    tensor = EVAL_TRANSFORMS(img).unsqueeze(0)   # (1, C, H, W)
    return tensor
