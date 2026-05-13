"""
prepare_dataset.py
------------------
NIH Malaria Cell Images veri setini indirir ve
train / val / test klasörlerine böler.

Kullanım:
    python src/prepare_dataset.py

Gereksinim:
    - Kaggle API kurulu ve ~/.kaggle/kaggle.json mevcut olmalı
    - Alternatif: dataset/raw/ klasörüne elle koyulabilir
"""

import os
import shutil
import random
from pathlib import Path
from config import (
    DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, TRAIN_RATIO, VAL_RATIO, SEED
)


def split_dataset(raw_dir: str) -> dict:
    """
    raw_dir içindeki görüntüleri train/val/test olarak böler.
    Oranlar: %70 / %15 / %15
    """
    random.seed(SEED)
    stats = {}

    for cls in CLASSES:
        src = os.path.join(raw_dir, cls)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Klasör bulunamadı: {src}\n"
                f"Lütfen dataset/raw/{cls}/ altına görüntüleri koyun."
            )

        images = [f for f in os.listdir(src)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        random.shuffle(images)

        n      = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train:n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }

        for dest_root, files in splits.items():
            dest = os.path.join(dest_root, cls)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src, f), os.path.join(dest, f))

        stats[cls] = {
            "total": n,
            "train": len(splits[TRAIN_DIR]),
            "val":   len(splits[VAL_DIR]),
            "test":  len(splits[TEST_DIR]),
        }

    return stats


def download_from_kaggle(dest: str) -> None:
    """Kaggle API ile veri setini indirir."""
    import subprocess
    os.makedirs(dest, exist_ok=True)
    print("Kaggle'dan indiriliyor...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "iarunava/cell-images-for-detecting-malaria",
        "-p", dest, "--unzip"
    ], check=True)
    print("✅ İndirme tamamlandı.")


if __name__ == "__main__":
    raw_dir = os.path.join(DATA_DIR, "raw", "cell_images")

    # Ham veri yoksa Kaggle'dan indir
    if not os.path.exists(raw_dir):
        download_from_kaggle(os.path.join(DATA_DIR, "raw"))

    print("Veri seti bölünüyor...")
    stats = split_dataset(raw_dir)

    print("\n📊 Veri Dağılımı:")
    print(f"{'Sınıf':<15} {'Toplam':>8} {'Train':>8} {'Val':>6} {'Test':>6}")
    print("-" * 45)
    for cls, s in stats.items():
        print(f"{cls:<15} {s['total']:>8} {s['train']:>8} {s['val']:>6} {s['test']:>6}")
    print("\n✅ Veri seti hazır!")
