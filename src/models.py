"""
models.py
---------
İki model tanımı:
  1. CustomCNN   — sıfırdan tasarlanmış küçük CNN
  2. get_efficientnet — EfficientNetB0 transfer learning versiyonu

Hem eğitim hem değerlendirme scriptleri bu modüldeki fonksiyonları kullanır.
"""

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, IMG_SIZE


# ─────────────────────────────────────────────────────────────
# 1. Custom CNN
# ─────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """
    3 Konvolüsyon bloğu + 2 tam bağlantılı katmandan oluşan
    basit ama etkin bir CNN mimarisi.

    Blok yapısı:
        Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Blok 1: 3 → 32 kanal
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112
            nn.Dropout2d(0.1),

            # Blok 2: 32 → 64 kanal
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56
            nn.Dropout2d(0.2),

            # Blok 3: 64 → 128 kanal
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
            nn.Dropout2d(0.2),
        )

        # Global Average Pooling — parametre sayısını azaltır
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # (B, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────
# 2. EfficientNetB0 — Transfer Learning
# ─────────────────────────────────────────────────────────────

def get_efficientnet(num_classes: int = NUM_CLASSES,
                     freeze_base: bool = True) -> nn.Module:
    """
    ImageNet ağırlıklarıyla EfficientNetB0 yükler.
    Son sınıflandırma başlığını binary classification için değiştirir.

    Args:
        freeze_base:  True → tüm feature katmanları dondurulur (fine-tuning phase 1)
                      False → tüm ağ eğitilebilir (fine-tuning phase 2)
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Orijinal başlık: Linear(1280, 1000)
    # Yeni başlık:     Linear(1280, 256) → ReLU → Dropout → Linear(256, 2)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def unfreeze_all(model: nn.Module) -> None:
    """Fine-tuning phase 2: tüm katmanları açar."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Toplam ve eğitilebilir parametre sayısını döndürür."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
