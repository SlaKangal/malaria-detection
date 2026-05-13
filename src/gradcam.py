"""
gradcam.py
----------
EfficientNetB0 için Grad-CAM ısı haritaları üretir.
Her örnek için:
  - Orijinal görüntü
  - Grad-CAM ısı haritası
  - Üst üste bindirme (overlay)
  - Gerçek sınıf / Tahmin / Confidence

Kullanım:
    python src/gradcam.py                    # Test setinden rastgele 8 görüntü
    python src/gradcam.py --image path/img.png  # Tek görüntü
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EFFICIENTNET_MODEL_PATH, GRADCAM_DIR,
    TEST_DIR, CLASSES, GRADCAM_NUM_IMAGES, SEED
)
from dataset import get_single_image_tensor, EVAL_TRANSFORMS
from models import get_efficientnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(GRADCAM_DIR, exist_ok=True)


# ── Grad-CAM core ─────────────────────────────────────────────

class GradCAM:
    """
    Hook tabanlı Grad-CAM implementasyonu.
    EfficientNetB0'ın son konvolüsyon katmanına (features[-1]) uygulanır.
    """

    def __init__(self, model: torch.nn.Module):
        self.model     = model
        self.gradients = None
        self.activations = None

        # EfficientNetB0'ın son Conv bloğuna hook ekle
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor) -> tuple:
        """
        Args:
            img_tensor: (1, C, H, W) tensor

        Returns:
            (pred_class, confidence, heatmap_normalized)
        """
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE)

        # İleri geçiş
        output = self.model(img_tensor)
        probs  = F.softmax(output, dim=1)
        pred_class  = output.argmax(1).item()
        confidence  = probs[0, pred_class].item()

        # Geri yayılım
        self.model.zero_grad()
        output[0, pred_class].backward()

        # Global Average Pooling ile ağırlıklar
        weights   = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam       = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam       = F.relu(cam)
        cam       = cam.squeeze().cpu().numpy()

        # 0-1 aralığına normalize et
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return pred_class, confidence, cam


# ── Görselleştirme ────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Normalize edilmiş tensörü 0-255 numpy dizisine çevirir."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_heatmap(img_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Grad-CAM ısı haritasını görüntü üzerine bindirir."""
    h, w   = img_rgb.shape[:2]
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.55, heatmap, 0.45, 0)
    return overlay


def visualize_gradcam(model, img_paths: list, true_labels: list,
                      save_name: str = "gradcam_grid.png"):
    """
    Birden fazla görüntü için Grad-CAM grid görseli oluşturur.
    Her satır: [Orijinal | Heatmap | Overlay]
    """
    gradcam   = GradCAM(model)
    n         = len(img_paths)
    fig_height = 3.5 * n
    fig, axes  = plt.subplots(n, 3, figsize=(12, fig_height))

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Orijinal Görüntü", "Grad-CAM Isı Haritası", "Üst Üste Bindirme"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i, (img_path, true_lbl) in enumerate(zip(img_paths, true_labels)):
        tensor    = get_single_image_tensor(img_path)
        pred_cls, conf, cam = gradcam.generate(tensor)
        img_rgb   = denormalize(tensor)
        overlay   = apply_heatmap(img_rgb, cam)
        heatmap_rgb = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.resize(heatmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        correct = (pred_cls == true_lbl)
        color   = "green" if correct else "red"
        status  = "✓" if correct else "✗"

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(
            f"Gerçek: {CLASSES[true_lbl]}\n"
            f"Tahmin: {CLASSES[pred_cls]} {status} ({conf:.2%})",
            fontsize=9, color=color, rotation=0,
            labelpad=130, va="center"
        )

        axes[i, 1].imshow(heatmap_rgb)
        axes[i, 2].imshow(overlay)

        for ax in axes[i]:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
                spine.set_visible(True)

    # Açıklama
    green_patch = mpatches.Patch(color="green", label="Doğru tahmin")
    red_patch   = mpatches.Patch(color="red",   label="Yanlış tahmin")
    fig.legend(handles=[green_patch, red_patch],
               loc="lower center", ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0))

    plt.suptitle(
        "Grad-CAM Açıklanabilirlik Analizi — EfficientNetB0\n"
        "Modelin karar verirken odaklandığı hücre bölgeleri",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = os.path.join(GRADCAM_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Grad-CAM görseli kaydedildi: {save_path}")
    return save_path


# ── Ana akış ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="Tek görüntü yolu (verilmezse test setinden seçilir)")
    parser.add_argument("--n", type=int, default=GRADCAM_NUM_IMAGES,
                        help="Görselleştirilecek görüntü sayısı")
    args = parser.parse_args()

    # Model yükle
    if not os.path.exists(EFFICIENTNET_MODEL_PATH):
        print(f"❌ Model bulunamadı: {EFFICIENTNET_MODEL_PATH}")
        print("   Önce: python src/train.py --model efficientnet")
        sys.exit(1)

    print("🧠 EfficientNetB0 yükleniyor...")
    model = get_efficientnet(freeze_base=False).to(DEVICE)
    model.load_state_dict(
        torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    if args.image:
        # Tek görüntü modu
        print(f"📷 Görüntü: {args.image}")
        visualize_gradcam(model, [args.image], [0], "gradcam_single.png")
    else:
        # Test setinden rastgele örnekle
        print(f"🔀 Test setinden {args.n} rastgele görüntü seçiliyor...")
        random.seed(SEED)
        dataset = ImageFolder(TEST_DIR, transform=EVAL_TRANSFORMS)
        indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))

        img_paths   = [dataset.samples[i][0]  for i in indices]
        true_labels = [dataset.samples[i][1]  for i in indices]

        # 4'er görüntüden oluşan iki grup (okunabilirlik için)
        chunk = 4
        for part, start in enumerate(range(0, len(img_paths), chunk)):
            paths  = img_paths[start:start + chunk]
            labels = true_labels[start:start + chunk]
            visualize_gradcam(
                model, paths, labels,
                f"gradcam_part{part + 1}.png"
            )

    print("✅ Grad-CAM analizi tamamlandı!")


if __name__ == "__main__":
    main()
