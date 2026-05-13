"""
predict.py
----------
Tek bir görüntü üzerinde tahmin yapar ve sonucu görselleştirir.
Çıktı: terminal + PNG dosyası

Kullanım:
    python src/predict.py --image path/to/cell.png
    python src/predict.py --image path/to/cell.png --model cnn
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    CLASSES, PRED_DIR
)
from dataset import get_single_image_tensor, EVAL_TRANSFORMS
from models import CustomCNN, get_efficientnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PRED_DIR, exist_ok=True)


def load_model(model_type: str):
    if model_type == "cnn":
        model = CustomCNN()
        path  = CNN_MODEL_PATH
    else:
        model = get_efficientnet(freeze_base=False)
        path  = EFFICIENTNET_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model ağırlığı bulunamadı: {path}\n"
            f"Önce: python src/train.py --model {model_type}"
        )

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def predict(model, img_path: str) -> dict:
    tensor  = get_single_image_tensor(img_path).to(DEVICE)
    with torch.no_grad():
        output  = model(tensor)
        probs   = F.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    return {
        "prediction":  CLASSES[pred_idx],
        "pred_index":  pred_idx,
        "confidence":  confidence,
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)}
    }


def save_prediction_plot(img_path: str, result: dict, model_name: str):
    """Görüntü ve tahmin bilgilerini birleştiren PNG oluşturur."""
    img  = Image.open(img_path).convert("RGB")
    pred = result["prediction"]
    conf = result["confidence"]
    probs= result["probabilities"]

    color = "#e53935" if pred == "Parasitized" else "#43a047"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # Görüntü
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Girdi Görüntüsü", fontsize=11)

    # Probability bar
    bars = axes[1].barh(
        CLASSES,
        [probs[c] for c in CLASSES],
        color=["#e53935" if c == "Parasitized" else "#43a047" for c in CLASSES],
        alpha=0.8, edgecolor="white", height=0.5
    )
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Olasılık", fontsize=11)
    axes[1].set_title("Sınıf Olasılıkları", fontsize=11)
    for bar, cls in zip(bars, CLASSES):
        w = bar.get_width()
        axes[1].text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{w:.2%}", va="center", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Tahmin: {pred}   |   Güven: {conf:.2%}   |   Model: {model_name}",
        fontsize=12, fontweight="bold", color=color, y=1.02
    )
    plt.tight_layout()

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(PRED_DIR, f"pred_{img_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Görüntü dosyası yolu")
    parser.add_argument("--model", default="efficientnet",
                        choices=["cnn", "efficientnet"])
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Görüntü bulunamadı: {args.image}")
        sys.exit(1)

    model_name = "EfficientNetB0" if args.model == "efficientnet" else "Custom CNN"
    print(f"🧠 Model yükleniyor: {model_name}")
    model  = load_model(args.model)

    print(f"🔍 Tahmin yapılıyor: {args.image}")
    result = predict(model, args.image)

    # Terminal çıktısı
    print("\n" + "─" * 40)
    print(f"  Tahmin      : {result['prediction']}")
    print(f"  Güven Skoru : {result['confidence']:.2%}")
    print(f"  Olasılıklar :")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"    {cls:<15} {bar:<20} {prob:.4f}")
    print("─" * 40)

    # Görsel kaydet
    save_path = save_prediction_plot(args.image, result, model_name)
    print(f"\n  📁 Görsel kaydedildi: {save_path}")
    print("✅ Tamamlandı!")


if __name__ == "__main__":
    main()
