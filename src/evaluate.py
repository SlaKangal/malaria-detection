"""
evaluate.py
-----------
Eğitilmiş modelleri test setinde değerlendirir.
Çıktılar:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix görseli
  - ROC Curve & AUC
  - Eğitim/Doğrulama accuracy-loss grafikleri
  - Karşılaştırma tablosu (JSON + terminal)

Kullanım:
    python src/evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    CM_DIR, PLOT_DIR, METRICS_DIR, CLASSES, SEED
)
from dataset import get_dataloaders
from models import CustomCNN, get_efficientnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CM_DIR,      exist_ok=True)
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ── Tahmin toplama ────────────────────────────────────────────

def get_predictions(model, loader):
    """Test seti üzerinde tüm tahminleri ve olasılıkları toplar."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds   = outputs.argmax(1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ── Metrik hesaplama ──────────────────────────────────────────

def compute_metrics(labels, preds, probs) -> dict:
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=0)
    roc_auc = auc(fpr, tpr)
    return {
        "accuracy":  round(accuracy_score(labels, preds),  4),
        "precision": round(precision_score(labels, preds, average="weighted"), 4),
        "recall":    round(recall_score(labels, preds, average="weighted"),    4),
        "f1":        round(f1_score(labels, preds, average="weighted"),        4),
        "auc":       round(roc_auc, 4),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
    }


# ── Görseller ────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, model_name: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    ax.set_title(f"Karmaşıklık Matrisi — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CM_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Confusion matrix kaydedildi: {path}")


def plot_roc_curves(results: dict):
    """İki modelin ROC eğrilerini tek grafikte gösterir."""
    colors = {"Custom CNN": "#2196F3", "EfficientNetB0": "#F44336"}
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, r in results.items():
        lbl = f"{name}  (AUC = {r['auc']:.4f})"
        ax.plot(r["fpr"], r["tpr"], lw=2, label=lbl, color=colors.get(name))

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Rastgele sınıflandırıcı")
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title("ROC Eğrisi Karşılaştırması", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 ROC eğrisi kaydedildi: {path}")


def plot_training_history(model_name: str):
    """Kaydedilmiş history JSON'undan acc/loss grafiği çizer."""
    hist_file = os.path.join(
        METRICS_DIR,
        f"{model_name.lower().replace(' ', '_')}_history.json"
    )
    if not os.path.exists(hist_file):
        print(f"  ⚠️  {hist_file} bulunamadı, grafik atlanıyor.")
        return

    with open(hist_file) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, h["train_acc"], "b-o", markersize=3, label="Train")
    axes[0].plot(epochs, h["val_acc"],   "r-o", markersize=3, label="Validation")
    axes[0].set_title(f"{model_name} — Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, h["train_loss"], "b-o", markersize=3, label="Train")
    axes[1].plot(epochs, h["val_loss"],   "r-o", markersize=3, label="Validation")
    axes[1].set_title(f"{model_name} — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(f"Eğitim Süreci — {model_name}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = f"history_{model_name.lower().replace(' ', '_')}.png"
    path  = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Eğitim grafikleri kaydedildi: {path}")


def plot_comparison_bar(results: dict):
    """İki modelin metriklerini yan yana bar grafiğiyle gösterir."""
    metrics   = ["accuracy", "precision", "recall", "f1", "auc"]
    labels    = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x         = np.arange(len(metrics))
    width     = 0.3
    colors    = ["#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=name, color=colors[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Model Karşılaştırması — Test Seti Metrikleri",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Karşılaştırma grafiği kaydedildi: {path}")


# ── Terminal tablosu ──────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "=" * 60)
    print(f"  {'MODEL':<22} {'ACC':>6} {'PREC':>6} {'REC':>6} {'F1':>6} {'AUC':>6}")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:<22} "
              f"{r['accuracy']:>6.4f} "
              f"{r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} "
              f"{r['f1']:>6.4f} "
              f"{r['auc']:>6.4f}")
    print("=" * 60)


# ── Ana akış ─────────────────────────────────────────────────

def main():
    print("📂 Veri yükleniyor...")
    _, _, test_loader, _ = get_dataloaders()

    all_results = {}

    # ── Custom CNN ──
    if os.path.exists(CNN_MODEL_PATH):
        print("\n🔍 Custom CNN değerlendiriliyor...")
        cnn = CustomCNN().to(DEVICE)
        cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
        labels, preds, probs = get_predictions(cnn, test_loader)
        all_results["Custom CNN"] = compute_metrics(labels, preds, probs)
        plot_confusion_matrix(labels, preds, "Custom CNN")
        plot_training_history("Custom_CNN")
        print(classification_report(labels, preds, target_names=CLASSES))
    else:
        print(f"⚠️  {CNN_MODEL_PATH} bulunamadı. Önce train.py çalıştırın.")

    # ── EfficientNetB0 ──
    if os.path.exists(EFFICIENTNET_MODEL_PATH):
        print("\n🔍 EfficientNetB0 değerlendiriliyor...")
        eff = get_efficientnet(freeze_base=False).to(DEVICE)
        eff.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE))
        labels, preds, probs = get_predictions(eff, test_loader)
        all_results["EfficientNetB0"] = compute_metrics(labels, preds, probs)
        plot_confusion_matrix(labels, preds, "EfficientNetB0")
        plot_training_history("EfficientNetB0")
        print(classification_report(labels, preds, target_names=CLASSES))
    else:
        print(f"⚠️  {EFFICIENTNET_MODEL_PATH} bulunamadı. Önce train.py çalıştırın.")

    # ── Karşılaştırma görselleri ──
    if len(all_results) == 2:
        plot_roc_curves(all_results)
        plot_comparison_bar(all_results)
        print_summary(all_results)

        # JSON kaydet
        save_path = os.path.join(METRICS_DIR, "comparison_results.json")
        save_data = {k: {m: v for m, v in r.items() if m not in ("fpr", "tpr")}
                     for k, r in all_results.items()}
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  📁 Sonuçlar kaydedildi: {save_path}")

    print("\n✅ Değerlendirme tamamlandı!")


if __name__ == "__main__":
    main()
