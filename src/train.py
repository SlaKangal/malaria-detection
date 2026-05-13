"""
train.py
--------
Her iki modeli sırayla eğitir ve en iyi ağırlıkları kaydeder.

Kullanım:
    python src/train.py --model all         # Her ikisini eğit
    python src/train.py --model cnn         # Sadece Custom CNN
    python src/train.py --model efficientnet # Sadece EfficientNet
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NUM_EPOCHS, EPOCHS_TL, LEARNING_RATE, LR_PATIENCE,
    EARLY_STOP, WEIGHT_DECAY, BATCH_SIZE,
    CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    PLOT_DIR, METRICS_DIR, SEED
)
from dataset import get_dataloaders
from models import CustomCNN, get_efficientnet, unfreeze_all, count_parameters


# ── Sabit ayarlar ─────────────────────────────────────────────
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")


# ── Yardımcı: tek epoch eğitim ────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds    = torch.max(outputs, 1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Ana eğitim fonksiyonu ─────────────────────────────────────

def train_model(model, model_name: str, save_path: str,
                num_epochs: int, train_loader, val_loader) -> dict:
    """
    Modeli eğitir, en iyi val_accuracy'de ağırlıkları kaydeder.
    EarlyStopping ve ReduceLROnPlateau içerir.
    """
    model = model.to(DEVICE)

    params   = count_parameters(model)
    print(f"\n{'='*55}")
    print(f"  Model: {model_name}")
    print(f"  Toplam parametre   : {params['total']:,}")
    print(f"  Eğitilebilir param.: {params['trainable']:,}")
    print(f"  Epoch: {num_epochs} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*55}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE, verbose=True
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    best_val_acc  = 0.0
    no_improve    = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, False)

        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{num_epochs}] "
              f"| Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} "
              f"| Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} "
              f"| {elapsed:.1f}s")

        # En iyi model kaydı
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
            print(f"  ✅ Yeni en iyi model kaydedildi! (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1

        # EarlyStopping
        if no_improve >= EARLY_STOP:
            print(f"  ⚠️  EarlyStopping: {EARLY_STOP} epoch gelişme yok. Durduruluyor.")
            break

    print(f"\n  ✅ Eğitim tamamlandı. En iyi val_acc: {best_val_acc:.4f}")

    # History JSON olarak kaydet
    os.makedirs(METRICS_DIR, exist_ok=True)
    hist_path = os.path.join(METRICS_DIR, f"{model_name.lower().replace(' ', '_')}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  📁 Eğitim geçmişi kaydedildi: {hist_path}")

    return history


# ── EfficientNet 2 aşamalı fine-tuning ───────────────────────

def train_efficientnet_two_phase(train_loader, val_loader) -> dict:
    """
    Aşama 1 (5 epoch): Sadece sınıflandırıcı başlığı eğitilir.
    Aşama 2 (kalan): Tüm ağ ince ayar yapılır, LR küçültülür.
    """
    model = get_efficientnet(freeze_base=True).to(DEVICE)

    # Aşama 1
    print("\n  🔒 Aşama 1: Feature katmanları donduruldu (5 epoch)")
    history = train_model(
        model, "EfficientNetB0",
        EFFICIENTNET_MODEL_PATH,
        num_epochs=5,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # En iyi ağırlıkları yükle ve tüm katmanları aç
    model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE))
    unfreeze_all(model)
    print(f"\n  🔓 Aşama 2: Tüm katmanlar açıldı, LR={LEARNING_RATE * 0.1}")

    # Aşama 2 — daha küçük LR ile
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE * 0.1,
                     weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=LR_PATIENCE, verbose=True)

    best_val_acc = max(history["val_acc"])
    no_improve   = 0

    for epoch in range(1, EPOCHS_TL - 5 + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, False)
        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"  [FT Epoch {epoch:02d}] "
              f"Train Acc: {tr_acc:.4f} | Val Acc: {vl_acc:.4f} | {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), EFFICIENTNET_MODEL_PATH)
            no_improve = 0
            print(f"  ✅ Fine-tune en iyisi kaydedildi! (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            print(f"  ⚠️  EarlyStopping devreye girdi.")
            break

    # Güncel history'i kaydet (üzerine yaz)
    hist_path = os.path.join(METRICS_DIR, "efficientnetb0_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ── Giriş noktası ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "cnn", "efficientnet"])
    args = parser.parse_args()

    print("📂 Veri yükleniyor...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    if args.model in ("all", "cnn"):
        cnn = CustomCNN()
        train_model(cnn, "Custom_CNN", CNN_MODEL_PATH,
                    NUM_EPOCHS, train_loader, val_loader)

    if args.model in ("all", "efficientnet"):
        train_efficientnet_two_phase(train_loader, val_loader)

    print("\n🎉 Tüm eğitimler tamamlandı!")
