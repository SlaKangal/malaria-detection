"""
config.py
---------
Projenin tüm sabit parametrelerini tek bir yerden yönetir.
Diğer tüm scriptler bu dosyayı import eder.
"""

import os

# ── Dizin yolları ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "weights")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")
CM_DIR      = os.path.join(OUTPUT_DIR, "confusion_matrix")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "grad_cam")
PRED_DIR    = os.path.join(OUTPUT_DIR, "predictions")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# ── Görüntü parametreleri ──────────────────────────────────────
IMG_SIZE    = 224          # Model giriş boyutu (224x224 px)
IMG_CHANNELS = 3           # RGB
IMG_FORMAT  = "PNG"        # Kaynak format (NIH dataset: PNG)

# ── Sınıf tanımları ───────────────────────────────────────────
CLASSES     = ["Parasitized", "Uninfected"]
NUM_CLASSES = 2
CLASS_MAP   = {"Parasitized": 0, "Uninfected": 1}

# ── Eğitim parametreleri ──────────────────────────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 30           # Custom CNN
EPOCHS_TL   = 20           # Transfer Learning (EfficientNetB0)
LEARNING_RATE = 1e-3
LR_PATIENCE = 5            # ReduceLROnPlateau patience
EARLY_STOP  = 8            # EarlyStopping patience
WEIGHT_DECAY = 1e-4

# ── Veri bölme oranları ───────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Model dosya isimleri ──────────────────────────────────────
CNN_MODEL_PATH        = os.path.join(MODEL_DIR, "custom_cnn_best.pth")
EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_best.pth")

# ── Grad-CAM parametresi ──────────────────────────────────────
GRADCAM_NUM_IMAGES = 8     # Görselleştirilecek örnek sayısı

# ── Seed (tekrar üretilebilirlik) ─────────────────────────────
SEED = 42
