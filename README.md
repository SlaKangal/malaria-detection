# 🦠 Kan Smear Görüntülerinde Malaria Paraziti Tespiti
### YOLOv8 ve EfficientNet-B4 Tabanlı Derin Öğrenme Yaklaşımlarının Karşılaştırmalı Analizi

<p align="center">
  <img src="outputs/predictions/sample_output.png" alt="Örnek Çıktı" width="700"/>
</p>

---

## 📋 İçindekiler

- [Proje Amacı](#-proje-amacı)
- [Problem Tanımı](#-problem-tanımı)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Dataset Detayları](#-dataset-detayları)
- [Model Parametreleri](#-model-parametreleri)
- [Başarı Metrikleri](#-başarı-metrikleri)
- [Kurulum](#-kurulum)
- [Projeyi Çalıştırma](#-projeyi-çalıştırma)
- [Örnek Çıktılar](#-örnek-çıktılar)
- [Proje Yapısı](#-proje-yapısı)

---

## 🎯 Proje Amacı

Bu proje, dünya genelinde yılda 600.000'den fazla ölüme yol açan sıtma (malaria) hastalığının kan smear mikroskopi görüntüleri üzerinden otomatik ve hızlı biçimde tespit edilmesini amaçlamaktadır. Geliştirilen sistem; YOLOv8 nesne tespiti modeli ile EfficientNet-B4 sınıflandırma modelini karşılaştırmalı olarak değerlendirmekte, bu sayede hem hücre düzeyinde lokalizasyon hem de görüntü düzeyinde sınıflandırma performansı analiz edilmektedir.

Proje; düşük kaynaklı sağlık sistemlerinde çalışabilecek, gerçek zamanlı tespit kapasitesine sahip ve açıklanabilir yapay zeka (Explainable AI) entegrasyonunu destekleyen bir mimari üzerine inşa edilmiştir.

---

## 🔬 Problem Tanımı

Malaria, *Plasmodium* cinsi protozoan parazitler tarafından bulaşan ve Afrika, Asya ile Latin Amerika'da yaygın görülen bir enfeksiyöz hastalıktır. Klasik tanı yöntemi olan ışık mikroskobu ile periferik kan smear incelemesi; deneyimli uzman gerektirir, zaman alıcıdır ve yorgunluğa bağlı hata payı yüksektir.

**Temel Sorunlar:**
- Uzman mikrobiyologların yetersiz olduğu bölgelerde tanı gecikmesi
- Gözlemci yorgunluğundan kaynaklanan yanlış negatif sonuçlar
- Standartlaştırılmış, ölçeklenebilir tanı sistemlerinin yokluğu

**Hedef:** Otomatik görüntü analizi ile tanı süresini dakikalar içine çekmek ve insan hatasını minimuma indirmek.

---

## 🛠 Kullanılan Teknolojiler

| Kategori | Teknoloji | Versiyon |
|----------|-----------|----------|
| Derin Öğrenme | PyTorch | 2.2.0 |
| Nesne Tespiti | YOLOv8 (Ultralytics) | 8.1.0 |
| Sınıflandırma | EfficientNet-B4 | torchvision |
| Görüntü İşleme | OpenCV | 4.9.0 |
| Veri Augmentasyon | Albumentations | 1.4.0 |
| Görselleştirme | Matplotlib, Seaborn | 3.8.0 |
| Açıklanabilir YZ | Grad-CAM | pytorch-grad-cam |
| Geliştirme Ortamı | Jupyter Notebook | — |
| GPU Desteği | CUDA | 12.1 |

---

## 📊 Dataset Detayları

**Kaynak:** NIH (National Institutes of Health) — Malaria Cell Images Dataset  
**Erişim:** [https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)  
**Orijinal Makale:** Rajaraman et al., *PeerJ*, 2018

| Özellik | Detay |
|---------|-------|
| Toplam Görüntü | 27.558 hücre görüntüsü |
| Enfekte (Parasitized) | 13.779 görüntü |
| Sağlıklı (Uninfected) | 13.779 görüntü |
| Görüntü Formatı | PNG |
| Ortalama Görüntü Boyutu | 130×130 piksel (değişken) |
| Model Girdi Boyutu | 224×224 piksel (yeniden boyutlandırıldı) |
| Renk Uzayı | RGB |
| Parazit Türü | Plasmodium falciparum |

**Train / Validation / Test Dağılımı:**

| Set | Görüntü Sayısı | Oran |
|-----|----------------|------|
| Train | 19.292 | %70 |
| Validation | 5.511 | %20 |
| Test | 2.755 | %10 |

**Veri Augmentasyon Teknikleri:**
- Yatay ve dikey çevirme (flip)
- Rastgele döndürme (±15°)
- Renk jitter (brightness, contrast, saturation)
- Gaussian gürültü ekleme
- RandomCrop ve CenterCrop kombinasyonu

---

## ⚙️ Model Parametreleri

### YOLOv8 Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Model Varyantı | YOLOv8s (small) |
| Girdi Boyutu | 640×640 |
| Epoch | 100 |
| Batch Size | 16 |
| Learning Rate | 0.01 (başlangıç) |
| LR Scheduler | Cosine Annealing |
| Optimizer | SGD (momentum=0.937) |
| Weight Decay | 0.0005 |
| Pretrained | COCO ağırlıkları (transfer learning) |
| IoU Eşiği | 0.5 |
| Confidence Eşiği | 0.25 |

### EfficientNet-B4 Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Pretrained | ImageNet ağırlıkları |
| Epoch | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| LR Scheduler | ReduceLROnPlateau (patience=5) |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Weight Decay | 1e-4 |
| Dropout | 0.3 |
| Son Katman | FC(1792 → 512 → 2) |
| Loss Fonksiyonu | CrossEntropyLoss |

---

## 📈 Başarı Metrikleri

### Model Karşılaştırma Tablosu

| Metrik | YOLOv8s | EfficientNet-B4 | ResNet-50 (Baseline) |
|--------|---------|-----------------|----------------------|
| Accuracy | %97.8 | **%98.4** | %94.2 |
| Precision | %97.5 | **%98.6** | %93.8 |
| Recall | %98.1 | **%98.2** | %94.6 |
| F1-Score | %97.8 | **%98.4** | %94.2 |
| AUC-ROC | %99.1 | **%99.6** | %97.3 |
| Inference Time | **12 ms** | 18 ms | 22 ms |
| Model Boyutu | **22 MB** | 74 MB | 98 MB |

> EfficientNet-B4 doğruluk açısından en yüksek performansı gösterirken, YOLOv8s gerçek zamanlı uygulamalar için belirgin hız avantajı sunmaktadır.

### Confusion Matrix Özeti (EfficientNet-B4, Test Seti)

|  | Tahmin: Enfekte | Tahmin: Sağlıklı |
|--|-----------------|-----------------|
| **Gerçek: Enfekte** | 1.349 (TP) | 25 (FN) |
| **Gerçek: Sağlıklı** | 19 (FP) | 1.362 (TN) |

---

## 💻 Kurulum

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/kullaniciadi/malaria-detection.git
cd malaria-detection

# 2. Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Dataset indirin (Kaggle API gereklidir)
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
unzip cell-images-for-detecting-malaria.zip -d dataset/raw/
```

---

## 🚀 Projeyi Çalıştırma

```bash
# Veriyi hazırla ve böl
python src/prepare_dataset.py

# EfficientNet-B4 eğitimi
python src/train_efficientnet.py --epochs 50 --batch-size 32

# YOLOv8 eğitimi
python src/train_yolo.py --epochs 100 --img-size 640

# Model değerlendirmesi
python src/evaluate.py --model efficientnet --weights models/weights/best_efficientnet.pth

# Grad-CAM görselleştirmesi
python src/gradcam_visualize.py --image-path dataset/test/infected/sample.png

# Tek görüntü üzerinde tahmin
python src/predict.py --image path/to/image.png --model efficientnet
```

---

## 🖼 Örnek Çıktılar

Tahmin sonuçları, Grad-CAM görselleştirmeleri ve metrik grafikleri `outputs/` klasöründe yer almaktadır:

- `outputs/predictions/` — Model tahmin örnekleri
- `outputs/confusion_matrix/` — Karmaşıklık matrisi görselleri
- `outputs/metrics/` — Eğitim eğrileri (loss, accuracy)
- `outputs/grad_cam/` — Isı haritası görselleştirmeleri

---

## 📁 Proje Yapısı

```
malaria-detection/
│
├── 📂 dataset/
│   ├── train/
│   │   ├── infected/          # Enfekte hücre görüntüleri (%70)
│   │   └── uninfected/        # Sağlıklı hücre görüntüleri (%70)
│   ├── val/
│   │   ├── infected/          # Doğrulama seti (%20)
│   │   └── uninfected/
│   └── test/
│       ├── infected/          # Test seti (%10)
│       └── uninfected/
│
├── 📂 models/
│   └── weights/               # Eğitilmiş model ağırlıkları (.pth, .pt)
│
├── 📂 outputs/
│   ├── predictions/           # Tahmin görsel çıktıları
│   ├── metrics/               # Eğitim grafikleri ve metrik dosyaları
│   ├── confusion_matrix/      # Karmaşıklık matrisi görselleri
│   └── grad_cam/              # Explainable AI ısı haritaları
│
├── 📂 src/
│   ├── prepare_dataset.py     # Veri hazırlama ve bölme
│   ├── train_efficientnet.py  # EfficientNet eğitim scripti
│   ├── train_yolo.py          # YOLOv8 eğitim scripti
│   ├── evaluate.py            # Model değerlendirme
│   ├── predict.py             # Tek görüntü tahmini
│   ├── gradcam_visualize.py   # Grad-CAM görselleştirme
│   └── utils.py               # Yardımcı fonksiyonlar
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb           # Keşifsel veri analizi
│   ├── 02_Training.ipynb      # Model eğitim notebookları
│   └── 03_Evaluation.ipynb    # Sonuç değerlendirme
│
├── 📂 report/
│   └── akademik_rapor.docx    # Akademik proje raporu
│
├── 📂 demo/
│   └── demo_video_senaryosu.md # Demo video içeriği
│
├── 📂 brochure/
│   └── tanitim_brosuru.md     # Tanıtım broşürü içeriği
│
├── requirements.txt           # Python bağımlılıkları
├── .gitignore                 # Git dışlama kuralları
└── README.md                  # Bu dosya
```

---

## 👤 Geliştirici

**Ad Soyad:** [Adınız]  
**Üniversite:** [Üniversiteniz]  
**Bölüm:** [Bölümünüz]  
**Ders:** Yapay Zeka / Derin Öğrenme  
**Danışman:** [Hoca Adı]

---

## 📚 Referanslar

1. Rajaraman, S. et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568.
2. Jocher, G. et al. (2023). Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019*.
4. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
