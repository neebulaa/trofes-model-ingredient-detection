# Trofes Model YOLOv8 Training.V2: Food Ingredient Recognition System (30 Classes)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/spaces/Bagusarya/Trofes_Ingredients_Detection)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8m-green)
![Status](https://img.shields.io/badge/Status-Stable-success)

##  1. Deskripsi Proyek
**Trofes Model.V2** adalah sistem deteksi objek otomatis berbasis Deep Learning untuk mengenali 30 jenis bahan makanan mentah. Model ini dirancang untuk membantu klasifikasi bahan makanan dalam skala besar dengan akurasi tinggi menggunakan arsitektur **YOLOv8m (Medium)**.

##  2. Dataset & Preprocessing (Roboflow)
Dataset dikelola melalui platform **Roboflow** untuk menjamin kualitas anotasi dan konsistensi label.
- **Total Kelas:** 30 Jenis Bahan Makanan.
- **Split Ratio:** 80% Training, 10% Validation, 10% Testing.
- **Preprocessing:** Auto-orientation, Resize to 640x640 (Square).
- **Health Check:** Memastikan distribusi data antar kelas berada di rentang 800 - 1.100 gambar untuk mencegah *bias* model.

### Distribusi Kelas:
| Kategori | Nama Bahan Makanan |
|----------|-------------------|
| **Vegetables** | Tomato, Spinach, Cabbage, Lettuce, Eggplant, Carrot, Bell Pepper, Pea, Garlic, Onion, Potato, Chili, Jalapeno |
| **Proteins** | Egg, Pork, Salmon, Chicken, Shrimp, Oyster, Beef, Tofu, Crab |
| **Others** | Mushroom, Ginger, Rice, Lemon, Kiwi, Banana, Mango, Apple |



##  3. Eksperimen & Pelatihan Model
Model dilatih menggunakan **Google Colab (Tesla T4 GPU)** dengan skema *Deep Refinement*.

- **Konfigurasi Utama:**
  - **Optimizer:** AdamW (Auto)
  - **Epochs:** 40 (30 Base + 10 Refinement)
  - **Augmentasi:** Mosaic (0.8), Mixup (0.05), Flips.
  
- **Strategi Close-Mosaic:** Pada epoch 31-40, augmentasi Mosaic dinonaktifkan (`close_mosaic=10`) untuk menstabilkan *loss* dan mempertajam akurasi koordinat *bounding box*.

##  4. Hasil Evaluasi & Performa
Hasil evaluasi diuji pada **Testing Set** (Data yang belum pernah dilihat model).

### Metrik Akurasi:
| Metrik | Skor |
|--------|------|
| **mAP50** | **0.7075** |
| **mAP50-95** | **0.5449** |


##  5. Versioning & Reproducibility
Proyek ini menggunakan **Hugging Face Hub** sebagai sistem *Model Versioning*. Setiap perubahan bobot model (`.pt`) dicatat menggunakan Git LFS.
- **Current Version:** v1.0 (Stable)
- **Model Card:** Dapat diakses di [Link Hugging Face]

##  6. Cara Penggunaan (Inference)
```python
from ultralytics import YOLO

# Load model dari Hugging Face atau Lokal
model = YOLO('weights/best.pt')```

# Prediksi gambar
results = model.predict(source='test_image.jpg', conf=0.25, save=True)
