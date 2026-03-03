import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --- KONFIGURASI ---
# Sesuaikan dengan Repo Hugging Face Anda
HF_REPO_ID = "Bagusarya/Trofes-YOLOv8m-Detct-Ingredients"
MODEL_FILENAME = "best.pt"  # Nama file model di repo HF (biasanya best.pt)
SOURCE_FOLDER = "test_images"  # Folder gambar uji coba

def main():
    print(" Memulai Skrip Deteksi Trofes...")

    # 1. Cek apakah folder test_images ada dan punya isi
    if not os.path.exists(SOURCE_FOLDER) or not os.listdir(SOURCE_FOLDER):
        print(f" Error: Folder '{SOURCE_FOLDER}' kosong atau tidak ditemukan!")
        print(f"   Silakan masukkan minimal 1 gambar ke folder '{SOURCE_FOLDER}' lalu jalankan ulang.")
        return

    # 2. Download Model dari Hugging Face (Jika belum ada di lokal)
    print(f" Mengunduh model dari Hugging Face: {HF_REPO_ID}...")
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        print(f" Model berhasil dimuat di: {model_path}")
    except Exception as e:
        print(f" Gagal mengunduh model. Pastikan repo publik dan koneksi internet stabil.")
        print(f"   Detail Error: {e}")
        return

    # 3. Load Model
    model = YOLO(model_path)

    # 4. Jalankan Prediksi
    print(f" Memproses gambar di folder '{SOURCE_FOLDER}'...")
    
    # Catatan: show=True akan membuka jendela pop-up gambar.
    # Jika dosen menjalankan ini di server/CLI tanpa GUI, setting show=False.
    results = model.predict(
        source=SOURCE_FOLDER,
        save=True,        # Hasil akan disimpan di folder 'runs/detect/predict'
        conf=0.5,         # Confidence threshold
        show=False,       # Ubah ke True jika ingin pop-up window (hanya di PC dengan GUI)
        verbose=True
    )

    print("\n✅ Proses Selesai!")
    print(f" Hasil deteksi gambar tersimpan di folder: 'runs/detect/predict'")

if __name__ == "__main__":
    main()