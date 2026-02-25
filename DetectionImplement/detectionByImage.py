from ultralytics import YOLO

# Load model hasil training
model = YOLO('MASUKKAN_NAMA_MODEL')

# Jalankan prediksi pada gambar
results = model.predict(
    source='[MASUKKAN_PATH_FOLDER_YG_BERISI_IMG',   # pastikan nama file sesuai!
    save=True,               # simpan hasil ke folder 'runs/detect/predict'
    show=True,
    conf=0.5,                # confidence threshold (bisa turunin ke 0.25 biar lebih sensitif)
    # source=frame,
    # conf=0.5, 
    verbose=False
)
