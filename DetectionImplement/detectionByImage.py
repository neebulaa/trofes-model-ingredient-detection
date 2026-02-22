from ultralytics import YOLO

# Load model hasil training
model = YOLO('best (1).pt')

# Jalankan prediksi pada gambar
results = model.predict(
    source='D:/Back Up Bagus Arya/Folder Bagus/cawu 4/Machine Learning/Model_Detection_Trofes/apple_test',   # pastikan nama file sesuai!
    save=True,               # simpan hasil ke folder 'runs/detect/predict'
    show=True,
    conf=0.5,                # confidence threshold (bisa turunin ke 0.25 biar lebih sensitif)
    # source=frame,
    # conf=0.5, 
    verbose=False
)
