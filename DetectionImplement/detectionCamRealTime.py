import cv2
from ultralytics import YOLO

model = YOLO('best (1).pt') 


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak bisa dibuka.")
    exit()

print("Menjalankan Model Trofes... Tekan 'q' untuk berhenti.")

while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Jalankan deteksi (Inference)
    # conf=0.5 artinya hanya tampilkan objek dengan tingkat keyakinan > 50%
    results = model.predict(source=frame, conf=0.5, show=False)

    # 4. Visualisasikan hasil deteksi di frame
    annotated_frame = results[0].plot()

    # Tampilkan jendela kamera
    cv2.imshow("Trofes AI - Food Detection", annotated_frame)

    # Berhenti jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()