from ultralytics import YOLO
import cv2

# Load model hasil training kamu
model = YOLO('best (1).pt')

# Buka kamera default (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Error: Tidak bisa mengakses kamera.")
    exit()

print("🎥 Kamera aktif!")
print("Tekan [SPASI] untuk ambil frame & deteksi")
print("Tekan 'q' untuk keluar")

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Tampilkan preview live (tanpa deteksi)
    cv2.imshow('Preview - Tekan SPASI untuk deteksi', frame)

    key = cv2.waitKey(1) & 0xFF

    # Keluar jika tekan 'q'
    if key == ord('q'):
        break

    # Deteksi saat tekan SPASI
    if key == 32:  # ASCII spasi = 32
        print("Mengambil frame dan mendeteksi...")

        # Jalankan deteksi langsung pada array frame (numpy array)
        results = model(frame)  # ← input berupa array, bukan path file!

        # Ambil gambar hasil deteksi (dengan bounding box)
        annotated_frame = results[0].plot()  # plot() menambahkan box & label

        # Tampilkan hasil deteksi di jendela baru
        cv2.imshow('Hasil Deteksi', annotated_frame)

        # Opsional: cetak jumlah objek terdeteksi
        num_boxes = len(results[0].boxes)
        print(f"Terdeteksi {num_boxes} objek")

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
print("Program selesai.")