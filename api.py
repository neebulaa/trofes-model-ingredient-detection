from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from ultralytics import YOLO
import numpy as np

# Inisialisasi FastAPI
app = FastAPI(title="Dewa Trofes Detection API")

# Muat model (Pastikan file best.pt ada di folder yang sama)
try:
    model = YOLO("best (1).pt") 
    print("Model Trofes berhasil dimuat!")
except Exception as e:
    print(f"Gagal muat model: {e}")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        # 2. Baca file gambar
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # 3. Jalankan inferensi dengan threshold 0.4
        results = model(image_np, conf=0.4, verbose=False) 
        
        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            return {"ingredients": []}

        # 4. Dictionary untuk filter akurasi tertinggi di "backstage"
        best_per_class = {}
        
        # Ekstrak data
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        names = result.names 

        for cls_id, conf in zip(class_ids, confidences):
            cls_name = names[cls_id]
            # Bandingkan untuk ambil yang paling akurat jika ada duplikat
            if cls_name not in best_per_class or conf > best_per_class[cls_name]:
                best_per_class[cls_name] = conf

        # 5. Output: Hanya kembalikan Array Nama (Bahan)
        final_ingredients = list(best_per_class.keys())

        return {
            "status": "success",
            "ingredients": final_ingredients
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API Trofes Aktif!"}

# --- Tambahkan ini di baris paling akhir file main.py ---

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 artinya API bisa diakses dari perangkat lain dalam satu WiFi (misal HP kamu)
    # Port 8000 adalah pintu masuknya
    uvicorn.run(app, host="0.0.0.0", port=8000)