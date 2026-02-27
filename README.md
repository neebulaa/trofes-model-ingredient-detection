# Trofes: Food Ingredient Detection API
### 🍎 Python | FastAPI | YOLOv8 | Hugging Face | Docker

An intelligent computer vision backend serving a **YOLOv8m** model for real-time food ingredient detection. This system identifies 30 types of raw food ingredients, designed to integrate seamlessly with the **Trofes Recipe Ecosystem**.

---

## 🚀 Machine Learning Architecture
This repository hosts the **Trofes Detection Model V2**, architected for high-precision object detection in culinary contexts.

### 1. Model Details: YOLOv8m (Medium)
* **Algorithm:** YOLOv8m (Convolutional Neural Network)
* **Use Case:** Real-time detection of 30 distinct food ingredient classes (Vegetables, Proteins, Fruits, etc.).
* **Deep Refinement Strategy:**
    * **Phase 1 (Epoch 1-30):** Heavy Mosaic augmentation for robust feature generalization.
    * **Phase 2 (Epoch 31-40):** Disabled Mosaic (`close_mosaic=10`) to stabilize bounding box coordinates, resulting in higher mAP.

### 2. Dataset & Training
* **Source:** Curated dataset from Roboflow.
* **Split:** 80% Training | 10% Validation | 10% Testing.
* **Health Check:** Balanced distribution (800 - 1,100 images per class) to prevent bias.
* **Performance Metrics:**
    * **mAP50:** `0.7075`
    * **mAP50-95:** `0.5449`

---

## 📂 Project Structure
```text
📦 TROFES-MODEL-DETECTION
 ┣ 📂 API                    # FastAPI application
 ┃ ┗ 📜 app.py               # Main API entry point & inference logic
 ┣ 📂 test_images            # Sample images for local testing
 ┣ 📂 weights                # Model artifacts (.pt)
 ┣ 📜 .env                   # Environment configuration
 ┣ 📜 .gitignore 
 ┣ 📜 Dockerfile             # Container config for Hugging Face Spaces
 ┣ 📜 main.py                # Local inference script (Standalone)
 ┣ 📜 README.md 
 ┗ 📜 requirements.txt       # Python dependencies


## 🐳 Docker & Deployment
Deployment Workflow
GitHub Repository: Source code & Dockerfile are pushed to main.

Hugging Face Spaces: Automatically builds the Docker image.

Runtime: API/app.py runs on port 7860, pulling weights from Hugging Face Hub at startup.

DockerFile
FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "7860"]
[!IMPORTANT]
Hugging Face Spaces requires the Dockerfile to be at the repository root and exposes port 7860 by default.

## 💻 Local Development
1. Setup Environment
# Clone the repo
git clone [https://github.com/username-anda/trofes-model-detection.git](https://github.com/username-anda/trofes-model-detection.git)
cd TROFES-MODEL-DETECTION

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install Dependencies
pip install -r requirements.txt

2. Run the API
# Start FastAPI with hot-reload
uvicorn API.app:app --reload
API akan tersedia di: http://127.0.0.1:8000

## 📡 API Endpoints
Method,Endpoint,Description
GET,/,Health check & Model status.
POST,/predict,Upload image file to detect ingredients.

Example Response:
{
  "success": true,
  "predictions": [
    {
      "class": "Tomato"
    }
  ]
}

##📄 License & Credits
Model Weights: Bagusarya/Trofes-YOLOv8m-Detct-Ingredients

License: MIT License.