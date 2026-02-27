
Trofes: Food Ingredient Detection API
Python FastAPI YOLOv8 OpenCV Hugging Face Docker

An intelligent computer vision backend serving a YOLOv8m model for real-time food ingredient detection. This system identifies 30 types of raw food ingredients, designed to integrate seamlessly with the Trofes Recipe Ecosystem.

🚀 Machine Learning Architecture
This repository hosts the Trofes Detection Model V2, architected for high-precision object detection in culinary contexts.

1. Model Architecture: YOLOv8m (Medium)
Algorithm: YOLOv8m (Convolutional Neural Network)
Use Case: Real-time detection of 30 distinct food ingredient classes (e.g., Vegetables, Proteins, Fruits).
Why this approach? YOLOv8m offers the optimal balance between inference speed and accuracy for real-time applications. We implemented a Deep Refinement Strategy:
Phase 1 (Epoch 1-30): Heavy Mosaic augmentation for robust feature generalization.
Phase 2 (Epoch 31-40): Disabled Mosaic (close_mosaic=10) to stabilize bounding box coordinates, resulting in higher mAP.
2. Dataset & Training
Source: Curated dataset from Roboflow.
Split: 80% Training, 10% Validation, 10% Testing.
Health Check: Balanced distribution (800 - 1.100 images per class) to prevent bias.
Performance:
mAP50: 0.7075
mAP50-95: 0.5449
📂 Project Structure
📦 TROFES-MODEL-DETECTION ┣ 📂 API                    # FastAPI application (deployed to Hugging Face Spaces) ┃ ┗ 📜 app.py               # Main API entry point, handles routing & inference ┣ 📂 test_images            # Sample images for local testing ┣ 📂 weights                # Model artifacts (.pt) — auto-generated/downloaded ┣ 📜 .env                   # Environment configuration (Model path, thresholds) ┣ 📜 .gitignore ┣ 📜 Dockerfile             # Container config for Hugging Face Spaces deployment ┣ 📜 main.py                # Local inference script (Standalone usage) ┣ 📜 README.md ┗ 📜 requirements.txt       # Python dependencies

🐳 Docker & Deployment
Dockerfile
The included Dockerfile containerizes the API/app.py for deployment on Hugging Face Spaces:
FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "7860"]
⚠️ Hugging Face Spaces requires the Dockerfile to be at the repository root and exposes port 7860 by default.

Deployment Architecture

GitHub Repository
 ┣ Source code + Dockerfile (root)
        │
        ▼ Build Process
        │
        ├──► Hugging Face Hub
        │    └── Model Weights stored here
        │         (Bagusarya/Trofes-YOLOv8m-Detct-Ingredients)
        │
        └──► Hugging Face Spaces (Docker)
             └── API/app.py running on port 7860
                  └── pulls model artifacts from HF Hub at runtime
                       via hf_hub_download()

🔗 Model Repository: Bagusarya/Trofes-YOLOv8m-Detct-Ingredients

💻 Local Development
Prerequisites
Python 3.9+ — Download here
Git — Download here
Docker (optional, for container testing) — Download here
1. Clone the Repository
git clone https://github.com/username-anda/trofes-model-detection.git
cd TROFES-MODEL-DETECTION
2. Create a Virtual Environment
# Create
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. (Optional) Run Inference Script Locally
If you want to run detection on local images without starting the server:
python main.py
5. Run the API Locally
uvicorn API.app:app --reload
The API will be live at http://127.0.0.1:8000

💡 On first run, the app will automatically download model artifacts from
Bagusarya/Trofes-YOLOv8m-Detct-Ingredients
via Hugging Face Hub. Make sure you have a stable internet connection.
6. Interactive API Docs
| Interface | URL |
|-----------|-----|
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |


Run with Docker (Local Container)
To replicate the exact production environment:
# Build the image
docker build -t trofes-detection-api .

# Run the container
docker run -p 7860:7860 trofes-detection-api

API available at http://localhost:7860

📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check & Model status. |
| `POST` | `/predict` | Upload an image file to detect ingredients. Returns JSON bounding boxes. |

Example Request (cURL)
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

Example Response
{
  "success": true,
  "predictions": [
    {
      "class": "Tomato",
    }
  ]
}

🤝 Contributing
Fork the repository
Create a new branch (git checkout -b feature/your-feature)
Commit your changes (git commit -m 'feat: add some feature')
Push to the branch (git push origin feature/your-feature)
Open a Pull Request

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
