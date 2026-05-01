# Backend FastAPI untuk Klasifikasi Gunung Berapi

## Langkah 1: Setup Virtual Environment

```bash
# Buat folder project
mkdir volcano-fastapi
cd volcano-fastapi

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

## Langkah 2: Buat requirements.txt

```txt
fastapi>=0.104.0
uvicorn>=0.24.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

## Langkah 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Langkah 4: Struktur Folder Project

```
volcano-fastapi/
├── venv/
├── requirements.txt
├── main.py                    # FastAPI application
├── volcano_classifier_model.joblib    # Model file (from notebook)
└── test_api.py                # Testing script
```

## Langkah 5: Pastikan Model File Ada

Pastikan file `volcano_classifier_model.joblib` dari notebook Anda sudah ada di folder yang sama. Jika belum, copy dari hasil training notebook.

## Langkah 6: Buat FastAPI Application

**main.py**:

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# ============================================
# DATA MODELS (Pydantic)
# ============================================

class VolcanoInput(BaseModel):
    """Single volcano input data"""
    tinggi_meter: float = Field(..., ge=0, le=10000, description="Height in meters")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    
    @validator('tinggi_meter')
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError('Height must be greater than 0')
        return v

class VolcanoBatchInput(BaseModel):
    """Batch volcano input data"""
    data: List[VolcanoInput] = Field(..., description="List of volcano data")

class PredictionResponse(BaseModel):
    """Single prediction response"""
    success: bool
    prediction: str
    confidence: float
    input: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    success: bool
    results: List[Dict[str, Any]]
    total: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_file: str

# ============================================
# GLOBAL VARIABLES
# ============================================

model = None
label_encoder = None
feature_columns = ['tinggi_meter', 'lat', 'lon']

# ============================================
# LIFESPAN MANAGEMENT (Startup & Shutdown)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    print("=" * 50)
    print("LOADING MODEL...")
    print("=" * 50)
    
    try:
        model = joblib.load('volcano_classifier_model.joblib')
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down...")

# ============================================
# INITIALIZE FASTAPI APP
# ============================================

app = FastAPI(
    title="Volcano Shape Classifier API",
    description="API for predicting volcano shapes based on height and coordinates",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================
# HELPER FUNCTIONS
# ============================================

def predict_single(height: float, lat: float, lon: float):
    """
    Make prediction for single volcano
    
    Returns:
        tuple: (prediction_class, confidence_score)
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Create DataFrame with correct feature order
    input_data = pd.DataFrame([[height, lat, lon]], columns=feature_columns)
    
    # Make prediction
    prediction_encoded = model.predict(input_data)[0]
    
    # Get confidence (probability)
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(np.max(probabilities))
    
    # Note: Since we don't have label_encoder from notebook,
    # we need to extract classes from model or define manually
    # For now, we'll return the encoded value and you can map it later
    # Alternatively, train and save label_encoder separately
    
    return prediction_encoded, confidence

def get_class_name(encoded_class: int) -> str:
    """
    Map encoded class to actual volcano shape name
    These classes are from the notebook's label_encoder.classes_
    """
    # These are the classes from the original notebook
    classes = [
        'Fumarol', 'bawah laut', 'kaldera', 'kerucut bara', 
        'kompleks', 'kubah lava', 'perisai', 'stratovulkan', 'supervulkan'
    ]
    
    if 0 <= encoded_class < len(classes):
        return classes[encoded_class]
    return f"unknown_class_{encoded_class}"

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Volcano Shape Classifier API",
        "version": "1.0.0",
        "endpoints": "/predict, /predict/batch, /health, /info"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_file="volcano_classifier_model.joblib"
    )

@app.get("/info")
async def get_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "success": True,
        "model_type": type(model).__name__,
        "features": feature_columns,
        "n_features": len(feature_columns),
        "is_trained": hasattr(model, 'predict'),
        "has_probability": hasattr(model, 'predict_proba')
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_volcano(volcano: VolcanoInput):
    """
    Predict volcano shape for a single volcano
    
    Example request:
    {
        "tinggi_meter": 1500,
        "lat": -7.0,
        "lon": 110.0
    }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model file."
        )
    
    try:
        # Make prediction
        prediction_encoded, confidence = predict_single(
            volcano.tinggi_meter,
            volcano.lat,
            volcano.lon
        )
        
        # Get class name
        prediction_class = get_class_name(prediction_encoded)
        
        return PredictionResponse(
            success=True,
            prediction=prediction_class,
            confidence=round(confidence, 4),
            input={
                "tinggi_meter": volcano.tinggi_meter,
                "lat": volcano.lat,
                "lon": volcano.lon
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: VolcanoBatchInput):
    """
    Predict volcano shapes for multiple volcanoes
    
    Example request:
    {
        "data": [
            {"tinggi_meter": 1500, "lat": -7.0, "lon": 110.0},
            {"tinggi_meter": 2801, "lat": 4.914, "lon": 96.329},
            {"tinggi_meter": 617, "lat": 5.820, "lon": 95.280}
        ]
    }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    results = []
    
    for volcano in batch.data:
        try:
            prediction_encoded, confidence = predict_single(
                volcano.tinggi_meter,
                volcano.lat,
                volcano.lon
            )
            
            prediction_class = get_class_name(prediction_encoded)
            
            results.append({
                "input": {
                    "tinggi_meter": volcano.tinggi_meter,
                    "lat": volcano.lat,
                    "lon": volcano.lon
                },
                "prediction": prediction_class,
                "confidence": round(confidence, 4),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "input": {
                    "tinggi_meter": volcano.tinggi_meter,
                    "lat": volcano.lat,
                    "lon": volcano.lon
                },
                "error": str(e),
                "status": "failed"
            })
    
    return BatchPredictionResponse(
        success=True,
        results=results,
        total=len(results)
    )

@app.post("/predict/form")
async def predict_form(
    tinggi_meter: float,
    lat: float,
    lon: float
):
    """
    Predict using form data (for web forms)
    
    Example: POST with form-urlencoded
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        prediction_encoded, confidence = predict_single(tinggi_meter, lat, lon)
        prediction_class = get_class_name(prediction_encoded)
        
        return {
            "success": True,
            "prediction": prediction_class,
            "confidence": round(confidence, 4),
            "input": {
                "tinggi_meter": tinggi_meter,
                "lat": lat,
                "lon": lon
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
```

## Langkah 7: Buat Testing Script

**test_api.py**:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_info():
    """Test info endpoint"""
    response = requests.get(f"{BASE_URL}/info")
    print("Model Info:", response.json())
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction"""
    data = {
        "tinggi_meter": 1500,
        "lat": -7.0,
        "lon": 110.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("\nSingle Prediction:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction"""
    data = {
        "data": [
            {"tinggi_meter": 2801, "lat": 4.914, "lon": 96.329},
            {"tinggi_meter": 617, "lat": 5.820, "lon": 95.280},
            {"tinggi_meter": 2245, "lat": 3.850, "lon": 97.664},
            {"tinggi_meter": 1500, "lat": -7.0, "lon": 110.0}
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print("\nBatch Prediction:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_form_prediction():
    """Test form prediction"""
    data = {
        "tinggi_meter": 1500,
        "lat": -7.0,
        "lon": 110.0
    }
    response = requests.post(f"{BASE_URL}/predict/form", data=data)
    print("\nForm Prediction:")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING VOLCANO CLASSIFIER API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Get Model Info", test_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Form Prediction", test_form_prediction),
    ]
    
    for name, test_func in tests:
        print(f"\n▶ Testing: {name}")
        try:
            if test_func():
                print(f"✅ {name} passed")
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ {name} error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")
```

## Langkah 8: Buat Script untuk Menjalankan Server

**run.sh** (Mac/Linux):

```bash
#!/bin/bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**run.bat** (Windows):

```batch
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Langkah 9: Cara Menjalankan

### 1. Setup dan Install:

```bash
# Buat dan aktivasi virtual environment
cd volcano-fastapi
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pastikan file model ada
ls volcano_classifier_model.joblib  # cek file model
```

### 2. Jalankan Server:

```bash
# Langsung dengan uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Atau pakai script
# Mac/Linux:
chmod +x run.sh
./run.sh

# Windows:
run.bat
```

### 3. Testing API:

```bash
# Buka terminal baru, jalankan test
python test_api.py
```

## Langkah 10: Dokumentasi API Otomatis

FastAPI menyediakan dokumentasi otomatis:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Langkah 11: Contoh Curl Commands

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tinggi_meter": 1500, "lat": -7.0, "lon": 110.0}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"tinggi_meter": 2801, "lat": 4.914, "lon": 96.329},
      {"tinggi_meter": 617, "lat": 5.820, "lon": 95.280}
    ]
  }'

# Form prediction
curl -X POST http://localhost:8000/predict/form \
  -d "tinggi_meter=1500&lat=-7.0&lon=110.0"
```

## Langkah 12: Production Ready dengan Gunicorn (Optional)

**gunicorn_config.py**:

```python
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 60
keepalive = 5
```

**run_production.sh**:

```bash
pip install gunicorn
gunicorn -c gunicorn_config.py main:app
```

## Langkah 13: Docker Support (Optional)

**Dockerfile**:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build dan run**:

```bash
docker build -t volcano-api .
docker run -p 8000:8000 volcano-api
```

## Ringkasan

```bash
# Complete workflow
mkdir volcano-fastapi
cd volcano-fastapi
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn joblib pandas numpy python-multipart pydantic

# Copy main.py dan volcano_classifier_model.joblib ke folder ini

# Jalankan server
uvicorn main:app --reload --port 8000

# Test (di terminal baru)
python test_api.py

# Buka browser: http://localhost:8000/docs
```

API siap digunakan dengan endpoint:
- `GET /health` - Cek status
- `GET /info` - Info model
- `POST /predict` - Prediksi tunggal
- `POST /predict/batch` - Prediksi batch
- `POST /predict/form` - Prediksi dari form
- `GET /docs` - Dokumentasi Swagger
