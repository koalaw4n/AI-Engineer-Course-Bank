# Sesi 7: Deploy Project dan Insight Dunia Kerja

---

## 🎯 TUJUAN BELAJAR
Setelah sesi ini, kamu akan mampu:
1. Memahami konsep **Deployment** (Membawa model dari notebook ke user).
2. Membangun API menggunakan **FastAPI** untuk melayani prediksi secara real-time.
3. Melakukan deployment ke platform cloud seperti **Streamlit Cloud**, **Railway**, atau **Render**.
4. Memahami **Roadmap Karir** AI Engineer di era Gen-AI (2025/2026).

---

# 🚀 7.1 PERSIAPAN DEPLOYMENT

Sebelum mendeploy, kita harus memastikan model kita sudah "matang" dan tersimpan dalam format yang bisa dibaca oleh server (Serialization).

### 📦 Menyimpan Model (Export)
Gunakan `joblib` untuk menyimpan object model dan scaler hasil training.

```python
import joblib

# Contoh simpan model dan scaler dari sesi sebelumnya
joblib.dump(best_model_oil, 'models/oil_model.pkl')
joblib.dump(scaler_oil, 'models/oil_scaler.pkl')

print("✅ Model & Scaler berhasil diekspor ke folder models/")
```

### 📋 Membuat `requirements.txt`
Ini adalah file "daftar belanja" library yang harus diinstal oleh server cloud.
```text
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
fastapi==0.103.1
uvicorn==0.23.2
joblib==1.3.2
pydantic==2.3.0
```

---

# ⚡ 7.2 MEMBANGUN API DENGAN FASTAPI

API (Application Programming Interface) adalah pintu gerbang agar aplikasi lain (Web, Mobile) bisa "berbicara" dengan model AI kita.

### 🛠️ Implementasi `main.py`
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. Inisialisasi App
app = FastAPI(
    title="AI Engineer Course API",
    description="API untuk Prediksi Produksi Minyak & Deteksi Diabetes",
    version="1.0.0"
)

# 2. Load Model & Scaler (Gunakan Path yang Benar)
try:
    oil_model = joblib.load('models/oil_model.pkl')
    oil_scaler = joblib.load('models/oil_scaler.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# 3. Definisi Schema Input
class OilInput(BaseModel):
    temperature: float
    pressure: float
    flow_rate: float
    viscosity: float

# 4. Endpoint Utama
@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Model API is ready!"}

# 5. Endpoint Prediksi
@app.post("/predict/oil")
def predict_oil(data: OilInput):
    try:
        # Konversi input ke DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Preprocessing (Scaling)
        df_scaled = oil_scaler.transform(df)
        
        # Prediksi
        prediction = oil_model.predict(df_scaled)
        
        return {
            "prediction": float(prediction[0]),
            "unit": "barrels/day",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

# 🌍 7.3 STRATEGI DEPLOY KE CLOUD

Pilih platform sesuai kebutuhan proyekmu:

| Platform | Cocok Untuk | Tingkat Kesulitan |
|----------|-------------|-------------------|
| **Streamlit Cloud** | Dashboard Web & Visualisasi | ⭐ (Sangat Mudah) |
| **Railway / Render** | Backend API (FastAPI/Flask) | ⭐⭐ (Mudah) |
| **Hugging Face Spaces** | Demo Model AI & Gradio | ⭐⭐ (Menengah) |
| **AWS / GCP / Azure** | Skala Perusahaan (Besar) | ⭐⭐⭐⭐ (Sulit) |

### 💡 Alur Push ke Production:
1. **Local Test**: Pastikan aplikasi jalan di komputermu (`uvicorn main:app`).
2. **Git Push**: Upload semua file ke **GitHub**.
3. **Connect Cloud**: Hubungkan GitHub ke platform cloud (misal: Streamlit Cloud).
4. **Deploy**: Tunggu sistem menginstal requirements dan menjalankan aplikasimu.

---

# 💼 7.4 ROADMAP KARIR AI ENGINEER 2025

Dunia AI berubah sangat cepat. Berikut adalah skill yang wajib kamu miliki tahun ini:

### 🛠️ Core Skills:
1. **MLOps**: Bukan cuma buat model, tapi tahu cara memantau (*monitoring*) model di production.
2. **LLM & Prompt Engineering**: Mampu mengintegrasikan model bahasa (GPT, Llama) ke aplikasi.
3. **Data Engineering Skills**: Tahu cara mengambil data dari database (SQL/NoSQL) secara otomatis.

### 📝 Tips Portofolio di GitHub:
- **README** adalah Resume-mu. Gunakan gambar/GIF demo aplikasi.
- Tuliskan **Alasan Memilih Model** (Contoh: "Saya memilih Random Forest karena memberikan akurasi 92% dan tahan terhadap outlier").
- Tuliskan **Business Impact** (Contoh: "Model ini bisa membantu optimasi biaya produksi hingga 10%").

---

# 🎯 7.5 TUGAS: DEPLOY API PERTAMAMU

1. Buat folder baru di komputermu bernama `my-ai-deployment`.
2. Masukkan file model `.pkl` kedalam folder `models/`.
3. Buat file `main.py` menggunakan FastAPI.
4. Jalankan secara lokal dan buka `http://127.0.0.1:8000/docs`.
5. Screenshot tampilan dokumentasi Swagger API kamu dan simpan sebagai bukti praktik.

---

> **"A model in a notebook is a prototype. A model in an API is a product."**
