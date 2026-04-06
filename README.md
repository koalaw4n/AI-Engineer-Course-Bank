# Materi 8 Sesi Machine Learning: Prediksi, Klasifikasi & Pengelompokan

---

## Sesi 1: Fundamental AI dan Roadmap Karir

### 1.1 Apa itu AI, Machine Learning, Deep Learning?
- Definisi dan hierarki AI → ML → DL
- Perbedaan dengan pemrograman tradisional (rule-based vs data-driven)
- Contoh aplikasi sehari-hari (rekomendasi, voice assistant, fraud detection)

### 1.2 3 Jenis Machine Learning (Fokus ke Dataset)
- **Supervised Learning** (Regresi & Klasifikasi) → ada label/target
- **Unsupervised Learning** (Clustering) → tidak ada label
- Kapan menggunakan masing-masing jenis

### 1.3 Studi Kasus dari Repository
- **Regresi** (oil): Memprediksi nilai kontinu → produksi minyak
- **Klasifikasi** (diabetes): Memprediksi kategori → sakit/tidak sakit
- **Clustering** (mall): Mengelompokkan tanpa label → segmentasi pelanggan

### 1.4 Roadmap Karir AI Engineer
- Peran: AI Engineer vs Data Scientist vs ML Engineer vs Data Analyst
- Skill tree: Matematika → Python → Data Wrangling → Modeling → Deployment
- Sertifikasi relevan (TensorFlow, AWS ML, Databricks)
- Portofolio yang menarik recruiter (proyek end-to-end > notebook)

### 1.5 Setup Environment
- Install Python, Jupyter Notebook/VS Code, Git
- Library: pandas, numpy, matplotlib, seaborn, scikit-learn
- Buat repository GitHub pertama untuk portofolio

---

## Sesi 2: Python untuk Kebutuhan AI

### 2.1 Review Python Cepat untuk Data
- List comprehension, dictionary, function, lambda, error handling
- Baca file dengan berbagai format (CSV, Excel, HTML)
- **Praktik**: Baca 3 dataset dari repository dengan `pd.read_html()`

### 2.2 NumPy untuk Operasi Numerik
- Array vs list Python (performa & kemampuan)
- Broadcasting, indexing, slicing
- Operasi matematika vektor (dot product, matriks)

### 2.3 Pandas untuk Manipulasi Data
- Series & DataFrame
- Seleksi data (loc, iloc, filtering)
- Handling missing value (isnull, dropna, fillna)
- Groupby, aggregate, merge, join
- **Praktik**: Eksplorasi dataset oil, diabetes, mall

### 2.4 Visualisasi Data dengan Matplotlib & Seaborn
- Line plot, bar chart, histogram, scatter plot
- Heatmap korelasi
- Customisasi plot (label, title, legend, color)
- **Praktik**: Visualisasi distribusi tiap fitur di 3 dataset

---

## Sesi 3: Mengolah dan Memahami Data (EDA)

### 3.1 Exploratory Data Analysis (EDA) Framework
- Univariat: analisis 1 fitur (mean, median, std, skewness)
- Bivariat: hubungan 2 fitur (korelasi, scatter plot)
- Multivariat: interaksi banyak fitur (pairplot, heatmap)

### 3.2 Data Cleaning
- Handling missing value (drop, mean/median imputation, forward fill)
- Deteksi & treatment outlier (IQR, Z-score, winsorizing)
- Handling duplicate data

### 3.3 Feature Engineering Dasar
- Encoding kategorikal (LabelEncoder, OneHotEncoder)
- Scaling numerik (MinMaxScaler, StandardScaler)
- Feature selection berdasarkan korelasi

### 3.4 EDA Khusus per Dataset
- **Oil (regresi)**: Cek linearitas fitur vs target
- **Diabetes (klasifikasi)**: Cek distribusi kelas (balance/imbalance)
- **Mall (clustering)**: Cek skala fitur (penting untuk scaling)

### 3.5 Praktik
- Buat laporan EDA lengkap untuk 3 dataset dalam 1 notebook
- Identifikasi insight bisnis dari data (misal: usia paling berisiko diabetes)

---

## Sesi 4: Bangun Model Machine Learning

### 4.1 Persiapan Modeling
- Split data: train-test split (80:20) untuk supervised
- Untuk clustering: tidak perlu split
- Validasi sederhana: cross-validation (k-fold)

### 4.2 Model untuk Regresi (Oil)
- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Evaluasi awal: MSE, MAE, R2

### 4.3 Model untuk Klasifikasi (Diabetes)
- Logistic Regression (baseline)
- Decision Tree Classifier
- Random Forest Classifier
- Evaluasi awal: accuracy, precision, recall, F1

### 4.4 Model untuk Clustering (Mall)
- K-Means Clustering
- Menentukan K dengan Elbow Method
- Evaluasi: silhouette score, inertia

### 4.5 Praktik
- Bangun 3 model untuk oil (pilih terbaik sementara)
- Bangun 3 model untuk diabetes (pilih terbaik sementara)
- Bangun K-Means untuk mall dengan K=3,4,5

---

## Sesi 5: Optimasi dan Evaluasi Model AI

### 5.1 Hyperparameter Tuning
- Perbedaan parameter vs hyperparameter
- GridSearchCV (exhaustive)
- RandomizedSearchCV (sampling)
- Optuna (Bayesian optimization) untuk advanced

### 5.2 Optimasi untuk Regresi (Oil)
- Tuning RandomForestRegressor: n_estimators, max_depth, min_samples_split
- Coba XGBoost Regressor (lebih powerfull)
- Bandingkan performa sebelum vs setelah tuning

### 5.3 Optimasi untuk Klasifikasi (Diabetes)
- Tuning XGBoost Classifier: learning_rate, max_depth, subsample
- Handle imbalance dengan class_weight atau SMOTE
- Gunakan ROC-AUC sebagai metric utama

### 5.4 Optimasi untuk Clustering (Mall)
- Coba berbagai K dengan silhouette score
- Bandingkan K-Means vs DBSCAN (tanpa perlu menentukan K)
- Evaluasi stabilitas cluster dengan bootstrap

### 5.5 Evaluasi Mendalam
- **Regresi**: learning curve, residual plot, cross-validation score
- **Klasifikasi**: confusion matrix, ROC-AUC, precision-recall curve
- **Clustering**: silhouette plot, Davies-Bouldin index
- **Semua**: identifikasi overfitting/underfitting

---

## Sesi 6: Bangun Project AI (End-to-End)

### 6.1 Struktur Proyek AI Profesional
```
project-ai/
├── data/
│   ├── raw/           (data asli dari repository)
│   └── processed/     (data setelah cleaning)
├── notebooks/         (EDA, eksperimen modeling)
│   ├── 1_eda_oil.ipynb
│   ├── 2_eda_diabetes.ipynb
│   └── 3_eda_mall.ipynb
├── src/
│   ├── preprocess.py  (fungsi cleaning & feature engineering)
│   ├── train.py       (training & saving model)
│   └── predict.py     (inference function)
├── models/            (saved model .pkl atau .joblib)
├── app/               (untuk sesi 7)
├── requirements.txt
├── README.md
└── .gitignore
```

### 6.2 Modular Code
- Pisahkan fungsi preprocessing (bisa dipakai ulang)
- Buat script training yang menyimpan model & metrics
- Gunargakan `if __name__ == "__main__"` untuk testing

### 6.3 Experiment Tracking
- **MLflow** untuk logging parameter, metrics, artifacts
- Alternatif sederhana: simpan metrics ke CSV atau JSON
- Bandingkan eksperimen: baseline vs tuned model

### 6.4 Version Control dengan Git
- Commit per perubahan signifikan
- Branching: main, develop, feature/xxx
- Push ke GitHub sebagai portofolio

### 6.5 Praktik
- Buat struktur proyek seperti di atas
- Refactor kode dari sesi 4-5 ke dalam src/
- Track 3 eksperimen (baseline, tuned, XGBoost) dengan MLflow

---

## Sesi 7: Deploy Project dan Insight Dunia Kerja

### 7.1 Persiapan Deployment
- Save model terbaik dengan `joblib.dump()` atau `pickle`
- Buat file `requirements.txt` (pin versi library)
- Optional: Dockerfile untuk containerization

### 7.2 Membangun API dengan FastAPI
- Endpoint `/predict` (single prediction)
- Endpoint `/predict-batch` (multiple predictions)
- Endpoint `/health` untuk health check
- Request/response model dengan Pydantic
- **Praktik**: Wrap model oil, diabetes, mall ke 3 endpoint berbeda

### 7.3 Deploy ke Production
- **Sanberlify.com** (platform Indonesia, 300 kredit gratis)
- Alternatif: Railway, Render, atau Streamlit Cloud
- Langkah deploy: push ke Git → connect ke platform → auto-deploy
- Set environment variables & custom domain (opsional)

### 7.4 Insight Dunia Kerja AI
- **CV & Portofolio**: Tampilkan proyek end-to-end (bukan hanya notebook)
- **Skill yang paling dicari**: Deployment (FastAPI/Docker), Cloud (AWS/GCP), MLOps
- **Tren 2025-2026**: LLM (RAG, Fine-tuning), Edge AI, AI Agent
- **Tips wawancara**: 
  - Take-home challenge: bersihkan kode, dokumentasi baik
  - System design: pikirkan scaling, monitoring, retraining
  - Case study: hubungkan ke business impact

### 7.5 Praktik
- Deploy FastAPI ke Sanberlify
- Uji coba dengan Postman atau `curl`
- Buat dokumentasi API di README (endpoint, contoh request/response)

---

## Sesi 8: Final Project & Portfolio Review

### 8.1 Integrasi 3 ML dalam 1 Aplikasi
- Pilih framework: **Streamlit** (UI interaktif) atau **FastAPI** (backend murni)
- Fitur aplikasi:
  - Pilih jenis analisis (Regresi / Klasifikasi / Clustering)
  - Input manual atau upload CSV
  - Tampilkan hasil prediksi atau cluster assignment
  - Visualisasi hasil (plot, tabel)

### 8.2 Final Project Requirements
**Wajib:**
- Menggunakan minimal 2 dari 3 dataset (oil, diabetes, mall)
- Model yang sudah dioptimasi (dari sesi 5)
- API atau aplikasi web yang bisa diakses publik
- Repository GitHub dengan README lengkap
- Video demo (max 3 menit)

**Nilai plus:**
- Menggunakan ketiga jenis ML dalam 1 aplikasi
- Deploy dengan Docker
- Unit testing
- Monitoring sederhana (logging)

### 8.3 Struktur Final Project
```
final-project-ai/
├── README.md           (penjelasan proyek, cara run, link demo, API docs)
├── requirements.txt
├── Dockerfile          (opsional)
├── app/
│   ├── main.py         (FastAPI) atau streamlit_app.py
│   ├── models/         (3 model terbaik: oil, diabetes, mall)
│   └── utils.py        (helper functions)
├── notebooks/          (EDA & eksperimen, untuk dokumentasi)
├── tests/              (unit test untuk preprocessing & prediction)
└── .github/workflows/  (opsional: CI/CD)
```

### 8.4 Portfolio Review
- Presentasi final project ke kelas (10 menit/orang)
- Peer review: beri feedback konstruktif
- Checklist portofolio GitHub:
  - [ ] README menarik (badge, gambar, link demo)
  - [ ] Kode rapi & terstruktur
  - [ ] Ada dokumentasi API (jika FastAPI)
  - [ ] Link live demo berfungsi

### 8.5 Next Step Setelah Program
- Advanced: Deep Learning (TensorFlow/PyTorch), NLP, Computer Vision
- MLOps: CI/CD untuk ML, model monitoring, data drift detection
- Cloud certification: AWS ML Specialty, Google Professional ML Engineer
- Komunitas: Kaggle, Indonesia AI, local meetup

---

## Ringkasan Cepat 8 Sesi dengan 7 Pilar

| Sesi | Pilar | Topik | Dataset |
|------|-------|-------|---------|
| 1 | Fundamental AI & Roadmap | Pengenalan 3 jenis ML + karir | Oil, Diabetes, Mall |
| 2 | Python untuk AI | NumPy, Pandas, visualisasi | Oil, Diabetes, Mall |
| 3 | Mengolah & memahami data | EDA, cleaning, feature engineering | Oil, Diabetes, Mall |
| 4 | Bangun model ML | Linear Reg, Logistic Reg, K-Means | Oil, Diabetes, Mall |
| 5 | Optimasi & evaluasi | Hyperparameter tuning, XGBoost, evaluasi mendalam | Oil, Diabetes, Mall |
| 6 | Bangun project AI | Struktur proyek, modular code, MLflow | Oil, Diabetes, Mall |
| 7 | Deploy + insight kerja | FastAPI, Sanberlify, tips karir | Oil, Diabetes, Mall |
| 8 | Final project & review | Integrasi 3 ML, portfolio, presentasi | Oil, Diabetes, Mall |

---
