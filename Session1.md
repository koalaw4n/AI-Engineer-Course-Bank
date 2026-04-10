# Sesi 1: Fundamental AI dan Roadmap Karir

## 1.1 Apa itu AI, Machine Learning, Deep Learning?

### Definisi dan Hierarki AI → ML → DL

```mermaid
flowchart TD
    A[Artificial Intelligence<br>Kecerdasan Buatan] --> B[Machine Learning<br>Pembelajaran Mesin]
    B --> C[Deep Learning<br>Pembelajaran Mendalam]
    
    A --- D[Rule-based Systems<br>Sistem berbasis aturan]
    
    subgraph Penjelasan
        A1["AI: Mesin meniru inteligensi manusia<br>(berpikir, belajar, memecahkan masalah)"]
        B1["ML: Belajar dari data tanpa diprogram eksplisit<br>(mengenali pola dari contoh)"]
        C1["DL: Jaringan saraf tiruan dengan banyak lapisan<br>(memproses data kompleks seperti gambar, suara)"]
    end
    
    A -.-> A1
    B -.-> B1
    C -.-> C1
```

**Penjelasan Hierarki:**
- **AI (Artificial Intelligence)** : Payung terbesar - semua upaya membuat mesin cerdas
- **ML (Machine Learning)** : Subset AI - mesin belajar dari data
- **DL (Deep Learning)** : Subset ML - menggunakan neural network berlapis-lapis

**Sumber:**
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson.
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.

---

### Perbedaan dengan Pemrograman Tradisional (Rule-based vs Data-driven)

```mermaid
flowchart LR
    subgraph Tradisional["Rule-based Programming (Tradisional)"]
        direction TB
        T1[Data] --> T2["Rules (Aturan Eksplisit)<br>If... Then... Else..."]
        T2 --> T3[Output]
        T4["Programmer menulis semua aturan"] -.-> T2
    end
    
    subgraph ML["Machine Learning (Data-driven)"]
        direction TB
        M1[Data] --> M2[Model]
        M2 --> M3[Output]
        M4["Model belajar pola dari data<br>+ Label sebagai jawaban"] -.-> M2
    end
    
    style Tradisional fill:#f9f,stroke:#333,stroke-width:2px
    style ML fill:#9cf,stroke:#333,stroke-width:2px
```

| Aspek | Rule-based | Data-driven (ML) |
|-------|-----------|------------------|
| **Pendekatan** | Programmer menulis aturan | Model belajar dari contoh |
| **Adaptasi** | Perlu update manual aturan | Belajar otomatis dari data baru |
| **Kompleksitas** | Terbatas pada kemampuan programmer | Dapat menangani pola kompleks |
| **Contoh** | Sistem pakar, kalkulator, lampu lalu lintas | Pengenalan wajah, NLP, rekomendasi |
| **Performa pada data besar** | Menurun (aturan terlalu banyak) | Meningkat (semakin banyak data semakin baik) |

**Sumber:** Chollet, F. (2018). *Deep Learning with Python*. Manning.

---

### Contoh Aplikasi Sehari-hari

```mermaid
mindmap
  root((Aplikasi AI Sehari-hari))
    Rekomendasi
      Netflix (film/series)
      Shopee/Tokopedia (produk)
      Spotify (lagu)
    Voice Assistant
      Siri (Apple)
      Google Assistant
      Alexa (Amazon)
    Fraud Detection
      Bank (transaksi mencurigakan)
      E-commerce (pembayaran palsu)
    Computer Vision
      Face Unlock (HP)
      OCR (scan teks)
    NLP
      ChatGPT (teks)
      Google Translate (terjemahan)
    Lainnya
      Google Maps (estimasi waktu)
      Email (filter spam)
```

**Sumber:** https://ai.googleblog.com/ , https://netflixtechblog.com/

---

## 1.2 Tiga Jenis Machine Learning (Fokus ke Dataset)

### Perbedaan Klasifikasi, Regresi, dan Pengelompokan (Clustering)

Berdasarkan gambar dari [Mercubuana Yogya](https://imam.mercubuana-yogya.ac.id/wp-content/uploads/2024/11/img.197Jenis-jenis-utama-Machine-Learning-Pembelajaran-Mesin.jpg), ketiga metode ini memiliki perbedaan fundamental:

```mermaid
flowchart TD
    subgraph ML["Machine Learning"]
        direction TB
        S["Supervised Learning<br>(Ada Label/Target)"] --> R["Regresi<br>Label = Nilai Numerik Kontinu"]
        S --> C["Klasifikasi<br>Label = Kategori/Class"]
        U["Unsupervised Learning<br>(Tanpa Label)"] --> CL["Clustering / Pengelompokan<br>Mencari kelompok alami"]
    end
    
    style R fill:#9cf,stroke:#333
    style C fill:#9cf,stroke:#333
    style CL fill:#f9f,stroke:#333
```

### Tabel Perbandingan Utama

| Aspek | **Regresi** | **Klasifikasi** | **Pengelompokan (Clustering)** |
|-------|-------------|----------------|-------------------------------|
| **Tipe Pembelajaran** | Supervised | Supervised | Unsupervised |
| **Label/Target** | Ada (numeric/angka) | Ada (kategori) | Tidak ada |
| **Output** | Nilai kontinu (contoh: 150.000, 37.5°C) | Kelas diskrit (contoh: "sakit", "sehat") | Kelompok alami (cluster) |
| **Contoh Pertanyaan** | "Berapa banyak?" "Seberapa besar?" | "Apakah termasuk kategori A atau B?" | "Bagaimana data ini mengelompok?" |
| **Evaluasi** | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 | Silhouette Score, Inertia |

---

### Penjelasan Detail per Metode

#### 1. Regresi (Regression) - Memprediksi Nilai Numerik

**Definisi:** Memprediksi *output berupa angka kontinu* berdasarkan data input. Label adalah nilai numerik.

**Contoh dari referensi gambar:**
> *"Predict the number of ice creams sold based on day, season, and weather"*

**Penjelasan:** Kita ingin memprediksi **berapa banyak** es krim terjual (nilai: 50, 127, 300, dst.), bukan hanya "laku" atau "tidak laku".

```mermaid
flowchart LR
    Input["Fitur: Hari, Musim, Cuaca<br>(Data Input)"] --> Model["Model Regresi<br>(Linear Regression, Random Forest)"]
    Model --> Output["Output: Jumlah Es Krim<br>Misal: 245 unit (angka kontinu)"]
    
    subgraph Label
        L["Target berupa angka kontinu<br>Contoh: 150, 200, 350"]
    end
    
    Model -.-> L
```

**Karakteristik Regresi:**
- Output tidak terbatas pada nilai diskrit (bisa 150.5, 200.75, dst.)
- Mencari hubungan fungsional antara input dan output
- Contoh lain: prediksi harga rumah, suhu besok, pendapatan perusahaan

---

#### 2. Klasifikasi (Classification) - Memprediksi Kategori

**Definisi:** Memprediksi *output berupa kategori/kelas* berdasarkan data input. Label adalah kelas diskrit.

**Contoh dari referensi gambar:**

> *"Predict whether a patient is at-risk for diabetes based on clinical data"*

**Penjelasan:** Outputnya adalah **kategori** (berisiko diabetes atau tidak) - hanya dua kemungkinan = **Binary Classification**.

> *"Predict the species of a penguin based on its measurements"*

**Penjelasan:** Outputnya adalah **salah satu dari beberapa spesies** (Gentoo, Adelie, Chinstrap) - ini adalah **Multiclass Classification**.

```mermaid
flowchart TD
    subgraph Binary["Binary Classification (2 kelas)"]
        B1["Data Pasien<br>(glukosa, BMI, usia)"] --> B2[Model Klasifikasi] --> B3["At-risk Diabetes?<br>Ya / Tidak"]
    end
    
    subgraph Multiclass["Multiclass Classification (>2 kelas)"]
        M1["Data Pengukuran Penguin<br>(paruh, sirip, berat)"] --> M2[Model Klasifikasi] --> M3["Spesies:<br>Gentoo / Adelie / Chinstrap"]
    end
```

**Karakteristik Klasifikasi:**
- Output bersifat kualitatif/kategorikal
- Dua jenis utama:
  - **Binary classification** (2 kelas, misal: spam/bukan spam, sehat/sakit)
  - **Multiclass classification** (lebih dari 2 kelas, misal: jenis bunga iris, merek mobil)

---

#### 3. Pengelompokan (Clustering) - Mencari Kelompok Alami Tanpa Label

**Definisi:** Mengelompokkan data ke dalam *cluster* (kelompok) di mana anggota dalam satu kelompok memiliki kemiripan, **tanpa** menggunakan label yang sudah diketahui sebelumnya.

**Contoh dari referensi gambar:**
> *"Separate plants into groups based on common characteristics"*

**Penjelasan:** Tidak ada label "jenis tanaman A/B/C" yang disediakan. Algoritma akan mencari pola alami dalam data (misal: berdasarkan tinggi, warna daun, kebutuhan air) dan membentuk kelompok sendiri.

```mermaid
flowchart LR
    Data["Data Tanaman<br>Tanpa Label<br>(tinggi, warna, kebutuhan air)"] --> Model["Algoritma Clustering<br>K-Means, DBSCAN, Hierarchical"]
    Model --> Kel1["Cluster 1:<br>Tanaman Tinggi (>2m)"]
    Model --> Kel2["Cluster 2:<br>Tanaman Pendek (<0.5m)"]
    Model --> Kel3["Cluster 3:<br>Tanaman Merambat"]
    
    style Data fill:#f9f,stroke:#333
```

**Karakteristik Clustering:**
- Data **tanpa label** (unsupervised learning)
- Tujuan: eksplorasi data, segmentasi, menemukan struktur tersembunyi
- Contoh lain: segmentasi pelanggan toko online, pengelompokan berita, analisis gen

---

### Visualisasi Perbandingan Konseptual (2D)

```mermaid
flowchart LR
    subgraph Regression["📈 Regresi (Nilai Kontinu)"]
        direction TB
        R_Plot["<br>Nilai<br> ▲<br> 80 ┤     •<br> 60 ┤  •     •<br> 40 ┤•   •<br> 20 ┤• <br>   └──────▶ Fitur<br>"]
        R_Ket["Mencari garis/kurva<br>yang memprediksi angka<br><br>Contoh: Harga → Luas rumah"]
    end
    
    subgraph Classification["🏷️ Klasifikasi (Kategori)"]
        direction TB
        C_Plot["<br>Fitur2 ▲<br>   │  △ △<br>   │ □   □<br>   │    ○ ○<br>   └──────▶ Fitur1<br>"]
        C_Ket["Mencari batas pemisah<br>antar kelas (△, □, ○)<br><br>Contoh: Jenis bunga → ukuran kelopak"]
    end
    
    subgraph Clustering["🔍 Clustering (Tanpa Label)"]
        direction TB
        CL_Plot["<br>Fitur2 ▲<br>   │  • •<br>   │ •   •<br>   │    • •<br>   └──────▶ Fitur1<br>"]
        CL_Ket["Mengelompokkan titik<br>berdasarkan kedekatan alami<br><br>Contoh: Segmentasi pelanggan"]
    end
```

---

### Kapan Menggunakan Masing-masing Jenis? (Decision Tree)

```mermaid
flowchart TD
    START[Mulai dengan Data] --> Q1{Apakah ada label/target?}
    Q1 -->|Ya, ada target| Q2{Apa tipe target?}
    Q2 -->|Angka kontinu<br>Contoh: harga, suhu, jumlah| R[**Regresi**<br>Contoh: Prediksi harga rumah]
    Q2 -->|Kategori diskrit<br>Contoh: ya/tidak, warna, jenis| C[**Klasifikasi**<br>Contoh: Deteksi spam email]
    Q1 -->|Tidak ada target sama sekali| Q3{Apa tujuan analisis?}
    Q3 -->|Mencari kelompok alami| CL[**Clustering / Pengelompokan**<br>Contoh: Segmentasi pelanggan]
    Q3 -->|Mencari aturan asosiasi| AS[**Association Rule**<br>Contoh: Market basket analysis]
    
    style R fill:#9cf,stroke:#333
    style C fill:#9cf,stroke:#333
    style CL fill:#f9f,stroke:#333
    style AS fill:#f9f,stroke:#333
```

**Sumber:**
- James, G., et al. (2021). *An Introduction to Statistical Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning*. MIT Press.
- Gambar referensi: [Mercubuana Yogya](https://imam.mercubuana-yogya.ac.id/wp-content/uploads/2024/11/img.197Jenis-jenis-utama-Machine-Learning-Pembelajaran-Mesin.jpg)

---

## 1.3 Studi Kasus dari Repository

### Ringkasan Tiga Studi Kasus

```mermaid
flowchart TD
    subgraph Kasus1["1. Regresi (Oil Production)"]
        K1A[Input: Waktu, lokasi sumur, tekanan] --> K1B[Target: Produksi minyak<br>Nilai kontinu dalam barel/hari]
        K1C[Metode: Linear Regression, Random Forest Regressor]
        K1D[Evaluasi: MAE, RMSE, R²]
    end
    
    subgraph Kasus2["2. Klasifikasi (Diabetes)"]
        K2A[Input: Glukosa, BMI, usia, tekanan darah] --> K2B[Target: Sakit / Tidak Sakit<br>Binary classification]
        K2C[Metode: Logistic Regression, SVM, Random Forest]
        K2D[Evaluasi: Accuracy, Precision, Recall, F1]
    end
    
    subgraph Kasus3["3. Clustering (Mall Customers)"]
        K3A[Input: Pendapatan, skor belanja, usia, jenis kelamin] --> K3B[Output: Segmentasi pelanggan<br>Tanpa label sebelumnya]
        K3C[Metode: K-Means, Hierarchical Clustering]
        K3D[Evaluasi: Silhouette Score, Inertia, Elbow Method]
    end
```

### Detail Dataset dan Implementasi

| Studi Kasus | Jenis | Dataset Source | Jumlah Data | Fitur Utama | Metrik Evaluasi |
|-------------|-------|----------------|-------------|-------------|-----------------|
| **Oil Production** | Regresi | [Oil Dataset](https://www.kaggle.com/datasets) | ±1000 baris | Waktu, tekanan, suhu, lokasi | MAE, RMSE, R² |
| **Diabetes** | Klasifikasi | [PIMA Indian Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 baris | Glukosa, BMI, usia, kehamilan | Accuracy, Precision, Recall, F1, AUC |
| **Mall Customers** | Clustering | [Mall Customer Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | 200 baris | Pendapatan tahunan, skor belanja | Silhouette Score, Inertia |

### Contoh Implementasi Sederhana (Pseudocode)

```python
# 1. REGRESI - Prediksi Produksi Minyak
from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)  # y_train = produksi minyak (angka)
prediksi = model_reg.predict(X_test)  # output: 1250.5 barel

# 2. KLASIFIKASI - Deteksi Diabetes
from sklearn.ensemble import RandomForestClassifier
model_clf = RandomForestClassifier()
model_clf.fit(X_train, y_train)  # y_train = 0 (sehat) atau 1 (diabetes)
prediksi = model_clf.predict(X_test)  # output: 0 atau 1

# 3. CLUSTERING - Segmentasi Pelanggan
from sklearn.cluster import KMeans
model_clust = KMeans(n_clusters=5)
model_clust.fit(X)  # X tanpa label!
segment = model_clust.predict(X)  # output: 0,1,2,3,4 (nomor cluster)
```

**Sumber Dataset:**
- Kaggle: https://www.kaggle.com/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Scikit-learn datasets: https://scikit-learn.org/stable/datasets.html

---

## 1.4 Roadmap Karir AI Engineer

### Perbandingan Role: AI Engineer vs Data Scientist vs ML Engineer vs Data Analyst

```mermaid
flowchart LR
    subgraph Roles["Data & AI Roles"]
        direction TB
        DA[Data Analyst<br>📊 Fokus: Insight & Dashboard]
        DS[Data Scientist<br>🔬 Fokus: Modeling & Eksperimen]
        MLE[ML Engineer<br>⚙️ Fokus: Deployment & Scalability]
        AIE[AI Engineer<br>🤖 Fokus: End-to-end AI System]
    end
    
    DA --> DS --> MLE --> AIE
    
    style DA fill:#f9f,stroke:#333
    style DS fill:#bbf,stroke:#333
    style MLE fill:#bfb,stroke:#333
    style AIE fill:#fbf,stroke:#333
```

### Tabel Perbandingan Detail

| Role | Tanggung Jawab Utama | Tools Utama | Gaji Rata-rata (IDR/USD) |
|------|---------------------|-------------|--------------------------|
| **Data Analyst** | Visualisasi data, Dashboard, SQL queries, Laporan bisnis | Tableau, Power BI, SQL, Excel | Rp 8-15 juta / $4k-6k |
| **Data Scientist** | Eksperimen model, Feature engineering, A/B testing, Statistika | Python, R, scikit-learn, Jupyter | Rp 15-30 juta / $8k-12k |
| **ML Engineer** | Deployment model, CI/CD pipeline, Monitoring, Scalability | Docker, Kubernetes, MLflow, TF Serving | Rp 20-40 juta / $10k-15k |
| **AI Engineer** | End-to-end pipeline, Optimization, System design, MLOps | TensorFlow, PyTorch, Airflow, Kubeflow | Rp 25-50 juta / $12k-18k |

**Sumber:** Glassdoor, Indeed, LinkedIn Salary (2024)

---

### Skill Tree AI Engineer

```mermaid
flowchart TD
    subgraph Level1["Level 1: Foundation (3-4 bulan)"]
        M1["📐 Matematika<br>• Linear Algebra (matriks, vektor)<br>• Kalkulus (turunan, gradien)<br>• Statistika & Probabilitas"]
        P1["🐍 Python Dasar<br>• Syntax & Data Structures<br>• Functions & OOP<br>• Libraries (NumPy)"]
    end
    
    subgraph Level2["Level 2: Data Wrangling (2-3 bulan)"]
        D1["📊 pandas & numpy<br>• Data manipulation<br>• Array operations"]
        D2["🧹 Data Cleaning<br>• Handling missing values<br>• Outlier detection"]
        D3["📈 Exploratory Data Analysis<br>• Visualization (matplotlib, seaborn)<br>• Statistical summary"]
    end
    
    subgraph Level3["Level 3: Modeling (3-4 bulan)"]
        M3["🤖 scikit-learn<br>• Model selection<br>• Pipeline"]
        M4["🔧 Feature Engineering<br>• Scaling, encoding<br>• Feature selection"]
        M5["📉 Model Evaluation & Tuning<br>• Cross-validation<br>• Hyperparameter tuning"]
    end
    
    subgraph Level4["Level 4: Deployment (2-3 bulan)"]
        DP1["🌐 API Development<br>• FastAPI / Flask"]
        DP2["🐳 Containerization<br>• Docker"]
        DP3["☁️ Cloud Platform<br>• AWS/GCP/Azure"]
    end
    
    Level1 --> Level2 --> Level3 --> Level4
    
    style Level1 fill:#f9f,stroke:#333
    style Level2 fill:#bbf,stroke:#333
    style Level3 fill:#bfb,stroke:#333
    style Level4 fill:#fbf,stroke:#333
```

---

### Sertifikasi Relevan

| Sertifikasi | Provider | Tingkat | Harga (USD) | Waktu Persiapan |
|-------------|----------|---------|-------------|-----------------|
| **TensorFlow Developer Certificate** | Google | Intermediate | $100 | 1-2 bulan |
| **AWS Certified Machine Learning - Specialty** | Amazon | Advanced | $300 | 2-3 bulan |
| **Databricks ML Associate** | Databricks | Intermediate | $200 | 1-2 bulan |
| **Azure Data Scientist Associate** | Microsoft | Intermediate | $165 | 1-2 bulan |
| **Deep Learning Specialization** | DeepLearning.AI | Beginner-Intermediate | $49/bulan | 2-3 bulan |

**Sumber:**
- TensorFlow: https://www.tensorflow.org/certificate
- AWS ML: https://aws.amazon.com/certification/certified-machine-learning-specialty/
- Databricks: https://www.databricks.com/learn/certification

---

### Portofolio yang Menarik Recruiter

```mermaid
flowchart LR
    subgraph Bad["❌ Tidak Direkomendasikan"]
        B1["Hanya Notebook (.ipynb)<br>tanpa struktur"]
        B2["Dataset Titanic/MNIST saja<br>(terlalu umum)"]
        B3["Tidak ada dokumentasi<br>atau README"]
        B4["Kode berantakan<br>tanpa komentar"]
    end
    
    subgraph Good["✅ Direkomendasikan"]
        G1["End-to-end project<br>Dari data ke deployment API"]
        G2["Problem bisnis nyata<br>(bukan dataset akademik)"]
        G3["Clean code + dokumentasi<br>README yang jelas"]
        G4["Unit test + CI/CD pipeline"]
        G5["Monitoring + logging<br>setelah deployment"]
    end
```

### Checklist Portofolio End-to-End

```mermaid
flowchart TD
    P1[1. Problem Definition<br>Definisi masalah bisnis] --> P2[2. Data Collection<br>Kaggle, API, Web Scraping]
    P2 --> P3[3. EDA & Visualization<br>pandas, matplotlib, seaborn]
    P3 --> P4[4. Feature Engineering<br>pipeline preprocessing]
    P4 --> P5[5. Model Training & Tuning<br>scikit-learn, Optuna]
    P5 --> P6[6. Model Evaluation<br>cross-validation, metrics]
    P6 --> P7[7. Model Deployment<br>FastAPI, Docker, Cloud]
    P7 --> P8[8. Monitoring<br>MLflow, Evidently AI]
    
    style P1 fill:#f9f,stroke:#333
    style P2 fill:#bbf,stroke:#333
    style P3 fill:#bfb,stroke:#333
    style P4 fill:#fbf,stroke:#333
    style P5 fill:#f9f,stroke:#333
    style P6 fill:#bbf,stroke:#333
    style P7 fill:#bfb,stroke:#333
    style P8 fill:#fbf,stroke:#333
```

**Sumber:** Data Science Portfolio Guide - https://github.com/DataSciencePortfolio

---

## 1.5 Setup Environment

### Arsitektur Setup

```mermaid
flowchart TD
    subgraph Local["💻 Local Machine"]
        IDE["IDE/Editor<br>• VS Code<br>• Jupyter Lab<br>• PyCharm"]
        Python["Python 3.9+<br>• Virtual Environment<br>• pip/conda"]
        Git["Git<br>• Version Control<br>• Git Bash"]
    end
    
    subgraph Libraries["📚 Python Libraries"]
        DPL["Data Processing<br>• pandas<br>• numpy"]
        VIZ["Visualisasi<br>• matplotlib<br>• seaborn"]
        ML["Machine Learning<br>• scikit-learn<br>• (opsional: tensorflow, pytorch)"]
    end
    
    subgraph Remote["☁️ Remote Repository"]
        GH["GitHub<br>Portfolio Repository"]
    end
    
    Local --> Libraries
    Local --> Remote
```

### Langkah Instalasi (Lengkap)

```bash
# ============================================
# 1. INSTALL PYTHON
# ============================================
# Windows: Download dari https://www.python.org/downloads/
# Mac: brew install python@3.9
# Linux: sudo apt install python3.9 python3-pip

# ============================================
# 2. INSTALL VS CODE (Opsional, tapi direkomendasikan)
# ============================================
# Download dari: https://code.visualstudio.com/
# Ekstensi yang diinstall:
#   - Python (Microsoft)
#   - Jupyter
#   - GitLens
#   - Prettier

# ============================================
# 3. BUAT VIRTUAL ENVIRONMENT
# ============================================
# Membuat environment baru
python -m venv ai_env

# Aktivasi (Mac/Linux)
source ai_env/bin/activate

# Aktivasi (Windows)
# ai_env\Scripts\activate

# ============================================
# 4. INSTALL LIBRARIES
# ============================================
# Core libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Additional (opsional untuk development)
pip install jupyter notebook ipykernel black flake8

# Untuk deep learning (opsional, butuh resource besar)
# pip install tensorflow pytorch

# ============================================
# 5. VERIFIKASI INSTALASI
# ============================================
python -c "import pandas as pd; import numpy as np; import sklearn; print(f'pandas: {pd.__version__}'); print(f'numpy: {np.__version__}'); print(f'sklearn: {sklearn.__version__}'); print('✅ Setup berhasil!')"

# ============================================
# 6. INSTALL GIT
# ============================================
# Download dari: https://git-scm.com/
# Konfigurasi awal:
git config --global user.name "Nama Kamu"
git config --global user.email "email@kamu.com"
```

---

### Create GitHub Repository untuk Portofolio

```mermaid
flowchart LR
    A[1. Buat repo di GitHub.com<br>Klik New Repository] --> B[2. git clone <url>]
    B --> C[3. Buat struktur folder]
    C --> D[4. git add .]
    D --> E[5. git commit -m 'initial commit']
    E --> F[6. git push origin main]
    
    style A fill:#9cf,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#fbf,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#9cf,stroke:#333
```

### Struktur Portofolio yang Direkomendasikan

```
portfolio-ai/
│
├── README.md                          # Dokumentasi utama portfolio
├── requirements.txt                   # Daftar library yang digunakan
├── .gitignore                         # File yang tidak perlu di-commit
│
├── notebooks/                         # Jupyter notebooks untuk eksplorasi
│   ├── 01_oil_regression.ipynb       # Studi kasus regresi
│   ├── 02_diabetes_classification.ipynb  # Studi kasus klasifikasi
│   └── 03_mall_clustering.ipynb      # Studi kasus clustering
│
├── src/                               # Source code moduler
│   ├── __init__.py
│   ├── data_preprocessing.py         # Fungsi preprocessing
│   ├── models.py                     # Definisi model
│   └── utils.py                      # Utility functions
│
├── data/                              # Dataset (jangan di-commit jika besar)
│   ├── raw/                          # Data mentah
│   └── processed/                    # Data yang sudah diproses
│
├── deployment/                        # Kode untuk deployment
│   ├── app.py                        # FastAPI/Flask app
│   ├── Dockerfile                    # Docker configuration
│   └── requirements_deploy.txt       # Requirements khusus deployment
│
├── tests/                             # Unit testing
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── docs/                              # Dokumentasi tambahan
│   └── project_report.pdf
│
└── .github/                           # GitHub Actions (CI/CD)
    └── workflows/
        └── ci.yml
```

### Contoh .gitignore untuk Proyek AI

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python

# Virtual Environment
venv/
env/
ai_env/

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb~

# Data files (besar)
*.csv
*.pkl
*.h5
*.parquet
data/raw/
data/processed/

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/

# Model files (besar)
*.joblib
*.pkl
models/
```

### Contoh README.md untuk Portofolio

```markdown
# AI Portfolio - [Nama Kamu]

## Tentang Saya
Data Scientist/AI Engineer dengan fokus pada [bidang spesialisasi].

## Proyek Unggulan

### 1. Prediksi Produksi Minyak (Regresi)
- **Dataset**: Oil production data
- **Metode**: Random Forest Regressor
- **Hasil**: R² = 0.89, RMSE = 125 barel
- **[Link ke notebook](notebooks/01_oil_regression.ipynb)**

### 2. Deteksi Diabetes (Klasifikasi)
- **Dataset**: PIMA Indian Diabetes
- **Metode**: Logistic Regression, Random Forest
- **Hasil**: Accuracy 85%, F1-score 0.82
- **[Link ke notebook](notebooks/02_diabetes_classification.ipynb)**

### 3. Segmentasi Pelanggan Mall (Clustering)
- **Dataset**: Mall customer data
- **Metode**: K-Means
- **Hasil**: 5 segmen optimal dengan silhouette score 0.55
- **[Link ke notebook](notebooks/03_mall_clustering.ipynb)**

## Skills
- Python, pandas, numpy, scikit-learn
- Data visualization (matplotlib, seaborn)
- Git, GitHub

## Kontak
- LinkedIn: [link]
- Email: [email]
- GitHub: [username]
```

---

## Ringkasan Akhir Sesi 1

```mermaid
mindmap
  root((Sesi 1<br>Fundamental AI<br>& Roadmap Karir))
    Konsep Dasar
      AI vs ML vs DL
      Rule-based vs Data-driven
      Aplikasi sehari-hari
    3 Jenis ML
      Supervised
        Regresi (angka kontinu)
        Klasifikasi (kategori)
      Unsupervised
        Clustering (kelompok alami)
      Kapan menggunakan
    Studi Kasus
      Oil - Regresi
      Diabetes - Klasifikasi
      Mall - Clustering
    Roadmap Karir
      4 Role utama
      Skill tree 4 level
      Sertifikasi
      Portfolio end-to-end
    Setup Environment
      Python + Jupyter
      4 library utama
      GitHub portfolio
```

---

## Tugas Praktik Sesi 1

### Tugas Wajib
1. **Install** semua tools yang disebutkan di bagian 1.5
2. **Buat repository** GitHub dengan nama `ai-portfolio-[namamu]`
3. **Clone** repository tersebut ke lokal
4. **Buat virtual environment** dan install library yang diperlukan
5. **Buat notebook sederhana** yang menampilkan:
   - Load dataset (gunakan seaborn bawaan: `load_dataset('tips')`)
   - EDA sederhana (info, describe, histogram)
   - Simpan ke repository dan push ke GitHub

---

## Daftar Pustaka Lengkap

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning*. Springer.
4. Chollet, F. (2018). *Deep Learning with Python*. Manning.
5. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. O'Reilly.
7. VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly.
8. Gambar referensi: [Mercubuana Yogya](https://imam.mercubuana-yogya.ac.id/wp-content/uploads/2024/11/img.197Jenis-jenis-utama-Machine-Learning-Pembelajaran-Mesin.jpg)
9. Kaggle Datasets: https://www.kaggle.com/
10. UCI Machine Learning Repository: https://archive.ics.uci.edu/

---
