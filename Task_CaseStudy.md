# 🛠️ TUGAS: CASE STUDY END-TO-END (REGRESI, KLASIFIKASI, KLUSTERING)

Selamat! Kamu telah sampai di tahap integrasi. Sesuai dengan materi di repository ini, kamu diminta untuk menyelesaikan 3 studi kasus. Di bawah ini adalah daftar dataset publik (Raw CSV) yang bisa kamu gunakan langsung di Google Colab.

---

# 🇮🇩 POJOK INDONESIA: DATASET LOKAL (REKOMENDASI!)
Gunakan dataset dengan konteks lokal Indonesia untuk portofolio yang lebih relevan dengan industri dalam negeri.

### 📈 Regresi: Prediksi Saham BBCA (Perbankan ID)
- **Link Raw CSV**: [🔗 BBCA.JK Data](https://raw.githubusercontent.com/kanadarma/indonesia-stock-market-data-set/master/BBCA.JK.csv)
- **Tugas**: Prediksi harga `Close` berdasarkan volume dan harga pembukaan.

### 🏥 Klasifikasi: Kualitas Udara Jakarta (ISPU)
- **Link Raw CSV**: [🔗 ISPU Jakarta 2021](https://raw.githubusercontent.com/yofisunarta/ispu-jakarta/master/data/ispu_jakarta_2021.csv)
- **Tugas**: Tebak kategori kualitas udara (Baik/Sedang/Tidak Sehat) berdasarkan polutan (PM10, SO2, CO, dll).

### 🛍️ Klustering: Konsumsi Pangan Per Provinsi
- **Link Raw CSV**: [🔗 Konsumsi Pangan ID](https://raw.githubusercontent.com/KurniawanDwi/Dataset-Indonesia/master/konsumsi_pangan.csv)
- **Tugas**: Kelompokkan provinsi di Indonesia berdasarkan pola konsumsi karbohidrat, protein, dan lemak.

---

# 📈 CASE STUDY 1: REGRESI (GLOBAL)
- **Prediksi Harga Rumah**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
- **Prediksi Efisiensi BBM**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv)

---

# 🏥 CASE STUDY 2: KLASIFIKASI (GLOBAL)
- **Diagnosa Diabetes (Pima)**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Titanic Survival**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

# 🛍️ CASE STUDY 3: KLUSTERING (GLOBAL)
- **Mall Customer Segmentation**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/SteffiPauly/Machine-Learning-Datasets/master/Mall_Customers.csv)
- **Wine Clustering**:  
  [🔗 Link Raw CSV](https://raw.githubusercontent.com/sharmaroshan/Wine-Clustering/master/Wine.csv)

---

# 📝 INSTRUKSI PENGERJAAN:

Gunakan fungsi `pd.read_csv("link_pilihan_kamu")` langsung di Google Colab.
1. Gunakan struktur folder modular (`src/preprocess.py`, `src/train.py`).
2. Simpan 3 model terbaik hasil pengerjaanmu ke dalam folder `models/`.
3. Buat satu file PDF/README yang berisi screenshot hasil evaluasi tiap Case Study.

---

> **"Membangun solusi untuk masalah lokal adalah langkah pertama menjadi AI Engineer yang berdampak."**
