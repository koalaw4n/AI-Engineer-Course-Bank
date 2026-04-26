# 📑 Case Study Assignment: Indonesian Public Data Analytics

## 📝 Deskripsi Tugas
Tugas ini dirancang untuk menguji kemampuan Anda dalam mengolah data publik Indonesia melalui alur kerja *Machine Learning* yang lengkap. Anda akan belajar bagaimana mengubah data mentah menjadi wawasan (*insight*) dan model prediksi yang akurat.

---

## 📂 Alur Kerja (Workflow) & Penjelasan Langkah
Ikuti 7 langkah standar industri ini untuk setiap tugas:

1.  **Data Preparation**: Tahap awal untuk memastikan data siap diolah. Meliputi pemuatan file CSV, pengecekan nilai kosong (`isnull()`), dan pembersihan teks agar menjadi angka (numerik).
2.  **EDA (Exploratory Data Analysis)**: Mengenali data Anda. Gunakan statistik deskriptif dan grafik untuk melihat pola, tren, atau anomali sebelum membuat model.
3.  **Feature Engineering**: Proses mengubah data mentah menjadi fitur yang lebih informatif bagi model. Contohnya: memecah string koordinat menjadi angka Latitude/Longitude atau mengubah tanggal menjadi "usia".
4.  **Model Training**: Pemilihan algoritma dan proses "belajar" model dari data latih.
5.  **Model Validation**: Mengukur seberapa pintar model Anda menggunakan data uji. Metrik seperti Akurasi atau RMSE sangat penting di sini.
6.  **Tuning & Finalize**: Mencoba berbagai kombinasi parameter untuk mencari performa terbaik dan menyimpan model akhir.
7.  **Referensi**: Mencantumkan sumber dokumentasi untuk memudahkan penelusuran kembali.

---

## 🛠️ Tugas 1: Klasifikasi (Classification)
**Dataset**: [indonesia_volcanoes.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv)  
**Tujuan**: Memprediksi tipe/bentuk gunung berapi (`bentuk`) berdasarkan fitur geografis.

### 📋 Panduan Langkah Demi Langkah:
*   **Step 1**: Load data dan bersihkan kolom `tinggi_meter`. Hilangkan kata " meter" menggunakan `.str.replace()` agar kolom bisa dihitung secara matematis.
*   **Step 2**: Cek distribusi kelas `bentuk`. Apakah ada tipe gunung yang sangat sedikit? Visualisasikan dengan `sns.countplot()`.
*   **Step 3**: Pecah kolom `geolokasi`. Gunakan Regex untuk mengekstrak angka. Koordinat ini adalah fitur kunci untuk membedakan tipe gunung berdasarkan lokasi.
*   **Step 4**: Pisahkan data (X: fitur, y: target). Train menggunakan `RandomForestClassifier`.
*   **Step 5**: Gunakan `classification_report` untuk melihat presisi dan recall setiap tipe gunung.
*   **Step 6**: Eksperimen dengan parameter `n_estimators` (jumlah pohon) untuk melihat pengaruhnya terhadap akurasi.
*   **Step 7**: Dokumentasikan fungsi Regex yang Anda gunakan.

#### 💡 Contoh Kode:
```python
import re
import pandas as pd

# Contoh Regex untuk mengambil angka koordinat
def extract_coords(text):
    # Mencari pola angka desimal
    match = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(match[0]), float(match[1]) if len(match) >= 2 else (0, 0)

df['lat'], df['long'] = zip(*df['geolokasi'].map(extract_coords))
```

---

## 📍 Tugas 2: Klastering (Clustering)
**Dataset**: [indonesia_volcanoes.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv)  
**Tujuan**: Mengelompokkan gunung berapi berdasarkan lokasi geografis untuk memetakan zona vulkanik.

### 📋 Panduan Langkah Demi Langkah:
*   **Step 1**: Ambil fitur Latitude dan Longitude yang sudah bersih dari Tugas 1. Pastikan tidak ada nilai `NaN`.
*   **Step 2**: Visualisasikan lokasi gunung pada peta sederhana menggunakan `plt.scatter(long, lat)`. Anda akan melihat "garis" cincin api Indonesia.
*   **Step 3**: Penting! Lakukan `StandardScaler` agar skala Latitude dan Longitude seimbang sebelum masuk ke algoritma.
*   **Step 4**: Jalankan `KMeans` dengan beberapa nilai K (misal 1 sampai 10).
*   **Step 5**: Gunakan **Elbow Method**. Cari titik di mana penurunan inersia mulai melambat (membentuk "siku").
*   **Step 6**: Pilih K optimal, jalankan ulang model, dan tambahkan label klaster ke dataframe asli.
*   **Step 7**: Pelajari dokumentasi K-Means tentang inisialisasi `k-means++`.

#### 💡 Contoh Kode:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['lat', 'long']])

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

---

## 📈 Tugas 3: Regresi (Regression)
**Dataset**: [indonesia_public_companies.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_public_companies.csv)  
**Tujuan**: Memprediksi `jumlah_saham` berdasarkan `listing_age`.

### 📋 Panduan Langkah Demi Langkah:
*   **Step 1**: Ubah `tanggal_listing` menjadi tipe data datetime. Tangani data yang tidak valid jika ada.
*   **Step 2**: Gunakan `df.corr()` untuk melihat apakah ada hubungan linear antara fitur numerik.
*   **Step 3**: Buat fitur `listing_age`. Caranya: `Tahun_Sekarang - Tahun_Listing`. Fitur ini mencerminkan kematangan perusahaan di bursa.
*   **Step 4**: Gunakan `LinearRegression`. Fit model menggunakan `listing_age` sebagai X.
*   **Step 5**: Lihat skor `R-Squared`. Semakin mendekati 1, semakin baik model Anda menjelaskan data.
*   **Step 6**: Coba gunakan `PolynomialFeatures` jika hubungan datanya tidak benar-benar lurus (linear).
*   **Step 7**: Referensi: Dokumentasi `pd.to_datetime`.

#### 💡 Contoh Kode:
```python
# Menghitung usia perusahaan
df['thn_listing'] = pd.to_datetime(df['tanggal_listing']).dt.year
df['usia_listing'] = 2024 - df['thn_listing']

# Regresi Linear
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['usia_listing']], df['jumlah_saham'])
```

---

## 📤 Kriteria Pengumpulan
- **Format**: File `.ipynb` (Jupyter Notebook).
- **Narasi**: Sertakan penjelasan singkat (1-2 kalimat) mengapa Anda melakukan langkah tersebut di setiap sel Markdown.
- **Output**: Pastikan semua plot/grafik tampil sebelum file dikirim.
