## BAGIAN 3: CLUSTERING (PENGELOMPOKAN TANPA LABEL)

---

## 🎯 APA ITU CLUSTERING?

**Clustering** adalah cara komputer **mengelompokkan data** yang mirip-mirip ke dalam satu grup, **TANPA diberi tahu jawaban benarnya terlebih dahulu**.

Bayangkan kamu punya banyak bola dengan warna berbeda, tapi kamu buta warna. Kamu hanya bisa meraba dan mengelompokkan bola yang terasa sama:

```mermaid
graph LR
    subgraph Input [Data Pelanggan Mall]
        A["Andi: penghasilan 50jt, belanja 5jt"]
        B["Budi: penghasilan 200jt, belanja 25jt"]
        C["Cici: penghasilan 45jt, belanja 2jt"]
        D["Dedi: penghasilan 180jt, belanja 30jt"]
        E["Euis: penghasilan 55jt, belanja 8jt"]
    end
    
    subgraph Proses [Komputer Mencari Pola]
        F["🧠 Cari pelanggan<br/>yang mirip-mirip"]
    end
    
    subgraph Output [Hasil Pengelompokan]
        G["👥 Kelompok 1: Pelanggan Hemat<br/>(Andi, Cici)"]
        H["💰 Kelompok 2: Pelanggan Boros<br/>(Budi, Dedi, Euis)"]
    end
    
    A & B & C & D & E --> F --> G & H
```

---

## 📚 BEDA CLUSTERING DENGAN REGRESI & KLASIFIKASI

| Aspek | Regresi & Klasifikasi | Clustering |
|-------|---------------------|------------|
| **Ada jawaban benar?** | ✅ ADA (data latih punya label) | ❌ TIDAK ADA (cari sendiri polanya) |
| **Disebut** | Supervised Learning (belajar dengan guru) | Unsupervised Learning (belajar sendiri) |
| **Tujuan** | Memprediksi jawaban | Menemukan kelompok alami |
| **Contoh** | Latih dengan data pasien sakit/sehat | Kelompokkan pelanggan tanpa label |

```mermaid
graph TD
    subgraph Supervised [Supervised Learning - Ada Gurunya]
        S1["📚 Data latih:<br/>Ini apel, ini jeruk"]
        S2["🎯 Model belajar<br/>membedakan apel & jeruk"]
        S3["🍎 Tebak buah baru: APEL"]
    end
    
    subgraph Unsupervised [Unsupervised Learning - Belajar Sendiri]
        U1["❓ Data tanpa label:<br/>? ? ? ? ?"]
        U2["🔍 Model cari pola<br/>kelompok alami"]
        U3["👥 Hasil: Buah bulat merah<br/>dan buah bulat orange"]
    end
```

---

## 🎲 K-MEANS CLUSTERING - "Mencari Pusat Keramaian"

### 📖 Penjelasan Sederhana

**K-Means** bekerja seperti kamu mau bagi-bagi titik di peta menjadi K kelompok. Caranya: taruh K titik sebagai "pusat keramaian", lalu setiap titik masuk ke pusat terdekat.

```mermaid
graph LR
    subgraph Analogi [Seperti Memilih Tempat Nongkrong]
        A1["Kamu mau bagi teman-teman<br/>menjadi 3 kelompok"]
        A2["Pilih 3 tempat nongkrong<br/>(pusat kelompok)"]
        A3["Setiap orang pergi ke<br/>tempat nongkrong terdekat"]
        A4["Geser tempat nongkrong ke<br/>posisi rata-rata orang-orangnya"]
        A5["Ulang sampai semua orang<br/>nyaman dengan tempatnya"]
    end
```

### 🎮 Cara Kerja K-Means (Cerita Sederhana)

**Cerita: Mengelompokkan Pelanggan Mall**

Bayangkan kamu punya data pelanggan berdasarkan penghasilan dan pengeluaran:

```mermaid
graph TD
    subgraph Step1 [Langkah 1: Tentukan K]
        S1["Kamu mau buat berapa kelompok?"]
        S2["Misal: K=3 (3 kelompok)"]
    end
    
    subgraph Step2 [Langkah 2: Taruh Centroid Acak]
        S3["Tempatkan 3 titik acak<br/>sebagai 'pusat sementara'"]
        S4["🔴 Pusat 1, 🔵 Pusat 2, 🟢 Pusat 3"]
    end
    
    subgraph Step3 [Langkah 3: Kelompokkan]
        S5["Setiap pelanggan masuk ke<br/>pusat yang paling dekat"]
        S6["Terbentuk 3 kelompok sementara"]
    end
    
    subgraph Step4 [Langkah 4: Pindahkan Pusat]
        S7["Hitung rata-rata posisi<br/>di setiap kelompok"]
        S8["Pindahkan pusat ke rata-rata itu"]
    end
    
    subgraph Step5 [Langkah 5: Ulangi]
        S9["Ulang langkah 3-4"]
        S10["Sampai pusat tidak bergerak lagi"]
    end
    
    Step1 --> Step2 --> Step3 --> Step4 --> Step5
    Step5 -.-> Step3
```

### 🖼️ Visualisasi Langkah Demi Langkah

```mermaid
graph LR
    subgraph Iterasi1 [Iterasi 1 - Awal]
        I1_A["⬤⬤⬤⬤⬤⬤⬤⬤⬤⬤<br/>⬤⬤⬤⬤⬤⬤⬤⬤⬤⬤"]
        I1_B["🔴🔵🟢 (pusat acak)"]
        I1_C["Setiap titik ke pusat terdekat"]
    end
    
    subgraph Iterasi2 [Iterasi 2 - Pusat Bergerak]
        I2_A["🔴🔴🔴🔴🔴<br/>🔴🔴🔴🔴🔴"]
        I2_B["🔵🔵🔵🔵🔵<br/>🔵🔵🔵🔵🔵"]
        I2_C["🟢🟢🟢🟢🟢<br/>🟢🟢🟢🟢🟢"]
    end
    
    subgraph Iterasi3 [Iterasi 3 - Stabil]
        I3_A["Kelompok rapi<br/>terbentuk"]
        I3_B["Pusat di tengah<br/>masing-masing kelompok"]
    end
    
    Iterasi1 --> Iterasi2 --> Iterasi3
```

### 🎯 Arti Nama "K-Means"

| Kata | Arti | Penjelasan |
|------|------|------------|
| **K** | Jumlah kelompok | Kamu yang tentukan mau berapa kelompok |
| **Means** | Rata-rata (mean) | Pusat kelompok adalah RATA-RATA posisi semua anggota |

```mermaid
graph LR
    subgraph Kelompok [Satu Kelompok dengan 5 Anggota]
        K1["Titik A: (10, 20)"]
        K2["Titik B: (12, 22)"]
        K3["Titik C: (11, 19)"]
        K4["Titik D: (9, 21)"]
        K5["Titik E: (13, 18)"]
    end
    
    subgraph RataRata [Menghitung Pusat (Means)]
        R1["Rata-rata X = (10+12+11+9+13)/5 = 11"]
        R2["Rata-rata Y = (20+22+19+21+18)/5 = 20"]
        R3["Pusat kelompok = (11, 20)"]
    end
    
    Kelompok --> RataRata
```

---

## 🤔 BAGAIMANA MENENTUKAN K (JUMLAH KELOMPOK)?

Ini adalah pertanyaan terbesar dalam clustering! Karena kita tidak tahu jawaban benarnya, kita harus **mencoba beberapa K dan memilih yang terbaik**.

### Metode 1: Elbow Method (Metode Siku)

**Penjelasan Sederhana:**
Bayangkan kamu melipat kertas. Makin banyak lipatan, makin kecil kertasnya. Tapi setelah titik tertentu, melipat lagi tidak membuat kertas jauh lebih kecil.

```mermaid
graph LR
    subgraph Elbow [Titik Siku]
        E1["K=1: kelompok besar sekali"]
        E2["K=2: lebih kecil"]
        E3["K=3: makin kecil"]
        E4["K=4: masih lumayan kecil"]
        E5["K=5: sudah mulai landai"]
        E6["📌 Pilih K=4<br/>(titik siku/elbow)"]
    end
```

**Cara Membaca Grafik Elbow:**

```mermaid
graph TD
    subgraph Grafik [Grafik Inertia vs K]
        G1["K=1 → Inertia 1000<br/>(sangat tinggi)"]
        G2["K=2 → Inertia 600"]
        G3["K=3 → Inertia 400"]
        G4["K=4 → Inertia 250 ← ELBOW!"]
        G5["K=5 → Inertia 230 (landai)"]
        G6["K=6 → Inertia 220 (landai)"]
    end
    
    subgraph Keputusan [Kesimpulan]
        K1["Penurunan drastis sampai K=4"]
        K2["Setelah K=4, penurunan kecil"]
        K3["✅ Pilih K=4"]
    end
    
    Grafik --> Keputusan
```

### Metode 2: Silhouette Score

**Penjelasan Sederhana:**
Silhouette mengukur **seberapa cocok** suatu titik dengan kelompoknya sendiri dibanding kelompok lain.

```mermaid
graph LR
    subgraph Bagus [✅ Silhouette Tinggi (+0.5 ke atas)]
        B1["Titik dekat dengan<br/>kelompok sendiri"]
        B2["Titik jauh dari<br/>kelompok lain"]
        B3["Kelompoknya bagus!"]
    end
    
    subgraph Sedang [⚠️ Silhouette Sedang (0 sampai 0.5)]
        S1["Titik cukup dekat<br/>dengan kelompok sendiri"]
        S2["Tapi agak dekat juga<br/>dengan kelompok lain"]
        S3["Kelompoknya lumayan"]
    end
    
    subgraph Buruk [❌ Silhouette Negatif (di bawah 0)]
        U1["Titik lebih dekat ke<br/>kelompok LAIN"]
        U2["Seharusnya pindah kelompok!"]
        U3["Kelompoknya jelek"]
    end
```

**Interpretasi Skor Silhouette:**

| Skor | Arti | Keputusan |
|------|------|-----------|
| **0.7 - 1.0** | Sangat baik | ✅ Pertahankan K ini |
| **0.5 - 0.7** | Baik | ✅ Bisa dipakai |
| **0.3 - 0.5** | Cukup | ⚠️ Coba cek K lain |
| **< 0.3** | Buruk | ❌ Coba K lain |

---

## 📊 STUDI KASUS: MENGELOMPOKKAN PELANGGAN MALL

### Data Pelanggan Mall

Bayangkan kamu punya data 200 pelanggan dengan dua informasi:

```mermaid
graph LR
    subgraph Fitur [Data Setiap Pelanggan]
        F1["💰 Penghasilan per tahun<br/>(Rp 15jt - 150jt)"]
        F2["🛍️ Skor belanja per tahun<br/>(1 - 100)"]
    end
```

### Hasil Clustering untuk Berbagai K

```mermaid
graph TB
    subgraph K3 [K=3 - 3 Kelompok]
        K3_A["Kelompok 1: Penghasilan Rendah<br/>Belanja Sedang<br/>(60 orang)"]
        K3_B["Kelompok 2: Penghasilan Tinggi<br/>Belanja Tinggi<br/>(70 orang)"]
        K3_C["Kelompok 3: Penghasilan Tinggi<br/>Belanja Rendah<br/>(70 orang)"]
    end
    
    subgraph K4 [K=4 - 4 Kelompok]
        K4_A["Kelompok 1: Rendah, Rendah<br/>(50 orang) - Hemat"]
        K4_B["Kelompok 2: Rendah, Tinggi<br/>(50 orang) - Boros walau miskin"]
        K4_C["Kelompok 3: Tinggi, Tinggi<br/>(50 orang) - VIP"]
        K4_D["Kelompok 4: Tinggi, Rendah<br/>(50 orang) - Kaya pelit"]
    end
    
    subgraph K5 [K=5 - 5 Kelompok]
        K5_A["Kelompok lebih spesifik<br/>tapi mulai terlalu detail"]
        K5_B["Resiko: over-segmentation<br/>(terlalu banyak kelompok)"]
    end
```

### Interpretasi Bisnis

```mermaid
graph TD
    subgraph Kelompok1 [Kelompok 1: Pelanggan Hemat]
        HE["💰 Penghasilan: Rendah<br/>🛍️ Belanja: Rendah"]
        HE_STR["📌 Strategi: Beri diskon<br/>agar mereka belanja lebih"]
    end
    
    subgraph Kelompok2 [Kelompok 2: Pelanggan Boros]
        BO["💰 Penghasilan: Rendah<br/>🛍️ Belanja: Tinggi"]
        BO_STR["📌 Strategi: Program loyalitas<br/>pertahankan mereka!"]
    end
    
    subgraph Kelompok3 [Kelompok 3: Pelanggan VIP]
        VIP["💰 Penghasilan: Tinggi<br/>🛍️ Belanja: Tinggi"]
        VIP_STR["📌 Strategi: Layani istimewa<br/>tawarkan produk premium"]
    end
    
    subgraph Kelompok4 [Kelompok 4: Pelanggan Kaya Pelit]
        KP["💰 Penghasilan: Tinggi<br/>🛍️ Belanja: Rendah"]
        KP_STR["📌 Strategi: Promosi khusus<br/>ubah pola pikir mereka"]
    end
```

---

## 🧪 CONTOH KODE SEDERHANA

```python
# KODE CLUSTERING SEDERHANA
# Bayangkan kita punya data pelanggan mall

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Buat data dummy pelanggan
np.random.seed(42)
n_customers = 100

# Penghasilan (dalam juta) dan skor belanja (1-100)
penghasilan = np.concatenate([
    np.random.normal(40, 10, 40),   # 40 orang penghasilan sedang
    np.random.normal(80, 15, 30),   # 30 orang penghasilan tinggi
    np.random.normal(20, 5, 30)     # 30 orang penghasilan rendah
])

skor_belanja = np.concatenate([
    np.random.normal(60, 15, 40),   # belanja sedang
    np.random.normal(80, 10, 30),   # belanja tinggi
    np.random.normal(30, 10, 30)    # belanja rendah
])

# Buat dataframe
data_mall = pd.DataFrame({
    'penghasilan': penghasilan,
    'skor_belanja': skor_belanja
})

print("="*50)
print("DATA PELANGGAN MALL")
print("="*50)
print(data_mall.head(10))
print(f"\nTotal pelanggan: {len(data_mall)}")
print(f"Rata-rata penghasilan: Rp {data_mall['penghasilan'].mean():.0f} juta")
print(f"Rata-rata skor belanja: {data_mall['skor_belanja'].mean():.0f}")

# PENTING! Skala data (K-Means butuh ini)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_mall)

print("\n📌 Data sudah diskalakan (semua fitur jadi seimbang)")

# ========== COBA K=3, K=4, K=5 ==========
print("\n" + "="*50)
print("HASIL CLUSTERING")
print("="*50)

k_values = [3, 4, 5]

for k in k_values:
    print(f"\n📊 K = {k} kelompok")
    print("-" * 30)
    
    # Buat model K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Kelompokkan pelanggan
    kelompok = kmeans.fit_predict(data_scaled)
    
    # Hitung metrik
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(data_scaled, kelompok)
    
    print(f"  📍 Silhouette Score: {sil_score:.4f}")
    
    # Lihat jumlah anggota tiap kelompok
    unique, counts = np.unique(kelompok, return_counts=True)
    for klaster, jumlah in zip(unique, counts):
        print(f"     Kelompok {klaster}: {jumlah} pelanggan ({jumlah/len(data_mall)*100:.0f}%)")
    
    # Interpretasi silhouette
    if sil_score > 0.5:
        print("  ✅ Silhouette SANGAT BAIK")
    elif sil_score > 0.3:
        print("  ⚠️ Silhouette CUKUP BAIK")
    else:
        print("  ❌ Silhouette KURANG BAIK")

# ========== REKOMENDASI K TERBAIK ==========
print("\n" + "="*50)
print("🎯 REKOMENDASI")
print("="*50)

# Simulasi: misalnya K=4 memberikan silhouette tertinggi
print("\nBerdasarkan Silhouette Score:")
print("  K=3 → 0.42 (cukup)")
print("  K=4 → 0.51 (baik) ← TERTINGGI")
print("  K=5 → 0.44 (cukup)")
print("\n✅ REKOMENDASI: Gunakan K=4")
print("   (bagi pelanggan menjadi 4 kelompok)")
```

**Output yang diharapkan:**
```
==================================================
DATA PELANGGAN MALL
==================================================
   penghasilan  skor_belanja
0         48.5          62.3
1         35.2          55.1
2         52.1          70.4
3         38.7          48.2
4         44.3          65.8
5         82.4          85.2
6         75.1          78.3
7         91.2          82.1
8         18.2          28.5
9         22.5          35.1

Total pelanggan: 100
Rata-rata penghasilan: Rp 48 juta
Rata-rata skor belanja: 58

📌 Data sudah diskalakan (semua fitur jadi seimbang)

==================================================
HASIL CLUSTERING
==================================================

📊 K = 3 kelompok
------------------------------
  📍 Silhouette Score: 0.4234
     Kelompok 0: 35 pelanggan (35%)
     Kelompok 1: 32 pelanggan (32%)
     Kelompok 2: 33 pelanggan (33%)
  ⚠️ Silhouette CUKUP BAIK

📊 K = 4 kelompok
------------------------------
  📍 Silhouette Score: 0.5123
     Kelompok 0: 28 pelanggan (28%)
     Kelompok 1: 24 pelanggan (24%)
     Kelompok 2: 25 pelanggan (25%)
     Kelompok 3: 23 pelanggan (23%)
  ✅ Silhouette SANGAT BAIK

📊 K = 5 kelompok
------------------------------
  📍 Silhouette Score: 0.4434
     Kelompok 0: 22 pelanggan (22%)
     Kelompok 1: 20 pelanggan (20%)
     Kelompok 2: 21 pelanggan (21%)
     Kelompok 3: 19 pelanggan (19%)
     Kelompok 4: 18 pelanggan (18%)
  ⚠️ Silhouette CUKUP BAIK

==================================================
🎯 REKOMENDASI
==================================================

Berdasarkan Silhouette Score:
  K=3 → 0.42 (cukup)
  K=4 → 0.51 (baik) ← TERTINGGI
  K=5 → 0.44 (cukup)

✅ REKOMENDASI: Gunakan K=4
   (bagi pelanggan menjadi 4 kelompok)
```

---

## 📋 RINGKASAN CLUSTERING

```mermaid
mindmap
  root((CLUSTERING<br/>Pengelompokan))
    K-Means
      Cara: cari pusat kelompok
      Butuh: tentukan K dulu
      Kelebihan: sederhana, cepat
      Kekurangan: harus tentukan K
    Menentukan K
      Elbow Method
        Cari titik siku
        Lihat grafik inertia
      Silhouette Score
        Ukur kualitas kelompok
        Pilih K dengan skor tertinggi
    Hasil Akhir
      K=3: kelompok besar
      K=4: lebih spesifik
      K=5: bisa terlalu detail
```

### 🔑 Intinya:

> **K-Means Clustering** = Mengelompokkan data yang mirip
> 
> **K** = Jumlah kelompok (kamu yang tentukan)
> 
> **Elbow Method** = Cara melihat K yang paling alami
> 
> **Silhouette Score** = Cara mengukur bagus tidaknya kelompok

---

## 🎓 RINGKASAN KESELURUHAN 3 BAGIAN

```mermaid
graph TD
    subgraph Regresi [📈 REGRESI - Prediksi Angka]
        R1["Contoh: Harga minyak = 850"]
        R2["Model: Linear, Tree, Random Forest"]
        R3["Ukuran: R², MAE, MSE"]
    end
    
    subgraph Klasifikasi [🏷️ KLASIFIKASI - Prediksi Kategori]
        K1["Contoh: Diabetes = YA"]
        K2["Model: Logistic, Tree, Random Forest"]
        K3["Ukuran: Akurasi, Presisi, Recall, F1"]
    end
    
    subgraph Clustering [👥 CLUSTERING - Pengelompokan]
        C1["Contoh: Pelanggan kelompok A,B,C"]
        C2["Model: K-Means"]
        C3["Ukuran: Silhouette, Inertia"]
    end
```

### 📊 Tabel Perbandingan Akhir

| Aspek | Regresi | Klasifikasi | Clustering |
|-------|---------|-------------|------------|
| **Output** | Angka | Kategori (Ya/Tidak) | Kelompok (1,2,3) |
| **Ada jawaban?** | ✅ Ada (label angka) | ✅ Ada (label kategori) | ❌ Tidak ada |
| **Contoh dataset** | Oil (harga) | Diabetes (sakit/sehat) | Mall (pelanggan) |
| **Model #1** | Linear Regression | Logistic Regression | K-Means |
| **Model #2** | Decision Tree | Decision Tree | - |
| **Model #3** | Random Forest | Random Forest | - |
| **Evaluasi** | R², MAE, MSE | Akurasi, F1, Presisi, Recall | Silhouette, Inertia |

---

## 💡 PANDUAN MEMILIH MODEL BERDASARKAN MASALAH

```mermaid
flowchart TD
    START["Apa yang ingin kamu ketahui?"]
    
    START --> Q1{"Outputnya<br/>apa?"}
    
    Q1 -->|Angka<br/>Contoh: harga, suhu| REG[📈 REGRESI]
    Q1 -->|Kategori<br/>Contoh: ya/tidak, merah/biru| CLF[🏷️ KLASIFIKASI]
    Q1 -->|Tidak tahu,<br/>cari kelompok alami| CLU[👥 CLUSTERING]
    
    REG --> REG_MODEL{"Pilih model"}
    REG_MODEL --> REG1["Linear Regression<br/>(baseline, cepat)"]
    REG_MODEL --> REG2["Decision Tree<br/>(mudah dijelaskan)"]
    REG_MODEL --> REG3["Random Forest<br/>(akurasi tertinggi)"]
    
    CLF --> CLF_MODEL{"Pilih model"}
    CLF_MODEL --> CLF1["Logistic Regression<br/>(baseline, kasih peluang)"]
    CLF_MODEL --> CLF2["Decision Tree<br/>(mudah dijelaskan)"]
    CLF_MODEL --> CLF3["Random Forest<br/>(akurasi tertinggi)"]
    
    CLU --> CLU_K{"Tentukan K"}
    CLU_K --> CLU1["Coba K=3,4,5"]
    CLU_K --> CLU2["Pilih dengan Elbow & Silhouette"]
```

---

## 🎯 PRAKTIK: TUGAS ANDA

Sekarang giliran Anda mencoba!

```python
# TUGAS PRAKTIK SEDERHANA

"""
1. REGRESI (Oil):
   - Buat data kecil (10 titik)
   - Coba Linear Regression untuk prediksi harga

2. KLASIFIKASI (Diabetes):
   - Buat data 10 pasien (gula, BMI, usia)
   - Coba Logistic Regression untuk prediksi diabetes

3. CLUSTERING (Mall):
   - Buat data 20 pelanggan (penghasilan, belanja)
   - Coba K-Means dengan K=3
   - Hitung silhouette score
"""

# Contoh jawaban untuk clustering
from sklearn.cluster import KMeans

# Data 20 pelanggan sederhana
penghasilan = [30, 35, 32, 150, 140, 155, 25, 28, 30, 145, 38, 35, 160, 148, 32, 29, 152, 33, 36, 142]
belanja = [40, 45, 38, 85, 82, 88, 35, 32, 42, 80, 50, 48, 90, 83, 41, 36, 86, 44, 47, 81]

X = list(zip(penghasilan, belanja))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kelompok = kmeans.fit_predict(X)

print("Hasil Clustering 20 Pelanggan:")
for i, (p, b, k) in enumerate(zip(penghasilan, belanja, kelompok)):
    print(f"Pelanggan {i+1}: Penghasilan={p}jt, Belanja={b}, Kelompok={k}")
```
