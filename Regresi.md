## BAGIAN 1: REGRESI (PREDIKSI ANGKA)

---

## 🎯 APA ITU REGRESI?

**Regresi** adalah cara komputer menebak **angka**. 

Bayangkan kamu ingin menebak:
- 🔢 "Besok harga minyak berapa?" → Rp 850.000
- 🔢 "Nilai ujian kamu berapa?" → 85
- 🔢 "Suhu besok berapa?" → 28°C

Nah, regresi adalah teknik yang membantu komputer membuat tebakan angka yang akurat!

```mermaid
graph LR
    subgraph Input [Informasi yang Diketahui]
        A["📊 Produksi minyak<br/>sebanyak 100 barrel"]
        B["📈 Permintaan pasar<br/>tinggi"]
        C["📦 Stok minyak<br/>sedikit"]
    end
    
    subgraph Proses [Komputer Belajar]
        D["🧠 Model Regresi<br/>mencari pola"]
    end
    
    subgraph Output [Hasil Tebakan]
        E["💲 Harga minyak<br/>Rp 850.000"]
    end
    
    A & B & C --> D --> E
```

---

## 📚 TIGA JENIS MODEL REGRESI

Ada 3 model yang akan kita pelajari:

| No | Model | Gambaran Sederhana |
|----|-------|-------------------|
| 1 | **Linear Regression** | Seperti menggambar garis lurus di antara titik-titik data |
| 2 | **Decision Tree** | Seperti bermain tebak-tebakan dengan pertanyaan ya/tidak |
| 3 | **Random Forest** | Seperti mengumpulkan pendapat banyak teman lalu dirata-rata |

---

## 1️⃣ LINEAR REGRESSION - "Garis Lurus Sederhana"

### 📖 Penjelasan Sederhana

Bayangkan kamu punya kumpulan titik di kertas. Linear Regression adalah **cara menggambar GARIS LURUS** yang paling pas melewati titik-titik tersebut.

```mermaid
graph TD
    subgraph Analogi [Seperti Menimbang Badan]
        A1["Setiap kali makan lebih banyak<br/>berat badan naik"]
        A2["Hubungannya: MAKAN → BERAT<br/>(semakin banyak makan, semakin berat)"]
        A3["Linear Regression mencari<br/>'berapa banyak kenaikan berat<br/>setiap 1 porsi makan'"]
    end
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Menebak Harga Minyak**

```
Kamu punya data:
- Jika produksi 100 barrel → harga Rp 800.000
- Jika produksi 200 barrel → harga Rp 850.000  
- Jika produksi 300 barrel → harga Rp 900.000

Linear Regression akan melihat:
"Setiap produksi naik 100 barrel, harga naik Rp 50.000"

Maka rumusnya:
HARGA = (500 × PRODUKSI) + 750.000

Coba tebak produksi 250 barrel:
HARGA = (500 × 250) + 750.000 = Rp 875.000
```

### ✅ Kelebihan & ❌ Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Super sederhana, gampang dipahami | Hanya bisa garis lurus |
| Cepat banget (detik) | Kalau datanya melengkung, jelek hasilnya |
| Cocok untuk tebakan awal (baseline) | Gampang terganggu data aneh (outlier) |

### 💬 Kapan Pakai?

> "Pakai Linear Regression kalau kamu baru mulai dan ingin lihat gambaran kasar dulu"

```mermaid
graph LR
    subgraph Cocok [✅ Cocok untuk]
        C1["Data cenderung lurus"]
        C2["Pengen cepet dapet hasil"]
        C3["Butuh rumus yang simpel"]
    end
    
    subgraph TidakCocok [❌ Tidak cocok untuk]
        T1["Data melengkung/rumit"]
        T2["Butuh tebakan sangat akurat"]
    end
```

---

## 2️⃣ DECISION TREE - "Tebak-tebakan Ya/Tidak"

### 📖 Penjelasan Sederhana

Decision Tree bekerja seperti **permainan 20 pertanyaan**. Kamu bertanya ya/tidak terus menerus sampai sampai pada satu jawaban.

```mermaid
graph TD
    subgraph Game [🎲 Permainan Tebak Harga]
        START["Mulai: Tebak harga minyak"]
        
        START --> Q1["Apakah produksi > 200 barrel?"]
        
        Q1 -->|Ya| Q2["Apakah permintaan tinggi?"]
        Q1 -->|Tidak| Q3["Apakah stok menipis?"]
        
        Q2 -->|Ya| A1["💰 Harga: Rp 900.000"]
        Q2 -->|Tidak| A2["💰 Harga: Rp 850.000"]
        
        Q3 -->|Ya| A3["💰 Harga: Rp 800.000"]
        Q3 -->|Tidak| A4["💰 Harga: Rp 750.000"]
    end
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Memilih Restoran**

Bayangkan kamu mau milih restoran berdasarkan 3 pertanyaan:

```mermaid
graph TD
    R["Apakah murah?"] -->|Ya| R1["Apakah dekat?"]
    R -->|Tidak| R2["Apakah enak?"]
    
    R1 -->|Ya| RR1["✅ Restoran A<br/>(murah & dekat)"]
    R1 -->|Tidak| RR2["❌ Skip"]
    
    R2 -->|Ya| RR3["✅ Restoran B<br/>(mahal tapi enak)"]
    R2 -->|Tidak| RR4["❌ Skip"]
```

**Sama persis dengan Decision Tree!** Setiap pertanyaan memecah pilihan sampai ketemu jawaban.

### 🌳 Bagaimana Pohon Tumbuh?

```mermaid
graph TB
    subgraph Step1 [Langkah 1: Cari pertanyaan terbaik]
        S1["Pertanyaan apa yang paling<br/>membedakan harga?"]
        S2["Contoh: 'Produksi > 200?'<br/>atau 'Permintaan tinggi?'"]
    end
    
    subgraph Step2 [Langkah 2: Bagi data]
        S3["Data pecah jadi 2 kelompok:<br/>yang jawab YA dan TIDAK"]
    end
    
    subgraph Step3 [Langkah 3: Ulangi]
        S4["Di setiap kelompok, cari<br/>pertanyaan terbaik lagi"]
        S5["Terus sampai puas"]
    end
    
    Step1 --> Step2 --> Step3
```

### 🛑 Bahaya Overfitting (Menghafal)

```mermaid
graph LR
    subgraph Masalah [❌ Decision Tree Suka Menghafal]
        M1["Pohon terlalu dalam<br/>(banyak pertanyaan)"]
        M2["Menghafal data latih<br/>sempurna"]
        M3["Saat dites data baru<br/>JELEK! Error besar"]
    end
    
    subgraph Solusi [✅ Solusi]
        S1["Batasi kedalaman pohon"]
        S2["Jangan terlalu banyak pertanyaan"]
    end
    
    Masalah --> Solusi
```

**Contoh Sederhana:**
```
Pohon yang sehat (kedalaman 3):
- 3 pertanyaan → 8 kemungkinan jawaban

Pohon yang overfit (kedalaman 10):
- 10 pertanyaan → 1024 kemungkinan
- Terlalu detail, jadi hafal mati!
```

### 💬 Kapan Pakai?

> "Pakai Decision Tree kalau kamu mau lihat alur keputusan yang jelas dan mudah dijelaskan ke bos"

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Mudah dijelaskan ke orang lain | Suka menghafal (overfitting) |
| Bisa menangkap pola yang tidak lurus | Kurang stabil (sedikit perubahan data bisa ubah pohon banyak) |
| Tidak perlu repot skala data | Akurasi biasa aja |

---

## 3️⃣ RANDOM FOREST - "Musyawarah Banyak Ahli"

### 📖 Penjelasan Sederhana

Random Forest = **Kumpulan BANYAK Decision Tree**. Seperti rapat dewan: minta pendapat banyak orang, lalu ambil rata-ratanya.

```mermaid
graph TD
    subgraph Dewan [👥 Rapat Penentuan Harga]
        A["Pakar 1 (Pohon 1)<br/>menebak: Rp 900.000"]
        B["Pakar 2 (Pohon 2)<br/>menebak: Rp 850.000"]
        C["Pakar 3 (Pohon 3)<br/>menebak: Rp 880.000"]
        D["Pakar 4 (Pohon 4)<br/>menebak: Rp 870.000"]
        E["Pakar 5 (Pohon 5)<br/>menebak: Rp 890.000"]
    end
    
    subgraph Rapat [🤝 Musyawarah]
        F["Rata-rata semua pendapat"]
    end
    
    subgraph Keputusan [🎯 Hasil Akhir]
        G["💰 HARGA = Rp 878.000"]
    end
    
    A & B & C & D & E --> F --> G
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Mau beli HP baru**

Bayangin kamu minta saran ke 5 teman:

```mermaid
graph LR
    subgraph Teman1 [Teman 1]
        T1A["Lihat: harga"]
        T1B["Kesimpulan: Samsung"]
    end
    
    subgraph Teman2 [Teman 2]
        T2A["Lihat: kamera"]
        T2B["Kesimpulan: iPhone"]
    end
    
    subgraph Teman3 [Teman 3]
        T3A["Lihat: baterai"]
        T3B["Kesimpulan: Xiaomi"]
    end
    
    subgraph Voting [🗳️ Voting]
        V["Samsung: 2 suara<br/>iPhone: 2 suara<br/>Xiaomi: 1 suara"]
    end
    
    Teman1 --> Voting
    Teman2 --> Voting
    Teman3 --> Voting
```

**Random Forest melakukan hal yang sama!** Setiap pohon (teman) punya sudut pandang berbeda karena:
- Melihat data yang berbeda (seperti pengalaman berbeda)
- Melihat fitur yang berbeda (seperti preferensi berbeda)

### 🌲 Kenapa Namanya "Random" Forest?

```mermaid
graph TB
    subgraph Random1 [Random #1: Data]
        R1["Setiap pohon dilatih dengan<br/>data yang diacak"]
        R2["Seperti: setiap ahli punya<br/>pengalaman berbeda"]
    end
    
    subgraph Random2 [Random #2: Fitur]
        R3["Setiap pohon hanya boleh<br/>melihat sebagian fitur"]
        R4["Seperti: ada ahli kamera,<br/>ada ahli baterai"]
    end
    
    Random1 --> RandomForest["🌲🌲🌲 RANDOM FOREST"]
    Random2 --> RandomForest
```

### 💪 Keunggulan Random Forest

```mermaid
graph LR
    subgraph SatuPohon [Satu Decision Tree]
        SP["🎯 Akurasi: 75%<br/>😰 Suka overfit"]
    end
    
    subgraph BanyakPohon [Random Forest (100 pohon)]
        BP["🎯 Akurasi: 88%<br/>😎 Anti overfit"]
    end
    
    SatuPohon -->|💪 Gabungkan| BanyakPohon
```

### 💬 Kapan Pakai?

> "Pakai Random Forest kalau kamu mau tebakan paling akurat, meskipun agak lambat"

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Akurasi paling TINGGI | Lebih lambat (karena banyak pohon) |
| Tidak mudah overfit | Susah dijelaskan (hitam-hitam) |
| Bisa tangkap pola rumit | Butuh memori lebih besar |
| Paling andal untuk produksi | |

---

## 📊 PERBANDINGAN KETIGANYA (BAHASA SEDERHANA)

```mermaid
graph TB
    subgraph Linear [📏 Linear Regression]
        L1["Cara kerja: Garis lurus"]
        L2["Kecepatan: 🚀 Super cepat"]
        L3["Akurasi: ⭐⭐ (biasa saja)"]
        L4["Mudah dijelaskan? ✅ Sangat"]
    end
    
    subgraph Tree [🌳 Decision Tree]
        T1["Cara kerja: Tebak-tebakan"]
        T2["Kecepatan: ⚡ Cepat"]
        T3["Akurasi: ⭐⭐⭐ (lumayan)"]
        T4["Mudah dijelaskan? ✅ Ya"]
    end
    
    subgraph Forest [🌲🌲🌲 Random Forest]
        F1["Cara kerja: Musyawarah"]
        F2["Kecepatan: 🐢 Lambat"]
        F3["Akurasi: ⭐⭐⭐⭐⭐ (terbaik!)"]
        F4["Mudah dijelaskan? ❌ Sulit"]
    end
```

### 🎯 Tabel Pilih-pilih Model

| Situasi Kamu | Model yang Cocok |
|--------------|------------------|
| Baru belajar, mau lihat gambaran | Linear Regression |
| Data masih kecil (<1000) | Decision Tree |
| Data besar (>5000) | Random Forest |
| Perlu jelaskan ke atasan | Decision Tree |
| Mau akurasi setinggi mungkin | Random Forest |
| Mau cepet dapet hasil | Linear Regression |
| Data berantakan tidak beraturan | Random Forest |

---

## 🧪 CONTOH KASUS NYATA

### Kasus: Memprediksi Harga Minyak

```python
# KODE SEDERHANA (tanpa istilah ribet)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Data: produksi dan harga minyak
produksi = [[100], [150], [200], [250], [300]]  # barrel
harga = [800, 820, 850, 880, 900]  # ribu rupiah

# Coba ketiga model
linear = LinearRegression()
pohon = DecisionTreeRegressor()
hutan = RandomForestRegressor(n_estimators=10)  # 10 pohon

# Latih semua model
linear.fit(produksi, harga)
pohon.fit(produksi, harga)
hutan.fit(produksi, harga)

# Tebak harga jika produksi 180 barrel
tebakan_linear = linear.predict([[180]])
tebakan_pohon = pohon.predict([[180]])
tebakan_hutan = hutan.predict([[180]])

print("Hasil tebakan untuk produksi 180 barrel:")
print(f"Linear Regression (garis lurus): Rp {tebakan_linear[0]:.0f} ribu")
print(f"Decision Tree (tebak-tebakan): Rp {tebakan_pohon[0]:.0f} ribu")
print(f"Random Forest (musyawarah): Rp {tebakan_hutan[0]:.0f} ribu")
```

**Output:**
```
Hasil tebakan untuk produksi 180 barrel:
Linear Regression (garis lurus): Rp 832 ribu
Decision Tree (tebak-tebakan): Rp 820 ribu
Random Forest (musyawarah): Rp 835 ribu
```

---

## 💡 RINGKASAN SINGKAT

```mermaid
mindmap
  root((PILIH MODEL<br/>REGRESI))
    Linear Regression
      Garis lurus
      Cepat tapi biasa aja
      Buat baseline
    Decision Tree
      Tebak-tebakan ya/tidak
      Mudah dijelaskan
      Hati-hati overfit
    Random Forest
      Kumpulan banyak pohon
      Paling akurat
      Rekomendasi final
```

### 🔑 Intinya:

> **Linear Regression** = Tebakan kasar, cepet
> 
> **Decision Tree** = Tebakan pakai pertanyaan, jelas
> 
> **Random Forest** = Tebakan musyawarah, paling jitu

**Mulai dari Linear Regression dulu buat lihat gambaran. Kalau kurang akurat, naik ke Random Forest!**
