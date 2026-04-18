## BAGIAN 2: KLASIFIKASI (PREDIKSI KATEGORI)

---

## 🎯 APA ITU KLASIFIKASI?

**Klasifikasi** adalah cara komputer menebak **kategori atau golongan**.

Bayangkan kamu ingin menebak:
- 🏥 "Apakah pasien ini kena diabetes?" → Ya / Tidak
- 📧 "Apakah email ini spam?" → Spam / Bukan Spam
- 🍎 "Ini buah apa?" → Apel / Jeruk / Pisang

Nah, klasifikasi adalah teknik untuk menjawab pertanyaan **"YA atau TIDAK"** atau **"INI ATAU ITU"**!

```mermaid
graph LR
    subgraph Input [Informasi Pasien]
        A["🩸 Kadar gula: 180"]
        B["⚖️ BMI: 32"]
        C["📅 Usia: 45 tahun"]
        D["🍽️ Riwayat makan: manis"]
    end
    
    subgraph Proses [Komputer Belajar]
        E["🧠 Model Klasifikasi<br/>mencari pola"]
    end
    
    subgraph Output [Hasil Tebakan]
        F["🏥 DIABETES?<br/>⚠️ YA"]
    end
    
    A & B & C & D --> E --> F
```

---

## 📚 PERBEDAAN REGRESI VS KLASIFIKASI

| Aspek | Regresi (sebelumnya) | Klasifikasi (sekarang) |
|-------|---------------------|----------------------|
| **Output** | Angka (75, 200000, 28.5) | Kategori (Ya/Tidak, Merah/Biru) |
| **Contoh** | "Harga minyak besok 850rb" | "Apakah harga naik? Ya" |
| **Pertanyaan** | "Berapa banyak?" | "Apakah ini?" |

```mermaid
graph TD
    subgraph Regresi [📈 REGRESI - Prediksi Angka]
        R1["Input: Produksi 100 barrel"]
        R2["Output: Harga = Rp 850.000"]
    end
    
    subgraph Klasifikasi [🏷️ KLASIFIKASI - Prediksi Kategori]
        K1["Input: Gula darah 180"]
        K2["Output: Diabetes = YA"]
    end
```

---

## 🎲 TIGA JENIS MODEL KLASIFIKASI

Ada 3 model yang akan kita pelajari:

| No | Model | Gambaran Sederhana |
|----|-------|-------------------|
| 1 | **Logistic Regression** | Seperti garis lengkung S yang memisahkan YA dan TIDAK |
| 2 | **Decision Tree Classifier** | Sama seperti sebelumnya, tapi outputnya YA/TIDAK |
| 3 | **Random Forest Classifier** | Musyawarah banyak pohon, hasil voting |

---

## 1️⃣ LOGISTIC REGRESSION - "Garis Lengkung S"

### 📖 Penjelasan Sederhana

Kalau Linear Regression pake garis lurus, **Logistic Regression pake garis berbentuk huruf S**. Kenapa? Karena untuk menjawab YA/TIDAK, kita perlu sesuatu yang menghasilkan antara 0 dan 1.

```mermaid
graph LR
    subgraph GarisLurus [Garis Lurus - Tidak Cocok]
        L1["Bisa menghasilkan -100 atau 200"]
        L2["Tidak masuk akal untuk<br/>'kemungkinan diabetes'"]
    end
    
    subgraph GarisS [Garis S - Cocok]
        S1["Hasil selalu antara 0-1"]
        S2["Bisa diartikan sebagai<br/>'peluang' atau 'persen'"]
    end
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Dokter Mendiagnosis Diabetes**

```mermaid
graph TD
    subgraph Proses [Cara Dokter Menentukan]
        P1["Kadar gula pasien: 150"]
        P2["Dokter pikir-pikir..."]
        P3["'Semakin tinggi gula,<br/>semakin mungkin kena diabetes'"]
        P4["Tapi tidak 100% pasti<br/>(ada faktor lain)"]
    end
    
    subgraph Hasil [Hasil Diagnosa]
        H1["Peluang diabetes: 75%"]
        H2["Kesimpulan:⚠️ YA, diabetes"]
    end
    
    P1 --> P2 --> P3 --> P4 --> H1 --> H2
```

**Logistic Regression melakukan hal yang sama!** Dia menghitung **PELUANG** (antara 0% sampai 100%) lalu memutuskan:

```mermaid
graph LR
    A["Peluang < 50%"] --> B["📌 Prediksi: TIDAK"]
    C["Peluang ≥ 50%"] --> D["📌 Prediksi: YA"]
```

### 📊 Visualisasi Kurva S

```mermaid
graph LR
    subgraph Grafik [Kurva Bentuk S]
        GX["Sumbu X: Kadar Gula Darah"]
        GY["Sumbu Y: Peluang Diabetes"]
        
        P1["Gula 80 → Peluang 5%<br/>→ Tidak diabetes"]
        P2["Gula 140 → Peluang 50%<br/>→ Batas antara"]
        P3["Gula 200 → Peluang 95%<br/>→ Ya diabetes"]
    end
```

### 🎯 Batas Keputusan (Threshold)

```mermaid
graph TD
    subgraph Threshold50 [Batas 50% - Standar]
        T50A["Peluang 49% → TIDAK"]
        T50B["Peluang 51% → YA"]
    end
    
    subgraph Threshold30 [Batas 30% - Lebih Waspada]
        T30A["Peluang 29% → TIDAK"]
        T30B["Peluang 31% → YA"]
        T30C["⚠️ Lebih banyak 'YA'<br/>(lebih aman, tapi banyak false alarm)"]
    end
    
    subgraph Threshold70 [Batas 70% - Lebih Yakin]
        T70A["Peluang 69% → TIDAK"]
        T70B["Peluang 71% → YA"]
        T70C["✅ Lebih yakin kalau 'YA'<br/>❌ Bisa kelewat kasus diabetes"]
    end
```

### ✅ Kelebihan & ❌ Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Sederhana dan cepat | Hanya bisa garis pemisah yang sederhana |
| Memberikan peluang (bukan tebakan kaku) | Kurang akurat untuk pola rumit |
| Mudah diinterpretasi | Sensitif terhadap data aneh (outlier) |
| Cocok sebagai baseline | |

### 💬 Kapan Pakai?

> "Pakai Logistic Regression kalau baru mulai atau butuh tebakan cepat dengan penjelasan peluang"

```mermaid
graph LR
    subgraph Cocok [✅ Cocok untuk]
        C1["Data bisa dipisahkan<br/>dengan garis lengkung"]
        C2["Pengen tahu peluang,<br/>bukan cuma ya/tidak"]
        C3["Butuh kecepatan"]
    end
```

---

## 2️⃣ DECISION TREE CLASSIFIER - "Tebak-tebakan YA/TIDAK"

### 📖 Penjelasan Sederhana

Decision Tree untuk klasifikasi **sama persis** dengan yang sudah kita pelajari di regresi! Bedanya hanya:
- **Regresi**: daunnya berisi ANGKA (rata-rata harga)
- **Klasifikasi**: daunnya berisi KATEGORI (suara terbanyak)

```mermaid
graph TD
    subgraph DecisionTreeRegresi [🌳 Decision Tree untuk Regresi]
        R1["Daun: Harga = Rp 850.000<br/>(rata-rata dari data)"]
    end
    
    subgraph DecisionTreeKlasifikasi [🌳 Decision Tree untuk Klasifikasi]
        K1["Daun: DIABETES = YA<br/>(kebanyakan data ya)"]
    end
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Deteksi Diabetes dengan Pertanyaan**

```mermaid
graph TD
    START["Mulai: Apakah pasien diabetes?"]
    
    START --> Q1["Apakah kadar gula > 140?"]
    
    Q1 -->|Ya| Q2["Apakah BMI > 30?"]
    Q1 -->|Tidak| Q3["Apakah usia > 50?"]
    
    Q2 -->|Ya| A1["🏥 DIABETES: YA<br/>(90% pasien di sini kena)"]
    Q2 -->|Tidak| A2["🏥 DIABETES: MUNGKIN<br/>(50% pasien di sini)"]
    
    Q3 -->|Ya| A3["🏥 DIABETES: MUNGKIN<br/>(40% pasien)"]
    Q3 -->|Tidak| A4["🏥 DIABETES: TIDAK<br/>(10% pasien)"]
```

### 🌳 Bagaimana Pohon Memutuskan?

```mermaid
graph TB
    subgraph Contoh [Data Pasien: Gula=160, BMI=32, Usia=45]
        C1["Langkah 1: Gula > 140? → YA"]
        C2["Langkah 2: BMI > 30? → YA"]
        C3["Sampai di daun: DIABETES = YA"]
    end
    
    subgraph Pohon [Pohon Keputusan]
        P1["Gula > 140?"]
        P1 -->|Ya| P2["BMI > 30?"]
        P2 -->|Ya| P3["✅ YA"]
    end
    
    C1 --> C2 --> C3
```

### ⚠️ Masalah yang Sama: Overfitting

```mermaid
graph LR
    subgraph TerlaluDalam [❌ Pohon Terlalu Dalam]
        TD1["Banyak pertanyaan"]
        TD2["Terlalu detail"]
        TD3["Hasil: Hafal data latih,<br/>jelek di data baru"]
    end
    
    subgraph Pas [✅ Pohon Pas]
        P1["Sedikit pertanyaan"]
        P2("Cukup detail")
        P3["Hasil: Generalisasi bagus"]
    end
```

**Contoh Overfitting:**
```
Pohon yang terlalu detail:
"Apakah gula > 140? → Ya → Apakah BMI > 30? → Ya → 
 Apakah usia > 45? → Ya → Apakah tekanan darah > 130? → Ya → 
 Apakah kolesterol > 200? → Ya → DIABETES"

Terlalu banyak syarat! Kalau ada pasien yang sedikit berbeda, bisa salah tebak.
```

### ✅ Kelebihan & ❌ Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Mudah dijelaskan ke pasien | Suka overfitting (terlalu detail) |
| Bisa lihat alur keputusan | Kurang stabil |
| Tidak perlu persiapan data ribet | Akurasi biasa aja |

### 💬 Kapan Pakai?

> "Pakai Decision Tree kalau kamu mau jelaskan ke dokter atau pasien alasan di balik diagnosis"

---

## 3️⃣ RANDOM FOREST CLASSIFIER - "Musyawarah Banyak Ahli"

### 📖 Penjelasan Sederhana

Sama seperti di regresi, **Random Forest untuk klasifikasi** juga kumpulan banyak pohon yang **voting** untuk menentukan jawaban.

```mermaid
graph TD
    subgraph DewanDokter [👥 Rapat Dewan Dokter]
        A["Dokter 1 (Pohon 1)<br/>Diagnosis: DIABETES"]
        B["Dokter 2 (Pohon 2)<br/>Diagnosis: DIABETES"]
        C["Dokter 3 (Pohon 3)<br/>Diagnosis: TIDAK"]
        D["Dokter 4 (Pohon 4)<br/>Diagnosis: DIABETES"]
        E["Dokter 5 (Pohon 5)<br/>Diagnosis: DIABETES"]
    end
    
    subgraph Voting [🗳️ Voting]
        F["DIABETES: 4 suara<br/>TIDAK: 1 suara"]
    end
    
    subgraph Keputusan [🎯 Hasil Akhir]
        G["🏥 DIABETES: YA"]
    end
    
    A & B & C & D & E --> F --> G
```

### 🎮 Cara Kerja (Cerita Sederhana)

**Cerita: Second Opinion ke Banyak Dokter**

Bayangkan kamu sakit dan minta pendapat ke 5 dokter berbeda:

```mermaid
graph LR
    subgraph Dokter1 [Dokter Spesialis Gula]
        D1A["Lihat: kadar gula"]
        D1B["Kesimpulan: DIABETES"]
    end
    
    subgraph Dokter2 [Dokter Spesialis Berat Badan]
        D2A["Lihat: BMI"]
        D2B["Kesimpulan: DIABETES"]
    end
    
    subgraph Dokter3 [Dokter Spesialis Usia]
        D3A["Lihat: usia"]
        D3B["Kesimpulan: TIDAK"]
    end
    
    subgraph Voting [🗳️ Hasil Voting]
        V["DIABETES: 2<br/>TIDAK: 1"]
    end
    
    Dokter1 --> Voting
    Dokter2 --> Voting
    Dokter3 --> Voting
```

**Random Forest melakukan hal yang sama!** Setiap pohon (dokter) punya spesialisasi berbeda:

```mermaid
graph TB
    subgraph DataSama [Data Pasien yang Sama]
        DS["Gula: 160, BMI: 32, Usia: 45"]
    end
    
    subgraph Pohon1 [Pohon 1 - Fokus Gula]
        P1["Gula tinggi → DIABETES"]
    end
    
    subgraph Pohon2 [Pohon 2 - Fokus BMI]
        P2["BMI tinggi → DIABETES"]
    end
    
    subgraph Pohon3 [Pohon 3 - Fokus Usia]
        P3["Usia 45 (masih muda) → TIDAK"]
    end
    
    subgraph Voting [Voting]
        V["DIABETES: 2 suara<br/>TIDAK: 1 suara<br/>Keputusan: DIABETES"]
    end
    
    DataSama --> Pohon1 & Pohon2 & Pohon3
    Pohon1 & Pohon2 & Pohon3 --> Voting
```

### 🗳️ Cara Voting pada Random Forest

```mermaid
graph LR
    subgraph VotingMurni [Voting Murni]
        VM1["Pohon: YA, YA, TIDAK, YA, YA"]
        VM2["Hasil: YA (4 vs 1)"]
    end
    
    subgraph VotingBobot [Voting dengan Bobot]
        VB1["Pohon yang lebih akurat<br/>suaranya lebih berbobot"]
        VB2["Hasil: lebih akurat"]
    end
```

### 💪 Kenapa Random Forest Paling Akurat?

```mermaid
graph TB
    subgraph SatuDokter [Satu Dokter]
        SD1["Bisa salah diagnosis<br/>karena faktor tertentu"]
        SD2["Akurasi: 75%"]
    end
    
    subgraph BanyakDokter [Banyak Dokter]
        BD1["Kesalahan masing-masing dokter<br/>saling meniadakan"]
        BD2["Akurasi: 88%"]
    end
    
    SatuDokter -->|Voting| BanyakDokter
```

### ✅ Kelebihan & ❌ Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|---------------|
| Akurasi PALING TINGGI | Lambat (banyak pohon) |
| Tidak mudah salah | Susah jelasin ke pasien |
| Bisa tangkap pola rumit | Butuh memori besar |
| Paling andal | |

### 💬 Kapan Pakai?

> "Pakai Random Forest kalau akurasi adalah prioritas #1 kamu"

---

## 📊 PERBANDINGAN KETIGA KLASIFIKASI

```mermaid
graph TB
    subgraph Logistic [📈 Logistic Regression]
        L1["Cara: Garis lengkung S"]
        L2["Kelebihan: Cepat, kasih peluang"]
        L3["Kekurangan: Kurang akurat"]
        L4["🎯 Akurasi: ⭐⭐"]
    end
    
    subgraph Tree [🌳 Decision Tree]
        T1["Cara: Tebak-tebakan"]
        T2["Kelebihan: Mudah dijelaskan"]
        T3["Kekurangan: Suka overfit"]
        T4["🎯 Akurasi: ⭐⭐⭐"]
    end
    
    subgraph Forest [🌲🌲🌲 Random Forest]
        F1["Cara: Voting banyak ahli"]
        F2["Kelebihan: Akurasi tinggi"]
        F3["Kekurangan: Sulit dijelaskan"]
        F4["🎯 Akurasi: ⭐⭐⭐⭐⭐"]
    end
```

### 🎯 Tabel Pilih-pilih Model Klasifikasi

| Situasi Kamu | Model yang Cocok |
|--------------|------------------|
| Baru belajar, mau coba-coba | Logistic Regression |
| Perlu tahu peluang (bukan cuma ya/tidak) | Logistic Regression |
| Perlu jelaskan ke pasien/atasan | Decision Tree |
| Data masih kecil (<1000) | Decision Tree |
| Mau akurasi setinggi mungkin | Random Forest |
| Data besar (>5000) | Random Forest |
| Aplikasi kritis (medis, keamanan) | Random Forest |

---

## 📏 CARA MENGUKUR KEBERHASILAN KLASIFIKASI

### Matriks Kebingungan (Confusion Matrix) - Penjelasan Sederhana

Bayangkan kamu punya 100 pasien, modelmu memprediksi siapa yang kena diabetes:

```mermaid
graph TD
    subgraph Aktual [Kondisi Sebenarnya]
        A1["30 orang BENAR-BENAR DIABETES"]
        A2["70 orang BENAR-BENAR SEHAT"]
    end
    
    subgraph Prediksi [Hasil Prediksi Model]
        P1["Model bilang 25 orang DIABETES"]
        P2["Model bilang 75 orang SEHAT"]
    end
    
    subgraph CocokCek [Dicocokkan]
        C1["✅ 20 orang: benar diabetes (TP)"]
        C2["❌ 5 orang: salah sehat (FN)"]
        C3["✅ 65 orang: benar sehat (TN)"]
        C4["❌ 5 orang: salah diabetes (FP)"]
    end
```

**Empat kemungkinan hasil:**

```mermaid
graph LR
    subgraph TP [✅ True Positive - Tebakan Benar Positif]
        TP1["Model: DIABETES"]
        TP2["Aktual: DIABETES"]
        TP3["Berhasil! Model benar"]
    end
    
    subgraph TN [✅ True Negative - Tebakan Benar Negatif]
        TN1["Model: SEHAT"]
        TN2["Aktual: SEHAT"]
        TN3["Berhasil! Model benar"]
    end
    
    subgraph FP [❌ False Positive - Salah Positif]
        FP1["Model: DIABETES"]
        FP2["Aktual: SEHAT"]
        FP3["False Alarm!<br/>Orang sehat dibilang sakit"]
    end
    
    subgraph FN [❌ False Negative - Salah Negatif]
        FN1["Model: SEHAT"]
        FN2["Aktual: DIABETES"]
        FN3["Berbahaya!<br/>Orang sakit tidak terdeteksi"]
    end
```

### Metrik Evaluasi - Penjelasan Sederhana

```mermaid
graph TD
    subgraph Accuracy [Akurasi - Seberapa Sering Benar?]
        A1["(TP + TN) / Total"]
        A2["(20 + 65) / 100 = 85%"]
        A3["Model benar 85% dari waktu"]
    end
    
    subgraph Precision [Presisi - Kalau Bilang Diabetes, Seberapa Yakin?]
        P1["TP / (TP + FP)"]
        P2["20 / (20 + 5) = 80%"]
        P3["Kalau model bilang diabetes,<br/>80% benerannya diabetes"]
    end
    
    subgraph Recall [Recall - Dari Semua Penderita, Berapa yang Terdeteksi?]
        R1["TP / (TP + FN)"]
        R2["20 / (20 + 5) = 80%"]
        R3["Dari 25 penderita,<br/>model mendeteksi 20 orang (80%)"]
    end
    
    subgraph F1 [F1-Score - Keseimbangan Presisi & Recall]
        F1["Rata-rata khusus dari Presisi & Recall"]
        F2["2 × (P×R)/(P+R)"]
        F3["= 2 × (0.8×0.8)/(0.8+0.8) = 80%"]
    end
```

### 📖 Analogi Sederhana Metrik

Bayangkan kamu jadi **pemeriksa di bandara** yang menangkap penumpang berbahaya:

| Metrik | Analogi di Bandara |
|--------|-------------------|
| **Akurasi** | Dari semua penumpang, berapa persen keputusanmu yang benar? |
| **Presisi** | Dari yang kamu tangkap, berapa persen yang benar-benar berbahaya? |
| **Recall** | Dari semua penumpang berbahaya, berapa persen yang kamu tangkap? |
| **F1** | Keseimbangan antara presisi dan recall |

```mermaid
graph LR
    subgraph Waspada [Kalau Mau Lebih Waspada]
        W1["Prioritaskan RECALL"]
        W2["Tangkap sebanyak mungkin<br/>penumpang berbahaya"]
        W3["Konsekuensi: Banyak false alarm<br/>(orang baik ditangkap)"]
    end
    
    subgraph Yakin [Kalau Mau Pasti]
        Y1["Prioritaskan PRESISI"]
        Y2["Hanya tangkap yang benar-benar<br/>terbukti berbahaya"]
        Y3["Konsekuensi: Banyak yang lolos<br/>(penumpang bahaya tidak tertangkap)"]
    end
    
    subgraph Seimbang [Kalau Mau Seimbang]
        S1["Prioritaskan F1"]
        S2["Keseimbangan antara<br/>waspada dan kepastian"]
    end
```

---

## 🧪 CONTOH KODE SEDERHANA

```python
# KODE KLASIFIKASI SEDERHANA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Data pasien (gula, BMI, usia) dan diagnosis (0=Tidak, 1=Ya)
# Data kecil untuk contoh
gula = [120, 150, 180, 110, 200, 130, 170, 140]  # kadar gula
bmi = [25, 30, 35, 22, 32, 28, 33, 29]            # BMI
usia = [30, 45, 50, 25, 55, 35, 48, 42]           # usia
diabetes = [0, 1, 1, 0, 1, 0, 1, 1]               # 0=Tidak, 1=Ya

# Gabungkan fitur
X = list(zip(gula, bmi, usia))
y = diabetes

# Bagi data: 6 data untuk latih, 2 data untuk uji
X_train = X[:6]  # 6 pasien pertama untuk latih
y_train = y[:6]
X_test = X[6:]   # 2 pasien terakhir untuk uji
y_test = y[6:]

print("Data latih (6 pasien):", X_train)
print("Data uji (2 pasien):", X_test)
print("Jawaban uji sebenarnya:", y_test)
print()

# Model 1: Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
tebakan_logistic = logistic.predict(X_test)
print(f"Logistic Regression tebak: {tebakan_logistic}")

# Model 2: Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tebakan_tree = tree.predict(X_test)
print(f"Decision Tree tebak: {tebakan_tree}")

# Model 3: Random Forest
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X_train, y_train)
tebakan_forest = forest.predict(X_test)
print(f"Random Forest tebak: {tebakan_forest}")

print()
print("Jawaban yang benar:", y_test)
```

**Output yang diharapkan:**
```
Data latih (6 pasien): [(120,25,30), (150,30,45), (180,35,50), (110,22,25), (200,32,55), (130,28,35)]
Data uji (2 pasien): [(170,33,48), (140,29,42)]
Jawaban uji sebenarnya: [1, 1]

Logistic Regression tebak: [1 1]
Decision Tree tebak: [1 1]
Random Forest tebak: [1 1]

Jawaban yang benar: [1, 1]
```

---

## 💡 RINGKASAN SINGKAT KLASIFIKASI

```mermaid
mindmap
  root((PILIH MODEL<br/>KLASIFIKASI))
    Logistic Regression
      Garis lengkung S
      Kasih peluang
      Buat baseline
    Decision Tree
      Tebak-tebakan
      Mudah dijelaskan
      Hati-hati overfit
    Random Forest
      Voting banyak pohon
      Paling akurat
      Rekomendasi final
```

### 🔑 Intinya:

> **Logistic Regression** = Tebakan dengan peluang, cepet
> 
> **Decision Tree** = Tebakan pakai pertanyaan, jelas
> 
> **Random Forest** = Tebakan musyawarah, paling jitu

### 📌 Ingat 4 Metrik Penting:

| Metrik | Pertanyaan yang Dijawab |
|--------|------------------------|
| **Akurasi** | Seberapa sering model benar? |
| **Presisi** | Kalau bilang YA, seberapa yakin? |
| **Recall** | Dari yang sebenarnya YA, berapa yang terdeteksi? |
| **F1** | Keseimbangan presisi & recall |
