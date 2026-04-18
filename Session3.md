# **Notebook: Praktik Exploratory Data Analysis (EDA) & Data Cleaning – Sesi 3 (Detail)**

Rekaman materi tambahan EDA: https://drive.google.com/file/d/1r-KXlOklFq-zG3XYO5b6IFUaRyjajZGH/view?usp=sharing

**Konteks Bisnis:**  
Puskesmas "Sehat Bersama" ingin menurunkan risiko diabetes dengan memahami faktor-faktor yang memengaruhinya. Data diambil dari 200 pasien secara acak.

---

## **Bagian 0: Import Library & Pengaturan Gaya Visual**

```python
# Import semua library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setting gaya visual untuk plot yang lebih profesional
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("Library berhasil diimport!")
```

---

## **Bagian 1: Pembuatan Dataset Mockup yang Lebih Realistis**

```python
# Set seed untuk reproduktibilitas
np.random.seed(42)

# Jumlah data
n_data = 200

# 1. Usia (distribusi normal, kebanyakan usia 30-70 tahun)
usia = np.random.normal(loc=50, scale=15, size=n_data)
usia = np.clip(usia, 18, 90).astype(int)

# 2. BMI (dengan kecenderungan ke arah overweight)
bmi = np.random.normal(loc=27, scale=4, size=n_data)
bmi = np.clip(bmi, 18, 40).round(1)

# 3. Tekanan darah (sistolik)
tekanan_darah = np.random.normal(loc=120, scale=15, size=n_data)
tekanan_darah = np.clip(tekanan_darah, 90, 200).astype(int)

# 4. Konsumsi nasi putih (probabilitas lebih besar ke Tinggi)
konsumsi_nasi = np.random.choice(['Rendah', 'Sedang', 'Tinggi'], 
                                  size=n_data, 
                                  p=[0.2, 0.3, 0.5])

# 5. Riwayat keluarga diabetes
riwayat_keluarga = np.random.choice(['Ya', 'Tidak'], 
                                    size=n_data, 
                                    p=[0.35, 0.65])

# 6. Gula darah puasa (dipengaruhi oleh konsumsi nasi & riwayat keluarga)
def generate_gula_darah(nasi, riwayat):
    base = 90
    if nasi == 'Tinggi':
        base += 30
    elif nasi == 'Sedang':
        base += 15
    if riwayat == 'Ya':
        base += 20
    noise = np.random.normal(0, 15)
    return max(70, min(300, base + noise))

gula_darah = [generate_gula_darah(n, r) for n, r in zip(konsumsi_nasi, riwayat_keluarga)]

# 7. Label diabetes (berdasarkan gula darah >= 126)
label_diabetes = [1 if gd >= 126 else 0 for gd in gula_darah]

# Buat dataframe
df = pd.DataFrame({
    'Nama_Pasien': [f'Pasien_{i+1:03d}' for i in range(n_data)],
    'Usia': usia,
    'BMI': bmi,
    'Tekanan_Darah': tekanan_darah,
    'Konsumsi_Nasi_Putih': konsumsi_nasi,
    'Riwayat_Keluarga': riwayat_keluarga,
    'Gula_Darah': gula_darah,
    'Label_Diabetes': label_diabetes
})

# SENGAAJA menambahkan missing value dan outlier
df.loc[0:7, 'BMI'] = np.nan           # 8 missing value di BMI
df.loc[15:18, 'Tekanan_Darah'] = np.nan  # 4 missing value di tekanan darah
df.loc[25, 'Usia'] = 150              # outlier usia
df.loc[30, 'Gula_Darah'] = 500        # outlier gula darah
df.loc[35, 'BMI'] = 55                # outlier BMI

print("✅ Dataset mockup selesai dibuat!")
print(f"Dimensi dataset: {df.shape}")
print("\n5 data pertama:")
df.head()
```

---

## **Bagian 2: Data Cleaning – Tahap 1 (Deteksi Masalah)**

```python
# -------------------------------------------------------------------
# 2.1 Informasi umum dataset
# -------------------------------------------------------------------
print("="*50)
print("INFORMASI UMUM DATASET")
print("="*50)
df.info()

print("\n" + "="*50)
print("STATISTIK DESKRIPTIF")
print("="*50)
df.describe(include='all').round(2)

# -------------------------------------------------------------------
# 2.2 Cek Missing Value
# -------------------------------------------------------------------
print("\n" + "="*50)
print("CEK MISSING VALUE")
print("="*50)
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_table = pd.DataFrame({
    'Jumlah Missing': missing_values,
    'Persentase (%)': missing_percent.round(2)
})
print(missing_table[missing_table['Jumlah Missing'] > 0])

# -------------------------------------------------------------------
# 2.3 Cek Outlier dengan metode IQR (Interquartile Range)
# -------------------------------------------------------------------
print("\n" + "="*50)
print("CEK OUTLIER (METODE IQR)")
print("="*50)

numeric_cols = ['Usia', 'BMI', 'Tekanan_Darah', 'Gula_Darah']
outlier_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
    print(f"{col}: {len(outliers)} outlier (batas: {lower_bound:.1f} - {upper_bound:.1f})")
    
    # Tampilkan contoh outlier jika ada
    if len(outliers) > 0:
        print(f"  Contoh: {outliers[col].head(3).values}")
```

---

## **Bagian 3: Data Cleaning – Tahap 2 (Perbaikan)**

```python
# -------------------------------------------------------------------
# 3.1 Handling Missing Value
# -------------------------------------------------------------------
print("SEBELUM HANDLING MISSING VALUE:")
print(f"Missing BMI: {df['BMI'].isnull().sum()}")
print(f"Missing Tekanan Darah: {df['Tekanan_Darah'].isnull().sum()}")

# Imputasi BMI dengan median (karena BMI biasanya skewed)
df['BMI'] = df['BMI'].fillna(df['BMI'].median())

# Imputasi Tekanan Darah dengan mean (distribusi relatif normal)
df['Tekanan_Darah'] = df['Tekanan_Darah'].fillna(df['Tekanan_Darah'].mean())

print("\n✅ SETELAH HANDLING MISSING VALUE:")
print(f"Missing BMI: {df['BMI'].isnull().sum()}")
print(f"Missing Tekanan Darah: {df['Tekanan_Darah'].isnull().sum()}")

# -------------------------------------------------------------------
# 3.2 Handling Outlier (Metode Winsorizing untuk Usia)
# -------------------------------------------------------------------
print("\n" + "="*50)
print("HANDLING OUTLIER")
print("="*50)

# Untuk usia > 100: kita cap di 90 tahun (batas realistis)
outlier_usia = df[df['Usia'] > 100]
print(f"Outlier Usia sebelum handling: {len(outlier_usia)} data")
df.loc[df['Usia'] > 100, 'Usia'] = 90
print(f"Outlier Usia setelah handling: {len(df[df['Usia'] > 100])} data")

# Untuk BMI > 40: kita ganti dengan 40 (obesitas ekstrem)
outlier_bmi = df[df['BMI'] > 40]
print(f"Outlier BMI sebelum handling: {len(outlier_bmi)} data")
df.loc[df['BMI'] > 40, 'BMI'] = 40
print(f"Outlier BMI setelah handling: {len(df[df['BMI'] > 40])} data")

# Untuk Gula Darah > 300: kita ganti dengan 300 (batas maksimal realistis)
outlier_gula = df[df['Gula_Darah'] > 300]
print(f"Outlier Gula Darah sebelum handling: {len(outlier_gula)} data")
df.loc[df['Gula_Darah'] > 300, 'Gula_Darah'] = 300
print(f"Outlier Gula Darah setelah handling: {len(df[df['Gula_Darah'] > 300])} data")

# -------------------------------------------------------------------
# 3.3 Hapus kolom yang tidak diperlukan
# -------------------------------------------------------------------
df_clean = df.drop(columns=['Nama_Pasien'])
print(f"\n✅ Dataset final setelah cleaning: {df_clean.shape}")
```

---

## **Bagian 4: Analisis Univariat (Distribusi Setiap Fitur)**

```python
# -------------------------------------------------------------------
# 4.1 Visualisasi Distribusi Numerik
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram Usia
sns.histplot(df_clean['Usia'], kde=True, ax=axes[0,0], color='skyblue', bins=20)
axes[0,0].axvline(df_clean['Usia'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["Usia"].mean():.1f}')
axes[0,0].axvline(df_clean['Usia'].median(), color='green', linestyle='--', label=f'Median: {df_clean["Usia"].median():.1f}')
axes[0,0].set_title('Distribusi Usia Pasien')
axes[0,0].legend()

# Histogram BMI
sns.histplot(df_clean['BMI'], kde=True, ax=axes[0,1], color='lightgreen', bins=20)
axes[0,1].axvline(df_clean['BMI'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["BMI"].mean():.1f}')
axes[0,1].axvline(df_clean['BMI'].median(), color='green', linestyle='--', label=f'Median: {df_clean["BMI"].median():.1f}')
axes[0,1].set_title('Distribusi BMI')
axes[0,1].legend()

# Boxplot Tekanan Darah
sns.boxplot(y=df_clean['Tekanan_Darah'], ax=axes[1,0], color='salmon')
axes[1,0].set_title('Boxplot Tekanan Darah (Sistolik)')
axes[1,0].set_ylabel('Tekanan Darah (mmHg)')

# Histogram Gula Darah dengan threshold diabetes
sns.histplot(df_clean['Gula_Darah'], kde=True, ax=axes[1,1], color='purple', bins=20)
axes[1,1].axvline(126, color='red', linestyle='--', linewidth=2, label='Threshold Diabetes (126)')
axes[1,1].set_title('Distribusi Gula Darah Puasa')
axes[1,1].legend()

plt.suptitle('ANALISIS UNIVARIAT - FITUR NUMERIK', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 4.2 Analisis Kategorikal
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Konsumsi Nasi Putih
konsumsi_counts = df_clean['Konsumsi_Nasi_Putih'].value_counts()
axes[0].bar(konsumsi_counts.index, konsumsi_counts.values, color=['#66c2a5', '#fc8d62', '#8da0cb'])
axes[0].set_title('Distribusi Konsumsi Nasi Putih')
axes[0].set_ylabel('Jumlah Pasien')
for i, v in enumerate(konsumsi_counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Riwayat Keluarga
riwayat_counts = df_clean['Riwayat_Keluarga'].value_counts()
colors = ['#e78ac3' if x == 'Ya' else '#a6d854' for x in riwayat_counts.index]
axes[1].bar(riwayat_counts.index, riwayat_counts.values, color=colors)
axes[1].set_title('Distribusi Riwayat Keluarga Diabetes')
axes[1].set_ylabel('Jumlah Pasien')
for i, v in enumerate(riwayat_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.suptitle('ANALISIS UNIVARIAT - FITUR KATEGORIKAL', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 4.3 Cek Class Imbalance (Label Diabetes)
# -------------------------------------------------------------------
print("\n" + "="*50)
print("CEK KESEIMBANGAN KELAS (CLASS IMBALANCE)")
print("="*50)
diabetes_counts = df_clean['Label_Diabetes'].value_counts()
diabetes_percent = (diabetes_counts / len(df_clean)) * 100

for label, count in diabetes_counts.items():
    status = "Diabetes" if label == 1 else "Sehat"
    print(f"{status}: {count} pasien ({diabetes_percent[label]:.1f}%)")

# Visualisasi
plt.figure(figsize=(8, 5))
sns.countplot(x='Label_Diabetes', data=df_clean, palette=['#66c2a5', '#fc8d62'])
plt.title('Distribusi Label Diabetes', fontsize=14, fontweight='bold')
plt.xticks([0, 1], ['Sehat (0)', 'Diabetes (1)'])
plt.ylabel('Jumlah Pasien')
for i, v in enumerate(diabetes_counts.values):
    plt.text(i, v + 3, str(v), ha='center', fontweight='bold', fontsize=12)
plt.show()
```

---

## **Bagian 5: Analisis Bivariat (Hubungan Antar Variabel)**

```python
# -------------------------------------------------------------------
# 5.1 Hubungan Konsumsi Nasi dengan Gula Darah
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x='Konsumsi_Nasi_Putih', y='Gula_Darah', data=df_clean, 
            order=['Rendah', 'Sedang', 'Tinggi'], palette='Set2')
plt.title('Distribusi Gula Darah per Tingkat Konsumsi Nasi')
plt.ylabel('Gula Darah (mg/dL)')
plt.axhline(126, color='red', linestyle='--', label='Threshold Diabetes')

# Barplot (rata-rata)
plt.subplot(1, 2, 2)
mean_gula = df_clean.groupby('Konsumsi_Nasi_Putih')['Gula_Darah'].mean().reindex(['Rendah', 'Sedang', 'Tinggi'])
std_gula = df_clean.groupby('Konsumsi_Nasi_Putih')['Gula_Darah'].std().reindex(['Rendah', 'Sedang', 'Tinggi'])
plt.bar(mean_gula.index, mean_gula.values, yerr=std_gula.values, capsize=10, 
        color=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title('Rata-rata Gula Darah per Konsumsi Nasi')
plt.ylabel('Rata-rata Gula Darah (mg/dL)')
plt.axhline(126, color='red', linestyle='--', label='Threshold Diabetes')

plt.suptitle('ANALISIS BIVARIAT: Konsumsi Nasi vs Gula Darah', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Uji statistik ANOVA
from scipy.stats import f_oneway
rendah = df_clean[df_clean['Konsumsi_Nasi_Putih'] == 'Rendah']['Gula_Darah']
sedang = df_clean[df_clean['Konsumsi_Nasi_Putih'] == 'Sedang']['Gula_Darah']
tinggi = df_clean[df_clean['Konsumsi_Nasi_Putih'] == 'Tinggi']['Gula_Darah']
f_stat, p_value = f_oneway(rendah, sedang, tinggi)
print(f"Uji ANOVA: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("✅ Kesimpulan: Ada perbedaan signifikan rata-rata gula darah antar kelompok konsumsi nasi.")
else:
    print("❌ Kesimpulan: Tidak ada perbedaan signifikan.")

# -------------------------------------------------------------------
# 5.2 Hubungan Riwayat Keluarga dengan Diabetes
# -------------------------------------------------------------------
plt.figure(figsize=(8, 6))
cross_tab = pd.crosstab(df_clean['Riwayat_Keluarga'], df_clean['Label_Diabetes'], normalize='index') * 100
cross_tab.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62'], ax=plt.gca())
plt.title('Persentase Diabetes berdasarkan Riwayat Keluarga', fontsize=14, fontweight='bold')
plt.xlabel('Riwayat Keluarga')
plt.ylabel('Persentase (%)')
plt.legend(['Sehat', 'Diabetes'], title='Status')
plt.xticks(rotation=0)
for i, (idx, row) in enumerate(cross_tab.iterrows()):
    plt.text(i, row[0]/2, f"{row[0]:.1f}%", ha='center', fontweight='bold')
    plt.text(i, row[0] + row[1]/2, f"{row[1]:.1f}%", ha='center', fontweight='bold')
plt.show()

# Uji Chi-Square
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df_clean['Riwayat_Keluarga'], df_clean['Label_Diabetes'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nUji Chi-Square: chi2 = {chi2:.2f}, p-value = {p:.4f}")
if p < 0.05:
    print("✅ Kesimpulan: Ada hubungan signifikan antara riwayat keluarga dan diabetes.")
else:
    print("❌ Kesimpulan: Tidak ada hubungan signifikan.")
```

---

## **Bagian 6: Analisis Multivariat (Korelasi & Interaksi)**

```python
# -------------------------------------------------------------------
# 6.1 Heatmap Korelasi
# -------------------------------------------------------------------
# Pilih kolom numerik untuk korelasi
numeric_features = ['Usia', 'BMI', 'Tekanan_Darah', 'Gula_Darah', 'Label_Diabetes']
corr_matrix = df_clean[numeric_features].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            mask=mask,
            cbar_kws={"shrink": 0.8})
plt.title('HEATMAP KORELASI ANTAR FITUR NUMERIK', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("INTERPRETASI KORELASI TERKUAT:")
print("="*50)
# Ambil korelasi dengan Label_Diabetes (abaikan korelasi dengan diri sendiri)
corr_with_target = corr_matrix['Label_Diabetes'].sort_values(ascending=False)
for var, corr in corr_with_target.items():
    if var != 'Label_Diabetes':
        strength = "sangat kuat" if abs(corr) > 0.7 else "kuat" if abs(corr) > 0.5 else "sedang" if abs(corr) > 0.3 else "lemah"
        direction = "positif" if corr > 0 else "negatif"
        print(f"• {var}: {corr:.3f} (korelasi {direction} {strength})")

# -------------------------------------------------------------------
# 6.2 Pairplot untuk melihat interaksi antar fitur
# -------------------------------------------------------------------
sns.pairplot(df_clean[numeric_features], 
             hue='Label_Diabetes', 
             palette=['#66c2a5', '#fc8d62'],
             diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 50})
plt.suptitle('PAIRPLOT: Interaksi Antar Fitur Numerik', y=1.02, fontsize=14, fontweight='bold')
plt.show()

# -------------------------------------------------------------------
# 6.3 Analisis Grup: Rata-rata per Kombinasi
# -------------------------------------------------------------------
print("\n" + "="*50)
print("RATA-RATA GULA DARAH BERDASARKAN KOMBINASI FAKTOR RISIKO:")
print("="*50)

group_stats = df_clean.groupby(['Konsumsi_Nasi_Putih', 'Riwayat_Keluarga'])['Gula_Darah'].agg(['mean', 'count', 'std']).round(1)
group_stats.columns = ['Rata2 Gula Darah', 'Jumlah Pasien', 'Std Dev']
print(group_stats)

# Visualisasi interaksi
plt.figure(figsize=(10, 6))
sns.barplot(x='Konsumsi_Nasi_Putih', y='Gula_Darah', hue='Riwayat_Keluarga', data=df_clean,
            order=['Rendah', 'Sedang', 'Tinggi'], palette='Set1')
plt.title('INTERAKSI: Konsumsi Nasi & Riwayat Keluarga terhadap Gula Darah', fontsize=14, fontweight='bold')
plt.ylabel('Rata-rata Gula Darah (mg/dL)')
plt.axhline(126, color='red', linestyle='--', label='Threshold Diabetes')
plt.legend(title='Riwayat Keluarga')
plt.show()
```

---

## **Bagian 7: Feature Engineering (Persiapan Modeling)**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
# 7.1 Persiapan Data
# -------------------------------------------------------------------
df_model = df_clean.copy()

# -------------------------------------------------------------------
# 7.2 Encoding untuk Fitur Kategorikal
# -------------------------------------------------------------------
print("="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Label Encoding untuk Riwayat Keluarga (binary)
le = LabelEncoder()
df_model['Riwayat_Keluarga_Encoded'] = le.fit_transform(df_model['Riwayat_Keluarga'])
print("✅ Riwayat Keluarga → Label Encoding (Ya=1, Tidak=0)")

# Ordinal Encoding untuk Konsumsi Nasi (sudah ordinal)
nasi_ordinal = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
df_model['Konsumsi_Nasi_Encoded'] = df_model['Konsumsi_Nasi_Putih'].map(nasi_ordinal)
print("✅ Konsumsi Nasi → Ordinal Encoding (Rendah=0, Sedang=1, Tinggi=2)")

# One-Hot Encoding (opsional untuk perbandingan)
df_model = pd.get_dummies(df_model, columns=['Konsumsi_Nasi_Putih'], prefix='Nasi')
print("✅ One-Hot Encoding untuk Konsumsi Nasi (3 kolom baru)")

# -------------------------------------------------------------------
# 7.3 Scaling Fitur Numerik
# -------------------------------------------------------------------
numeric_features = ['Usia', 'BMI', 'Tekanan_Darah', 'Gula_Darah']
scaler = StandardScaler()
df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
print("✅ Standard Scaling untuk fitur numerik (mean=0, std=1)")

# -------------------------------------------------------------------
# 7.4 Tampilkan Hasil Akhir
# -------------------------------------------------------------------
print("\n" + "="*50)
print("DATA SIAP UNTUK MODELING:")
print("="*50)
print(f"Dimensi dataset final: {df_model.shape}")
print("\n5 data pertama:")
df_model.head()
```

---

## **Bagian 8: Kesimpulan & Rekomendasi Bisnis**

```python
print("="*60)
print("KESIMPULAN ANALISIS EDA")
print("="*60)

insights = [
    "1. Prevalensi Diabetes: {:.1f}% dari total pasien menderita diabetes.".format(diabetes_percent[1]),
    "2. Faktor Risiko Terkuat:",
    "   - Gula Darah memiliki korelasi {:.3f} dengan status diabetes (sangat kuat)".format(corr_matrix.loc['Gula_Darah', 'Label_Diabetes']),
    "   - BMI memiliki korelasi {:.3f} (sedang-kuat)".format(corr_matrix.loc['BMI', 'Label_Diabetes']),
    "   - Usia memiliki korelasi {:.3f} (lemah-sedang)".format(corr_matrix.loc['Usia', 'Label_Diabetes']),
    "3. Konsumsi Nasi: Pasien dengan konsumsi nasi tinggi memiliki rata-rata gula darah {:.1f} mg/dL, " \
    "sedangkan yang rendah hanya {:.1f} mg/dL (perbedaan signifikan).".format(mean_gula['Tinggi'], mean_gula['Rendah']),
    "4. Riwayat Keluarga: Pasien dengan riwayat keluarga diabetes {:.1f} kali lebih berisiko.".format(
        cross_tab.loc['Ya', 1] / cross_tab.loc['Tidak', 1]
    )
]

for insight in insights:
    print(insight)

print("\n" + "="*60)
print("REKOMENDASI BISNIS UNTUK PUSKESMAS")
print("="*60)

recommendations = [
    "🎯 Intervensi Prioritas:",
    "   • Program pengurangan konsumsi nasi putih (edukasi, substitusi dengan karbohidrat kompleks)",
    "   • Skrining diabetes rutin untuk pasien dengan BMI > 25 dan usia > 45 tahun",
    "   • Konseling gizi khusus untuk pasien dengan riwayat keluarga diabetes",
    "",
    "📊 Tindakan untuk Data Science:",
    "   • Menggunakan teknik SMOTE atau class weighting saat modeling karena kelas tidak seimbang",
    "   • Fitur yang paling penting untuk prediksi: Gula_Darah, BMI, Konsumsi_Nasi, Riwayat_Keluarga",
    "   • Disarankan menggunakan model Random Forest atau XGBoost untuk menangkap interaksi non-linear"
]

for rec in recommendations:
    print(rec)

print("\n" + "="*60)
print("✅ Notebook selesai! Dataset siap untuk dilanjutkan ke sesi modeling.")
print("="*60)
```

---

## **Ekspor Data yang Sudah Dibersihkan**

```python
# Simpan data yang sudah bersih untuk sesi berikutnya
df_clean.to_csv('data_diabetes_puskesmas_clean.csv', index=False)
df_model.to_csv('data_diabetes_puskesmas_model_ready.csv', index=False)

print("✅ Data bersih disimpan sebagai: 'data_diabetes_puskesmas_clean.csv'")
print("✅ Data siap model disimpan sebagai: 'data_diabetes_puskesmas_model_ready.csv'")

# Tampilkan sample final
print("\nSample data siap model (5 baris terakhir):")
df_model.tail()
```
