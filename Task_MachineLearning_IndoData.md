# 📑 Case Study Assignment: Indonesian Public Data Analytics

## 📝 Deskripsi Tugas
Tugas ini dirancang untuk menguji kemampuan Anda dalam mengolah data publik Indonesia melalui alur kerja *Machine Learning* yang lengkap. Anda akan belajar bagaimana mengubah data mentah menjadi wawasan (*insight*) dan model prediksi yang akurat.

---

## 📂 Alur Kerja (Workflow) & Referensi Kode
Ikuti 7 langkah standar industri ini untuk setiap tugas:

1.  **Data Preparation**: `pd.read_csv()`, `df.isnull().sum()`, `df.dropna()`.
2.  **EDA**: `df.describe()`, `sns.histplot()`, `plt.show()`.
3.  **Feature Engineering**: `df.apply()`, `LabelEncoder()`, `StandardScaler()`.
4.  **Model Training**: `model.fit(X_train, y_train)`.
5.  **Model Validation**: `accuracy_score()`, `mean_squared_error()`.
6.  **Tuning & Finalize**: `GridSearchCV()`, `model.predict()`.
7.  **Referensi**: Dokumentasi resmi Scikit-learn & Pandas.

---

## 🛠️ Tugas 1: Klasifikasi (Classification)
**Dataset**: [indonesia_volcanoes.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv)  
**Tujuan**: Memprediksi tipe/bentuk gunung berapi (`bentuk`) berdasarkan fitur geografis.

### 📋 Panduan & Referensi Kode:
*   **Step 1 (Data Prep)**: Bersihkan `tinggi_meter`.
    ```python
    df['tinggi_meter'] = df['tinggi_meter'].str.extract('(\d+)').astype(float)
    ```
*   **Step 2 (EDA)**: Visualisasi kategori gunung.
    ```python
    import seaborn as sns
    sns.countplot(data=df, x='bentuk')
    ```
*   **Step 3 (Feature Engineering)**: Ekstrak koordinat dengan Regex.
    ```python
    import re
    def extract_lat(text):
        res = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
        return float(res[0]) if res else 0
    df['lat'] = df['geolokasi'].apply(extract_lat)
    ```
*   **Step 4 (Model Training)**: Latih model Random Forest.
    ```python
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```
*   **Step 5 (Model Validation)**: Cek akurasi.
    ```python
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    ```
*   **Step 6 (Tuning)**: Eksperimen parameter.
    ```python
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    ```
*   **Step 7 (Referensi)**: [Regex101](https://regex101.com/) untuk testing pola geolokasi.

---

## 📍 Tugas 2: Klastering (Clustering)
**Dataset**: [indonesia_volcanoes.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_volcanoes.csv)  
**Tujuan**: Mengelompokkan gunung berapi berdasarkan lokasi geografis.

### 📋 Panduan & Referensi Kode:
*   **Step 1 (Data Prep)**: Pastikan koordinat `lat` dan `long` sudah numerik dan tidak ada `NaN`.
    ```python
    df = df.dropna(subset=['lat', 'long'])
    ```
*   **Step 2 (EDA)**: Plot sebaran lokasi.
    ```python
    import matplotlib.pyplot as plt
    plt.scatter(df['long'], df['lat'])
    ```
*   **Step 3 (Feature Engineering)**: Standarisasi data (Penting untuk KMeans!).
    ```python
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(df[['lat', 'long']])
    ```
*   **Step 4 (Model Training)**: Jalankan KMeans.
    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X_scaled)
    ```
*   **Step 5 (Model Validation)**: Elbow Method (Inertia).
    ```python
    wcss = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k).fit(X_scaled)
        wcss.append(km.inertia_)
    ```
*   **Step 6 (Finalize)**: Tambahkan label klaster ke data.
    ```python
    df['cluster'] = kmeans.labels_
    ```
*   **Step 7 (Referensi)**: [Sklearn KMeans Docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

---

## 📈 Tugas 3: Regresi (Regression)
**Dataset**: [indonesia_public_companies.csv](https://raw.githubusercontent.com/yogski/indonesian_public_data/master/csv/indonesia_public_companies.csv)  
**Tujuan**: Memprediksi `jumlah_saham` berdasarkan `listing_age`.

### 📋 Panduan & Referensi Kode:
*   **Step 1 (Data Prep)**: Konversi kolom tanggal.
    ```python
    df['tanggal_listing'] = pd.to_datetime(df['tanggal_listing'])
    ```
*   **Step 2 (EDA)**: Cek korelasi.
    ```python
    print(df.corr())
    ```
*   **Step 3 (Feature Engineering)**: Buat fitur usia listing.
    ```python
    df['listing_age'] = 2024 - df['tanggal_listing'].dt.year
    ```
*   **Step 4 (Model Training)**: Linear Regression.
    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[['listing_age']], df['jumlah_saham'])
    ```
*   **Step 5 (Model Validation)**: Metrik RMSE & R2.
    ```python
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    ```
*   **Step 6 (Tuning)**: Encoding kategori perusahaan.
    ```python
    df_encoded = pd.get_dummies(df, columns=['kategori'])
    ```
*   **Step 7 (Referensi)**: [Pandas Get Dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).

---

## 📤 Kriteria Pengumpulan
- **Format**: File `.ipynb` (Jupyter Notebook).
- **Narasi**: Gunakan Markdown untuk menjelaskan alur kerja Anda.
- **Output**: Grafik harus terlihat jelas.
