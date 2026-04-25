## BAGIAN 3: KLUSTERING (PENGELOMPOKAN DATA)

**Segmentasi Pelanggan & Wilayah**

---

### 📦 Dataset untuk Klustering:

**1. Konteks Lokal (Indonesia):**
- **Konsumsi Pangan Per Provinsi**: [🔗 Download CSV](https://raw.githubusercontent.com/KurniawanDwi/Dataset-Indonesia/master/konsumsi_pangan.csv)
- **Ekspor Impor ID (Historical)**: [🔗 Download CSV](https://raw.githubusercontent.com/KurniawanDwi/Dataset-Indonesia/master/ekspor_impor_indonesia.csv)

**2. Konteks Global:**
- **Mall Customer Segmentation**: [🔗 Download CSV](https://raw.githubusercontent.com/SteffiPauly/Machine-Learning-Datasets/master/Mall_Customers.csv)
- **Wine Chemistry**: [🔗 Download CSV](https://raw.githubusercontent.com/sharmaroshan/Wine-Clustering/master/Wine.csv)

---

### 🎯 Apa itu Klustering?
Klustering adalah teknik **Unsupervised Learning** untuk mencari pola tersembunyi.

### 📋 Tahapan Model:
1. **Analisis Struktur**: Tentukan fitur untuk pengelompokan.
2. **Preprocessing**: **Wajib Scaler** (K-Means sensitif terhadap skala).
3. **Training**: Gunakan **Elbow Method** untuk mencari nilai K.
4. **Interpretasi**: Berikan profil pada tiap kelompok hasil cluster.

---

> **"Klustering menjawab pertanyaan: Apa pola tersembunyi di data ini?"**
