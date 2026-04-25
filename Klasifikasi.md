## BAGIAN 2: KLASIFIKASI (PREDIKSI KATEGORI)

**Diagnosa Medis & Kualitas Lingkungan**

---

### 📦 Dataset untuk Klasifikasi:

**1. Konteks Lokal (Indonesia):**
- **Kualitas Udara Jakarta (ISPU)**: [🔗 Download CSV](https://raw.githubusercontent.com/yofisunarta/ispu-jakarta/master/data/ispu_jakarta_2021.csv)
- **Status Gizi Balita (Simulasi)**: [🔗 Link Referensi](https://raw.githubusercontent.com/arifbe/sembako-scraper/master/data/sembako.csv) (Mirror)

**2. Konteks Global:**
- **Diabetes Pima Indians**: [🔗 Download CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Titanic Survival**: [🔗 Download CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

### 🎯 Apa itu Klasifikasi?
Klasifikasi adalah teknik Machine Learning untuk memprediksi **kategori atau label**.

### 📋 Tahapan Model:
1. **Analisis Distribusi**: Cek keseimbangan kelas.
2. **Preprocessing**: Lakukan **Standard Scaling**.
3. **Training**: Gunakan `XGBClassifier` atau `RandomForestClassifier`.
4. **Evaluasi**: Cek **Confusion Matrix** dan **Recall**.

---

> **"Klasifikasi menjawab pertanyaan: Termasuk kelompok mana ini?"**
