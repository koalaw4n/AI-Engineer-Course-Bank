# **Sesi 2: Python untuk Kebutuhan AI**
## Fondasi Data Science - Pandas, NumPy, & Visualisasi untuk AI

---

## 🎯 **Tujuan Pembelajaran**
Setelah sesi ini, Anda akan mampu:
- Mengolah dan membersihkan data dengan Pandas
- Melakukan operasi numerik efisien dengan NumPy
- Membuat visualisasi data yang informatif
- Mempersiapkan data untuk model AI

---

## 📦 **Installasi Library yang Diperlukan**

```bash
# Jalankan di terminal/command prompt
pip install pandas numpy matplotlib seaborn openpyxl requests beautifulsoup4
```

```python
# Import library yang akan digunakan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Setting untuk tampilan yang lebih baik
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0-8-darkgrid')
sns.set_palette("husl")

print("✅ Semua library berhasil diimport!")
```

---

## 📚 **2.1 Review Python Cepat untuk Data**

### **2.1.1 List Comprehension**

List comprehension adalah cara membuat list baru dengan lebih cepat dan efisien.

```python
print("="*60)
print("LIST COMPREHENSION - Mempercepat Kode Python")
print("="*60)

# CONTOH 1: Tanpa List Comprehension (Cara Lama)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_numbers = []
for num in numbers:
    if num % 2 == 0:
        squared_numbers.append(num ** 2)
print("Cara lama (tanpa list comprehension):", squared_numbers)

# CONTOH 2: Dengan List Comprehension (Cara Pythonic)
squared_numbers = [num ** 2 for num in numbers if num % 2 == 0]
print("Cara Pythonic (dengan list comprehension):", squared_numbers)

# CONTOH 3: List Comprehension Bersarang
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print("Flatten matrix (3x3 → 1x9):", flattened)

# CONTOH 4: Dictionary Comprehension
names = ['Andi', 'Budi', 'Citra', 'Dewi']
scores = [85, 92, 78, 90]
student_scores = {name: score for name, score in zip(names, scores)}
print("Dictionary dari 2 list:", student_scores)

# CONTOH 5: Set Comprehension
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_squares = {x**2 for x in numbers}
print("Set comprehension (unique values):", unique_squares)
```

### **2.1.2 Lambda Function**

Lambda adalah fungsi kecil satu baris yang sangat berguna untuk operasi sederhana.

```python
print("\n" + "="*60)
print("LAMBDA FUNCTION - Fungsi Cepat Satu Baris")
print("="*60)

# CONTOH 1: Lambda Dasar
kuadrat = lambda x: x ** 2
print(f"Lambda kuadrat(5): {kuadrat(5)}")

# CONTOH 2: Lambda dengan Multiple Parameter
tambah = lambda a, b, c: a + b + c
print(f"Lambda tambah(1,2,3): {tambah(1, 2, 3)}")

# CONTOH 3: Lambda dengan map() - Transformasi Data
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(f"Map (double): {doubled}")

# CONTOH 4: Lambda dengan filter() - Seleksi Data
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Filter (even numbers): {even_numbers}")

# CONTOH 5: Lambda dengan sorted() - Custom Sorting
products = [
    {'name': 'Laptop', 'price': 15000000},
    {'name': 'Mouse', 'price': 150000},
    {'name': 'Keyboard', 'price': 500000},
    {'name': 'Monitor', 'price': 2500000}
]
sorted_by_price = sorted(products, key=lambda x: x['price'])
print("Produk termurah ke termahal:")
for product in sorted_by_price:
    print(f"  - {product['name']}: Rp{product['price']:,}")
```

### **2.1.3 Error Handling**

Error handling membuat program tetap berjalan meskipun ada kesalahan.

```python
print("\n" + "="*60)
print("ERROR HANDLING - Membuat Program Lebih Robust")
print("="*60)

def safe_calculate():
    """Fungsi dengan error handling yang baik"""
    try:
        # Mencoba melakukan operasi yang mungkin error
        angka1 = float(input("Masukkan angka pertama: "))
        angka2 = float(input("Masukkan angka kedua: "))
        hasil = angka1 / angka2
        
    except ValueError:
        print("❌ ERROR: Input harus berupa angka!")
        return None
        
    except ZeroDivisionError:
        print("❌ ERROR: Tidak bisa membagi dengan nol!")
        return None
        
    except Exception as e:
        print(f"❌ ERROR tidak terduga: {e}")
        return None
        
    else:
        print(f"✅ Hasil: {angka1} / {angka2} = {hasil}")
        return hasil
        
    finally:
        print("🔚 Eksekusi fungsi selesai.\n")

# Test fungsi
# safe_calculate()  # Uncomment untuk test interaktif

# Contoh error handling untuk file
def safe_read_file(filename):
    """Membaca file dengan aman"""
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"❌ File '{filename}' tidak ditemukan!")
        return None
    except PermissionError:
        print(f"❌ Tidak memiliki izin untuk membaca '{filename}'!")
        return None

# Test
safe_read_file("file_yang_tidak_ada.txt")
```

### **2.1.4 Baca File dengan Pandas (CSV, Excel, HTML)**

Pandas dapat membaca berbagai format file dengan mudah.

```python
print("\n" + "="*60)
print("MEMBACA BERBAGAI FORMAT FILE DENGAN PANDAS")
print("="*60)

# CONTOH 1: Membaca CSV (Comma Separated Values)
print("\n1. MEMBACA FILE CSV:")
csv_url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
try:
    df_csv = pd.read_csv(csv_url)
    print(f"CSV berhasil dibaca! Shape: {df_csv.shape}")
    print(df_csv.head())
except:
    print("Membuat contoh CSV dari dictionary:")
    df_csv = pd.DataFrame({
        'Name': ['Andi', 'Budi', 'Citra'],
        'Age': [25, 30, 28],
        'City': ['Jakarta', 'Bandung', 'Surabaya']
    })
    df_csv.to_csv('sample.csv', index=False)
    df_csv = pd.read_csv('sample.csv')
    print(df_csv)

# CONTOH 2: Membaca Excel
print("\n2. MEMBACA FILE EXCEL:")
from openpyxl import Workbook
wb = Workbook()
ws = wb.active
ws.append(['Product', 'Price', 'Stock'])
ws.append(['Laptop', 15000000, 10])
ws.append(['Mouse', 150000, 50])
ws.append(['Keyboard', 500000, 25])
wb.save('sample.xlsx')
df_excel = pd.read_excel('sample.xlsx', engine='openpyxl')
print(df_excel)

# CONTOH 3: Membaca HTML (Tabel dari Website)
print("\n3. MEMBACA TABEL HTML DARI WEBSITE:")
# Contoh Wikipedia - tabel populasi
url = "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"
try:
    tables = pd.read_html(url)
    print(f"Ditemukan {len(tables)} tabel di halaman tersebut")
    if len(tables) > 0:
        df_html = tables[0]  # Ambil tabel pertama
        print(f"Shape tabel pertama: {df_html.shape}")
        print(df_html.head())
except Exception as e:
    print(f"Error membaca HTML: {e}")
    print("Membuat contoh tabel HTML sederhana:")
    html_string = """
    <table>
        <tr><th>Nama</th><th>Nilai</th></tr>
        <tr><td>Andi</td><td>85</td></tr>
        <tr><td>Budi</td><td>92</td></tr>
    </table>
    """
    df_html = pd.read_html(StringIO(html_string))[0]
    print(df_html)
```

### **2.1.5 PRAKTIK: Baca 3 Dataset dari Repository dengan pd.read_html()**

**Dataset yang akan digunakan:**
1. **Tabel Populasi Negara** - Untuk belajar data statistik
2. **Tabel FIFA World Rankings** - Data peringkat sepakbola
3. **Tabel Harga Saham** - Data keuangan real-time

```python
print("\n" + "="*60)
print("PRAKTIK: MEMBACA 3 DATASET DENGAN pd.read_html()")
print("="*60)

# ============================================
# DATASET 1: Populasi Negara (Wikipedia)
# ============================================
print("\n📊 DATASET 1: World Population by Country")
print("-" * 40)

url_populasi = "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)"

try:
    # Baca semua tabel dari halaman
    tables_populasi = pd.read_html(url_populasi)
    
    # Tabel populasi biasanya yang pertama atau kedua
    df_populasi = tables_populasi[0]
    
    print(f"✅ Berhasil membaca dataset populasi!")
    print(f"Shape: {df_populasi.shape} (baris, kolom)")
    print(f"\n5 Negara dengan populasi terbesar:")
    print(df_populasi.head())
    
    # Bersihkan nama kolom
    df_populasi.columns = df_populasi.columns.droplevel(0) if isinstance(df_populasi.columns, pd.MultiIndex) else df_populasi.columns
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback dataset
    df_populasi = pd.DataFrame({
        'Country': ['China', 'India', 'USA', 'Indonesia', 'Pakistan'],
        'Population': [1444216107, 1393409038, 332915073, 276361783, 225199937],
        'Year': [2023, 2023, 2023, 2023, 2023]
    })
    print("Menggunakan dataset fallback:")
    print(df_populasi)

# ============================================
# DATASET 2: FIFA World Rankings
# ============================================
print("\n\n📊 DATASET 2: FIFA World Rankings")
print("-" * 40)

url_fifa = "https://en.wikipedia.org/wiki/FIFA_World_Rankings"

try:
    tables_fifa = pd.read_html(url_fifa)
    
    # Cari tabel ranking
    df_fifa = None
    for i, table in enumerate(tables_fifa):
        if 'Rank' in str(table.columns) or 'Team' in str(table.columns):
            df_fifa = table
            print(f"✅ Menemukan tabel ranking di index {i}")
            break
    
    if df_fifa is None:
        df_fifa = tables_fifa[0]  # Ambil tabel pertama
    
    print(f"Shape: {df_fifa.shape}")
    print(f"\nTop 10 tim terbaik FIFA:")
    print(df_fifa.head(10))
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback dataset
    df_fifa = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Team': ['Argentina', 'France', 'Brazil', 'England', 'Belgium'],
        'Points': [1867.25, 1853.11, 1848.45, 1832.15, 1818.92]
    })
    print("Menggunakan dataset fallback:")
    print(df_fifa)

# ============================================
# DATASET 3: Tabel Keuangan/Stock Market
# ============================================
print("\n\n📊 DATASET 3: Stock Market Data (Example)")
print("-" * 40)

# Menggunakan URL dengan tabel keuangan
url_stock = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

try:
    tables_stock = pd.read_html(url_stock)
    
    # Tabel S&P 500 biasanya yang pertama
    df_stock = tables_stock[0]
    
    print(f"✅ Berhasil membaca dataset S&P 500!")
    print(f"Shape: {df_stock.shape}")
    print(f"\n5 perusahaan pertama:")
    print(df_stock[['Symbol', 'Security', 'GICS Sector']].head())
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback dataset
    df_stock = pd.DataFrame({
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'Company': ['Apple Inc.', 'Google', 'Microsoft', 'Amazon', 'Tesla'],
        'Price': [175.50, 142.30, 380.75, 145.20, 240.50],
        'Volume': [50000000, 25000000, 35000000, 40000000, 80000000]
    })
    print("Menggunakan dataset fallback:")
    print(df_stock)

# Ringkasan semua dataset
print("\n" + "="*60)
print("RINGKASAN 3 DATASET YANG TELAH DIBACA")
print("="*60)

datasets = {
    'Population Data': df_populasi,
    'FIFA Rankings': df_fifa,
    'Stock Market': df_stock
}

for name, df in datasets.items():
    print(f"\n📁 {name}:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)[:5]}...")  # 5 kolom pertama
    print(f"   - Missing values: {df.isnull().sum().sum()}")
```

**📚 Link Dataset untuk Praktik:**
- Populasi Negara: https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)
- FIFA Rankings: https://en.wikipedia.org/wiki/FIFA_World_Rankings
- S&P 500 Companies: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

**💡 Tips:**
- Gunakan `pd.read_html()` untuk mengekstrak tabel dari website
- Website dengan banyak tabel perlu iterasi untuk menemukan tabel yang tepat
- Handle error dengan try-except untuk mengantisipasi perubahan struktur website

---

## 🔢 **2.2 NumPy untuk Operasi Numerik**

### **2.2.1 Array vs List Python (Performa & Kemampuan)**

```python
print("\n" + "="*60)
print("NUMPY ARRAY vs PYTHON LIST - Perbandingan Performa")
print("="*60)

import time

# Membuat data besar
size = 10_000_000  # 10 juta data
print(f"Memproses {size:,} data...")

# PYTHON LIST
python_list = list(range(size))
start = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start
print(f"\n🐍 Python List: {python_time:.4f} detik")

# NUMPY ARRAY
numpy_array = np.arange(size)
start = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start
print(f"🚀 NumPy Array: {numpy_time:.4f} detik")
print(f"⚡ Percepatan: {python_time/numpy_time:.1f}x lebih cepat dengan NumPy!")

# Perbedaan kemampuan
print("\n📊 PERBEDAAN KEMAMPUAN:")
print("-" * 40)

# List bisa menyimpan berbagai tipe data
mixed_list = [1, "text", 3.14, [1, 2], {"key": "value"}]
print(f"List bisa campur tipe data: {mixed_list}")

# Array hanya bisa satu tipe data
mixed_array = np.array([1, 2, 3.14])
print(f"Array semua jadi float: {mixed_array}")

# Array memiliki atribut tambahan
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nArray shape: {arr.shape}")
print(f"Array dimension: {arr.ndim}D")
print(f"Array size (total elemen): {arr.size}")
print(f"Array data type: {arr.dtype}")
```

### **2.2.2 Broadcasting, Indexing, Slicing**

```python
print("\n" + "="*60)
print("BROADCASTING - Operasi Array dengan Shape Berbeda")
print("="*60)

# BROADCASTING: Scalar ke Array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array asli: {arr}")
print(f"Array + 10: {arr + 10}")  # 10 di-broadcast ke semua elemen
print(f"Array * 3: {arr * 3}")
print(f"Array ** 2: {arr ** 2}")

# BROADCASTING: 2D dengan 1D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vector = np.array([10, 20, 30])
col_vector = np.array([[10], [20], [30]])

print(f"\nMatrix 3x3:\n{matrix}")
print(f"Row vector: {row_vector}")
print(f"Matrix + Row vector (broadcast ke baris):\n{matrix + row_vector}")
print(f"Matrix + Col vector (broadcast ke kolom):\n{matrix + col_vector}")

print("\n" + "="*60)
print("INDEXING & SLICING - Mengakses Data")
print("="*60)

# 1D Indexing & Slicing
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"Array 1D: {arr_1d}")
print(f"arr_1d[0] (elemen pertama): {arr_1d[0]}")
print(f"arr_1d[-1] (elemen terakhir): {arr_1d[-1]}")
print(f"arr_1d[2:5] (index 2-4): {arr_1d[2:5]}")
print(f"arr_1d[::2] (lompat 2): {arr_1d[::2]}")
print(f"arr_1d[::-1] (reverse): {arr_1d[::-1]}")

# 2D Indexing & Slicing
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print(f"\nArray 2D (3x4):\n{arr_2d}")
print(f"arr_2d[0, 1] (baris 0, kolom 1): {arr_2d[0, 1]}")
print(f"arr_2d[1, :] (baris 1 semua kolom): {arr_2d[1, :]}")
print(f"arr_2d[:, 2] (kolom 2 semua baris): {arr_2d[:, 2]}")
print(f"arr_2d[0:2, 1:3] (submatrix 2x2):\n{arr_2d[0:2, 1:3]}")

# Boolean Indexing (Filtering)
data = np.random.randn(10)  # 10 random numbers
print(f"\nRandom data: {data}")
print(f"Data > 0: {data[data > 0]}")
print(f"Jumlah data positif: {(data > 0).sum()}")
print(f"Rata-rata data positif: {data[data > 0].mean() if any(data > 0) else 0}")

# Fancy Indexing (Index dengan array)
indices = np.array([0, 2, 4, 6, 8])
print(f"Data di index {indices}: {data[indices]}")
```

### **2.2.3 Operasi Matematika Vektor (Dot Product, Matriks)**

```python
print("\n" + "="*60)
print("OPERASI VEKTOR & MATRIKS - Fondasi AI")
print("="*60)

# DOT PRODUCT - Dasar Neural Network
print("1. DOT PRODUCT (Perkalian Titik)")
print("-" * 40)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
dot_product_alt = v1 @ v2  # Cara alternatif (Python 3.5+)

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"Dot product (v1 · v2) = {dot_product}")
print(f"Dot product alternatif = {dot_product_alt}")
print("Kegunaan: Menghitung similarity, weighted sum di neural network")

# PERKALIAN MATRIKS - Forward Propagation Neural Network
print("\n2. PERKALIAN MATRIKS (Matrix Multiplication)")
print("-" * 40)

# Neural network layer: 3 input → 4 hidden → 2 output
input_layer = np.array([[1, 2, 3],     # Batch 2 samples
                        [4, 5, 6]])    # 3 features each

weights_1 = np.array([[0.1, 0.2, 0.3, 0.4],  # 3 inputs → 4 neurons
                      [0.5, 0.6, 0.7, 0.8],
                      [0.9, 1.0, 1.1, 1.2]])

bias_1 = np.array([0.1, 0.2, 0.3, 0.4])

# Hidden layer calculation
hidden = np.dot(input_layer, weights_1) + bias_1
print(f"Input shape: {input_layer.shape}")
print(f"Weights shape: {weights_1.shape}")
print(f"Hidden layer output shape: {hidden.shape}")
print(f"Hidden layer values:\n{hidden}")

# OPERASI MATRIKS LAINNYA
print("\n3. OPERASI MATRIKS LAINNYA")
print("-" * 40)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

print(f"\nTranspose A^T:\n{A.T}")
print(f"Matrix Identity 2x2:\n{np.eye(2)}")
print(f"Inverse A^-1:\n{np.linalg.inv(A)}")
print(f"Determinant of A: {np.linalg.det(A):.2f}")

# EIGENVALUES & EIGENVECTORS (PCA, Dimensionality Reduction)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

---

## 🐼 **2.3 Pandas untuk Manipulasi Data**

### **2.3.1 Series & DataFrame**

```python
print("\n" + "="*60)
print("PANDAS SERIES & DATAFRAME - Struktur Data Core")
print("="*60)

# SERIES = 1D labeled array (seperti kolom di Excel)
print("1. SERIES (1 Dimensi dengan Label)")
print("-" * 40)

scores = pd.Series([85, 92, 78, 90, 88],
                   index=['Andi', 'Budi', 'Citra', 'Dewi', 'Eka'],
                   name='Ujian')
print(scores)
print(f"\nAkses dengan label: scores['Budi'] = {scores['Budi']}")
print(f"Akses dengan posisi: scores.iloc[2] = {scores.iloc[2]}")
print(f"Statistik: mean={scores.mean():.1f}, max={scores.max()}, min={scores.min()}")

# DATAFRAME = 2D labeled structure (seperti spreadsheet)
print("\n2. DATAFRAME (2 Dimensi dengan Baris & Kolom)")
print("-" * 40)

df_students = pd.DataFrame({
    'Nama': ['Andi', 'Budi', 'Citra', 'Dewi', 'Eka'],
    'Usia': [25, 30, 28, 35, 26],
    'Kota': ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung'],
    'Skor': [85, 92, 78, 90, 88],
    'Status': ['Aktif', 'Aktif', 'Lulus', 'Aktif', 'Lulus']
})

print(df_students)
print(f"\nShape: {df_students.shape}")
print(f"Index: {df_students.index.tolist()}")
print(f"Columns: {df_students.columns.tolist()}")
print(f"Data types:\n{df_students.dtypes}")
print(f"\nInfo lengkap:")
df_students.info()
```

### **2.3.2 Seleksi Data (loc, iloc, filtering)**

```python
print("\n" + "="*60)
print("SELEKSI DATA - Cara Mengambil Data dari DataFrame")
print("="*60)

# Gunakan dataset yang sudah ada
print("Dataset contoh (5 baris pertama):")
print(df_students)

# LOC - Seleksi berdasarkan LABEL
print("\n1. loc (Label-based selection)")
print("-" * 40)
print("df_students.loc[0:2, 'Nama':'Skor']")
print(df_students.loc[0:2, 'Nama':'Skor'])

print("\ndf_students.loc[df_students['Status'] == 'Aktif', ['Nama', 'Skor']]")
print(df_students.loc[df_students['Status'] == 'Aktif', ['Nama', 'Skor']])

# ILOC - Seleksi berdasarkan POSISI
print("\n2. iloc (Position-based selection)")
print("-" * 40)
print("df_students.iloc[0:3, 0:2]  # 3 baris pertama, 2 kolom pertama")
print(df_students.iloc[0:3, 0:2])

print("\ndf_students.iloc[:, -2:]  # Semua baris, 2 kolom terakhir")
print(df_students.iloc[:, -2:])

# CONDITIONAL FILTERING - Seleksi dengan Kondisi
print("\n3. Conditional Filtering (Filter dengan Kondisi)")
print("-" * 40)

# Filter tunggal
aktif_students = df_students[df_students['Status'] == 'Aktif']
print(f"Student aktif:\n{aktif_students}")

# Multiple conditions (AND = &, OR = |)
high_score_aktif = df_students[(df_students['Skor'] >= 85) & (df_students['Status'] == 'Aktif')]
print(f"\nStudent aktif dengan skor >= 85:\n{high_score_aktif}")

# Filter dengan isin()
cities = ['Jakarta', 'Bandung']
jakarta_bandung = df_students[df_students['Kota'].isin(cities)]
print(f"\nStudent dari Jakarta atau Bandung:\n{jakarta_bandung}")

# Filter dengan query (alternatif)
high_score = df_students.query('Skor > 85')
print(f"\nStudent dengan skor > 85 (menggunakan query):\n{high_score}")
```

### **2.3.3 Handling Missing Values**

```python
print("\n" + "="*60)
print("HANDLING MISSING VALUES - Menangani Data Kosong")
print("="*60)

# Membuat dataset dengan missing values
df_missing = pd.DataFrame({
    'Nama': ['Andi', 'Budi', 'Citra', 'Dewi', 'Eka'],
    'Usia': [25, None, 28, None, 26],
    'Gaji': [8000000, 9500000, None, 12000000, 7000000],
    'Kota': ['Jakarta', 'Bandung', None, 'Jakarta', 'Surabaya'],
    'Pengalaman': [2, 5, None, None, 3]
})

print("Dataset dengan missing values:")
print(df_missing)

# 1. DETEKSI MISSING VALUES
print("\n1. DETEKSI MISSING VALUES")
print("-" * 40)
print(f"isnull():\n{df_missing.isnull()}")
print(f"\nJumlah missing per kolom:\n{df_missing.isnull().sum()}")
print(f"Persentase missing per kolom:\n{df_missing.isnull().sum() / len(df_missing) * 100:.1f}%")
print(f"Total missing values: {df_missing.isnull().sum().sum()}")

# 2. DROP MISSING VALUES
print("\n2. DROP MISSING VALUES (Hapus Data Kosong)")
print("-" * 40)

print("Drop baris dengan ANY missing value:")
print(df_missing.dropna())

print("\nDrop baris dengan ALL missing value (harus semua kolom kosong):")
print(df_missing.dropna(how='all'))

print("\nDrop kolom yang memiliki missing value:")
print(df_missing.dropna(axis=1))

# 3. FILL MISSING VALUES (IMPUTASI)
print("\n3. FILL MISSING VALUES (Isi Data Kosong)")
print("-" * 40)

# Fill dengan nilai konstan
print("Fill dengan nilai konstan (0 untuk numerik, 'Unknown' untuk string):")
df_filled_constant = df_missing.copy()
df_filled_constant['Usia'] = df_filled_constant['Usia'].fillna(0)
df_filled_constant['Kota'] = df_filled_constant['Kota'].fillna('Unknown')
print(df_filled_constant)

# Fill dengan mean/median
print("\nFill numeric dengan mean:")
df_filled_mean = df_missing.copy()
df_filled_mean['Usia'] = df_filled_mean['Usia'].fillna(df_filled_mean['Usia'].mean())
df_filled_mean['Gaji'] = df_filled_mean['Gaji'].fillna(df_filled_mean['Gaji'].median())
print(df_filled_mean)

# Fill dengan forward fill (gunakan nilai sebelumnya)
print("\nForward fill (menggunakan nilai dari baris sebelumnya):")
print(df_missing.fillna(method='ffill'))

# Fill dengan backward fill (gunakan nilai setelahnya)
print("\nBackward fill (menggunakan nilai dari baris setelahnya):")
print(df_missing.fillna(method='bfill'))

# 4. INTERPOLASI (Metode yang lebih sophisticated)
print("\n4. INTERPOLASI (Metode Lanjutan)")
print("-" * 40)
df_numeric = df_missing[['Usia', 'Gaji', 'Pengalaman']]
print("Interpolasi linear:")
print(df_numeric.interpolate())
```

### **2.3.4 Groupby, Aggregate, Merge, Join**

```python
print("\n" + "="*60)
print("GROUPBY & AGREGASI - Analisis Data Berkelompok")
print("="*60)

# GROUPBY - Mengelompokkan data berdasarkan kategori
print("1. GROUPBY - Pengelompokan Data")
print("-" * 40)

# Contoh dengan data penjualan
df_sales = pd.DataFrame({
    'Product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse', 'Laptop'],
    'Category': ['Elektronik', 'Aksesoris', 'Elektronik', 'Aksesoris', 'Aksesoris', 'Elektronik'],
    'Sales': [15000000, 150000, 15500000, 500000, 160000, 14800000],
    'Quantity': [1, 2, 1, 1, 3, 1],
    'City': ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung', 'Jakarta']
})

print("Dataset Penjualan:")
print(df_sales)

# Groupby sederhana
print("\nTotal sales per product:")
print(df_sales.groupby('Product')['Sales'].sum())

# Multiple aggregations
print("\nStatistik lengkap per category:")
agg_results = df_sales.groupby('Category').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Quantity': 'sum'
})
print(agg_results)

# Custom aggregation
print("\nRange harga per product:")
def price_range(x):
    return x.max() - x.min()

print(df_sales.groupby('Product')['Sales'].agg(price_range))

# ============================================
# MERGE & JOIN - Menggabungkan Dataset
# ============================================
print("\n" + "="*60)
print("MERGE & JOIN - Menggabungkan Multiple Dataset")
print("="*60)

# Dataset 1: Informasi Customer
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Andi', 'Budi', 'Citra', 'Dewi', 'Eka'],
    'city': ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung']
})

# Dataset 2: Transaksi
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 1, 2, 4, 4],
    'amount': [250000, 300000, 150000, 450000, 200000],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Mouse']
})

print("Tabel Customers:")
print(customers)
print("\nTabel Orders:")
print(orders)

# INNER JOIN - Hanya data yang ada di kedua tabel
print("\n1. INNER JOIN (hanya customer yang pernah order):")
inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
print(inner_join)

# LEFT JOIN - Semua data dari tabel kiri (customers)
print("\n2. LEFT JOIN (semua customer):")
left_join = pd.merge(customers, orders, on='customer_id', how='left')
print(left_join)

# RIGHT JOIN - Semua data dari tabel kanan (orders)
print("\n3. RIGHT JOIN (semua order):")
right_join = pd.merge(customers, orders, on='customer_id', how='right')
print(right_join)

# OUTER JOIN - Semua data dari kedua tabel
print("\n4. OUTER JOIN (semua data):")
outer_join = pd.merge(customers, orders, on='customer_id', how='outer')
print(outer_join)

# Analisis setelah merge
print("\n5. ANALISIS SETELAH MERGE:")
customer_spending = pd.merge(customers, orders, on='customer_id', how='left')
spending_summary = customer_spending.groupby('name').agg({
    'amount': ['sum', 'mean', 'count']
}).fillna(0)
print("Total belanja per customer:")
print(spending_summary)
```

### **2.3.5 PRAKTIK: Eksplorasi Dataset Oil, Diabetes, Mall**

**Link Dataset:**
1. **Oil Production**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-oil-production.csv
2. **Diabetes**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv  
3. **Mall Customers**: https://raw.githubusercontent.com/mohammadmaftoun/Mall-Customers-Segmentation/master/Mall_Customers.csv

```python
print("\n" + "="*60)
print("PRAKTIK EKSPLORASI 3 DATASET (Oil, Diabetes, Mall)")
print("="*60)

# ============================================
# DATASET 1: OIL PRODUCTION (Time Series / Regresi)
# ============================================
print("\n📊 DATASET 1: Monthly Oil Production")
print("-" * 40)

url_oil = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-oil-production.csv"
df_oil = pd.read_csv(url_oil, names=['Month', 'Production'])

print(f"Shape: {df_oil.shape}")
print(f"Info dataset:")
df_oil.info()
print(f"\n5 data pertama:")
print(df_oil.head())
print(f"\n5 data terakhir:")
print(df_oil.tail())
print(f"\nStatistik deskriptif:")
print(df_oil.describe())
print(f"\nMissing values: {df_oil.isnull().sum()}")
print(f"Unique months: {df_oil['Month'].nunique()}")

# ============================================
# DATASET 2: DIABETES (Classification)
# ============================================
print("\n\n📊 DATASET 2: Pima Indians Diabetes")
print("-" * 40)

columns_diabetes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
url_diabetes = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
df_diabetes = pd.read_csv(url_diabetes, names=columns_diabetes)

print(f"Shape: {df_diabetes.shape}")
print(f"Info dataset:")
df_diabetes.info()
print(f"\n5 data pertama:")
print(df_diabetes.head())
print(f"\nStatistik deskriptif:")
print(df_diabetes.describe())
print(f"\nDistribusi Outcome (0 = No Diabetes, 1 = Diabetes):")
print(df_diabetes['Outcome'].value_counts())
print(f"Persentase Diabetes: {df_diabetes['Outcome'].mean() * 100:.1f}%")

# Cek missing values (dalam dataset ini, 0 bisa berarti missing)
print(f"\nNilai 0 pada setiap kolom (bisa jadi missing value):")
zero_counts = (df_diabetes == 0).sum()
print(zero_counts)

# ============================================
# DATASET 3: MALL CUSTOMERS (Clustering)
# ============================================
print("\n\n📊 DATASET 3: Mall Customers Segmentation")
print("-" * 40)

url_mall = "https://raw.githubusercontent.com/mohammadmaftoun/Mall-Customers-Segmentation/master/Mall_Customers.csv"
df_mall = pd.read_csv(url_mall)

print(f"Shape: {df_mall.shape}")
print(f"Info dataset:")
df_mall.info()
print(f"\n5 data pertama:")
print(df_mall.head())
print(f"\nStatistik deskriptif:")
print(df_mall.describe())
print(f"\nUnique values Genre: {df_mall['Genre'].unique()}")
print(f"Distribusi Gender:")
print(df_mall['Genre'].value_counts())
print(f"\nKorelasi antara Annual Income dan Spending Score:")
print(df_mall['Annual Income (k$)'].corr(df_mall['Spending Score (1-100)']))

# ============================================
# EKSPLORASI LANJUTAN
# ============================================
print("\n" + "="*60)
print("EKSPLORASI LANJUTAN - Insight dari Data")
print("="*60)

# 1. Distribusi umur customer mall
print("\n1. Distribusi Umur Customer Mall:")
age_groups = pd.cut(df_mall['Age'], bins=[18, 25, 35, 45, 55, 70], 
                    labels=['18-25', '25-35', '35-45', '45-55', '55-70'])
print(age_groups.value_counts())

# 2. Rata-rata spending score berdasarkan gender dan usia
print("\n2. Rata-rata Spending Score per Gender & Age Group:")
spending_by_group = df_mall.groupby(['Genre', pd.cut(df_mall['Age'], bins=5)])['Spending Score (1-100)'].mean()
print(spending_by_group)

# 3. Korelasi fitur diabetes dengan outcome
print("\n3. Korelasi fitur dengan Diabetes Outcome:")
correlations = df_diabetes.corr()['Outcome'].sort_values(ascending=False)
print(correlations)
print("\nFitur paling berpengaruh untuk prediksi diabetes:")
print(f"- Glucose: {correlations['Glucose']:.3f}")
print(f"- BMI: {correlations['BMI']:.3f}")
print(f"- Age: {correlations['Age']:.3f}")

# 4. Analisis time series oil production
print("\n4. Analisis Produksi Minyak:")
df_oil['Year'] = pd.to_numeric(df_oil['Month'].str.split('-').str[0])
yearly_production = df_oil.groupby('Year')['Production'].mean()
print(f"Rata-rata produksi per tahun:")
print(yearly_production.head())
print(f"Tren: {'Meningkat' if yearly_production.iloc[-1] > yearly_production.iloc[0] else 'Menurun'}")
```

---

## 📊 **2.4 Visualisasi Data dengan Matplotlib & Seaborn**

### **2.4.1 Line Plot, Bar Chart, Histogram, Scatter Plot**

```python
print("\n" + "="*60)
print("VISUALISASI DATA DASAR")
print("="*60)

# LINE PLOT - Untuk data time series
print("\n1. LINE PLOT (Time Series)")
print("-" * 40)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df_oil['Production'][:100])
plt.title('Oil Production Over Time')
plt.xlabel('Month')
plt.ylabel('Production (thousands barrels)')
plt.grid(True, alpha=0.3)

# BAR CHART - Untuk data kategorikal
print("\n2. BAR CHART (Kategorikal)")
print("-" * 40)

plt.subplot(1, 3, 2)
gender_spending = df_mall.groupby('Genre')['Spending Score (1-100)'].mean()
plt.bar(gender_spending.index, gender_spending.values, color=['blue', 'pink'])
plt.title('Average Spending Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Spending Score')
for i, v in enumerate(gender_spending.values):
    plt.text(i, v + 1, f'{v:.1f}', ha='center')

# HISTOGRAM - Untuk distribusi data
print("\n3. HISTOGRAM (Distribusi)")
print("-" * 40)

plt.subplot(1, 3, 3)
plt.hist(df_diabetes['Glucose'], bins=20, edgecolor='black', alpha=0.7)
plt.axvline(df_diabetes['Glucose'].mean(), color='red', linestyle='dashed', 
            label=f'Mean: {df_diabetes["Glucose"].mean():.1f}')
plt.axvline(df_diabetes['Glucose'].median(), color='green', linestyle='dashed',
            label=f'Median: {df_diabetes["Glucose"].median():.1f}')
plt.title('Distribution of Glucose Levels')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# SCATTER PLOT - Hubungan 2 variabel
print("\n4. SCATTER PLOT (Hubungan Variabel)")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot dengan warna berdasarkan gender
scatter = axes[0].scatter(df_mall['Annual Income (k$)'], 
                          df_mall['Spending Score (1-100)'],
                          c=df_mall['Age'], cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('Income vs Spending Score (Colored by Age)')
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score (1-100)')
plt.colorbar(scatter, ax=axes[0], label='Age')

# Scatter plot dengan warna berdasarkan outcome diabetes
colors = {0: 'blue', 1: 'red'}
axes[1].scatter(df_diabetes['Glucose'], df_diabetes['BMI'], 
                c=df_diabetes['Outcome'].map(colors), alpha=0.6, s=30)
axes[1].set_title('Glucose vs BMI (Diabetes: Blue=No, Red=Yes)')
axes[1].set_xlabel('Glucose')
axes[1].set_ylabel('BMI')

plt.tight_layout()
plt.show()
```

### **2.4.2 Heatmap Korelasi**

```python
print("\n" + "="*60)
print("HEATMAP KORELASI - Memahami Hubungan Antar Fitur")
print("="*60)

# Heatmap untuk Diabetes Dataset
print("\n1. HEATMAP DIABETES DATASET")
print("-" * 40)

plt.figure(figsize=(10, 8))
corr_diabetes = df_diabetes.corr()
mask = np.triu(np.ones_like(corr_diabetes, dtype=bool))  # Mask untuk segitiga atas
sns.heatmap(corr_diabetes, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Diabetes Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Interpretasi korelasi
print("\nInterpretasi Korelasi Terkuat dengan Outcome (Diabetes):")
corr_with_outcome = corr_diabetes['Outcome'].drop('Outcome').sort_values(ascending=False)
for feature, corr in corr_with_outcome.head(3).items():
    print(f"  - {feature}: {corr:.3f} ({'Positif' if corr > 0 else 'Negatif'})")

# Heatmap untuk Mall Dataset
print("\n2. HEATMAP MALL DATASET")
print("-" * 40)

plt.figure(figsize=(8, 6))
mall_numeric = df_mall.select_dtypes(include=[np.number])
corr_mall = mall_numeric.corr()
sns.heatmap(corr_mall, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap - Mall Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nInsight dari Heatmap Mall:")
print(f"  - Korelasi Age vs Spending Score: {corr_mall.loc['Age', 'Spending Score (1-100)']:.3f}")
print(f"  - Korelasi Income vs Spending Score: {corr_mall.loc['Annual Income (k$)', 'Spending Score (1-100)']:.3f}")
```

### **2.4.3 Customisasi Plot (Label, Title, Legend, Color)**

```python
print("\n" + "="*60)
print("CUSTOMISASI PLOT - Membuat Visualisasi Profesional")
print("="*60)

# Membuat figure dengan multiple subplots yang di-customize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Advanced Data Visualization - Mall Customers Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Boxplot dengan custom colors
ax1 = axes[0, 0]
box_data = [df_mall[df_mall['Genre'] == genre]['Annual Income (k$)'] 
            for genre in df_mall['Genre'].unique()]
bp = ax1.boxplot(box_data, labels=df_mall['Genre'].unique(), 
                 patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightpink']):
    patch.set_facecolor(color)
ax1.set_title('Income Distribution by Gender', fontsize=12, fontweight='bold')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Annual Income (k$)')
ax1.grid(True, alpha=0.3)

# Plot 2: Violin plot dengan seaborn
ax2 = axes[0, 1]
sns.violinplot(data=df_mall, x='Genre', y='Spending Score (1-100)', 
               ax=ax2, palette=['lightblue', 'lightpink'])
ax2.set_title('Spending Score Distribution by Gender', fontsize=12, fontweight='bold')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Spending Score (1-100)')

# Plot 3: Stacked bar chart
ax3 = axes[1, 0]
age_groups = pd.cut(df_mall['Age'], bins=[18, 30, 40, 50, 70], 
                    labels=['18-30', '30-40', '40-50', '50-70'])
gender_age = pd.crosstab(age_groups, df_mall['Genre'])
gender_age.plot(kind='bar', stacked=True, ax=ax3, color=['lightblue', 'lightpink'])
ax3.set_title('Age Distribution by Gender', fontsize=12, fontweight='bold')
ax3.set_xlabel('Age Group')
ax3.set_ylabel('Count')
ax3.legend(title='Gender')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Scatter dengan trend line
ax4 = axes[1, 1]
for genre in df_mall['Genre'].unique():
    subset = df_mall[df_mall['Genre'] == genre]
    ax4.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                label=genre, alpha=0.6, s=50)
    # Tambah trend line
    z = np.polyfit(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], 1)
    p = np.poly1d(z)
    ax4.plot(subset['Annual Income (k$)'].sort_values(), 
             p(subset['Annual Income (k$)'].sort_values()), 
             linestyle='--', alpha=0.8)
ax4.set_title('Income vs Spending Score with Trend Lines', fontsize=12, fontweight='bold')
ax4.set_xlabel('Annual Income (k$)')
ax4.set_ylabel('Spending Score (1-100)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Customisasi dengan style yang berbeda
print("\n📌 TIPS CUSTOMISASI PLOT:")
print("1. Gunakan grid() untuk memudahkan pembacaan nilai")
print("2. Atur alpha untuk transparansi (0-1)")
print("3. Tambah trend line untuk melihat pola")
print("4. Gunakan colormap yang colorblind-friendly (viridis, plasma, inferno)")
print("5. Selalu beri label pada sumbu dan judul yang informatif")
```

### **2.4.4 PRAKTIK: Visualisasi Distribusi Tiap Fitur di 3 Dataset**

```python
print("\n" + "="*60)
print("PRAKTIK: Visualisasi Distribusi 3 Dataset")
print("="*60)

# Membuat visualisasi lengkap untuk semua fitur di 3 dataset
fig = plt.figure(figsize=(20, 15))
fig.suptitle('Complete Feature Distribution Analysis - 3 Datasets', 
             fontsize=16, fontweight='bold', y=0.98)

# ============================================
# DATASET 1: OIL PRODUCTION
# ============================================
print("\n📊 Visualisasi Oil Production Dataset...")

# Time series plot
ax1 = plt.subplot(3, 4, 1)
ax1.plot(df_oil['Production'][:100], color='blue', linewidth=1)
ax1.set_title('Oil Production (Time Series)', fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Production')
ax1.grid(True, alpha=0.3)

# Histogram distribusi
ax2 = plt.subplot(3, 4, 2)
ax2.hist(df_oil['Production'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
ax2.set_title('Distribution of Oil Production', fontweight='bold')
ax2.set_xlabel('Production')
ax2.set_ylabel('Frequency')
ax2.axvline(df_oil['Production'].mean(), color='red', linestyle='dashed', 
            label=f'Mean: {df_oil["Production"].mean():.1f}')
ax2.legend()

# Boxplot
ax3 = plt.subplot(3, 4, 3)
ax3.boxplot(df_oil['Production'].dropna(), vert=False)
ax3.set_title('Boxplot - Oil Production', fontweight='bold')
ax3.set_xlabel('Production')

# Rolling mean (moving average)
ax4 = plt.subplot(3, 4, 4)
rolling_mean = df_oil['Production'].rolling(window=12).mean()
ax4.plot(df_oil['Production'][:100], alpha=0.5, label='Original')
ax4.plot(rolling_mean[:100], color='red', linewidth=2, label='12-month MA')
ax4.set_title('Production with Moving Average', fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Production')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============================================
# DATASET 2: DIABETES
# ============================================
print("\n📊 Visualisasi Diabetes Dataset...")

# Feature distributions
features_diabetes = ['Glucose', 'BMI', 'Age']
for i, feature in enumerate(features_diabetes):
    ax = plt.subplot(3, 4, 5 + i)
    for outcome in [0, 1]:
        subset = df_diabetes[df_diabetes['Outcome'] == outcome][feature]
        ax.hist(subset, bins=20, alpha=0.5, label=f'Outcome {outcome}')
    ax.set_title(f'Distribution of {feature}', fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.axvline(df_diabetes[feature].mean(), color='black', linestyle='dashed', alpha=0.5)

# Correlation with outcome
ax8 = plt.subplot(3, 4, 8)
corr_with_outcome = df_diabetes.corr()['Outcome'].drop('Outcome').sort_values()
colors_bar = ['red' if x < 0 else 'green' for x in corr_with_outcome.values]
ax8.barh(corr_with_outcome.index, corr_with_outcome.values, color=colors_bar)
ax8.set_title('Feature Correlation with Diabetes', fontweight='bold')
ax8.set_xlabel('Correlation Coefficient')
ax8.axvline(0, color='black', linewidth=0.5)

# ============================================
# DATASET 3: MALL CUSTOMERS
# ============================================
print("\n📊 Visualisasi Mall Dataset...")

# Age distribution
ax9 = plt.subplot(3, 4, 9)
ax9.hist(df_mall['Age'], bins=20, edgecolor='black', alpha=0.7, color='purple')
ax9.set_title('Age Distribution of Customers', fontweight='bold')
ax9.set_xlabel('Age')
ax9.set_ylabel('Frequency')

# Income vs Spending by gender
ax10 = plt.subplot(3, 4, 10)
for genre in df_mall['Genre'].unique():
    subset = df_mall[df_mall['Genre'] == genre]
    ax10.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                 label=genre, alpha=0.6, s=30)
ax10.set_title('Income vs Spending Score', fontweight='bold')
ax10.set_xlabel('Annual Income (k$)')
ax10.set_ylabel('Spending Score')
ax10.legend()

# Heatmap correlation (small)
ax11 = plt.subplot(3, 4, 11)
mall_corr = df_mall.select_dtypes(include=[np.number]).corr()
im = ax11.imshow(mall_corr, cmap='coolwarm', aspect='auto')
ax11.set_xticks(range(len(mall_corr.columns)))
ax11.set_yticks(range(len(mall_corr.columns)))
ax11.set_xticklabels(mall_corr.columns, rotation=45, ha='right', fontsize=8)
ax11.set_yticklabels(mall_corr.columns, fontsize=8)
ax11.set_title('Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax11, shrink=0.8)

# Spending score distribution by gender (violin)
ax12 = plt.subplot(3, 4, 12)
violin_parts = ax12.violinplot([df_mall[df_mall['Genre'] == 'Male']['Spending Score (1-100)'],
                                 df_mall[df_mall['Genre'] == 'Female']['Spending Score (1-100)']],
                                positions=[1, 2], showmeans=True, showmedians=True)
ax12.set_xticks([1, 2])
ax12.set_xticklabels(['Male', 'Female'])
ax12.set_title('Spending Score Distribution', fontweight='bold')
ax12.set_ylabel('Spending Score (1-100)')
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ VISUALISASI SELESAI!")
print("\n📝 INSIGHT YANG DAPAT DIAMBIL:")
print("\n1. OIL DATASET:")
print("   - Produksi minyak menunjukkan tren tertentu (lihat moving average)")
print("   - Distribusi cenderung normal/skewed")
print("\n2. DIABETES DATASET:")
print("   - Glucose, BMI, dan Age adalah prediktor terkuat")
print("   - Distribusi fitur berbeda antara pasien diabetes dan non-diabetes")
print("\n3. MALL DATASET:")
print("   - Tidak ada korelasi kuat antara income dan spending score")
print("   - Distribusi spending score relatif sama antara gender")
print("   - Ada cluster alami dalam data (bisa untuk segmentation)")
```

---

## 📚 **Ringkasan & Referensi**

### **Link Dataset (Lengkap):**

| Dataset | Link | Tipe |
|---------|------|------|
| Oil Production | https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-oil-production.csv | Time Series/Regresi |
| Diabetes | https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv | Klasifikasi |
| Mall Customers | https://raw.githubusercontent.com/mohammadmaftoun/Mall-Customers-Segmentation/master/Mall_Customers.csv | Clustering |

### **Link Dokumentasi Resmi:**

| Library | Dokumentasi | Fitur Utama |
|---------|-------------|-------------|
| **Pandas** | https://pandas.pydata.org/docs/ | Data manipulation, I/O, groupby |
| **NumPy** | https://numpy.org/doc/stable/ | Array operations, linear algebra |
| **Matplotlib** | https://matplotlib.org/stable/ | Basic plotting, customization |
| **Seaborn** | https://seaborn.pydata.org/ | Statistical visualizations |

### **Cheat Sheet Quick Reference:**

```python
# PANDAS CHEAT SHEET
# Baca data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_html('url')[0]

# Lihat data
df.head(), df.tail(), df.sample()
df.info(), df.describe(), df.shape

# Seleksi data
df.loc[label], df.iloc[position]
df[df['kolom'] > nilai]
df.query('kolom > nilai')

# Handle missing
df.isnull().sum()
df.dropna()
df.fillna(df.mean())

# Groupby
df.groupby('kolom').agg({'kolom2': 'mean'})

# Merge
pd.merge(df1, df2, on='key', how='inner')

# NUMPY CHEAT SHEET
arr = np.array([1,2,3])
arr = np.zeros((3,3))
arr = np.random.randn(10)

# Operasi
arr + 10, arr * 2, arr ** 2
np.dot(v1, v2)
A @ B  # matrix multiplication
np.linalg.inv(A)

# MATPLOTLIB CHEAT SHEET
plt.plot(x, y)  # line plot
plt.scatter(x, y)  # scatter plot
plt.hist(x, bins=20)  # histogram
plt.bar(x, y)  # bar chart
plt.title(), plt.xlabel(), plt.ylabel()
plt.legend(), plt.grid(), plt.colorbar()

# SEABORN CHEAT SHEET
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df, hue='target')
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)
```

### **Tips Penting untuk AI/ML:**

1. **Selalu eksplorasi data sebelum modeling** - Visualisasi adalah kunci
2. **Handle missing values dengan tepat** - Jangan asal drop
3. **Normalisasi/Standarisasi** - Penting untuk banyak algoritma ML
4. **Cek multikolinearitas** - Korelasi tinggi antar fitur bisa jadi masalah
5. **Visualisasi hasil** - Jangan hanya lihat angka, lihat grafiknya
