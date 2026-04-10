# 🐍 Introduction to Python Programming

## Table of Contents & Materials

---

## 📍 Materi 1: Introduction to Python

[🔗 Link to Notebook](https://colab.research.google.com/drive/1dBvLlvZKBfkkEq_MCzORCioR9imydV1M?usp=sharing)

**Penjelasan:**
Pertemuan ini merupakan pengenalan dasar Python. Anda akan belajar cara menginstal Python di komputer, memahami aturan penulisan kode (syntax), mengenal editor yang cocok untuk pemrograman Python, membuat file program pertama, dan mencetak teks "Hello World" sebagai program tradisional pertama.

- **1.1 Installing Python** - Proses mengunduh dan menginstal Python dari python.org, serta memastikan PATH terkonfigurasi dengan benar.
- **1.2 Syntax Rules in Python** - Aturan dasar penulisan kode Python seperti indentasi, case-sensitive, dan penggunaan komentar.
- **1.3 Python Programming Editor** - Pengenalan berbagai editor seperti VS Code, PyCharm, Jupyter Notebook, dan Google Colab.
- **1.4 Creating a Program File** - Cara membuat file berekstensi .py dan menyimpannya untuk dieksekusi.
- **1.5 Printing "Hello World"** - Menggunakan fungsi `print()` untuk menampilkan teks ke layar sebagai program pertama.

---

## 📍 Materi 2: Data Types & Operators

[🔗 Link to Notebook](https://colab.research.google.com/drive/1VTUrKAHR8UpUa4tUunes7CdHfmXBUOPt?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas jenis-jenis data yang dapat disimpan dalam variabel Python serta operator-operator yang digunakan untuk memanipulasi data tersebut. Pemahaman tentang tipe data dan operator sangat penting karena menjadi fondasi dari semua operasi komputasi.

- **2.2 Data Types** - Jenis data dasar Python: integer (angka bulat), float (desimal), string (teks), boolean (True/False), dan complex (bilangan kompleks).
- **2.3 Data Type Conversion** - Mengubah tipe data menggunakan fungsi `int()`, `float()`, `str()`, dan `bool()`.
- **2.4 Operators** - Simbol khusus untuk melakukan operasi pada data:
  - **2.4.1 Arithmetic Operators** - Operator matematika: `+` (tambah), `-` (kurang), `*` (kali), `/` (bagi), `//` (bagi bulat), `%` (modulus/sisa bagi), `**` (pangkat).
  - **2.4.2 Assignment Operators** - Operator penugasan: `=`, `+=`, `-=`, `*=`, `/=` untuk menetapkan nilai ke variabel.
  - **2.4.3 Comparison Operators** - Operator perbandingan: `==`, `!=`, `>`, `<`, `>=`, `<=` menghasilkan nilai boolean.
  - **2.4.4 Logical Operators** - Operator logika: `and`, `or`, `not` untuk menggabungkan kondisi boolean.
  - **2.4.5 Bitwise Operators** - Operator bit-level: `&` (AND), `|` (OR), `^` (XOR), `~` (NOT), `<<` (shift kiri), `>>` (shift kanan).
  - **2.4.6 Ternary Operator** - Operator kondisional satu baris: `nilai_benar if kondisi else nilai_salah`.

---

## 📍 Materi 3: Conditions & Loops

[🔗 Link to Notebook](https://colab.research.google.com/drive/1Oa7pJrngnVK_Qpwyy6UADtc7pjZXCpw7?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas struktur kontrol alur program. Conditions (percabangan) memungkinkan program mengambil keputusan berdasarkan kondisi tertentu. Loops (perulangan) memungkinkan program mengeksekusi blok kode berulang kali.

- **3.1 Basic If Structure** - Struktur `if` dasar untuk mengeksekusi kode hanya jika kondisi bernilai True.
- **3.2 If/Elif/Else Branching** - Percabangan dengan banyak kondisi menggunakan `if`, `elif` (else if), dan `else` untuk menangani semua kemungkinan.
- **3.3 Looping** - Konsep perulangan dalam pemrograman.
- **3.4 For Loops** - Perulangan untuk mengiterasi sequence (list, string, range) menggunakan `for item in sequence:`.
- **3.5 While Loops** - Perulangan yang berjalan selama kondisi bernilai True, menggunakan `while kondisi:`.

---

## 📍 Materi 4: Functions

[🔗 Link to Notebook](https://colab.research.google.com/drive/1whtoGRHVKrd2TlEuMipQl7dhq-ikjqdt?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas fungsi, yaitu blok kode yang dapat digunakan kembali (reusable). Fungsi membantu memecah program menjadi bagian-bagian yang lebih kecil, terorganisir, dan mudah dikelola.

- **4.1 Functions with Parameters** - Membuat fungsi yang menerima input (parameter/argumen) untuk diproses di dalam fungsi.
- **4.2 Functions Returning Values** - Menggunakan keyword `return` untuk mengirimkan hasil olahan fungsi kembali ke pemanggil.
- **4.3 Variable Scope in Functions** - Konsep lingkup variabel: variabel lokal (di dalam fungsi) vs variabel global (di luar fungsi).
- **4.4 Built-in Python Functions** - Fungsi bawaan Python yang siap pakai seperti `print()`, `len()`, `type()`, `input()`, `range()`, `max()`, `min()`, `sum()`.

---

## 📍 Materi 5: Lists

[🔗 Link to Notebook](https://colab.research.google.com/drive/11EkpTvaurHZm8P8UNwBuu0gXArsmJpzI?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas List, yaitu tipe data koleksi yang paling fleksibel di Python. List dapat menyimpan berbagai tipe data, bersifat mutable (dapat diubah), dan memiliki urutan (ordered).

- **5.1 Lists** - Cara membuat list menggunakan kurung siku `[]` atau fungsi `list()`, misal: `buah = ["apel", "mangga", "jeruk"]`.
- **5.2 Accessing List Values** - Mengakses elemen list menggunakan indeks (dimulai dari 0) dan indeks negatif (dari belakang).
- **5.3 Modifying List Values** - Mengubah nilai elemen list dengan assignment: `buah[0] = "pisang"`.
- **5.4 Adding Items to a List** - Menambah elemen: `append()` (di akhir), `insert(posisi, nilai)` (di tengah), `extend()` (menggabungkan list).
- **5.5 Removing Items from a List** - Menghapus elemen: `remove(nilai)` (hapus berdasarkan nilai), `pop(indeks)` (hapus berdasarkan indeks), `clear()` (hapus semua).
- **5.6 Slicing Lists** - Mengambil sebagian list menggunakan `[start:end:step]`, misal `buah[1:3]`.
- **5.7 List Operations** - Operasi pada list seperti concatenation `+`, repetition `*`, `in` operator untuk pengecekan keanggotaan.
- **5.8 Multi-dimensional Lists** - List di dalam list (list bersarang) untuk merepresentasikan matriks atau struktur data kompleks.

---

## 📍 Materi 6: Tuples & Sets

[🔗 Link to Notebook](https://colab.research.google.com/drive/1lWseBChR2R2XxLbuQ2k2jrT6fDq0i0u4?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas dua tipe data koleksi lainnya: Tuple (mirip list tetapi tidak bisa diubah/immutable) dan Set (koleksi unik tanpa urutan).

- **6.1 Creating and Accessing Tuples** - Membuat tuple dengan kurung `()` atau fungsi `tuple()`. Akses elemen sama seperti list menggunakan indeks.
- **6.2 Slicing Tuples** - Pengambilan sebagian tuple dengan sintaks `[start:end:step]` (menghasilkan tuple baru).
- **6.4 Nested Tuples** - Tuple di dalam tuple untuk menyimpan data terstruktur.
- **6.5 Iterating Over Tuples** - Perulangan pada tuple menggunakan `for item in tuple:`.
- **6.6 Tuple Methods** - Method bawaan tuple: `count()` untuk menghitung kemunculan nilai, `index()` untuk mencari posisi nilai.
- **6.7 Sets** - Membuat set dengan kurung kurawal `{}` atau fungsi `set()`. Set menyimpan elemen unik (tidak ada duplikat) dan tidak berurutan.
- **6.8 Modifying and Removing Set Items** - Menambah dengan `add()`, menghapus dengan `remove()` atau `discard()`, menghapus semua dengan `clear()`.
- **6.9 Set Operations** - Operasi himpunan: union (`|` atau `union()`), intersection (`&` atau `intersection()`), difference (`-` atau `difference()`), symmetric difference (`^`).

---

## 📍 Materi 7: Dictionaries

[🔗 Link to Notebook](https://colab.research.google.com/drive/1HpccVEVa_ZCtqqJ2vt6X1YMBMxXXP8Mf?usp=sharing)

**Penjelasan:**
Pertemuan ini membahas Dictionary, yaitu tipe data koleksi yang menyimpan data dalam pasangan **key-value** (kunci-nilai). Dictionary sangat berguna untuk merepresentasikan data terstruktur seperti record atau objek.

- **7.1 Creating Dictionaries** - Membuat dictionary dengan kurung kurawal `{}` dan tanda titik dua `:`, misal: `siswa = {"nama": "Budi", "umur": 17}`.
- **7.2 Using Constructors** - Membuat dictionary menggunakan fungsi `dict()`, misal: `siswa = dict(nama="Budi", umur=17)`.
- **7.3 Accessing Dictionary Values** - Mengakses nilai menggunakan key: `siswa["nama"]` atau `siswa.get("nama")`.
- **7.4 Using Loops with Dictionaries** - Perulangan dictionary: `.keys()` untuk kunci, `.values()` untuk nilai, `.items()` untuk pasangan key-value.
- **7.5 Modifying and Deleting Dictionary Items** - Mengubah nilai dengan assignment `siswa["umur"] = 18`. Menghapus dengan `del siswa["umur"]` atau `pop()`.
- **7.6 Adding Items to a Dictionary** - Menambah pasangan baru cukup dengan assignment `siswa["kelas"] = "12A"`.
- **7.7 Getting Dictionary Length** - Mengetahui jumlah pasangan key-value dengan fungsi `len()`.
- **7.8 Dictionary Methods** - Method bawaan dictionary: `keys()`, `values()`, `items()`, `get()`, `pop()`, `popitem()`, `update()`, `clear()`, `copy()`.

---

**Catatan:** Semua notebook dapat dijalankan langsung di Google Colab tanpa perlu instalasi Python di komputer lokal. Cukup klik link pada setiap pertemuan.
