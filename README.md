# PPD — Praktikum Penambangan Data

## 1. Manfaat Mata Kuliah

Matakuliah praktikum penambangan data fokus pada penerapan konsep dan teknik dalam menggali serta menganalisis informasi berharga dari sejumlah besar data. Dalam praktikum ini, mahasiswa diberikan kesempatan untuk memahami dan mengimplementasikan metode-metode penambangan data seperti clustering, klasifikasi, asosiasi, dan regresi pada dataset nyata.

## 2. Deskripsi Perkuliahan

Mata kuliah ini akan mengimplementasikan teknik-teknik dalam data mining maupun machine learning menggunakan bahasa pemrograman Python. Detail dapat dilihat di bawah ini:

- Aplikasi kecerdasan buatan, machine learning dan aplikasinya pada kasus sederhana.
- Aplikasi Algoritma pembelajaran terawasi (supervised learning) pada kasus sederhana.
- Aplikasi Algoritma pembelajaran tak-terawasi (unsupervised learning) pada kasus sederhana.
- Aplikasi NLP (Natural Language Processing) pada kasus sederhana.
- Aplikasi Computer vision pada kasus sederhana.

---

## 3. Workflow Praktikum (Google Colab & GitHub)

Untuk menjalankan notebook praktikum secara efektif menggunakan Google Colab yang terhubung langsung dengan repositori ini:

### A. Cara Membuka & Menjalankan Notebook di Colab
1. Buka [Google Colaboratory](https://colab.research.google.com/).
2. Pilih tab **GitHub**.
3. Masukkan repositori: `januarsyah901/PPD` (atau URL repositori ini).
  4. Pilih branch `main`.
  5. Pilih file notebook `.ipynb` yang ingin dijalankan (misal: `P1/project/p1_numpy_pandas.ipynb`).

### B. Menyimpan Hasil Praktikum Kembali ke GitHub
1. Setelah selesai running dan mengerjakan tugas di Google Colab, klik menu **File** $\rightarrow$ **Save a copy in GitHub** (Simpan salinan di GitHub).
2. Pilih repositori `januarsyah901/PPD`, tentukan branch tujuan, dan isi pesan commit.

### C. Penamaan File Notebook

Format: `p{n}_{topik}.ipynb` — nomor pertemuan + isi yang dibahas, snake_case, tanpa spasi.

Contoh:
- `P1/project/p1_numpy_pandas.ipynb`
- `P4/project/p4_klasifikasi.ipynb`

Letakkan di `Pn/project/`.

### D. Standar Notebook Praktikum (Colab-Ready)
Semua notebook praktikum di repositori ini didesain agar kompatibel dan ramah Google Colab:
- **Penanganan Dataset**: Apabila praktikum memerlukan dataset lokal di repositori, cell awal akan otomatis menangani clone repo atau download dataset via URL Raw GitHub.
- **Dependencies**: Perintah instalasi paket (`!pip install ...`) disiapkan jika memerlukan library tambahan di luar bawaan Colab.

### E. Tips Penting & Penanganan Dataset di Colab
> **Catatan:** Saat membuka file `.ipynb` langsung dari GitHub via Google Colab, Colab **hanya memuat file notebook ke browser** dan tidak otomatis meng-clone seluruh repositori/dataset ke VM runtime Colab.

Untuk mengakses dataset atau modul pendukung di Colab, gunakan salah satu solusi berikut di cell inisialisasi:

#### 1. Auto-Clone Repositori (Rekomendasi)
```python
import os

# Clone repo jika runtime berjalan di Google Colab dan folder belum ada
if 'google.colab' in str(get_ipython()):
    if not os.path.exists('PPD'):
        !git clone https://github.com/januarsyah901/PPD.git
        %cd PPD/P1/project
```

#### 2. Membaca Dataset Langsung via Raw GitHub URL
```python
import pandas as pd

url = "https://raw.githubusercontent.com/januarsyah901/PPD/main/P1/project/dataset.csv"
df = pd.read_csv(url)
```

#### 3. Mount Google Drive (Jika dataset sangat besar)
```python
from google.colab import drive
drive.mount('/content/drive')
```
