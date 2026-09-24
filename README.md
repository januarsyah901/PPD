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

---

## 4. Standar Penulisan Laporan (LaTeX)

Setiap pertemuan praktikum disertai pembuatan laporan resmi menggunakan **LaTeX**. Acuan struktur: `P1/laporan/` dan `P2/laporan/`.

### A. Struktur Folder Laporan
Setiap folder pertemuan `Pn/` memiliki folder `laporan/`:
```text
Pn/
├── project/
│   └── p{n}_{topik}.ipynb                   # Notebook Colab
└── laporan/
    ├── PPD_P{n}_Januarsyah Akbar_535846.tex # Source code LaTeX
    ├── lambang ugm.png                      # Logo resmi UGM untuk cover
    └── gambar/                              # Tangkapan layar / grafik hasil praktikum
```

### B. Format Penamaan File Laporan
- **Format**: `PPD_P{n}_Januarsyah Akbar_535846.tex`
- **Contoh**: `P1/laporan/PPD_P1_Januarsyah Akbar_535846.tex`

### C. Sistematika Isi Laporan

Empat bab (`report` class). Heading LaTeX berhenti di `\subsection`. Isi tugas pakai `\enumerate` biasa (1, 2, 3), **bukan** `\subsubsection`.

```text
Cover
Daftar Isi

BAB I   Tujuan Praktikum
        enumerate capaian (tanpa section)

BAB II  Dasar Teori
        \section per konsep (contoh P1: NumPy, Pandas;
        P2: AI, ML, EDA, Data Preprocessing)

BAB III Hasil dan Pembahasan
        \section{Langkah Praktikum}          % P2: Langkah Percobaan
            \subsection{Langkah Percobaan ...}  per topik/dataset
        \section{Tugas}                      % P2: Tugas dan Analisis
            \subsection{Tugas N / Bagian X: ...}
                1. Deskripsi Tugas
                2. Implementasi Kode
                3. Hasil Eksekusi
                4. Analisis dan Pembahasan

BAB IV  Kesimpulan
        enumerate (tanpa section)

Daftar Pustaka
```

Contoh heading P1:
- `3.1` Langkah Praktikum → `3.1.1` NumPy, `3.1.2` Pandas
- `3.2` Tugas → `3.2.1` Tugas 1, `3.2.2` Tugas 2, `3.2.3` Tugas 3

Contoh heading P2:
- `3.1` Langkah Percobaan → `3.1.1` EDA, `3.1.2` Data Preprocessing (Titanic)
- `3.2` Tugas dan Analisis → `3.2.1` Bagian A (Bank Churners), `3.2.2` Bagian B (Bengaluru)

### D. Pengelolaan File Build LaTeX di Git
Sesuai konfigurasi `.gitignore`, **hanya file sumber `.tex` dan aset gambar (`gambar/`, `*.png`)** yang di-push ke repositori. File build sementara seperti `.aux`, `.log`, `.toc`, `.out`, `.synctex.gz`, dan `.pdf` otomatis diabaikan agar repositori tetap bersih dan ringan.

---

## 5. Otomasi Pembuatan Laporan (AI Assistant End-to-End)

Untuk mempercepat pengerjaan laporan praktikum di setiap pertemuan, gunakan format instruksi berikut kepada AI Assistant:

- **Trigger / Perintah:**
  - `"Gas buatin laprak pertemuan [N] full end-to-end sampai PDF jadi"`
  - *(atau: `"Buat laprak P[N] lengkap sama gambar dan compile LaTeX-nya"`)*

- **Standard Pipeline Otomatis yang Dijalankan:**
  1. **Analisis Modul & Notebook**: Membaca modul `Pn/MODUL_PDD_Pn.md` dan struktur kode di `Pn/project/`.
  2. **Ekstraksi Gambar/Grafik**: Menjalankan skrip Python untuk merender seluruh plot percobaan & tugas ke `Pn/laporan/gambar/`.
  3. **Penulisan LaTeX (`.tex`)**: Menyusun laporan lengkap mengikuti sistematika Bab III di atas (bukan 5 bab). Metadata nama/NIM/kelas UGM, code listings, tabel, analisis.
  4. **Kompilasi Otomatis PDF**: Menjalankan `latexmk -pdf -interaction=nonstopmode` hingga file `PPD_P{n}_Januarsyah Akbar_535846.pdf` siap kumpul.

---

## 6. Preferensi & Karakteristik Penulisan Bang Jan

Panduan khusus bagi asisten AI saat mendampingi atau menyusun laporan praktikum bersama Bang Jan:

1. **Otentisitas Laporan (No Hardcoded Output/Code)**
   - Jangan menyajikan output eksekusi atau tabel hasil evaluasi menggunakan teks tiruan (seperti environment `lstlisting` atau `tabularx` ganda yang di-hardcode).
   - Selalu gunakan bukti visual riil berupa tangkapan layar Google Colab (`\screenshotimage{...}`) lengkap dengan tanda centang hijau eksekusi dan tabel output aslinya.
   - Tabel LaTeX hanya digunakan jika benar-benar diperlukan sebagai pelengkap, bukan pengganti tangkapan layar.

2. **Kebersihan Ruang Kerja (Clean Workspace)**
   - Selalu bersihkan file sementara atau folder penampung gambar mentah (seperti folder `gambar ss`) setelah gambar dipilah dan direname ke folder `gambar/`.
   - Repositori harus selalu rapi tanpa sampah file yang menumpuk.

3. **Gaya Penulisan Natural, Ringkas, dan Mengalir**
   - **Bukan Format List Berlebihan**: Hindari memecah narasi teknis ke dalam deretan bullet points atau penomoran kaku (`itemize`/`enumerate`), terutama di bagian Pembahasan dan Bab Kesimpulan. Gunakan format paragraf utuh.
   - **Paragraf Pendek dan Nyaman Dibaca**: Jangan membuat paragraf tebal bertumpuk (*wall of text*). Pecah ide pembahasan menjadi beberapa paragraf pendek (2 hingga 4 kalimat per paragraf) agar alur membaca tetap ringan dan fokus.
   - **Bebas dari Slop AI**: Hindari frasa klise robotik, pembuka klise, em dash/en dash berlebihan, dan sampaikan analisis secara lugas langsung ke inti permasalahan.

4. **Instruksi Berorientasi Eksekusi Langsung**
   - Berikan solusi kode atau naskah yang siap pakai.
   - Jalankan proses hingga tahapan *final build* (kompilasi dokumen PDF) selesai secara mandiri tanpa bertele-tele.

