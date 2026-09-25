# PPD - Praktikum Penambangan Data

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

Format: `p{n}_{topik}.ipynb` (nomor pertemuan + topik pembahasan, snake_case, tanpa spasi).

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

Empat bab (`report` class). Penomoran tugas tidak menggunakan hirarki bertingkat (tanpa leveling seperti 3.2.1 atau 3.2.2). Cukup gunakan nomor tugas langsung: `Tugas 1: [Judul]` dan `Tugas 2: [Judul]`. Bab Kesimpulan disusun dalam bentuk paragraf naratif utuh, bukan daftar list bernomor.

```text
Cover (Desain standar P5 dengan titlingpage dan logo UGM height=7cm)
Daftar Isi (Penomoran romawi, awal isi bab mulai halaman 3 angka arab)

BAB I   Tujuan Praktikum
        enumerate capaian (tanpa nomor section)

BAB II  Dasar Teori
        \section per konsep materi

BAB III Hasil dan Pembahasan
        \section{Langkah Praktikum}
            \subsection{Langkah Percobaan ...}
        \section{Tugas}
            \subsection*{Tugas 1: ...} (tanpa leveling digit 3.2.1)
            \subsection*{Tugas 2: ...} (tanpa leveling digit 3.2.2)

BAB IV  Kesimpulan
        Paragraf naratif utuh (dilarang menggunakan list atau enumerate)

Daftar Pustaka
```

### D. Pengelolaan File Build dan Gambar di Git
Sesuai konfigurasi `.gitignore`, berkas tangkapan layar (`*.png`) serta file build sementara (`.aux`, `.log`, `.toc`, `.out`, `.synctex.gz`, dan `.pdf`) otomatis diabaikan agar tidak terdorong ke repositori remote. Hal ini menjaga ukuran repositori tetap ringan dan bersih, dengan fokus pelacakan pada berkas sumber naskah `.tex` dan modul kode praktikum.

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

## 6. Preferensi dan Karakteristik Penulisan Laporan Bang Jan

Panduan khusus dan kriteria wajib bagi asisten AI saat mendampingi atau menyusun laporan praktikum bersama Bang Jan:

1. **Desain Sampul dan Tata Letak (Standar P5)**
   - Format halaman judul selalu mengacu pada desain P5 menggunakan environment `titlingpage`.
   - Lambang resmi UGM dipasang proporsional dengan tinggi `height=7cm`.
   - Informasi penyusun wajib mencantumkan Nama, NIM, Kelas, serta Dosen Pengampu lengkap.
   - Penomoran romawi berlaku pada halaman awal dan penomoran arab dimulai dari halaman 3 setelah Daftar Isi. Judul bab pada Daftar Isi disusun rapi tanpa nomor ganda.

2. **Bukti Eksekusi Murni Screenshot (Anti-Output Manual)**
   - Tidak boleh ada hasil eksekusi terminal, log, atau tabel output yang diketik manual di dalam berkas LaTeX (hindari `verbatim` atau kotak teks imitasi).
   - Seluruh bukti keberhasilan run kode, pembentukan data, dan hasil metrik evaluasi wajib berupa tangkapan layar asli Google Colab (`\screenshotimage`).
   - Teks laporan murni difungsikan untuk narasi analisis, pembahasan pola, serta interpretasi bisnis atas hasil yang tampak pada screenshot.

3. **Penomoran Tugas Bebas Hirarki (Tanpa Leveling)**
   - Penulisan subbab tugas tidak menggunakan nomor bertingkat seperti 3.2.1 atau 3.2.2.
   - Format judul tugas ditulis lugas dengan nomor tugas langsung, misalnya `Tugas 1: [Topik]` dan `Tugas 2: [Topik]`.
   - Penyesuaian diimplementasikan menggunakan `\subsection*{...}` yang didampingi perintah `\addcontentsline` agar entri Daftar Isi tetap sejajar rapi di bawah seksi tugas.

4. **Bab Kesimpulan Wajib Paragraf Utuh (Tanpa List atau Enumerate)**
   - Bab Kesimpulan dilarang keras berbentuk poin-poin daftar bernomor (`enumerate` atau `itemize`).
   - Kesimpulan wajib ditulis dalam bentuk paragraf naratif utuh yang runtut, menghubungkan esensi metodologi, komparasi performa algoritma, objektivitas metrik pengujian, hingga implikasi praktis pada studi kasus.

5. **Desain Tabel Profesional dan Estetis**
   - Setiap tabel yang dibuat wajib memiliki tata letak modern dan elegan.
   - Gunakan kombinasi warna header biru tua (`#1E3A8A`), teks header putih tebal, zebra striping selang-seling abu-abu muda (`#F8FAFC`), dan garis pembatas tipis `booktabs`.
   - Lebar kolom dan spasi baris (`\arraystretch`) diatur longgar agar teks nyaman dibaca tanpa pemotongan kata yang canggung.

6. **Tipografi Bersih dan Bebas Residu Markdown**
   - Berkas `.tex` harus bersih dari kebocoran sintaks markdown seperti tanda bintang ganda (`**`), tanda bintang miring (`*`), atau angka list mentah (`1. `).
   - Gunakan sintaks resmi LaTeX seperti `\textbf{...}`, `\textit{...}`, dan blok `enumerate` yang sah jika list diperlukan pada bagian isi langkah percobaan.
   - Bahasa narasi menerapkan kaidah anti-slop yang ketat, tanpa tanda strip panjang (em dash atau en dash), kalimat bervariasi ritmenya, dan langsung fokus ke inti pembahasan tanpa basa-basi pembuka robotik.

7. **Manajemen Git dan Proteksi Aset Gambar**
   - Berkas gambar tangkapan layar (`*.png`) dimasukkan ke `.gitignore` agar tidak di-push ke repositori remote, menjaga repo tetap bersih dan ringan.
   - Hanya berkas sumber dokumen `.tex` dan naskah pendukung yang dilacak dalam kontrol versi git.

