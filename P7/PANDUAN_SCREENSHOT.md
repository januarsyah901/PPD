# Panduan Tangkapan Layar Praktikum Pertemuan 7: Imbalanced Dataset

Panduan ini berisi pemetaan lengkap seluruh grafik visualisasi dan titik tangkapan layar (screenshot) kode maupun output Google Colab untuk laporan praktikum Modul 7 (Imbalanced Dataset).

Semua berkas gambar disimpan ke dalam direktori:
`P7/laporan/gambar/`

Seluruh gambar di bawah ini telah terpasang ke dalam naskah LaTeX ([PPD_P7_Januarsyah Akbar_535846.tex](file:///Users/mrfrog/Documents/Kuliah/PPD/P7/laporan/PPD_P7_Januarsyah%20Akbar_535846.tex)) dan terkompilasi ke dokumen PDF ([PPD_P7_Januarsyah Akbar_535846.pdf](file:///Users/mrfrog/Documents/Kuliah/PPD/P7/laporan/PPD_P7_Januarsyah%20Akbar_535846.pdf)).

---

## A. Daftar Tangkapan Layar Langkah Percobaan (Langkah 1 sampai 9 Modul)

Bagian ini memetakan seluruh tangkapan layar eksekusi pada langkah percobaan dataset stroke (`healthcare-dataset-stroke-data.csv`):

| No | Nama Berkas Gambar | Sel di Google Colab (`p7_imbalanced_dataset.ipynb`) | Nomor Gambar di PDF / Modul | Status | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 1 | `colab-p7-imports.png` | Code Cell 2 | Gambar 1 (Modul 7.4.1) | **Terpasang** | Impor pustaka preprocessing, skikit-learn, seaborn, dan imbalanced-learn |
| 2 | `colab-p7-read-dataset.png` | Code Cell 3 | Gambar 2 (Modul 7.4.2) | **Terpasang** | Pemuatan dataset dan tampilan 5 baris pertama (`df.head()`) |
| 3 | `colab-p7-info.png` | Code Cell 4 | Gambar 3 di Laporan | **Terpasang** | Ringkasan struktur kolom, tipe data, dan memori (`df.info()`) |
| 4 | `colab-p7-null-before.png` | Code Cell 5 | Gambar 4 (Modul 7.4.3) | **Terpasang** | Output pengecekan nilai null awal (`df.isnull().sum()`, terlihat BMI ada 201 data kosong) |
| 5 | `colab-p7-missing-df.png` | Code Cell 5 | Gambar 5 di Laporan | **Terpasang** | Tabel persentase missing values per atribut |
| 6 | `colab-p7-pie-distribution.png` | Code Cell 6 | Gambar 6 (Modul 7.4.4) | **Terpasang** | Diagram lingkaran distribusi variabel target stroke (95,13% vs 4,87%) |
| 7 | `colab-p7-data-cleaning.png` | Code Cell 7 | Gambar 7 (Modul 7.4.5) | **Terpasang** | Potongan kode imputasi nilai rata-rata (*mean*) pada atribut BMI |
| 8 | `colab-p7-null-after.png` | Code Cell 7 | Gambar 8 (Modul 7.4.6) | **Terpasang** | Verifikasi ulang ketiadaan missing value pasca imputasi (`df.isnull().sum()`) |
| 9 | `colab-p7-rus-output.png` | Code Cell 9 | Gambar 9 (Modul 7.4.7) | **Terpasang** | Output teks frekuensi sampel sebelum vs sesudah Random Under Sampling (169 vs 169) |
| 10 | `colab-p7-melihat-data.png` | Code Cell 8 | Gambar 10 (Modul 7.4.8) | **Terpasang** | Cuplikan data kolom target stroke sebanyak 5.110 baris (`df[['stroke']]`) |
| 11 | `colab-p7-rus-bmi-hist.png` | Code Cell 10 | Gambar 11 (Modul 7.4.9) | **Terpasang** | Histogram perbandingan sebaran BMI sebelum vs sesudah Random Under Sampling |
| 12 | `colab-p7-rus-scatter.png` | Code Cell 11 | Gambar 12 (Modul 7.4.10) | **Terpasang** | Scatter plot kadar glukosa vs BMI sebelum vs sesudah Random Under Sampling |
| 13 | `colab-p7-ros-output.png` | Code Cell 12 | Gambar 13 (Modul 7.4.11) | **Terpasang** | Output teks frekuensi sampel sebelum vs sesudah Random Over Sampling (3.663 vs 3.663) |
| 14 | `colab-p7-ros-bmi-hist.png` | Code Cell 13 | Gambar 14 (Modul 7.4.12) | **Terpasang** | Histogram perbandingan sebaran BMI sebelum vs sesudah Random Over Sampling |
| 15 | `colab-p7-ros-scatter.png` | Code Cell 14 | Gambar 15 (Modul 7.4.13) | **Terpasang** | Scatter plot kadar glukosa vs BMI sebelum vs sesudah Random Over Sampling |
| 16 | `colab-p7-smote-output.png` | Code Cell 15 | Gambar 16 (Modul 7.4.14) | **Terpasang** | Output teks frekuensi sampel sebelum vs sesudah SMOTE (3.663 vs 3.663) |
| 17 | `colab-p7-smote-bmi-hist.png` | Code Cell 16 | Gambar 17 (Modul 7.4.15) | **Terpasang** | Histogram perbandingan sebaran BMI sebelum vs sesudah SMOTE |
| 18 | `colab-p7-smote-scatter.png` | Code Cell 17 | Gambar 18 (Modul 7.4.16) | **Terpasang** | Scatter plot kadar glukosa vs BMI sebelum vs sesudah SMOTE |
| 19 | `colab-p7-adasyn-output.png` | Code Cell 18 | Gambar 19 (Modul 7.4.17) | **Terpasang** | Output teks frekuensi sampel sebelum vs sesudah ADASYN (3.676 vs 3.663) |
| 20 | `colab-p7-adasyn-bmi-hist.png` | Code Cell 19 | Gambar 20 (Modul 7.4.18) | **Terpasang** | Histogram perbandingan sebaran BMI sebelum vs sesudah ADASYN |
| 21 | `colab-p7-adasyn-scatter.png` | Code Cell 20 | Gambar 21 (Modul 7.4.19) | **Terpasang** | Scatter plot kadar glukosa vs BMI sebelum vs sesudah ADASYN |
| 22 | `colab-p7-summary-table.png` | Code Cell 21 | Gambar 22 di Laporan | **Terpasang** | Tabel rekapitulasi jumlah sampel data latih dan rasio seluruh metode penyeimbang |
| 23 | `colab-p7-comparison-all.png` | Code Cell 21 | Gambar 23 di Laporan | **Terpasang** | Diagram batang komparatif sebaran kelas mayoritas dan minoritas seluruh teknik |

---

## B. Daftar Tangkapan Layar Tugas dan Analisis

Bagian ini memetakan tangkapan layar untuk Tugas 1 (Stroke Dataset) dan Tugas 2 (Heart Failure Records Dataset):

| No | Nama Berkas Gambar | Sel di Google Colab (`p7_imbalanced_dataset.ipynb`) | Nomor Gambar di PDF | Status | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 1 | `colab-p7-read-dataset.png` | Code Cell 22 | Gambar 24 (Tugas 1) | **Terpasang** | Pemuatan Stroke Prediction Dataset dalam DataFrame pandas (Tugas 1) |
| 2 | `colab-p7-t2-heart-dist.png` | Code Cell 24 | Gambar 25 (Tugas 2) | **Terpasang** | Visualisasi diagram lingkaran distribusi kelas target kematian pasien (`DEATH_EVENT`) |
| 3 | `colab-p7-t2-metrics-table.png` | Code Cell 27 | Gambar 26 (Tugas 2) | **Terpasang** | Tabel komparasi metrik performa (Accuracy, Precision, Recall, F1-Score) sebelum vs sesudah balancing |
| 4 | `colab-p7-t2-metrics-bar.png` | Code Cell 27 | Gambar 27 (Tugas 2) | **Terpasang** | Diagram batang perbandingan metrik evaluasi model Logistic Regression dan Random Forest |
| 5 | `colab-p7-t2-confusion-matrices.png` | Code Cell 28 | Gambar 28 (Tugas 2) | **Terpasang** | Heatmap matriks konfusi komparatif 6 skenario pengujian model |
| 6 | `colab-p7-t2-roc-pr-curves.png` | Code Cell 29 | Gambar 29 (Tugas 2) | **Terpasang** | Kurva karakteristik ROC dan Precision-Recall komparatif |

---

## C. Panduan Pengambilan Tangkapan Layar Mandiri di Google Colab (Opsional)

Jika bang jan ingin mengganti berkas gambar otomatis dengan hasil tangkapan layar langsung dari antarmuka Google Colab di peramban, ikuti langkah berikut:

1. **Jalankan Notebook**: Buka berkas [p7_imbalanced_dataset.ipynb](file:///Users/mrfrog/Documents/Kuliah/PPD/P7/project/p7_imbalanced_dataset.ipynb) di Google Colab, lalu jalankan semua sel (`Runtime -> Run all`).
2. **Ambil Tangkapan Layar**:
   * Di macOS, gunakan pintasan `Cmd + Shift + 4`, lalu pilih area sel kode beserta output yang diinginkan secara rapi.
   * Pastikan teks kode dan angka output terbaca jelas tanpa terpotong.
3. **Simpan Menggunakan Nama yang Sama**:
   * Simpan berkas hasil tangkapan layar langsung ke folder `P7/laporan/gambar/` menimpa nama berkas yang ada pada tabel di atas.
4. **Kompilasi Ulang Laporan**:
   * Jalankan perintah kompilasi LaTeX di terminal:
     ```bash
     cd P7/laporan
     pdflatex "PPD_P7_Januarsyah Akbar_535846.tex"
     pdflatex "PPD_P7_Januarsyah Akbar_535846.tex"
     ```
   * Berkas PDF baru akan otomatis memuat gambar tangkapan layar terbaru tanpa perlu merombak sintaks dokumen LaTeX.
