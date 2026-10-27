# Panduan Tangkapan Layar Praktikum Pertemuan 7: Imbalanced Dataset

Panduan ini berisi pemetaan lengkap seluruh grafik visualisasi dan tangkapan layar (screenshot) asli langsung dari Google Colab untuk laporan praktikum Modul 7 (Imbalanced Dataset).

Semua berkas gambar tersimpan di:
- `P7/laporan/gambar/` (aset aktif yang dimuat dokumen LaTeX)
- `P7/laporan/gambar/ss_asli/` (arsip tangkapan layar asli Colab)

Seluruh gambar di bawah ini telah terpasang ke dalam naskah LaTeX ([PPD_P7_Januarsyah Akbar_535846.tex](file:///Users/mrfrog/Documents/Kuliah/PPD/P7/laporan/PPD_P7_Januarsyah%20Akbar_535846.tex)) dan terkompilasi ke dokumen PDF ([PPD_P7_Januarsyah Akbar_535846.pdf](file:///Users/mrfrog/Documents/Kuliah/PPD/P7/laporan/PPD_P7_Januarsyah%20Akbar_535846.pdf)).

---

## A. Daftar Tangkapan Layar Langkah Percobaan (Langkah 1 sampai 9 Modul)

Bagian ini memetakan tangkapan layar eksekusi pada langkah percobaan dataset stroke (`healthcare-dataset-stroke-data.csv`):

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Sel di Google Colab (`p7_imbalanced_dataset.ipynb`) | Nomor Gambar di PDF / Modul | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 1 | `colab-p7-setup.png` | `colab-p7-setup.png` | Code Cell 1 | Tersedia di arsip | Sel inisialisasi lingkungan Colab dan pengunduhan dataset otomatis |
| 2 | `colab-p7-imports.png` | `colab-p7-imports.png` | Code Cell 2 | Gambar 1 (Modul 7.4.1) | Impor pustaka preprocessing, scikit-learn, seaborn, dan imbalanced-learn |
| 3 | `colab-p7-read-dataset.png` | `colab-p7-read-dataset.png` | Code Cell 3 | Gambar 2 (Modul 7.4.2) | Pemuatan dataset dan tampilan 5 baris pertama (`df.head()`) |
| 4 | `colab-p7-info.png` | `colab-p7-info.png` | Code Cell 4 | Gambar 3 di Laporan | Ringkasan struktur kolom, tipe data, dan alokasi memori (`df.info()`) |
| 5 | `colab-p7-null-before.png` | `colab-p7-null-before.png` | Code Cell 5 | Gambar 4 (Modul 7.4.3) | Pengecekan missing value awal (`df.isnull().sum()`, BMI = 201 data kosong) |
| 6 | `colab-p7-pie-distribution.png` | `colab-p7-pie-distribution.png` | Code Cell 6 | Gambar 5 (Modul 7.4.4) | Diagram lingkaran distribusi variabel target stroke (95,13% vs 4,87%) |
| 7 | `colab-p7-data-cleaning.png` | `colab-p7-data-cleaning.png` | Code Cell 7 | Gambar 6 (Modul 7.4.5 & 7.4.6) | Imputasi mean pada BMI dan verifikasi ketiadaan missing value |
| 8 | `colab-p7-rus-output.png` | `colab-p7-rus-output.png` | Code Cell 9 | Gambar 7 (Modul 7.4.7) | Output frekuensi sampel sebelum vs sesudah Random Under Sampling (169 vs 169) |
| 9 | `colab-p7-melihat-data.png` | `colab-p7-melihat-data.png` | Code Cell 8 | Gambar 8 (Modul 7.4.8) | Cuplikan data kolom target stroke sebanyak 5.110 baris (`df[['stroke']]`) |
| 10 | `colab-p7-rus-bmi-hist.png` | `colab-p7-rus-bmi-hist.png` | Code Cell 10 | Gambar 9 (Modul 7.4.9) | Histogram perbandingan sebaran BMI sebelum vs sesudah Random Under Sampling |
| 11 | `colab-p7-rus-scatter.png` | `colab-p7-rus-scatter.png` | Code Cell 11 | Gambar 10 (Modul 7.4.10) | Scatter plot kadar glukosa vs BMI sebelum vs sesudah Random Under Sampling |
| 12 | `colab-p7-ros-output.png` | `colab-p7-ros-output.png` | Code Cell 12 | Gambar 11 (Modul 7.4.11) | Output frekuensi sampel sebelum vs sesudah Random Over Sampling (3.663 vs 3.663) |
| 13 | `colab-p7-ros-bmi-hist.png` | `colab-p7-ros-bmi-hist.png` | Code Cell 13 | Gambar 12 (Modul 7.4.12) | Histogram perbandingan sebaran BMI sebelum vs sesudah Random Over Sampling |
| 14 | `colab-p7-ros-scatter.png` | `colab-p7-ros-scatter.png` | Code Cell 14 | Gambar 13 (Modul 7.4.13) | Scatter plot kadar glukosa vs BMI sebelum vs sesudah Random Over Sampling |
| 15 | `colab-p7-smote-output.png` | `colab-p7-smote-output.png` | Code Cell 15 | Gambar 14 (Modul 7.4.14) | Output frekuensi sampel sebelum vs sesudah SMOTE (3.663 vs 3.663) |
| 16 | `colab-p7-smote-bmi-hist.png` | `colab-p7-smote-bmi-hist.png` | Code Cell 16 | Gambar 15 (Modul 7.4.15) | Histogram perbandingan sebaran BMI sebelum vs sesudah SMOTE |
| 17 | `colab-p7-smote-scatter.png` | `colab-p7-smote-scatter.png` | Code Cell 17 | Gambar 16 (Modul 7.4.16) | Scatter plot kadar glukosa vs BMI sebelum vs sesudah SMOTE |
| 18 | `colab-p7-adasyn-output.png` | `colab-p7-adasyn-output.png` | Code Cell 18 | Gambar 17 (Modul 7.4.17) | Output frekuensi sampel sebelum vs sesudah ADASYN (3.676 vs 3.663) |
| 19 | `colab-p7-adasyn-bmi-hist.png` | `colab-p7-adasyn-bmi-hist.png` | Code Cell 19 | Gambar 18 (Modul 7.4.18) | Histogram perbandingan sebaran BMI sebelum vs sesudah ADASYN |
| 20 | `colab-p7-adasyn-scatter.png` | `colab-p7-adasyn-scatter.png` | Code Cell 20 | Gambar 19 (Modul 7.4.19) | Scatter plot kadar glukosa vs BMI sebelum vs sesudah ADASYN |
| 21 | `colab-p7-summary-table.png` | `colab-p7-summary-table.png` | Code Cell 21 | Gambar 20 di Laporan | Tabel ringkasan jumlah sampel latih dan rasio seluruh teknik penyeimbang |
| 22 | `colab-p7-comparison-all.png` | `colab-p7-comparison-all.png` | Code Cell 21 | Gambar 21 di Laporan | Diagram batang komparatif distribusi sampel data latih seluruh teknik |

---

## B. Daftar Tangkapan Layar Tugas dan Analisis

Bagian ini memetakan tangkapan layar untuk Tugas 1 (Stroke Dataset) dan Tugas 2 (Heart Failure Records Dataset):

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Sel di Google Colab (`p7_imbalanced_dataset.ipynb`) | Nomor Gambar di PDF | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 1 | `colab-p7-t1-head.png` | `colab-p7-t1-head.png` | Code Cell 22 | Gambar 22 (Tugas 1) | Pemuatan Stroke Prediction Dataset dalam DataFrame pandas (`df_stroke.head(10)`) |
| 2 | `colab-p7-t1-features.png` | `colab-p7-t1-features.png` | Cell Markdown 23 | Tersedia di arsip | Tabel deskripsi seluruh fitur dataset stroke pada notebook |
| 3 | `colab-p7-t2-load.png` | `colab-p7-t2-load.png` | Code Cell 23 | Gambar 23 (Tugas 2) | Pemuatan dataset Heart Failure Clinical Records (`df_hf.head()`) |
| 4 | `colab-p7-t2-heart-dist.png` | `colab-p7-t2-heart-dist.png` | Code Cell 24 | Gambar 24 (Tugas 2) | Diagram lingkaran sebaran target kematian pasien (`DEATH_EVENT`) |
| 5 | `colab-p7-t2-prep-balancing.png` | `colab-p7-t2-prep-balancing.png` | Code Cell 25 & 26 | Gambar 25 (Tugas 2) | Preprocessing, standardisasi, split berstrata, dan balancing SMOTE serta ADASYN |
| 6 | `colab-p7-t2-metrics-table.png` | `colab-p7-t2-metrics-table.png` | Code Cell 27 | Gambar 26 (Tugas 2) | Tabel metrik performa (Accuracy, Precision, Recall, F1-Score) model evaluasi |
| 7 | `colab-p7-t2-metrics-bar.png` | `colab-p7-t2-metrics-bar.png` | Code Cell 27 | Gambar 27 (Tugas 2) | Diagram batang perbandingan metrik evaluasi model |
| 8 | `colab-p7-t2-confusion-matrices.png` | `colab-p7-t2-confusion-matrices.png` | Code Cell 28 | Gambar 28 (Tugas 2) | Heatmap matriks konfusi komparatif seluruh skenario model |
| 9 | `colab-p7-t2-roc-pr-curves.png` | `colab-p7-t2-roc-pr-curves.png` | Code Cell 29 | Gambar 29 (Tugas 2) | Kurva karakteristik ROC dan Precision-Recall komparatif |
