# Panduan Tangkapan Layar Praktikum Pertemuan 8: Neural Network

Panduan ini berisi pemetaan lengkap seluruh grafik visualisasi dan tangkapan layar eksekusi langsung dari Google Colab / Jupyter Notebook untuk laporan praktikum Modul 8 (Neural Network - Multilayer Perceptron).

Semua berkas gambar tersimpan di:
- `P8/laporan/gambar/` (aset aktif yang dimuat dokumen LaTeX)
- `P8/laporan/gambar/ss_asli/` (arsip tangkapan layar asli Google Colab)

Seluruh gambar di bawah ini telah terintegrasi ke dalam naskah LaTeX ([PPD_P8_Januarsyah Akbar_535846.tex](file:///Users/mrfrog/Documents/Kuliah/PPD/P8/laporan/PPD_P8_Januarsyah%20Akbar_535846.tex)) dan terkompilasi ke dokumen PDF final ([PPD_P8_Januarsyah Akbar_535846.pdf](file:///Users/mrfrog/Documents/Kuliah/PPD/P8/laporan/PPD_P8_Januarsyah%20Akbar_535846.pdf)).

---

## A. Daftar Tangkapan Layar Langkah Percobaan (Langkah 1 sampai 7 Modul)

### 1. Percobaan 1: Klasifikasi Bank Customer Churn (MLPClassifier)

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di PDF / Modul | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 1 | `colab-p8-setup.png` | `colab-p8-setup.png` | Code Cell 1 | Gambar 1 di Laporan | Inisialisasi lingkungan dan pemilihan direktori dataset pada Google Colab |
| 2 | `colab-p8-imports.png` | `colab-p8-imports.png` | Code Cell 2 | Gambar 2 (Modul 8.4.1) | Impor pustaka preprocessing, scikit-learn MLP, metrik, dan visualisasi |
| 3 | `colab-p8-churn-head.png` | `colab-p8-churn-head.png` | Code Cell 3 | Gambar 3 (Modul 8.4.2) | Tampilan 5 baris pertama Bank Customer Churn Dataset (`df.head()`) |
| 4 | `colab-p8-churn-hist.png` | Hasil visualisasi | Code Cell 4 | Gambar 4 (Modul 8.4.3) | Distribusi histogram variabel Bank Customer Churn (`df.hist()`) |
| 5 | `colab-p8-churn-pie.png` | Hasil visualisasi | Code Cell 5 | Gambar 5 (Modul 8.4.4) | Diagram lingkaran proporsi target class Attrition_Flag (83,93% vs 16,07%) |
| 6 | `colab-p8-churn-null.png` | `colab-p8-churn-null.png` | Code Cell 6 | Gambar 6 (Modul 8.4.5) | Pengecekan missing value Bank Churn (`df.isnull().sum()`, seluruh kolom 0) |
| 7 | `colab-p8-churn-cats.png` | `colab-p8-churn-cats.png` | Code Cell 7 | Gambar 7 (Modul 8.4.6) | Identifikasi atribut kategorikal (`Gender`, `Education_Level`, `Marital_Status`, dll.) |
| 8 | `colab-p8-churn-split-train.png` | `colab-p8-churn-split-train.png` | Code Cell 8 | Gambar 8 (Modul 8.4.7) | Pembagian data 70:30, Label Encoding, StandardScaler, dan pelatihan MLPClassifier |
| 9 | `colab-p8-churn-coefs.png` | `colab-p8-churn-coefs.png` | Code Cell 9 | Gambar 9 (Modul 8.4.8) | Inspeksi bobot sinaptik jaringan tiruan (`model.coefs_`, dimensi (19, 100) dan (100, 1)) |
| 10 | `colab-p8-churn-dfx.png` | `colab-p8-churn-dfx.png` | Code Cell 10 | Gambar 10 (Modul 8.4.9 & 8.4.10) | Tampilan dataframe fitur `df_X` pasca encoding dan vektor numerik `X_train[0]` |
| 11 | `colab-p8-churn-cm.png` | Hasil visualisasi | Code Cell 11 | Gambar 11 (Modul 8.4.11) | Visualisasi Confusion Matrix MLPClassifier pada data uji nasabah bank |
| 12 | `colab-p8-churn-inference-code.png` | `colab-p8-churn-inference-code.png` | Code Cell 12 | Gambar 12 (Modul 8.4.12 & 8.4.13) | Penyiapan dictionary data profil nasabah baru dan hasil interpretasi prediksi (*Existing Customer*, keyakinan 100.00%) |

### 2. Percobaan 2: Regresi Harga Rumah KC House (MLPRegressor)

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di PDF / Modul | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|---|
| 13 | `colab-p8-kc-head.png` | `colab-p8-kc-head.png` | Code Cell 13 | Gambar 13 (Modul 8.4.14) | Cuplikan 5 baris pertama dataset KC House (`df.head()`) |
| 14 | `colab-p8-kc-null.png` | `colab-p8-kc-null.png` | Code Cell 14 | Gambar 14 (Modul 8.4.15) | Pengecekan missing values KC House Data (`df.isnull().sum()`, seluruh kolom bersih) |
| 15 | `colab-p8-kc-hist.png` | Hasil visualisasi | Code Cell 15 | Gambar 15 (Modul 8.4.16) | Histogram distribusi fitur numerik properti rumah KC House |
| 16 | `colab-p8-kc-cats.png` | `colab-p8-kc-cats.png` | Code Cell 16 | Gambar 16 (Modul 8.4.17) | Pemeriksaan tipe kolom fitur non-numerik dan imputasi mean |
| 17 | `colab-p8-kc-perf.png` | `colab-p8-kc-perf.png` | Code Cell 17 | Gambar 17 (Modul 8.4.18) | Skor determinasi latih & uji serta prediksi 10 data pertama vs nilai asli |
| 18 | `colab-p8-kc-eval.png` | `colab-p8-kc-eval.png` | Code Cell 18 | Gambar 18 (Modul 8.4.19) | Output evaluasi metrik RMSE (253.743,42 USD) dan $R^2$ (0,5540) |
| 19 | `colab-p8-kc-bar-pred.png` | Hasil visualisasi | Code Cell 19 | Gambar 19 (Modul 8.4.20) | Diagram batang perbandingan harga rumah riil vs estimasi MLP (30 sampel) |

---

## B. Daftar Tangkapan Layar Tugas dan Analisis

### 1. Tugas 1: Analisis Karakteristik dan Fitur Dataset

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Nomor Gambar di Laporan | Keterangan & Isi Gambar |
|---|---|---|---|---|
| 20 | `colab-p8-t1-churn-features.png` | `colab-p8-t1-churn-features.png` | Gambar 20 di Laporan | Kamus data lengkap 21 atribut Bank Customer Churn Dataset beserta peran analitis |
| 21 | `colab-p8-t1-kc-features.png` | `colab-p8-t1-kc-features.png` | Gambar 21 di Laporan | Kamus data lengkap 21 atribut King County House Sales Dataset beserta peran domain properti |

### 2. Tugas 2: Eksperimen Model MLP pada Dataset Baru

| No | Nama Berkas Gambar | Sumber Tangkapan Layar Asli (`ss_asli/`) | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di Laporan | Keterangan & Isi Gambar |
|---|---|---|---|---|---|
| 22 | `colab-p8-t2-diab-cm.png` | Hasil visualisasi | Code Cell 22 | Gambar 22 di Laporan | Confusion Matrix MLPClassifier multi-layer (64, 32) pada diagnosa diabetes |
| 23 | `colab-p8-t2-diab-loss.png` | Hasil visualisasi | Code Cell 22 | Gambar 23 di Laporan | Kurva penurunan nilai loss fungsi objektif selama pelatihan MLP diabetes |
| 24 | `colab-p8-t2-diab-report.png` | `colab-p8-t2-diab-report.png` | Code Cell 22 | Gambar 24 di Laporan | Tabel laporan metrik klasifikasi (Presisi, Recall, F1-Score) diagnosa diabetes |
| 25 | `colab-p8-t2-ins-scatter.png` | Hasil visualisasi | Code Cell 23 | Gambar 25 di Laporan | Scatter plot korelasi nilai aktual vs estimasi tagihan asuransi kesehatan |
| 26 | `colab-p8-t2-ins-bar.png` | Hasil visualisasi | Code Cell 23 | Gambar 26 di Laporan | Diagram batang komparatif biaya asuransi aktual vs prediksi (30 pasien) |
| 27 | `colab-p8-t2-ins-loss.png` | Hasil visualisasi | Code Cell 23 | Gambar 27 di Laporan | Kurva konvergensi nilai loss pelatihan model MLPRegressor tagihan asuransi |
| 28 | `colab-p8-t2-ins-eval.png` | `colab-p8-t2-ins-eval.png` | Code Cell 23 | Gambar 28 di Laporan | Rangkuman metrik kinerja MLPRegressor ($R^2$ 0,8437, RMSE 4.787,68 USD) |
