# Panduan Tangkapan Layar Praktikum Pertemuan 8: Neural Network

Panduan ini berisi pemetaan lengkap seluruh grafik visualisasi dan tangkapan layar eksekusi langsung dari Google Colab / Jupyter Notebook untuk laporan praktikum Modul 8 (Neural Network - Multilayer Perceptron).

Semua berkas gambar tersimpan di:
- `P8/laporan/gambar/` (aset aktif yang dimuat dokumen LaTeX)

Seluruh gambar di bawah ini telah terintegrasi ke dalam naskah LaTeX ([PPD_P8_Januarsyah Akbar_535846.tex](file:///Users/mrfrog/Documents/Kuliah/PPD/P8/laporan/PPD_P8_Januarsyah%20Akbar_535846.tex)) dan terkompilasi ke dokumen PDF final ([PPD_P8_Januarsyah Akbar_535846.pdf](file:///Users/mrfrog/Documents/Kuliah/PPD/P8/laporan/PPD_P8_Januarsyah%20Akbar_535846.pdf)).

---

## A. Daftar Tangkapan Layar Langkah Percobaan (Langkah 1 sampai 7 Modul)

### 1. Percobaan 1: Klasifikasi Bank Customer Churn (MLPClassifier)

| No | Nama Berkas Gambar | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di PDF / Modul | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|
| 1 | `colab-p8-imports.png` | Code Cell 2 | Gambar 1 (Modul 8.4.1) | Impor pustaka preprocessing, scikit-learn MLP, metrik, dan visualisasi |
| 2 | `colab-p8-churn-head.png` | Code Cell 3 | Gambar 2 (Modul 8.4.2) | Tampilan 5 baris pertama Bank Customer Churn Dataset (`df.head()`) |
| 3 | `colab-p8-churn-hist.png` | Code Cell 4 | Gambar 3 (Modul 8.4.3) | Distribusi histogram variabel Bank Customer Churn (`df.hist()`) |
| 4 | `colab-p8-churn-pie.png` | Code Cell 5 | Gambar 4 (Modul 8.4.4) | Diagram lingkaran proporsi target class Attrition_Flag (83,93% vs 16,07%) |
| 5 | `colab-p8-churn-null.png` | Code Cell 6 | Gambar 5 (Modul 8.4.5) | Pengecekan missing value Bank Churn (`df.isnull().sum()`, seluruh kolom 0) |
| 6 | `colab-p8-churn-cats.png` | Code Cell 7 | Gambar 6 (Modul 8.4.6) | Identifikasi atribut kategorikal (`Gender`, `Education_Level`, `Marital_Status`, dll.) |
| 7 | `colab-p8-churn-split-train.png` | Code Cell 8 | Gambar 7 (Modul 8.4.7) | Pembagian data 70:30, Label Encoding, StandardScaler, dan pelatihan MLPClassifier |
| 8 | `colab-p8-churn-coefs.png` | Code Cell 9 | Gambar 8 (Modul 8.4.8) | Inspeksi bobot sinaptik jaringan tiruan (`model.coefs_`, dimensi (19, 100) dan (100, 1)) |
| 9 | `colab-p8-churn-dfx.png` | Code Cell 10 | Gambar 9 (Modul 8.4.9) | Tampilan dataframe fitur `df_X` setelah encoding numerik |
| 10 | `colab-p8-churn-xtrain0.png` | Code Cell 10 | Gambar 10 (Modul 8.4.10) | Representasi vektor numerik satu baris data latih ter-standarisasi (`X_train[0]`) |
| 11 | `colab-p8-churn-eval-metrics.png` | Code Cell 11 | Gambar 11 di Laporan | Output numerik metrik evaluasi (Akurasi 93,81%, Presisi 96,05%, Recall 96,58%) |
| 12 | `colab-p8-churn-cm.png` | Code Cell 11 | Gambar 12 (Modul 8.4.11) | Visualisasi Confusion Matrix MLPClassifier pada data uji nasabah bank |
| 13 | `colab-p8-churn-inference-code.png` | Code Cell 12 | Gambar 13 (Modul 8.4.12) | Penyiapan dictionary data profil nasabah baru dan alur standarisasi |
| 14 | `colab-p8-churn-inference-out.png` | Code Cell 12 | Gambar 14 (Modul 8.4.13) | Hasil inferensi model terhadap nasabah baru (prediksi: *Existing Customer*, 96,52%) |

### 2. Percobaan 2: Regresi Harga Rumah KC House (MLPRegressor)

| No | Nama Berkas Gambar | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di PDF / Modul | Keterangan & Isi Tangkapan Layar |
|---|---|---|---|---|
| 15 | `colab-p8-kc-head.png` | Code Cell 13 | Gambar 15 (Modul 8.4.14) | Cuplikan 5 baris pertama dataset KC House (`df.head()`) |
| 16 | `colab-p8-kc-null.png` | Code Cell 14 | Gambar 16 (Modul 8.4.15) | Pengecekan missing values KC House Data (`df.isnull().sum()`, seluruh kolom bersih) |
| 17 | `colab-p8-kc-hist.png` | Code Cell 15 | Gambar 17 (Modul 8.4.16) | Histogram distribusi fitur numerik properti rumah KC House |
| 18 | `colab-p8-kc-cats.png` | Code Cell 16 | Gambar 18 (Modul 8.4.17) | Pemeriksaan tipe kolom fitur non-numerik dan imputasi mean |
| 19 | `colab-p8-kc-perf.png` | Code Cell 17 | Gambar 19 (Modul 8.4.18) | Skor determinasi latih & uji serta prediksi 10 data pertama vs nilai asli |
| 20 | `colab-p8-kc-eval.png` | Code Cell 18 | Gambar 20 (Modul 8.4.19) | Output evaluasi metrik RMSE (255.319,80 USD) dan $R^2$ (0,5485) |
| 21 | `colab-p8-kc-bar-pred.png` | Code Cell 19 | Gambar 21 (Modul 8.4.20) | Diagram batang perbandingan harga rumah riil vs estimasi MLP (30 sampel) |

---

## B. Daftar Tangkapan Layar Tugas dan Analisis

### 1. Tugas 1: Analisis Karakteristik dan Fitur Dataset

| No | Nama Berkas Gambar | Sumber | Nomor Gambar di Laporan | Keterangan & Isi Gambar |
|---|---|---|---|---|
| 22 | `colab-p8-t1-churn-features.png` | Tabel Ringkasan Fitur | Gambar 22 di Laporan | Kamus data lengkap 21 atribut Bank Customer Churn Dataset beserta peran analitis |
| 23 | `colab-p8-t1-kc-features.png` | Tabel Ringkasan Fitur | Gambar 23 di Laporan | Kamus data lengkap 21 atribut King County House Sales Dataset beserta peran domain properti |

### 2. Tugas 2: Eksperimen Model MLP pada Dataset Baru

| No | Nama Berkas Gambar | Sel di Notebook (`p8_neural_network.ipynb`) | Nomor Gambar di Laporan | Keterangan & Isi Gambar |
|---|---|---|---|---|
| 24 | `colab-p8-t2-diab-cm.png` | Code Cell 22 | Gambar 24 di Laporan | Confusion Matrix MLPClassifier multi-layer (64, 32) pada diagnosa diabetes |
| 25 | `colab-p8-t2-diab-loss.png` | Code Cell 22 | Gambar 25 di Laporan | Kurva penurunan nilai loss fungsi objektif selama pelatihan MLP diabetes |
| 26 | `colab-p8-t2-diab-report.png` | Code Cell 22 | Gambar 26 di Laporan | Tabel laporan metrik klasifikasi (Presisi, Recall, F1-Score) diagnosa diabetes |
| 27 | `colab-p8-t2-ins-scatter.png` | Code Cell 23 | Gambar 27 di Laporan | Scatter plot korelasi nilai aktual vs estimasi tagihan asuransi kesehatan |
| 28 | `colab-p8-t2-ins-bar.png` | Code Cell 23 | Gambar 28 di Laporan | Diagram batang komparatif biaya asuransi aktual vs prediksi (30 pasien) |
| 29 | `colab-p8-t2-ins-loss.png` | Code Cell 23 | Gambar 29 di Laporan | Kurva konvergensi nilai loss pelatihan model MLPRegressor tagihan asuransi |
| 30 | `colab-p8-t2-ins-eval.png` | Code Cell 23 | Gambar 30 di Laporan | Rangkuman metrik kinerja MLPRegressor ($R^2$ 0,8437, RMSE 4.787,68 USD) |
