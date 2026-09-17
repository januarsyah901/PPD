# Panduan Tangkapan Layar Praktikum Pertemuan 5

Panduan ini berisi daftar grafik visualisasi dan titik tangkapan layar output kode Google Colab untuk laporan praktikum Modul 5 (Forecasting).

---

## A. Daftar Grafik Visualisasi (Sudah Masuk ke Naskah Laporan)

Seluruh grafik berikut telah dirender dan tersimpan di folder `P5/laporan/gambar/`:

| No | Nama File Gambar | Letak di Laporan (PDF) | Keterangan Grafik |
|---|---|---|---|
| 1 | `bike_sharing_timeseries.png` | Bab III, Subbab 3.1 poin 4 | Plot deret waktu peminjaman sepeda harian (cnt) 2011-2012 |
| 2 | `bike_sharing_predictions.png` | Bab III, Subbab 3.1 poin 6 | Plot perbandingan nilai aktual vs prediksi model (MLP, KNN, DT, RF) pada data uji Bike Sharing |
| 3 | `bike_sharing_eval_comp.png` | Bab III, Subbab 3.1 poin 7 | Diagram batang perbandingan metrik evaluasi RMSE dan Korelasi Pearson (R) model Bike Sharing |
| 4 | `bike_sharing_summary_table.png` | Bab III, Subbab 3.1 poin 7 | Tabel visual ringkasan performa model peramalan Bike Sharing |
| 5 | `tsla_timeseries.png` | Bab III, Subbab 3.2 poin 1 | Plot tren pergerakan harga saham Tesla (Close vs Open) periode 2010-2020 |
| 6 | `tsla_u_predictions.png` | Bab III, Subbab 3.2 poin 2 | Plot hasil prediksi model Tugas 2 (Univariate: Close 7 hari ke Target Close t+2) |
| 7 | `tsla_m_predictions.png` | Bab III, Subbab 3.2 poin 3 | Plot hasil prediksi model Tugas 3 (Multivariate: Close & Open 7 hari ke Target Close t+2) |
| 8 | `tsla_eval_comp.png` | Bab III, Subbab 3.2 poin 4 | Diagram batang komparasi performa Univariate vs Multivariate untuk seluruh model |
| 9 | `tsla_extrapolation_analysis.png` | Bab III, Subbab 3.2 poin 5 | Grafik analisis fenomena ekstrapolasi (perbedaan mendasar antara MLP dan algoritma berbasis pohon pada lonjakan harga) |
| 10 | `tsla_summary_table.png` | Bab III, Subbab 3.2 poin 4 | Tabel visual ringkasan lengkap evaluasi Tugas 2 dan Tugas 3 |

---

## B. Daftar Tangkapan Layar Output Kode Colab (Sudah Masuk ke Naskah Laporan)

Seluruh tangkapan layar output sel eksekusi Google Colab telah diintegrasikan ke dalam folder `P5/laporan/gambar/` dan naskah laporan LaTeX:

| No | Nama File Gambar | Sel di Google Colab (`p5_forecasting.ipynb`) | Letak di Laporan LaTeX / PDF | Keterangan & Isi Output |
|---|---|---|---|---|
| 1 | `colab-bike-head.png` | Cell 4 (`df.head()`) | Bab III, Subbab 3.1 poin 1 | Tabel 5 baris pertama dataset Bike Sharing |
| 2 | `colab-bike-datetime.png` | Cell 5 (Konversi format tanggal) | Bab III, Subbab 3.1 poin 2 | Verifikasi tipe data datetime dan cuplikan 10 baris nilai target `cnt` |
| 3 | `colab-bike-sliding-shape.png` | Cell 8 & 9 (Sliding window & split) | Bab III, Subbab 3.1 poin 3 | Dimensi data setelah sliding window `(579, 15)` dan `(145, 15)` serta sampel `X[0]` dan `y[0]` |
| 4 | `colab-bike-eval-output.png` | Cell 14 (Evaluasi model Bike Sharing) | Bab III, Subbab 3.1 poin 7 | Teks output nilai RMSE dan Pearson R untuk MLP, KNN, DT, dan RF |
| 5 | `colab-tsla-head.png` | Cell 16 (`df_tsla.head()`) | Bab III, Subbab 3.2 poin 1 | Dimensi dataset Tesla `(2416, 7)` dan cuplikan 5 baris pertama |
| 6 | `colab-tsla-u-eval.png` | Cell 19 (Evaluasi Tugas 2) | Bab III, Subbab 3.2 poin 2 | Tabel hasil evaluasi Tugas 2 Univariate (MLP RMSE 19.01, R 0.9679) |
| 7 | `colab-tsla-m-eval.png` | Cell 22 (Evaluasi Tugas 3) | Bab III, Subbab 3.2 poin 3 | Tabel hasil evaluasi Tugas 3 Multivariate (MLP RMSE 18.27, R 0.9708) |
| 8 | `colab-tsla-compare-table.png` | Cell 24 (Komparasi Tugas 2 & 3) | Bab III, Subbab 3.2 poin 4 | Tabel perbandingan langsung metrik performa Univariate vs Multivariate |
