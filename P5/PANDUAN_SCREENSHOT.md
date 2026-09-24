# Panduan Tangkapan Layar Praktikum Pertemuan 5

Panduan ini berisi daftar grafik visualisasi dan titik tangkapan layar (screenshot) kode maupun output Google Colab untuk laporan praktikum Modul 5 (Forecasting).

Semua file gambar disimpan ke dalam direktori:
`P5/laporan/gambar/`

Seluruh tangkapan layar di bawah ini telah terpasang ke dalam naskah LaTeX dan berhasil di-compile ke dokumen PDF (`PPD_P5_Januarsyah Akbar_535846.pdf`). Tidak ada lagi potongan kode atau output yang di-hardcode menggunakan blok `lstlisting`.

---

## A. Daftar Grafik Visualisasi

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

## B. Daftar Tangkapan Layar Output Kode Colab

Seluruh tangkapan layar output sel eksekusi Google Colab telah diintegrasikan ke dalam folder `P5/laporan/gambar/` dan naskah laporan LaTeX:

| No | Nama File Gambar | Sel di Google Colab (`p5_forecasting.ipynb`) | Letak di Laporan LaTeX / PDF | Keterangan & Isi Output |
|---|---|---|---|---|
| 1 | `colab-bike-head.png` | Code Cell 5 (`df.head()`) | Bab III, Subbab 3.1 poin 1 | Tabel 5 baris pertama dataset Bike Sharing |
| 2 | `colab-bike-datetime.png` | Code Cell 7 (Konversi format tanggal) | Bab III, Subbab 3.1 poin 2 | Verifikasi tipe data datetime dan cuplikan 10 baris nilai target `cnt` |
| 3 | `colab-bike-sliding-shape.png` | Code Cell 13 (Sliding window & split) | Bab III, Subbab 3.1 poin 3 | Dimensi data setelah sliding window `(579, 15)` dan `(145, 15)` |
| 4 | `colab-bike-sliding-sample.png` | Code Cell 15 (Inspeksi transformasi) | Bab III, Subbab 3.1 poin 3 | Cuplikan sampel nilai input `X[0]` dan target `y[0]` |
| 5 | `colab-bike-eval-output.png` | Code Cell 25 (Evaluasi model Bike Sharing) | Bab III, Subbab 3.1 poin 7 | Teks output nilai RMSE dan Pearson R untuk MLP, KNN, DT, dan RF |
| 6 | `colab-tsla-head.png` | Code Cell 28 (`df_tsla.head()`) | Bab III, Subbab 3.2 poin 1 | Dimensi dataset Tesla `(2416, 7)` dan cuplikan 5 baris pertama |
| 7 | `colab-tsla-u-eval.png` | Code Cell 34 (Evaluasi Tugas 2) | Bab III, Subbab 3.2 poin 2 | Tabel hasil evaluasi Tugas 2 Univariate (MLP RMSE 19.01, R 0.9679) |
| 8 | `colab-tsla-m-eval.png` | Code Cell 40 (Evaluasi Tugas 3) | Bab III, Subbab 3.2 poin 3 | Tabel hasil evaluasi Tugas 3 Multivariate (MLP RMSE 18.27, R 0.9708) |
| 9 | `colab-tsla-compare-table.png` | Code Cell 44 (Komparasi Tugas 2 & 3) | Bab III, Subbab 3.2 poin 4 | Tabel perbandingan langsung metrik performa Univariate vs Multivariate |

---

## C. Daftar Tangkapan Layar Potongan Kode (Menggantikan Seluruh `lstlisting`)

Seluruh blok kode program di naskah laporan telah diganti dengan tangkapan layar sel Google Colab berikut:

| No | Nama File Gambar | Letak di Naskah LaTeX / PDF | Sel di Google Colab (`p5_forecasting.ipynb`) | Deskripsi Tangkapan Layar Kode |
|---|---|---|---|---|
| 1 | `colab-code-sliding.png` | Bab III, Subbab 3.1 poin 3 | Code Cell 9 (Baris 1–15) | Fungsi jendela geser `split_sequences(sequences, n_steps_in, n_steps_out)` |
| 2 | `colab-code-stats.png` | Bab III, Subbab 3.1 poin 3 | Code Cell 11 (Baris 1–16) | Fungsi rekayasa fitur statistik `stats_features(input_data)` |
| 3 | `colab-code-models-def.png` | Bab III, Subbab 3.1 poin 5 | Code Cell 19 (Baris 1–39) | Definisi fungsi pelatihan empat algoritma model (`mlp`, `knn`, `dt`, `rf`) |
| 4 | `colab-code-bike-train.png` | Bab III, Subbab 3.1 poin 5 | Code Cell 21 (Baris 1–5) | Pemanggilan fungsi pelatihan model Bike Sharing |
| 5 | `colab-code-tsla-u-prep.png` | Bab III, Subbab 3.2 poin 2 | Code Cell 32 (Baris 1–14) | Pembentukan `seq_u`, split data kronologis 80:20, dan ekstraksi fitur Tugas 2 (Univariate) |
| 6 | `colab-code-tsla-m-prep.png` | Bab III, Subbab 3.2 poin 3 | Code Cell 38 (Baris 1–13) | Pembentukan `seq_m`, split data kronologis 80:20, dan ekstraksi fitur Tugas 3 (Multivariate) |
