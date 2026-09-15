# Panduan Tangkapan Layar Praktikum Pertemuan 4

Panduan ini berisi daftar grafik visualisasi dan titik tangkapan layar output kode Google Colab untuk laporan praktikum Modul 4 (Regression).

---

## A. Daftar Grafik Visualisasi (Sudah Masuk ke Naskah Laporan)

Seluruh grafik berikut sudah dirender otomatis ke dalam folder `P4/laporan/gambar/`:

| No | Nama File Gambar | Letak di Laporan (PDF) | Keterangan Grafik |
|---|---|---|---|
| 1 | `hist_kc_house.png` | Bab III (Hal. 7, Gambar 1) | Histogram distribusi seluruh variabel numerik KC House Data |
| 2 | `corr_kc_house.png` | Bab III (Hal. 8, Gambar 2) | Heatmap matriks korelasi Pearson fitur terhadap harga rumah |
| 3 | `bar_comp_lr.png` | Bab III (Hal. 9, Gambar 3) | Diagram batang nilai riil vs prediksi Linear Regression (50 sampel) |
| 4 | `bar_comp_dt.png` | Bab III (Hal. 10, Gambar 4) | Diagram batang nilai riil vs prediksi Decision Tree (50 sampel) |
| 5 | `bar_comp_rf.png` | Bab III (Hal. 11, Gambar 5) | Diagram batang nilai riil vs prediksi Random Forest (50 sampel) |
| 6 | `kc_house_eval_comp.png` | Bab III (Hal. 12, Gambar 6) | Diagram batang komparasi metrik RMSE dan R2 ketiga model KC House |
| 7 | `corr_car_price.png` | Bab III (Hal. 15, Gambar 7) | Diagram batang korelasi Pearson fitur numerik terhadap harga mobil |
| 8 | `car_price_lr_comp.png` | Bab III (Hal. 17, Gambar 8 atas) | Diagram batang nilai aktual vs prediksi Linear Regression mobil |
| 9 | `car_price_dt_comp.png` | Bab III (Hal. 17, Gambar 8 tengah) | Diagram batang nilai aktual vs prediksi Decision Tree mobil |
| 10 | `car_price_rf_comp.png` | Bab III (Hal. 17, Gambar 8 bawah) | Diagram batang nilai aktual vs prediksi Random Forest mobil |
| 11 | `car_price_summary_table.png` | Bab III (Hal. 18, Gambar 9) | Visualisasi tabel performa model regresi mobil (RMSE, R2, R) |
| 12 | `car_price_metrics_comp.png` | Bab III (Hal. 18, Gambar 10) | Diagram batang komparasi metrik RMSE, R2, dan R ketiga model mobil |

---

## B. Daftar Tangkapan Layar Output Kode Colab (Sudah Masuk ke Naskah Laporan)

Seluruh tangkapan layar output kode Google Colab telah diintegrasikan ke dalam folder `P4/laporan/gambar/` dan naskah laporan LaTeX:

| No | Nama File Gambar | Sel di Google Colab (`p4_regression.ipynb`) | Letak di Laporan LaTeX / PDF | Status & Keterangan |
|---|---|---|---|---|
| 1 | `colab-eda-head.png` | Cell 5 (`df.head()` dan `df.tail()`) | Bab III, Subbab 3.2 poin 1 (Gambar 1) | Terpasang (output 5 baris awal & akhir) |
| 2 | `colab-eda-describe.png` | Cell 6 (`df.describe()`) | Bab III, Subbab 3.2 poin 2 (Gambar 2) | Terpasang (tabel statistik deskriptif KC House) |
| 3 | `colab-eda-corr.png` | Cell 8 (`df.corr(numeric_only=True)`) | Bab III, Subbab 3.2 poin 4 (Gambar 4) | Terpasang (matriks nilai angka korelasi) |
| 4 | `colab-eda-null.png` | Cell 9 (`df.isnull().sum()`) | Bab III, Subbab 3.2 poin 5 (Gambar 6) | Terpasang (verifikasi 0 missing value) |
| 5 | `colab-model-lr-output.png` | Cell 12 (Evaluasi Linear Regression) | Bab III, Subbab 3.3.1 (Gambar 7) | Terpasang (output RMSE dan R2 Linear Regression) |
| 6 | `colab-model-dt-output.png` | Cell 15 (Evaluasi Decision Tree) | Bab III, Subbab 3.3.2 (Gambar 9) | Terpasang (output RMSE dan R2 Decision Tree) |
| 7 | `colab-model-rf-output.png` | Cell 18 (Evaluasi Random Forest) | Bab III, Subbab 3.3.3 (Gambar 11) | Terpasang (output RMSE dan R2 Random Forest) |
| 8 | `colab-kc-eval-table.png` | Cell 20 (`df_kc_eval`) | Bab III, Subbab 3.4 (Gambar 13) | Terpasang (tabel evaluasi komparasi KC House) |
| 9 | `colab-car-head.png` | Cell 23 (`df_car.head()`) | Bab III, Subbab 3.5 (Gambar 15) | Terpasang (dimensi dan cuplikan data mobil) |
| 10 | `colab-car-results-table.png` | Cell 28 (`df_results_car`) | Bab III, Subbab 3.8 (Gambar 18) | Terpasang (tabel performa akhir model mobil) |
