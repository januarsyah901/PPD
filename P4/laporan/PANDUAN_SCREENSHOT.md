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

## B. Daftar Titik Output Kode Colab (Area Screenshot Teks / Tabel)

Berikut daftar sel Google Colab yang menghasilkan teks/tabel output kode (termasuk output Train R2, Test R2, dan RMSE):

| No | Nama Potensial File | Sel di Google Colab (`p4_regression.ipynb`) | Letak di Laporan LaTeX / PDF | Isi Output Kode yang Muncul |
|---|---|---|---|---|
| 1 | `colab-eda-head.png` | Cell 5 (`df.head()` dan `df.tail()`) | Bab III, Subbab 3.2 poin 1 (Hal. 6) | Tabel 5 baris pertama dan 5 baris terakhir dataset KC House |
| 2 | `colab-eda-describe.png` | Cell 6 (`df.describe()`) | Bab III, Subbab 3.2 poin 2 (Hal. 6) | Tabel ringkasan statistik deskriptif (count, mean, std, min, quartile, max) |
| 3 | `colab-eda-corr.png` | Cell 8 (`df.corr(numeric_only=True)`) | Bab III, Subbab 3.2 poin 4 (Hal. 7) | Matriks angka korelasi antarfitur numerik |
| 4 | `colab-eda-null-cats.png` | Cell 9 & 10 (`isnull().sum()` & `cats`) | Bab III, Subbab 3.2 poin 5 (Hal. 8) | Teks pengecekan missing value (total 0) dan `Index([], dtype='object')` |
| 5 | `colab-model-lr-output.png` | Cell 11 & 12 (Linear Regression) | Bab III, Subbab 3.3.1 (Hal. 9) | `Train R2: 0.6995`, `Test R2: 0.6995`, `RMSE: 208296.73`, `Intercept`, koefisien, dan 10 prediksi |
| 6 | `colab-model-dt-output.png` | Cell 14 & 15 (Decision Tree) | Bab III, Subbab 3.3.2 (Hal. 10) | `Train R2: 0.9185`, `Test R2: 0.7586`, `RMSE: 186675.48`, dan 10 perbandingan prediksi |
| 7 | `colab-model-rf-output.png` | Cell 17 & 18 (Random Forest) | Bab III, Subbab 3.3.3 (Hal. 11) | `Train R2: 0.982463...`, `Test R2: 0.853950...`, `RMSE: 145205.694...`, dan 10 perbandingan prediksi |
| 8 | `colab-kc-eval-table.png` | Cell 20 (`df_kc_eval`) | Bab III, Subbab 3.4 (Hal. 12, Tabel 2) | Output tabel perbandingan performa 3 model regresi KC House |
| 9 | `colab-car-head.png` | Cell 23 (`df_car.head()`) | Bab III, Subbab 3.5 (Hal. 13-14) | Output dimensi data (205, 26) dan tabel cuplikan data mobil |
| 10 | `colab-car-corr-text.png` | Cell 24 (`corr_car`) | Bab III, Subbab 3.6 (Hal. 15) | Teks daftar koefisien korelasi Pearson fitur numerik terhadap price |
| 11 | `colab-car-results-table.png` | Cell 28 (`df_results_car`) | Bab III, Subbab 3.8 (Hal. 18, Tabel 4) | Output tabel evaluasi akhir model mobil: Linear (3722.33 / 0.8000 / 0.8968), DT (2994.77 / 0.8706 / 0.9449), RF (1955.87 / 0.9448 / 0.9727) |
