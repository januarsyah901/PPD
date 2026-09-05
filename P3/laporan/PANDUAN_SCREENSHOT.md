# Panduan Tangkapan Layar (Screenshot) Praktikum Pertemuan 3

Panduan ini berisi daftar nama file, status tangkapan layar, dan area sel Google Colab untuk laporan praktikum modul 3 (Classification).

Semua file gambar disimpan ke dalam direktori:
`P3/laporan/gambar/`

---

## A. Daftar Tangkapan Layar yang Sudah Masuk ke Naskah

| No | Nama File | Status | Keterangan Tangkapan Layar |
|---|---|---|---|
| 1 | `colab-import-load.png` | **Sudah Terpasang** | Sel import seluruh pustaka dan pemanggilan dataset stroke serta `df.head()` |
| 2 | `colab-eda-describe.png` | **Sudah Terpasang** | Output tabel ringkasan statistik deskriptif data (`df.describe()`) |
| 3 | `colab-eda-null.png` | **Sudah Terpasang** | Output pengecekan nilai hilang (`df.isnull().sum()`, terlihat bmi bernilai 201) |
| 4 | `colab-eda-cats.png` | **Sudah Terpasang** | Output daftar kolom kategorikal bertipe object |
| 5 | `colab-prep-full.png` | **Sudah Terpasang** | Sel lengkap pra-pemrosesan data stroke (imputasi, encoding, split, scaler, dimensi data latih) |
| 6 | `colab-loan-summary-table.png` | **Sudah Terpasang** | Output tabel ringkasan performa 4 model klasifikasi pinjaman bank |
| 7 | `pie_stroke.png` | **Sudah Terpasang** | Grafik diagram lingkaran distribusi kelas target stroke |
| 8 | `hist_stroke.png` | **Sudah Terpasang** | Grafik kisi matriks histogram seluruh fitur numerik dataset stroke |
| 9 | `cm_lr.png` | **Sudah Terpasang** | Heatmap matriks konfusi Logistic Regression pada dataset stroke |
| 10 | `cm_knn.png` | **Sudah Terpasang** | Heatmap matriks konfusi KNN pada dataset stroke |
| 11 | `cm_dt.png` | **Sudah Terpasang** | Heatmap matriks konfusi Decision Tree pada dataset stroke |
| 12 | `cm_rf.png` | **Sudah Terpasang** | Heatmap matriks konfusi Random Forest pada dataset stroke |
| 13 | `cm_ada.png` | **Sudah Terpasang** | Heatmap matriks konfusi AdaBoost pada dataset stroke |
| 14 | `cm_loan_lr.png` | **Sudah Terpasang** | Heatmap matriks konfusi Logistic Regression pada dataset loan |
| 15 | `cm_loan_knn.png` | **Sudah Terpasang** | Heatmap matriks konfusi KNN pada dataset loan |
| 16 | `cm_loan_dt.png` | **Sudah Terpasang** | Heatmap matriks konfusi Decision Tree pada dataset loan |
| 17 | `cm_loan_rf.png` | **Sudah Terpasang** | Heatmap matriks konfusi Random Forest pada dataset loan |

---

## B. Tangkapan Layar Tambahan (Opsional)

Jika bang jan ingin menambahkan screenshot sel Colab yang masih tersisa, gunakan nama file berikut:

| No | Nama File | Kode / Cell Google Colab | Keterangan Tangkapan Layar |
|---|---|---|---|
| 1 | `colab-model-lr.png` | `LogisticRegression()` | Sel eksekusi pelatihan dan teks metrik Logistic Regression |
| 2 | `colab-model-lr-coef.png` | `model_lr.coef_` | Nilai koefisien dan konstanta regresi logistik |
| 3 | `colab-model-knn.png` | `KNeighborsClassifier(10)` | Sel eksekusi pelatihan dan teks metrik KNN |
| 4 | `colab-model-dt.png` | `DecisionTreeClassifier(...)` | Sel eksekusi pelatihan dan teks metrik Decision Tree |
| 5 | `colab-model-rf.png` | `RandomForestClassifier()` | Sel eksekusi pelatihan dan teks metrik Random Forest |
| 6 | `colab-model-ada.png` | `AdaBoostClassifier()` | Sel eksekusi pelatihan dan teks metrik AdaBoost |
| 7 | `colab-loan-load-clean.png` | `df_loan` missing values | Sel penanganan missing values dataset loan |
| 8 | `colab-loan-prep.png` | Encoding & standardisasi loan | Sel persiapan fitur dan split dataset loan |
| 9 | `colab-loan-modelling.png` | Loop model loan status | Sel proses perulangan evaluasi model loan |
| 10 | `colab-stroke-summary-table.png` | `df_stroke_results` | Output tabel ringkasan performa 5 model pada dataset stroke |
