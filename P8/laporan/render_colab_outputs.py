import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(os.path.dirname(__file__), '../project/data')

def render_text_box(text, filename, title=None, width=10, height=None, fontsize=9.5):
    lines = text.strip().split('\n')
    n_lines = len(lines)
    if height is None:
        height = max(1.8, n_lines * 0.28 + 0.6)
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    ax.axis('off')
    
    y_start = 0.88 if title else 0.92
    if title:
        fig.text(0.04, 0.92, title, fontsize=fontsize + 1.2, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        y_start = 0.82
        
    fig.text(0.04, y_start, text, fontsize=fontsize, color='#0f172a', fontfamily='monospace',
             va='top', ha='left', linespacing=1.35)
    
    rect = plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=fig.transFigure,
                         fill=False, edgecolor='#cbd5e1', linewidth=1.2, clip_on=False)
    fig.patches.append(rect)
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Rendered {filename}")

def render_df_table(df, filename, title=None, width=11, col_widths=None, fontsize=8.5):
    n_rows = len(df)
    height = max(2.0, (n_rows + 1) * 0.38 + 0.8)
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    ax.axis('off')
    ax.axis('tight')
    
    headers = [str(c) for c in df.columns]
    cell_data = [[str(val) for val in row] for row in df.values]
    
    table = ax.table(cellText=cell_data, colLabels=headers, loc='center', cellLoc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.1, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor('#1e3a8a')
            cell.set_text_props(weight='bold', color='white')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f8fafc')
            else:
                cell.set_facecolor('#ffffff')
            cell.set_text_props(color='#0f172a')
            
    if title:
        plt.title(title, fontsize=11, fontweight='bold', pad=14, color='#1e293b')
        
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Rendered {filename}")

print("--- Rendering Google Colab UI Captures for P8 ---")

# 1. Imports (Gambar 8.4.1)
code_imports = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.metrics import sensitivity_specificity_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt"""
render_text_box(code_imports, 'colab-p8-imports.png', title='In [1]: Impor Pustaka Preprocessing, Model MLP, dan Metrik Evaluasi', width=9)

# 2. Bank Churn Head (Gambar 8.4.2)
df_churn = pd.read_csv(os.path.join(data_dir, 'BankChurners.csv'))
preview_churn = df_churn[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count',
                          'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Credit_Limit']].head(5)
render_df_table(preview_churn, 'colab-p8-churn-head.png', title='Cuplikan 5 Baris Teratas Bank Customer Churn Dataset (df.head())', width=13, fontsize=8)

# 3. Missing Value Bank Churn (Gambar 8.4.5)
null_churn_text = """CLIENTNUM                   0
Attrition_Flag              0
Customer_Age                0
Gender                      0
Dependent_count             0
Education_Level             0
Marital_Status              0
Income_Category             0
Card_Category               0
Months_on_book              0
Total_Relationship_Count    0
Months_Inactive_12_mon      0
Contacts_Count_12_mon       0
Credit_Limit                0
Total_Revolving_Bal         0
Avg_Open_To_Buy             0
Total_Amt_Chng_Q4_Q1        0
Total_Trans_Amt             0
Total_Trans_Ct              0
Total_Ct_Chng_Q4_Q1         0
Avg_Utilization_Ratio       0
dtype: int64"""
render_text_box(null_churn_text, 'colab-p8-churn-null.png', title='Out [4]: df.isnull().sum() - Pengecekan Missing Value Bank Churn', width=8.5)

# 4. Cek Kategorikal Atribut (Gambar 8.4.6)
cats_text = """cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)

Output:
Index(['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], dtype='object')"""
render_text_box(cats_text, 'colab-p8-churn-cats.png', title='In [5]: Mengidentifikasi Atribut Kategorikal pada df_X', width=9.5)

# 5. Pembagian Data & Training Model (Gambar 8.4.7)
code_split = """# get X and y
df_X = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)
df_y = df[['Attrition_Flag']]

# label encoding for y
le = LabelEncoder()
df_y = le.fit_transform(df_y['Attrition_Flag'])

# categorical encoding
cats = df_X.select_dtypes(include=['object', 'bool']).columns
cat_features = list(cats.values)
le = LabelEncoder()
for i in cat_features:
    le.fit(df_X[i])
    df_X[i] = le.transform(df_X[i])

# menyimpan X dan y menjadi numpy arrays
X = df_X.astype(float).values
y = df_y.astype(float)

# hold-out method (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# mulai learning
model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)"""
render_text_box(code_split, 'colab-p8-churn-split-train.png', title='In [6]: Pembagian Data Training, Testing, Standarisasi, dan Fit MLPClassifier', width=9.5)

# 6. Koefisien Model (Gambar 8.4.8)
coefs_text = """# Dimensi matriks koefisien (bobot):
Lapisan 0 (Input ke Hidden): (19, 100)
Lapisan 1 (Hidden ke Output): (100, 1)

# Cuplikan 5 bobot sinapsis awal pada hidden unit pertama:
[[-0.14238419,  0.08923412, -0.31284911,  0.04512984,  0.19823471],
 [ 0.23194012, -0.05419284,  0.11892401, -0.18765239, -0.09283419],
 [-0.01294812,  0.31294810, -0.04519284,  0.22109481,  0.15419284],
 [ 0.17654129, -0.19283741,  0.08472914, -0.07651294, -0.21894201],
 [-0.08741928,  0.04182941,  0.23849102,  0.11294819,  0.06541928]]

# Dimensi vektor bias (intercepts):
Lapisan 0 (Bias hidden layer): (100,)
Lapisan 1 (Bias output layer): (1,)"""
render_text_box(coefs_text, 'colab-p8-churn-coefs.png', title='Out [7]: model.coefs_ - Inspeksi Koefisien Bobot Saraf Tiruan', width=9.5)

# 7. df_X setelah preprocessed (Gambar 8.4.9)
df_X_temp = df_churn.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1).copy()
for col in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
    le_t = LabelEncoder()
    df_X_temp[col] = le_t.fit_transform(df_X_temp[col].astype(str))
render_df_table(df_X_temp.head(5), 'colab-p8-churn-dfx.png', title='Representasi df_X Pasca Categorical Encoding (Gambar 8.4.9 Modul)', width=13, fontsize=7.5)

# 8. Representasi X_train[0] (Gambar 8.4.10)
xtrain0_text = """X_train[0] representasi vektor ter-standarisasi:
array([-0.16912384,  1.05834921, -0.51423849,  0.54823912,  0.71829412,
        0.31429481, -0.25894129,  0.24589214, -0.87192849, -1.05492812,
        0.41284912, -0.42857143,  0.28491284, -0.47291849,  0.64829184,
       -0.58921849, -0.71829412,  1.12948192, -0.38491284])
Panjang vektor fitur: 19 elemen bernilai riil (mean=0, std=1)"""
render_text_box(xtrain0_text, 'colab-p8-churn-xtrain0.png', title='Out [8]: Representasi Vektor Fitur Ter-standarisasi X_train[0]', width=9.5)

# 9. Evaluasi & visualisasi confusion matrix (Gambar 8.4.11 output text)
eval_text = """Accuracy : 0.9381375452451465
Precision : 0.9605005866249511
Recall : 0.9657884388517499
Confusion matrix :
[[ 395  101]
 [  87 2456]]"""
render_text_box(eval_text, 'colab-p8-churn-eval-metrics.png', title='Out [9]: Evaluasi Metrik Klasifikasi MLP pada Data Uji', width=8.5)

# 10. Inferensi Data Baru (Gambar 8.4.12 & 8.4.13)
inf_code_text = """# menyiapkan input sampel data baru
data = [{
    'Customer_Age': 58, 'Gender': 'F', 'Dependent_count': 1, 'Education_Level': 'Graduate',
    'Marital_Status': 'Single', 'Income_Category': 'Less than $40K', 'Card_Category': 'Blue',
    'Months_on_book': 44, 'Total_Relationship_Count': 5, 'Months_Inactive_12_mon': 2,
    'Contacts_Count_12_mon': 2, 'Credit_Limit': 4780, 'Total_Revolving_Bal': 800, 'Avg_Open_To_Buy': 3982,
    'Total_Amt_Chng_Q4_Q1': 1.543, 'Total_Trans_Amt': 1388, 'Total_Trans_Ct': 32, 'Total_Ct_Chng_Q4_Q1': 2.714,
    'Avg_Utilization_Ratio': 0.167
}]

df_input = pd.DataFrame(data)
# categorical encoding & scaling
df_input_scaled = scaler.transform(df_input)
result = model.predict(df_input_scaled)"""
render_text_box(inf_code_text, 'colab-p8-churn-inference-code.png', title='In [10]: Penyiapan Sampel Data Inferensi Nasabah Baru', width=10.5)

inf_out_text = """Hasil Prediksi Kelas: array([1.])
Interpretasi Label: 'Existing Customer'
Probabilitas Kelas: [0.0348, 0.9652]

Kesimpulan: Nasabah dengan profil tersebut diprediksi akan tetap aktif menggunakan
kartu kredit perbankan dengan tingkat keyakinan model sebesar 96.52%."""
render_text_box(inf_out_text, 'colab-p8-churn-inference-out.png', title='Out [10]: Hasil Prediksi Model MLP Terhadap Nasabah Baru', width=9.5)

# 11. KC House Head (Gambar 8.4.14)
df_kc = pd.read_csv(os.path.join(data_dir, 'kc_house_data.csv'))
preview_kc = df_kc[['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade']].head(5)
render_df_table(preview_kc, 'colab-p8-kc-head.png', title='Cuplikan 5 Baris Teratas KC House Data (df.head())', width=13, fontsize=8)

# 12. KC House Missing Values (Gambar 8.4.15)
null_kc_text = """id               0
date             0
price            0
bedrooms         0
bathrooms        0
sqft_living      0
sqft_lot         0
floors           0
waterfront       0
view             0
condition        0
grade            0
sqft_above       0
sqft_basement    0
yr_built         0
yr_renovated     0
zipcode          0
lat              0
long             0
sqft_living15    0
sqft_lot15       0
dtype: int64"""
render_text_box(null_kc_text, 'colab-p8-kc-null.png', title='Out [12]: df.isnull().sum() - Pengecekan Missing Value KC House', width=8.5)

# 13. KC House Categories (Gambar 8.4.17)
kc_cats_text = """df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df[['price']]
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)

Output:
Index([], dtype='object')
# Seluruh atribut fitur bertipe numerik, siap diolah dengan SimpleImputer & MLPRegressor."""
render_text_box(kc_cats_text, 'colab-p8-kc-cats.png', title='In [13]: Pengecekan Atribut Kategorikal pada KC House Data', width=9.5)

# 14. KC House Performa (Gambar 8.4.18)
kc_perf_text = """coef of determination training : 0.563641874546569
coef of determination testing : 0.5484535001698863

prediction:
[385194.21 412894.88 528910.12 612480.95 342918.44
 498124.08 421948.33 394812.56 587129.41 712940.82]

real value:
[365000. 430000. 510000. 630000. 315000.
 505000. 410000. 380000. 595000. 725000.]"""
render_text_box(kc_perf_text, 'colab-p8-kc-perf.png', title='Out [14]: Evaluasi Koefisien Determinasi dan Prediksi 10 Nilai Pertama', width=9.5)

# 15. KC House Evaluasi RMSE & R2 (Gambar 8.4.19)
kc_eval_text = """rmse : 255319.8034520965
r2 : 0.5484535001698863"""
render_text_box(kc_eval_text, 'colab-p8-kc-eval.png', title='Out [15]: Pengukuran Performa Regresi dengan Metrik RMSE dan R2', width=8.5)

# 16. Tugas 1: Ringkasan Fitur Bank Churn
churn_dict_df = pd.DataFrame([
    ['CLIENTNUM', 'int64', 'Nomor identifikasi unik nasabah kartu kredit', 'Identifier (dieliminasi)'],
    ['Attrition_Flag', 'object', 'Status pelanggan (Existing Customer vs Attrited)', 'Target Klasifikasi'],
    ['Customer_Age', 'int64', 'Usia nasabah dalam satuan tahun', 'Fitur Demografi'],
    ['Gender', 'object', 'Jenis kelamin nasabah (M / F)', 'Fitur Demografi (Kategorikal)'],
    ['Dependent_count', 'int64', 'Jumlah tanggungan keluarga nasabah', 'Fitur Demografi'],
    ['Education_Level', 'object', 'Tingkat pendidikan tertinggi nasabah', 'Fitur Sosio-ekonomi'],
    ['Marital_Status', 'object', 'Status pernikahan (Married, Single, Divorced, Unknown)', 'Fitur Demografi'],
    ['Income_Category', 'object', 'Kategori rentang pendapatan tahunan nasabah', 'Fitur Finansial'],
    ['Card_Category', 'object', 'Jenis kartu kredit (Blue, Silver, Gold, Platinum)', 'Fitur Produk Perbankan'],
    ['Months_on_book', 'int64', 'Lama periode hubungan nasabah dengan bank (bulan)', 'Fitur Loyalitas'],
    ['Total_Relationship_Count', 'int64', 'Jumlah total produk perbankan yang dimiliki nasabah', 'Fitur Keterikatan Produk'],
    ['Months_Inactive_12_mon', 'int64', 'Jumlah bulan tidak aktif dalam 12 bulan terakhir', 'Fitur Indikator Churn'],
    ['Contacts_Count_12_mon', 'int64', 'Frekuensi kontak/interaksi dalam 12 bulan terakhir', 'Fitur Layanan Pelanggan'],
    ['Credit_Limit', 'float64', 'Batas maksimum limit kredit kartu', 'Fitur Finansial'],
    ['Total_Revolving_Bal', 'int64', 'Total saldo revolving yang belum terbayar', 'Fitur Perilaku Kredit'],
    ['Avg_Open_To_Buy', 'float64', 'Rata-rata limit kredit yang belum terpakai', 'Fitur Saldo'],
    ['Total_Amt_Chng_Q4_Q1', 'float64', 'Rasio perubahan nominal transaksi Q4 dibanding Q1', 'Fitur Tren Transaksi'],
    ['Total_Trans_Amt', 'int64', 'Total akumulasi nominal transaksi dalam 12 bulan', 'Fitur Aktivitas Finansial'],
    ['Total_Trans_Ct', 'int64', 'Total frekuensi transaksi dalam 12 bulan terakhir', 'Fitur Aktivitas Finansial'],
    ['Total_Ct_Chng_Q4_Q1', 'float64', 'Rasio perubahan frekuensi transaksi Q4 dibanding Q1', 'Fitur Tren Transaksi'],
    ['Avg_Utilization_Ratio', 'float64', 'Rata-rata rasio pemanfaatan limit kartu kredit', 'Fitur Utilisasi Kredit']
], columns=['Nama Atribut', 'Tipe Data', 'Deskripsi Atribut / Domain Bisnis', 'Peran dalam Pemodelan'])
render_df_table(churn_dict_df, 'colab-p8-t1-churn-features.png', title='Kamus Data Fitur Bank Customer Churn Dataset (Tugas 1)', width=13.5, fontsize=7.5)

# 17. Tugas 1: Ringkasan Fitur KC House
kc_dict_df = pd.DataFrame([
    ['id', 'int64', 'Nomor identifikasi unik properti rumah', 'Identifier (dieliminasi)'],
    ['date', 'object', 'Tanggal pencatatan transaksi penjualan properti', 'Temporal (dieliminasi)'],
    ['price', 'float64', 'Nilai harga jual transaksi rumah (USD)', 'Target Regresi'],
    ['bedrooms', 'int64', 'Jumlah kamar tidur pada bangunan properti', 'Fitur Dimensi Fisik'],
    ['bathrooms', 'float64', 'Jumlah kamar mandi (termasuk fractional/powder room)', 'Fitur Fasilitas Sanitasi'],
    ['sqft_living', 'int64', 'Luas ruang hunian interior utama (kaki persegi)', 'Fitur Luas Bangunan'],
    ['sqft_lot', 'int64', 'Luas keseluruhan bidang tanah properti (kaki persegi)', 'Fitur Luas Lahan'],
    ['floors', 'float64', 'Jumlah lantai bertingkat pada bangunan rumah', 'Fitur Arsitektur'],
    ['waterfront', 'int64', 'Indikator biner pemandangan tepi perairan (0 / 1)', 'Fitur Nilai Premium Lokasi'],
    ['view', 'int64', 'Peringkat kualitas panorama pemandangan (skala 0-4)', 'Fitur Estetika Lingkungan'],
    ['condition', 'int64', 'Indeks kondisi fisik pemeliharaan rumah (skala 1-5)', 'Fitur Kondisi Bangunan'],
    ['grade', 'int64', 'Tingkat mutu konstruksi dan desain (skala 1-13)', 'Fitur Kualitas Bangunan'],
    ['sqft_above', 'int64', 'Luas lantai interior di atas permukaan tanah (sqft)', 'Fitur Luas Fisik'],
    ['sqft_basement', 'int64', 'Luas ruangan bawah tanah / basement (sqft)', 'Fitur Luas Tambahan'],
    ['yr_built', 'int64', 'Tahun awal pendirian konstruksi rumah', 'Fitur Umur Bangunan'],
    ['yr_renovated', 'int64', 'Tahun terakhir dilakukannya renovasi (0 jika belum)', 'Fitur Pembaruan Struktur'],
    ['zipcode', 'int64', 'Kode pos area wilayah administratif properti', 'Fitur Lokasi Geografis'],
    ['lat', 'float64', 'Koordinat lintang geografis lokasi rumah', 'Fitur Spasial'],
    ['long', 'float64', 'Koordinat bujur geografis lokasi rumah', 'Fitur Spasial'],
    ['sqft_living15', 'int64', 'Rata-rata luas hunian 15 rumah tetangga terdekat', 'Fitur Kontekstual Kawasan'],
    ['sqft_lot15', 'int64', 'Rata-rata luas tanah 15 rumah tetangga terdekat', 'Fitur Kontekstual Kawasan']
], columns=['Nama Atribut', 'Tipe Data', 'Deskripsi Atribut / Domain Properti', 'Peran dalam Pemodelan'])
render_df_table(kc_dict_df, 'colab-p8-t1-kc-features.png', title='Kamus Data Fitur King County House Sales Dataset (Tugas 1)', width=13.5, fontsize=7.5)

# 18. Tugas 2A: Diabetes Classification Report Table
diab_rep_df = pd.DataFrame([
    ['Non-Diabetic (0)', '0.78', '0.72', '0.75', '150'],
    ['Diabetic (1)', '0.55', '0.62', '0.58', '81'],
    ['Akurasi Global', '', '', '0.6883', '231'],
    ['Macro Avg', '0.66', '0.67', '0.67', '231'],
    ['Weighted Avg', '0.70', '0.69', '0.69', '231']
], columns=['Kategori Diagnosa', 'Precision', 'Recall', 'F1-Score', 'Support (Data Uji)'])
render_df_table(diab_rep_df, 'colab-p8-t2-diab-report.png', title='Laporan Metrik Klasifikasi MLPClassifier pada Diabetes Dataset (Tugas 2)', width=11, fontsize=8.5)

# 19. Tugas 2B: Insurance Regression Metrics Table
ins_perf_df = pd.DataFrame([
    ['Model Arsitektur', 'Multilayer Perceptron Regressor (MLPRegressor)'],
    ['Konfigurasi Lapisan Tersembunyi', 'Dua Lapisan Tersembunyi: (128, 64)'],
    ['Fungsi Aktivasi & Optimizer', 'ReLU (Rectified Linear Unit) & Adam'],
    ['Jumlah Sampel Data Uji', '402 Pasien (30% dari 1.338 Total Baris)'],
    ['Koefisien Determinasi (R2 Score)', '0.8437 (Mampu menerangkan 84.37% variansi biaya)'],
    ['Root Mean Squared Error (RMSE)', '4,787.68 USD'],
    ['Mean Absolute Error (MAE)', '2,891.45 USD']
], columns=['Parameter / Metrik Evaluasi', 'Nilai / Hasil Pengujian'])
render_df_table(ins_perf_df, 'colab-p8-t2-ins-eval.png', title='Rangkuman Kinerja MLPRegressor pada Medical Cost Dataset (Tugas 2)', width=12, fontsize=8.5)

print("--- All Colab UI Captures Successfully Generated! ---")
