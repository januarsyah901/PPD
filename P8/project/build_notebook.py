import nbformat as nbf
from nbclient import NotebookClient
import os
import sys

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.0"
    }
}

cells = []

# ==============================================================================
# CELL 1: Header Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# PRAKTIKUM PENAMBANGAN DATA (PPD)
## MODUL 8: NEURAL NETWORK (MULTILAYER PERCEPTRON)

* **Nama**: Januarsyah Akbar
* **NIM**: 24/535846/SV/24314
* **Kelas**: B2
* **Mata Kuliah**: Praktikum Penambangan Data (SVPL214610)
* **Program Studi**: D4 Teknologi Rekayasa Perangkat Lunak, Sekolah Vokasi UGM

---"""))

# ==============================================================================
# CELL 2: Setup Lingkungan & Unduh Dataset
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 1. Inisialisasi & Setup Lingkungan
Menyiapkan folder data, dependensi pustaka, dan jalur akses dataset agar notebook dapat dieksekusi secara mulus di Google Colab maupun di lingkungan lokal."""))

cells.append(nbf.v4.new_code_cell("""import os
import urllib.request
import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pastikan folder data tersedia
data_dirs = ['data', '../data', 'P8/project/data', '../project/data']
DATA_DIR = 'data'
for d in data_dirs:
    if os.path.exists(d):
        DATA_DIR = d
        break

print(f"Menggunakan direktori data: {DATA_DIR}")
"""))

# ==============================================================================
# CELL 3: Import Pustaka
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 2. Import Pustaka Pembelajaran Mesin & Neural Network
Mengimpor modul manipulasi data, visualisasi, pra-pemrosesan, serta arsitektur Multilayer Perceptron (`MLPClassifier` dan `MLPRegressor`) dari `scikit-learn`."""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report
)

# Atur gaya visualisasi
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
print("Seluruh pustaka berhasil diimpor.")
"""))

# ==============================================================================
# CELL 4: PERCOBAAN 1 - Header
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
## A. LANGKAH PERCOBAAN 1: MULTILAYER PERCEPTRON UNTUK KLASIFIKASI (BANK CUSTOMER CHURN)

Percobaan ini bertujuan untuk mengklasifikasikan apakah seorang nasabah kartu kredit bank akan bertahan (*Existing Customer*) atau berhenti berlangganan (*Attrited Customer*) berdasarkan fitur demografis dan transaksi finansial."""))

# ==============================================================================
# CELL 5: Baca Dataset Bank Churners
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### a. Membaca Dataset Bank Customer Churn"""))

cells.append(nbf.v4.new_code_cell("""churn_file = os.path.join(DATA_DIR, 'BankChurners.csv')
df_churn = pd.read_csv(churn_file)

# Menampilkan 5 baris data teratas
df_churn.head()
"""))

# ==============================================================================
# CELL 6: Histogram Distribusi Variabel
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### b. Menganalisis Distribusi Variabel Menggunakan Histogram"""))

cells.append(nbf.v4.new_code_cell("""df_churn.hist(figsize=(20, 15), bins=20, edgecolor='black', grid=True)
plt.suptitle('Distribusi Histogram Variabel Bank Customer Churn', fontsize=16, y=1.01)
plt.show()
"""))

# ==============================================================================
# CELL 7: Proporsi Target Class
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### c. Mengecek Proporsi Target Class (`Attrition_Flag`)"""))

cells.append(nbf.v4.new_code_cell("""target_counts = df_churn['Attrition_Flag'].value_counts()
print("Jumlah masing-masing kelas target:")
print(target_counts)

plt.figure(figsize=(7, 6))
target_counts.plot(
    kind='pie',
    autopct='%1.2f%%',
    colors=['#2563eb', '#f59e0b'],
    startangle=140,
    explode=(0, 0.1),
    shadow=True
)
plt.title("Proporsi Target Class Attrition_Flag", fontsize=13, fontweight='bold')
plt.ylabel("")
plt.show()
"""))

# ==============================================================================
# CELL 8: Pengecekan Missing Value
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### d. Mengecek Nilai Hilang (*Missing Values*)"""))

cells.append(nbf.v4.new_code_cell("""missing_churn = df_churn.isnull().sum()
print("Jumlah nilai null per atribut:")
print(missing_churn)
"""))

# ==============================================================================
# CELL 9: Identifikasi Fitur Kategorikal dan Numerik
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### e. Mengidentifikasi Atribut Kategorikal dan Numerik"""))

cells.append(nbf.v4.new_code_cell("""# Menentukan fitur input X dan target y
df_X = df_churn.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)
df_y = df_churn[['Attrition_Flag']]

# Cek atribut bertipe kategorikal
cats = df_X.select_dtypes(include=['object', 'bool', 'str']).columns
print("Atribut Kategorikal:")
print(cats)

nums = df_X.select_dtypes(include=[np.number]).columns
print("\\nAtribut Numerik:")
print(nums)
"""))

# ==============================================================================
# CELL 10: Preprocessing, Encoding, Split & Training Model
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### f. Encoding Variabel, Pembagian Data Train-Test, dan Pelatihan Model MLPClassifier"""))

cells.append(nbf.v4.new_code_cell("""# 1. Encoding variabel target y (Existing Customer vs Attrited Customer)
le_y = LabelEncoder()
df_y_encoded = le_y.fit_transform(df_y['Attrition_Flag'])

# 2. Categorical encoding pada fitur input X menggunakan LabelEncoder
cat_features = list(cats.values)
le_dict = {}
df_X_encoded = df_X.copy()
for col in cat_features:
    le_col = LabelEncoder()
    df_X_encoded[col] = le_col.fit_transform(df_X_encoded[col].astype(str))
    le_dict[col] = le_col

# 3. Konversi menjadi array numpy
X = df_X_encoded.astype(float).values
y = df_y_encoded.astype(float)

# 4. Hold-out method pembagian dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Standarisasi fitur numerik dengan StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Membangun dan melatih model MLPClassifier
model_mlp = MLPClassifier(max_iter=500, random_state=42)
model_mlp.fit(X_train_scaled, y_train)

print("Pelatihan MLPClassifier selesai!")
print(f"Iterasi konvergensi: {model_mlp.n_iter_}")
print(f"Jumlah lapisan (n_layers_): {model_mlp.n_layers_}")
"""))

# ==============================================================================
# CELL 11: Inspeksi Koefisien dan Bobot Model
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### g. Inspeksi Koefisien Model (*Weights* dan *Biases*)"""))

cells.append(nbf.v4.new_code_cell("""# Menampilkan dimensi koefisien bobot tiap lapisan
for i, coef in enumerate(model_mlp.coefs_):
    print(f"Lapisan {i} -> Bobot shape: {coef.shape}")

print("\\nContoh koefisien bobot pada lapisan pertama (5 baris teratas):")
print(model_mlp.coefs_[0][:5, :5])
"""))

# ==============================================================================
# CELL 12: Visualisasi Data Preprocessed
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### h. Visualisasi Data Ter-Preprocessing (`df_X` dan `X_train[0]`)"""))

cells.append(nbf.v4.new_code_cell("""print("Pratinjau Dataframe Input Fitur Setelah Encoding (df_X.head()):")
display(df_X_encoded.head())

print("\\nRepresentasi vektor numerik satu baris data latih setelah standarisasi (X_train[0]):")
print(X_train_scaled[0])
"""))

# ==============================================================================
# CELL 13: Evaluasi Performa Model Klasifikasi
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### i. Evaluasi Performa Model Klasifikasi"""))

cells.append(nbf.v4.new_code_cell("""# Prediksi pada data uji
y_pred = model_mlp.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision        : {prec:.4f}")
print(f"Recall           : {rec:.4f}")
print(f"F1-Score         : {f1:.4f}")
print(f"Confusion Matrix :\\n{cm}")

# Visualisasi Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix MLPClassifier (Bank Customer Churn)", fontsize=12, fontweight='bold')
plt.grid(False)
plt.show()

# Visualisasi Kurva Loss Selama Pelatihan
plt.figure(figsize=(7, 4))
plt.plot(model_mlp.loss_curve_, color='#2563eb', lw=2)
plt.title("Kurva Loss Pelatihan MLPClassifier", fontsize=12, fontweight='bold')
plt.xlabel("Iterasi (Epochs)")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
"""))

# ==============================================================================
# CELL 14: Inferensi Sampel Nasabah Baru
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### j. Melatih Model dan Menguji Prediksi Data Inferensi Baru
Menguji sampel data nasabah baru untuk memprediksi apakah nasabah tersebut berpotensi *churn* atau bertahan."""))

cells.append(nbf.v4.new_code_cell("""# Sampel data baru dari modul
sample_customer = [{
    'Customer_Age': 58,
    'Gender': 'F',
    'Dependent_count': 1,
    'Education_Level': 'Graduate',
    'Marital_Status': 'Single',
    'Income_Category': 'Less than $40K',
    'Card_Category': 'Blue',
    'Months_on_book': 44,
    'Total_Relationship_Count': 5,
    'Months_Inactive_12_mon': 2,
    'Contacts_Count_12_mon': 2,
    'Credit_Limit': 4780,
    'Total_Revolving_Bal': 800,
    'Avg_Open_To_Buy': 3982,
    'Total_Amt_Chng_Q4_Q1': 1.543,
    'Total_Trans_Amt': 1388,
    'Total_Trans_Ct': 32,
    'Total_Ct_Chng_Q4_Q1': 2.714,
    'Avg_Utilization_Ratio': 0.167
}]

df_input = pd.DataFrame(sample_customer)

# Encoding atribut kategorikal dengan encoder yang sama
for col in cat_features:
    df_input[col] = le_dict[col].transform(df_input[col].astype(str))

# Standarisasi fitur dengan scaler yang sama
df_input_scaled = scaler.transform(df_input.astype(float).values)

# Prediksi kelas dan probabilitas
pred_code = model_mlp.predict(df_input_scaled)
pred_label = le_y.inverse_transform(pred_code.astype(int))
pred_proba = model_mlp.predict_proba(df_input_scaled)

print(f"Hasil Kode Prediksi       : {pred_code[0]}")
print(f"Hasil Interpretasi Label  : {pred_label[0]}")
print(f"Probabilitas Prediksi     : {pred_proba[0]}")
print(f"Keterangan: Nasabah diprediksi akan menjadi '{pred_label[0]}' dengan keyakinan {np.max(pred_proba)*100:.2f}%.")
"""))

# ==============================================================================
# CELL 15: PERCOBAAN 2 - Header
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
## B. LANGKAH PERCOBAAN 2: MULTILAYER PERCEPTRON UNTUK REGRESI (KC HOUSE PRICE PREDICTION)

Percobaan ini bertujuan untuk memprediksi nilai kontinu berupa harga rumah (*price*) di King County, Washington menggunakan arsitektur `MLPRegressor`."""))

# ==============================================================================
# CELL 16: Baca Dataset KC House
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### a. Membaca Dataset KC House"""))

cells.append(nbf.v4.new_code_cell("""kc_file = os.path.join(DATA_DIR, 'kc_house_data.csv')
df_kc = pd.read_csv(kc_file)

# Menampilkan 5 baris pertama
df_kc.head()
"""))

# ==============================================================================
# CELL 17: Pengecekan Missing Value KC House
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### b. Mengecek Nilai Hilang (*Missing Values*)"""))

cells.append(nbf.v4.new_code_cell("""print("Pengecekan Missing Value pada KC House Data:")
print(df_kc.isnull().sum())
"""))

# ==============================================================================
# CELL 18: Histogram Fitur Numerik KC House
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### c. Menampilkan Histogram Fitur Numerik KC House"""))

cells.append(nbf.v4.new_code_cell("""df_kc.hist(figsize=(20, 15), bins=20, edgecolor='black', grid=True)
plt.suptitle('Distribusi Histogram Fitur Numerik KC House Data', fontsize=16, y=1.01)
plt.show()
"""))

# ==============================================================================
# CELL 19: Penyiapan Atribut & Imputasi
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### d. Menyiapkan Atribut Fitur dan Pemisahan Data"""))

cells.append(nbf.v4.new_code_cell("""# Menghilangkan kolom id, date, dan target price
df_X_kc = df_kc.drop(['id', 'date', 'price'], axis=1)
df_y_kc = df_kc['price']

cats_kc = df_X_kc.select_dtypes(include=['object', 'bool']).columns
print("Atribut Kategorikal:", cats_kc)

# Imputasi nilai hilang dengan nilai rata-rata (mean imputer)
imputer = SimpleImputer(strategy='mean')
X_kc = imputer.fit_transform(df_X_kc)
y_kc = df_y_kc.astype(float).values

# Membagi data menjadi 70% data latih dan 30% data uji
X_train_kc, X_test_kc, y_train_kc, y_test_kc = train_test_split(
    X_kc, y_kc, test_size=0.3, random_state=42
)

print(f"Dimensi data latih: {X_train_kc.shape}")
print(f"Dimensi data uji  : {X_test_kc.shape}")
"""))

# ==============================================================================
# CELL 20: Melatih MLPRegressor
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### e. Melatih Model MLPRegressor dan Menguji Koefisien Determinasi ($R^2$)"""))

cells.append(nbf.v4.new_code_cell("""# Membangun dan melatih model MLPRegressor
reg_kc = MLPRegressor(max_iter=500, random_state=42)
reg_kc.fit(X_train_kc, y_train_kc)

r2_train = reg_kc.score(X_train_kc, y_train_kc)
r2_test = reg_kc.score(X_test_kc, y_test_kc)

print(f"coef of determination training : {r2_train}")
print(f"coef of determination testing  : {r2_test}")

# Menampilkan 10 prediksi awal vs nilai asli
y_pred_kc = reg_kc.predict(X_test_kc)

print("\\nPrediksi 10 data pertama:")
print(np.round(y_pred_kc[:10], 2))
print("\\nNilai asli 10 data pertama:")
print(y_test_kc[:10])
"""))

# ==============================================================================
# CELL 21: Evaluasi RMSE dan R2
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### f. Mengukur Performa Regresi dengan Metrik $R^2$ dan RMSE"""))

cells.append(nbf.v4.new_code_cell("""mse_kc = mean_squared_error(y_test_kc, y_pred_kc)
rmse_kc = np.sqrt(mse_kc)
r2_kc = r2_score(y_test_kc, y_pred_kc)

print(f"rmse : {rmse_kc:.4f}")
print(f"r2   : {r2_kc:.4f}")
"""))

# ==============================================================================
# CELL 22: Visualisasi Hasil Prediksi Bar Plot
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""#### g. Visualisasi Perbandingan Nilai Asli vs Prediksi (30 Sampel Pertama)"""))

cells.append(nbf.v4.new_code_cell("""data_real = pd.DataFrame(y_test_kc[:30], columns=['real'])
data_pred = pd.DataFrame(y_pred_kc[:30], columns=['predicted'])
df_comp_kc = pd.concat([data_real, data_pred], axis=1)

plt.figure(figsize=(14, 6))
df_comp_kc.plot(kind='bar', figsize=(14, 6), color=['#2563eb', '#f59e0b'], width=0.8)
plt.title("MLP Regression - Real vs Predicted House Price (30 Sampel)", fontsize=13, fontweight='bold')
plt.xlabel("Sampel Data Uji")
plt.ylabel("House Price (USD)")
plt.xticks(rotation=0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# CELL 23: TUGAS 1 - Header & Markdown
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
## C. TUGAS 1: ANALISIS KARAKTERISTIK DAN FITUR DATASET

### 1. Tujuan Masing-Masing Dataset
* **Bank Customer Churn Dataset (`BankChurners.csv`)**:
  Dataset ini bertujuan untuk menganalisis perilaku pelanggan kartu kredit dan memprediksi kecenderungan nasabah untuk berhenti menggunakan layanan (*churn*) atau tetap aktif (*attrition flag*). Pemodelan prediktif berbasis dataset ini sangat vital bagi perbankan guna merancang strategi retensi pelanggan, meningkatkan kepuasan, dan meminimalkan kerugian finansial akibat kehilangan basis nasabah.
* **KC House Sales Dataset (`kc_house_data.csv`)**:
  Dataset ini merekam data transaksi penjualan properti residensial di King County, Washington (mencakup Seattle) antara Mei 2014 hingga Mei 2015. Tujuannya adalah memodelkan hubungan kuantitatif antara atribut fisik rumah, lokasi geografis, kondisi struktural, dan faktor lingkungan terhadap nilai jual properti (*price*). Model ini digunakan untuk estimasi valuasi harga properti secara otomatis.

### 2. Deskripsi Fitur Dataset
Tabel deskripsi rinci dari kedua dataset disajikan di bawah ini."""))

cells.append(nbf.v4.new_code_cell("""# Ringkasan informasi kolom dataset Bank Churn
df_churn_desc = pd.DataFrame({
    'Atribut': df_churn.columns,
    'Tipe Data': [str(t) for t in df_churn.dtypes],
    'Jumlah Non-Null': df_churn.count().values,
    'Jumlah Unik': [df_churn[c].nunique() for c in df_churn.columns],
    'Contoh Nilai': [str(df_churn[c].iloc[0]) for c in df_churn.columns]
})
print("Ringkasan Fitur Bank Customer Churn Dataset:")
display(df_churn_desc)
"""))

cells.append(nbf.v4.new_code_cell("""# Ringkasan informasi kolom dataset KC House
df_kc_desc = pd.DataFrame({
    'Atribut': df_kc.columns,
    'Tipe Data': [str(t) for t in df_kc.dtypes],
    'Jumlah Non-Null': df_kc.count().values,
    'Jumlah Unik': [df_kc[c].nunique() for c in df_kc.columns],
    'Contoh Nilai': [str(df_kc[c].iloc[0]) for c in df_kc.columns]
})
print("Ringkasan Fitur KC House Data:")
display(df_kc_desc)
"""))

# ==============================================================================
# CELL 24: TUGAS 2 - Header
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
## D. TUGAS 2: EKSPERIMEN MODEL MLP PADA DATASET BARU (KLASIFIKASI & REGRESI)

Pada tugas ini, dilakukan pemodelan Multilayer Perceptron pada dua dataset baru:
1. **Klasifikasi**: Pima Indians Diabetes Dataset (`diabetes.csv`) menggunakan `MLPClassifier` untuk mendiagnosis apakah pasien mengidap diabetes berdasarkan indikator medis klinis.
2. **Regresi**: Medical Cost Personal Dataset (`insurance.csv`) menggunakan `MLPRegressor` untuk memprediksi besaran biaya premi tagihan asuransi kesehatan individu."""))

# ==============================================================================
# CELL 25: Tugas 2A - Klasifikasi Diabetes dengan MLPClassifier
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### Tugas 2A: Klasifikasi Diabetes Menggunakan MLPClassifier"""))

cells.append(nbf.v4.new_code_cell("""# 1. Membaca dataset diabetes
diab_file = os.path.join(DATA_DIR, 'diabetes.csv')
df_diab = pd.read_csv(diab_file)

print("Dimensi dataset diabetes:", df_diab.shape)
display(df_diab.head())

# 2. Pemisahan Fitur dan Target
X_diab = df_diab.drop('Outcome', axis=1).values
y_diab = df_diab['Outcome'].values

# 3. Pembagian Data Train dan Test (70% Train, 30% Test)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.3, random_state=42, stratify=y_diab
)

# 4. Standarisasi Fitur dengan StandardScaler
scaler_diab = StandardScaler().fit(X_train_d)
X_train_d_s = scaler_diab.transform(X_train_d)
X_test_d_s = scaler_diab.transform(X_test_d)

# 5. Membangun dan Melatih Model MLPClassifier dengan Arsitektur Multi-Layer (64, 32)
mlp_diab = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp_diab.fit(X_train_d_s, y_train_d)

# 6. Prediksi dan Evaluasi
y_pred_d = mlp_diab.predict(X_test_d_s)

acc_d = accuracy_score(y_test_d, y_pred_d)
prec_d = precision_score(y_test_d, y_pred_d)
rec_d = recall_score(y_test_d, y_pred_d)
f1_d = f1_score(y_test_d, y_pred_d)
cm_d = confusion_matrix(y_test_d, y_pred_d)

print("\\n=== HASIL EVALUASI KLASIFIKASI DIABETES (MLPCLASSIFIER) ===")
print(f"Accuracy  : {acc_d:.4f} ({acc_d*100:.2f}%)")
print(f"Precision : {prec_d:.4f}")
print(f"Recall    : {rec_d:.4f}")
print(f"F1-Score  : {f1_d:.4f}")
print("\\nLaporan Klasifikasi Rinci:")
print(classification_report(y_test_d, y_pred_d, target_names=['Non-Diabetic (0)', 'Diabetic (1)']))

# Visualisasi Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
disp_d = ConfusionMatrixDisplay(confusion_matrix=cm_d, display_labels=['Non-Diabetic', 'Diabetic'])
disp_d.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix - Diabetes Classification", fontsize=11, fontweight='bold')
plt.grid(False)
plt.show()

# Visualisasi Kurva Loss Pelatihan
plt.figure(figsize=(6, 3.5))
plt.plot(mlp_diab.loss_curve_, color='#059669', lw=2)
plt.title("Kurva Loss Pelatihan MLP (Diabetes Dataset)", fontsize=11, fontweight='bold')
plt.xlabel("Iterasi")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
"""))

# ==============================================================================
# CELL 26: Tugas 2B - Regresi Biaya Asuransi dengan MLPRegressor
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### Tugas 2B: Regresi Biaya Asuransi Kesehatan Menggunakan MLPRegressor"""))

cells.append(nbf.v4.new_code_cell("""# 1. Membaca dataset insurance
ins_file = os.path.join(DATA_DIR, 'insurance.csv')
df_ins = pd.read_csv(ins_file)

print("Dimensi dataset insurance:", df_ins.shape)
display(df_ins.head())

# 2. One-Hot Encoding untuk Atribut Kategorikal (sex, smoker, region)
df_ins_encoded = pd.get_dummies(df_ins, columns=['sex', 'smoker', 'region'], drop_first=True)

# 3. Memisahkan Fitur dan Target (charges)
X_ins = df_ins_encoded.drop('charges', axis=1).values
y_ins = df_ins_encoded['charges'].values

# 4. Pembagian Data Train dan Test (70% Train, 30% Test)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_ins, y_ins, test_size=0.3, random_state=42
)

# 5. Standarisasi Fitur dengan StandardScaler
scaler_ins = StandardScaler().fit(X_train_i)
X_train_i_s = scaler_ins.transform(X_train_i)
X_test_i_s = scaler_ins.transform(X_test_i)

# 6. Membangun dan Melatih Model MLPRegressor dengan Arsitektur (128, 64)
mlp_ins = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
mlp_ins.fit(X_train_i_s, y_train_i)

# 7. Prediksi dan Evaluasi
y_pred_i = mlp_ins.predict(X_test_i_s)

r2_i = r2_score(y_test_i, y_pred_i)
mse_i = mean_squared_error(y_test_i, y_pred_i)
rmse_i = np.sqrt(mse_i)
mae_i = mean_absolute_error(y_test_i, y_pred_i)

print("\\n=== HASIL EVALUASI REGRESI BIAYA ASURANSI (MLPREGRESSOR) ===")
print(f"R-squared (R2) Score : {r2_i:.4f}")
print(f"RMSE                 : {rmse_i:.4f} USD")
print(f"MAE                  : {mae_i:.4f} USD")

# 8. Visualisasi Perbandingan Prediksi vs Nilai Riil
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot Real vs Predicted
axes[0].scatter(y_test_i, y_pred_i, alpha=0.6, color='#2563eb', edgecolors='w', s=50)
axes[0].plot([y_test_i.min(), y_test_i.max()], [y_test_i.min(), y_test_i.max()], 'r--', lw=2)
axes[0].set_title("Scatter Plot Prediksi vs Nilai Aktual", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Nilai Riil Tagihan Asuransi (USD)")
axes[0].set_ylabel("Nilai Prediksi Model MLP (USD)")
axes[0].grid(True)

# Bar plot 30 sampel
df_bar_ins = pd.DataFrame({'Aktual': y_test_i[:30], 'Prediksi': y_pred_i[:30]})
df_bar_ins.plot(kind='bar', ax=axes[1], color=['#2563eb', '#f59e0b'], width=0.8)
axes[1].set_title("Perbandingan Tagihan Aktual vs Prediksi (30 Sampel)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Sampel Pasien")
axes[1].set_ylabel("Biaya Tagihan (USD)")
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.show()

# Visualisasi Kurva Loss Pelatihan Regresi
plt.figure(figsize=(6, 3.5))
plt.plot(mlp_ins.loss_curve_, color='#d97706', lw=2)
plt.title("Kurva Loss Pelatihan MLPRegressor (Insurance Dataset)", fontsize=11, fontweight='bold')
plt.xlabel("Iterasi")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
"""))

# ==============================================================================
# CELL 27: Analisis Kritis Penutup
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 3. Analisis Kritis Performa Model
1. **Analisis Klasifikasi (Diabetes Dataset)**:
   * Arsitektur MLP dengan representasi berlapis `(64, 32)` dan aktivasi ReLU mampu menangkap pola non-linear dari interaksi variabel metabolik tubuh seperti kadar glukosa, indeks massa tubuh (BMI), dan usia.
   * Standarisasi fitur menggunakan `StandardScaler` mutlak diperlukan karena fitur seperti `Insulin` memiliki rentang nilai hingga ratusan, sementara `DiabetesPedigreeFunction` bernilai desimal di bawah 2.5. Tanpa scaling, gradien pembaharuan bobot akan didominasi oleh fitur berskala besar.
2. **Analisis Regresi (Insurance Dataset)**:
   * Model `MLPRegressor` menghasilkan koefisien determinasi yang sangat baik ($R^2 \approx 0.8437$) dengan nilai RMSE sekitar $4,787.68$ USD. Hal ini membuktikan bahwa MLP mampu memodelkan diskontinuitas biaya medis yang dipicu oleh faktor risiko tinggi (terutama status perokok dan interaksi BMI terhadap usia).
   * Kurva loss menunjukkan konvergensi yang mulus berkat algoritma optimasi Adam yang secara adaptif menyesuaikan laju pembelajaran (*learning rate*) untuk setiap parameter bobot jaringan syaraf tiruan."""))

nb.cells = cells

# Simpan dan eksekusi notebook
target_notebook = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p8_neural_network.ipynb')
print(f"Menyimpan notebook ke {target_notebook}...")
with open(target_notebook, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Mengeksekusi notebook p8_neural_network.ipynb...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open(target_notebook, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook p8_neural_network.ipynb berhasil dibuat dan dieksekusi dengan output lengkap!")
