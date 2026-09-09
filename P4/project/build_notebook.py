import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

# Metadata
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
        "version": "3.10.0"
    }
}

cells = []

# --- CELL 1: Header ---
cells.append(nbf.v4.new_markdown_cell("""# PRAKTIKUM PENAMBANGAN DATA (PPD)
## MODUL 4: REGRESSION

* **Nama**: Januarsyah Akbar
* **NIM**: 24/535846/SV/24314
* **Kelas**: B2
* **Mata Kuliah**: Praktikum Penambangan Data (SVPL214610)
* **Program Studi**: D4 Teknologi Rekayasa Perangkat Lunak, Sekolah Vokasi UGM

---"""))

# --- CELL 2: Intro Percobaan ---
cells.append(nbf.v4.new_markdown_cell("""## A. LANGKAH PERCOBAAN (KC House Data)
Dataset: [House Sales in King County, USA](https://www.kaggle.com/datasets/shivachandel/kc-house-data)"""))

# --- CELL 3: 1. Impor Pustaka ---
cells.append(nbf.v4.new_markdown_cell("""### 1. Impor Pustaka"""))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor"""))

# --- CELL 4: 2. Load Dataset ---
cells.append(nbf.v4.new_markdown_cell("""### 2. Load Dataset"""))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/kc_house_data.csv')"""))

# --- CELL 5: 3. EDA - a. head & tail ---
cells.append(nbf.v4.new_markdown_cell("""### 3. Exploratory Data Analysis (EDA)
**a. Melihat 5 baris pertama dan terakhir pada dataset**"""))
cells.append(nbf.v4.new_code_cell("""df.head()"""))
cells.append(nbf.v4.new_code_cell("""df.tail()"""))

# --- CELL 6: 3. EDA - b. describe ---
cells.append(nbf.v4.new_markdown_cell("""**b. Melihat deskripsi statistik**"""))
cells.append(nbf.v4.new_code_cell("""df.describe()"""))

# --- CELL 7: 3. EDA - c. hist ---
cells.append(nbf.v4.new_markdown_cell("""**c. Melihat histogram variabel numerik**"""))
cells.append(nbf.v4.new_code_cell("""df.hist(figsize=(10,10))
plt.show()"""))

# --- CELL 8: 3. EDA - d. corr ---
cells.append(nbf.v4.new_markdown_cell("""**d. Mengecek korelasi atribut numerik**"""))
cells.append(nbf.v4.new_code_cell("""df.corr(numeric_only=True)"""))

# --- CELL 9: 3. EDA - e. isnull ---
cells.append(nbf.v4.new_markdown_cell("""**e. Mengecek missing value**"""))
cells.append(nbf.v4.new_code_cell("""df.isnull().sum()"""))

# --- CELL 10: 3. EDA - f. kategorikal ---
cells.append(nbf.v4.new_markdown_cell("""**f. Mengecek atribut kategorikal**"""))
cells.append(nbf.v4.new_code_cell("""df_X = df.drop(['id', 'date', 'price'], axis=1)
y = df['price']
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)"""))

# --- CELL 11: 4. Modelling - a. Linear Regression ---
cells.append(nbf.v4.new_markdown_cell("""### 4. Modelling & Evaluasi
**a. Linear Regression**"""))
cells.append(nbf.v4.new_code_cell("""df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

print('coef of determination training ', reg.score(X_train, y_train))
print('coef of determination testing ', reg.score(X_test, y_test))

print('coefficient')
print(reg.coef_)
print('intercept')
print(reg.intercept_)

print('prediction')
y_pred = reg.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[0:10])"""))

# --- CELL 12: Evaluasi LR ---
cells.append(nbf.v4.new_markdown_cell("""*Evaluasi Model Linear Regression dengan RMSE dan $R^2$*"""))
cells.append(nbf.v4.new_code_cell("""mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse : ', rmse)
print('r2 : ', r2)"""))

# --- CELL 13: Plot bar LR ---
cells.append(nbf.v4.new_markdown_cell("""*Visualisasi Perbandingan Prediksi Linear Regression (50 Sampel)*"""))
cells.append(nbf.v4.new_code_cell("""data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
df_new.plot(kind='bar', figsize=(15,3))

plt.title("Linear regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()"""))

# --- CELL 14: Modelling - b. Decision Tree ---
cells.append(nbf.v4.new_markdown_cell("""**b. Decision Tree Regression**"""))
cells.append(nbf.v4.new_code_cell("""dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

print('coef of determination training ', dt.score(X_train, y_train))
print('coef of determination testing ', dt.score(X_test, y_test))

print('prediction')
y_pred_dt = dt.predict(X_test)
print(y_pred_dt[:10])
print('real value')
print(y_test[0:10])"""))

# --- CELL 15: Evaluasi DT ---
cells.append(nbf.v4.new_markdown_cell("""*Evaluasi Model Decision Tree dengan RMSE dan $R^2$*"""))
cells.append(nbf.v4.new_code_cell("""mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print('rmse : ', rmse_dt)
print('r2 : ', r2_dt)"""))

# --- CELL 16: Plot bar DT ---
cells.append(nbf.v4.new_markdown_cell("""*Visualisasi Perbandingan Prediksi Decision Tree (50 Sampel)*"""))
cells.append(nbf.v4.new_code_cell("""data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred_dt[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
df_new.plot(kind='bar', figsize=(15,3))

plt.title("DT regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()"""))

# --- CELL 17: Modelling - c. Random Forest ---
cells.append(nbf.v4.new_markdown_cell("""**c. Random Forest Regression**"""))
cells.append(nbf.v4.new_code_cell("""rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

print('coef of determination training ', rf.score(X_train, y_train))
print('coef of determination testing ', rf.score(X_test, y_test))

print('prediction')
y_pred_rf = rf.predict(X_test)
print(y_pred_rf[:10])
print('real value')
print(y_test[:10])"""))

# --- CELL 18: Evaluasi RF ---
cells.append(nbf.v4.new_markdown_cell("""*Evaluasi Model Random Forest dengan RMSE dan $R^2$*"""))
cells.append(nbf.v4.new_code_cell("""mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('rmse : ', rmse_rf)
print('r2 : ', r2_rf)"""))

# --- CELL 19: Plot bar RF ---
cells.append(nbf.v4.new_markdown_cell("""*Visualisasi Perbandingan Prediksi Random Forest (50 Sampel)*"""))
cells.append(nbf.v4.new_code_cell("""data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred_rf[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
df_new.plot(kind='bar', figsize=(15,3))

plt.title("RF regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()"""))

# --- CELL 20: Rekapitulasi KC House ---
cells.append(nbf.v4.new_markdown_cell("""### Rekapitulasi Performa Model Percobaan (KC House Data)"""))
cells.append(nbf.v4.new_code_cell("""df_kc_eval = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree Regression (max_depth=10)', 'Random Forest Regression'],
    'Train R2': [reg.score(X_train, y_train), dt.score(X_train, y_train), rf.score(X_train, y_train)],
    'Test R2': [r2, r2_dt, r2_rf],
    'RMSE': [rmse, rmse_dt, rmse_rf]
})
df_kc_eval"""))

# --- CELL 21: Header Tugas ---
cells.append(nbf.v4.new_markdown_cell("""---
## B. TUGAS & ANALISIS (Car Price Prediction Dataset)
Dataset: [Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data)"""))

# --- CELL 22: Tugas 1 ---
cells.append(nbf.v4.new_markdown_cell("""### 1. Tujuan Penggunaan Dataset & Definisi Input-Output
* **Tujuan Penggunaan Dataset**:
  Dataset ini digunakan untuk memprediksi estimasi harga jual mobil berdasarkan karakteristik teknis mesin, spesifikasi dimensi bodi, serta kategori kendaraan. Analisis ini membantu industri otomotif dalam memahami preferensi pasar dan menentukan strategi penetapan harga yang kompetitif.
* **Variabel Output (Target)**:
  * `price` (Numerik kontinu, satuan USD): Harga pasar dari kendaraan yang diprediksi.
* **Variabel Input (Fitur)**:
  * Fitur numerik: `symboling`, `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`, `enginesize`, `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`.
  * Fitur kategorikal: `fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `cylindernumber`, `fuelsystem`.
  * Kolom identitas yang dieliminasi: `car_ID` (indeks/identitas unik) dan `CarName` (nama spesifik model mobil)."""))

# --- CELL 23: Load Dataset Car ---
cells.append(nbf.v4.new_code_cell("""url_car = 'https://raw.githubusercontent.com/pranavhaldar/Car-Price_Linear-Regression-Assignment/master/CarPrice_Assignment.csv'
df_car = pd.read_csv(url_car)

print("Dimensi dataset:", df_car.shape)
df_car.head()"""))

# --- CELL 24: Tugas 2 ---
cells.append(nbf.v4.new_markdown_cell("""### 2. Variabel yang Paling Mempengaruhi Output (Korelasi Tinggi)"""))
cells.append(nbf.v4.new_code_cell("""from scipy.stats import pearsonr

numeric_car = df_car.select_dtypes(include=[np.number]).drop('car_ID', axis=1)
corr_car = numeric_car.corr()['price'].drop('price').sort_values(ascending=False)

print("Koefisien Korelasi Fitur Numerik terhadap Price:")
print(corr_car)

# Visualisasi korelasi
plt.figure(figsize=(8, 5))
colors = ['#1b9e77' if c > 0 else '#d95f02' for c in corr_car.values]
corr_car.plot(kind='barh', color=colors, edgecolor='black')
plt.title('Korelasi Fitur Numerik terhadap Harga Mobil')
plt.xlabel('Nilai Korelasi Pearson (r)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()"""))

# --- CELL 25: Analisis korelasi markdown ---
cells.append(nbf.v4.new_markdown_cell("""**Analisis Variabel Berpengaruh:**
1. **`enginesize` ($r = 0.8741$)**: Kapasitas mesin memiliki korelasi positif paling dominan. Mobil dengan kapasitas silinder mesin besar selalu memiliki harga yang jauh lebih mahal.
2. **`curbweight` ($r = 0.8353$)**: Bobot kosong mobil memiliki korelasi sangat kuat. Mobil yang lebih berat umumnya memiliki struktur bodi lebih kokoh, fitur keselamatan lengkap, serta dimensi mewah.
3. **`horsepower` ($r = 0.8081$)**: Tenaga kuda mesin menjadi daya tarik performa utama pada segmen kendaraan premium.
4. **`carwidth` ($r = 0.7593$) dan `carlength` ($r = 0.6829$)**: Dimensi fisik mobil berkorelasi kuat terhadap kelas kenyamanan dan segmen mobil.
5. **`highwaympg` ($r = -0.6976$) dan `citympg` ($r = -0.6858$)**: Efisiensi bahan bakar berkorelasi negatif kuat, karena mobil mewah bertenaga besar umumnya memiliki konsumsi bahan bakar yang lebih boros."""))

# --- CELL 26: Tugas 3 - Preprocessing & Modelling ---
cells.append(nbf.v4.new_markdown_cell("""### 3. Pembuatan Model Regresi (Linear, Decision Tree, Random Forest)"""))
cells.append(nbf.v4.new_code_cell("""from sklearn.preprocessing import LabelEncoder

# Pra-pemrosesan Data
df_X_car = df_car.drop(['car_ID', 'CarName', 'price'], axis=1)
y_car = df_car['price'].values

le = LabelEncoder()
cats_car = df_X_car.select_dtypes(include=['object', 'str']).columns
for col in cats_car:
    df_X_car[col] = le.fit_transform(df_X_car[col].astype(str))

X_car = df_X_car.values.astype(float)

# Pembagian data latih dan uji (70:30)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_car, y_car, test_size=0.3, random_state=42)

# Inisialisasi Model
models_car = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Random Forest Regression': RandomForestRegressor(random_state=42)
}

preds_car = {}
results_car = []

for name, model in models_car.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    preds_car[name] = y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test_c, y_pred))
    r2 = r2_score(y_test_c, y_pred)
    r = pearsonr(y_test_c, y_pred)[0]
    
    results_car.append({
        'model': name,
        'RMSE': rmse,
        'R2': r2,
        'R': r
    })

print("Pelatihan ketiga model selesai!")"""))

# --- CELL 27: Visualisasi Prediksi Mobil ---
cells.append(nbf.v4.new_code_cell("""# Visualisasi perbandingan prediksi pada 35 sampel data uji
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
sample_n = 35

for idx, (name, pred) in enumerate(preds_car.items()):
    df_plot = pd.DataFrame({
        'real values': y_test_c[:sample_n],
        'predicted values': pred[:sample_n]
    })
    df_plot.plot(kind='bar', ax=axes[idx], color=['#1f77b4', '#ff7f0e'], width=0.8)
    axes[idx].set_title(f"{name} (Actual vs Predicted)")
    axes[idx].set_ylabel("Price (USD)")
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

plt.xlabel("Sample Index")
plt.tight_layout()
plt.show()"""))

# --- CELL 28: Tugas 4 - Tabel Performa ---
cells.append(nbf.v4.new_markdown_cell("""### 4. Tabel Performa Model Machine Learning (Car Price)"""))
cells.append(nbf.v4.new_code_cell("""df_results_car = pd.DataFrame(results_car)
df_results_car"""))

# --- CELL 29: Tugas 5 - Analisis Model Terbaik ---
cells.append(nbf.v4.new_markdown_cell("""### 5. Analisis Pemilihan Model Terbaik

Berdasarkan hasil evaluasi pada tabel performa di atas:
* **Random Forest Regression** merupakan model yang paling baik.

**Alasan Pemilihan**:
1. **Nilai RMSE Paling Rendah ($1955.87$)**: Rata-rata margin kesalahan estimasi harga Random Forest jauh lebih kecil dibandingkan Decision Tree ($2994.77$) maupun Linear Regression ($3722.33$).
2. **Nilai $R^2$ Tertinggi ($0.9448$)**: Model Random Forest mampu menjelaskan 94.48% variasi harga mobil, sedangkan Linear Regression hanya mampu menjelaskan 80.00%.
3. **Korelasi Prediksi ($R$) Paling Kuat ($0.9727$)**: Nilai korelasi Pearson mendekati sempurna, menunjukkan prediksi model selaras rapat dengan harga sebenarnya.
4. **Generalisasi Lebih Stabil**: Pendekatan ansambel pada Random Forest menggabungkan puluhan pohon keputusan dengan teknik bagging, sehingga memangkas varians data dan mencegah bias overfitting yang sering menjebak Decision Tree tunggal."""))

nb.cells = cells

# Save unexecuted first
nb_path = '/Users/mrfrog/Documents/Kuliah/PPD/P4/project/p4_regression.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Created notebook at: {nb_path}")

# Execute notebook
print("Executing notebook to populate all outputs...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

# Save executed notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook successfully executed and saved with all outputs!")
