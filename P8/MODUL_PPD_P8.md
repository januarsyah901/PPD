# PERTEMUAN VIII

## NEURAL NETWORK

### 8.1 TUJUAN PEMBELAJARAN

A. Mahasiswa memahami konsep Multilayer Perceptron (MLP) dan mampu menerapkannya pada teknik klasifikasi dan regresi.  
B. Mahasiswa mampu menganalisis hasil klasifikasi menggunakan MLP dengan memanfaatkan metrik evaluasi dan pengujian model.  
C. Mahasiswa mampu menganalisis hasil regresi menggunakan MLP dengan membandingkan hasil prediksi dengan data asli.  

---

### 8.2 DASAR TEORI

#### A. Neural Network

Neural network atau jaringan syaraf tiruan adalah jaringan atau jaring laba-laba neuron, merupakan tiruan buatan dari otak manusia yang menghubungkan lapisan input ke lapisan output. Neural network adalah teknik pemodelan yang sangat canggih dan dapat memodelkan fungsi yang sangat kompleks. Menggunakan jaringan syaraf tiruan, dimungkinkan untuk memprediksi harapan masa depan, mengklasifikasikan nilai data ke dalam kelompok yang memungkinkan keputusan dibuat menggunakan mesin yang cerdas secara artifisial.

Seperti manusia, jaringan saraf tiruan dilatih oleh supervisor sesuai dengan paradigma pembelajaran yang diawasi. Neural network adalah bagian dari kecerdasan buatan dan pembelajaran mesin. Pembelajaran mendalam menjadi lebih maju dengan bertambahnya data. Oleh karena itu, jaringan saraf menyesuaikan kinerjanya menjadi lebih besar saat mereka tumbuh lebih besar dan menangani arus informasi yang sangat besar (Hukubun, 2022).

#### B. Multilayer Perceptron

Multi Layer Perceptron (MLP) merupakan salah satu algoritma deep learning yang sudah banyak digunakan dalam penelitian masalah klasifikasi karena memiliki kelebihan dibandingkan metode klasifikasi konvensional lain. MLP memiliki kelebihan seperti mampu mengadaptasikan dirinya dengan data, dapat memperkirakan hubungan antar keanggotaan kelas dengan atribut dari objek, lebih reliabel terhadap noise dalam data, dan mampu menghitung nilai probabilitas posterior, yang merupakan dasar untuk membangun aturan klasifikasi dan analisis statistika.

Secara umum, langkah-langkah dalam pelatihan algoritma multilayer perceptron menggunakan backpropagation untuk mengubah nilai bobot-bobotnya dalam arah mundur (backward) (Pardede, Hayadi, & Iskandar, 2022).

#### C. Backpropagation Learning

Backpropagation adalah algoritma pembelajaran yang digunakan untuk mengoptimalkan bobot (weights) neuron dalam jaringan syaraf tiruan. Prinsipnya adalah menghitung error antara output jaringan dan target, kemudian menyebarkan error tersebut ke seluruh lapisan untuk memperbaiki bobot secara bertahap (Goodfellow, Bengio, & Courville, 2016). Dengan metode ini, jaringan dapat belajar dari kesalahan dan meningkatkan performa prediksi baik pada klasifikasi maupun regresi (Haykin, 2009).

#### D. Activation Function

Fungsi aktivasi adalah komponen penting dalam neural network yang menentukan output dari sebuah neuron berdasarkan input yang diterima. Fungsi ini memungkinkan jaringan untuk mempelajari hubungan non-linear pada data, sehingga MLP dapat melakukan klasifikasi atau regresi yang kompleks (Goodfellow, Bengio, & Courville, 2016). Tanpa fungsi aktivasi, jaringan hanya akan menjadi kombinasi linear dari input (Haykin, 2009). Beberapa fungsi aktivasi populer adalah Sigmoid, Tanh, ReLU, dan Softmax (Nielsen, 2015).

#### E. Klasifikasi

Klasifikasi adalah salah satu teknik dalam supervised learning yang bertujuan untuk mengelompokkan data ke dalam kategori berdasarkan label yang telah ditentukan. Dalam klasifikasi, model dilatih menggunakan dataset berlabel untuk mengenali pola dan menentukan kelas suatu data baru yang belum diketahui.

Beberapa algoritma yang umum digunakan dalam klasifikasi meliputi Support Vector Machines (SVM), Normal Bayesian Classifier (NBC), K-Nearest Neighbor (KNN), Trees Gradient Boosted (GBT), Random Trees (RT), dan Artificial Neural Networks (ANN). Algoritma-algoritma ini membantu dalam memprediksi kelas suatu data berdasarkan fitur yang dimilikinya, memungkinkan aplikasi luas dalam berbagai bidang, termasuk deteksi penyakit, analisis gambar, dan sistem rekomendasi (Roihan, Sunarya, & Rafika, 2020).

#### F. Regresi

Linear regression merupakan salah satu algoritma yang memodelkan suatu persamaan untuk menghitung estimasi. Metode ini bertujuan untuk mencari pola pada nilai numerik, sehingga data yang dibutuhkan berupa data numerik agar dapat diolah dengan model algoritma ini. Linear regression memprediksi nilai data yang tidak diketahui dengan menggunakan nilai data lain yang terkait dan diketahui. Secara matematis memodelkan variabel yang tidak diketahui atau tergantung (dependent) dan variabel yang dikenal atau independen (independent) sebagai persamaan linier (Mardiani, et al., 2023).

---

### 8.3 ALAT DAN BAHAN

#### A. Perangkat Keras
1. Laptop dengan minimal RAM 4 GB
2. Koneksi internet stabil

#### B. Perangkat Lunak
1. Browser (Chrome, Edge, Firefox, dsb)
2. Google Colab / Jupyter Notebook
3. Library ML (pandas, numpy, scikit-learn, matplotlib, dsb)

---

### 8.4 LANGKAH PERCOBAAN

#### 1. Persiapkan IDE
Mahasiswa memulai percobaan dengan membuka Google Colab sebagai lingkungan pengembangan utama. Google Colab dipilih karena telah mendukung berbagai library machine learning yang dibutuhkan. Colab juga menyediakan resource komputasi berbasis CPU atau GPU secara gratis sehingga proses pelatihan model MLP dapat berjalan lebih optimal tanpa membutuhkan perangkat keras khusus.

#### 2. Instalasi dan Impor Library
Setelah lingkungan siap, mahasiswa mengimpor library yang diperlukan untuk manipulasi data, preprocessing, visualisasi, dan pembangunan model neural network. Library yang digunakan antara lain: pandas, numpy, matplotlib, seaborn, scikit-learn (MLPClassifier dan MLPRegressor), serta imbalanced-learn untuk metrik tambahan pada data tidak seimbang.

```python
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
```
*Gambar 8.4.1 Impor Pustaka*

#### 3. Import dan Load Dataset Information
Dataset diambil dari sumber online (Kaggle/GitHub) dan dibaca menggunakan pandas. Untuk percobaan:
1. Klasifikasi: dataset Bank Customer Churn digunakan untuk memprediksi pelanggan yang akan churn.
2. Regresi: dataset KC House Data digunakan untuk memprediksi harga rumah.

#### 4. Exploratory Data Analysis (EDA)
Pada tahap ini, mahasiswa melakukan eksplorasi awal terhadap dataset untuk memahami karakteristik data sebelum digunakan dalam model MLP.

#### 5. Data Cleaning
Mahasiswa memeriksa keberadaan nilai hilang (missing values) dan memproses variabel kategorikal menggunakan teknik encoding (Label Encoding atau Ordinal Encoding). Fitur numerik kemudian dinormalisasi dengan StandardScaler untuk meningkatkan stabilitas pelatihan model MLP.

#### 6. MLP untuk Klasifikasi
Mahasiswa membangun model MLPClassifier untuk memprediksi kategori target pada dataset churn. Proses mencakup:

**a. Melihat beberapa baris data awal (head())**
```python
# read file
df = pd.read_csv('/content/drive/MyDrive/Penambangan Data Sem 6/BankChurners.csv')

# baris data dari atas
df.head()
```
*Gambar 8.4.2 Membaca dataset*

**b. Menganalisis distribusi variabel menggunakan histogram**
```python
df.hist(figsize=(20, 15))
plt.show()
```
*Gambar 8.4.3 Menganalisis dataset*

**c. Mengecek proporsi target class**
```python
# cek persentase target class
data = df['Attrition_Flag'].value_counts()
data.plot(kind='pie', autopct='%1.2f%%')
plt.show()
```
Output Distribusi:
- Existing Customer: 83.93%
- Attrited Customer: 16.07%

*Gambar 8.4.4 Mengecek proporsi data*

**d. Mengecek missing value**
```python
# missing value
df.isnull().sum()
```
*Gambar 8.4.5 Mengecek nilai null*

**e. Mengidentifikasi atribut kategorikal dan numerik**
```python
# cek kategorikal atribut
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)
```
Output:
```
Index(['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], dtype='object')
```
*Gambar 8.4.6 Mengidentifikasi atribut kategorikal dan numerik*

**f. Membagi data menjadi train dan test serta training model**
```python
# get X and y
df_X = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)
df_y = df[['Attrition_Flag']]

# label encoding for y
# merubah nilai yg ada di y menjadi 0 atau 1
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

# hold-out method, dibagi menjadi training dan testing set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# mulai learning
model = MLPClassifier(max_iter=500)
# model = MLPClassifier(max_iter=1000, random_state=42, activation='logistic')
model.fit(X_train, y_train)
```
*Gambar 8.4.7 Pembagian data training dan testing*

```python
# koef model
model.coefs_
```
*Gambar 8.4.8 Koefisien model*

**g. Visualisasi Data Preprocessed**  
*Gambar 8.4.9 & Gambar 8.4.10: Visualisasi Data df_X dan representasi X_train[0].*

**h. Mengevaluasi performa model**
```python
# evaluasi & visualisasi confusion matrix
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("Confusion matrix :", confusion_matrix(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=model.classes_)
disp.plot()
plt.show()
```
*Gambar 8.4.11 Visualisasi Confusion Matrix*

**i. Melatih model dan prediksi data inferensi baru**
```python
# menyiapkan input sampel data baru
data = [{
    'Customer_Age': 58, 'Gender': 'F', 'Dependent_count': 1, 'Education_Level': 'Graduate',
    'Marital_Status': 'Single', 'Income_Category': 'Less than $40K', 'Card_Category': 'Blue',
    'Months_on_book': 44, 'Total_Relationship_Count': 5, 'Months_Inactive_12_mon': 2,
    'Contacts_Count_12_mon': 2, 'Credit_Limit': 4780, 'Total_Revolving_Bal': 800, 'Avg_Open_To_Buy': 3982,
    'Total_Amt_Chng_Q4_Q1': 1.543, 'Total_Trans_Amt': 1388, 'Total_Trans_Ct': 32, 'Total_Ct_Chng_Q4_Q1': 2.714,
    'Avg_Utilization_Ratio': 0.167
}]

df_input = pd.DataFrame(data)

# categorical encoding dan preprocessing yang selaras
df_input_scaled = scaler.transform(df_input)
result = model.predict(df_input_scaled)
```
*Gambar 8.4.12 Melatih model MLP & Gambar 8.4.13 Melihat hasil prediksi*

---

#### 7. MLP untuk Regresi
Mahasiswa membangun model MLPRegressor untuk memprediksi nilai kontinu (harga rumah). Langkah yang dilakukan:

**a. Membaca dataset**
```python
# read dataset
df = pd.read_csv('/content/drive/MyDrive/Penambangan Data Sem 6/kc_house_data.csv')
df.head()
```
*Gambar 8.4.14 Membaca dataset*

**b. Mengecek missing value**
```python
# cek missing values
df.isnull().sum()
```
*Gambar 8.4.15 Mengecek missing value*

**c. Menampilkan histogram fitur numerik**
```python
# menampilkan histogram fitur numerik
df.hist(figsize=(20, 15))
plt.show()
```
*Gambar 8.4.16 Histogram fitur numerik*

**d. Menyiapkan atribut dan pemisahan data**
```python
# cek kategori atribut
df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df[['price']]
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)
# Index([], dtype='object')
```
*Gambar 8.4.17 Mengecek Performa*

**e. Melatih model MLPRegressor**
```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df['price']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df_X)
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = MLPRegressor()
reg.fit(X_train, y_train)

print('coef of determination training :', reg.score(X_train, y_train))
print('coef of determination testing :', reg.score(X_test, y_test))

print('prediction')
y_pred = reg.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[:10])
```
Output Skor Determinasi:
- coef of determination training : 0.563641874546569
- coef of determination testing : 0.5484535001698863

*Gambar 8.4.18 Mengecek performa*

**f. Mengukur performa dengan R² dan RMSE**
```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse :', rmse)
print('r2 :', r2)
```
Output Evaluasi:
- rmse : 255319.8034520965
- r2 : 0.5484535001698863

*Gambar 8.4.19 Evaluasi model*

**g. Visualisasi perbandingan prediksi vs nilai asli**
```python
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.DataFrame(y_test[:30], columns=['real'])
data2 = pd.DataFrame(y_pred[:30], columns=['predicted'])
df_new = pd.concat([data1, data2], axis=1)
df_new.plot(kind='bar')
plt.title("MLP regression")
plt.xlabel('Sample')
plt.ylabel('House Price')
plt.show()
```
*Gambar 8.4.20 Visualisasi hasil prediksi*

---

### 8.5 TUGAS DAN ANALISIS

#### 1. Tugas 1
- Unduh dataset: Bank Customer Churn Dataset dan kc_house_data.
- Jelaskan tujuan dari masing-masing dataset tersebut.
- Jelaskan fitur-fitur yang ada pada kedua dataset tersebut.

#### 2. Tugas 2
- Carilah dataset baru untuk permasalahan klasifikasi, lalu bangun model MLPClassifier.
- Carilah dataset baru untuk permasalahan regresi, lalu bangun model MLPRegressor.
- Tampilkan dan analisis performa masing-masing model (metrik klasifikasi dan regresi).

---

### 8.6 REFERENSI

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- Haykin, S. (2009). *Neural networks and learning machines* (3rd ed.). Pearson.
- Hukubun, A. J. (2022). *Neural Network*. Diambil kembali dari ResearchGate: https://www.researchgate.net/publication/365581579_Neural_Network
- Mardiani, E., Rahmansyah, N., Ningsih, S., Lantana, D. A., Wirawan, A. S., Wijaya, S. A., & Putri, D. N. (2023). Komparasi Metode Knn, Naive Bayes, Decision Tree, Ensemble, Linear Regression Terhadap Analisis Performa Pelajar SMA. *INNOVATIVE: Journal Of Social Science Research*, 13880-13892.
- Nielsen, M. A. (2015). *Neural networks and deep learning: A textbook*. Determination Press.
- Pardede, D., Hayadi, B. H., & Iskandar. (2022). KAJIAN LITERATUR MULTI LAYER PERCEPTRON: SEBERAPA BAIK PERFORMA ALGORITMA INI. *Journal of ICT Aplication and System*, 23-34.
- Roihan, A., Sunarya, P. A., & Rafika, A. S. (2020). Pemanfaatan Machine Learning dalam Berbagai Bidang: Review paper. *IJCIT (Indonesian Journal on Computer and Information Technology)*, 75-82.
