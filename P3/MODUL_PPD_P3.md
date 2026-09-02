> 🖼️ **[Document Watermark/Background Image]:** {A faint, semi-transparent watermark pattern depicting a multi-pointed star or floral design with a circular center and ornate, symmetrical botanical flourishes extending outward.}
> 
> 

Modul Praktikum Penambangan Data - Teknologi Rekayasa Perangkat Lunak - 2025

# PERTEMUAN III



## Classification



### 3.1. TUJUAN PEMBELAJARAN



A. Mahasiswa mampu memahami konsep dasar beberapa algoritma klasifikasi, seperti Logistic Regression, KNN, Decision Tree, dan Random Forest
B. Mahasiswa mengetahui fungsi dan peran metrik evaluasi klasifikasi, seperti accuracy, precision, dan recall dalam menilai performa model
C. Mahasiswa dapat menerapkan berbagai algoritma klasifikasi untuk membangun model
D. Mahasiswa mampu melakukan evaluasi model dengan menyusun tabel performa berisi accuracy, precision, dan recall dari setiap model

### 3.2. DASAR TEORI



* **Classification**


Klasifikasi merupakan salah satu teknik supervised learning yang bertujuan untuk memprediksi label atau kategori dari sebuah sampel berdasarkan fitur yang dimiliki. Model klasifikasi belajar dari data berlabel, kemudian menghasilkan suatu fungsi yang mampu menentukan kelas dari data baru yang belum pernah dilihat. Klasifikasi bekerja dengan cara mencari pola dari data historis untuk menentukan keputusan kelas yang paling mendekati. Klasifikasi umum digunakan dalam aplikasi seperti deteksi spam, diagnosis medis, hingga rekomendasi konten.


* **Algoritma Classification**


Ada banyak metode yang bisa dipakai untuk melakukan klasifikasi. Beberapa algoritma di bawah ini termasuk yang paling sering digunakan dan dijelaskan dalam dokumentasi Scikit-Learn dan TensorFlow karena sederhana, efektif, dan cocok untuk berbagai jenis dataset.


1. **Logistic Regression**


Logistic Regression merupakan algoritma klasifikasi berbasis regresi linier yang memodelkan probabilitas suatu sampel termasuk ke dalam kelas tertentu menggunakan fungsi sigmoid. Output model berada dalam rentang 0-1, yang kemudian dikonversi menjadi label kelas. Meskipun sederhana, algoritma ini efisien dan bekerja baik pada data yang memiliki hubungan linear separable.


2. **K-Nearest Neighbors (KNN)**


K-Nearest Neighbors mengklasifikasikan suatu data baru berdasarkan mayoritas kelas dari k tetangga terdekatnya. Kedekatan dihitung menggunakan metrik jarak seperti Euclidean distance. Algoritma ini bersifat non-parametric dan sangat bergantung pada struktur distribusi data serta nilai k yang dipilih.


3. **Decision Tree**


Decision Tree bekerja dengan cara membagi dataset secara rekursif berdasarkan fitur yang memberikan informasi terbaik. Proses pemilihan fitur dilakukan menggunakan ukuran seperti Gini impurity atau entropy/information gain. Kelebihannya adalah interpretasi yang mudah, tetapi mudah mengalami overfitting tanpa pengaturan parameter yang tepat.


4. **Random Forest**


Random Forest adalah algoritma ensemble yang menggabungkan banyak Decision Tree melalui teknik bagging. Setiap pohon dilatih pada subset data yang berbeda, dan hasil akhirnya ditentukan melalui voting. Random Forest membuat model lebih stabil, mengurangi overfitting, dan meningkatkan akurasi dibanding satu pohon tunggal.


5. **Adaboost (Adaptive Boosting)**


AdaBoost bekerja dengan menggabungkan beberapa weak learner (biasanya Decision Tree kecil) secara berurutan. Pada setiap iterasi, bobot sampel diperbarui sehingga model berikutnya lebih fokus pada data yang sebelumnya salah diklasifikasikan. Dengan cara ini, AdaBoost mampu meningkatkan performa model secara signifikan terutama pada dataset yang kompleks.




* **Metrik Evaluasi Classification**


Dalam machine learning khususnya pada model klasifikasi, diperlukan metrik evaluasi untuk menilai seberapa baik model mampu memprediksi label dengan benar. Evaluasi ini dilakukan dengan membandingkan hasil prediksi model terhadap nilai aktual melalui sebuah struktur yang disebut confusion matrix.


1. **Confusion Matrix**


Confusion matrix terdiri dari empat komponen penting, antara lain True Positive (TP) yaitu prediksi positif yang benar, True Negative (TN) yaitu prediksi negatif yang benar, False Positive (FP) yaitu prediksi positif yang salah, dan False Negative (FN) yaitu prediksi negatif yang salah. Berdasarkan nilai-nilai ini, beberapa metrik evaluasi dapat dihitung untuk memahami performa model secara lebih komprehensif.


2. **Accuracy, Precision, Recall, F1 Score**


Accuracy merupakan metrik paling dasar yang menunjukkan proporsi prediksi yang benar dari seluruh prediksi yang dibuat oleh model. Rumusnya didefinisikan sebagai berikut:


$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$




Precision mengukur tingkat ketepatan model dalam melakukan prediksi positif, yaitu berapa banyak prediksi positif yang benar-benar positif. Precision dihitung dengan rumus berikut:


$$Precision=\frac{TP}{TP + FP}$$




Sementara itu, Recall mengukur kemampuan model dalam menemukan seluruh sampel yang benar-benar positif. Recall dihitung menggunakan rumus:


$$Recall=\frac{TP}{TP+FN}$$




Karena precision dan recall memiliki fokus yang berbeda, F1-Score digunakan sebagai metrik yang menyeimbangkan keduanya. Metrik ini sangat berguna ketika dataset bersifat imbalanced dan diperlukan keseimbangan antara kemampuan model menghindari false positives serta false negatives. F1-score merupakan harmonic mean dari precision dan recall, dengan rumus:


$$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$






### 3.3. ALAT DAN BAHAN



* **Perangkat Keras**


Komputer/Laptop


* **Perangkat Lunak**


Python3
Google Colaboratory



### 3.4. LANGKAH PERCOBAAN



[Isi dengan instruksi langkah demi langkah yang jelas disertai penjelasan singkat. Tambahkan gambar ilustrasi jika diperlukan]

1. **Impor Pustaka**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.metrics import sensitivity_specificity_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

```


> 🖼️ **[Gambar 3.4.1 Import Pustaka]:** {A screenshot of a code editor showing a list of Python import statements for data manipulation and machine learning libraries including pandas, numpy, matplotlib, and various modules from sklearn and imblearn.}
> 
> 


2. **Load dataset**


Dataset dapat diunduh pada link berikut:
[https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)


```python
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/refs/heads/main/healthcare-dataset-stroke-data.csv')
```[cite: 1]


```


3. **Exploratory Data Analysis (EDA)**


a. Melihat lima baris pertama dan terakhir pada dataset dengan `df.head()` dan `df.tail()`.


```python
df.head()
```[cite: 1]

| | id | gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi | smoking_status | stroke |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9046 | Male | 67.0 | 0 | 1 | Yes | Private | Urban | 228.69 | 36.6 | formerly smoked | 1 |
| 1 | 51676 | Female | 61.0 | 0 | 0 | Yes | Self-employed | Rural | 202.21 | NaN | never smoked | 1 |
| 2 | 31112 | Male | 80.0 | 0 | 1 | Yes | Private | Rural | 105.92 | 32.5 | never smoked | 1 |
| 3 | 60182 | Female | 49.0 | 0 | 0 | Yes | Private | Urban | 171.23 | 34.4 | smokes | 1 |
| 4 | 1665 | Female | 79.0 | 1 | 0 | Yes | Self-employed | Rural | 174.12 | 24.0 | never smoked | 1 |

```



```
> 🖼️ **[Gambar 3.4.2 Lima Baris Pertama Dataset]:** {A screenshot showing a tabular DataFrame output of the first 5 rows of a stroke prediction dataset, with columns like id, gender, age, hypertension, etc.}[cite: 1]

```python
df.tail()
```[cite: 1]

| | id | gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi | smoking_status | stroke |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5105 | 18234 | Female | 80.0 | 1 | 0 | Yes | Private | Urban | 83.75 | NaN | never smoked | 0 |
| 5106 | 44873 | Female | 81.0 | 0 | 0 | Yes | Self-employed | Urban | 125.20 | 40.0 | never smoked | 0 |
| 5107 | 19723 | Female | 35.0 | 0 | 0 | Yes | Self-employed | Rural | 82.99 | 30.6 | never smoked | 0 |
| 5108 | 37544 | Male | 51.0 | 0 | 0 | Yes | Private | Rural | 166.29 | 25.6 | formerly smoked | 0 |
| 5109 | 44679 | Female | 44.0 | 0 | 0 | Yes | Govt_job | Urban | 85.28 | 26.2 | Unknown | 0 |

```

```
> 🖼️ **[Gambar 3.4.3 Lima Baris Terakhir Dataset]:** {A screenshot showing a tabular DataFrame output of the last 5 rows of a stroke prediction dataset, displaying indices 5105 to 5109.}[cite: 1]

b. Melihat deskripsi statistik dengan `df.describe()`[cite: 1]
```python
df.describe()
```[cite: 1]

| | id | age | hypertension | heart_disease | avg_glucose_level | bmi | stroke |
|---|---|---|---|---|---|---|---|
| count | 5110.000000 | 5110.000000 | 5110.000000 | 5110.000000 | 5110.000000 | 4909.000000 | 5110.000000 |
| mean | 36517.829354 | 43.226614 | 0.097456 | 0.054012 | 106.147677 | 28.893237 | 0.048728 |
| std | 21161.721625 | 22.612647 | 0.296607 | 0.226063 | 45.283560 | 7.854067 | 0.215320 |
| min | 67.000000 | 0.080000 | 0.000000 | 0.000000 | 55.120000 | 10.300000 | 0.000000 |
| 25% | 17741.250000 | 25.000000 | 0.000000 | 0.000000 | 77.245000 | 23.500000 | 0.000000 |
| 50% | 36932.000000 | 45.000000 | 0.000000 | 0.000000 | 91.885000 | 28.100000 | 0.000000 |
| 75% | 54682.000000 | 61.000000 | 0.000000 | 0.000000 | 114.090000 | 33.100000 | 0.000000 |
| max | 72940.000000 | 82.000000 | 1.000000 | 1.000000 | 271.740000 | 97.600000 | 1.000000 |

```

```
> 🖼️ **[Gambar 3.4.4 Hasil Deskripsi Statistik Dataset]:** {A screenshot displaying the statistical summary table for numeric variables in the dataset, detailing count, mean, standard deviation, min, percentiles (25%, 50%, 75%), and max.}[cite: 1]

c. Melihat presentase target kelas[cite: 1]
```python
data = df['stroke'].value_counts()
data.plot(kind='pie', autopct='%.2f%%')
plt.show()
```[cite: 1]

> 🖼️ **[Gambar 3.4.5 Presentase Target Kelas]:** {A pie chart visualizing the distribution of the 'stroke' target variable. The vast majority of the chart is a blue slice labeled '0' representing 95.13%. A very small orange slice is labeled '1', representing 4.87%. The y-axis label is 'count'.}[cite: 1]

Pie chart tersebut menunjukkan distribusi variabel stroke, di mana kelas tidak mengalami stroke (0) mendominasi dengan proporsi sekitar 95.13%, sedangkan kelas mengalami stroke (1) hanya sebesar 4.87%.[cite: 1] Hal ini menandakan bahwa dataset bersifat imbalanced, karena jumlah sampel pada kelas positif jauh lebih sedikit dibanding kelas negatif.[cite: 1]

d. Melihat histogram[cite: 1]
```python
df.hist(figsize=(10,10))
plt.show()
```[cite: 1]

> 🖼️ **[Gambar 3.4.6 Kode untuk Menampilkan Histogram]:** {A screenshot of the two lines of Python code: `df.hist(figsize=(10,10))` and `plt.show()`.}[cite: 1]

> 🖼️ **[Gambar 3.4.7 Histogram Seluruh Variabel Numerik]:** {A grid of 9 histograms displaying the distributions for the dataset's numerical variables: 'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', and 'stroke'. Most distributions such as 'avg_glucose_level' and 'bmi' are heavily right-skewed, while 'hypertension', 'heart_disease', and 'stroke' show extreme imbalance heavily weighted toward the value 0.0.}[cite: 1]

e. Mengecek missing value dengan `df.isnull().sum()`[cite: 1]
```python
df.isnull().sum()
```[cite: 1]

> 🖼️ **[Gambar 3.4.8 Pengecekan Missing Value]:** {A screenshot of the code `df.isnull().sum()` output, showing a list of columns with their missing value counts. All columns show 0 missing values except for 'bmi', which has 201 missing values. The dtype is listed as int64.}[cite: 1]

Hasil pengecekan missing value di atas menunjukkan bahwa hanya variabel bmi memiliki nilai kosong sebanyak 201 baris, sementara kolom lainnya tidak memiliki missing value.[cite: 1]

f. Mengecek atribut kategorikal[cite: 1]
```python
df_X = df.drop(['id', 'stroke'], axis=1) #definisikan kolom yg jadi input
df_y = df[['stroke']] #definisikan kolom yg jadi output
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)
```[cite: 1]
```text
Index(['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], dtype='object')
```[cite: 1]

> 🖼️ **[Gambar 3.4.9 Pengecekan Atribut Kategorikal]:** {A screenshot of the Python code block filtering the DataFrame for object and boolean data types, printing the resulting column index which includes 'gender', 'ever_married', 'work_type', 'Residence_type', and 'smoking_status'.}[cite: 1]

Hasil pengecekan atribut kategorikal menunjukkan bahwa variabel kategorikal yang perlu dilakukan encoding yaitu gender, ever_married, work_type, Residence_type, dan smoking_status.[cite: 1]

```

4. **Data Preprocessing**

> 🖼️ **[Gambar 3.4.10 Kode untuk Melakukan Data Preprocessing]:** {A screenshot of a comprehensive Python script for data preprocessing. The code outlines steps including dropping 'id' and 'stroke' for the input features (X), isolating 'stroke' for the target (y), imputing missing values in the 'bmi' column using the median, label encoding categorical columns, converting X and y to float numpy arrays, splitting the data into a 70/30 train-test split using `train_test_split`, and finally scaling the training and testing sets using `StandardScaler`.}
> 
> 


a. Membuat variable `df_X` untuk input variable dan `df_y` untuk target class
Pada tahap awal, dataset dipisahkan menjadi dua, yaitu variabel fitur dan variabel target. Fitur (X) merupakan atribut independen yang digunakan sebagai inputan model, sedangkan target (y) adalah variabel dependen yang ingin diprediksi.
b. Label encoding untuk y
Proses label encoding dilakukan untuk memastikan variabel y berada dalam format numerik.
c. Imputation dengan nilai tengah (median)
Nilai hilang pada fitur numerik diatasi menggunakan metode pengisian median, karena median lebih stabil terhadap keberadaan outlier.
d. Categorical encoding
Model machine learning hanya dapat memproses data numerik. Oleh karena itu, semua fitur bertipe object atau boolean dikonversi menjadi nilai numerik menggunakan Label Encoding.
e. Menyimpan X dan y menjadi numpy arrays
Agar kompatibel dengan fungsi-fungsi pada pustaka Scikit-Learn, variabel fitur dan target dikonversi menjadi array NumPy bertipe float.
f. Membagi data training dan testing
Dataset dibagi menjadi data training dan data testing dengan komposisi 70% data training dan 30% data testing. Pembagian ini dilakukan agar model dapat dievaluasi menggunakan data yang tidak pernah digunakan selama proses pelatihan.
g. Normalisasi data (standardization)
Standardisasi dilakukan menggunakan StandardScaler, yang mengubah distribusi setiap fitur sehingga memiliki nilai rata-rata 0 dan standar deviasi 1. Proses fit hanya dilakukan pada data training untuk mencegah data leakage, kemudian digunakan untuk mentransformasi data training dan data testing.


Tahapan preprocessing di atas memastikan bahwa dataset berada dalam kondisi optimal sebelum digunakan untuk membangun model klasifikasi. Langkah berikutnya, dilakukan proses pemeriksaan hasil preprocessing untuk memastikan seluruh transformasi sudah sesuai.


a. Variabel input X
Seluruh nilai kategorik telah dikonversi menjadi numerik.


```text
X
array([[ 1.  , 67.  ,  0.  , 228.69,  36.6 ,  1.  ],
       [ 0.  , 61.  ,  0.  , 202.21,  28.1 ,  2.  ],
       [ 1.  , 80.  ,  0.  , 105.92,  32.5 ,  2.  ],
       [ 0.  , 35.  ,  0.  ,  82.99,  30.6 ,  2.  ],
       [ 1.  , 51.  ,  0.  , 166.29,  25.6 ,  1.  ],
       [ 0.  , 44.  ,  0.  ,  85.28,  26.2 ,  0.  ]])
```[cite: 1]
> 🖼️ **[Gambar 3.4.11 Variabel X Setelah Dikonversi Menjadi Numerik]:** {A screenshot of a NumPy array containing rows of numeric data values, demonstrating the result of categorical encoding.}[cite: 1]

b. Variabel target y[cite: 1]
Variabel target atau label berupa nilai biner (0 dan 1) hasil dari proses label encoding.[cite: 1]
```text
y
array([1., 1., 1., 0., 0., 0.])
```[cite: 1]
> 🖼️ **[Gambar 3.4.12 Variabel y Setelah Proses Label Encoding]:** {A screenshot showing the output of the target variable y as a NumPy array containing the values 1., 1., 1., 0., 0., 0.}[cite: 1]

c. Data training yang telah dinormalisasi (X_train)[cite: 1]
X_train menampilkan nilai fitur yang telah distandardisasi, sehingga setiap kolom memiliki rata-rata 0 dan standar deviasi 1.[cite: 1]
```text
X_train
array([[ 1.18418848, -1.7467638 , -0.31719928, ..., -0.340693  ,
        -1.64591683, -1.29622579],
       [ 1.18418848, -0.63635252, -0.31719928, ...,  2.26654137,
        -0.78843307,  1.52066342],
       [ 1.18418848,  0.02969425,  3.15259225, ..., -0.32155489,
        -0.30772248,  0.58170035],
       ...,
       [-0.84446587, -1.87290052, -0.31719928, ..., -0.18803315,
         1.43804198, -1.29622579],
       [ 1.18418848,  1.62888649, -0.31719928, ...,  2.01062472,
         0.27692553, -0.35720272],
       [-0.84446587,  0.11872715, -0.31719928, ..., -0.12416526,
         2.78441501,  1.52066342]])
```[cite: 1]
> 🖼️ **[Gambar 3.4.13 Hasil Transformasi pada Data Training]:** {A screenshot displaying the highly truncated output of the `X_train` NumPy array, containing continuous, standardized float values resulting from a StandardScaler transformation.}[cite: 1]

d. Data testing yang telah dinormalisasi (X_test)[cite: 1]
X_test menunjukkan hasil transformasi fitur pada data testing menggunakan scaler yang sama seperti data training, sehingga menghasilkan distribusi nilai yang serupa dengan angka-angka pada skala terstandarisasi.[cite: 1]
```text
X_test
array([[ 1.18418848, -0.54751962,  0.31719928, ..., -0.90971812,
        -0.76244872, -1.29622579],
       [ 1.18418848, -0.14777156, -0.31719928, ..., -0.89992653,
        -0.07386327,  0.58170035],
       [-0.84446587, -1.569098  , -0.31719928, ...,  0.69675096,
        -0.82740961, -1.29622579],
       ...,
       [ 1.18418848, -0.05893866,  0.31719928, ...,  0.26569829,
        -0.21677723,  0.58170035],
       [-0.84446587,  0.60730811,  0.31719928, ..., -0.80846414,
        -0.63252693, -1.29622579],
       [-0.84446587,  0.74055746,  0.31719928, ...,  0.72746095,
        -0.46362862,  0.58170035]])
```[cite: 1]
> 🖼️ **[Gambar 3.4.14 Hasil Transformasi pada Data Testing]:** {A screenshot showing a similar truncated NumPy array for `X_test`, displaying standard scaled float data values for the testing dataset.}[cite: 1]


```


5. **Modelling**


a. Logistic Regression


> 🖼️ **[Gambar 3.4.15 Kode untuk Logistic Regression]:** {A screenshot of a Python code block initializing a `LogisticRegression()` model, fitting it to the training data, creating predictions on `X_test`, and printing the evaluation metrics including accuracy, precision, recall, and computing a Confusion Matrix using `ConfusionMatrixDisplay`.}
> 
> 


* **Melatih model**


Pada tahap ini, objek `LogisticRegression()` dibuat sebagai model klasifikasi. Model kemudian dilatih menggunakan data training (`X_train` dan `y_train`) agar dapat mempelajari hubungan antara fitur dan label target.


* **Membuat prediksi**


Model yang telah dilatih digunakan untuk memprediksi kelas dari data testing (`X_test`). Hasil prediksi disimpan pada variabel `y_pred`, yang berupa vektor berisi nilai kelas hasil prediksi.


* **Mengukur performa model**


Performa model diukur dengan metrik accuracy, precision, recall, dan confusion matrix. Penggunaan parameter `average='macro'` memastikan bahwa setiap kelas memiliki kontribusi yang sama dalam perhitungan precision dan recall.


```text
Accuracy 0.9419439008480104
Precision 0.4709719504240052
Recall 0.5
Confusion matrix [[1444    0]
 [  89    0]]
```[cite: 1]

```




> 🖼️ **[Gambar 3.4.16 Output Performa Model]:** {A text block output showing the Accuracy at roughly 94.19%, Precision at 47.09%, Recall at 50%, and a raw text Confusion matrix representing `[[1444 0] [89 0]]`.}
> 
> 


* **Visualisasi confusion matrix**

> 🖼️ **[Gambar 3.4.17 Confusion Matrix]:** {A heatmap visualization of the Confusion Matrix for Logistic Regression. The horizontal axis is 'Predicted label' with ticks 0.0 and 1.0. The vertical axis is 'True label' with ticks 0.0 and 1.0. The top-left cell (True Negative) is deeply shaded blue with a value of 1444. The bottom-left cell (False Negative) is slightly shaded light blue with a value of 89. The right column cells are both 0 and white. A color bar scaling from 0 to 1400 is on the right.}
> 
> 


Confusion matrix divisualisasikan menggunakan warna (heatmap) untuk memudahkan interpretasi. Visualisasi ini membantu melihat pola kesalahan model, misalnya apakah model cenderung salah pada kelas tertentu.


* **Menghitung F1 score**


F1-score merupakan kombinasi harmonis dari precision dan recall, sehingga memberikan gambaran performa yang lebih seimbang terutama pada dataset dengan distribusi kelas yang tidak merata.


```text
F1 0.485052065838092
```[cite: 1]
> 🖼️ **[Gambar 3.4.18 Output F1 Score]:** {A screenshot displaying the printed F1 score output: `F1 0.485052065838092`.}[cite: 1]


```


* **Model coefficient**

```text
model.coef_
array([[-0.01878829,  1.55249921,  0.10370412,  0.07921647, -0.18371621,
        -0.06959651,  0.06033662,  0.19742689, -0.03156246,  0.02218138]])
```[cite: 1]
> 🖼️ **[Gambar 3.4.19 Pengecekan Nilai Koefisien dari Model Logistic Rgression]:** {A screenshot showing the output of `model.coef_`, which is an array containing the learned weights for the 10 features.}[cite: 1]

Output `model.coef_` menunjukkan nilai koefisien yang diperoleh model Logistic Regression untuk setiap fitur setelah proses pelatihan.[cite: 1] Nilai positif menunjukkan bahwa fitur tersebut cenderung meningkatkan peluang prediksi kelas 1, sedangkan nilai negatif menurunkan peluang tersebut.[cite: 1]


```


* **Model intercept**

```text
model.intercept_
array([-4.04531553])
```[cite: 1]
> 🖼️ **[Gambar 3.4.20 Pengecekan Nilai Konstanta pada Model Logistic Regression]:** {A screenshot showing the output of `model.intercept_` resulting in an array containing a single value: `-4.04531553`.}[cite: 1]

Output `model.intercept_` merupakan nilai konstanta dalam fungsi keputusan model.[cite: 1] Intercept bernilai negative menunjukkan bahwa, ketika seluruh fitur bernilai nol (setelah melalui proses normalisasi), model memiliki kecenderungan awal untuk memprediksi kelas 0.[cite: 1]


```




b. K-Nearest Neighbors


> 🖼️ **[Gambar 3.4.21 Kode untuk Melatih Model KNN]:** {A screenshot of Python code initializing `KNeighborsClassifier(n_neighbors=10)`, fitting it to the training data, generating predictions, and printing accuracy, precision, recall, and drawing the Confusion Matrix.}
> 
> 


* **Melatih model**


Model KNN dibentuk dengan menetapkan jumlah tetangga (k) sebanyak 10, yang berarti keputusan klasifikasi akan ditentukan berdasarkan 10 sampel terdekat pada fitur. Data training (`X_train` dan `y_train`) digunakan untuk menyimpan representasi data sehingga model dapat menghitung jarak antar sampel pada tahap prediksi.


* **Membuat prediksi**


Model kemudian digunakan untuk memprediksi kelas dari data testing (`X_test`). Proses prediksi dilakukan dengan menghitung jarak antara sampel testing dan seluruh sampel training, kemudian memilih kelas mayoritas dari 10 tetangga terdekat.


* **Mengukur performa model**

```text
Accuracy 0.9419439008480104
Precision 0.4709719504240052
Recall 0.5
Confusion matrix [[1444    0]
 [  89    0]]
```[cite: 1]
> 🖼️ **[Gambar 3.4.22 Output Performa Model]:** {A text block displaying the performance metrics for the KNN model, matching the exact scores seen in the Logistic Regression output.}[cite: 1]


```


* **Visualisasi confusion matrix**

> 🖼️ **[Gambar 3.4.23 Confusion Matrix]:** {A heatmap visualization of the Confusion Matrix for KNN. Identical structure to the Logistic Regression matrix, showing 1444 in the top-left, 89 in the bottom-left, and 0 in the right column cells.}
> 
> 




c. Decision Tree


> 🖼️ **[Gambar 3.4.24 Kode untuk Melatih Model Decision Tree]:** {A screenshot of Python code initializing `DecisionTreeClassifier(criterion="entropy")`, training the model, making predictions, and printing the subsequent evaluation metrics and Confusion Matrix plot.}
> 
> 


* **Melatih model**


Model Decision Tree dibentuk menggunakan kriteria pemilihan split entropy, yang mengukur tingkat ketidakteraturan data pada setiap node. Semakin rendah nilai entropy, semakin baik kualitas pemisahan yang dilakukan. Data training (`X_train` dan `y_train`) digunakan untuk membangun struktur pohon keputusan yang optimal berdasarkan pola data.


* **Membuat prediksi**


Model yang telah terbentuk kemudian digunakan untuk memprediksi kelas dari data testing (`X_test`). Prediksi dilakukan dengan menelusuri jalur pada pohon keputusan, mulai dari root hingga mencapai node daun yang menentukan kelas akhir.


* **Mengukur performa model**

```text
Accuracy 0.9021526418786693
Precision 0.5294517089478175
Recall 0.5263235706059946
Confusion matrix [[1374   70]
 [  80    9]]
```[cite: 1]
> 🖼️ **[Gambar 3.4.25 Output Performa Model]:** {A text block output showing the Decision Tree metrics: Accuracy at ~90.2%, Precision at 52.9%, Recall at 52.6%, and a confusion matrix of `[[1374 70] [80 9]]`.}[cite: 1]


```


* **Visualisasi confusion matrix**

> 🖼️ **[Gambar 3.4.26 Confusion Matrix]:** {A heatmap visualization of the Confusion Matrix for the Decision Tree model. The top-left cell shows 1374, bottom-left is 80, top-right is 70, and bottom-right is 9.}
> 
> 




d. Random Forest


> 🖼️ **[Gambar 3.4.27 Kode untuk Melatih Model Random Forest]:** {A screenshot of Python code initializing a `RandomForestClassifier()` with default parameters, fitting the model, making predictions on the test set, and printing performance evaluations.}
> 
> 


* **Melatih model**


Model Random Forest dibentuk dengan menggunakan parameter default, yang secara otomatis membuat sejumlah pohon keputusan dan melatihnya pada subset acak dari dataset. Algoritma ini bekerja berdasarkan prinsip bagging, yaitu menggabungkan hasil prediksi dari banyak pohon untuk mengurangi varians dan meningkatkan akurasi. Data training (`X_train` dan `y_train`) digunakan untuk membangun forest tersebut.


* **Membuat prediksi**


Model yang telah dilatih kemudian digunakan untuk memprediksi kelas data testing (`X_test`). Prediksi akhir ditentukan melalui proses voting mayoritas dari seluruh pohon yang ada dalam ensemble.


* **Mengukur performa model**

```text
Accuracy 0.9419439008480104
Precision 0.4709719504240052
Recall 0.5
Confusion matrix [[1444    0]
 [  89    0]]
```[cite: 1]
> 🖼️ **[Gambar 3.4.28 Output Performa Model]:** {A text block output showing performance metrics for the Random Forest model, matching the exact scores and confusion matrix `[[1444 0] [89 0]]` seen in the earlier Logistic Regression output.}[cite: 1]


```


* **Visualisasi confusion matrix**

> 🖼️ **[Gambar 3.4.29 Confusion Matrix]:** {A heatmap visualization of the Confusion Matrix for Random Forest. It is identical to the first Logistic Regression plot, showing 1444 (True Negative), 89 (False Negative), and 0 in the remaining cells.}
> 
> 




e. Ada Boost


> 🖼️ **[Gambar 3.4.30 Kode untuk Melatih Model Ada Boost]:** {A screenshot of Python code setting up the `AdaBoostClassifier()`, training it, computing predictions, and printing accuracy metrics along with the Confusion Matrix display command.}
> 
> 


* **Melatih model**


Model AdaBoost dibentuk menggunakan parameter default, di mana algoritma akan melatih beberapa weak learners (biasanya decision tree berukuran kecil) secara berurutan. Setiap weak learner diberikan bobot berdasarkan tingkat kesalahannya, sehingga model berikutnya lebih fokus pada sampel-sampel yang sulit diprediksi. Proses ini meningkatkan akurasi keseluruhan model. Data latih (`X_train` dan `y_train`) digunakan sebagai dasar pembelajaran.


* **Melakukan prediksi**


Setelah proses pelatihan selesai, model digunakan untuk memprediksi kelas data testing (`X_test`). Prediksi akhir merupakan kombinasi berbobot dari seluruh weak learners yang telah dilatih sebelumnya.


* **Mengukur performa model**

```text
Accuracy 0.9419439008480104
Precision 0.4709719504240052
Recall 0.5
Confusion matrix [[1444    0]
 [  89    0]]
```[cite: 1]
> 🖼️ **[Gambar 3.4.31 Output Performa Model]:** {A text block displaying AdaBoost performance metrics, mirroring the outputs of the previous baseline models with an accuracy of 94.19% and all minority class predictions defaulting to 0.}[cite: 1]


```


* **Visualisasi confusion matrix**

> 🖼️ **[Gambar 3.4.32 Confusion Matrix]:** {A heatmap visualization of the Confusion Matrix for AdaBoost, identical to the plots displaying a `[[1444, 0], [89, 0]]` configuration.}
> 
> 





### 3.5. TUGAS & ANALISIS



Dataset: [https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction)

1. Jelaskan apa tujuan penggunaan dataset ini?


2. Definisikan atribut yang menjadi input dan output nya?


3. Silahkan membuat model klasifikasi logistic regression, KNN, Decision Tree, dan Random Forest untuk kasus dataset diatas!


4. Buatlah tabel yang menjelaskan performa dari model machine learning untuk kasus dataset diatas! Kolom pertama "model", kolom selanjutnya "accuracy, precision, recall"


5. Jelaskan dari hasil eksperimen di atas, model mana yang paling baik? Jelaskan alasan anda!



### 3.6. REFERENSI



[https://www.python.org/doc/](https://www.python.org/doc/)


[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)


[https://numpy.org/](https://numpy.org/)


Han, J., Pei, J., & Tong, H. (2022). Data Mining: Concepts and Techniques (4th ed.). Morgan Kaufmann/Elsevier.