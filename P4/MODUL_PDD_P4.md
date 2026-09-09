# PERTEMUAN IV

## Regression

### 4.1. TUJUAN PEMBELAJARAN

A. Mahasiswa mampu memahami konsep dasar regresi
B. Mahasiswa mengetahui prinsip kerja berbagai algoritma regresi, khususnya Linear Regression, Decision Tree Regression, dan Random Forest Regression
C. Mahasiswa dapat menerapkan berbagai algoritma regresi untuk membangun model
D. Mahasiswa mampu mengevaluasi performa model regresi menggunakan metrik RMSE, R², dan koefisien korelasi (R)

---

### 4.2. DASAR TEORI

* **Regresi**
Regresi adalah metode statistik dalam machine learning yang digunakan untuk memodelkan hubungan antara satu atau lebih variabel independen (fitur) dengan variabel dependen (target) yang bersifat numerik atau kontinu. Tujuan utama regresi adalah untuk memahami pola hubungan antara fitur dan target, memperkirakan nilai masa depan berdasarkan data sebelumnya, dan mengidentifikasi seberapa besar pengaruh masing-masing fitur terhadap output.

Dalam machine learning, regresi termasuk dalam supervised learning, karena model dilatih menggunakan pasangan input-output sehingga bisa belajar memprediksi nilai target yang belum diketahui. Contoh penggunaan regresi antara lain yaitu prediksi harga rumah, estimasi permintaan barang, prediksi curah hujan, dan evaluasi risiko. Regresi bekerja dengan cara menemukan fungsi atau model matematis yang paling sesuai dengan pola data. Model tersebut kemudian digunakan untuk memprediksi nilai baru berdasarkan input baru.

* **Algoritma Regresi**
Algoritma regresi yang umum digunakan adalah Linear Regression, Decision Tree Regression, dan Random Forest Regression. Masing-masing algoritma memiliki karakteristik dan cara kerja yang berbeda sesuai jenis data dan pola hubungan antar variabel.

#### 1. Linear Regression

Linear Regression adalah algoritma regresi paling dasar yang mengasumsikan bahwa hubungan antara fitur dan target bersifat linier. Model ini berusaha membangun sebuah garis atau hyperplane terbaik yang meminimalkan error antara prediksi dan nilai sebenarnya. Berikut merupakan persamaan linear regression.

$$y = b_{0} + b_{1}x_{1} + b_{2}x_{2} + \dots + b_{n}x_{n}$$

Dimana:

* $b_{0} =$ intercept (nilai dasar)
* $b_{n} =$ koefisien regresi
* $x_{n} =$ nilai fitur
* $y =$ target/prediksi

Model linear regression mencari koefisien terbaik menggunakan teknik Least Squares, yaitu metode yang meminimalkan jumlah kuadrat error. Kelebihan linear regression yaitu sangat cepat dan efisien untuk dataset besar, mudah dipahami dan dijelaskan (interpretable), cocok untuk hubungan yang mendekati linear. Sedangkan kekurangannya adalah kurang cocok untuk pola non-linear dan sensitif terhadap outlier.

#### 2. Decision Tree Regression

Decision Tree Regression merupakan salah satu algoritma regresi yang bekerja dengan membangun struktur pohon keputusan sebagai representasi model. Algoritma ini membagi data menjadi beberapa subset berdasarkan titik split yang dianggap paling optimal dalam mengurangi error. Setiap pembagian node dilakukan secara rekursif hingga mencapai leaf node yang berisi nilai prediksi. Pemilihan split biasanya dilakukan berdasarkan pengukuran error seperti Mean Squared Error (MSE) sehingga pemisahan data menghasilkan kelompok dengan variasi nilai target yang lebih homogen.

Decision Tree Regression sangat efektif untuk memodelkan hubungan non-linear karena struktur pohon memungkinkan model membuat batas keputusan yang kompleks. Selain itu, decision tree tidak memerlukan normalisasi atau scaling data dan dapat menangani fitur numerik maupun kategorikal. Meskipun demikian, algoritma ini memiliki kelemahan utama berupa kecenderungan overfitting, yaitu ketika model terlalu menyesuaikan diri dengan data training sehingga performanya menurun saat diuji pada data baru.

#### 3. Random Forest Regression

Random Forest Regression merupakan pengembangan dari Decision Tree yang menggunakan pendekatan ensemble learning, yaitu membangun banyak pohon keputusan dan menggabungkan prediksinya. Setiap pohon dibangun menggunakan sampel data yang dipilih secara acak melalui bootstrap sampling, serta subset fitur yang juga dipilih secara acak pada setiap pemisahan node. Proses ini menciptakan berbagai variasi pohon yang kemudian memberikan hasil prediksi lebih stabil dan tidak mudah overfitting dibanding tree tunggal.

Prediksi akhir pada Random Forest biasanya diperoleh dari rata-rata hasil prediksi semua pohon, sehingga model mampu menangkap pola kompleks tanpa mengorbankan akurasi. Selain itu, Random Forest menyediakan informasi mengenai feature importance, yang menunjukkan fitur mana yang paling berpengaruh dalam proses prediksi. Hal ini membuat Random Forest menjadi salah satu algoritma regresi yang umum digunakan, terutama pada dataset yang besar dan memiliki banyak fitur. Meski demikian, model ini memerlukan sumber daya komputasi yang lebih besar.

* **Metrik Evaluasi Regresi**
Evaluasi model regresi dilakukan menggunakan metrik khusus yang mampu mengukur kesalahan prediksi nilai numerik. Untuk menilai performa model regresi, digunakan dua metrik utama.

#### 1. RMSE

RMSE mengukur rata-rata jarak antara nilai prediksi dan nilai sebenarnya dengan memberikan penalti lebih besar terhadap kesalahan yang besar. RMSE merupakan akar dari Mean Squared Error, sehingga unitnya sama dengan unit target dan lebih mudah diinterpretasikan dalam masalah nyata. Nilai RMSE yang lebih kecil menandakan bahwa model menghasilkan prediksi yang lebih akurat.

$$RMSE = \sqrt{\frac{1}{n}\sum(y_{pred} - y_{true})^{2}}$$

RMSE yang memiliki nilai rendah menunjukkan bahwa model mampu menghasilkan prediksi yang lebih mendekati nilai sebenarnya, sehingga tingkat akurasinya semakin baik. Karena RMSE memberikan penalti yang lebih besar pada kesalahan yang ekstrem, metrik ini menjadi sensitif terhadap outlier. Sensitivitas tersebut justru menjadikan RMSE cocok digunakan pada data kontinu seperti harga rumah, di mana perbedaan nilai yang besar dapat memberikan dampak signifikan terhadap evaluasi performa model.

#### 2. Coefficient of Determination (R²)

Coefficient of Determination atau $R^{2}$ menunjukkan seberapa besar proporsi variansi target yang dapat dijelaskan oleh model. Nilai $R^{2}$ berada pada rentang 0 hingga 1, di mana nilai mendekati 1 menandakan bahwa model memiliki kemampuan prediksi yang baik dan dapat menangkap pola hubungan dengan baik. Berbeda dengan metrik klasifikasi seperti akurasi atau precision, RMSE dan $R^{2}$ merupakan metrik yang dirancang khusus untuk kasus regresi, karena mampu menilai kualitas prediksi numerik secara tepat.

$$0 \le R^{2} \le 1$$

Nilai $R^{2}$ yang semakin mendekati 1 menunjukkan bahwa model mampu menjelaskan proporsi variansi target dengan lebih baik, sehingga performanya dapat dikatakan semakin sesuai dengan pola data sebenarnya. Sebaliknya, jika nilai $R^{2}$ rendah, berarti model kurang mampu menangkap hubungan antara fitur dan target. Karena sifatnya yang mengukur seberapa besar variasi data yang dapat dijelaskan oleh model, $R^{2}$ sangat berguna dalam menilai tingkat kecocokan model regresi, terutama ketika ingin memahami seberapa representatif model tersebut terhadap pola keseluruhan dalam dataset.

---

### 4.3. ALAT DAN BAHAN

* **Perangkat Keras**
* Komputer/Laptop


* **Perangkat Lunak**
* Python3
* Google Colaboratory



---

### 4.4. LANGKAH PERCOBAAN

#### 1. Impor Pustaka

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

```

> 🖼️ **[Gambar 4.4.1 Import Library]:** Tangkapan layar dari cell code editor yang menampilkan skrip Python untuk mengimpor pustaka dasar manipulasi data dan visualisasi (pandas, matplotlib.pyplot, numpy, seaborn) serta modul-modul machine learning dari scikit-learn (LinearRegression, train_test_split, mean_squared_error, r2_score, DecisionTreeRegressor, RandomForestRegressor).

#### 2. Load dataset

Dataset dapat diunduh pada link berikut:

`[https://www.kaggle.com/datasets/shivachandel/kc-house-data](https://www.kaggle.com/datasets/shivachandel/kc-house-data)`

```python
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/kc_house_data.csv')

```

#### 3. Exploratory Data Analysis (EDA)

**a. Melihat 5 baris pertama dan terakhir pada dataset dengan `df.head()` dan `df.tail()`.**

> 🖼️ **[Gambar 4.4.2 Lima Baris Pertama Dataset]:** Tangkapan layar tabel data keluaran dari fungsi `df.head()`, menampilkan 5 baris pertama dari dataset KC House Data beserta 21 atribut kolomnya seperti id, date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, dan sqft_lot15.

> 🖼️ **[Gambar 4.4.3 Lima Baris Terakhir Dataset]:** Tangkapan layar tabel data keluaran dari perintah `df.tail()`, menampilkan 5 baris data paling akhir (indeks 21608 hingga 21612) yang memperlihatkan nilai observasi pada seluruh kolom dataset penjualan rumah.

**b. Melihat deskripsi statistik dengan `df.describe()`.**

> 🖼️ **[Gambar 4.4.4 Hasil Deskripsi Statistik Dataset]:** Tangkapan layar ringkasan statistik deskriptif dari `df.describe()` yang mencakup metrik count, mean, standard deviation (std), nilai minimum, persentil ke-25, 50 (median), 75, dan nilai maksimum untuk seluruh kolom bertipe numerik.

**c. Melihat histogram**

```python
df.hist(figsize=(10,10))
plt.show()

```

> 🖼️ **[Gambar 4.4.5 Kode untuk Menampilkan Histogram]:** Potongan kode Python berisikan perintah pemanggilan visualisasi histogram distribusi seluruh variabel numerik menggunakan `df.hist(figsize=(10,10))` diikuti dengan `plt.show()`.

> 🖼️ **[Gambar 4.4.6 Histogram Seluruh Variabel Numerik]:** Kisi-kisi grafik histogram 5x4 yang menampilkan visualisasi sebaran frekuensi untuk masing-masing atribut (id, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, dan sqft_lot15), sebagian besar menunjukkan kurva kemiringan positif (right-skewed).

**d. Mengecek korelasi atribut numeric dengan `df.corr(numeric_only = True)**`

> 🖼️ **[Gambar 4.4.7 Korelasi Atribut Numerik]:** Matriks korelasi tabular yang menunjukkan nilai koefisien korelasi Pearson antar seluruh pasangan atribut numerik pada dataset, digunakan untuk menganalisis hubungan linier terhadap variabel harga (price).

**e. Mengecek missing value dengan `df.isnull().sum()**`

```text
id               0
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
dtype: int64

```

> 🖼️ **[Gambar 4.4.8 Pengecekan Missing Value]:** Tampilan output eksekusi `df.isnull().sum()` pada terminal/notebook, membuktikan bahwa setiap kolom bernilai 0 (tidak ada data yang hilang/missing values).

**f. Mengecek atribut kategorikal**

```python
df_X = df.drop(['id', 'date', 'price'], axis=1)
y = df['price']
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)

```

Output:

```text
Index([], dtype='object')

```

> 🖼️ **[Gambar 4.4.9 Pengecekan Atribut Kategorikal]:** Tangkapan layar kode seleksi tipe data menggunakan `select_dtypes` dan hasil print menunjukkan objek Index kosong (`Index([], dtype='object')`), menandakan semua fitur yang tersisa bertipe numerik.

---

#### 4. Modelling

**a. Linear Regression**

```python
df_X = df.drop(['id', 'date', 'price'], axis=1)
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
print(y_test[0:10])

```

> 🖼️ **[Gambar 4.4.10 Kode untuk Membuat Model Linear Regression]:** Tangkapan layar editor kode skrip Python untuk alur lengkap pelatihan model Linear Regression, mulai dari pemisahan data fitur-target, splitting dataset (70:30), inisialisasi dan fitting model, hingga pencetakan skor determinasi, koefisien, intercept, serta 10 nilai awal perbandingan antara prediksi dan label riil.

* **Menentukan fitur (x) dan target (y)**
Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
Selanjutnya, objek model Linear Regression dibuat dengan `reg = LinearRegression()`. Model kemudian dilatih menggunakan data training melalui perintah `reg.fit(X_train, y_train)`, di mana model mempelajari hubungan linier antara fitur dan target.
* **Mengukur kemampuan model**
Setelah tahap training selesai, nilai coefficient of determination atau $R^{2}$ dihitung baik untuk data training maupun testing.
* **Melihat parameter model**
Model Linear Regression menghasilkan dua parameter, yaitu koefisien dan intercept. Koefisien menunjukkan pengaruh masing-masing fitur terhadap nilai prediksi, sedangkan intercept merupakan nilai dasar ketika semua fitur bernilai nol. Kedua parameter ini ditampilkan dalam kode melalui `reg.coef_` dan `reg.intercept_`.
* **Membuat prediksi model**
Tahap selanjutnya adalah membuat prediksi pada data testing menggunakan `reg.predict(X_test)`. Hasil prediksi kemudian disimpan dalam variabel `y_pred`, kemudian ditampilkan untuk melihat 10 prediksi pertama. Untuk memastikan hasilnya masuk akal, nilai prediksi tersebut dibandingkan dengan 10 nilai sebenarnya dari `y_test`. Langkah ini memberikan gambaran awal mengenai akurasi model sebelum dilakukan evaluasi lebih lanjut menggunakan metrik seperti RMSE atau R².

```text
coef of determination training 0.6995155846436756
coef of determination testing 0.6994627057969904
coefficient
[-3.43081477e+04  4.03129700e+04  1.12001375e+02  9.91841247e-02
  5.27154218e+03  5.43877177e+05  5.50830616e+04  2.31460673e+04
  9.49081794e+04  7.22190668e+01  3.97823082e+01 -2.59441847e+03
  2.19209734e+01 -5.56358731e+02  5.95216324e+05 -1.96904658e+05
  1.62077488e+01 -3.30430480e-01]
intercept
6641646.708095346
prediction
[ 458597.0676416   748993.75994823 1243303.75799061 1665116.95095454
  737302.05741732  283239.58524967  831732.8758231   495383.02095346
  385779.81919014  474179.42285135]
real value
[ 365000.  865000. 1038000. 1490000.  711000.  211000.  790000.  680000.
  384500.  605000.]

```

> 🖼️ **[Gambar 4.4.11 Hasil Prediksi Model Linear Regression]:** Tangkapan layar keluaran konsol model Linear Regression: skor $R^2$ training (~0.6995), testing (~0.6994), larik koefisien fitur, nilai bias/intercept (6641646.71), serta perbandingan 10 baris pertama nilai prediksi terhadap data aktual.

Tahap selanjutnya adalah evaluasi model dengan metriks RMSE dan $R^{2}$ untuk mengukur kualitas prediksi dari model regresi.

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse : ', rmse)
print('r2 : ', r2)

```

> 🖼️ **[Gambar 4.4.12 Kode Evaluasi Model Menggunakan RMSE dan R2]:** Snippet kode perhitungan metrik performa model dengan scikit-learn menghitung nilai MSE, melakukan penarikan akar (akar kuadrat) melalui `np.sqrt` untuk mendapatkan RMSE, serta evaluasi koefisien determinasi `r2_score`.

Pertama, nilai mean squared error (MSE) dihitung menggunakan fungsi `mean_squared_error(y_test, y_pred)`. Nilai MSE menggambarkan rata-rata kuadrat selisih antara nilai prediksi dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik performa model. Selanjutnya, nilai Root Mean Squared Error (RMSE) diperoleh dengan mengambil akar kuadrat dari MSE menggunakan `np.sqrt(mse)`. RMSE memberikan interpretasi kesalahan dalam satuan yang sama dengan target, sehingga lebih mudah untuk dibandingkan.

Kemudian, performa model juga dievaluasi menggunakan $R^{2}$ (coefficient of determination) melalui fungsi `r2_score(y_test, y_pred)`. Nilai $R^{2}$ menunjukkan seberapa besar variansi data yang dapat dijelaskan oleh model. Nilai yang mendekati 1 berarti model memiliki kemampuan prediksi yang sangat baik, sedangkan nilai mendekati 0 menandakan model kurang mampu menjelaskan pola hubungan dalam data.

```text
rmse :  208296.72772118862
r2 :  0.6994627057969904

```

> 🖼️ **[Gambar 4.4.13 Hasil Evaluasi Menggunakan Metrik RMSE dan R]:** Output konsol metrik evaluasi Linear Regression menunjukkan skor Root Mean Squared Error sebesar 208296.73 dan nilai $R^2$ sebesar 0.6995.

Berdasarkan hasil evaluasi, model menghasilkan nilai RMSE sebesar 208296.72, yang berarti rata-rata kesalahan prediksi model berada di kisaran angka tersebut terhadap nilai harga rumah sebenarnya. Sementara itu, nilai R² sebesar 0.699 menunjukkan bahwa model mampu menjelaskan sekitar 69.9% variansi dalam data harga rumah. Nilai ini mengindikasikan bahwa model Linear Regression memiliki kemampuan yang cukup baik dalam memodelkan hubungan antara fitur dan harga rumah, meskipun masih terdapat sekitar 30% variansi yang belum dapat dijelaskan oleh model.

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

```python
data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
#df_new.plot.bar() #bisa pakai cara 1
df_new.plot(kind='bar', figsize=(15,3)) # bisa pakai cara 2

plt.title("Linear regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()

```

> 🖼️ **[Gambar 4.4.14 Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Skrip Python yang menyatukan 50 data pertama dari array aktual dan prediksi menjadi satu objek DataFrame bernama `df_new`, kemudian menghasilkan diagram batang perbandingan dengan `df_new.plot(kind='bar')`.

* Pertama, diambil 50 sampel awal dari `y_test` dan `y_pred`, kemudian masing-masing diubah menjadi objek Series.
* Kedua Series ini digabungkan menjadi sebuah DataFrame bernama `df_new`, dengan dua kolom yang diberi label real values dan predicted values.
* Setelah data digabung, grafik batang (bar chart) dibuat menggunakan perintah `df_new.plot(kind='bar', figsize=(15,3))`. Visualisasi ini berfungsi sebagai alat untuk mengevaluasi apakah pola prediksi model mendekati pola sebenarnya.

> 🖼️ **[Gambar 4.4.15 Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Diagram batang ganda berwarna biru (real values) dan jingga (predicted values) untuk 50 sampel data pertama pada model Linear Regression, menunjukkan tingkat kemiripan prediksi terhadap harga asli rumah dengan beberapa deviasi pada sampel nilai tinggi.

---

**b. Decision Tree Regression**

```python
df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeRegressor(max_depth=10)
dt.fit(X_train, y_train)

print('coef of determination training ', dt.score(X_train, y_train))
print('coef of determination testing ', dt.score(X_test, y_test))

print('prediction')
y_pred = dt.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[0:10])

```

> 🖼️ **[Gambar 4.4.16 Kode untuk Membuat Model Decision Tree Regression]:** Skrip implementasi Decision Tree Regressor menggunakan pembatasan kedalaman struktur pohon `max_depth=10`, proses fitting data latih, kalkulasi nilai R² latih dan uji, serta inspeksi 10 hasil prediksi.

* **Menentukan fitur (x) dan target (y)**
Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
Model Decision Tree Regression dibuat dengan objek `DecisionTreeRegressor(max_depth=10)` yang berfungsi sebagai representasi pohon keputusan untuk tugas regresi. Parameter `max_depth=10` digunakan untuk membatasi kedalaman maksimum pohon sehingga model tidak terlalu kompleks dan dapat mengurangi risiko overfitting. Setelah objek model dibuat, tahap training dilakukan dengan memanggil `dt.fit(X_train, y_train)` sehingga model mempelajari pola hubungan antara fitur pada data latih dan nilai targetnya.
* **Mengukur kemampuan model**
Setelah tahap training selesai, nilai coefficient of determination atau $R^{2}$ dihitung baik untuk data training maupun testing.
* **Membuat prediksi model**
Setelah model selesai dilatih menggunakan data training, langkah berikutnya adalah membuat prediksi terhadap data yang belum pernah dilihat model, yaitu data testing. Proses ini dilakukan menggunakan perintah `dt.predict(X_test)`, yang menghasilkan serangkaian nilai hasil prediksi dan disimpan dalam variabel `y_pred`.

```text
coef of determination training 0.9185274272472612
coef of determination testing 0.7689346283378805
prediction
[ 372512.5         837693.06862745 1087247.96153846 1984617.64705882
  656314.08658009  237079.87179487  849106.21276596  517991.26363636
  366278.875       519039.40166667]
real value
[ 365000.  865000. 1038000. 1490000.  711000.  211000.  790000.  680000.
  384500.  605000.]

```

> 🖼️ **[Gambar 4.4.17 Hasil Prediksi Model Decision Tree Regression]:** Hasil log output untuk Decision Tree Regression menunjukkan koefisien determinasi data latih sebesar 0.9185 dan data uji sebesar 0.7689 beserta deretan 10 prediksi vs nilai asli.

Selanjutnya dilakukan evaluasi model menggunakan metrik seperti RMSE dan $R^{2}$. Berikut hasil evaluasi model menggunakan algoritma Decision Tree Regression.

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse : ', rmse)
print('r2 : ', r2)

```

> 🖼️ **[Gambar 4.4.18 Kode Evaluasi Model Menggunakan RMSE dan R2]:** Potongan kode eksekusi fungsi evaluasi eror kuadrat rata-rata dan skor koefisien korelasi R² untuk algoritma Decision Tree.

```text
rmse :  182642.01673989513
r2 :  0.7689346283378805

```

> 🖼️ **[Gambar 4.4.19 Hasil Evaluasi Model]:** Tampilan teks skor evaluasi model Decision Tree: RMSE menurun ke angka 182642.02 dan $R^2$ meningkat mencapai 0.7689.

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

```python
data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
#df_new.plot.bar() #bisa pakai cara 1
df_new.plot(kind='bar', figsize=(15,3)) # bisa pakai cara 2

plt.title("DT regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()

```

> 🖼️ **[Gambar 4.4.20 Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Blok kode Python untuk membentuk grafik perbandingan nilai prediksi Decision Tree dengan nilai riil target harga rumah pada 50 sampel data pertama.

Berikut hasil perbandingan antara nilai harga rumah sebenarnya dan nilai hasil prediksi model Decision Tree Regression untuk 50 sampel pertama dari data testing.

> 🖼️ **[Gambar 4.4.21 Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Diagram batang komparasi 50 sampel antara harga riil vs prediksi model Decision Tree Regression; terlihat batang oranye (prediksi) mengikuti profil batang biru (aktual) lebih rapat dibandingkan model linear sebelumnya.

---

**c. Random Forest Regression**

```python
df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

print('coef of determination training ', rf.score(X_train, y_train))
print('coef of determination testing ', rf.score(X_test, y_test))

print('prediction')
y_pred = rf.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[:10])

```

> 🖼️ **[Gambar 4.4.22 Kode untuk Membuat Model Random Forest Regression]:** Tangkapan layar sintaks pemodelan Random Forest Regressor default, proses training ensemble decision trees, evaluasi skor determinasi internal, dan penampilan 10 prediksi awal testing.

* **Menentukan fitur (x) dan target (y)**
Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
Tahap pemodelan Random Forest Regression dimulai dengan membentuk objek model melalui `RandomForestRegressor()`, yang secara default akan membuat sekumpulan pohon keputusan sebagai dasar prediksi. Model ini kemudian dilatih menggunakan data training dengan memanggil `rf.fit(X_train, y_train)`, sehingga setiap pohon dalam ensemble mempelajari pola hubungan antara fitur dan harga rumah dari subset data yang berbeda.
* **Mengukur kemampuan model**
Setelah tahap training selesai, nilai coefficient of determination atau $R^{2}$ dihitung baik untuk data training maupun testing.
* **Membuat prediksi model**
Tahap berikutnya adalah membuat prediksi menggunakan `rf.predict(X_test)`, yang menghasilkan nilai estimasi harga rumah dengan data testing.

```text
coef of determination training 0.9822001440062388
coef of determination testing 0.8596889494795431
prediction
[ 379509.          885196.         1130435.43333333 2179107.
  703632.1         249551.38        838150.5         630879.5
  404713.92        541167.67      ]
real value
[ 365000.  865000. 1038000. 1490000.  711000.  211000.  790000.  680000.
  384500.  605000.]

```

> 🖼️ **[Gambar 4.4.23 Hasil Prediksi Model Random Forest Regression]:** Output terminal model Random Forest Regression dengan akurasi koefisien determinasi data training mencapai 0.9822 dan data testing melonjak ke 0.8597, disertai sampel array hasil estimasi target.

Selanjutnya dilakukan evaluasi model menggunakan metrik seperti RMSE dan $R^{2}$. Berikut hasil evaluasi model menggunakan algoritma Decision Tree Regression.

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse : ', rmse)
print('r2 : ', r2)

```

> 🖼️ **[Gambar 4.4.24 Kode Evaluasi Model Menggunakan RMSE dan R2]:** Snippet instruksi kalkulasi metrik RMSE dan koefisien R2 untuk model ensemble Random Forest.

```text
rmse :  142364.91502742676
r2 :  0.8596889494795431

```

> 🖼️ **[Gambar 4.4.25 Hasil Evaluasi Model]:** Hasil kalkulasi metrik evaluasi model Random Forest: mencatat nilai RMSE terkecil yaitu 142364.92 dan skor R² tertinggi mencapai 0.8597.

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

```python
data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
#df_new.plot.bar() #bisa pakai cara 1
df_new.plot(kind='bar', figsize=(15,3)) # bisa pakai cara 2

plt.title("RF regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()

```

> 🖼️ **[Gambar 4.4.26 Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Skrip Python pembuatan plot diagram batang perbandingan 50 sampel harga rumah aktual terhadap estimasi hasil Random Forest Regression bertajuk "RF regression".

Berikut hasil perbandingan antara nilai harga rumah sebenarnya dan nilai hasil prediksi model Random Forest Regression untuk 50 sampel pertama dari data testing.

> 🖼️ **[Gambar 4.4.27 Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]:** Diagram batang komparasi bar `real values` (biru) dan `predicted values` (oranye) dari Random Forest Regression pada 50 sampel pengujian, memperlihatkan tingkat kecocokan pola tinggi dengan deviasi margin error paling minimal di antara algoritma yang diuji.

---

### 4.5. TUGAS & ANALISIS

**Dataset:** `[https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data)`

1. Jelaskan apa tujuan penggunaan dataset ini dan definisikan atribut yang menjadi input dan output-nya!
2. Sebutkan variabel yang paling mempengaruhi output atau mempunyai nilai korelasi tinggi!
3. Silahkan membuat model regresi dengan Linear Regression, Decision Tree Regression, Random Forest Regression dari dataset diatas!
4. Buatlah tabel yang menjelaskan performa dari model machine learning untuk kasus dataset diatas! Kolom pertama "model", kolom selanjutnya "RMSE, R2, R".
5. Jelaskan dari hasil eksperimen di atas, model mana yang paling baik? Jelaskan alasan anda!

---

### 4.6. REFERENSI

Han, J., Pei, J., & Tong, H. (2022). *Data Mining: Concepts and Techniques* (4th ed.). Morgan Kaufmann/Elsevier.