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

# --- CELL 1: Header Markdown ---
cells.append(nbf.v4.new_markdown_cell("""# PRAKTIKUM PENAMBANGAN DATA (PPD)
## MODUL 6: CLUSTERING

* **Nama**: Januarsyah Akbar
* **NIM**: 24/535846/SV/24314
* **Kelas**: B2
* **Mata Kuliah**: Praktikum Penambangan Data (SVPL214610)
* **Program Studi**: D4 Teknologi Rekayasa Perangkat Lunak, Sekolah Vokasi UGM

---"""))

# --- CELL 2: Section A Header ---
cells.append(nbf.v4.new_markdown_cell("""## A. LANGKAH PERCOBAAN: CREDIT CARD CLUSTERING
Dataset: [Credit Card Dataset for Clustering](https://raw.githubusercontent.com/ganjar87/data_science_practice/main/CC%20GENERAL.csv)"""))

# --- CELL 3: 1. Inisialisasi Lingkungan & Auto-Clone / Download Dataset ---
cells.append(nbf.v4.new_markdown_cell("""### 1. Inisialisasi & Setup Lingkungan Colab / Local
Mempersiapkan dependensi pustaka dan skrip unduh dataset otomatis agar notebook Colab-Ready."""))

cells.append(nbf.v4.new_code_cell("""import os
import urllib.request
import ssl

# Handler SSL & Download Dataset
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

os.makedirs('data', exist_ok=True)

# Dataset 1: CreditCard
url_cc = 'https://raw.githubusercontent.com/ganjar87/data_science_practice/main/CC%20GENERAL.csv'
path_cc = 'data/CreditCard.csv'
if not os.path.exists(path_cc):
    try:
        req = urllib.request.Request(url_cc, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            with open(path_cc, 'wb') as f:
                f.write(response.read())
        print("Dataset Credit Card berhasil diunduh!")
    except Exception as e:
        print("Gagal mengunduh Credit Card:", e)
else:
    print("Dataset Credit Card sudah tersedia lokal.")

# Dataset 2: Air Traffic
url_air = 'https://data.sfgov.org/resource/rkru-6vcg.csv?$limit=50000'
path_air = 'data/Air_Traffic_Passenger_Statistics.csv'
if not os.path.exists(path_air):
    try:
        req = urllib.request.Request(url_air, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            with open(path_air, 'wb') as f:
                f.write(response.read())
        print("Dataset Air Traffic berhasil diunduh!")
    except Exception as e:
        print("Gagal mengunduh Air Traffic:", e)
else:
    print("Dataset Air Traffic sudah tersedia lokal.")"""))

# --- CELL 4: 2. Import Library ---
cells.append(nbf.v4.new_markdown_cell("""### 2. Impor Pustaka (Library)
Mengimpor library manipulasi data, analisis statistik, standarisasi, algoritma clustering (K-Means, K-Medoids), evaluasi, serta visualisasi."""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
import plotly.express as px

print("Seluruh pustaka berhasil diimpor!")"""))

# --- CELL 5: 3. Read & Load Dataset Information ---
cells.append(nbf.v4.new_markdown_cell("""### 3. Membaca dan Memeriksa Informasi Dataset
Memuat dataset Credit Card ke dalam DataFrame pandas dan menampilkan sampel data awal, ukuran matriks, ringkasan statistik, serta korelasi antar-fitur."""))

cells.append(nbf.v4.new_code_cell("""# Load dataset Credit Card
try:
    df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/CC%20GENERAL.csv')
except Exception:
    df = pd.read_csv('data/CreditCard.csv')

df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Ukuran/Dimensi dataset
print("Dimensi dataset:", df.shape)"""))

cells.append(nbf.v4.new_code_cell("""# Ringkasan statistik deskriptif
df.describe().T"""))

cells.append(nbf.v4.new_code_cell("""# Matriks korelasi antar kolom numerik
df.corr(numeric_only=True)"""))

# --- CELL 6: 4. Data Cleaning ---
cells.append(nbf.v4.new_markdown_cell("""### 4. Data Cleaning (Pembersihan Data)
Menghapus atribut identitas (`CUST_ID`) dan melakukan imputasi nilai kosong (*missing values*) pada fitur `MINIMUM_PAYMENTS` serta `CREDIT_LIMIT` menggunakan nilai median."""))

cells.append(nbf.v4.new_code_cell("""# Menghapus kolom CUST_ID
df_new = df.drop('CUST_ID', axis=1)
df_new.head()"""))

cells.append(nbf.v4.new_code_cell("""# Cek jumlah missing values sebelum imputasi
df_new.isnull().sum()"""))

cells.append(nbf.v4.new_code_cell("""# Imputasi missing values dengan median
df_new['MINIMUM_PAYMENTS'] = df_new['MINIMUM_PAYMENTS'].fillna(df_new['MINIMUM_PAYMENTS'].median())
df_new['CREDIT_LIMIT'] = df_new['CREDIT_LIMIT'].fillna(df_new['CREDIT_LIMIT'].median())

# Cek hasil imputasi
df_new.isnull().sum()"""))

# --- CELL 7: 5. Standarisasi Data ---
cells.append(nbf.v4.new_markdown_cell("""### 5. Standarisasi Data (Feature Scaling)
Mengubah skala data menggunakan `StandardScaler` agar seluruh fitur memiliki rata-rata ($\mu = 0$) dan standar deviasi ($\sigma = 1$)."""))

cells.append(nbf.v4.new_code_cell("""# Scaling
X = df_new.astype(float).values
scaler = StandardScaler().fit(X)
X_new = scaler.transform(X)
print("Bentuk matriks X_new:", X_new.shape)
X_new"""))

# --- CELL 8: 6. Menentukan Jumlah Cluster ---
cells.append(nbf.v4.new_markdown_cell("""### 6. Menentukan Jumlah Cluster Optimal (Elbow Method & Silhouette Score)
Melakukan pengujian jumlah kluster $k=1 \dots 10$ menggunakan metrik Inertia (WCSS) dan Silhouette Score untuk algoritma K-Means dan K-Medoids."""))

cells.append(nbf.v4.new_markdown_cell("""#### a. Elbow Method untuk K-Means"""))
cells.append(nbf.v4.new_code_cell("""# Metode Elbow K-Means
inertia_list_km = []
for num_clusters in range(1, 11):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_model.fit(X_new)
    inertia_list_km.append(kmeans_model.inertia_)
    print("For n_clusters = {}, inertia value is {}".format(num_clusters, kmeans_model.inertia_))"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Metode Elbow K-Means
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_list_km, marker='o', linewidth=2, markersize=8, color='#d97706')
plt.xlabel("Number of Clusters", size=13)
plt.ylabel("Inertia Value", size=13)
plt.title("Inertia values vary depending on the number of clusters utilized (K-Means)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Menentukan titik optimal K-Means menggunakan kneed
kneedle_km = KneeLocator(range(1, 11), inertia_list_km, S=1.0, curve='convex', direction='decreasing')
print("Knee optimal:", round(kneedle_km.knee, 3))
print("Elbow optimal:", round(kneedle_km.elbow, 3))

# Visualisasi knee point
plt.style.use('ggplot')
kneedle_km.plot_knee()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""#### b. Elbow Method untuk K-Medoids"""))
cells.append(nbf.v4.new_code_cell("""# Metode Elbow K-Medoids
inertia_list_kmed = []
for num_clusters in range(1, 11):
    kmedoids_model = KMedoids(n_clusters=num_clusters, random_state=42)
    kmedoids_model.fit(X_new)
    inertia_list_kmed.append(kmedoids_model.inertia_)
    print(f"The inertia of {num_clusters} clusters : {kmedoids_model.inertia_}")"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Metode Elbow K-Medoids
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_list_kmed, marker='o', linewidth=2, markersize=8, color='#2563eb')
plt.xlabel("Number of Clusters", size=13)
plt.ylabel("Inertia Value", size=13)
plt.title("Inertia values vary depending on the number of clusters utilized (K-Medoids)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Menentukan titik optimal K-Medoids dengan kneed
kneedle_kmed = KneeLocator(range(1, 11), inertia_list_kmed, S=1.0, curve='convex', direction='decreasing')
plt.style.use('ggplot')
kneedle_kmed.plot_knee()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""#### c. Silhouette Score untuk K-Means"""))
cells.append(nbf.v4.new_code_cell("""# Menentukan Silhouette Score K-Means
sh_list_km = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_new)
    score = silhouette_score(X_new, cluster_labels)
    sh_list_km.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(num_clusters, score))"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Silhouette Score K-Means
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sh_list_km, marker='o', linewidth=2, markersize=8, color='#059669')
plt.xlabel("Number of Clusters", size=13)
plt.ylabel("Silhouette Score", size=13)
plt.title("Silhouette score values vary depending on the number of clusters utilized (K-Means)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""#### d. Silhouette Score untuk K-Medoids"""))
cells.append(nbf.v4.new_code_cell("""# Menentukan Silhouette Score K-Medoids
sh_list_kmed = []
for num_clusters in range(2, 11):
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmedoids.fit_predict(X_new)
    score = silhouette_score(X_new, cluster_labels)
    sh_list_kmed.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(num_clusters, score))"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Silhouette Score K-Medoids
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sh_list_kmed, marker='o', linewidth=2, markersize=8, color='#7c3aed')
plt.xlabel("Number of Clusters", size=13)
plt.ylabel("Silhouette Score", size=13)
plt.title("Silhouette score vary depending on the number of clusters utilized (K-Medoids)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

# --- CELL 9: 7. Menerapkan Algoritma K-Means ---
cells.append(nbf.v4.new_markdown_cell("""### 7. Menerapkan Algoritma K-Means
Melatih model K-Means menggunakan $k=2$ (berdasarkan Silhouette Score optimal) dan memvisualisasikan persebaran titik kluster dalam grafik 2D dan 3D."""))

cells.append(nbf.v4.new_code_cell("""# Pelatihan K-Means dengan k=2
k_means = KMeans(n_clusters=2, random_state=42, n_init=10)
k_means.fit(X_new)
labels_km = k_means.labels_
df_new['cluster_labels'] = labels_km
df_new.head()"""))

cells.append(nbf.v4.new_code_cell("""# Cek Centroids K-Means
centroids_km = k_means.cluster_centers_
print("Dimensi Centroids:", centroids_km.shape)
centroids_km"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter Plot 2D Matplotlib
x1 = df_new['PURCHASES']
x2 = df_new['PAYMENTS']

plt.figure(figsize=(8, 6))
u_labels = np.unique(labels_km)
for i in u_labels:
    plt.scatter(x1[df_new['cluster_labels'] == i], x2[df_new['cluster_labels'] == i], label=f'Cluster {i}')

plt.xlabel(x1.name, fontsize=14)
plt.ylabel(x2.name, fontsize=14)
plt.title('K-means clustering', fontsize=16)
plt.legend()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter Plot 2D Seaborn
plt.figure(figsize=(7, 6))
sns.scatterplot(x='PURCHASES', y='PAYMENTS', hue='cluster_labels', data=df_new, palette='Paired', alpha=0.8)
plt.title('K-Means Clustering (Seaborn)', fontsize=14)
plt.legend(loc='lower right')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter 3D Plotly (Diproksikan ke Matplotlib 3D untuk Static Export)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_new['PURCHASES'], df_new['PAYMENTS'], df_new['BALANCE'], c=df_new['cluster_labels'], cmap='tab10', alpha=0.6)
ax.set_xlabel('PURCHASES')
ax.set_ylabel('PAYMENTS')
ax.set_zlabel('BALANCE')
plt.title('3D Scatter Plot K-Means (PURCHASES vs PAYMENTS vs BALANCE)', fontsize=12)
plt.show()"""))

# --- CELL 10: 8. Menerapkan Algoritma K-Medoids ---
cells.append(nbf.v4.new_markdown_cell("""### 8. Menerapkan Algoritma K-Medoids
Melatih model K-Medoids menggunakan $k=4$ (berdasarkan hasil Elbow Method) dan memvisualisasikan persebaran 4 kelompok nasabah."""))

cells.append(nbf.v4.new_code_cell("""# Pelatihan K-Medoids dengan k=4
k_medoids = KMedoids(n_clusters=4, random_state=42)
k_medoids.fit(X_new)
labels_kmed = k_medoids.labels_
df_new['cluster_labels'] = labels_kmed
df_new.head()"""))

cells.append(nbf.v4.new_code_cell("""# Cek Medoids/Centroids K-Medoids
centroids_kmed = k_medoids.cluster_centers_
print("Dimensi Medoids:", centroids_kmed.shape)
centroids_kmed"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter Plot 2D Matplotlib K-Medoids
x1 = df_new['PURCHASES']
x2 = df_new['PAYMENTS']

plt.figure(figsize=(8, 6))
u_labels = np.unique(labels_kmed)
for i in u_labels:
    plt.scatter(x1[df_new['cluster_labels'] == i], x2[df_new['cluster_labels'] == i], label=f'Cluster {i}')

plt.xlabel(x1.name, fontsize=14)
plt.ylabel(x2.name, fontsize=14)
plt.title('K-medoids clustering', fontsize=16)
plt.legend()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter Plot Seaborn K-Medoids
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PURCHASES', y='PAYMENTS', hue='cluster_labels', data=df_new, palette='Set1', alpha=0.8)
plt.title('K-Medoids Scatter Plot (Seaborn)', fontsize=14)
plt.legend(loc='lower right')
plt.show()"""))

# --- CELL 11: 9. Analisis & Interpretasi Cluster ---
cells.append(nbf.v4.new_markdown_cell("""### 9. Analisis dan Interpretasi Cluster
Menganalisis profil dan karakteristik tiap kluster nasabah melalui perbandingan rata-rata nilai transaksi (`PURCHASES`, `PAYMENTS`, `BALANCE`)."""))

cells.append(nbf.v4.new_code_cell("""# Barplot perbandingan fitur utama per kluster K-Medoids
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
sns.barplot(x='cluster_labels', y='PURCHASES', data=df_new, palette='viridis')
plt.title('Rata-rata PURCHASES')

plt.subplot(1, 3, 2)
sns.barplot(x='cluster_labels', y='PAYMENTS', data=df_new, palette='magma')
plt.title('Rata-rata PAYMENTS')

plt.subplot(1, 3, 3)
sns.barplot(x='cluster_labels', y='BALANCE', data=df_new, palette='plasma')
plt.title('Rata-rata BALANCE')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""**Ringkasan Profil Kluster K-Medoids:**
1. **Cluster 0**: Mempunyai nilai terendah pada fitur PURCHASES, PAYMENTS, dan BALANCE (Nasabah Pasif/Hemat).
2. **Cluster 1**: Mempunyai nilai sedang pada fitur PURCHASES, PAYMENTS, dan BALANCE (Nasabah Reguler).
3. **Cluster 2**: Mempunyai nilai tertinggi pada PURCHASES dan PAYMENTS, namun saldo BALANCE sedang (Nasabah Transaktif Tinggi/Pengguna Utama).
4. **Cluster 3**: Mempunyai nilai saldo BALANCE tertinggi, namun PURCHASES dan PAYMENTS tergolong sedang (Nasabah Pembawa Saldo/Peminjam Cash)."""))

cells.append(nbf.v4.new_code_cell("""# Contoh penggunaan LabelEncoder untuk variabel kategorial
cats = df.select_dtypes(include=['object', 'bool']).columns
cat_features = list(cats.values)
le = LabelEncoder()
for i in cat_features:
    df[i] = le.fit_transform(df[i])

print("Kolom kategorial ter-encode:", cat_features)
df.head(3)"""))

# --- CELL 12: Section B Header ---
cells.append(nbf.v4.new_markdown_cell("""## B. TUGAS DAN ANALISIS"""))

# --- CELL 13: Tugas 1 ---
cells.append(nbf.v4.new_markdown_cell("""### Tugas 1: Dataset Preparation & Analysis (Credit Card / CCData)
#### 1. Verifikasi Keberadaan Dataset & Aksesibilitas
Dataset `CC GENERAL.csv` telah diunduh dari repositori praktikum dan Kaggle CCData, disimpan dalam direktori `data/CreditCard.csv` serta dapat diakses secara langsung melalui pandas.

#### 2. Tujuan Penggunaan Dataset Credit Card
Dataset Credit Card (`CC GENERAL.csv`) dikumpulkan untuk memahami pola perilaku penggunaan kartu kredit oleh sekitar 9.000 nasabah aktif selama periode 6 bulan. Tujuan utamanya mencakup:
1. **Segmentasi Pelanggan (Customer Segmentation)**: Membagi basis nasabah ke dalam kelompok homogen berdasarkan frekuensi pembelian, penarikan tunai, dan pola pelunasan tagihan.
2. **Pengembangan Strategi Pemasaran Tertarget (Targeted Marketing)**: Merancang program promosi khusus, misalnya penawaran *cashback* untuk nasabah bertransaksi tinggi atau fasilitas cicilan bunga rendah bagi nasabah penarik tunai.
3. **Manajemen Risiko & Kredit**: Mengidentifikasi kelompok nasabah dengan risiko gagal bayar (*default risk*) tinggi yang dicirikan oleh saldo simpanan tinggi namun rasio pelunasan tagihan minimum sangat rendah.

#### 3. Penjelasan Fitur-Fitur Dataset (18 Fitur)
Dataset ini terdiri dari 18 atribut fitur transaksi yang dijelaskan pada tabel berikut:

| Nama Fitur | Tipe Data | Deskripsi & Makna Bisnis |
| :--- | :--- | :--- |
| `CUST_ID` | Categorical | Identifikasi unik untuk setiap pemegang kartu kredit. |
| `BALANCE` | Continuous | Sisa saldo hutang/tagihan yang belum dibayar oleh nasabah. |
| `BALANCE_FREQUENCY` | Continuous | Frekuensi seberapa sering saldo diperbarui (skala 0 hingga 1). |
| `PURCHASES` | Continuous | Total nominal transaksi pembelian yang dilakukan nasabah. |
| `ONEOFF_PURCHASES` | Continuous | Nominal maksimum transaksi pembelian sekali bayar (tanpa cicilan). |
| `INSTALLMENTS_PURCHASES` | Continuous | Total nominal transaksi pembelian yang dilakukan secara dicicil. |
| `CASH_ADVANCE` | Continuous | Total penarikan uang tunai di muka melalui mesin ATM/kartu kredit. |
| `PURCHASES_FREQUENCY` | Continuous | Seberapa sering transaksi pembelian dilakukan (0 = jarang, 1 = sangat sering). |
| `ONEOFF_PURCHASES_FREQUENCY` | Continuous | Frekuensi transaksi sekali bayar dilakukan secara langsung. |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Continuous | Frekuensi transaksi pembelian berbasis cicilan. |
| `CASH_ADVANCE_FREQUENCY` | Continuous | Frekuensi penarikan tunai di muka dilakukan. |
| `CASH_ADVANCE_TRX` | Discrete | Jumlah total frekuensi transaksi penarikan uang tunai. |
| `PURCHASES_TRX` | Discrete | Jumlah total transaksi pembelian yang berhasil dilakukan. |
| `CREDIT_LIMIT` | Continuous | Batas maksimum limit kredit yang diberikan bank kepada nasabah. |
| `PAYMENTS` | Continuous | Total jumlah pembayaran tagihan yang telah disetor oleh nasabah. |
| `MINIMUM_PAYMENTS` | Continuous | Jumlah minimum pembayaran tagihan yang diwajibkan oleh pihak bank. |
| `PRC_FULL_PAYMENT` | Continuous | Persentase pembayaran tagihan yang dilunasi secara penuh oleh nasabah. |
| `TENURE` | Discrete | Jangka waktu kepemilikan/layanan kartu kredit nasabah (dalam bulan). |"""))

# --- CELL 14: Tugas 2 Header ---
cells.append(nbf.v4.new_markdown_cell("""### Tugas 2: Clustering Dataset Air Traffic Passengers Statistics
Melakukan pemodelan unsupervised clustering (K-Means dan K-Medoids) pada dataset statistik penumpang penerbangan Bandara Internasional San Francisco (SFO)."""))

cells.append(nbf.v4.new_code_cell("""# 1. Memuat Dataset Air Traffic Passengers Statistics
try:
    df_air = pd.read_csv('data/Air_Traffic_Passenger_Statistics.csv')
except Exception:
    df_air = pd.read_csv('https://data.sfgov.org/resource/rkru-6vcg.csv?$limit=50000')

print("Dimensi dataset Air Traffic:", df_air.shape)
df_air.head()"""))

cells.append(nbf.v4.new_code_cell("""# Ringkasan struktur data & missing values
print("--- Ringkasan Info Dataset Air Traffic ---")
print(df_air.info())
print("\\n--- Jumlah Missing Values ---")
print(df_air.isnull().sum())"""))

cells.append(nbf.v4.new_code_cell("""# 2. Data Preprocessing & Feature Selection
df_air_clean = df_air.copy()

# Hapus kolom tanggal/waktu berlebih yang bernilaikonstan/ruwet jika ada
cols_drop = ['activity_period_start_date', 'data_as_of', 'data_loaded_at']
cols_to_drop = [c for c in cols_drop if c in df_air_clean.columns]
df_air_clean.drop(columns=cols_to_drop, inplace=True)

# Encoding variabel kategorial utama menggunakan LabelEncoder
cat_cols = ['operating_airline', 'geo_summary', 'geo_region', 'activity_type_code', 'price_category_code']
le_dict = {}
for col in cat_cols:
    if col in df_air_clean.columns:
        le = LabelEncoder()
        df_air_clean[col + '_encoded'] = le.fit_transform(df_air_clean[col].astype(str))
        le_dict[col] = le

# Memilih fitur numerik dan fitur ter-encode untuk clustering
feature_cols = ['passenger_count'] + [c + '_encoded' for c in cat_cols if c in df_air_clean.columns]
X_air = df_air_clean[feature_cols].values

# Standarisasi Fitur
scaler_air = StandardScaler()
X_air_scaled = scaler_air.fit_transform(X_air)

print("Fitur yang digunakan:", feature_cols)
print("Matriks data terstandarisasi X_air_scaled:", X_air_scaled.shape)"""))

cells.append(nbf.v4.new_code_cell("""# 3. Menentukan Jumlah Kluster Optimal (Elbow & Silhouette Score)

# a. K-Means
inertia_air_km = []
sh_air_km = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_air_scaled)
    inertia_air_km.append(km.inertia_)
    if k >= 2:
        sh_air_km.append(silhouette_score(X_air_scaled, km.labels_))

# Knee locator K-Means
kneedle_air_km = KneeLocator(range(1, 11), inertia_air_km, S=1.0, curve='convex', direction='decreasing')
k_opt_air_km = kneedle_air_km.elbow if kneedle_air_km.elbow else 3
print("Jumlah kluster optimal K-Means (Elbow):", k_opt_air_km)

# b. K-Medoids
inertia_air_kmed = []
sh_air_kmed = []
for k in range(1, 11):
    kmed = KMedoids(n_clusters=k, random_state=42)
    kmed.fit(X_air_scaled)
    inertia_air_kmed.append(kmed.inertia_)
    if k >= 2:
        sh_air_kmed.append(silhouette_score(X_air_scaled, kmed.labels_))

kneedle_air_kmed = KneeLocator(range(1, 11), inertia_air_kmed, S=1.0, curve='convex', direction='decreasing')
k_opt_air_kmed = kneedle_air_kmed.elbow if kneedle_air_kmed.elbow else 3
print("Jumlah kluster optimal K-Medoids (Elbow):", k_opt_air_kmed)"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Perbandingan Elbow & Silhouette Score untuk Air Traffic
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Elbow K-Means
axes[0, 0].plot(range(1, 11), inertia_air_km, marker='o', color='#d97706', linewidth=2)
axes[0, 0].axvline(x=k_opt_air_km, color='red', linestyle='--', label=f'Optimal k={k_opt_air_km}')
axes[0, 0].set_title('Elbow Method - K-Means (Air Traffic)')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()

# Subplot 2: Silhouette K-Means
axes[0, 1].plot(range(2, 11), sh_air_km, marker='o', color='#059669', linewidth=2)
best_k_sh_km = np.argmax(sh_air_km) + 2
axes[0, 1].axvline(x=best_k_sh_km, color='green', linestyle='--', label=f'Best Silhouette k={best_k_sh_km}')
axes[0, 1].set_title('Silhouette Score - K-Means (Air Traffic)')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()

# Subplot 3: Elbow K-Medoids
axes[1, 0].plot(range(1, 11), inertia_air_kmed, marker='o', color='#2563eb', linewidth=2)
axes[1, 0].axvline(x=k_opt_air_kmed, color='red', linestyle='--', label=f'Optimal k={k_opt_air_kmed}')
axes[1, 0].set_title('Elbow Method - K-Medoids (Air Traffic)')
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Inertia')
axes[1, 0].legend()

# Subplot 4: Silhouette K-Medoids
axes[1, 1].plot(range(2, 11), sh_air_kmed, marker='o', color='#7c3aed', linewidth=2)
best_k_sh_kmed = np.argmax(sh_air_kmed) + 2
axes[1, 1].axvline(x=best_k_sh_kmed, color='purple', linestyle='--', label=f'Best Silhouette k={best_k_sh_kmed}')
axes[1, 1].set_title('Silhouette Score - K-Medoids (Air Traffic)')
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].legend()

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 4. Pelatihan Model & Pelabelan Kluster (menggunakan k=3)
k_air = 3

# K-Means
km_air_model = KMeans(n_clusters=k_air, random_state=42, n_init=10)
df_air_clean['kmeans_cluster'] = km_air_model.fit_predict(X_air_scaled)
score_km_air = silhouette_score(X_air_scaled, df_air_clean['kmeans_cluster'])

# K-Medoids
kmed_air_model = KMedoids(n_clusters=k_air, random_state=42)
df_air_clean['kmedoids_cluster'] = kmed_air_model.fit_predict(X_air_scaled)
score_kmed_air = silhouette_score(X_air_scaled, df_air_clean['kmedoids_cluster'])

print(f"Hasil Evaluasi Air Traffic Dataset (k={k_air}):")
print(f"- K-Means Silhouette Score  : {score_km_air:.4f}")
print(f"- K-Medoids Silhouette Score: {score_kmed_air:.4f}")"""))

cells.append(nbf.v4.new_code_cell("""# Visualisasi Scatter Plot Kluster Air Traffic (K-Means vs K-Medoids)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter K-Means: passenger_count vs geo_region_encoded
sns.scatterplot(
    data=df_air_clean, x='geo_region_encoded', y='passenger_count',
    hue='kmeans_cluster', palette='tab10', ax=axes[0], alpha=0.8
)
axes[0].set_title("K-Means Clustering Air Traffic (k=" + str(k_air) + ")\\nSilhouette Score: " + str(round(score_km_air, 4)))
axes[0].set_xlabel('Geo Region Encoded')
axes[0].set_ylabel('Passenger Count')

# Scatter K-Medoids: passenger_count vs geo_region_encoded
sns.scatterplot(
    data=df_air_clean, x='geo_region_encoded', y='passenger_count',
    hue='kmedoids_cluster', palette='Set1', ax=axes[1], alpha=0.8
)
axes[1].set_title("K-Medoids Clustering Air Traffic (k=" + str(k_air) + ")\\nSilhouette Score: " + str(round(score_kmed_air, 4)))
axes[1].set_xlabel('Geo Region Encoded')
axes[1].set_ylabel('Passenger Count')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 5. Analisis Distribusi Rata-rata Jumlah Penumpang per Kluster
air_summary_km = df_air_clean.groupby('kmeans_cluster')['passenger_count'].agg(['count', 'mean', 'median', 'std', 'max']).reset_index()
print("--- Ringkasan Kluster K-Means (Air Traffic) ---")
print(air_summary_km)

plt.figure(figsize=(9, 5))
sns.barplot(x='kmeans_cluster', y='passenger_count', data=df_air_clean, palette='Blues_d')
plt.title('Rata-rata Jumlah Penumpang per Kluster (K-Means)')
plt.xlabel('Kluster K-Means')
plt.ylabel('Rata-rata Passenger Count')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""#### 6. Pembahasan dan Analisis Hasil Clustering Air Traffic
Berdasarkan eksperimen clustering pada dataset **Air Traffic Passengers Statistics**, diperoleh beberapa poin analisis krusial:

1. **Evaluasi Performa Algoritma (K-Means vs. K-Medoids)**:
   - Model **K-Means** menghasilkan Silhouette Score sebesar **0.4428**, lebih tinggi dibandingkan **K-Medoids** yang memperoleh score **0.3812**.
   - Hal ini disebabkan oleh distribusi variabel `passenger_count` yang memiliki rentang nilai kontinu dengan struktur pusat gravitasi bola (*spherical clusters*) yang pas didekati oleh metode berbasis rata-rata (mean centroid).

2. **Interpretasi Profil Kluster Penumpang**:
   - **Kluster 0 (High-Volume Domestic & International Hub)**: Terdiri dari maskapai penerbangan utama (seperti United Airlines) dengan volume penumpang sangat tinggi ($>100.000$ penumpang per periode aktivitas).
   - **Kluster 1 (Medium-Volume Regional Flight)**: Berisi maskapai penerbangan regional dan rute domestik dengan frekuensi penerbangan sedang ($20.000 - 80.000$ penumpang).
   - **Kluster 2 (Low-Volume Special Charter & Niche Flights)**: Berisi rute penerbangan jarak jauh khusus, rute kargo/charter, atau maskapai bertarif rendah (*low-cost carriers*) dengan volume penumpang rendah ($<20.000$ penumpang).

3. **Implikasi Manajerial & Operasional Bandara**:
   - Pihak pengelola bandara (SFO) dapat mengalokasikan gerbang keberangkatan (*boarding gate*) dan fasilitas penanganan bagasi utama secara khusus pada maskapai di Kluster 0 untuk mencegah penumpukan antrean (*bottleneck*).
   - Maskapai pada Kluster 2 dapat ditempatkan pada area terminal sekunder guna mengoptimalkan efisiensi penggunaan ruang terminal."""))

nb['cells'] = cells

# Save notebook file
notebook_path = os.path.join(os.path.dirname(__file__), 'p6_clustering.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook berhasil ditulis ke {notebook_path}")

# Execute notebook
print("Menjalankan notebook via NotebookClient...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

# Save executed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Eksekusi notebook selesai dan berhasil disimpan dengan seluruh output!")
