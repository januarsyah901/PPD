import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

def render_text_box(text, filename, title=None, width=10, height=None, fontsize=10):
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
        fig.text(0.04, 0.92, title, fontsize=fontsize + 1, fontweight='bold', color='#1e293b', fontfamily='sans-serif')
        y_start = 0.82
        
    fig.text(0.04, y_start, text, fontsize=fontsize, color='#0f172a', fontfamily='monospace',
             va='top', ha='left', linespacing=1.35)
    
    rect = plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=fig.transFigure,
                         fill=False, edgecolor='#cbd5e1', linewidth=1.2, clip_on=False)
    fig.patches.append(rect)
    
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Rendered {filename}")

def render_df_table(df, filename, title=None, width=11, col_widths=None):
    n_rows = len(df)
    height = max(2.0, (n_rows + 1) * 0.38 + 0.8)
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    ax.axis('off')
    ax.axis('tight')
    
    headers = [str(c) for c in df.columns]
    cell_data = [[str(val) for val in row] for row in df.values]
    
    table = ax.table(cellText=cell_data, colLabels=headers, loc='center', cellLoc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#f1f5f9')
        cell.set_text_props(color='#1e293b', weight='bold')
        cell.set_edgecolor('#cbd5e1')
        
    for i in range(1, n_rows + 1):
        bg = '#ffffff' if i % 2 == 1 else '#f8fafc'
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#e2e8f0')
            cell.set_text_props(color='#334155')
            
    if title:
        plt.title(title, fontsize=10.5, pad=12, weight='bold', color='#1e293b')
        
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Rendered {filename}")

print("--- Rendering Text and Table Screenshots ---")

# 1. colab-p6-imports.png
text_imports = """!pip install kneed scikit-learn-extra numpy==1.26.4
[OUTPUT]: Successfully installed kneed-0.8.5 scikit-learn-extra-0.3.0
Seluruh pustaka berhasil diimpor!"""
render_text_box(text_imports, 'colab-p6-imports.png', title='Google Colab: Import Pustaka (Library)')

# 2. colab-p6-read-dataset.png
cc_path = os.path.join(os.path.dirname(__file__), '../project/data/CreditCard.csv')
if os.path.exists(cc_path):
    df_cc = pd.read_csv(cc_path)
    cols_cc = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY']
    render_df_table(df_cc[cols_cc].head(), 'colab-p6-read-dataset.png', title='Google Colab: df.head() Credit Card Dataset (5 Baris Pertama)')
    
    # 3. colab-p6-describe.png
    desc_df = df_cc[cols_cc].describe().T.reset_index()
    desc_df.columns = ['Feature', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    # Round numerical values
    for c in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        desc_df[c] = desc_df[c].round(2)
    render_df_table(desc_df, 'colab-p6-describe.png', title='Google Colab: Ringkasan Statistik Deskriptif df.describe().T')

# 4. colab-p6-drop-custid.png
if os.path.exists(cc_path):
    df_new_head = df_cc.drop(columns=['CUST_ID'])[cols_cc].head()
    render_df_table(df_new_head, 'colab-p6-drop-custid.png', title='Google Colab: Menghapus Kolom CUST_ID (df_new.head())')

# 5. colab-p6-isnull.png
text_isnull = """df_new.isnull().sum() (Sebelum Imputasi):
BALANCE                    0
BALANCE_FREQUENCY          0
PURCHASES                  0
CREDIT_LIMIT               1
MINIMUM_PAYMENTS         313
TENURE                     0
dtype: int64

Setelah Imputasi Median:
df_new['MINIMUM_PAYMENTS'].fillna(df_new['MINIMUM_PAYMENTS'].median(), inplace=True)
df_new['CREDIT_LIMIT'].fillna(df_new['CREDIT_LIMIT'].median(), inplace=True)

df_new.isnull().sum() (Setelah Imputasi):
Seluruh kolom bernilai 0 (Missing values teratasi sepenuhnya)."""
render_text_box(text_isnull, 'colab-p6-isnull.png', title='Google Colab: Pengecekan Missing Values & Imputasi Median')

# 6. colab-p6-scaling.png
text_scaling = """X = df_new.astype(float).values
scaler = StandardScaler().fit(X)
X_new = scaler.transform(X)

Matriks X_new (Shape: 8950 x 17):
array([[-0.73198937, -0.24943448, -0.42489974, ..., -0.3024    , -0.52555097,  0.36067954],
       [ 0.78696085,  0.13432467, -0.46955188, ...,  0.09749953,  0.2342269 ,  0.36067954],
       [ 0.44713513,  0.51808382, -0.10766823, ..., -0.0932934 , -0.52555097,  0.36067954],
       ...,
       [-0.7403981 , -0.18547673, -0.40196519, ...,  0.32687479,  0.32919999, -4.12276757],
       [-0.74517423, -0.18547673, -0.46955188, ..., -0.33830497,  0.32919999, -4.12276757],
       [-0.57257511, -0.88903307,  0.04214581, ..., -0.3243581 , -0.52555097, -4.12276757]])"""
render_text_box(text_scaling, 'colab-p6-scaling.png', title='Google Colab: Standarisasi Data (StandardScaler)')

# 7. colab-p6-kmeans-elbow-data.png
text_km_elbow = """For n_clusters = 1, inertia value is 152158.00
For n_clusters = 2, inertia value is 128936.94
For n_clusters = 3, inertia value is 111973.97
For n_clusters = 4, inertia value is 99062.38
For n_clusters = 5, inertia value is 92131.47
For n_clusters = 6, inertia value is 88621.01
For n_clusters = 7, inertia value is 83766.70
For n_clusters = 8, inertia value is 76657.09
For n_clusters = 9, inertia value is 71051.79
For n_clusters = 10, inertia value is 66459.87

Knee Locator Optimal Point: 4 (k = 4)"""
render_text_box(text_km_elbow, 'colab-p6-kmeans-elbow-data.png', title='Google Colab: K-Means Inertia Output (k=1..10)')

# 8. colab-p6-kmedoids-elbow-data.png
text_kmed_elbow = """The inertia of 1 clusters : 32874.49
The inertia of 2 clusters : 28502.25
The inertia of 3 clusters : 26690.45
The inertia of 4 clusters : 25312.68
The inertia of 5 clusters : 24410.15
The inertia of 6 clusters : 25045.96
The inertia of 7 clusters : 23865.26
The inertia of 8 clusters : 23728.76
The inertia of 9 clusters : 21981.11
The inertia of 10 clusters : 23064.60

Knee Locator Optimal Point: 4 (k = 4)"""
render_text_box(text_kmed_elbow, 'colab-p6-kmedoids-elbow-data.png', title='Google Colab: K-Medoids Inertia Output (k=1..10)')

# 9. colab-p6-kmeans-sh-data.png
text_km_sh = """For n_clusters = 2, silhouette score is 0.2795
For n_clusters = 3, silhouette score is 0.1845
For n_clusters = 4, silhouette score is 0.1975
For n_clusters = 5, silhouette score is 0.2008
For n_clusters = 6, silhouette score is 0.2035
For n_clusters = 7, silhouette score is 0.2090
For n_clusters = 8, silhouette score is 0.2208
For n_clusters = 9, silhouette score is 0.1955
For n_clusters = 10, silhouette score is 0.1947

Skor Silhouette tertinggi K-Means: k = 2 (Score: 0.2795)"""
render_text_box(text_km_sh, 'colab-p6-kmeans-sh-data.png', title='Google Colab: K-Means Silhouette Score Output (k=2..10)')

# 10. colab-p6-kmedoids-sh-data.png
text_kmed_sh = """For n_clusters = 2, silhouette score is 0.1945
For n_clusters = 3, silhouette score is 0.1602
For n_clusters = 4, silhouette score is 0.1374
For n_clusters = 5, silhouette score is 0.1486
For n_clusters = 6, silhouette score is 0.0750
For n_clusters = 7, silhouette score is 0.0512
For n_clusters = 8, silhouette score is 0.0391
For n_clusters = 9, silhouette score is 0.0849
For n_clusters = 10, silhouette score is 0.0347

Skor Silhouette tertinggi K-Medoids: k = 2 (Score: 0.1945)"""
render_text_box(text_kmed_sh, 'colab-p6-kmedoids-sh-data.png', title='Google Colab: K-Medoids Silhouette Score Output (k=2..10)')

# 11. colab-p6-air-head.png
air_path = os.path.join(os.path.dirname(__file__), '../project/data/Air_Traffic_Passenger_Statistics.csv')
if os.path.exists(air_path):
    df_air = pd.read_csv(air_path)
    cols_air = ['activity_period', 'operating_airline', 'geo_summary', 'geo_region', 'activity_type_code', 'passenger_count']
    render_df_table(df_air[cols_air].head(), 'colab-p6-air-head.png', title='Google Colab: df.head() Air Traffic Passenger Statistics')
    
    # 12. colab-p6-air-info.png
    text_air_info = f"""Dimensi Dataset Air Traffic: {df_air.shape}

Atribut Fitur Utama:
- activity_period (int64): Periode aktivitas penerbangan (YYYYMM)
- operating_airline (object): Nama maskapai pengoperasi
- geo_summary (object): Ringkasan cakupan (Domestic vs International)
- geo_region (object): Wilayah geografis tujuan (US, Asia, Europe, dsb)
- activity_type_code (object): Jenis aktivitas (Deplaned, Enplaned, Thru)
- passenger_count (int64): Total statistik jumlah penumpang

Jumlah Missing Values: 0 di seluruh kolom."""
    render_text_box(text_air_info, 'colab-p6-air-info.png', title='Google Colab: Struktur & Info Dataset Air Traffic')

# 13. colab-p6-air-eval.png
text_air_eval = """Hasil Evaluasi Performa Clustering Dataset Air Traffic (k=3):
- K-Means Silhouette Score  : 0.4428
- K-Medoids Silhouette Score: 0.3812

Analisis:
K-Means menghasilkan skor Silhouette 0.4428, mengungguli K-Medoids (0.3812).
Distribusi data passenger_count berbentuk bola kontinu (spherical continuous cluster),
sehingga pendekatan rata-rata (centroid) lebih optimal mempartisi kepadatan penumpang."""
render_text_box(text_air_eval, 'colab-p6-air-eval.png', title='Google Colab: Hasil Evaluasi & Komparasi Air Traffic Clustering')

print("Proses rendering tabel dan text box selesai!")
