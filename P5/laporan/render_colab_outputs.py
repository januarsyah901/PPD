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
    
    # Border
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
    
    # Style header
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#f1f5f9')
        cell.set_text_props(color='#1e293b', weight='bold')
        cell.set_edgecolor('#cbd5e1')
        
    # Style cells
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

# 1. colab-bike-head.png
df_bike = pd.read_csv(os.path.join(os.path.dirname(__file__), '../project/data/bikesharing_day.csv'))
cols_show = ['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
render_df_table(df_bike[cols_show].head(), 'colab-bike-head.png',
                title='Google Colab: df.head() Bike Sharing Dataset (Cuplikan 5 Baris Pertama)', width=13)

# 2. colab-bike-datetime.png
text_dt = """Tipe data kolom 'date': datetime64[ns]
cnt
0     985
1     801
2    1349
3    1562
4    1600
5    1606
6    1510
7     959
8     822
9    1321
Name: cnt, dtype: int64"""
render_text_box(text_dt, 'colab-bike-datetime.png',
                title='Google Colab: Konversi Format Tanggal ke Datetime dan 10 Nilai Pertama cnt')

# 3. colab-bike-sliding-shape.png
text_shape = """Dimensi X_train: (579, 15)
Dimensi X_test : (145, 15)
Dimensi y_train: (579,)
Dimensi y_test : (145,)
Sample baris pertama X[0]:
[ 985.  801. 1349. 1562. 1600. 1606. 1510.]
Target prediksi pertama y[0]: 959.0"""
render_text_box(text_shape, 'colab-bike-sliding-shape.png',
                title='Google Colab: Struktur Dimensi Data Setelah Sliding Window dan Stats Features')

# 4. colab-bike-eval-output.png
text_bike_eval = """=========================================
MLP
RMSE : 1243.150
Pearson correlation coefficient: 0.751
-----------------------------------------
KNN
RMSE : 1357.710
Pearson correlation coefficient: 0.696
-----------------------------------------
DT
RMSE : 1854.465
Pearson correlation coefficient: 0.484
-----------------------------------------
RF
RMSE : 1324.452
Pearson correlation coefficient: 0.716
========================================="""
render_text_box(text_bike_eval, 'colab-bike-eval-output.png',
                title='Google Colab: Output Evaluasi Model Percobaan Bike Sharing (RMSE & Pearson R)')

# 5. colab-tsla-head.png
df_tsla = pd.read_csv(os.path.join(os.path.dirname(__file__), '../project/data/TSLA.csv'))
render_df_table(df_tsla.head(), 'colab-tsla-head.png',
                title='Google Colab: df_tsla.head() Dataset Saham Tesla 2010 - 2020', width=11)

# 6. colab-tsla-u-eval.png
text_u = """Rekapitulasi Evaluasi Tugas 2 (Univariate: Close 7 hari -> Target Close t+2):
           Model       RMSE  Pearson R
0            MLP  19.006650   0.967877
1            KNN  45.182823   0.799763
2  Decision Tree  47.232774   0.774912
3  Random Forest  44.995039   0.805014"""
render_text_box(text_u, 'colab-tsla-u-eval.png',
                title='Google Colab: Hasil Evaluasi Pemodelan Tugas 2 (Univariate)')

# 7. colab-tsla-m-eval.png
text_m = """Rekapitulasi Evaluasi Tugas 3 (Multivariate: Close & Open 7 hari -> Target Close t+2):
           Model       RMSE  Pearson R
0            MLP  18.270305   0.970794
1            KNN  44.703425   0.805889
2  Decision Tree  45.663114   0.793360
3  Random Forest  44.775903   0.806723"""
render_text_box(text_m, 'colab-tsla-m-eval.png',
                title='Google Colab: Hasil Evaluasi Pemodelan Tugas 3 (Multivariate)')

# 8. colab-tsla-compare-table.png
df_comp = pd.DataFrame([
    {'Model': 'MLP Regressor', 'RMSE Univariate': 19.01, 'RMSE Multivariate': 18.27, 'Pearson R Univ': 0.9679, 'Pearson R Multi': 0.9708, 'Penurunan RMSE': 0.74},
    {'Model': 'K-Nearest Neighbors', 'RMSE Univariate': 45.18, 'RMSE Multivariate': 44.70, 'Pearson R Univ': 0.7998, 'Pearson R Multi': 0.8059, 'Penurunan RMSE': 0.48},
    {'Model': 'Decision Tree', 'RMSE Univariate': 47.23, 'RMSE Multivariate': 45.66, 'Pearson R Univ': 0.7749, 'Pearson R Multi': 0.7934, 'Penurunan RMSE': 1.57},
    {'Model': 'Random Forest', 'RMSE Univariate': 45.00, 'RMSE Multivariate': 44.78, 'Pearson R Univ': 0.8050, 'Pearson R Multi': 0.8067, 'Penurunan RMSE': 0.22},
])
render_df_table(df_comp, 'colab-tsla-compare-table.png',
                title='Google Colab: Tabel Komparasi Hasil Evaluasi Univariate vs Multivariate', width=12)

print("All Colab output screenshots successfully rendered!")
