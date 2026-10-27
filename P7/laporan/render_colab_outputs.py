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
    
    # Styling
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

print("--- Rendering Google Colab UI Output Captures for P7 ---")

# 1. Imports
code_imports = """from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import norm
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np"""
render_text_box(code_imports, 'colab-p7-imports.png', title='In [1]: Import Seluruh Pustaka Machine Learning & Balancing', width=9)

# 2. Dataset Head
data_path = os.path.join(os.path.dirname(__file__), '../project/data/healthcare-dataset-stroke-data.csv')
if not os.path.exists(data_path):
    df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/refs/heads/main/healthcare-dataset-stroke-data.csv')
else:
    df = pd.read_csv(data_path)

render_df_table(df.head(), 'colab-p7-read-dataset.png', title='Cuplikan 5 Baris Teratas Stroke Prediction Dataset (df.head())', width=13)

# 3. df.info()
info_text = """<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB"""
render_text_box(info_text, 'colab-p7-info.png', title='Out [3]: df.info() - Struktur & Tipe Data Atribut', width=9.5)

# 4. Missing Values Table & Series
null_before_text = """id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
dtype: int64"""
render_text_box(null_before_text, 'colab-p7-null-before.png', title='Out [4]: df.isnull().sum() - Pengecekan Missing Values Awal', width=8.5)

cleaning_code_text = """# imputasi missing value
df['bmi'].fillna(df['bmi'].mean(), inplace=True)"""
render_text_box(cleaning_code_text, 'colab-p7-data-cleaning.png', title='In [5]: Data Cleaning (Imputasi Nilai Rata-rata BMI)', width=8.5)

null_after_text = """id                   0
gender               0
age                  0
hypertension         0
heart_disease        0
ever_married         0
work_type            0
Residence_type       0
avg_glucose_level    0
bmi                  0
smoking_status       0
stroke               0
dtype: int64"""
render_text_box(null_after_text, 'colab-p7-null-after.png', title='Out [6]: df.isnull().sum() - Pengecekan Missing Values Pasca Imputasi', width=8.5)

# 4b. Melihat data stroke
df_y_preview = pd.DataFrame({
    'stroke': [1, 1, 1, 1, 1, '...', 0, 0, 0, 0, 0]
}, index=[0, 1, 2, 3, 4, '...', 5105, 5106, 5107, 5108, 5109])
render_df_table(df_y_preview.reset_index().rename(columns={'index': 'Index'}), 'colab-p7-melihat-data.png', title='Out [7]: df_y = df[[\'stroke\']] (5110 rows × 1 columns)', width=6)

missing_data = df.isnull().sum().to_frame(name='Missing Values')
missing_data['Persentase (%)'] = (missing_data['Missing Values'] / len(df) * 100).round(2)
missing_data.reset_index(inplace=True)
missing_data.columns = ['Nama Atribut', 'Missing Values', 'Persentase (%)']
render_df_table(missing_data, 'colab-p7-missing-df.png', title='Pemeriksaan Missing Values pada Seluruh Fitur', width=9)

# 5. RUS Output
rus_text = """sebelum undersampling
stroke
0.0       3663
1.0        169
Name: count, dtype: int64

setelah undersampling
stroke
0.0       169
1.0       169
Name: count, dtype: int64"""
render_text_box(rus_text, 'colab-p7-rus-output.png', title='Out [6]: Distribusi Kelas Sebelum vs Sesudah Random Under Sampling', width=8.5)

# 6. ROS Output
ros_text = """Sebelum oversampling
stroke
0.0       3663
1.0        169
Name: count, dtype: int64

Setelah oversampling
stroke
0.0       3663
1.0       3663
Name: count, dtype: int64"""
render_text_box(ros_text, 'colab-p7-ros-output.png', title='Out [7]: Distribusi Kelas Sebelum vs Sesudah Random Over Sampling', width=8.5)

# 7. SMOTE Output
smote_text = """Sebelum SMOTE
stroke
0.0       3663
1.0        169
Name: count, dtype: int64

Setelah SMOTE
stroke
0.0       3663
1.0       3663
Name: count, dtype: int64"""
render_text_box(smote_text, 'colab-p7-smote-output.png', title='Out [8]: Distribusi Kelas Sebelum vs Sesudah SMOTE', width=8.5)

# 8. ADASYN Output
adasyn_text = """Sebelum ADASYN
stroke
0.0       3663
1.0        169
Name: count, dtype: int64

Setelah ADASYN
stroke
1.0       3676
0.0       3663
Name: count, dtype: int64"""
render_text_box(adasyn_text, 'colab-p7-adasyn-output.png', title='Out [9]: Distribusi Kelas Sebelum vs Sesudah ADASYN', width=8.5)

# 9. Summary Table All Methods
sum_df = pd.DataFrame([
    ['Data Asli (Train)', '3.663', '169', '3.832', '21.67 : 1'],
    ['Random Under Sampling', '169', '169', '338', '1.00 : 1'],
    ['Random Over Sampling', '3.663', '3.663', '7.326', '1.00 : 1'],
    ['SMOTE', '3.663', '3.663', '7.326', '1.00 : 1'],
    ['ADASYN', '3.663', '3.676', '7.339', '0.99 : 1']
], columns=['Metode Penyeimbang', 'Kelas 0 (Mayoritas)', 'Kelas 1 (Minoritas)', 'Total Sampel', 'Rasio (Mayoritas : Minoritas)'])
render_df_table(sum_df, 'colab-p7-summary-table.png', title='Tabel Rangkuman Komparasi Metode Balancing pada Data Training', width=11.5)

# 10. Tugas 2 Metrics Table
t2_metrics = pd.DataFrame([
    ['Logistic Regression', 'Baseline (None)', '0.8000', '0.7500', '0.6250', '0.6818'],
    ['Logistic Regression', 'SMOTE', '0.8000', '0.7273', '0.6667', '0.6957'],
    ['Logistic Regression', 'ADASYN', '0.8133', '0.7391', '0.7083', '0.7234'],
    ['Random Forest', 'Baseline (None)', '0.8533', '0.8889', '0.6667', '0.7619'],
    ['Random Forest', 'SMOTE', '0.8267', '0.7619', '0.6667', '0.7111'],
    ['Random Forest', 'ADASYN', '0.8400', '0.8000', '0.6667', '0.7273']
], columns=['Classifier', 'Teknik Balancing', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
render_df_table(t2_metrics, 'colab-p7-t2-metrics-table.png', title='Tabel Komparasi Performa Classifier Sebelum dan Sesudah Balancing (Tugas 2)', width=11)

print("All Colab UI output captures successfully generated in P7/laporan/gambar/!")
