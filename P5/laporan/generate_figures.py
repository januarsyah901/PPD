import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kurtosis, skew
from math import sqrt

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2 = list(input_data[i])
        min_v = float(np.min(inp2))
        max_v = float(np.max(inp2))
        diff = (max_v - min_v)
        std = float(np.std(inp2))
        mean = float(np.mean(inp2))
        median = float(np.median(inp2))
        kurt = float(kurtosis(inp2))
        sk = float(skew(inp2))
        inp2.extend([min_v, max_v, diff, std, mean, median, kurt, sk])
        inp.append(inp2)
    return np.array(inp)

# =====================================================================
# 1. PERCOBAAN: BIKE SHARING FORECASTING
# =====================================================================
print("--- 1. Generating Bike Sharing Figures ---")
bike_path = os.path.join(os.path.dirname(__file__), '../project/data/bikesharing_day.csv')
if not os.path.exists(bike_path):
    bike_path = 'https://raw.githubusercontent.com/ganjar87/data_science_practice/main/bikesharing_day.csv'

df_bike = pd.read_csv(bike_path)
df_bike['date'] = pd.to_datetime(df_bike['dteday'])

# 1.1 Time series plot of Bike Sharing cnt
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df_bike['date'], df_bike['cnt'], color='#1f77b4', linewidth=1.5, label='Total Peminjaman Sepeda (cnt)')
ax.set_title('Tren Deret Waktu Peminjaman Sepeda Harian (Bike Sharing Dataset 2011-2012)', fontsize=12, pad=10)
ax.set_xlabel('Tanggal Observasi', fontsize=10)
ax.set_ylabel('Jumlah Peminjaman Sepeda', fontsize=10)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'bike_sharing_timeseries.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved bike_sharing_timeseries.png")

# Data preparation
df_x_bike = df_bike[['cnt', 'cnt']]
in_seq_bike = df_x_bike.astype(float).values
n_in_b, n_out_b = 7, 1
X_b, y_b = split_sequences(in_seq_bike, n_in_b, n_out_b)
n_input_b = X_b.shape[1] * X_b.shape[2]
X_b = X_b.reshape((X_b.shape[0], n_input_b))

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_b, y_b, test_size=0.2, random_state=42, shuffle=False
)
X_b_train = stats_features(X_b_train)
X_b_test = stats_features(X_b_test)

# Train models
models_bike = {
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

preds_bike = {}
results_bike = []

for name, model in models_bike.items():
    model.fit(X_b_train, y_b_train)
    pred = np.round(model.predict(X_b_test), 0)
    preds_bike[name] = pred
    rmse = sqrt(mean_squared_error(y_b_test, pred))
    corr, _ = pearsonr(y_b_test, pred)
    results_bike.append({
        'Model': name,
        'RMSE': rmse,
        'Pearson R': corr
    })

df_res_bike = pd.DataFrame(results_bike)
print("Bike Sharing Results Summary:")
print(df_res_bike)

# 1.2 Comparison plot: Real data vs Models
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_b_test, label='Real Data', color='#111827', linewidth=2.0)
ax.plot(preds_bike['MLP'], label='MLP (Neural Network)', color='#2563eb', linestyle='--', linewidth=1.6)
ax.plot(preds_bike['KNN'], label='KNN', color='#059669', linestyle=':', linewidth=1.5)
ax.plot(preds_bike['Decision Tree'], label='Decision Tree', color='#d97706', linestyle='-.', linewidth=1.3)
ax.plot(preds_bike['Random Forest'], label='Random Forest', color='#dc2626', linestyle='-', alpha=0.8, linewidth=1.4)

ax.set_title('Perbandingan Prediksi Model Supervised Learning terhadap Data Uji Bike Sharing', fontsize=12, pad=10)
ax.set_xlabel('Indeks Data Uji (Hari)', fontsize=10)
ax.set_ylabel('Jumlah Peminjaman Sepeda (cnt)', fontsize=10)
ax.legend(loc='upper right', frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'bike_sharing_predictions.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved bike_sharing_predictions.png")

# 1.3 Bar chart comparison of metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
b_names = [r['Model'] for r in results_bike]
b_rmses = [r['RMSE'] for r in results_bike]
b_corrs = [r['Pearson R'] for r in results_bike]

bars1 = ax1.bar(b_names, b_rmses, color=['#2563eb', '#059669', '#d97706', '#dc2626'], edgecolor='black', width=0.6)
ax1.set_title('Root Mean Squared Error (RMSE)\n(Nilai lebih rendah lebih baik)', fontsize=10.5)
ax1.set_ylabel('RMSE')
for b in bars1:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 25, f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.set_ylim(0, max(b_rmses) * 1.18)

bars2 = ax2.bar(b_names, b_corrs, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'], edgecolor='black', width=0.6)
ax2.set_title('Koefisien Korelasi Pearson (R)\n(Nilai mendekati 1 lebih baik)', fontsize=10.5)
ax2.set_ylabel('Korelasi Pearson (R)')
for b in bars2:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax2.set_ylim(0, 1.0)

plt.suptitle('Evaluasi Metrik Performa Model Forecasting (Bike Sharing Dataset)', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'bike_sharing_eval_comp.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved bike_sharing_eval_comp.png")

# 1.4 Render summary table as image
fig, ax = plt.subplots(figsize=(7.5, 2.6))
ax.axis('off')
ax.axis('tight')
tbl_data_b = [
    ['Algoritma Forecasting', 'RMSE', 'Korelasi Pearson (R)'],
    ['Multilayer Perceptron (MLP)', f"{results_bike[0]['RMSE']:.3f}", f"{results_bike[0]['Pearson R']:.3f}"],
    ['K-Nearest Neighbors (KNN)', f"{results_bike[1]['RMSE']:.3f}", f"{results_bike[1]['Pearson R']:.3f}"],
    ['Decision Tree Regressor', f"{results_bike[2]['RMSE']:.3f}", f"{results_bike[2]['Pearson R']:.3f}"],
    ['Random Forest Regressor', f"{results_bike[3]['RMSE']:.3f}", f"{results_bike[3]['Pearson R']:.3f}"]
]
tbl_b = ax.table(cellText=tbl_data_b, loc='center', cellLoc='center')
tbl_b.auto_set_font_size(False)
tbl_b.set_fontsize(9.5)
tbl_b.scale(1.2, 1.8)
for i in range(3):
    cell = tbl_b[(0, i)]
    cell.set_facecolor('#003366')
    cell.set_text_props(color='white', weight='bold')
for row in range(1, 5):
    for col in range(3):
        cell = tbl_b[(row, col)]
        if row % 2 == 1:
            cell.set_facecolor('#f0f4f8')
        if row == 1: # Best model MLP
            cell.set_text_props(weight='bold')
plt.title('Tabel Rekapitulasi Evaluasi Model Forecasting Bike Sharing', fontsize=11, pad=12, weight='bold')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'bike_sharing_summary_table.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print("Saved bike_sharing_summary_table.png")


# =====================================================================
# 2. TUGAS DAN ANALISIS: TESLA STOCK PRICE FORECASTING
# =====================================================================
print("\n--- 2. Generating Tesla Stock Figures ---")
tsla_path = os.path.join(os.path.dirname(__file__), '../project/data/TSLA.csv')
if not os.path.exists(tsla_path):
    tsla_path = 'https://raw.githubusercontent.com/vidvatabuch/Tesla-Stock-Analysis/master/TSLA.csv'

df_tsla = pd.read_csv(tsla_path)
df_tsla['Date'] = pd.to_datetime(df_tsla['Date'])

# 2.1 Time series plot of Tesla Stock (Close and Open)
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df_tsla['Date'], df_tsla['Close'], color='#dc2626', linewidth=1.5, label='Close Price (USD)')
ax.plot(df_tsla['Date'], df_tsla['Open'], color='#2563eb', linewidth=1.2, alpha=0.7, linestyle='--', label='Open Price (USD)')
ax.set_title('Pergerakan Harga Saham Tesla (TSLA) Periode 2010 - 2020', fontsize=12, pad=10)
ax.set_xlabel('Tahun', fontsize=10)
ax.set_ylabel('Harga Saham (USD)', fontsize=10)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_timeseries.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_timeseries.png")

# --- TUGAS 2: UNIVARIATE (Close 7 hari -> Target Close t+2) ---
seq_u = df_tsla[['Close', 'Close']].astype(float).values
n_in_tsla, n_out_tsla = 7, 2
X_u, y_u = split_sequences(seq_u, n_in_tsla, n_out_tsla)
X_u = X_u.reshape((X_u.shape[0], X_u.shape[1] * X_u.shape[2]))

X_u_tr, X_u_te, y_u_tr, y_u_te = train_test_split(
    X_u, y_u, test_size=0.2, random_state=42, shuffle=False
)
X_u_tr = stats_features(X_u_tr)
X_u_te = stats_features(X_u_te)

models_tsla_u = {
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

preds_tsla_u = {}
results_tsla_u = []

for name, model in models_tsla_u.items():
    model.fit(X_u_tr, y_u_tr)
    pred = np.round(model.predict(X_u_te), 2)
    preds_tsla_u[name] = pred
    rmse = sqrt(mean_squared_error(y_u_te, pred))
    corr, _ = pearsonr(y_u_te, pred)
    results_tsla_u.append({
        'Model': name,
        'RMSE': rmse,
        'Pearson R': corr
    })

print("Tugas 2 Results Summary:")
print(pd.DataFrame(results_tsla_u))

# 2.2 Prediction comparison Tugas 2
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_u_te, label='Harga Riil (Actual Close t+2)', color='#111827', linewidth=2.0)
ax.plot(preds_tsla_u['MLP'], label='MLP (RMSE: 19.01, R: 0.968)', color='#2563eb', linestyle='--', linewidth=1.6)
ax.plot(preds_tsla_u['KNN'], label='KNN (RMSE: 45.18, R: 0.800)', color='#059669', linestyle=':', linewidth=1.4)
ax.plot(preds_tsla_u['Decision Tree'], label='Decision Tree (RMSE: 47.23, R: 0.775)', color='#d97706', linestyle='-.', linewidth=1.2)
ax.plot(preds_tsla_u['Random Forest'], label='Random Forest (RMSE: 45.00, R: 0.805)', color='#dc2626', linestyle='-', alpha=0.8, linewidth=1.4)

ax.set_title('Tugas 2: Prediksi Harga Saham Tesla 2 Hari ke Depan Berbasis Close 7 Hari Sebelumnya (Univariate)', fontsize=11.5, pad=10)
ax.set_xlabel('Indeks Sampel Uji (Hari)', fontsize=10)
ax.set_ylabel('Harga Close (USD)', fontsize=10)
ax.legend(loc='upper left', frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_u_predictions.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_u_predictions.png")


# --- TUGAS 3: MULTIVARIATE (Close & Open 7 hari -> Target Close t+2) ---
seq_m = df_tsla[['Close', 'Open', 'Close']].astype(float).values
X_m, y_m = split_sequences(seq_m, n_in_tsla, n_out_tsla)
X_m = X_m.reshape((X_m.shape[0], X_m.shape[1] * X_m.shape[2]))

X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42, shuffle=False
)
X_m_tr = stats_features(X_m_tr)
X_m_te = stats_features(X_m_te)

models_tsla_m = {
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

preds_tsla_m = {}
results_tsla_m = []

for name, model in models_tsla_m.items():
    model.fit(X_m_tr, y_m_tr)
    pred = np.round(model.predict(X_m_te), 2)
    preds_tsla_m[name] = pred
    rmse = sqrt(mean_squared_error(y_m_te, pred))
    corr, _ = pearsonr(y_m_te, pred)
    results_tsla_m.append({
        'Model': name,
        'RMSE': rmse,
        'Pearson R': corr
    })

print("Tugas 3 Results Summary:")
print(pd.DataFrame(results_tsla_m))

# 2.3 Prediction comparison Tugas 3
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_m_te, label='Harga Riil (Actual Close t+2)', color='#111827', linewidth=2.0)
ax.plot(preds_tsla_m['MLP'], label='MLP (RMSE: 18.27, R: 0.971)', color='#2563eb', linestyle='--', linewidth=1.6)
ax.plot(preds_tsla_m['KNN'], label='KNN (RMSE: 44.70, R: 0.806)', color='#059669', linestyle=':', linewidth=1.4)
ax.plot(preds_tsla_m['Decision Tree'], label='Decision Tree (RMSE: 45.66, R: 0.793)', color='#d97706', linestyle='-.', linewidth=1.2)
ax.plot(preds_tsla_m['Random Forest'], label='Random Forest (RMSE: 44.78, R: 0.807)', color='#dc2626', linestyle='-', alpha=0.8, linewidth=1.4)

ax.set_title('Tugas 3: Prediksi Harga Saham Tesla 2 Hari ke Depan Berbasis Close dan Open 7 Hari Sebelumnya (Multivariate)', fontsize=11.5, pad=10)
ax.set_xlabel('Indeks Sampel Uji (Hari)', fontsize=10)
ax.set_ylabel('Harga Close (USD)', fontsize=10)
ax.legend(loc='upper left', frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_m_predictions.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_m_predictions.png")


# 2.4 Comparison between Univariate and Multivariate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
m_names = [r['Model'] for r in results_tsla_u]
x_pos = np.arange(len(m_names))
width = 0.35

# RMSE Comparison
rmse_u_vals = [r['RMSE'] for r in results_tsla_u]
rmse_m_vals = [r['RMSE'] for r in results_tsla_m]

b1 = ax1.bar(x_pos - width/2, rmse_u_vals, width=width, label='Univariate (Close saja)', color='#93c5fd', edgecolor='black')
b2 = ax1.bar(x_pos + width/2, rmse_m_vals, width=width, label='Multivariate (Close + Open)', color='#1d4ed8', edgecolor='black')
ax1.set_title('Komparasi Nilai RMSE (USD)\n(Nilai lebih rendah lebih baik)', fontsize=10.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(m_names)
ax1.set_ylabel('RMSE (USD)')
ax1.legend(frameon=True, fontsize=9)

for b in b1:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 0.8, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for b in b2:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 0.8, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_ylim(0, max(rmse_u_vals) * 1.18)

# Pearson R Comparison
corr_u_vals = [r['Pearson R'] for r in results_tsla_u]
corr_m_vals = [r['Pearson R'] for r in results_tsla_m]

b3 = ax2.bar(x_pos - width/2, corr_u_vals, width=width, label='Univariate (Close saja)', color='#a7f3d0', edgecolor='black')
b4 = ax2.bar(x_pos + width/2, corr_m_vals, width=width, label='Multivariate (Close + Open)', color='#047857', edgecolor='black')
ax2.set_title('Komparasi Koefisien Korelasi Pearson (R)\n(Nilai mendekati 1 lebih baik)', fontsize=10.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(m_names)
ax2.set_ylabel('Korelasi Pearson (R)')
ax2.legend(frameon=True, fontsize=9)

for b in b3:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for b in b4:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_ylim(0.65, 1.05)

plt.suptitle('Perbandingan Performa Model Univariate (Tugas 2) vs Multivariate (Tugas 3)', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_eval_comp.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_eval_comp.png")


# 2.5 Extrapolation Analysis (Why Tree Models Plateau vs MLP)
fig, ax = plt.subplots(figsize=(13, 5))
test_idx_subset = np.arange(len(y_m_te) - 100, len(y_m_te)) # last 100 days of test set where TSLA spiked
ax.plot(test_idx_subset, y_m_te[test_idx_subset], label='Harga Riil (Actual Spike)', color='#111827', linewidth=2.5)
ax.plot(test_idx_subset, preds_tsla_m['MLP'][test_idx_subset], label='MLP (Mampu Ekstrapolasi Tren Naik)', color='#2563eb', linewidth=2.0, linestyle='--')
ax.plot(test_idx_subset, preds_tsla_m['Random Forest'][test_idx_subset], label='Random Forest (Terjebak Plafon / Plateau)', color='#dc2626', linewidth=1.8, linestyle=':')
ax.plot(test_idx_subset, preds_tsla_m['KNN'][test_idx_subset], label='KNN (Terjebak Tetangga Maksimum)', color='#059669', linewidth=1.5, linestyle='-.')

ax.set_title('Analisis Fenomena Ekstrapolasi: Lonjakan Harga Saham Tesla pada 100 Sampel Uji Terakhir', fontsize=11.5, pad=10)
ax.set_xlabel('Indeks Sampel Uji', fontsize=10)
ax.set_ylabel('Harga Saham (USD)', fontsize=10)
ax.legend(loc='upper left', frameon=True, fontsize=9.5)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_extrapolation_analysis.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_extrapolation_analysis.png")


# 2.6 Render Tesla Summary Table as image
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis('off')
ax.axis('tight')

tbl_data_t = [
    ['Skenario Forecasting', 'Model ML', 'RMSE (USD)', 'Korelasi Pearson (R)', 'Evaluasi'],
    ['Tugas 2 (Univariate)', 'MLP Regressor', f"{results_tsla_u[0]['RMSE']:.2f}", f"{results_tsla_u[0]['Pearson R']:.4f}", 'Sangat Baik'],
    ['Tugas 2 (Univariate)', 'K-Nearest Neighbors', f"{results_tsla_u[1]['RMSE']:.2f}", f"{results_tsla_u[1]['Pearson R']:.4f}", 'Cukup'],
    ['Tugas 2 (Univariate)', 'Decision Tree', f"{results_tsla_u[2]['RMSE']:.2f}", f"{results_tsla_u[2]['Pearson R']:.4f}", 'Kurang'],
    ['Tugas 2 (Univariate)', 'Random Forest', f"{results_tsla_u[3]['RMSE']:.2f}", f"{results_tsla_u[3]['Pearson R']:.4f}", 'Cukup'],
    ['Tugas 3 (Multivariate)', 'MLP Regressor (Terbaik)', f"{results_tsla_m[0]['RMSE']:.2f}", f"{results_tsla_m[0]['Pearson R']:.4f}", 'Terbaik'],
    ['Tugas 3 (Multivariate)', 'K-Nearest Neighbors', f"{results_tsla_m[1]['RMSE']:.2f}", f"{results_tsla_m[1]['Pearson R']:.4f}", 'Cukup Baik'],
    ['Tugas 3 (Multivariate)', 'Decision Tree', f"{results_tsla_m[2]['RMSE']:.2f}", f"{results_tsla_m[2]['Pearson R']:.4f}", 'Cukup'],
    ['Tugas 3 (Multivariate)', 'Random Forest', f"{results_tsla_m[3]['RMSE']:.2f}", f"{results_tsla_m[3]['Pearson R']:.4f}", 'Cukup Baik']
]

tbl_t = ax.table(cellText=tbl_data_t, loc='center', cellLoc='center')
tbl_t.auto_set_font_size(False)
tbl_t.set_fontsize(9)
tbl_t.scale(1.15, 1.6)

for i in range(5):
    cell = tbl_t[(0, i)]
    cell.set_facecolor('#003366')
    cell.set_text_props(color='white', weight='bold')

for row in range(1, 9):
    for col in range(5):
        cell = tbl_t[(row, col)]
        if row % 2 == 1:
            cell.set_facecolor('#f0f4f8')
        if row == 5: # Best model MLP multivariate
            cell.set_facecolor('#dbeafe')
            cell.set_text_props(weight='bold')

plt.title('Tabel Perbandingan Performa Model Forecasting Saham Tesla (Tugas 2 & 3)', fontsize=11, pad=12, weight='bold')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'tsla_summary_table.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print("Saved tsla_summary_table.png")

print("\nAll figures successfully generated in:", output_dir)
