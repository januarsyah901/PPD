import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

print("--- 1. Generating KC House Data Figures ---")
url_kc = 'https://raw.githubusercontent.com/ganjar87/data_science_practice/main/kc_house_data.csv'
df_kc = pd.read_csv(url_kc)

# 1. Histogram all numeric attributes
fig, ax = plt.subplots(figsize=(12, 10))
df_kc.hist(ax=ax, bins=20, color='#2b5c8f', edgecolor='black', alpha=0.8)
plt.suptitle('Histogram Seluruh Variabel Numerik KC House Data', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'hist_kc_house.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved hist_kc_house.png")

# 2. Correlation heatmap with price
plt.figure(figsize=(10, 8))
numeric_kc = df_kc.select_dtypes(include=[np.number]).drop('id', axis=1)
corr_kc = numeric_kc.corr()
mask = np.triu(np.ones_like(corr_kc, dtype=bool))
sns.heatmap(corr_kc, mask=mask, annot=True, fmt='.2f', cmap='Blues', cbar=True, square=True, annot_kws={'size': 7})
plt.title('Matriks Korelasi Pearson Fitur Numerik KC House Data', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'corr_kc_house.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved corr_kc_house.png")

# Prepare KC data for modelling
df_X_kc = df_kc.drop(['id', 'date', 'price'], axis=1)
df_y_kc = df_kc['price']
X_kc = df_X_kc.astype(float).values
y_kc = df_y_kc.astype(float).values

X_train_kc, X_test_kc, y_train_kc, y_test_kc = train_test_split(X_kc, y_kc, test_size=0.3, random_state=42)

# Linear Regression
reg_kc = LinearRegression()
reg_kc.fit(X_train_kc, y_train_kc)
y_pred_reg_kc = reg_kc.predict(X_test_kc)

# Decision Tree Regression
dt_kc = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_kc.fit(X_train_kc, y_train_kc)
y_pred_dt_kc = dt_kc.predict(X_test_kc)

# Random Forest Regression
rf_kc = RandomForestRegressor(random_state=42)
rf_kc.fit(X_train_kc, y_train_kc)
y_pred_rf_kc = rf_kc.predict(X_test_kc)

# 3. Bar plot comparison Linear Regression (50 samples)
d1 = pd.Series(y_test_kc[:50].ravel())
d2 = pd.Series(y_pred_reg_kc[:50].ravel())
df_comp_lr = pd.concat([d1, d2], keys=['real values', 'predicted values'], axis=1)

plt.figure(figsize=(15, 4))
df_comp_lr.plot(kind='bar', figsize=(15, 4), color=['#1f77b4', '#ff7f0e'], width=0.8)
plt.title('Linear Regression: Perbandingan Nilai Riil dan Prediksi (50 Sampel Awal)', fontsize=12)
plt.xlabel('Sample i')
plt.ylabel('House Price (USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bar_comp_lr.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved bar_comp_lr.png")

# 4. Bar plot comparison Decision Tree (50 samples)
d2_dt = pd.Series(y_pred_dt_kc[:50].ravel())
df_comp_dt = pd.concat([d1, d2_dt], keys=['real values', 'predicted values'], axis=1)

plt.figure(figsize=(15, 4))
df_comp_dt.plot(kind='bar', figsize=(15, 4), color=['#1f77b4', '#ff7f0e'], width=0.8)
plt.title('Decision Tree Regression: Perbandingan Nilai Riil dan Prediksi (50 Sampel Awal)', fontsize=12)
plt.xlabel('Sample i')
plt.ylabel('House Price (USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bar_comp_dt.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved bar_comp_dt.png")

# 5. Bar plot comparison Random Forest (50 samples)
d2_rf = pd.Series(y_pred_rf_kc[:50].ravel())
df_comp_rf = pd.concat([d1, d2_rf], keys=['real values', 'predicted values'], axis=1)

plt.figure(figsize=(15, 4))
df_comp_rf.plot(kind='bar', figsize=(15, 4), color=['#1f77b4', '#ff7f0e'], width=0.8)
plt.title('Random Forest Regression: Perbandingan Nilai Riil dan Prediksi (50 Sampel Awal)', fontsize=12)
plt.xlabel('Sample i')
plt.ylabel('House Price (USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bar_comp_rf.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved bar_comp_rf.png")

# 6. Evaluation metrics comparison for KC House
models_kc = ['Linear Regression', 'Decision Tree (d=10)', 'Random Forest']
rmse_kc_vals = [
    np.sqrt(mean_squared_error(y_test_kc, y_pred_reg_kc)),
    np.sqrt(mean_squared_error(y_test_kc, y_pred_dt_kc)),
    np.sqrt(mean_squared_error(y_test_kc, y_pred_rf_kc))
]
r2_kc_vals = [
    r2_score(y_test_kc, y_pred_reg_kc),
    r2_score(y_test_kc, y_pred_dt_kc),
    r2_score(y_test_kc, y_pred_rf_kc)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
bars1 = ax1.bar(models_kc, [v / 1000 for v in rmse_kc_vals], color=['#4292c6', '#2171b5', '#084594'], edgecolor='black')
ax1.set_title('Root Mean Squared Error (dalam ribuan USD)', fontsize=11)
ax1.set_ylabel('RMSE (x $1,000)')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 3, f'{yval:.1f}k', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 250)

bars2 = ax2.bar(models_kc, r2_kc_vals, color=['#74c476', '#31a354', '#006d2c'], edgecolor='black')
ax2.set_title('Coefficient of Determination ($R^2$)', fontsize=11)
ax2.set_ylabel('$R^2$')
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylim(0, 1.05)

plt.suptitle('Perbandingan Evaluasi Model Regresi pada KC House Data', fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kc_house_eval_comp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved kc_house_eval_comp.png")

print("--- 2. Generating Car Price Data Figures ---")
url_car = 'https://raw.githubusercontent.com/pranavhaldar/Car-Price_Linear-Regression-Assignment/master/CarPrice_Assignment.csv'
df_car = pd.read_csv(url_car)

# 7. Correlation bar plot for Car Price
numeric_car = df_car.select_dtypes(include=[np.number]).drop('car_ID', axis=1)
corr_car = numeric_car.corr()['price'].drop('price').sort_values()

plt.figure(figsize=(8, 6))
colors = ['#d95f02' if c < 0 else '#1b9e77' for c in corr_car.values]
bars = plt.barh(corr_car.index, corr_car.values, color=colors, edgecolor='black', height=0.65)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Koefisien Korelasi Pearson Fitur Numerik terhadap Harga Mobil (Price)', fontsize=11)
plt.xlabel('Nilai Korelasi ($r$)')
for bar in bars:
    val = bar.get_width()
    offset = 0.02 if val >= 0 else -0.08
    plt.text(val + offset, bar.get_y() + bar.get_height()/4, f'{val:.2f}', fontsize=8, fontweight='bold')
plt.xlim(-0.9, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'corr_car_price.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved corr_car_price.png")

# Prepare Car Price data for modelling
df_X_car = df_car.drop(['car_ID', 'CarName', 'price'], axis=1)
y_car = df_car['price'].values

le = LabelEncoder()
cats_car = df_X_car.select_dtypes(include=['object']).columns
for col in cats_car:
    df_X_car[col] = le.fit_transform(df_X_car[col].astype(str))

X_car = df_X_car.values.astype(float)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_car, y_car, test_size=0.3, random_state=42)

models_car = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
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
        'Model': name,
        'RMSE': f"{rmse:.2f}",
        'R2': f"{r2:.4f}",
        'R': f"{r:.4f}",
        'raw_rmse': rmse,
        'raw_r2': r2,
        'raw_r': r
    })

df_res_car = pd.DataFrame(results_car)
print("Car Price Results Summary:")
print(df_res_car[['Model', 'RMSE', 'R2', 'R']])

# 8. Comparison bar charts for Car Price (test set)
n_samples = min(35, len(y_test_c))
idx = np.arange(n_samples)
width = 0.35

# Linear Regression
plt.figure(figsize=(14, 4))
plt.bar(idx - width/2, y_test_c[:n_samples], width=width, label='Nilai Riil', color='#1f77b4', edgecolor='black')
plt.bar(idx + width/2, preds_car['Linear Regression'][:n_samples], width=width, label='Prediksi LR', color='#ff7f0e', edgecolor='black')
plt.title('Linear Regression: Nilai Aktual vs Prediksi Harga Mobil (35 Sampel Data Uji)', fontsize=11)
plt.xlabel('Indeks Sampel Uji')
plt.ylabel('Car Price (USD)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'car_price_lr_comp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved car_price_lr_comp.png")

# Decision Tree
plt.figure(figsize=(14, 4))
plt.bar(idx - width/2, y_test_c[:n_samples], width=width, label='Nilai Riil', color='#1f77b4', edgecolor='black')
plt.bar(idx + width/2, preds_car['Decision Tree'][:n_samples], width=width, label='Prediksi DT', color='#2ca02c', edgecolor='black')
plt.title('Decision Tree Regression: Nilai Aktual vs Prediksi Harga Mobil (35 Sampel Data Uji)', fontsize=11)
plt.xlabel('Indeks Sampel Uji')
plt.ylabel('Car Price (USD)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'car_price_dt_comp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved car_price_dt_comp.png")

# Random Forest
plt.figure(figsize=(14, 4))
plt.bar(idx - width/2, y_test_c[:n_samples], width=width, label='Nilai Riil', color='#1f77b4', edgecolor='black')
plt.bar(idx + width/2, preds_car['Random Forest'][:n_samples], width=width, label='Prediksi RF', color='#d62728', edgecolor='black')
plt.title('Random Forest Regression: Nilai Aktual vs Prediksi Harga Mobil (35 Sampel Data Uji)', fontsize=11)
plt.xlabel('Indeks Sampel Uji')
plt.ylabel('Car Price (USD)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'car_price_rf_comp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved car_price_rf_comp.png")

# 9. Render Summary Table as Image
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis('off')
ax.axis('tight')

table_data = [
    ['Model', 'RMSE', 'R2', 'R'],
    ['Linear Regression', f"{results_car[0]['raw_rmse']:.2f}", f"{results_car[0]['raw_r2']:.4f}", f"{results_car[0]['raw_r']:.4f}"],
    ['Decision Tree Regression', f"{results_car[1]['raw_rmse']:.2f}", f"{results_car[1]['raw_r2']:.4f}", f"{results_car[1]['raw_r']:.4f}"],
    ['Random Forest Regression', f"{results_car[2]['raw_rmse']:.2f}", f"{results_car[2]['raw_r2']:.4f}", f"{results_car[2]['raw_r']:.4f}"]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#003366')
    cell.set_text_props(color='white', weight='bold')

for row in range(1, 4):
    for col in range(4):
        cell = table[(row, col)]
        if row % 2 == 1:
            cell.set_facecolor('#f0f4f8')
        if row == 3:
            cell.set_text_props(weight='bold')

plt.title('Tabel Performa Model Machine Learning (Car Price Dataset)', fontsize=11, pad=15, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'car_price_summary_table.png'), dpi=220, bbox_inches='tight')
plt.close()
print("Saved car_price_summary_table.png")

# 10. Metrics Comparison Bar Chart for Car Price
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
m_names = ['Linear Reg', 'Decision Tree', 'Random Forest']
rmses = [r['raw_rmse'] for r in results_car]
r2s = [r['raw_r2'] for r in results_car]
rs = [r['raw_r'] for r in results_car]

# RMSE
b1 = axes[0].bar(m_names, rmses, color=['#e6550d', '#fdae6b', '#31a354'], edgecolor='black')
axes[0].set_title('Root Mean Squared Error (RMSE)\n(Lebih rendah lebih baik)', fontsize=10)
axes[0].set_ylabel('RMSE (USD)')
for b in b1:
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width()/2, h + 60, f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[0].set_ylim(0, 4200)

# R2
b2 = axes[1].bar(m_names, r2s, color=['#74c476', '#31a354', '#006d2c'], edgecolor='black')
axes[1].set_title('Coefficient of Determination ($R^2$)\n(Lebih tinggi lebih baik)', fontsize=10)
axes[1].set_ylabel('$R^2$')
for b in b2:
    h = b.get_height()
    axes[1].text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[1].set_ylim(0, 1.08)

# R (Correlation)
b3 = axes[2].bar(m_names, rs, color=['#6baed6', '#3182bd', '#08519c'], edgecolor='black')
axes[2].set_title('Pearson Correlation ($R$)\n(Lebih tinggi lebih baik)', fontsize=10)
axes[2].set_ylabel('$R$')
for b in b3:
    h = b.get_height()
    axes[2].text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[2].set_ylim(0, 1.08)

plt.suptitle('Evaluasi Metrik Performa Tiga Algoritma Regresi (Car Price Prediction)', fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'car_price_metrics_comp.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved car_price_metrics_comp.png")

print("All figures successfully generated in:", output_dir)
