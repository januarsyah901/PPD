import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report
)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join(os.path.dirname(__file__), '../project/data')

print("--- Generating High-Resolution Plot Figures for P8 ---")

# ==============================================================================
# 1. Bank Churn: Histogram Distribusi
# ==============================================================================
df_churn = pd.read_csv(os.path.join(data_dir, 'BankChurners.csv'))
fig, ax = plt.subplots(figsize=(16, 12), dpi=220)
df_churn.hist(bins=20, edgecolor='black', grid=True, ax=ax, color='#3b82f6')
plt.suptitle('Distribusi Histogram Variabel Bank Customer Churn', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-churn-hist.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-churn-hist.png")

# ==============================================================================
# 2. Bank Churn: Pie Chart Proporsi Target Class
# ==============================================================================
plt.figure(figsize=(7, 6), dpi=220)
target_counts = df_churn['Attrition_Flag'].value_counts()
labels = [f"Existing Customer\n{target_counts['Existing Customer']:,} nasabah",
          f"Attrited Customer\n{target_counts['Attrited Customer']:,} nasabah"]
colors = ['#2563eb', '#f59e0b']
plt.pie(
    target_counts,
    labels=labels,
    autopct='%1.2f%%',
    colors=colors,
    startangle=140,
    explode=(0, 0.1),
    shadow=True,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)
plt.title("Proporsi Target Class Attrition_Flag", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-churn-pie.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-churn-pie.png")

# ==============================================================================
# 3. Bank Churn: Confusion Matrix Plot
# ==============================================================================
df_X = df_churn.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)
df_y = df_churn[['Attrition_Flag']]
le_y = LabelEncoder()
y = le_y.fit_transform(df_y['Attrition_Flag'])

cats = df_X.select_dtypes(include=['object', 'bool', 'str']).columns
for col in cats:
    le = LabelEncoder()
    df_X[col] = le.fit_transform(df_X[col].astype(str))

X = df_X.astype(float).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

clf = MLPClassifier(max_iter=500, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5), dpi=220)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix MLPClassifier (Bank Customer Churn)", fontsize=12, fontweight='bold', pad=12)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-churn-cm.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-churn-cm.png")

# ==============================================================================
# 4. KC House: Histogram Fitur Numerik
# ==============================================================================
df_kc = pd.read_csv(os.path.join(data_dir, 'kc_house_data.csv'))
fig, ax = plt.subplots(figsize=(16, 12), dpi=220)
df_kc.drop(['id'], axis=1, errors='ignore').hist(bins=20, edgecolor='black', grid=True, ax=ax, color='#10b981')
plt.suptitle('Distribusi Histogram Fitur Numerik KC House Data', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-kc-hist.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-kc-hist.png")

# ==============================================================================
# 5. KC House: Bar Plot Real vs Predicted (30 Sampel)
# ==============================================================================
df_X_kc = df_kc.drop(['id', 'date', 'price'], axis=1)
df_y_kc = df_kc['price']
imputer = SimpleImputer(strategy='mean')
X_kc = imputer.fit_transform(df_X_kc)
y_kc = df_y_kc.astype(float).values
X_train_kc, X_test_kc, y_train_kc, y_test_kc = train_test_split(X_kc, y_kc, test_size=0.3, random_state=42)

reg_kc = MLPRegressor(max_iter=500, random_state=42)
reg_kc.fit(X_train_kc, y_train_kc)
y_pred_kc = reg_kc.predict(X_test_kc)

data_real = pd.DataFrame(y_test_kc[:30], columns=['real'])
data_pred = pd.DataFrame(y_pred_kc[:30], columns=['predicted'])
df_comp_kc = pd.concat([data_real, data_pred], axis=1)

fig, ax = plt.subplots(figsize=(13, 5), dpi=220)
df_comp_kc.plot(kind='bar', ax=ax, color=['#2563eb', '#f59e0b'], width=0.8)
plt.title("MLP Regression - Real vs Predicted House Price (30 Sampel)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Sampel Data Uji")
plt.ylabel("House Price (USD)")
plt.xticks(rotation=0, fontsize=8)
plt.legend(['Nilai Riil (Real)', 'Hasil Prediksi (Predicted)'], fontsize=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-kc-bar-pred.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-kc-bar-pred.png")

# ==============================================================================
# 6. Tugas 2A: Diabetes Classification Figures (Confusion Matrix & Loss Curve)
# ==============================================================================
df_diab = pd.read_csv(os.path.join(data_dir, 'diabetes.csv'))
X_diab = df_diab.drop('Outcome', axis=1).values
y_diab = df_diab['Outcome'].values
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_diab, y_diab, test_size=0.3, random_state=42, stratify=y_diab)

scaler_d = StandardScaler().fit(X_tr_d)
X_tr_d_s = scaler_d.transform(X_tr_d)
X_te_d_s = scaler_d.transform(X_te_d)

mlp_diab = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_diab.fit(X_tr_d_s, y_tr_d)
y_pred_d = mlp_diab.predict(X_te_d_s)
cm_d = confusion_matrix(y_te_d, y_pred_d)

fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=220)
disp_d = ConfusionMatrixDisplay(confusion_matrix=cm_d, display_labels=['Non-Diabetic (0)', 'Diabetic (1)'])
disp_d.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix - Diabetes Classification (Tugas 2)", fontsize=11, fontweight='bold', pad=10)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-t2-diab-cm.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-t2-diab-cm.png")

fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=220)
plt.plot(mlp_diab.loss_curve_, color='#059669', lw=2.2)
plt.title("Kurva Loss Pelatihan MLPClassifier (Diabetes)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Iterasi (Epochs)")
plt.ylabel("Loss Nilai Fungsi Error")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-t2-diab-loss.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-t2-diab-loss.png")

# ==============================================================================
# 7. Tugas 2B: Medical Insurance Figures (Scatter, Bar, Loss)
# ==============================================================================
df_ins = pd.read_csv(os.path.join(data_dir, 'insurance.csv'))
df_ins_encoded = pd.get_dummies(df_ins, columns=['sex', 'smoker', 'region'], drop_first=True)
X_ins = df_ins_encoded.drop('charges', axis=1).values
y_ins = df_ins_encoded['charges'].values
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_ins, y_ins, test_size=0.3, random_state=42)

scaler_i = StandardScaler().fit(X_tr_i)
X_tr_i_s = scaler_i.transform(X_tr_i)
X_te_i_s = scaler_i.transform(X_te_i)

mlp_ins = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_ins.fit(X_tr_i_s, y_tr_i)
y_pred_i = mlp_ins.predict(X_te_i_s)

# Scatter plot
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=220)
ax.scatter(y_te_i, y_pred_i, alpha=0.6, color='#2563eb', edgecolors='k', linewidth=0.5, s=45)
min_val = min(y_te_i.min(), y_pred_i.min())
max_val = max(y_te_i.max(), y_pred_i.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Garis Ideal Prediksi Tepat')
ax.set_title("Scatter Plot Prediksi vs Nilai Aktual Tagihan Asuransi", fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel("Biaya Aktual Tagihan Asuransi (USD)")
ax.set_ylabel("Prediksi Biaya Tagihan Asuransi (USD)")
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-t2-ins-scatter.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-t2-ins-scatter.png")

# Bar plot 30 sampel
fig, ax = plt.subplots(figsize=(13, 5), dpi=220)
df_comp_ins = pd.DataFrame({'Aktual': y_te_i[:30], 'Prediksi': y_pred_i[:30]})
df_comp_ins.plot(kind='bar', ax=ax, color=['#2563eb', '#f59e0b'], width=0.8)
plt.title("Perbandingan Tagihan Asuransi Aktual vs Prediksi (30 Sampel Pertama)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Sampel Pasien Uji")
plt.ylabel("Biaya Tagihan Asuransi (USD)")
plt.xticks(rotation=0, fontsize=8)
plt.legend(['Biaya Riil (Actual)', 'Prediksi Model (Predicted)'], fontsize=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-t2-ins-bar.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-t2-ins-bar.png")

# Loss curve
fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=220)
plt.plot(mlp_ins.loss_curve_, color='#d97706', lw=2.2)
plt.title("Kurva Loss Pelatihan MLPRegressor (Insurance Charges)", fontsize=11, fontweight='bold', pad=10)
plt.xlabel("Iterasi (Epochs)")
plt.ylabel("Loss (MSE)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p8-t2-ins-loss.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p8-t2-ins-loss.png")

print("--- All High-Resolution Plot Figures Successfully Generated! ---")
