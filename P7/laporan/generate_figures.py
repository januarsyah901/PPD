import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Config styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

print("--- Generating High-Resolution Plot Figures for P7 ---")

# Cari data stroke
data_path = os.path.join(os.path.dirname(__file__), '../project/data/healthcare-dataset-stroke-data.csv')
if not os.path.exists(data_path):
    df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/refs/heads/main/healthcare-dataset-stroke-data.csv')
else:
    df = pd.read_csv(data_path)

# 1. Pie Chart Distribusi Target Stroke
plt.figure(figsize=(7, 6), dpi=220)
target_counts = df['stroke'].value_counts()
labels = [f"Tidak Stroke (0)\n{target_counts[0]:,} pasien", f"Stroke (1)\n{target_counts[1]:,} pasien"]
plt.pie(
    target_counts,
    labels=labels,
    autopct='%.2f%%',
    startangle=140,
    colors=['#2563eb', '#dc2626'],
    explode=(0, 0.12),
    shadow=True,
    textprops={'fontsize': 11, 'weight': 'bold'}
)
plt.title("Distribusi Kelas Target 'stroke' (Rasio Ketimpangan ~19.5:1)", fontsize=13, pad=15, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'colab-p7-pie-distribution.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-pie-distribution.png")

# 2. Missing Value Imputation Plot
plt.figure(figsize=(8, 4.5), dpi=220)
null_counts = [df['bmi'].isnull().sum(), 0]
categories = ['Sebelum Imputasi', 'Sesudah Imputasi (Mean)']
bars = plt.bar(categories, null_counts, color=['#ef4444', '#10b981'], width=0.45)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)} null", ha='center', va='bottom', fontweight='bold')
plt.title("Pemeriksaan Missing Values pada Atribut 'bmi'", fontsize=12, pad=12, fontweight='bold')
plt.ylabel("Jumlah Missing Values (Baris)", fontsize=11)
plt.ylim(0, 240)
plt.savefig(os.path.join(output_dir, 'colab-p7-missing-values.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-missing-values.png")

# Preprocessing Dataset Stroke
df_clean = df.copy()
df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].mean())

df_X = df_clean.drop(['id', 'stroke'], axis=1).copy()
df_y = df_clean['stroke'].copy()

cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in cat_cols:
    df_X[col] = le.fit_transform(df_X[col].astype(str))

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

df_ori = pd.concat([pd.DataFrame(X_train, columns=df_X.columns), pd.DataFrame(y_train, columns=['stroke'])], axis=1)

# Helper function plotting distribution bar, hist, scatter
def generate_balancing_plots(X_res, y_res, method_name, prefix):
    # 1. Bar count
    plt.figure(figsize=(7, 4.5), dpi=220)
    counts_before = [int(np.sum(y_train == 0)), int(np.sum(y_train == 1))]
    counts_after = [int(np.sum(y_res == 0)), int(np.sum(y_res == 1))]
    
    x_idx = np.arange(2)
    w = 0.35
    plt.bar(x_idx - w/2, counts_before, width=w, label='Sebelum Penyeimbangan', color='#64748b')
    plt.bar(x_idx + w/2, counts_after, width=w, label=f'Sesudah {method_name}', color='#2563eb')
    
    for i, v in enumerate(counts_before):
        plt.text(i - w/2, v + 40, f"{v}", ha='center', fontweight='bold', fontsize=9)
    for i, v in enumerate(counts_after):
        plt.text(i + w/2, v + 40, f"{v}", ha='center', fontweight='bold', fontsize=9, color='#1d4ed8')
        
    plt.xticks(x_idx, ['Tidak Stroke (0)', 'Stroke (1)'])
    plt.ylabel('Jumlah Sampel', fontsize=11)
    plt.title(f"Perbandingan Frekuensi Kelas Sebelum vs Sesudah {method_name}", fontsize=12, pad=12, fontweight='bold')
    plt.legend()
    plt.ylim(0, max(max(counts_before), max(counts_after)) * 1.15)
    plt.savefig(os.path.join(output_dir, f'{prefix}-dist.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Histogram BMI
    df_res = pd.concat([pd.DataFrame(X_res, columns=df_X.columns), pd.DataFrame(y_res, columns=['stroke'])], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    sns.histplot(data=df_ori, x='bmi', hue='stroke', ax=axes[0], palette=['#2563eb', '#dc2626']).set(title="Sebelum Penyeimbangan")
    sns.histplot(data=df_res, x='bmi', hue='stroke', ax=axes[1], palette=['#2563eb', '#dc2626']).set(title=f"Sesudah {method_name}")
    plt.suptitle(f"Distribusi Fitur BMI: Sebelum vs Sesudah {method_name}", fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}-bmi-hist.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    sns.scatterplot(data=df_ori, x='avg_glucose_level', y='bmi', hue='stroke', alpha=0.6, ax=axes[0], palette=['#2563eb', '#dc2626']).set(title="Sebelum Penyeimbangan")
    sns.scatterplot(data=df_res, x='avg_glucose_level', y='bmi', hue='stroke', alpha=0.6, ax=axes[1], palette=['#2563eb', '#dc2626']).set(title=f"Sesudah {method_name}")
    plt.suptitle(f"Sebaran Fitur (Glukosa vs BMI): Sebelum vs Sesudah {method_name}", fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}-scatter.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved plots for {method_name} ({prefix})")

# A. Random Under Sampling
rus = RandomUnderSampler(random_state=0)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
generate_balancing_plots(X_rus, y_rus, "Random Under Sampling (RUS)", "colab-p7-rus")

# B. Random Over Sampling
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
generate_balancing_plots(X_ros, y_ros, "Random Over Sampling (ROS)", "colab-p7-ros")

# C. SMOTE
smote = SMOTE(random_state=0)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
generate_balancing_plots(X_smote, y_smote, "SMOTE", "colab-p7-smote")

# D. ADASYN
adasyn = ADASYN(random_state=0)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
generate_balancing_plots(X_adasyn, y_adasyn, "ADASYN", "colab-p7-adasyn")

# 7. Comparison All Methods Bar Chart
plt.figure(figsize=(10, 5), dpi=220)
methods = ['Data Asli', 'RUS', 'ROS', 'SMOTE', 'ADASYN']
c0 = [np.sum(y_train==0), np.sum(y_rus==0), np.sum(y_ros==0), np.sum(y_smote==0), np.sum(y_adasyn==0)]
c1 = [np.sum(y_train==1), np.sum(y_rus==1), np.sum(y_ros==1), np.sum(y_smote==1), np.sum(y_adasyn==1)]

x = np.arange(len(methods))
w = 0.35
plt.bar(x - w/2, c0, width=w, label='Kelas Mayoritas (Tidak Stroke)', color='#3b82f6')
plt.bar(x + w/2, c1, width=w, label='Kelas Minoritas (Stroke)', color='#ef4444')

for i in range(len(methods)):
    plt.text(i - w/2, c0[i] + 45, f"{c0[i]}", ha='center', fontsize=9, fontweight='bold')
    plt.text(i + w/2, c1[i] + 45, f"{c1[i]}", ha='center', fontsize=9, fontweight='bold')

plt.xticks(x, methods, fontweight='bold')
plt.ylabel('Jumlah Sampel pada Training Set', fontsize=11)
plt.title("Komparasi Distribusi Sampel Training pada Berbagai Teknik Penyeimbang", fontsize=12, pad=12, fontweight='bold')
plt.legend(frameon=True)
plt.ylim(0, 4200)
plt.savefig(os.path.join(output_dir, 'colab-p7-comparison-all.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-comparison-all.png")

# --- TUGAS 2 EXPERIMENTS & PLOTS ---
print("--- Generating Plots for Tugas 2 (Heart Failure Records) ---")
hf_path = os.path.join(os.path.dirname(__file__), '../project/data/heart_failure_clinical_records_dataset.csv')
if not os.path.exists(hf_path):
    df_hf = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/heart_failure_clinical_records_dataset.csv')
else:
    df_hf = pd.read_csv(hf_path)

# Pie chart target Tugas 2
plt.figure(figsize=(6.5, 5.5), dpi=220)
hf_counts = df_hf['DEATH_EVENT'].value_counts()
labels = [f"Pasien Selamat (0)\n{hf_counts[0]} pasien", f"Meninggal Dunia (1)\n{hf_counts[1]} pasien"]
plt.pie(
    hf_counts,
    labels=labels,
    autopct='%.2f%%',
    startangle=135,
    colors=['#0ea5e9', '#f43f5e'],
    explode=(0, 0.1),
    shadow=True,
    textprops={'fontsize': 11, 'weight': 'bold'}
)
plt.title("Distribusi Target Kematian Pasien (DEATH_EVENT)", fontsize=13, pad=15, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'colab-p7-t2-heart-dist.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-t2-heart-dist.png")

# Preprocessing & Model Evaluation for Tugas 2
X_hf = df_hf.drop('DEATH_EVENT', axis=1).values
y_hf = df_hf['DEATH_EVENT'].values

X_train_hf, X_test_hf, y_train_hf, y_test_hf = train_test_split(
    X_hf, y_hf, test_size=0.25, random_state=42, stratify=y_hf
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_hf)
X_te_sc = scaler.transform(X_test_hf)

# Balancing
smote_hf = SMOTE(random_state=42)
X_tr_smote, y_tr_smote = smote_hf.fit_resample(X_tr_sc, y_train_hf)

adasyn_hf = ADASYN(random_state=42)
X_tr_adasyn, y_tr_adasyn = adasyn_hf.fit_resample(X_tr_sc, y_train_hf)

models = [
    ('Logistic Regression', 'Baseline', LogisticRegression(random_state=42), X_tr_sc, y_train_hf),
    ('Logistic Regression', 'SMOTE', LogisticRegression(random_state=42), X_tr_smote, y_tr_smote),
    ('Logistic Regression', 'ADASYN', LogisticRegression(random_state=42), X_tr_adasyn, y_tr_adasyn),
    ('Random Forest', 'Baseline', RandomForestClassifier(random_state=42), X_tr_sc, y_train_hf),
    ('Random Forest', 'SMOTE', RandomForestClassifier(random_state=42), X_tr_smote, y_tr_smote),
    ('Random Forest', 'ADASYN', RandomForestClassifier(random_state=42), X_tr_adasyn, y_tr_adasyn),
]

eval_res = []
for name, bal, clf, x_t, y_t in models:
    clf.fit(x_t, y_t)
    preds = clf.predict(X_te_sc)
    probas = clf.predict_proba(X_te_sc)[:, 1]
    
    acc = accuracy_score(y_test_hf, preds)
    prec = precision_score(y_test_hf, preds, zero_division=0)
    rec = recall_score(y_test_hf, preds, zero_division=0)
    f1 = f1_score(y_test_hf, preds, zero_division=0)
    cm = confusion_matrix(y_test_hf, preds)
    
    eval_res.append({
        'name': name,
        'bal': bal,
        'clf': clf,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'proba': probas
    })

# 8. Confusion Matrices Heatmap (2x3)
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=220)
for idx, r in enumerate(eval_res):
    ax = axes[idx // 3, idx % 3]
    cm = r['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f"{r['name']} ({r['bal']})\nRecall: {r['rec']:.3f} | F1: {r['f1']:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediksi Model', fontsize=10)
    ax.set_ylabel('Label Aktual', fontsize=10)
    ax.set_xticklabels(['Selamat (0)', 'Wafat (1)'])
    ax.set_yticklabels(['Selamat (0)', 'Wafat (1)'])

plt.suptitle("Confusion Matrix Komparatif Model Klasifikasi (Heart Failure)", fontsize=14, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p7-t2-confusion-matrices.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-t2-confusion-matrices.png")

# 9. Bar Chart Metrik Komparatif
plt.figure(figsize=(12, 5.5), dpi=220)
model_labels = [f"{r['name']}\n({r['bal']})" for r in eval_res]
x = np.arange(len(model_labels))
w = 0.2

plt.bar(x - 1.5*w, [r['acc'] for r in eval_res], width=w, label='Accuracy', color='#3b82f6')
plt.bar(x - 0.5*w, [r['prec'] for r in eval_res], width=w, label='Precision', color='#10b981')
plt.bar(x + 0.5*w, [r['rec'] for r in eval_res], width=w, label='Recall', color='#f59e0b')
plt.bar(x + 1.5*w, [r['f1'] for r in eval_res], width=w, label='F1-Score', color='#8b5cf6')

plt.xticks(x, model_labels, fontsize=9, fontweight='bold')
plt.ylabel('Skor Nilai Metrik', fontsize=11)
plt.title("Komparasi Metrik Evaluasi Model Sebelum dan Sesudah Balancing", fontsize=13, pad=12, fontweight='bold')
plt.ylim(0, 1.15)
plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=True)

# Berikan highlight text pada recall
for i, r in enumerate(eval_res):
    plt.text(i + 0.5*w, r['rec'] + 0.02, f"{r['rec']:.2f}", ha='center', fontsize=8, fontweight='bold', color='#b45309')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p7-t2-metrics-bar.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-t2-metrics-bar.png")

# 10. ROC Curves & PR Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=220)

for r in eval_res:
    fpr, tpr, _ = roc_curve(y_test_hf, r['proba'])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{r['name']} ({r['bal']}) AUC={roc_auc:.3f}", linewidth=1.8)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate', fontsize=10)
axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=10)
axes[0].legend(loc='lower right', fontsize=8.5)

for r in eval_res:
    pr, rc, _ = precision_recall_curve(y_test_hf, r['proba'])
    axes[1].plot(rc, pr, label=f"{r['name']} ({r['bal']})", linewidth=1.8)

axes[1].set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Recall', fontsize=10)
axes[1].set_ylabel('Precision', fontsize=10)
axes[1].legend(loc='lower left', fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p7-t2-roc-pr-curves.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p7-t2-roc-pr-curves.png")

print("All figures successfully generated in P7/laporan/gambar/!")
