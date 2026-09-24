import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D

# Config styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

print("--- Generating Plot Figures for P6 ---")

# Load Credit Card Dataset
cc_path = os.path.join(os.path.dirname(__file__), '../project/data/CreditCard.csv')
if not os.path.exists(cc_path):
    print("CreditCard dataset tidak ditemukan di local, menggunakan URL raw GitHub...")
    df_cc = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/CC%20GENERAL.csv')
else:
    df_cc = pd.read_csv(cc_path)

# Preprocessing
df_new = df_cc.drop('CUST_ID', axis=1)
df_new['MINIMUM_PAYMENTS'] = df_new['MINIMUM_PAYMENTS'].fillna(df_new['MINIMUM_PAYMENTS'].median())
df_new['CREDIT_LIMIT'] = df_new['CREDIT_LIMIT'].fillna(df_new['CREDIT_LIMIT'].median())

X = df_new.astype(float).values
scaler = StandardScaler().fit(X)
X_new = scaler.transform(X)

# 1. Heatmap Korelasi
plt.figure(figsize=(10, 8), dpi=200)
sns.heatmap(df_new.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Atribut Credit Card', pad=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'colab-p6-corr.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-corr.png")

# Compute KMeans Inertia & Silhouette
inertia_km = []
sh_km = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_new)
    inertia_km.append(km.inertia_)
    if k >= 2:
        sh_km.append(silhouette_score(X_new, km.labels_))

# 2. KMeans Elbow Plot
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(range(1, 11), inertia_km, marker='o', linewidth=2.2, markersize=8, color='#d97706')
plt.xlabel("Number of Clusters", size=12)
plt.ylabel("Inertia Value", size=12)
plt.title("Inertia values vary depending on the number of clusters utilized (K-Means)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, 'colab-p6-kmeans-elbow-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmeans-elbow-plot.png")

# 3. KMeans Knee Locator Plot
kneedle_km = KneeLocator(range(1, 11), inertia_km, S=1.0, curve='convex', direction='decreasing')
plt.figure(figsize=(7, 5), dpi=200)
plt.plot(range(1, 11), inertia_km, 'b-', label='Inertia')
plt.axvline(x=kneedle_km.knee, color='r', linestyle='--', label=f'Knee/Elbow Point (k={kneedle_km.knee})')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('K-Means KneeLocator Optimal Detection', fontweight='bold')
plt.legend()
plt.savefig(os.path.join(output_dir, 'colab-p6-kmeans-knee-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmeans-knee-plot.png")

# Compute KMedoids Inertia & Silhouette
inertia_kmed = []
sh_kmed = []
for k in range(1, 11):
    kmed = KMedoids(n_clusters=k, random_state=42)
    kmed.fit(X_new)
    inertia_kmed.append(kmed.inertia_)
    if k >= 2:
        sh_kmed.append(silhouette_score(X_new, kmed.labels_))

# 4. KMedoids Elbow Plot
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(range(1, 11), inertia_kmed, marker='o', linewidth=2.2, markersize=8, color='#2563eb')
plt.xlabel("Number of Clusters", size=12)
plt.ylabel("Inertia Value", size=12)
plt.title("Inertia values vary depending on the number of clusters utilized (K-Medoids)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, 'colab-p6-kmedoids-elbow-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmedoids-elbow-plot.png")

# 5. KMedoids Knee Locator Plot
kneedle_kmed = KneeLocator(range(1, 11), inertia_kmed, S=1.0, curve='convex', direction='decreasing')
plt.figure(figsize=(7, 5), dpi=200)
plt.plot(range(1, 11), inertia_kmed, 'b-', label='Inertia')
plt.axvline(x=kneedle_kmed.knee if kneedle_kmed.knee else 4, color='r', linestyle='--', label='Knee/Elbow Point (k=4)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('K-Medoids KneeLocator Optimal Detection', fontweight='bold')
plt.legend()
plt.savefig(os.path.join(output_dir, 'colab-p6-kmedoids-knee-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmedoids-knee-plot.png")

# 6. KMeans Silhouette Plot
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(range(2, 11), sh_km, marker='o', linewidth=2.2, markersize=8, color='#059669')
plt.xlabel("Number of Clusters", size=12)
plt.ylabel("Silhouette Score", size=12)
plt.title("Silhouette score values vary depending on the number of clusters utilized (K-Means)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, 'colab-p6-kmeans-sh-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmeans-sh-plot.png")

# 7. KMedoids Silhouette Plot
plt.figure(figsize=(8, 5), dpi=200)
plt.plot(range(2, 11), sh_kmed, marker='o', linewidth=2.2, markersize=8, color='#7c3aed')
plt.xlabel("Number of Clusters", size=12)
plt.ylabel("Silhouette Score", size=12)
plt.title("Silhouette score vary depending on the number of clusters utilized (K-Medoids)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, 'colab-p6-kmedoids-sh-plot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmedoids-sh-plot.png")

# 8. KMeans 2D Scatter Plot
km_opt = KMeans(n_clusters=2, random_state=42, n_init=10)
km_opt.fit(X_new)
df_new['km_label'] = km_opt.labels_

plt.figure(figsize=(8, 6), dpi=200)
sns.scatterplot(x='PURCHASES', y='PAYMENTS', hue='km_label', data=df_new, palette='Paired', alpha=0.8)
plt.title('K-means clustering (PURCHASES vs PAYMENTS)', fontsize=14, fontweight='bold')
plt.xlabel('PURCHASES', fontsize=12)
plt.ylabel('PAYMENTS', fontsize=12)
plt.legend(title='Cluster')
plt.savefig(os.path.join(output_dir, 'colab-p6-kmeans-scatter2d.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmeans-scatter2d.png")

# 9. KMeans 3D Scatter Plot
fig = plt.figure(figsize=(9, 7), dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_new['PURCHASES'], df_new['PAYMENTS'], df_new['BALANCE'], c=df_new['km_label'], cmap='tab10', alpha=0.6)
ax.set_xlabel('PURCHASES')
ax.set_ylabel('PAYMENTS')
ax.set_zlabel('BALANCE')
plt.title('3D Scatter Plot K-Means (PURCHASES vs PAYMENTS vs BALANCE)', fontweight='bold')
plt.savefig(os.path.join(output_dir, 'colab-p6-kmeans-scatter3d.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmeans-scatter3d.png")

# 10. KMedoids 2D Scatter Plot
kmed_opt = KMedoids(n_clusters=4, random_state=42)
kmed_opt.fit(X_new)
df_new['kmed_label'] = kmed_opt.labels_

plt.figure(figsize=(8, 6), dpi=200)
sns.scatterplot(x='PURCHASES', y='PAYMENTS', hue='kmed_label', data=df_new, palette='Set1', alpha=0.8)
plt.title('K-medoids clustering (PURCHASES vs PAYMENTS)', fontsize=14, fontweight='bold')
plt.xlabel('PURCHASES', fontsize=12)
plt.ylabel('PAYMENTS', fontsize=12)
plt.legend(title='Cluster')
plt.savefig(os.path.join(output_dir, 'colab-p6-kmedoids-scatter2d.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-kmedoids-scatter2d.png")

# 11. KMedoids Cluster Analysis Barplots
plt.figure(figsize=(14, 5), dpi=200)
plt.subplot(1, 3, 1)
sns.barplot(x='kmed_label', y='PURCHASES', data=df_new, palette='viridis')
plt.title('Rata-rata PURCHASES')
plt.xlabel('Cluster Label')

plt.subplot(1, 3, 2)
sns.barplot(x='kmed_label', y='PAYMENTS', data=df_new, palette='magma')
plt.title('Rata-rata PAYMENTS')
plt.xlabel('Cluster Label')

plt.subplot(1, 3, 3)
sns.barplot(x='kmed_label', y='BALANCE', data=df_new, palette='plasma')
plt.title('Rata-rata BALANCE')
plt.xlabel('Cluster Label')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'colab-p6-cluster-analysis-barplot.png'), bbox_inches='tight')
plt.close()
print("Saved colab-p6-cluster-analysis-barplot.png")

# =====================================================================
# AIR TRAFFIC FIGURES
# =====================================================================
air_path = os.path.join(os.path.dirname(__file__), '../project/data/Air_Traffic_Passenger_Statistics.csv')
if os.path.exists(air_path):
    df_air = pd.read_csv(air_path)
    df_air_clean = df_air.copy()
    
    cat_cols = ['operating_airline', 'geo_summary', 'geo_region', 'activity_type_code', 'price_category_code']
    for col in cat_cols:
        if col in df_air_clean.columns:
            df_air_clean[col + '_encoded'] = LabelEncoder().fit_transform(df_air_clean[col].astype(str))
            
    feature_cols = ['passenger_count'] + [c + '_encoded' for c in cat_cols if c in df_air_clean.columns]
    X_air_scaled = StandardScaler().fit_transform(df_air_clean[feature_cols].values)
    
    # Air Traffic Elbow & Silhouette
    inertia_air_km, sh_air_km = [], []
    inertia_air_kmed, sh_air_kmed = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_air_scaled)
        kmed = KMedoids(n_clusters=k, random_state=42).fit(X_air_scaled)
        inertia_air_km.append(km.inertia_)
        inertia_air_kmed.append(kmed.inertia_)
        if k >= 2:
            sh_air_km.append(silhouette_score(X_air_scaled, km.labels_))
            sh_air_kmed.append(silhouette_score(X_air_scaled, kmed.labels_))
            
    # 12. colab-p6-air-elbow-sh.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    axes[0, 0].plot(range(1, 11), inertia_air_km, marker='o', color='#d97706', linewidth=2)
    axes[0, 0].set_title('Elbow Method - K-Means (Air Traffic)', fontweight='bold')
    axes[0, 0].set_xlabel('k')
    axes[0, 0].set_ylabel('Inertia')
    
    axes[0, 1].plot(range(2, 11), sh_air_km, marker='o', color='#059669', linewidth=2)
    axes[0, 1].set_title('Silhouette Score - K-Means (Air Traffic)', fontweight='bold')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('Silhouette Score')
    
    axes[1, 0].plot(range(1, 11), inertia_air_kmed, marker='o', color='#2563eb', linewidth=2)
    axes[1, 0].set_title('Elbow Method - K-Medoids (Air Traffic)', fontweight='bold')
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('Inertia')
    
    axes[1, 1].plot(range(2, 11), sh_air_kmed, marker='o', color='#7c3aed', linewidth=2)
    axes[1, 1].set_title('Silhouette Score - K-Medoids (Air Traffic)', fontweight='bold')
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'colab-p6-air-elbow-sh.png'), bbox_inches='tight')
    plt.close()
    print("Saved colab-p6-air-elbow-sh.png")
    
    # 13. colab-p6-air-scatter-comp.png
    km_air = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_air_scaled)
    kmed_air = KMedoids(n_clusters=3, random_state=42).fit(X_air_scaled)
    df_air_clean['km_air_cluster'] = km_air.labels_
    df_air_clean['kmed_air_cluster'] = kmed_air.labels_
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
    sns.scatterplot(data=df_air_clean, x='geo_region_encoded', y='passenger_count', hue='km_air_cluster', palette='tab10', ax=axes[0], alpha=0.8)
    axes[0].set_title('K-Means Clustering Air Traffic (k=3)', fontweight='bold')
    axes[0].set_xlabel('Geo Region Encoded')
    axes[0].set_ylabel('Passenger Count')
    
    sns.scatterplot(data=df_air_clean, x='geo_region_encoded', y='passenger_count', hue='kmed_air_cluster', palette='Set1', ax=axes[1], alpha=0.8)
    axes[1].set_title('K-Medoids Clustering Air Traffic (k=3)', fontweight='bold')
    axes[1].set_xlabel('Geo Region Encoded')
    axes[1].set_ylabel('Passenger Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'colab-p6-air-scatter-comp.png'), bbox_inches='tight')
    plt.close()
    print("Saved colab-p6-air-scatter-comp.png")
    
    # 14. colab-p6-air-bar-summary.png
    plt.figure(figsize=(9, 5), dpi=200)
    sns.barplot(x='km_air_cluster', y='passenger_count', data=df_air_clean, palette='Blues_d')
    plt.title('Rata-rata Jumlah Penumpang per Kluster K-Means (Air Traffic)', fontweight='bold')
    plt.xlabel('Kluster K-Means')
    plt.ylabel('Rata-rata Passenger Count')
    plt.savefig(os.path.join(output_dir, 'colab-p6-air-bar-summary.png'), bbox_inches='tight')
    plt.close()
    print("Saved colab-p6-air-bar-summary.png")

print("Seluruh visualisasi gambar P6 selesai diproses!")
