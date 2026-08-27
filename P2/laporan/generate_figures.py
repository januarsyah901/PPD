import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup dir
output_dir = os.path.join(os.path.dirname(__file__), 'gambar')
os.makedirs(output_dir, exist_ok=True)

print("Generating figures for P2 Laporan...")

# 1. Line Chart Dummy
plt.figure(figsize=(6, 4))
x = np.array([2, 4, 6, 8])
y1 = np.array([3, 8, 1, 10])
y2 = np.array([4, 7, 10, 12])
plt.plot(x, y1, marker='o', label='Data 1')
plt.plot(x, y2, marker='x', label='Data 2')
plt.title('Line Chart Dummy')
plt.xlabel('Sumbu X')
plt.ylabel('Sumbu Y')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_line_dummy.png'), dpi=200)
plt.close()

# 2. IoT Humidity Time Series
df_iot = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/datatraining.txt')
df_iot['date'] = pd.to_datetime(df_iot['date'])
plt.figure(figsize=(10, 3.5))
plt.plot(df_iot['date'], df_iot['Humidity'], color='tab:blue', linewidth=1)
plt.title('IoT Humidity Over Time')
plt.xlabel('Date')
plt.ylabel('Humidity')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_iot_humidity.png'), dpi=200)
plt.close()

# 3. Pie Chart Dummy
plt.figure(figsize=(5, 5))
y = np.array([35, 25, 25, 15])
mylabels = ["Web Programmer", "Data Scientist", "DB Admin", "Manager"]
plt.pie(y, labels=mylabels, autopct='%.2f%%', startangle=90)
plt.title('Pie Chart Dummy (Profesi)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_pie_dummy.png'), dpi=200)
plt.close()

# Load BankChurners
df_churning = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/BankChurners.csv')
existing_data = df_churning[(df_churning.Attrition_Flag == 'Existing Customer')]
attrited_data = df_churning[(df_churning.Attrition_Flag == 'Attrited Customer')]

# 4. Pie Chart Marital Status
plt.figure(figsize=(5.5, 5.5))
data_marital = df_churning['Marital_Status'].value_counts()
plt.pie(data_marital, labels=data_marital.index, autopct='%.2f%%', startangle=140)
plt.title('Marital Status (Bank Churners)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_pie_marital.png'), dpi=200)
plt.close()

# 5. Bar Plot Dummy
plt.figure(figsize=(6, 4))
y_bar = np.array([35, 25, 25, 15])
x_bar = ["Web Programmer", "Data Scientist", "DB Admin", "Manager"]
plt.bar(x_bar, y_bar, color='skyblue', edgecolor='black')
plt.xticks(rotation=30)
plt.ylabel('Jumlah')
plt.title('Bar Plot Dummy')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_bar_dummy.png'), dpi=200)
plt.close()

# 6. Grouped Bar Plot Marital Status
plt.figure(figsize=(7, 4.5))
y1_m = existing_data['Marital_Status'].value_counts()
y2_m = attrited_data['Marital_Status'].value_counts()
x1_m = np.arange(len(y1_m.index))
x2_m = np.arange(len(y2_m.index))
bar_width = 0.4
plt.bar(x1_m, y1_m, width=bar_width, label='Existing Customer', color='tab:blue')
plt.bar(x2_m + bar_width, y2_m, width=bar_width, label='Attrited Customer', color='tab:orange')
plt.title('Marital Status: Existing vs Attrited')
plt.ylabel('Count')
plt.xticks(x2_m + bar_width / 2, labels=y1_m.index)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_bar_marital_grouped.png'), dpi=200)
plt.close()

# 7. Stacked Bar Plot Marital Status
plt.figure(figsize=(6.5, 4.5))
df_stack = pd.concat([y1_m, y2_m], keys=['existing', 'attrited'], axis=1)
df_stack.plot(kind='bar', stacked=True, color=['tab:blue', 'tab:orange'], figsize=(6.5, 4.5))
plt.title('Stacked Bar Plot Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_bar_marital_stacked.png'), dpi=200)
plt.close()

# 8. Histogram Dummy
plt.figure(figsize=(6, 4))
data_hist = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
bins = [0, 25, 50, 75, 100]
plt.hist(data_hist, bins=bins, color='mediumpurple', edgecolor='black')
plt.title("Histogram Data Dummy")
plt.xticks([0, 25, 50, 75, 100])
plt.xlabel('Nilai')
plt.ylabel('Jumlah Siswa')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_hist_dummy.png'), dpi=200)
plt.close()

# 9. Histogram Customer Age
plt.figure(figsize=(6.5, 4.5))
plt.hist(existing_data['Customer_Age'], alpha=0.7, label='Existing', bins=15, color='tab:blue')
plt.hist(attrited_data['Customer_Age'], alpha=0.7, label='Attrited', bins=15, color='tab:orange')
plt.title('Distribusi Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_hist_age.png'), dpi=200)
plt.close()

# 10. Scatter Plot Age vs Months on Book
plt.figure(figsize=(6.5, 4.5))
plt.scatter(existing_data['Customer_Age'], existing_data['Months_on_book'], color='tab:blue', alpha=0.4, label='Existing', s=20)
plt.scatter(attrited_data['Customer_Age'], attrited_data['Months_on_book'], color='tab:orange', alpha=0.6, label='Attrited', s=20)
plt.title('Customer Age vs Months on Book')
plt.xlabel('Customer Age')
plt.ylabel('Months on Book')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_scatter_age_months.png'), dpi=200)
plt.close()

# 11. Boxplot Single & Grouped
plt.figure(figsize=(5, 4))
plt.boxplot(df_churning['Customer_Age'])
plt.title('Boxplot Customer Age (All Customers)')
plt.ylabel('Age')
plt.xticks([1], labels=['All Customers'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_boxplot_single.png'), dpi=200)
plt.close()

plt.figure(figsize=(6, 4))
df_box = pd.concat([existing_data['Customer_Age'], attrited_data['Customer_Age']], keys=['Existing', 'Attrited'], axis=1)
df_box.boxplot()
plt.title('Boxplot Customer Age: Existing vs Attrited')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_boxplot_grouped.png'), dpi=200)
plt.close()

# 12. Violin Plot
plt.figure(figsize=(6, 4))
dict_violin = {'Existing': existing_data['Customer_Age'].values, 'Attrited': attrited_data['Customer_Age'].values}
plt.violinplot(list(dict_violin.values()), showmeans=True)
plt.title('Violin Plot Customer Age')
plt.ylabel('Age')
plt.xticks([1, 2], labels=list(dict_violin.keys()))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_violin_age.png'), dpi=200)
plt.close()

# 13. Subplot 2x2
plt.figure(figsize=(9, 8))
plt.subplot(2, 2, 1)
plt.boxplot(existing_data['Customer_Age'])
plt.ylabel('Age')
plt.xticks([1], labels=['Existing Customer'])

plt.subplot(2, 2, 2)
plt.boxplot(attrited_data['Customer_Age'])
plt.ylabel('Age')
plt.xticks([1], labels=['Attrited Customer'])

plt.subplot(2, 2, 3)
existing_data['Customer_Age'].plot.hist(color='tab:blue')
plt.ylabel('Count')
plt.xlabel('Age of Existing Customer')
plt.ylim([0, 2100])

plt.subplot(2, 2, 4)
attrited_data['Customer_Age'].plot.hist(color='tab:red')
plt.ylabel('Count')
plt.xlabel('Age of Attrited Customer')
plt.ylim([0, 2100])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_subplot_2x2.png'), dpi=200)
plt.close()

# 14. Annotation Plot
plt.figure(figsize=(6, 4))
data_ann = df_churning['Customer_Age']
plt.hist(data_ann, color='teal', edgecolor='black', bins=15)
plt.text(60, 1600, f"Median Age: {round(data_ann.median(), 2)}", style='italic', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.text(60, 1800, f"Mean Age: {round(data_ann.mean(), 2)}", style='italic', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.title('Histogram Customer Age dengan Anotasi')
plt.xlabel('Customer Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_annotation.png'), dpi=200)
plt.close()

# 15. Axis & Legend Setup Demo
plt.figure(figsize=(6.5, 4))
x_dates = np.array(["01/02/2020", "01/03/2020", "01/04/2020", "01/05/2020"])
y1_axis = np.array([3, 8, 1, 10])
y2_axis = np.array([13, 4, 10, 12])
plt.plot(x_dates, y1_axis, marker='o', label='Sensor 1')
plt.plot(x_dates, y2_axis, marker='x', label='Sensor 2')
plt.legend()
plt.ylabel('Suhu (°C)', fontsize=10)
plt.xlabel('Tanggal', fontsize=10)
plt.ylim([-20, 30])
plt.title('Pengaturan Axis & Ylim')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_axis_demo.png'), dpi=200)
plt.close()

# --- PREPROCESSING TITANIC FIGURES ---
df_titanic = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/train.csv')

plt.figure(figsize=(5, 5))
titanic_surv = df_titanic['Survived'].value_counts()
plt.pie(titanic_surv, labels=['Tidak Selamat (0)', 'Selamat (1)'], autopct='%.2f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
plt.title('Distribusi Kelas Target Survived (Titanic)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_titanic_survived_pie.png'), dpi=200)
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(9, 8))
axes = axes.ravel()
num_cols_tit = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for i, c in enumerate(num_cols_tit):
    axes[i].hist(df_titanic[c].dropna(), color='steelblue', edgecolor='black', bins=15)
    axes[i].set_title(c)
for j in range(len(num_cols_tit), 9):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_titanic_num_hist.png'), dpi=200)
plt.close()

# --- TUGAS BENGALURU FIGURES ---
URL_BENGALURU = 'https://raw.githubusercontent.com/dphi-official/Datasets/master/Bengaluru_House_Data.csv'
df_house = pd.read_csv(URL_BENGALURU)

def parse_sqft(val):
    if pd.isna(val): return np.nan
    text = str(val).strip()
    if '-' in text:
        parts = text.split('-')
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return np.nan
    num = ''
    for ch in text:
        if ch.isdigit() or ch == '.': num += ch
        elif num: break
    try: return float(num) if num else np.nan
    except: return np.nan

def parse_bhk(val):
    if pd.isna(val): return np.nan
    try: return float(str(val).split()[0])
    except: return np.nan

df_house['total_sqft_num'] = df_house['total_sqft'].apply(parse_sqft)
df_house['bhk'] = df_house['size'].apply(parse_bhk)

num_cols_beng = ['total_sqft_num', 'bath', 'balcony', 'bhk', 'price']
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.ravel()
for i, c in enumerate(num_cols_beng):
    axes[i].hist(df_house[c].dropna(), bins=30, color='darkseagreen', edgecolor='black')
    axes[i].set_title(c)
    axes[i].set_xlabel(c)
    axes[i].set_ylabel('Count')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_bengaluru_hist.png'), dpi=200)
plt.close()

corr_beng = df_house[num_cols_beng].corr(numeric_only=True)
plt.figure(figsize=(6, 5))
im = plt.imshow(corr_beng, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_beng.columns)), corr_beng.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_beng.columns)), corr_beng.columns)
plt.title('Correlation Heatmap (Bengaluru House Data)')
for (r, c), val in np.ndenumerate(corr_beng.values):
    plt.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_bengaluru_corr.png'), dpi=200)
plt.close()

print("All figures generated successfully!")
