# Rencana Slide Presentasi UTS Praktikum Penambangan Data (Kelompok 4)

**Topik:** Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Platform Crash Game Menggunakan Algoritma K-Means Clustering  
**Waktu:** 10 Menit Presentasi + 5 Menit Tanya Jawab  
**Anggota Kelompok:**
1. MUHAMMAD RAKAN HIBRIZI
2. JANUARSYAH AKBAR
3. Devin Sotya Prathama
4. SHINOSUKE ALEXANDER SWANDJAYA

---

### Slide 1: Judul & Anggota Tim (1 Menit)
- **Judul Proyek:** Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Platform Crash Game Menggunakan Algoritma K-Means Clustering
- **Mata Kuliah:** Praktikum Penambangan Data (D.4 TRPL SV UGM)
- **Dosen Pengampu:** Dr.Eng. Ir. Ganjar Alfian, S.T., M.Eng. & Dr. Imam Fahrurrozi, S.T., M.Cs.
- **Nama Anggota:** Tim Kelompok 4

---

### Slide 2: Pendahuluan & Latar Belakang Masalah (2 Menit)
- **Fenomena Crash Game (Bustabit):** Mekanisme permainan pengganda taruhan real-time berbasis Bitcoin/Bits, di mana pemain harus menekan tombol cashout sebelum grafik bust.
- **Permasalahan Utama:**
  1. Tingginya volatilitas dan adiksi taruhan daring akibat dorongan psikologis ilusi kendali (*illusion of control*).
  2. Kebutuhan segmentasi profil perilaku pemain untuk mendeteksi pola taruhan berisiko ekstrem versus pemain kasual.
- **Tujuan Analisis:** Mengidentifikasi kelompok perilaku dan toleransi risiko pemain menggunakan pendekatan pembelajaran tanpa pengawasan (*unsupervised learning*).
- **Paper Acuan:** *AI Personalization and Its Influence on Online Gamblers' Behavior* (MDPI Behavioral Sciences, 2025).

---

### Slide 3: Karakteristik Dataset & Preprocessing (2 Menit)
- **Sumber Data:** Kaggle (Gambling Behavior Bustabit, 50.000 log transaksi).
- **Penanganan Missing Values:**
  - Kolom `CashedOut`, `Bonus`, dan `Profit` kosong sebanyak ~42.5%.
  - Fakta domain: Nilai kosong menandakan pemain mengalami bust (kalah).
  - Solusi imputasi logis: `actual_profit = -Bet`.
- **Feature Engineering (Agregasi Profil per Pemain):**
  - Mengubah 50.000 log ronde transaksi menjadi 4.149 profil pengguna unik (`Username`).
  - Fitur kunci: `total_bets` (frekuensi), `avg_bet` (ukuran taruhan), `win_rate` (rasio menang), dan `avg_cashout` (target pengganda).
- **Transformasi & Standarisasi:**
  - Log-transformation (`np.log1p`) untuk mengatasi kemiringan distribusi (*skewness*) ekstrem.
  - Penskalaan fitur dengan `StandardScaler` (Z-score).

---

### Slide 4: Metodologi & Penentuan Klaster Optimal (2 Menit)
- **Algoritma:** K-Means Clustering dengan jarak Euclidean.
- **Model Evaluation Multi-Metrik:**
  1. **Elbow Method (WCSS / Inertia):** Penurunan tajam melandai pada rentang $k=3$ hingga $k=5$.
  2. **Silhouette Score:** Puncak tertinggi dicapai pada **$k=4$ (skor 0.3189)**.
  3. **Davies-Bouldin Index (DBI):** Titik rendah optimal pada **$k=4$ (1.1713)**.
- **Keputusan:** Memilih **$k = 4$** karena memberikan separasi matematis terbaik dan interpretasi psikologis paling tajam.

---

### Slide 5: Hasil Segmentasi & Interpretasi 4 Profil Perilaku (2 Menit)
- **Klaster 0: Active Grinders (1.067 pemain, 25.7%)**
  - Profil: Taruhan sangat sering (rata-rata 33 taruhan), target multiplier konservatif (1.59x), win rate stabil (63%), dan profit positif konsisten (+4.963 bits).
- **Klaster 1: Casual Conservative Winners (1.732 pemain, 41.7%)**
  - Profil: Frekuensi rendah (2.6 taruhan), win rate tertinggi (89%), cashout cepat di multiplier rendah (1.43x), profit positif (+4.022 bits).
- **Klaster 2: High-Risk Seekers / Chasers (354 pemain, 8.5%)**
  - Profil: Mengejar multiplier ekstrem (rata-rata 7.13x), win rate rendah (30%), akumulasi kerugian rata-rata -2.335 bits. Kelompok paling rentan adiksi.
- **Klaster 3: Unlucky Casuals (996 pemain, 24.0%)**
  - Profil: Pemain singkat (2.4 taruhan) dengan tingkat kemenangan sangat rendah (9%) dan kerugian cepat (-9.186 bits).
- **Visualisasi:** Pemetaan 2D menggunakan Principal Component Analysis (PCA).

---

### Slide 6: Kesimpulan & Saran (1 Menit)
- **Kesimpulan:**
  1. Algoritma K-Means terbukti efektif mengelompokkan 4.149 pemain ke dalam 4 spektrum risiko yang terpisah jelas.
  2. Ditemukan kelompok *High-Risk Seekers* yang konsisten mengejar multiplier tinggi meskipun menanggung kerugian sistematis.
- **Kepatuhan Batasan CPMK2:** Pengerjaan berhenti tepat pada tahap **Model Evaluation & Interpretation**.
- **Saran Pengembangan Lanjutan:**
  - Eksplorasi algoritma clustering berbasis densitas (DBSCAN) untuk memisahkan noise pencilan ekstrem.
  - Analisis pola transisi dinamis pemain (*gambler's fallacy*) dari waktu ke waktu.
