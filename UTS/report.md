# Laporan Progress dan Hasil Analisis Proyek UTS (Kelompok 4)

**Mata Kuliah:** Praktikum Penambangan Data (SVPL261503)  
**Program Studi:** Sarjana Terapan (D.4) Teknologi Rekayasa Perangkat Lunak  
**Dosen Pengampu:** Dr.Eng. Ir. Ganjar Alfian, S.T., M.Eng. & Dr. Imam Fahrurrozi, S.T., M.Cs.  

**Anggota Kelompok 4:**
1. MUHAMMAD RAKAN HIBRIZI (muhammadrakanhibrizi2005@mail.ugm.ac.id)
2. JANUARSYAH AKBAR (januarsyahakbar791@gmail.com)
3. Devin Sotya Prathama (devinevos09@gmail.com)
4. SHINOSUKE ALEXANDER SWANDJAYA

**Judul Proyek:** Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Platform Crash Game Menggunakan Algoritma K-Means Clustering  
**Dataset:** Bustabit Gambling Behavior (Kaggle, 50.000 log transaksi taruhan)  
**Paper Acuan:** *AI Personalization and Its Influence on Online Gamblers' Behavior* (MDPI Behavioral Sciences, 2025 / PMC12189489)  
**Tautan Google Colab:** [PPD UTS Kelompok 4 Google Colab](https://colab.research.google.com/github/januarsyah901/PPD/blob/main/UTS/PPD_UTS_Kelompok_4_KMeans_Clustering.ipynb)  

---

## 1. Rangkuman Pekerjaan yang Telah Diselesaikan

### A. Data Pipeline & Otomasi Pemuatan Data
- Dataset `bustabit.csv` (3,29 MB, 50.000 baris) diintegrasikan langsung melalui tautan GitHub Raw publik:
  `https://raw.githubusercontent.com/januarsyah901/PPD/main/UTS/bustabit.csv`
- Notebook dirancang dengan skema *fallback* otomatis: jika dijalankan secara daring di Google Colab, dataset langsung diunduh lewat URL GitHub Raw. Jika dijalankan secara luring, skrip otomatis beralih membaca berkas lokal.

### B. Data Cleaning & Penanganan Missing Values
- Analisis awal mendeteksi kekosongan data (~42,53% atau 21.266 baris) pada kolom `CashedOut`, `Bonus`, dan `Profit`.
- Berdasarkan aturan main crash game Bustabit, kondisi ini bukan *data entry error*, melainkan penanda bahwa pemain mengalami *bust* (multiplier permainan meledak sebelum pemain sempat menekan tombol pencairan).
- Dilakukan penyesuaian domain:
  - Variabel biner status kemenangan `is_win = 1` jika `CashedOut` terisi, dan `0` jika kosong.
  - Imputasi keuntungan riil `actual_profit = Profit.fillna(-Bet)`, di mana saat kalah pemain merugi sebesar taruhan modal awal.

### C. Feature Engineering (User Profiling)
- Mengagregasi 50.000 log transaksi per ronde menjadi 4.149 profil pengguna unik (`Username`) untuk menangkap pola perilaku pemain:
  1. `total_bets`: Total frekuensi taruhan pemain (ukuran ketahanan dan intensitas bermain).
  2. `avg_bet`: Rata-rata besaran taruhan dalam satuan Bits (*stake size*).
  3. `win_rate`: Rasio kemenangan taruhan (total menang dibagi total taruhan).
  4. `avg_cashout`: Rata-rata target multiplier saat menang (indikator toleransi risiko).
  5. `total_profit`: Akumulasi keuntungan atau kerugian bersih.

### D. Penskalaan dan Transformasi Fitur
- Distribusi variabel `total_bets`, `avg_bet`, dan `avg_cashout` memiliki tingkat kemiringan (*skewness*) ekstrem akibat pencilan bernilai sangat besar.
- Diterapkan transformasi logaritmik `np.log1p` guna merapatkan rentang data tanpa menghilangkan urutan informasi.
- Dilanjutkan dengan standarisasi menggunakan `StandardScaler` (Z-score scaling, rata-rata 0 dan variansi 1) agar perhitungan jarak Euclidean pada algoritma K-Means tidak didominasi variabel tertentu.

### E. Evaluasi Komprehensif dan Pelatihan K-Means
- Menjalankan pengujian rentang $k = 2$ hingga $k = 8$ menggunakan 3 parameter evaluasi: Elbow Method (Inertia/WCSS), Silhouette Score, dan Davies-Bouldin Index (DBI).
- Melatih model final K-Means dengan parameter optimal $k = 4$.
- Melakukan pemetaan reduksi dimensi visual menggunakan Principal Component Analysis (PCA 2D).

---

## 2. Tabel Hasil Evaluasi Model Clustering

| Jumlah Klaster ($k$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Keterangan Evaluasi |
| :---: | :---: | :---: | :---: | :--- |
| 2 | 12880,17 | 0,2652 | 1,6244 | Pemisahan terlalu kasar |
| 3 | 9848,39 | 0,3001 | 1,3029 | Mulai stabil |
| **4** | **7840,79** | **0,3189** | **1,1713** | **Optimal (Puncak Silhouette & DBI Rendah)** |
| 5 | 6270,25 | 0,3141 | 1,0361 | Silhouette mulai menurun |
| 6 | 5580,01 | 0,3020 | 1,0666 | Kerapatan klaster menurun |
| 7 | 5036,15 | 0,2996 | 1,0255 | Klaster makin terfragmentasi |
| 8 | 4511,54 | 0,3072 | 1,0268 | Sub-klaster redundan |

---

## 3. Temuan Karakteristik 4 Klaster Perilaku Pemain

Berikut rincian profil rata-rata tiap klaster pada skala nilai riil (original scale):

| Klaster | Jumlah Pemain | Persentase | Rata-rata Taruhan (Bets) | Rata-rata Nominal (Bits) | Win Rate | Rata-rata Multiplier | Total Profit Riil (Bits) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1.067 | 25,7% | 33,24 | 2.838,22 | 63,0% | 1,59x | +4.963,32 |
| **1** | 1.732 | 41,7% | 2,62 | 6.389,10 | 88,9% | 1,43x | +4.022,74 |
| **2** | 354 | 8,5% | 21,49 | 2.083,20 | 30,2% | 7,13x | -2.335,37 |
| **3** | 996 | 24,0% | 2,40 | 5.292,29 | 8,9% | 1,12x | -9.186,16 |

### Deskripsi Profil Perilaku:

1. **Klaster 0: Active Grinders (Taruhan Konsisten, Win Rate Stabil, Profit Positif)**
   - Karakteristik: Pemain paling aktif dengan frekuensi rata-rata tertinggi (33,24 ronde).
   - Strategi: Menetapkan target multiplier realistis dan rendah (1,59x). Menghasilkan tingkat kemenangan stabil di angka 63,0% dan membukukan keuntungan kumulatif rata-rata tertinggi (+4.963,32 bits).
   - Interpretasi: Mewakili pemain disiplin atau pengguna bot taruhan terprogram yang mengejar margin keuntungan kecil secara konsisten.

2. **Klaster 1: Casual Conservative Winners (Frekuensi Rendah, Win Rate Sangat Tinggi, Multiplier Rendah)**
   - Karakteristik: Kelompok pemain terbesar (41,7% dari populasi) dengan waktu bermain singkat (rata-rata 2,62 ronde).
   - Strategi: Menarik taruhan sesegera mungkin di batas aman (rata-rata 1,43x).
   - Interpretasi: Pemain kasual yang disiplin dan berhenti bermain setelah memperoleh keuntungan cepat (win rate mencapai 88,9% dan profit +4.022,74 bits).

3. **Klaster 2: High-Risk Seekers / Chasers (Multiplier Ekstrem, Win Rate Rendah, Rugi Bersih)**
   - Karakteristik: Kelompok spekulatif ekstrem (8,5% dari populasi).
   - Strategi: Menunggu target multiplier fantastis hingga rata-rata 7,13x (beberapa di antaranya mengejar puluhan hingga ratusan kali lipat).
   - Dampak Finansial: Win rate anjlok hingga 30,2%, menyebabkan akumulasi kerugian rata-rata -2.335,37 bits meskipun volume taruhan cukup tinggi (21,49 ronde).
   - Interpretasi: Menunjukkan gejala klasik kecanduan taruhan (*loss chasing* dan *sensation seeking*), kelompok yang paling membutuhkan intervensi sistem perjudian bertanggung jawab (*responsible gambling*).

4. **Klaster 3: Unlucky Casuals (Frekuensi Rendah, Win Rate Sangat Rendah, Cepat Rugi)**
   - Karakteristik: Pemain kasual baru atau eksperimental (24,0% populasi) dengan rata-rata 2,40 ronde.
   - Dampak Finansial: Menghadapi win rate sangat rendah (hanya 8,9%) dan menderita kerugian bersih tercepat (-9.186,16 bits).
   - Interpretasi: Pemain yang mengalami kekalahan beruntun di awal mencoba permainan lalu segera meninggalkan platform.

---

## 4. Kepatuhan Terhadap Rubrik Penugasan (CPMK1 & CPMK2)

1. **Validasi Dataset (CPMK1):** Terpenuhi penuh. Dataset Kaggle terkonfirmasi dan disitasi secara resmi dalam jurnal ilmiah MDPI Behavioral Sciences (Februari 2025) untuk topik pemodelan perilaku penjudi.
2. **Batasan Implementasi Kode (CPMK2):** Kode Google Colab dikerjakan tepat sampai tahap **Model Evaluation & Interpretation**. Tidak melampaui batas penugasan (tanpa *deployment* atau pembuatan aplikasi web).
3. **Kesiapan Materi Presentasi:** Draf struktur slide 10 menit sudah dirancang pada berkas `outline_presentasi_kelompok_4.md` dan siap dialihkan ke Google Slides atau PowerPoint.
