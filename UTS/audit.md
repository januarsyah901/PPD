# Audit dan Hasil Revisi Materi Presentasi (Kelompok 4)

**Topik:** Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Crash Game Bustabit  
**Mata Kuliah:** Praktikum Penambangan Data (D.4 TRPL SV UGM)  
**Berkas yang Diaudit:** `ppt.md` dan `ppt.pdf` (ekspor Canva 11 slide)  
**Tanggal Audit:** 8 Oktober 2026  

---

## Bagian 1: Daftar Temuan dan Masalah Kritis

Berikut rincian kelemahan, salah ketik, dan ketidakkonsistenan yang ditemukan pada berkas presentasi:

### 1. Judul dan Penomoran Klaster Bertentangan (Slide 8 / Halaman 8 PDF)
* **Masalah:** Judul slide tertulis "Empat Klaster Perilaku", padahal isi teks memaparkan 5 kelompok ($n=297, 235, 226, 244, 109$).
* **Masalah Penomoran:** Indeks klaster di kolom kiri tertulis 0, 1, 2, 3, tetapi di kolom kanan klaster terakhir berlabel **5** (*Target Cashout Tinggi*). Pada algoritma K-Means dengan $k=5$, penomoran indeks adalah 0, 1, 2, 3, 4. Angka 4 hilang dan melompat ke 5.
* **Tindakan Koreksi:** Ubah judul menjadi "Karakteristik Lima Klaster Perilaku" dan perbaiki label klaster terakhir menjadi angka **4**.

### 2. Grafik Laptop di Slide Penutup Belum Sinkron (Slide 11 / Halaman 11 PDF)
* **Masalah:** Pada mockup laptop di sebelah kanan, teks header menampilkan label besar **`k = 5`**, tetapi diagram batang di bawahnya hanya memuat 4 item (Segmen 1, Segmen 2, Segmen 3, Segmen 4). Segmen 5 hilang.
* **Tindakan Koreksi:** Tambahkan satu batang grafik lagi untuk **Segmen 5** pada desain Canva agar selaras dengan $k=5$.

### 3. Duplikasi Judul dan Subjudul (Slide 9 dan Slide 10 / Halaman 9 & 10 PDF)
* **Masalah:** Slide 9 dan Slide 10 memakai judul dan subjudul yang persis sama ("Temuan dan Pemanfaatan" serta "Edukasi dan literasi publik berbasis bukti untuk kampanye anti-judol"). Kolom kiri Slide 10 juga terlihat kosong.
* **Tindakan Koreksi:**
  * Slide 9 difokuskan ke fakta analisis: Ubah judul menjadi **Temuan Kunci Dinamika Taruhan**.
  * Slide 10 difokuskan ke aksi dan limitasi: Ubah judul menjadi **Pemanfaatan Edukasi & Batasan Riset**.

### 4. Salah Ketik Kata (Slide 4 / Halaman 4 PDF)
* **Masalah:** Poin terakhir tertulis *"Melengkapi paper rujkan:"* (kurang huruf 'u').
* **Tindakan Koreksi:** Perbaiki menjadi *"Melengkapi paper rujukan:"*.

### 5. Format Variabel Kode (Slide 5 / Halaman 5 PDF)
* **Masalah:** Poin 01 tertulis `actual-profit` (memakai tanda minus/strip).
* **Tindakan Koreksi:** Ganti dengan `actual_profit` (memakai garis bawah) agar sesuai dengan nama kolom pada skrip Python.

### 6. Catatan Sinkronisasi Notebook Colab
* **Status:** Slide presentasi menggunakan penyaringan pemain $\ge 10$ taruhan ($n=1.111$) dan memilih model $k=5$ (Silhouette 0,2588; DBI 1.1430). Ini keputusan metodologis yang tepat karena menyaring 66% pemain acak.
* **Kewajiban:** Pastikan berkas `PPD_UTS_Kelompok_4_KMeans_Clustering.ipynb` di Google Colab sudah diperbarui kodenya dengan filter ini sebelum sesi presentasi, agar angka pada slide identik dengan angka pada notebook jika dibuka dosen.

---

## Bagian 2: Naskah Lengkap PPT Hasil Revisi (Siap Pakai untuk Canva)

Teks di bawah ini sudah dibersihkan dari seluruh kesalahan di atas dan bisa langsung disalin ke elemen teks Canva maupun berkas catatan presentasi.

```markdown
# Slide 1: Cover
### Segmentasi Perilaku dan Profil Risiko Penjudi Daring
Studi pada Crash Game Bustabit Menggunakan Algoritma K-Means Clustering

Kelompok 4:
- Muhammad Rakan Hibrizi
- Januarsyah Akbar
- Devin Sotya Prathama
- Shinosuke Alexander Swandjaya

Praktikum Penambangan Data (SVPL261503)
D.4 Teknologi Rekayasa Perangkat Lunak, Sekolah Vokasi UGM

---

# Slide 2: Latar Belakang dan Tujuan
Banyak orang terjerat judi daring karena ilusi bisa menang konsisten dengan trik tertentu.

Tujuan penelitian ini adalah memetakan segmen perilaku pemain crash game Bustabit dari data transaksi, lalu mendeskripsikan profil hasil tiap segmen (median profit dan persentase pemain rugi).

Acuan utama:
AI Personalization and Its Influence on Online Gamblers' Behavior (MDPI Behavioral Sciences, 2025)

---

# Slide 3: Karakteristik Dataset Bustabit
- Sumber data: Kaggle (Gambling Behavior Bustabit)
- Total data: 50.000 log transaksi taruhan

DATASET OVERVIEW:
- Kolom CashedOut: Kosong ± 42,53%
- Kolom Bonus: Kosong ± 42,53%
- Kolom Profit: Kosong ± 42,53%

Catatan Domain:
Kolom CashedOut, Bonus, dan Profit kosong pada sekitar 42,53% transaksi.
Nilai kosong menandakan transaksi bust (kalah), bukan kesalahan input data.

---

# Slide 4: Mengapa Clustering, Bukan Regresi
- Menemukan segmen perilaku yang berbeda, bukan satu hubungan rata-rata populasi.
- Profil dibentuk dari beberapa fitur sekaligus (frekuensi, nominal taruhan, target cashout, win rate), tanpa membatasi satu variabel target tunggal.
- Data tanpa label historis, sehingga murni bersifat unsupervised learning.
- Luaran berupa profil segmen deskriptif untuk profiling tindakan, bukan rumus prediksi.
- Melengkapi paper rujukan: paper rujukan menguji hipotesis dengan uji statistik dan regresi makro, sedangkan proyek ini mendeskripsikan sub-segmen mikro pemain.

---

# Slide 5: Preprocessing dan Feature Engineering
Transformasi dari Log Transaksi Menjadi Profil Pemain:

01. Label Status & Profit Riil
    Membuat label is_win dan actual_profit (kondisi kalah = minus nilai Bet).
02. Audit Kualitas Data
    Sebanyak 43,9% pemain hanya bertaruh <= 2 kali sehingga metriknya bersifat kebetulan acak. Analisis difokuskan pada pemain dengan minimal 10 taruhan (1.111 dari 4.149 pemain atau 26,8% populasi).
03. Reduksi Kemiringan (Skewness)
    Menerapkan transformasi log1p pada total_bets, avg_bet, dan avg_cashout.
04. Standarisasi Fitur
    Penskalaan nilai menggunakan StandardScaler (rata-rata 0, variansi 1).
05. Imputasi Batas Aman
    Nilai avg_cashout bagi pemain yang belum pernah menang diisi angka 1,0.

---

# Slide 6: Penentuan Jumlah Klaster
- Evaluasi pengujian rentang k = 2 sampai 7.
- Tiga parameter evaluasi: Elbow Method (Inertia), Silhouette Score, dan Davies-Bouldin Index.
- Kompromi seimbang antara pemisahan matematis, stabilitas klaster, dan keterbacaan profil perilaku.

MODEL TERPILIH (k = 5):
- Silhouette Score: 0,2588
- Davies-Bouldin Index: 1,1430

---

# Slide 7: Mengapa Bukan k = 4?
- Pada analisis awal (seluruh 4.149 pemain tanpa filter), Silhouette Score sempat mencapai puncak di k = 4 (0,3189).
- Masalah klaster awal: kelompok terbentuk karena jumlah observasi, bukan pola perilaku. Dua klaster awal berisi 2.728 pemain (66% populasi) dengan median hanya 2 taruhan (satu kelompok kebetulan menang win_rate = 1,00 dan kelompok lain kebetulan kalah win_rate = 0). Itu hasil acak dari 2 kali coba, bukan pola gaya bermain.
- Setelah audit filter >= 10 taruhan diterapkan, model k = 4 terbukti lebih buruk dibanding k = 5 pada kedua metrik evaluasi (Silhouette 0,2507 vs 0,2588; DBI 1,2585 vs 1,1430).

---

# Slide 8: Karakteristik Lima Klaster Perilaku
0. Target Rendah, Win Rate Tinggi (n = 297)
   Target cashout 1,19x, win rate 0,80, median profit +12,7 bits, 40% pemain rugi.
1. Frekuensi Tinggi, Target Rendah (n = 235)
   Median 69 taruhan, target 1,37x, win rate 0,68, median profit -10,2 bits, 51% pemain rugi.
2. Taruhan Nominal Tinggi (n = 226)
   Rata-rata bet ± 2.225 bits, target 1,49x, win rate 0,60, median profit +1.673 bits, 43% pemain rugi.
3. Target Menengah (n = 244)
   Target cashout 2,30x, win rate 0,33, median profit -65,5 bits, 61% pemain rugi.
4. Target Cashout Tinggi (n = 109)
   Target cashout 9,05x, win rate 0,14, median profit -172,6 bits, 61% pemain rugi.

---

# Slide 9: Temuan Kunci Dinamika Taruhan
Edukasi dan Literasi Publik Berbasis Bukti Nyata:

- Tingkat Risiko Transaksi:
  Sebanyak 42,53% dari 50.000 transaksi taruhan berakhir bust (kalah).
- Ilusi Multiplier Tinggi:
  Semakin tinggi target cashout yang dipasang, win rate anjlok drastis (dari 0,80 turun ke 0,33, lalu jatuh ke 0,14). Median profit negatif konsisten terjadi pada klaster dengan target >= 2x pada filter min_bets 5, 10, maupun 20.
- Faktor Skala Modal:
  Taruhan nominal besar tidak otomatis rugi karena kelompok ini mengamankan target rendah (1,49x).
- Dominasi Volume Transaksi:
  Klaster Taruhan Nominal Tinggi menguasai 88,6% volume perputaran dana sampel, membuktikan adanya konsentrasi modal pada segelintir akun besar (whales), sementara mayoritas pemain kasual tetap merugi.

---

# Slide 10: Pemanfaatan Edukasi & Batasan Riset
Pemanfaatan untuk Publik:
- Bahan edukasi dan literasi digital berbasis data empiris untuk membongkar mitos kemenangan judi daring.
- Pemetaan proksi perilaku untuk sistem deteksi keuangan: pola loss chasing di Klaster 4 dapat diadaptasi untuk mendeteksi anomali mutasi rekening atau e-wallet (seperti top-up panik berulang kali dalam hitungan menit).
- Forensik digital dan penegakan hukum: membantu otoritas (PPATK / Kepolisian) memilah rekening korban adiksi dari rekening sindikat penampung.
- Dasar profiling pola taruhan untuk membantu instrumen konseling adiksi dan regulasi perlindungan konsumen.
- Data ini bersifat observasional, bukan bukti langsung bahwa platform menarget tipe pemain tertentu secara personal.

Batasan Penelitian:
- Temuan ini berlaku khusus untuk kelompok pemain aktif (>= 10 taruhan) sehingga terdapat potensi bias ketahanan (survivorship bias).
- Struktur kepadatan klaster berada pada rentang lemah hingga sedang (Silhouette 0,25 - 0,29).

---

# Slide 11: Kesimpulan dan Batasan Pengerjaan
Kesimpulan Model:
- Model k = 5 dipilih sebagai titik temu terbaik antara pemisahan klaster, stabilitas sebaran, dan interpretasi psikologis. Model k = 2 unggul pada Silhouette murni dan k = 6 unggul pada DBI, sehingga k = 5 tidak diklaim mutlak optimal melainkan paling representatif.

Kepatuhan Batasan Rubrik (CPMK2):
- Penelitian dibatasi secara tegas hingga tahap evaluasi dan interpretasi model.
- Tidak mencakup tahap deployment aplikasi atau pembuatan antarmuka pengguna pada fase UTS ini.

Grafik Model (Mockup Laptop):
- Ringkasan Model: k = 5
  - Segmen 1 (Target Rendah, Win Rate Tinggi)
  - Segmen 2 (Frekuensi Tinggi, Target Rendah)
  - Segmen 3 (Taruhan Nominal Tinggi)
  - Segmen 4 (Target Menengah)
  - Segmen 5 (Target Cashout Tinggi)
```
