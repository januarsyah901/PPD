# Segmentasi Perilaku dan Profil Risiko Penjudi Daring



### Studi pada Crash Game Bustabit menggunakan algoritma K-Means Clustering



**Kelompok 4**

* Muhammad Rakan Hibrizi


* Januarsyah Akbar


* Devin Sotya Prathama


* Shinosuke Alexander Swandjaya



**Praktikum Penambangan Data (SVPL261503)**

**D.4 Teknologi Rekayasa Perangkat Lunak**

> 🖼️ **[Image/Diagram/Chart Description]:** Tampilan antarmuka (UI) dari Crash Game Bustabit berlatar belakang gelap dengan kurva pengganda (*multiplier*) berwarna biru sian yang menanjak secara eksponensial hingga angka 3.47x. Di bawah kurva terdapat daftar taruhan langsung dengan kolom User, Bet, Cashout, dan Profit: User123 (Bet: 10.00, Cashout: 1.45x, Profit: 4.50); User 456 (Bet: 20.00, Cashout: 3.21, Profit: 44.20); 799 (Bet: 15.00, Cashout: 2.10x, Profit: 16.50); User321 (Bet: 8.00, Cashout: kosong, Profit: -8.00); User654 (Bet: 25.00, Cashout: 1.88x, Profit: 22.00); User 987 (Bet: 12.00, Cashout: 4.05x, Profit: 36.60); dan User147 (Bet: 5.00, Cashout: kosong, Profit: -5.00). Bagian bawah menampilkan tombol taruhan cepat (1.00, 2.00, 5.00, 10.00) serta tombol aksi "BET".
> 
> 

---

# Latar Belakang dan Tujuan



Banyak orang terjerat judi daring karena ilusi bisa menang konsisten dengan trik tertentu.

Tujuan penelitian ini adalah memetakan segmen perilaku pemain crash game Bustabit dari data transaksi, lalu mendeskripsikan profil hasil tiap segmen (median profit dan % pemain rugi).

**Acuan utama:**

*AI Personalization and Its Influence on Online Gamblers' Behavior* (MDPI Behavioral Sciences, 2025)

> 🖼️ **[Image/Diagram/Chart Description]:** Ilustrasi antarmuka Crash Game yang menampilkan kurva eksponensial berwarna biru toska mencapai angka 3.47x. Di sebelah kanan grafik terlihat tabel ringkasan taruhan pengguna (User, Bet, Cashout, Profit) dan di bagian bawahnya terdapat kontrol tombol nilai taruhan serta tombol eksekusi "BET".
> 
> 

---

# Dataset Bustabit



* Menggunakan data dari **Kaggle**

* Terdiri dari **50.000 transaksi taruhan**


### DATASET OVERVIEW



| Kolom | Kondisi |
| --- | --- |
| CashedOut | Kosong ± 42,53% |
| Bonus | Kosong ± 42,53% |
| Profit | Kosong ± 42,53% |

> Kolom CashedOut, Bonus, dan Profit kosong sekitar **42,53% TRANSAKSI**.
> 
> 
> Nilai kosong berarti **transaksi bust (kalah)**, bukan data hilang.
> 
> 

---

# Mengapa Clustering, Bukan Regresi



* Menemukan segmen perilaku yang berbeda, bukan satu hubungan rata-rata populasi.


* Profil dibentuk dari beberapa fitur sekaligus (frekuensi, nominal taruhan, target cashout, win rate), tanpa memilih satu variabel target.


* Data tanpa label, sehingga bersifat **unsupervised**.


* Luaran berupa profil segmen deskriptif, bukan rumus prediksi.


* **Melengkapi paper rujukan:** paper menguji hipotesis dengan uji statistik dan regresi, proyek ini mendeskripsikan segmen pemain. Bukan replikasi, dan tidak menguji hipotesis paper.



---

# Preprocessing dan Feature Engineering



### Dari transaksi menjadi profil pemain



1. **01** Buat label `is_win` dan `actual-profit` (kalah = minus Bet)


2. **02** Audit: 43,9% pemain hanya punya $\le 2$ taruhan, sehingga metrik perilakunya tidak stabil. Analisis dibatasi pada pemain dengan minimal 10 taruhan (1.111 dari 4.149 pemain, 26,8%).


3. **03** Transformasi `log1p` untuk mengurangi *skewness*

4. **04** Standarisasi dengan `StandardScaler`

5. **05** Catatan: `avg_cashout` pemain yang belum pernah menang diisi 1,0 (imputasi).



> 🖼️ **[Image/Diagram/Chart Description]:** Diagram alur transformasi data horizontal yang menghubungkan empat tahapan utama: ikon tumpukan berkas "Data Transaksi" $\rightarrow$ "Preprocessing" $\rightarrow$ "Feature Engineering" $\rightarrow$ kartu ikon "Profil Pemain". Pada latar belakang atas terdapat tampilan tabel matriks berisi angka-angka mentah nilai taruhan, koefisien pengganda, dan profit.
> 
> 

---

# Penentuan Jumlah Klaster Optimal



* Evaluasi $k=2$ sampai 7


* Metode Elbow (Inertia), Silhouette Score, dan Davies-Bouldin Index


* Kompromi antara pemisahan klaster, stabilitas, dan interpretabilitas. Tidak diklaim "optimal".



### 5 TERPILIH



* **Silhouette Score:** 0,2588


* **Davies-Bouldin Index:** 1.1430



---

# Mengapa Bukan 4?



* Pada analisis awal (seluruh 4.149 pemain) Silhouette puncak di $k=4$ (0,3189).


* $k=4$ klaster awal dibedakan oleh jumlah observasi, bukan perilaku: dua klaster berisi 2.728 pemain (66%), median hanya 2 taruhan, satu `win_rate = 1,00` dan satu `win_rate = 0`. Itu hasil acak dari 2 taruhan, bukan segmen perilaku.


* Dilakukan filter $\ge 10$ taruhan, $k=4$ lebih buruk dari $k=5$ di kedua metrik (Silhouette 0,2507 vs 0,2588; DBI 1,2585 vs 1,1430).



---

# Empat Klaster Perilaku



* **0 - Target Rendah, Win Rate Tinggi**
($n=297$): cashout 1,19x, win rate 0,80, median profit +12,7, 40% rugi.


* **1 - Frekuensi Tinggi, Target Rendah**
($n=235$): median 69 taruhan, 1,37x, win rate 0,68, median profit -10,2, 51% rugi.


* **2 - Taruhan Nominal Tinggi**
($n=226$): avg bet $\approx 2.225$ bits, 1,49x, win rate 0,60, median profit +1.673, 43% rugi.


* **3 - Target Menengah**
($n=244$): 2,30x, win rate 0,33, median profit -65,5, 61% rugi.


* **5 - Target Cashout Tinggi**
($n=109$): 9,05x, win rate 0,14, median profit -172,6, 61% rugi.



> 🖼️ **[Image/Diagram/Chart Description]:** Foto sebuah ponsel pintar yang tergeletak miring di sebelah tumpukan koin kasino/poker chips di atas permukaan meja hitam. Layar smartphone menampilkan aplikasi Crash Game dengan grafik kurva bernuansa hijau kebiruan (cyan) menyala mencapai nilai 3.47x dan deretan tombol pilihan taruhan cepat di bawahnya.
> 
> 

---

# Temuan dan Pemanfaatan



### Edukasi dan literasi publik berbasis bukti untuk kampanye anti-judol



* 42,53% dari 50.000 taruhan berakhir bust (tingkat transaksi, tergantung target yang dipilih pemain).


* Makin tinggi target cashout, makin rendah win rate ($0,80 \rightarrow 0,33 \rightarrow 0,14$). Sebagian bersifat mekanis karena desain game. Median profit negatif pada klaster bertarget $\ge 2\text{x}$ di min_bets = 5, 10, dan 20.


* Taruhan nominal besar tidak otomatis rugi: median profit klaster itu positif.


* Klaster Taruhan Nominal Tinggi memegang 88,6% volume taruhan, dan ROI gabungan sampel ini positif (sekitar +5%). Jadi data ini tidak mendukung klaim bahwa pemain pasti rugi.



> 🖼️ **[Image/Diagram/Chart Description]:** Kumpulan grafik dan poster kampanye anti-judi daring. Terdapat lencana lingkaran hitam dengan angka persentase "42,53%", spanduk peringatan merah berbunyi "BERHENTI SEBELUM TERLAMBAT - Judi Daring Merugikan. Lindungi Diri dan Orang Lain.", serta lambang tameng perisai hijau berteks "PILIH KEUANGAN SEHAT BUKAN JUDI DARING".
> 
> 

---

# Pemanfaatan Edukasi & Batasan Riset

### Pemanfaatan untuk Publik dan Regulasi

* Bahan edukasi dan literasi digital berbasis data empiris untuk membongkar mitos kemenangan judi daring.

* Pemetaan proksi perilaku untuk sistem deteksi keuangan: pola loss chasing di Klaster 4 dapat diadaptasi untuk mendeteksi anomali mutasi rekening atau e-wallet (seperti top-up panik berulang kali dalam hitungan menit).

* Forensik digital dan penegakan hukum: membantu otoritas (PPATK / Kepolisian) memilah rekening korban adiksi dari rekening sindikat penampung.

* Dasar profiling pola taruhan untuk instrumen konseling adiksi dan regulasi perlindungan konsumen. Bukan bukti langsung bahwa platform menarget tipe pemain tertentu secara personal.

### Batasan

* Hasil hanya berlaku untuk pemain $\ge 10$ taruhan (ada potensi *survivorship bias*), dan struktur klaster lemah sampai sedang (Silhouette 0,25 - 0,29).

> 🖼️ **[Image/Diagram/Chart Description]:** Visualisasi poster kampanye publik di sisi kanan: spanduk tanda bahaya merah bertuliskan "BERHENTI SEBELUM TERLAMBAT", pesan edukasi "Judi Daring Merugikan. Lindungi Diri dan Orang Lain.", serta ilustrasi perisai pengawasan transaksi keuangan "DETEKSI DINI & PERLINDUNGAN KEUANGAN".
> 
> 

---

# Kesimpulan dan Batasan



* $k=5$ dipilih sebagai kompromi antara pemisahan klaster, stabilitas, dan interpretabilitas. $k=2$ unggul pada Silhouette dan $k=6$ unggul pada DBI, sehingga $k=5$ tidak diklaim "optimal".


* Penelitian dibatasi hingga tahap evaluasi dan interpretasi model.


* Tidak mencakup deployment aplikasi atau antarmuka pengguna.



**Terima kasih**

> 🖼️ **[Image/Diagram/Chart Description]:** Diagram alur pohon ringkasan model bertajuk "Ringkasan Model Segmentasi Perilaku k=5". Di bawah node utama, terdapat percabangan menuju empat kotak klasifikasi segmen berlabel "Segmen 1", "Segmen 2", "Segmen 3", dan "Segmen 4".
> 
>