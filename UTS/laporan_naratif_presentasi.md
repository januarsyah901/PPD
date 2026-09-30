# Panduan Narasi Presentasi Proyek UTS (Kelompok 4)
### Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Platform Crash Game Menggunakan Algoritma K-Means Clustering

Naskah ini disusun secara naratif runut mengikuti 10 seksi kode yang ada pada notebook Google Colab. Gaya bahasa dibuat mengalir dan komunikatif, sehingga bisa langsung dibaca saat presentasi maupun dijadikan contekan bicara (*speaker notes*).

---

### Bagian 1: Pembuka, Import Library, dan Pengaturan Lingkungan
Halo semuanya, perkenalkan kami dari Kelompok 4. Pada proyek Praktikum Penambangan Data kali ini, kami mengangkat topik segmentasi perilaku dan profil risiko penjudi daring pada platform crash game Bustabit menggunakan K-Means Clustering.

Di bagian pertama notebook, kami menyiapkan lingkungan kerja Python dengan pustaka standar sains data. Kami menggunakan Pandas dan NumPy untuk olah data, Matplotlib dan Seaborn untuk visualisasi, serta Scikit-Learn untuk kebutuhan penskalaan, pemodelan K-Means, reduksi dimensi PCA, dan penghitungan metrik evaluasi seperti Silhouette Score dan Davies-Bouldin Index.

---

### Bagian 2: Pemuatan Data (Data Loading)
Masuk ke bagian kedua, kami memuat berkas `bustabit.csv`. Dataset ini berukuran 3,29 MB dan berisi 50.000 baris log transaksi taruhan. 

Agar notebook ini fleksibel dan bisa dijalankan siapa saja di Google Colab tanpa repot mengunggah file manual, kami menghubungkannya langsung ke tautan GitHub Raw publik di repositori kami. Kami juga menyematkan blok proteksi otomatis (*try-except*). Jika notebook dijalankan tanpa sambungan internet, sistem otomatis beralih mencari file lokal di direktori kerja. Saat cell ini dieksekusi, seluruh 50.000 data transaksi langsung terbaca sempurna.

---

### Bagian 3: Eksplorasi Data dan Pemahaman Kasus (EDA)
Pada bagian ketiga, kami memeriksa struktur tabel dan distribusi data awal. Kami menemukan sembilan kolom utama, antara lain identitas game, username pemain, besaran taruhan (*Bet*), angka multiplier saat pemain menarik uang (*CashedOut*), nilai pengali saat game meledak (*BustedAt*), dan profit taruhan.

Temuan menarik muncul saat kami mengecek nilai kosong. Kolom `CashedOut`, `Bonus`, dan `Profit` sama-sama memiliki missing value sebanyak 21.266 baris, atau sekitar 42,53%. Dalam analisis data biasa, angka kosong sebesar ini sering dianggap cacat data. Namun dalam domain crash game, kondisi kosong ini justru fakta penting. Nilai `CashedOut` kosong karena grafik permainan meledak (*bust*) sebelum pemain sempat mencairkan uang. Artinya, pemain kalah total pada ronde tersebut.

---

### Bagian 4: Pembersihan Data pada Tingkat Transaksi
Di bagian keempat, kami menangani kondisi kekalahan tadi secara logis sebelum melangkah ke pemodelan. 

Pertama, kami membuat kolom biner `is_win`. Jika kolom `CashedOut` ada isinya, kita beri nilai 1 tanda menang. Jika kosong, kita beri nilai 0 tanda kalah. Hasilnya menunjukkan bahwa 57,47% transaksi berakhir menang dan 42,53% transaksi berakhir kalah. 

Kedua, kami membetulkan pencatatan profit riil. Pada data mentah, transaksi kalah tercatat bernilai NaN. Kami mengisi nilai kosong ini dengan `-Bet` karena ketika kalah, pemain kehilangan modal taruhan mereka seutuhnya. Langkah ini krusial agar penghitungan untung-rugi pemain menjadi akurat.

---

### Bagian 5: Rekayasa Fitur (User Profiling per Username)
Bagian kelima merupakan inti dari pemrosesan data kami. Masalah utama dataset mentah adalah satuannya berbentuk transaksi per ronde permainan, padahal tujuan kami adalah mengenali perilaku orangnya. Oleh sebab itu, kami mengagregasi 50.000 log transaksi menjadi 4.149 profil pengguna unik berdasarkan kolom `Username`.

Dari proses agregasi ini, kami mengekstrak empat metrik perilaku kunci:
1. `total_bets`: Berapa kali pemain memasang taruhan, untuk melihat frekuensi dan ketahanan bermain.
2. `avg_bet`: Rata-rata nominal yang dipertaruhkan setiap ronde, untuk mengukur ukuran modal (*stake size*).
3. `win_rate`: Rasio kemenangan pemain, dihitung dari total ronde menang dibagi total ronde yang dimainkan.
4. `avg_cashout`: Rata-rata pengali multiplier saat pemain menang, yang menjadi indikator toleransi risiko. Pemain yang rata-rata cashout di angka 10x tentu memiliki profil risiko yang jauh berbeda dibanding pemain yang selalu keluar di angka 1,2x.

Bagi pemain yang belum pernah menang sama sekali, rata-rata cashout mereka kami isi dengan baseline 1,0 sebagai batas impas.

---

### Bagian 6: Transformasi dan Penskalaan Fitur
Masuk ke bagian keenam, kami memeriksa sebaran data keempat fitur tadi. Variabel `total_bets`, `avg_bet`, dan `avg_cashout` ternyata memiliki ekor distribusi yang sangat panjang (*skewed* ekstrem). Ada pemain yang hanya bertaruh belasan bits, tetapi ada juga yang bertaruh ratusan ribu bits. Ada pemain yang bermain sekali, tetapi ada pula yang bermain hampir 300 ronde.

Karena algoritma K-Means sangat sensitif terhadap skala data dan mengandalkan jarak Euclidean lurus, angka raksasa ini bisa merusak hasil pengelompokan. Kami mengatasinya dengan menerapkan transformasi logaritmik `np.log1p` untuk merapatkan sebaran nilai tanpa mengubah urutan datanya. Setelah sebarannya lebih simetris, kami membakukan seluruh fitur menggunakan `StandardScaler` sehingga rata-ratanya berada di titik nol dan standar deviasinya bernilai satu.

---

### Bagian 7: Evaluasi Model dan Penentuan Jumlah Klaster Optimal
Di bagian ketujuh, kami mengevaluasi berapa jumlah klaster ($k$) yang paling tepat secara objektif, bukan sekadar menebak-nebak. Kami menguji rentang nilai $k$ dari 2 sampai 8 dengan tiga metrik sekaligus.

Pertama, Elbow Method. Grafik Within-Cluster Sum of Squares (WCSS) menunjukkan penurunan curam dari $k=2$ ke $k=3$, lalu mulai melandai stabil pada rentang $k=3$ hingga $k=5$.

Kedua, Silhouette Score untuk mengukur kerapatan di dalam kelompok dan keterpisahan antar kelompok. Nilai Silhouette tertinggi berhasil diraih pada **$k = 4$ dengan skor 0,3189**. Pada $k=5$ nilainya turun ke 0,3141, dan semakin anjlok pada $k=6$ ke atas.

Ketiga, Davies-Bouldin Index (DBI). Skor pada $k = 4$ mencapai 1,1713, menunjukkan rasio jarak antar pusat klaster yang solid. Dari ketiga indikator ini, kami sepakat menetapkan $k = 4$ sebagai model terbaik.

---

### Bagian 8: Pelatihan Model Akhir K-Means
Pada bagian kedelapan, kami melatih model final K-Means dengan parameter $k = 4$ dan `n_init = 20` agar inisialisasi pusat klaster benar-benar stabil. 

Model membagi 4.149 pemain ke dalam empat kelompok:
- Klaster 0 berisi 1.067 pemain (25,7% populasi).
- Klaster 1 berisi 1.732 pemain (41,7% populasi).
- Klaster 2 berisi 354 pemain (8,5% populasi).
- Klaster 3 berisi 996 pemain (24,0% populasi).

---

### Bagian 9: Profiling, Karakterisasi, dan Visualisasi Klaster
Bagian kesembilan menyajikan interpretasi bisnis dan psikologi perilaku dari keempat klaster tersebut saat kita kembalikan ke skala nilai aslinya.

**Klaster 0 kami beri label Active Grinders.**  
Kelompok ini bermain paling sering dengan rata-rata 33,2 taruhan. Strategi mereka sangat disiplin dengan target multiplier rendah di angka 1,59x. Pola ini membuat tingkat kemenangan mereka stabil di 63,0% dan menghasilkan profit rata-rata positif tertinggi, yaitu +4.963 bits. Mereka bertaruh secara konsisten layaknya pekerja harian atau pengguna bot terprogram.

**Klaster 1 kami beri label Casual Conservative Winners.**  
Ini adalah kelompok terbesar, mencakup 41,7% pemain. Mereka hanya bermain sebentar (rata-rata 2,6 taruhan) dan langsung mencairkan taruhan di multiplier sangat aman, yaitu 1,43x. Pendekatan ini memberi mereka win rate fantastis sebesar 88,9% dan keuntungan rata-rata +4.022 bits. Mereka tipe pemain yang puas dengan kemenangan kecil lalu langsung berhenti.

**Klaster 2 kami beri label High-Risk Seekers atau Chasers.**  
Jumlahnya hanya 8,5% (354 orang), tetapi perilakunya paling ekstrem. Mereka memasang target multiplier rata-rata 7,13x, bahkan ada yang menargetkan puluhan kali lipat. Karena mengejar penggali tinggi, tingkat kemenangan mereka merosot tajam ke 30,2% dan mereka menanggung rugi bersih rata-rata -2.335 bits. Dari kacamata psikologi adiksi, kelompok inilah yang paling rentan terjebak ilusi kemenangan besar (*loss chasing*).

**Klaster 3 kami beri label Unlucky Casuals.**  
Mencakup 24,0% pemain dengan durasi bermain singkat (rata-rata 2,4 taruhan). Sayangnya mereka bernasib buruk dengan win rate hanya 8,9% dan menderita kerugian tercepat sebesar -9.186 bits. Mereka biasanya pemain pemula yang langsung bangkrut di awal dan tidak kembali lagi.

Untuk membuktikan pemisahan ini secara visual, kami mereduksi dimensi data menjadi dua komponen utama menggunakan PCA. Pada grafik scatter plot 2D yang dihasilkan, tampak jelas keempat kelompok ini menempati kuadran yang berbeda tanpa saling bertumpuk acak.

---

### Bagian 10: Kesimpulan dan Kepatuhan Rubrik Evaluasi
Sebagai penutup pada bagian kesepuluh, proyek ini membuktikan bahwa algoritma K-Means tanpa pengawasan mampu mengenali pola perilaku penjudi daring secara akurat. Dari data mentah transaksi, algoritma ini berhasil membedakan spektrum risiko mulai dari pemain konservatif pencari untung stabil hingga pemain spekulatif yang berisiko tinggi kecanduan.

Kami juga menegaskan bahwa pengerjaan proyek ini mematuhi batas penugasan CPMK2, di mana implementasi dihentikan tepat pada tahap Model Evaluation dan interpretasi klaster, tanpa melangkah ke pembuatan antarmuka web atau deployment sistem.

Terima kasih, sekian presentasi dari Kelompok 4. Kami persilakan bagi bapak dosen atau rekan-rekan yang ingin mengajukan pertanyaan.
