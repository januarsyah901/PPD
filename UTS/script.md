# Panduan Naskah Bicara Presentasi UTS (Kelompok 4)

**Topik:** Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Crash Game Bustabit Menggunakan K-Means Clustering  
**Waktu Total:** Maksimal 10 Menit (rata-rata 45 sampai 60 detik per slide)  
**Anggota Tim:** Muhammad Rakan Hibrizi, Januarsyah Akbar, Devin Sotya Prathama, Shinosuke Alexander Swandjaya

---

### Slide 1: Cover (Estimasi Waktu: 30 Detik)

**Fokus Inti:** Menyapa audiens, mengenalkan anggota kelompok, dan menyebutkan judul proyek secara lugas.

**Naskah Lisan:**

> "Selamat pagi, bapak dosen pengampu dan rekan-rekan sekalian. Kami dari Kelompok 4 yang beranggotakan Rakan, Januarsyah, Devin, dan Shinosuke. Pada kesempatan kali ini, kami mempresentasikan proyek Praktikum Penambangan Data bertajuk _Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Crash Game Bustabit Menggunakan K-Means Clustering_. Langsung saja kita masuk ke pokok permasalahan."

---

### Slide 2: Latar Belakang dan Tujuan (Estimasi Waktu: 45 Detik)

**Fokus Inti:** Mengangkat ilusi kemenangan di platform judi daring dan tujuan riset kelompok.

**Naskah Lisan:**

> "Banyak masyarakat terjerumus ke dalam lingkaran judi daring karena terjebak ilusi kendali, mengira mereka bisa menang konsisten jika memiliki trik tertentu. Melalui proyek ini, tujuan kami adalah memetakan segmen perilaku pemain crash game Bustabit dari data transaksi riil. Kami mendeskripsikan profil hasil dari setiap kelompok, khususnya melihat median profit dan persentase pemain yang berakhir rugi. Riset kami mengacu pada paper ilmiah MDPI Behavioral Sciences tahun 2025 mengenai personalisasi platform terhadap perilaku pemain."

---

### Slide 3: Karakteristik Dataset Bustabit (Estimasi Waktu: 50 Detik)

**Fokus Inti:** Menjelaskan data mentah dan logika penting di balik missing values 42,53%.

**Naskah Lisan:**

> "Data yang kami gunakan bersumber dari Kaggle berisi 50.000 baris log transaksi taruhan. Saat memeriksa data, kami menemukan kolom CashedOut, Bonus, dan Profit memiliki kekosongan sekitar 42,53%. Dalam analisis data biasa, angka kosong ini sering dikira data cacat. Namun di domain crash game, ini adalah fakta perilaku penting. Nilai kosong berarti grafik permainan meledak sebelum pemain menekan tombol cashout. Artinya, sebanyak 42,53% transaksi berakhir bust atau kalah total."

---

### Slide 4: Mengapa Clustering, Bukan Regresi (Estimasi Waktu: 60 Detik)

**Fokus Inti:** Menjawab pertanyaan metodologi pembanding terhadap paper acuan yang memakai regresi.

**Naskah Lisan:**

> "Paper rujukan kami menggunakan regresi untuk melihat korelasi agregat populasi. Namun kami memilih K-Means Clustering karena perilaku penjudi sangat heterogen. Regresi memaksakan satu garis rata-rata populasi, sedangkan kami ingin menemukan arketipe kelompok yang berbeda tanpa membatasi satu variabel target tunggal. Profil risiko seseorang ditentukan oleh kombinasi frekuensi, modal, toleransi pengganda, dan win rate secara bersamaan. Karena data transaksi mentah ini tidak memiliki label kategori sejak awal, clustering tanpa pengawasan menjadi metode paling tepat untuk membedah segmen pemain."

---

### Slide 5: Preprocessing dan Feature Engineering (Estimasi Waktu: 60 Detik)

**Fokus Inti:** Transformasi data transaksi menjadi profil pemain dan penjelasan audit penyaringan data.

**Naskah Lisan:**

> "Sasaran kami adalah membaca perilaku manusianya, bukan ronde per ronde. Kami mengagregasi 50.000 transaksi ke tingkat profil pengguna unik. Pertama, kami membuat label is_win dan profit riil di mana kondisi kalah dicatat minus nilai taruhan. Kedua, ini audit penting kami, sebanyak 43,9% pemain ternyata hanya bertaruh 1 atau 2 kali. Metrik pada kelompok ini murni kebetulan acak, bukan pola perilaku. Karena itu, analisis kami batasi pada pemain aktif dengan minimal 10 taruhan, menghasilkan 1.111 profil valid. Data ini kemudian kami transformasi log1p untuk mereduksi skewness dan kami standarisasi menggunakan StandardScaler."

---

### Slide 6: Penentuan Jumlah Klaster (Estimasi Waktu: 50 Detik)

**Fokus Inti:** Memaparkan pengujian metrik k = 2 hingga 7 dan pemilihan model k = 5.

**Naskah Lisan:**

> "Untuk menentukan jumlah kelompok secara objektif, kami menguji nilai k dari 2 sampai 7 menggunakan Elbow Method, Silhouette Score, dan Davies-Bouldin Index. Kami memilih model k = 5 sebagai titik kompromi terbaik. Nilai Silhouette Score berada di angka 0,2588 dan Davies-Bouldin Index di 1,1430. Model ini tidak kami klaim mutlak optimal secara matematis murni, tetapi memberikan keterpisahan yang paling stabil dan interpretasi psikologis paling masuk akal."

---

### Slide 7: Mengapa Bukan k = 4? (Estimasi Waktu: 60 Detik)

**Fokus Inti:** Membedah kelemahan model awal k = 4 dan alasan ilmiah beralih ke k = 5.

**Naskah Lisan:**

> "Mungkin ada pertanyaan, mengapa kami tidak memakai k = 4? Pada analisis awal sebelum data difilter, Silhouette memang sempat memuncak di k = 4. Namun setelah kami audit, klaster tersebut terbentuk semu akibat jumlah ronde main, bukan gaya bermain. Dua klaster awal didominasi pemain yang hanya coba-coba 2 ronde, yang satu kebetulan menang win rate 1,0 dan yang satu langsung kalah win rate 0. Itu bukan segmen perilaku. Begitu kami saring dengan batas minimal 10 taruhan, performa k = 4 justru kalah di kedua metrik dibanding k = 5, baik dari sisi kerapatan Silhouette maupun DBI."

---

### Slide 8: Karakteristik Lima Klaster Perilaku (Estimasi Waktu: 70 Detik)

**Fokus Inti:** Menjelaskan profil 5 klaster dari sisi risiko dan performa finansial riil.

**Naskah Lisan:**

> "Model k = 5 memetakan pemain ke dalam lima tipologi:
> Klaster 0 adalah kelompok Target Rendah Win Rate Tinggi. Mereka sangat disiplin bermain di pengganda 1,19x dengan win rate 80%.
> Klaster 1 adalah Frekuensi Tinggi Target Rendah, bertaruh hingga median 69 kali di pengali 1,37x, namun separuh dari mereka tetap merugi.
> Klaster 2 adalah Taruhan Nominal Tinggi dengan modal rata-rata 2.225 bits. Mereka bermain aman di pengali 1,49x sehingga mengantongi profit positif.
> Klaster 3 adalah kelompok Target Menengah di pengali 2,30x, di mana win rate mereka anjlok ke 33% dan 61% pemain menanggung rugi.
> Terakhir, Klaster 4 adalah Target Cashout Tinggi. Mereka mengejar pengali ekstrem rata-rata 9,05x. Hasilnya win rate terpuruk ke 14% dan mencatatkan kerugian terbesar."

---

### Slide 9: Temuan Kunci Dinamika Taruhan (Estimasi Waktu: 60 Detik)

**Fokus Inti:** Menyajikan bukti empiris yang membongkar mitos kemenangan judi online.

**Naskah Lisan:**

> "Dari profil lima klaster tadi, kami menarik beberapa temuan kunci untuk kampanye literasi anti-judol. Pertama, makin tinggi pengganda yang dikejar pemain, win rate mereka turun bebas dari 80%, ke 33%, hingga tinggal 14%. Kerugian median negatif selalu terjadi pada pemain yang mengejar pengali 2x ke atas. Kedua, temuan profit positif pada Klaster 2 tidak membuktikan judi menguntungkan. Klaster ini menguasai 88,6% perputaran dana, menunjukkan bahwa kemenangan di platform terpusat pada segelintir akun bermodal raksasa, sementara mayoritas pemain kasual tetap habis tergerus."

---

### Slide 10: Pemanfaatan Edukasi & Batasan Riset (Estimasi Waktu: 60 Detik)

**Fokus Inti:** Menjelaskan secara santai arah guna data untuk edukasi publik, ide deteksi transaksi keuangan, dan batasan riset.

**Naskah Lisan:**

> "Nah, data hasil olahan ini sebenarnya mau kita pakai buat apa? Dari awal kami tegaskan, tujuannya bukan buat bantu bandar atau optimasi game, tapi murni buat edukasi dan perlindungan masyarakat.
> 
> Pemanfaatan pertama, buat materi kampanye literasi anti-judol. Teman-teman mahasiswa atau pegiat finansial bisa pakai angka konkret tadi untuk bikin infografis, nunjukin bukti matematis kalau makin nekat ngejar jackpot, sistemnya memang dirancang bikin kantong jebol.
> 
> Pemanfaatan kedua, ada potensi menarik di sistem keuangan perbankan dan e-wallet. Pihak bank memang tidak tahu multiplier game, tapi pola psikologis di Klaster 4 tadi bisa kita petakan ke jejak mutasi. Misalnya saat pemain kalah dan kalap, muncul anomali top-up panik berkali-kali dalam hitungan menit dengan nominal makin naik. Pola ini bisa jadi sinyal peringatan dini buat membatasi rekening, sekaligus bantu aparat membedakan mana rekening korban adiksi dan mana rekening penampung milik sindikat.
> 
> Tapi kami tetap jujur dengan batasannya. Analisis ini baru mencakup pemain aktif yang main minimal 10 kali, jadi ada faktor bias ketahanan atau survivorship bias. Selain itu, tingkat kerapatan klasternya masih berada di kategori sedang."

---

### Slide 11: Kesimpulan dan Batasan Pengerjaan (Estimasi Waktu: 40 Detik)

**Fokus Inti:** Menutup presentasi dan menegaskan kepatuhan terhadap rubrik batas pengerjaan CPMK2.

**Naskah Lisan:**

> "Kesimpulannya, model k = 5 berhasil membedakan spektrum risiko pemain secara objektif, mulai dari pemain konservatif target rendah hingga kelompok spekulatif berisiko tinggi kecanduan. Sesuai batasan rubrik penugasan UTS CPMK2, proyek kami selesaikan tepat pada tahap evaluasi dan interpretasi model, tanpa melangkah ke deployment antarmuka aplikasi. Demikian presentasi dari Kelompok 4, waktu kami kembalikan untuk sesi tanya jawab. Terima kasih."
