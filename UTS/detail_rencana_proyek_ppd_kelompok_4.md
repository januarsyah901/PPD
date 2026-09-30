# Detail Rencana Proyek Praktikum Penambangan Data (UTS/UAS)

**Mata Kuliah:** Praktikum Penambangan Data (SVPL261503)  
**Program Studi:** Sarjana Terapan (D.4) Teknologi Rekayasa Perangkat Lunak  
**Dosen Pengampu:**  
- Dr.Eng. Ir. Ganjar Alfian, S.T., M.Eng.  
- Dr. Imam Fahrurrozi, S.T., M.Cs.  

---

## 1. Identitas Kelompok
* **Kelompok:** 4
* **Anggota Kelompok:**
  1. MUHAMMAD RAKAN HIBRIZI
  2. JANUARSYAH AKBAR
  3. Devin Sotya Prathama
  4. SHINOSUKE ALEXANDER SWANDJAYA

---

## 2. Formulir Pendaftaran Spreadsheet (Materi Uji CPMK1)

| Field | Detail Isian |
| :--- | :--- |
| **Judul Proyek** | Segmentasi Perilaku dan Profil Risiko Penjudi Daring pada Platform Crash Game Menggunakan Algoritma K-Means Clustering |
| **Link Sumber Dataset** | [Kaggle - Gambling Behavior Bustabit](https://www.kaggle.com/datasets/kingabzpro/gambling-behavior-bustabit) |
| **Judul Paper** | AI Personalization and Its Influence on Online Gamblers' Behavior |
| **Link Paper** | [MDPI Behavioral Sciences / PubMed PMC12189489](https://www.mdpi.com/2076-328X/15/6/779) |

---

## 3. Rincian Teknis & Metodologi Proyek

### A. Ruang Lingkup & Task
* **Task:** Unsupervised Learning — *Clustering*
* **Format Data:** Tabular (`.csv`, 50.000 baris transaksi taruhan)
* **Tujuan Proyek:** Mengelompokkan pengguna ke dalam kelompok/klaster perilaku taruhan (misalnya: *Conservative Players*, *High-Risk High-Reward Gamblers*, *Grinders/Bonus Hunters*) berdasarkan variabel taruhan, frekuensi, dan toleransi risiko.

### B. Tahapan Analisis (Eksplorasi hingga Evaluasi)
Sesuai instruksi pengerjaan di Google Colab, alur dibatasi tepat sampai tahap **Model Evaluation**:

1. **Data Understanding & Preprocessing:**
   - Penanganan missing value pada kolom `CashedOut` dan `Profit` (kondisi pemain mengalami *bust* / kalah sebelum menarik taruhan).
   - Penanganan nilai *skewed* dan *outliers* pada variabel `Bet` dan `BustedAt` menggunakan transformasi logaritmik atau *Robust Scaling*.
2. **Feature Engineering (User Profiling):**
   - Agregasi data transaksi berdasarkan entitas `Username` untuk mendapatkan metrik perilaku:
     - Rata-rata besaran taruhan (*Average Bet Amount*)
     - Rasio penarikan target penggali (*Average Cashout Multiplier*)
     - Rasio kemenangan (*Win/Loss Ratio*)
     - Frekuensi taruhan (*Total Bets Played*)
3. **Pemodelan (Clustering):**
   - Implementasi algoritma **K-Means Clustering** (dengan alternatif pembanding seperti *DBSCAN* atau *Hierarchical Clustering*).
4. **Model Evaluation:**
   - **Elbow Method & WCSS (Within-Cluster Sum of Squares):** Menentukan jumlah $k$ klaster optimal.
   - **Silhouette Score & Analysis:** Mengukur kepadatan dan keterpisahan antar klaster.
   - **Davies-Bouldin Index (DBI):** Mengukur rasio kesamaan dalam klaster terhadap jarak antar klaster.

---

## 4. Rencana Sistematika Presentasi (10 Menit Presentasi + 5 Menit Q&A)
1. **Judul:** Memperkenalkan anggota kelompok 4 dan fokus topik clustering perilaku judi online.
2. **Pendahuluan:** Permasalahan kecanduan judi daring, volatilitas crash game, serta pentingnya segmentasi pola perilaku pemain.
3. **Metodologi:** Gambaran dataset Bustabit, teknik pembersihan/agregasi fitur, dan algoritma K-Means.
4. **Hasil dan Pembahasan:** Visualisasi klaster (PCA/t-SNE 2D/3D), interpretasi karakteristik tiap kelompok pemain, dan hasil metrik evaluasi (*Silhouette Score*).
5. **Kesimpulan dan Saran:** Ringkasan temuan segmentasi risiko serta batasan penelitian untuk pengembangan selanjutnya.