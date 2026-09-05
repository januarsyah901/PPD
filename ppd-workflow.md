# Workflow Praktikum Penambangan Data (PPD)

- **Eksekusi Notebook**: Semua file `.ipynb` dijalankan langsung di Google Colab.
- **Workflow Repository**:
  - Repo GitHub dibuka langsung via fitur `Open from GitHub` di Google Colab (`januarsyah901/PPD`).
  - Memilih branch `main` (atau branch per pertemuan jika ada).
  - Memilih file notebook `.ipynb` yang ingin di-run (format: `Pn/project/p{n}_{topik}.ipynb`).
- **Konvensi & Best Practices Pembuatan Notebook**:
  - **Dataset Handling**: Karena Colab hanya memuat file `.ipynb` tanpa clone repo secara otomatis, selalu sediakan cell inisialisasi untuk auto-clone repo (`!git clone ...`), load via GitHub Raw URL, atau Google Drive mount jika ada file dataset eksternal.
  - **Dependencies**: Tambahkan `!pip install ...` jika memerlukan library tambahan di luar bawaan Colab.
  - **Struktur File**: Simpan/strukturkan file di dalam folder pertemuan (misal `P1/project/p1_numpy_pandas.ipynb`).

- **Standar Laporan Praktikum (LaTeX)**:
  - Setiap folder pertemuan dibuat sub-folder `Pn/laporan/`.
  - Format file LaTeX: `PPD_P{n}_Januarsyah Akbar_535846.tex`.
  - Aset logo: `lambang ugm.png` di folder laporan.
  - Folder gambar: `Pn/laporan/gambar/`.
  - File build sementara LaTeX (`.aux`, `.log`, `.toc`, dsb.) diabaikan via `.gitignore`.

- **Trigger & Standar Otomasi Laporan End-to-End**:
  - Ketika bang jan meminta: *"Gas buatin laprak pertemuan [N] full end-to-end sampai PDF jadi"* (atau perintah serupa seperti *"Buat laprak P[N] lengkap sama gambar dan compile LaTeX-nya"*), langsung jalankan pipeline otomatis berikut tanpa perlu diminta langkah per langkah:
    1. **Analisis Modul & Proyek**: Baca `Pn/MODUL_PDD_Pn.md` dan cek notebook di `Pn/project/`.
    2. **Generate Gambar/Grafik**: Buat skrip python untuk mengekstrak dan merender seluruh grafik percobaan dan tugas ke folder `Pn/laporan/gambar/`.
    3. **Tulis Dokumen LaTeX (`.tex`)**: Tulis dokumen lengkap dengan format standar UGM (struktur 4 Bab, metadata nama/NIM/kelas, code listing, tabel deskripsi, output box, dan analisis/jawaban tugas mendalam). Pastikan semua karakter khusus seperti underscore di luar code environment sudah di-escape (`\_`).
    4. **Build PDF Otomatis**: Kompilasi langsung dengan `pdflatex` / `latexmk` sebanyak 2x di folder `Pn/laporan/` hingga menghasilkan file PDF siap kumpul `PPD_P{n}_Januarsyah Akbar_535846.pdf`.
