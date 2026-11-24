# 🪲 InsectWise — Aplikasi Klasifikasi Spesies Serangga Berbasis CNN

**InsectWise** adalah aplikasi berbasis *Deep Learning* menggunakan **Convolutional Neural Network (CNN)** untuk mengidentifikasi dan mengklasifikasikan spesies serangga berdasarkan citra habitus (morfologi tampak luar). Model dilatih menggunakan dataset spesimen serangga berbasis kode taksonomi **GBIF (Global Biodiversity Information Facility)** dan terintegrasi ke dalam aplikasi inferensi untuk memudahkan pengenalan spesies secara cepat.

---

## 👥 Anggota Kelompok

| No | Nama Lengkap |
| :-: | :--- |
| 1 | **Aulia Arief** |
| 2 | **Naufal Huseini** |
| 3 | **Herva Sulistia** |
| 4 | **Habil Rahman Soleh** |
| 5 | **Jefri Mulia Rezki** |

---

## ✨ Fitur Utama

* **Klasifikasi Citra Serangga Otomatis:** Mengidentifikasi spesies serangga dari gambar input beserta nilai probabilitas / tingkat kepercayaannya (*Top-K predictions*).
* **Pemetaan Kode GBIF & Informasi Spesies:** Menghubungkan hasil prediksi kelas (kode GBIF ID) dengan nama ilmiah serta informasi deskriptif serangga dari `class_names.json` dan `database_serangga.json`.
* **Pipeline Training Terstruktur:**
  * Pembagian data *stratified* (`train` / `val` / `test`) untuk menjaga proporsi kelas.
  * Penyeimbangan data (*data balancing*) khusus pada data latih untuk mengurangi bias kelas mayoritas/minoritas tanpa kebocoran data (*data leakage*).
  * Augmentasi citra otomatis (*flip*, rotasi, *zoom*, translasi, dan kontras) untuk meningkatkan kemampuan generalisasi model.

---

## 📂 Struktur Proyek

```text
InsectWise/
├── app.py                           # Aplikasi utama untuk menjalankan inferensi klasifikasi serangga
├── model_cnn_serangga_final.keras   # Model CNN terlatih siap pakai
├── class_names.json                 # Mapping indeks kelas ke kode GBIF, nama spesies, & konfigurasi model
├── database_serangga.json           # Basis data informasi tambahan mengenai spesies serangga
├── requirements.txt                 # Daftar dependensi library Python
├── LICENSE                          # Lisensi proyek
└── README.md                        # Dokumentasi proyek
```

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Pengolahan Data & Evaluasi:** NumPy, Scikit-learn, Matplotlib, Seaborn
* **Dataset:** *Insect Identification from Habitus Images* (Kaggle / GBIF Taxonomy)

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/rieppng/InsectWise.git
   cd InsectWise
   ```

2. **Buat dan aktifkan virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instal dependensi yang dibutuhkan:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```bash
   python app.py
   ```
   *(Catatan: Jika `app.py` menggunakan framework web seperti Streamlit, jalankan dengan `streamlit run app.py`).*

---

## 📊 Catatan Dataset & Model

* **Sumber Dataset:** [Insect Identification from Habitus Images (Kaggle)](https://www.kaggle.com/datasets/kmldas/insect-identification-from-habitus-images)
* **Artefak Model:** Pastikan file `model_cnn_serangga_final.keras`, `class_names.json`, dan `database_serangga.json` berada pada direktori yang sesuai dengan konfigurasi di dalam `app.py` saat aplikasi dijalankan.