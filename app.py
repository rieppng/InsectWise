"""
Sistem Deteksi & Klasifikasi Serangga berbasis CNN (Streamlit)
--------------------------------------------------------------
File yang dibutuhkan (satu folder dengan app.py):
  - model_cnn_serangga.h5       -> model CNN Anda
  - class_names.json            -> file hasil notebook training (berisi class_names,
                                   img_size, gbif_to_name)
  - database_serangga.json      -> kamus data serangga (offline)

Jalankan:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ----------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_cnn_serangga_final.keras"  # model CNN Anda (format .h5)
#MODEL_KERAS_PATH = BASE_DIR / "bud_model1.keras"   # cadangan jika .h5 gagal dimuat
DB_PATH = BASE_DIR / "database_serangga.json"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

DEFAULT_IMG_SIZE = (128, 128)   # (lebar, tinggi) -> dipakai jika ukuran input model tak terbaca
NORMALIZE_IN_APP = False        # False karena model sudah punya layer Rescaling.
                                # Ubah ke True jika model TIDAK punya Rescaling (bagi 255.0 di sini).
LOW_CONFIDENCE = 50.0           # di bawah ini (%) tampilkan peringatan
TOP_K = 3                       # jumlah kandidat teratas yang ditampilkan

st.set_page_config(page_title="Klasifikasi Serangga", page_icon="🐞", layout="wide")


# ----------------------------------------------------------------------
# LOADER (di-cache)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH))
    except Exception:
        # Cadangan: format .keras (lebih tahan beda versi TensorFlow/Keras)
        if MODEL_KERAS_PATH.exists():
            return tf.keras.models.load_model(str(MODEL_KERAS_PATH))
        raise


@st.cache_data
def load_json(path_str: str):
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_model():
    if not MODEL_PATH.exists() and not MODEL_KERAS_PATH.exists():
        st.error(f"File model tidak ditemukan: `{MODEL_PATH.name}`. "
                 "Letakkan di folder yang sama dengan `app.py`.")
        st.stop()
    try:
        return load_model()
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal memuat model: {e}")
        st.stop()


def safe_load_database() -> dict:
    if not DB_PATH.exists():
        st.warning(f"`{DB_PATH.name}` tidak ditemukan. Informasi serangga tidak dapat ditampilkan.")
        return {}
    try:
        return load_json(str(DB_PATH))
    except json.JSONDecodeError as e:
        st.warning(f"`{DB_PATH.name}` bukan JSON yang valid: {e}")
        return {}


def safe_load_class_names(num_classes: int):
    """
    Membaca class_names.json buatan notebook training.
    Format: {"img_size": 128, "class_names": [...], "index_to_gbif": {...}, "gbif_to_name": {...}}
    Urutan class_names == urutan indeks output model.
    Mengembalikan (daftar_id_kelas | None, dict ID->nama ilmiah dari GBIF).
    """
    if not CLASS_NAMES_PATH.exists():
        st.warning(f"`{CLASS_NAMES_PATH.name}` tidak ditemukan. Salin dari folder OUTPUT_DIR hasil training, "
                   "jika tidak hasil prediksi hanya berupa nomor indeks.")
        return None, {}
    try:
        meta = load_json(str(CLASS_NAMES_PATH))
    except json.JSONDecodeError as e:
        st.warning(f"`{CLASS_NAMES_PATH.name}` tidak valid: {e}")
        return None, {}

    gbif_names = {}
    if isinstance(meta, dict):
        names = meta.get("class_names")
        gbif_names = meta.get("gbif_to_name", {}) or {}
    else:                       # dukung juga format list sederhana
        names = meta

    if not isinstance(names, list) or len(names) != num_classes:
        n = len(names) if isinstance(names, list) else "?"
        st.warning(f"Jumlah kelas di `{CLASS_NAMES_PATH.name}` ({n}) tidak sama dengan output model "
                   f"({num_classes}). Pastikan file ini berasal dari training model yang sama.")
        return None, gbif_names
    return [str(n) for n in names], gbif_names


# ----------------------------------------------------------------------
# PREPROCESSING & INFERENSI
# ----------------------------------------------------------------------
def get_model_img_size(model) -> tuple:
    try:
        _, h, w, _ = model.input_shape
        if h and w:
            return (int(w), int(h))
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_IMG_SIZE


def preprocess(image: Image.Image, size: tuple) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")   # koreksi rotasi HP + pastikan 3 channel
    arr = np.asarray(image, dtype=np.float32)
    # Resize bilinear ala TF, sama seperti image_dataset_from_directory saat training
    arr = tf.image.resize(arr, (size[1], size[0]), method="bilinear").numpy()
    if NORMALIZE_IN_APP:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, batch: np.ndarray) -> np.ndarray:
    probs = model.predict(batch, verbose=0)[0].astype(np.float64)
    # Jika output belum berupa probabilitas (logits), ubah dengan softmax.
    if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    return probs


# ----------------------------------------------------------------------
# TAMPILAN INFORMASI
# ----------------------------------------------------------------------
def render_info(info: dict):
    kategori = str(info.get("kategori", "-"))
    low = kategori.lower()
    if "hama" in low:
        badge = "🔴"
    elif "guna" in low or "bermanfaat" in low:
        badge = "🟢"
    else:
        badge = "⚪"

    st.subheader(f"{info.get('nama_lokal', '-')}")
    st.markdown(f"*{info.get('nama_ilmiah', '-')}*")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kategori**")
        st.info(f"{badge} {kategori}")
    with c2:
        st.markdown("**Status Bahaya**")
        st.warning(f"⚠️ {info.get('status_bahaya', '-')}")

    st.markdown("**🌿 Kegunaan Ekologis**")
    st.success(info.get("kegunaan_ekologis", "-"))

    st.markdown("**💡 Fakta Unik**")
    st.info(info.get("fakta_unik", "-"))


# ----------------------------------------------------------------------
# APLIKASI UTAMA
# ----------------------------------------------------------------------
def main():
    st.title("🐞 Deteksi & Klasifikasi Serangga")
    st.caption("Unggah foto atau ambil gambar langsung dari kamera untuk mengenali serangga.")

    model = safe_load_model()
    database = safe_load_database()
    img_size = get_model_img_size(model)
    num_classes = int(model.output_shape[-1])
    class_names, gbif_names = safe_load_class_names(num_classes)

    # --- Input (multi-input) ---
    source = st.radio("Sumber gambar", ["📁 Upload file", "📷 Kamera"], horizontal=True)
    if source == "📁 Upload file":
        file = st.file_uploader("Pilih gambar (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Ambil foto serangga")

    if file is None:
        st.info("Belum ada gambar. Silakan unggah file atau ambil foto terlebih dahulu.")
        return

    try:
        image = Image.open(file)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("File tidak dapat dibaca sebagai gambar. Coba file lain.")
        return

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.image(image, caption="Gambar yang dianalisis", use_container_width=True)

    # --- Inferensi ---
    with st.spinner("Menganalisis gambar..."):
        try:
            probs = predict(model, preprocess(image, img_size))
        except Exception as e:  # noqa: BLE001
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            return

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100
    class_id = class_names[top_idx] if class_names else str(top_idx)

    with right:
        st.metric("Hasil Prediksi", class_id, f"{confidence:.2f}% yakin")
        st.progress(min(int(confidence), 100))

        if confidence < LOW_CONFIDENCE:
            st.warning("Tingkat keyakinan rendah. Coba foto yang lebih jelas, "
                       "dengan serangga di tengah dan latar yang sederhana.")

        info = database.get(class_id)
        if info:
            render_info(info)
        else:
            st.subheader(gbif_names.get(class_id, "Nama belum tersedia"))
            st.caption(f"GBIF ID: {class_id}")
            st.warning(f"ID kelas `{class_id}` belum ada di `{DB_PATH.name}`, "
                       "jadi info kategori/bahaya/ekologi belum tersedia.")

    # --- Kandidat teratas ---
    with st.expander(f"Lihat {TOP_K} kandidat teratas"):
        for i in np.argsort(probs)[::-1][:TOP_K]:
            cid = class_names[i] if class_names else str(i)
            nama = database.get(cid, {}).get("nama_lokal") or gbif_names.get(cid, "(tidak ada di database)")
            st.write(f"**{cid}** — {nama}: {probs[i] * 100:.2f}%")


if __name__ == "__main__":
    main()