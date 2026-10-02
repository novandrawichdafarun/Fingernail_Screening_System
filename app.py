"""
Fingernail Screening System (Streamlit)
Pengambilan dan peningkatan kualitas citra kuku.

Jalankan lokal : streamlit run app.py
"""

import io
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage import img_as_float
from skimage.restoration import richardson_lucy

st.set_page_config(
    page_title="Fingernail Screening System",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# FUNGSI SHARPNESS
# ============================================================

def calculate_sharpness(image):
    if image is None:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ============================================================
# FUNGSI ENHANCEMENT
# ============================================================

def enhance_image(image, num_iter=15):
    if image is None:
        return None

    rgb = image.copy()

    # Normalisasi
    image_float = img_as_float(rgb)

    # PSF (Gaussian)
    psf_size = 7
    sigma = 2.0

    x = np.arange(-psf_size // 2 + 1, psf_size // 2 + 1)
    X, Y = np.meshgrid(x, x)

    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / psf.sum()

    # Richardson-Lucy per channel
    deblurred = np.zeros_like(image_float)

    for c in range(3):
        deblurred[:, :, c] = richardson_lucy(
            image_float[:, :, c],
            psf,
            num_iter=num_iter,
            clip=False,
        )

    deblurred = np.clip(deblurred, 0, 1)
    deblurred = (deblurred * 255).astype(np.uint8)

    # Sharpening (unsharp mask)
    blur = cv2.GaussianBlur(deblurred, (0, 0), 1.5)
    sharpened = cv2.addWeighted(deblurred, 1.6, blur, -0.6, 0)

    # CLAHE pada channel L
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    L = clahe.apply(L)

    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ============================================================
# PERSIAPAN GAMBAR + PIPELINE
# ============================================================

def decode_image(data: bytes, max_side: int):
    """Bytes -> array RGB uint8, perbaiki rotasi EXIF, kecilkan jika terlalu besar."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)

    h, w = arr.shape[:2]
    longest = max(h, w)

    if longest > max_side:
        scale = max_side / longest
        arr = cv2.resize(
            arr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return arr


# Cache dibatasi (maks. 3 entri, 10 menit) agar gambar tidak menumpuk di memori server
@st.cache_data(show_spinner=False, max_entries=3, ttl=600)
def run_pipeline(data: bytes, max_side: int, num_iter: int):
    original = decode_image(data, max_side)
    enhanced = enhance_image(original, num_iter)

    return (
        original,
        enhanced,
        calculate_sharpness(original),
        calculate_sharpness(enhanced),
    )

FORMATS = {
    # label : (format Pillow, mime, ekstensi, opsi simpan)
    "JPG (q95)": ("JPEG", "image/jpeg", "jpg", {"quality": 95}),
    "PNG (lossless)": ("PNG", "image/png", "png", {}),
    "BMP (lossless, ukuran besar)": ("BMP", "image/bmp", "bmp", {}),
}

def encode_image(arr, fmt):
    pill_format, mime, ext, options = FORMATS[fmt]
    
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=pill_format, **options)
    return buf.getvalue(), mime, ext

# Cache dibatasi agar gambar tidak menumpuk di memori server
@st.cache_data(show_spinner=False, max_entries=3, ttl=600)
def build_downloads(data: bytes, max_side: int, num_iter: int, fmt_label: str):
    """Siapkan file unduhan original (resolusi asli) dan enhanced."""
    _, enhanced, _, _ = run_pipeline(data, max_side, num_iter)

    # Original diunduh pada resolusi aslinya (tanpa pengecilan)
    original_full = decode_image(data, max_side=10**9)

    orig_bytes, mime, ext = encode_image(original_full, fmt_label)
    enh_bytes, _, _ = encode_image(enhanced, fmt_label)

    return orig_bytes, enh_bytes, mime, ext

def quality_label(sharpness):
    if sharpness >= 500:
        return "success", "BAIK - Layak dianalisis"
    if sharpness >= 200:
        return "warning", "CUKUP - Perlu perhatian"
    return "error", "BURUK - Capture ulang"


# ============================================================
# SIDEBAR PENGATURAN
# ============================================================

st.sidebar.header("⚙️ Pengaturan")

max_side = st.sidebar.select_slider(
    "Resolusi maksimum (piksel sisi terpanjang)",
    options=[640, 960, 1280, 1600, 1920],
    value=1280,
)

num_iter = st.sidebar.slider(
    "Iterasi Richardson-Lucy",
    min_value=5,
    max_value=30,
    value=15,
)

save_format = st.sidebar.selectbox(
    "Format unduhan",
    list(FORMATS.keys()),
)

st.sidebar.caption(
    "Resolusi dan iterasi yang lebih tinggi memberi detail lebih baik "
    "tetapi proses lebih lama, terutama di server deploy."
)


# ============================================================
# HALAMAN UTAMA
# ============================================================

st.title("🩺 Fingernail Screening System")
st.subheader("Deteksi Dini Diabetes Mellitus Menggunakan Citra Kuku")
st.write(
    "Sistem pengambilan dan peningkatan kualitas citra kuku "
    "menggunakan kamera perangkat webcam Logitech Brio 100 "
    "atau gambar yang diupload."
)

left, right = st.columns(2)

# ---------------- INPUT ----------------
with left:
    st.markdown("## 📷 1. Input Gambar")

    source = st.radio(
        "Sumber gambar",
        ["Kamera", "Upload Gambar"],
        horizontal=True,
    )

    image_bytes = None

    if source == "Kamera":
        photo = st.camera_input("Ambil foto kuku")
        if photo is not None:
            image_bytes = photo.getvalue()
    else:
        uploaded = st.file_uploader(
            "Upload gambar (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()

# ---------------- HASIL ----------------
with right:
    st.markdown("## ✨ 2. Hasil")

    if image_bytes is None:
        st.info("Belum ada gambar. Ambil foto atau upload gambar di sebelah kiri.")
    else:
        try:
            with st.spinner("Memproses enhancement... mohon tunggu."):
                original, enhanced, s_before, s_after = run_pipeline(
                    image_bytes, max_side, num_iter
                )
        except Exception as e:
            st.error(f"Gagal memproses gambar: {e}")
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            st.image(original, caption="Gambar Original")
        with c2:
            st.image(enhanced, caption="Gambar Setelah Enhancement")

        st.markdown("### 📊 Informasi Gambar")

        h, w = original.shape[:2]
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolusi", f"{w} × {h}")
        m2.metric("Sharpness sebelum", f"{s_before:.2f}")
        m3.metric("Sharpness sesudah", f"{s_after:.2f}")

        level, text = quality_label(s_after)
        getattr(st, level)(f"**Kualitas:** {text}")

        orig_bytes, enh_bytes, mime, ext = build_downloads(
            image_bytes, max_side, num_iter, save_format
        )
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "💾 Unduh Gambar Original",
                data=orig_bytes,
                file_name=f"fingernail_original_{stamp}.{ext}",
                mime=mime,
            )
        with d2:
            st.download_button(
                "💾 Unduh Gambar Enhanced",
                data=enh_bytes,
                file_name=f"fingernail_enhanced_{stamp}.{ext}",
                mime=mime,
            )

        st.caption(
            "Gambar original diunduh pada resolusi aslinya. "
            "Gambar enhanced mengikuti resolusi maksimum di sidebar."
        )

        st.markdown("### 🧠 Hasil Pengujian")
        st.info(
            "**Belum dianalisis.** Gambar sudah melalui proses enhancement. "
            "Untuk menentukan Normal / Pre-Diabetes / Diabetes diperlukan "
            "model Deep Learning yang sudah dilatih."
        )

st.divider()
st.caption(
    "Aplikasi ini hanya untuk penelitian dan bukan alat diagnosis medis. "
    "Gambar diproses di memori dan tidak disimpan permanen oleh aplikasi."
)
