"""
Fingernail Screening System (Streamlit)
Pengambilan dan peningkatan kualitas citra kuku.

Jalankan lokal : python.exe -m streamlit run app.py
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
# TAHAPAN ENHANCEMENT
# Semua fungsi: input array RGB uint8 -> output array RGB uint8
# ============================================================

def white_balance_gray_world(img):
    """Koreksi warna dominan (color cast) dengan asumsi Gray World."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-6)
    gains = np.clip(gains, 0.7, 1.4)   # batasi agar warna tidak berubah ekstrem
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def correct_illumination(img, strength=0.8):
    """Meratakan pencahayaan tidak merata (bayangan / sisi lebih terang) pada channel L."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    h, w = L.shape
    small_w, small_h = max(w // 8, 8), max(h // 8, 8)

    small = cv2.resize(L, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small, (0, 0), sigmaX=max(small_w, small_h) / 6)
    background = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)

    corrected = L - strength * (background - background.mean())
    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def auto_gamma(img, target=0.5):
    """Koreksi gamma otomatis jika gambar terlalu gelap / terlalu terang."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = float(np.clip(gray.mean() / 255.0, 0.01, 0.99))

    if abs(mean - target) < 0.08:
        return img   # sudah cukup baik

    gamma = np.clip(np.log(target) / np.log(mean), 0.6, 1.6)

    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def denoise_nlm(img, strength=5):
    """Noise reduction Non-Local Means (dilakukan sebelum deblurring)."""
    return cv2.fastNlMeansDenoisingColored(
        img, None, strength, strength, 7, 21
    )


def deblur_richardson_lucy(img, num_iter=15):
    """Deblurring Richardson-Lucy dengan PSF Gaussian."""
    image_float = img_as_float(img)

    psf_size = 7
    sigma = 2.0

    x = np.arange(-psf_size // 2 + 1, psf_size // 2 + 1)
    X, Y = np.meshgrid(x, x)

    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / psf.sum()

    deblurred = np.zeros_like(image_float)

    for c in range(3):
        deblurred[:, :, c] = richardson_lucy(
            image_float[:, :, c],
            psf,
            num_iter=num_iter,
            clip=False,
        )

    deblurred = np.clip(deblurred, 0, 1)
    return (deblurred * 255).astype(np.uint8)


def unsharp_mask(img, amount=0.6, sigma=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def contrast_stretch(img, low=1, high=99):
    """Peregangan kontras berdasarkan persentil pada channel L."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    lo, hi = np.percentile(L, (low, high))
    if hi - lo < 10:
        return img   # kontras sudah sangat sempit, jangan diperbesar

    stretched = (L - lo) * 255.0 / (hi - lo)
    lab[:, :, 0] = np.clip(stretched, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_clahe(img, clip_limit=1.8):
    """CLAHE pada channel L (meningkatkan detail lokal)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L = clahe.apply(L)

    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)


def enhance_image(image, opts):
    """
    Menjalankan tahapan sesuai urutan:
    white balance -> pencahayaan -> gamma -> denoise ->
    deblurring -> unsharp -> peregangan kontras -> CLAHE
    """
    if image is None:
        return None, []

    out = image.copy()
    applied = []

    if opts["wb"]:
        out = white_balance_gray_world(out)
        applied.append("White Balance")

    if opts["illum"]:
        out = correct_illumination(out)
        applied.append("Koreksi Pencahayaan")

    if opts["gamma"]:
        out = auto_gamma(out)
        applied.append("Gamma Otomatis")

    if opts["denoise"]:
        out = denoise_nlm(out, opts["denoise_h"])
        applied.append("Noise Reduction")

    if opts["deblur"]:
        out = deblur_richardson_lucy(out, opts["num_iter"])
        applied.append("Deblurring (Richardson-Lucy)")

    if opts["sharpen"]:
        out = unsharp_mask(out, opts["sharpen_amount"])
        applied.append("Unsharp Mask")

    if opts["stretch"]:
        out = contrast_stretch(out)
        applied.append("Peregangan Kontras")

    if opts["clahe"]:
        out = apply_clahe(out, opts["clip_limit"])
        applied.append("CLAHE")

    return out, applied


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
def run_pipeline(data: bytes, max_side: int, opts: dict):
    original = decode_image(data, max_side)
    enhanced, applied = enhance_image(original, opts)

    return (
        original,
        enhanced,
        calculate_sharpness(original),
        calculate_sharpness(enhanced),
        applied,
    )


FORMATS = {
    # label        : (format Pillow, mime, ekstensi, opsi simpan)
    "JPG (q95)": ("JPEG", "image/jpeg", "jpg", {"quality": 95}),
    "PNG (lossless)": ("PNG", "image/png", "png", {}),
    "BMP (lossless, ukuran besar)": ("BMP", "image/bmp", "bmp", {}),
}


def encode_image(arr, fmt_label):
    pil_format, mime, ext, options = FORMATS[fmt_label]

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=pil_format, **options)

    return buf.getvalue(), mime, ext


@st.cache_data(show_spinner=False, max_entries=3, ttl=600)
def build_downloads(data: bytes, max_side: int, opts: dict, fmt_label: str):
    """Siapkan file unduhan original (resolusi asli) dan enhanced."""
    _, enhanced, _, _, _ = run_pipeline(data, max_side, opts)

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

save_format = st.sidebar.selectbox(
    "Format unduhan",
    list(FORMATS.keys()),
)

st.sidebar.header("🧪 Tahapan Enhancement")
st.sidebar.caption("Dijalankan berurutan dari atas ke bawah.")

opts = {}

opts["wb"] = st.sidebar.checkbox(
    "1. White Balance (Gray World)",
    value=False,
    help="Menghilangkan warna dominan akibat lampu / kamera.",
)

opts["illum"] = st.sidebar.checkbox(
    "2. Koreksi pencahayaan tidak merata",
    value=False,
    help="Meratakan sisi gambar yang lebih gelap / terang.",
)

opts["gamma"] = st.sidebar.checkbox(
    "3. Koreksi gamma otomatis",
    value=True,
    help="Hanya aktif jika gambar terlalu gelap atau terlalu terang.",
)

opts["denoise"] = st.sidebar.checkbox(
    "4. Noise Reduction (Non-Local Means)",
    value=True,
    help="Dilakukan sebelum deblurring agar noise tidak ikut diperkuat.",
)
opts["denoise_h"] = (
    st.sidebar.slider("Kekuatan noise reduction", 1, 15, 5)
    if opts["denoise"] else 5
)

opts["deblur"] = st.sidebar.checkbox(
    "5. Deblurring (Richardson-Lucy)",
    value=True,
)
opts["num_iter"] = (
    st.sidebar.slider("Iterasi Richardson-Lucy", 5, 30, 15)
    if opts["deblur"] else 15
)

opts["sharpen"] = st.sidebar.checkbox(
    "6. Sharpening (Unsharp Mask)",
    value=True,
)
opts["sharpen_amount"] = (
    st.sidebar.slider("Kekuatan sharpening", 0.1, 2.0, 0.6, 0.1)
    if opts["sharpen"] else 0.6
)

opts["stretch"] = st.sidebar.checkbox(
    "7. Peregangan kontras (persentil 1-99)",
    value=False,
    help="Melebarkan rentang kecerahan agar gambar tidak terlihat pucat.",
)

opts["clahe"] = st.sidebar.checkbox(
    "8. CLAHE (detail lokal)",
    value=True,
)
opts["clip_limit"] = (
    st.sidebar.slider("Clip limit CLAHE", 1.0, 4.0, 1.8, 0.1)
    if opts["clahe"] else 1.8
)

st.sidebar.caption(
    "Tahap 1, 2, 3, dan 7 mengubah warna / kecerahan gambar. Jika warna "
    "kuku dipakai sebagai fitur analisis, gunakan pengaturan yang sama "
    "untuk semua data. Resolusi dan iterasi yang lebih tinggi memberi "
    "detail lebih baik tetapi proses lebih lama."
)


# ============================================================
# HALAMAN UTAMA
# ============================================================

st.title("🩺 Fingernail Screening System")
st.subheader("Deteksi Dini Diabetes Mellitus Menggunakan Citra Kuku")
st.write(
    "Sistem pengambilan dan peningkatan kualitas citra kuku "
    "menggunakan kamera perangkat (misalnya webcam Logitech Brio 100) "
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
                original, enhanced, s_before, s_after, applied = run_pipeline(
                    image_bytes, max_side, opts
                )
        except Exception as e:
            st.error(f"Gagal memproses gambar: {e}")
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            st.image(original, caption="Gambar Original")
        with c2:
            st.image(enhanced, caption="Gambar Setelah Enhancement")

        if applied:
            st.caption("Tahapan diterapkan: " + " → ".join(applied))
        else:
            st.caption("Tidak ada tahapan enhancement yang dipilih.")

        st.markdown("### 📊 Informasi Gambar")

        h, w = original.shape[:2]
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolusi", f"{w} × {h}")
        m2.metric("Sharpness sebelum", f"{s_before:.2f}")
        m3.metric("Sharpness sesudah", f"{s_after:.2f}")

        level, text = quality_label(s_after)
        getattr(st, level)(f"**Kualitas:** {text}")

        orig_bytes, enh_bytes, mime, ext = build_downloads(
            image_bytes, max_side, opts, save_format
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