import os
import sys
from string import Template

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QRectF, QVariantAnimation, QEasingCurve
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPainterPath, QFontDatabase,
    QPen
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
    QSizePolicy, QDialog, QTabWidget, QPlainTextEdit
)


# ==========================================================
# 1. PROSES DETEKSI KEMATANGAN PISANG
# ==========================================================

# Piksel dengan V (kecerahan) sampai nilai ini dianggap hitam.
AMBANG_GELAP = 55

# Sisi terpanjang gambar disesuaikan ke ukuran ini (rasio aspek tetap).
# Gambar 3:2 tetap menjadi 600 x 400 seperti sebelumnya.
UKURAN_MAKS = 600


def ambang(kanal, batas, jenis):
    """
    Thresholding satu kanal dengan cv2.threshold, mengembalikan citra biner.
    THRESH_BINARY     : 255 jika nilai > batas, selain itu 0.
    THRESH_BINARY_INV : 255 jika nilai <= batas, selain itu 0.
    """
    _, biner = cv2.threshold(kanal, batas, 255, jenis)
    return biner


def deteksi_pisang(nama_file):
    gambar = cv2.imread(nama_file)

    if gambar is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    # Ukuran disesuaikan tanpa mengubah rasio aspek, jadi gambar potret,
    # persegi, maupun lanskap tidak gepeng.
    tinggi0, lebar0 = gambar.shape[:2]
    skala = UKURAN_MAKS / max(tinggi0, lebar0)
    gambar = cv2.resize(
        gambar,
        (max(1, round(lebar0 * skala)), max(1, round(tinggi0 * skala))),
    )
    blur = cv2.GaussianBlur(gambar, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # SEGMENTASI OBJEK PISANG (thresholding pada kanal H, S, V)
    # Mengambil objek pisang dan menghilangkan latar putih:
    #   H <= 90  (corak hijau sampai cokelat)
    #   S >= 35  (latar putih hampir tidak berwarna, jadi tersingkir)
    #   V >= 30  (bukan piksel yang terlalu gelap)
    h, s, v = cv2.split(hsv)
    mask_h = ambang(h, 90, cv2.THRESH_BINARY_INV)
    mask_s = ambang(s, 34, cv2.THRESH_BINARY)
    mask_v = ambang(v, 29, cv2.THRESH_BINARY)
    mask_pisang = cv2.bitwise_and(cv2.bitwise_and(mask_h, mask_s), mask_v)

    # Piksel gelap hampir tidak berwarna (saturasi rendah), jadi lolos dari
    # batas S di atas. Pisang busuk didominasi piksel seperti ini, maka
    # piksel gelap (V <= AMBANG_GELAP) ikut dimasukkan ke objek.
    mask_gelap = ambang(v, AMBANG_GELAP, cv2.THRESH_BINARY_INV)
    mask_pisang = cv2.bitwise_or(mask_pisang, mask_gelap)

    # Menghilangkan noise kecil, lalu menutup celah kecil pada objek
    kernel_kecil = np.ones((3, 3), np.uint8)
    kernel_sedang = np.ones((7, 7), np.uint8)
    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_OPEN, kernel_kecil)
    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_CLOSE, kernel_sedang)

    # Mencari kontur objek
    kontur, _ = cv2.findContours(
        mask_pisang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not kontur:
        raise ValueError("Objek pisang tidak ditemukan.")

    # Mengambil kontur terbesar
    kontur_terbesar = max(kontur, key=cv2.contourArea)

    tinggi, lebar = mask_pisang.shape
    if cv2.contourArea(kontur_terbesar) < tinggi * lebar * 0.05:
        raise ValueError("Objek pisang terlalu kecil atau tidak terdeteksi.")

    # Membuat mask khusus objek pisang
    mask_objek = np.zeros_like(mask_pisang)
    cv2.drawContours(
        mask_objek, [kontur_terbesar], -1, 255, thickness=cv2.FILLED
    )

    # Menghitung total piksel objek
    total_piksel = cv2.countNonZero(mask_objek)

    if total_piksel == 0:
        raise ValueError("Area objek pisang kosong.")

    # Rentang HSV untuk setiap warna.
    # Hitam (V sampai AMBANG_GELAP) untuk kategori busuk. Batas bawah V pada
    # warna lain berada tepat di atasnya agar rentangnya tidak tumpang tindih.
    v_min = AMBANG_GELAP + 1
    rentang_warna = {
        "Hijau": (np.array([35, 40, v_min]), np.array([85, 255, 255])),
        "Kuning": (np.array([18, 50, v_min]), np.array([34, 255, 255])),
        "Cokelat": (np.array([2, 40, v_min]), np.array([17, 255, 200])),
        "Hitam": (np.array([0, 0, 0]), np.array([179, 255, AMBANG_GELAP])),
    }

    persentase = {}

    for warna, (bawah, atas) in rentang_warna.items():
        mask_warna = cv2.inRange(hsv, bawah, atas)
        mask_warna = cv2.bitwise_and(mask_warna, mask_objek)
        persentase[warna] = cv2.countNonZero(mask_warna) / total_piksel * 100

    # Menentukan warna dominan
    warna_dominan = max(persentase, key=persentase.get)

    klasifikasi = {
        "Hijau": "Belum Matang",
        "Kuning": "Matang",
        "Cokelat": "Terlalu Matang",
        "Hitam": "Busuk",
    }

    kematangan = klasifikasi[warna_dominan]

    # Hasil segmentasi
    hasil_segmentasi = cv2.bitwise_and(gambar, gambar, mask=mask_objek)

    return (
        gambar, hasil_segmentasi, persentase, warna_dominan, kematangan,
        mask_objek > 0,
    )


# ==========================================================
# 2. INFORMASI DETAIL CITRA (SEBELUM DAN SESUDAH ANALISIS)
#    Hanya membaca hasil deteksi_pisang(), tidak mengubahnya.
# ==========================================================

def angka(x, d=2):
    """Format Indonesia: titik untuk ribuan, koma untuk desimal."""
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def format_ukuran(n):
    for satuan in ("B", "KB", "MB"):
        if n < 1024:
            return f"{angka(n, 0 if satuan == 'B' else 1)} {satuan}"
        n /= 1024
    return f"{angka(n, 1)} GB"


def bagian(judul, *isi):
    return "\n".join([judul, "-" * len(judul), *isi])


def kv(label, nilai):
    return f"{label:<20}: {nilai}"


def tabel_statistik(citra, kanal):
    baris = ["  kanal   min   maks   rata-rata   simpang baku"]
    for i, nama in enumerate(kanal):
        k = citra[:, :, i]
        baris.append(
            f"  {nama:<5} {int(k.min()):5d} {int(k.max()):6d} "
            f"{k.mean():11.2f} {k.std():14.2f}".replace(".", ",")
        )
    return "\n".join(baris)


def potong(citra, cx, cy, ukuran=5):
    """Cuplikan matriks ukuran x ukuran di sekitar titik (cx, cy)."""
    h, w = citra.shape[:2]
    u = min(ukuran, h, w)
    x0 = int(np.clip(cx - u // 2, 0, w - u))
    y0 = int(np.clip(cy - u // 2, 0, h - u))
    return citra[y0:y0 + u, x0:x0 + u], x0, y0


def blok_matriks(cuplikan, kanal):
    baris = []
    for i, nama in enumerate(kanal):
        baris.append(f"  Kanal {nama}")
        baris += [
            "    " + "  ".join(f"{int(v):3d}" for v in b)
            for b in cuplikan[:, :, i]
        ]
        baris.append("")
    return "\n".join(baris).rstrip()


def histogram_objek(hsv, mask):
    """
    Histogram kanal H (180 bin), S (256 bin), dan V (256 bin) dengan
    cv2.calcHist, hanya dihitung pada piksel objek pisang.
    """
    mask8 = mask.astype(np.uint8) * 255
    return {
        "H": cv2.calcHist([hsv], [0], mask8, [180], [0, 180]).flatten(),
        "S": cv2.calcHist([hsv], [1], mask8, [256], [0, 256]).flatten(),
        "V": cv2.calcHist([hsv], [2], mask8, [256], [0, 256]).flatten(),
    }


def ringkas_histogram(hist, n_obj):
    """Baris teks ringkasan histogram: modus dan beberapa persentase."""
    h, s, v = hist["H"], hist["S"], hist["V"]
    n = max(n_obj, 1)
    return [
        "Dihitung dengan cv2.calcHist, hanya pada piksel objek.",
        kv("Modus H", f"{int(h.argmax())}  ({angka(h.max(), 0)} piksel)"),
        kv("Modus S", f"{int(s.argmax())}  ({angka(s.max(), 0)} piksel)"),
        kv("Modus V", f"{int(v.argmax())}  ({angka(v.max(), 0)} piksel)"),
        kv("Piksel H 35-85", f"{angka(h[35:86].sum() / n * 100)}%  (rentang hijau)"),
        kv("Piksel H 18-34", f"{angka(h[18:35].sum() / n * 100)}%  (rentang kuning)"),
        kv("Piksel H 2-17", f"{angka(h[2:18].sum() / n * 100)}%  (rentang cokelat)"),
        kv("Piksel V <= %d" % AMBANG_GELAP, f"{angka(v[:AMBANG_GELAP + 1].sum() / n * 100)}%  (hitam)"),
    ]


def info_citra(nama_file, gambar, segmentasi, mask, persentase, dominan, kematangan):
    """Mengembalikan (teks_sebelum, teks_sesudah, histogram)."""
    asli = cv2.imread(nama_file)          # citra mentah, sebelum diubah ukuran
    h0, w0, c0 = asli.shape
    h1, w1 = gambar.shape[:2]

    blur = cv2.GaussianBlur(gambar, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    hist = histogram_objek(hsv, mask)

    ys, xs = np.nonzero(mask)
    n_obj, n_all = int(mask.sum()), h1 * w1

    if n_obj:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = int(xs.mean()), int(ys.mean())
        rata_bgr, rata_hsv = gambar[mask].mean(axis=0), hsv[mask].mean(axis=0)
    else:
        cx, cy = w1 // 2, h1 // 2
        x0 = x1 = cx
        y0 = y1 = cy
        rata_bgr = rata_hsv = np.zeros(3)

    # ---------- Sebelum analisis ----------
    cup0, ax, ay = potong(asli, cx * w0 // w1, cy * h0 // h1)
    sebelum = "\n\n".join([
        bagian(
            "Citra asli (belum diproses)",
            kv("Berkas", os.path.basename(nama_file)),
            kv("Ukuran berkas", format_ukuran(os.path.getsize(nama_file))),
            kv("Dimensi", f"{angka(w0, 0)} × {angka(h0, 0)} piksel (lebar × tinggi)"),
            kv("Bentuk matriks", f"({h0}, {w0}, {c0})  [tinggi, lebar, kanal]"),
            kv("Kanal", f"{c0}  (B, G, R)"),
            kv("Tipe data", asli.dtype),
            kv("Rasio aspek", angka(w0 / h0)),
            kv("Jumlah piksel", angka(h0 * w0, 0)),
            kv("Jumlah elemen", angka(asli.size, 0)),
            kv("Memori matriks", format_ukuran(asli.nbytes)),
        ),
        bagian("Statistik per kanal", tabel_statistik(asli, "BGR")),
        bagian(
            f"Cuplikan matriks {cup0.shape[0]}×{cup0.shape[1]} dari titik ({ax}, {ay})",
            "Diambil di sekitar pusat pisang pada citra asli.", "",
            blok_matriks(cup0, "BGR"),
        ),
    ])

    # ---------- Sesudah analisis ----------
    selisih_blur = np.abs(gambar.astype(np.int16) - blur.astype(np.int16)).mean()

    komposisi, terklasifikasi = [], 0
    for warna, nilai in persentase.items():
        n = int(round(nilai / 100 * n_obj))
        terklasifikasi += n
        komposisi.append(kv(warna, f"{angka(n, 0):>9} piksel  {angka(nilai):>7}%"))
    komposisi.append(kv(
        "Lainnya",
        f"{angka(max(0, n_obj - terklasifikasi), 0):>9} piksel  "
        f"{angka(max(0.0, 100 - sum(persentase.values()))):>7}%",
    ))

    banding = [
        "Sebelum: seluruh citra asli.  Sesudah: hanya piksel objek.",
        "  kanal   sebelum   sesudah   selisih",
    ]
    for i, k in enumerate("BGR"):
        a, b = asli[:, :, i].mean(), rata_bgr[i]
        banding.append(f"  {k:<5} {a:8.2f} {b:9.2f} {b - a:+9.2f}".replace(".", ","))

    cup1, bx, by = potong(gambar, cx, cy)
    cup_hsv, _, _ = potong(hsv, cx, cy)
    cup_seg, _, _ = potong(segmentasi, cx, cy)

    sesudah = "\n\n".join([
        bagian(
            "1. Pengubahan ukuran",
            kv("Dimensi", f"{w0} × {h0}  menjadi  {w1} × {h1} piksel"),
            kv("Faktor skala", f"x {angka(w1 / w0, 3)},  y {angka(h1 / h0, 3)}"),
            kv("Bentuk matriks", gambar.shape),
        ),
        bagian(
            "2. Gaussian blur",
            kv("Kernel", "5 × 5, sigma otomatis"),
            kv("Selisih rata-rata", f"{angka(selisih_blur)} tingkat keabuan per elemen"),
        ),
        bagian(
            "3. Konversi ke HSV",
            "Rentang OpenCV: H 0 sampai 179, S dan V 0 sampai 255.",
            tabel_statistik(hsv, "HSV"),
        ),
        bagian(
            "4. Segmentasi objek pisang",
            kv("Metode", "cv2.threshold pada kanal H, S, V"),
            kv("Piksel objek", f"{angka(n_obj, 0)}  ({angka(n_obj / n_all * 100)}% dari citra)"),
            kv("Piksel latar", f"{angka(n_all - n_obj, 0)}  ({angka((n_all - n_obj) / n_all * 100)}%)"),
            kv("Kotak pembatas", f"({x0}, {y0}) sampai ({x1}, {y1})"),
            kv("Ukuran kotak", f"{x1 - x0 + 1} × {y1 - y0 + 1} piksel"),
            kv("Titik pusat", f"({cx}, {cy})"),
            kv("Rata-rata B, G, R", ", ".join(angka(x) for x in rata_bgr)),
            kv("Rata-rata H, S, V", ", ".join(angka(x) for x in rata_hsv)),
        ),
        bagian(
            "5. Komposisi warna pada objek", *komposisi,
            "Jumlah piksel dihitung dari persentase.", "",
            kv("Warna dominan", dominan),
            kv("Klasifikasi", kematangan),
        ),
        bagian("6. Rata-rata nilai sebelum dan sesudah segmentasi", *banding),
        bagian(
            f"7. Cuplikan matriks {cup1.shape[0]}×{cup1.shape[1]} dari titik ({bx}, {by})",
            "Titik yang sama pada tiga tahap. Nilai 0 pada hasil segmentasi adalah latar.", "",
            "Citra yang diubah ukuran (B, G, R)", blok_matriks(cup1, "BGR"), "",
            "Citra HSV (H, S, V)", blok_matriks(cup_hsv, "HSV"), "",
            "Hasil segmentasi (B, G, R)", blok_matriks(cup_seg, "BGR"),
        ),
        bagian("8. Histogram kanal HSV pada objek", *ringkas_histogram(hist, n_obj)),
    ])

    return sebelum, sesudah, hist


# ==========================================================
# 3. DESAIN
#    Antarmuka netral: satu-satunya warna adalah warna data.
#    Panggung foto gelap karena hasil segmentasi berlatar hitam.
# ==========================================================

INK, MUTED, GARIS = "#1C2124", "#5F6A71", "#D3D8DA"
PANEL, STAGE, STAGE_TEKS = "#F2F4F3", "#1E2225", "#8E989E"

WARNA_DATA = {
    "Hijau": "#7FA83A",
    "Kuning": "#EFC31A",
    "Cokelat": "#7B4A24",
    "Hitam": "#2A2522",
}
WARNA_LAINNYA = "#D9DDDE"

PESAN_AWAL = "Mendukung berkas JPG, PNG, dan BMP."
TEKS_KOSONG = "Buka foto pisang untuk melihat tingkat kematangannya."

STYLE = Template("""
QMainWindow, QDialog { background: $PANEL; }
QFrame#stage { background: $STAGE; }
QFrame#panel { background: $PANEL; border-left: 1px solid $GARIS; }

QLabel#area { color: $STAGE_TEKS; font-size: 13px; background: transparent; }
QLabel#area[kosong="true"] { border: 1px solid #343A3E; }
QLabel#caption { color: $STAGE_TEKS; font-size: 12px; }

QLabel#verdict { color: $INK; font-size: 30px; font-weight: 700; }
QLabel#keterangan { color: $MUTED; font-size: 13px; }
QLabel#namaWarna, QLabel#nilaiWarna { color: $INK; font-size: 13px; }
QLabel[tebal="true"] { font-weight: 700; }

QPushButton#utama, QPushButton#sekunder {
    border-radius: 4px; padding: 10px 16px; min-height: 20px;
    font-size: 13px; font-weight: 600;
}
QPushButton#utama { background: $INK; color: white; border: 2px solid $INK; }
QPushButton#utama:hover { background: #343C41; border-color: #343C41; }
QPushButton#utama:pressed { background: #0F1214; border-color: #0F1214; }
QPushButton#sekunder {
    background: transparent; color: $INK; border: 2px solid #C4CACC;
}
QPushButton#sekunder:hover { background: #E6E9E8; }
QPushButton#sekunder:pressed { background: #D9DDDC; }
QPushButton#sekunder:disabled { color: #A3ACB1; border-color: #DEE2E3; }
QPushButton:focus { border-color: #4A7BD0; }

QTabWidget::pane { border: none; }
QTabBar::tab {
    padding: 8px 14px; margin-right: 6px; color: $MUTED; font-size: 13px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover:!selected { color: $INK; }
QTabBar::tab:selected {
    color: $INK; font-weight: 600; border-bottom: 2px solid $INK;
}
QPlainTextEdit#detail {
    background: #FBFCFB; color: $INK; border: 1px solid $GARIS; padding: 10px;
    selection-background-color: #F6E39A; selection-color: $INK;
}

QStatusBar {
    background: $PANEL; color: $MUTED; font-size: 12px;
    border-top: 1px solid $GARIS;
}
QStatusBar::item { border: none; }
""").substitute(
    INK=INK, MUTED=MUTED, GARIS=GARIS, PANEL=PANEL,
    STAGE=STAGE, STAGE_TEKS=STAGE_TEKS,
)


def set_prop(widget, nama, nilai):
    widget.setProperty(nama, nilai)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def numpy_ke_pixmap(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    )


def label(teks="", nama=None, wrap=False, rata=None):
    lb = QLabel(teks)
    if nama:
        lb.setObjectName(nama)
    lb.setWordWrap(wrap)
    if rata:
        lb.setAlignment(rata)
    return lb


def tombol(teks, nama, fungsi, pintasan=None):
    tb = QPushButton(teks)
    tb.setObjectName(nama)
    tb.setCursor(Qt.PointingHandCursor)
    tb.clicked.connect(fungsi)
    if pintasan:
        tb.setShortcut(pintasan)
    return tb


# ==========================================================
# 4. KOMPONEN
# ==========================================================

class AreaGambar(QLabel):
    """Menampilkan gambar dengan rasio asli, menyesuaikan ukuran jendela."""

    def __init__(self, teks_kosong):
        super().__init__()
        self._teks_kosong = teks_kosong
        self._pixmap = None
        self.setObjectName("area")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.hapus()

    def set_gambar(self, pixmap):
        self._pixmap = pixmap
        set_prop(self, "kosong", False)
        self._perbarui()

    def hapus(self):
        self._pixmap = None
        self.clear()
        self.setText(self._teks_kosong)
        set_prop(self, "kosong", True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._perbarui()

    def _perbarui(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))


class JalurWarna(QWidget):
    """
    Satu jalur horizontal yang diisi seperti kulit pisang yang menua:
    hijau, kuning, cokelat, hitam. Lebar tiap bagian sama dengan
    persentasenya; sisa piksel yang tidak masuk rentang berwarna abu-abu.
    """

    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self._persen, self._t = {}, 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(450)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._ubah)

    def _ubah(self, nilai):
        self._t = float(nilai)
        self.update()

    def atur(self, persen=None):
        """Isi jalur dengan animasi; tanpa argumen, kosongkan."""
        self._anim.stop()
        self._persen = dict(persen or {})
        if self._persen:
            self._anim.start()
        else:
            self._t = 0.0
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        bentuk = QPainterPath()
        bentuk.addRoundedRect(r, 4, 4)
        p.setClipPath(bentuk)
        p.fillRect(r, QColor(WARNA_LAINNYA))

        x = r.left()
        for warna, kode in WARNA_DATA.items():
            lebar = r.width() * self._persen.get(warna, 0) / 100 * self._t
            p.fillRect(QRectF(x, r.top(), lebar, r.height()), QColor(kode))
            x += lebar


class BarisWarna:
    """Satu baris legenda: kotak warna, nama, persentase."""

    def __init__(self, grid, indeks, nama, kode):
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background: {kode}; border-radius: 2px;")
        self.nama = label(nama, "namaWarna")
        self.nilai = label("", "nilaiWarna", rata=Qt.AlignRight | Qt.AlignVCenter)
        for kolom, widget in enumerate((swatch, self.nama, self.nilai)):
            grid.addWidget(widget, indeks, kolom)

    def isi(self, teks, tebal=False):
        self.nilai.setText(teks)
        set_prop(self.nama, "tebal", tebal)
        set_prop(self.nilai, "tebal", tebal)


class GrafikHistogram(QWidget):
    """
    Grafik batang satu histogram. Pita berwarna menandai rentang warna
    pada klasifikasi, garis putus-putus menandai ambang thresholding.
    """

    def __init__(self, judul, data, pita=(), garis=()):
        super().__init__()
        self._judul, self._data = judul, np.asarray(data, dtype=float)
        self._pita, self._garis = pita, garis
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#FBFCFB"))
        n = len(self._data)
        area = QRectF(54, 26, self.width() - 54 - 14, self.height() - 26 - 24)
        if n == 0 or area.width() <= 0 or area.height() <= 0:
            return
        maks = max(self._data.max(), 1.0)
        lebar = area.width() / n

        for a, b, kode in self._pita:
            warna = QColor(kode)
            warna.setAlpha(55)
            x0 = area.left() + a * lebar
            p.fillRect(QRectF(x0, area.top(), (b + 1 - a) * lebar, area.height()), warna)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(INK))
        for i, nilai in enumerate(self._data):
            if nilai > 0:
                t = area.height() * nilai / maks
                p.drawRect(QRectF(
                    area.left() + i * lebar, area.bottom() - t, max(lebar, 1.0), t
                ))

        huruf = p.font()
        huruf.setPointSize(8)
        p.setFont(huruf)
        for nilai, teks in self._garis:
            x = area.left() + nilai * lebar
            p.setPen(QPen(QColor("#C0392B"), 1, Qt.DashLine))
            p.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))
            p.setPen(QColor("#C0392B"))
            p.drawText(int(x) + 3, int(area.top()) + 11, teks)

        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(GARIS))
        p.drawRect(area)
        p.setPen(QColor(INK))
        huruf.setBold(True)
        p.setFont(huruf)
        p.drawText(int(area.left()), 15, self._judul)
        huruf.setBold(False)
        p.setFont(huruf)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, area.top() - 2, area.left() - 6, 14),
                   Qt.AlignRight | Qt.AlignTop, angka(maks, 0))
        p.drawText(QRectF(0, area.bottom() - 12, area.left() - 6, 14),
                   Qt.AlignRight | Qt.AlignBottom, "0")
        p.drawText(int(area.left()), int(self.height()) - 8, "0")
        p.drawText(int(area.right()) - 24, int(self.height()) - 8, str(n - 1))


class DialogDetail(QDialog):
    """Detail citra: tab sebelum, tab sesudah, dan tab histogram."""

    def __init__(self, parent, nama_berkas, teks_sebelum, teks_sesudah,
                 histogram=None):
        super().__init__(parent)
        self.setWindowTitle(f"Detail citra: {nama_berkas}")
        self.resize(800, 680)
        self._nama_berkas = nama_berkas
        self._teks_semua = f"{teks_sebelum}\n\n\n{teks_sesudah}\n"

        tata = QVBoxLayout(self)
        tata.setContentsMargins(20, 14, 20, 18)
        tata.setSpacing(12)

        huruf = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        huruf.setPointSize(10)

        tab = QTabWidget()
        for judul, teks in (
            ("Sebelum analisis", teks_sebelum),
            ("Sesudah analisis", teks_sesudah),
        ):
            editor = QPlainTextEdit(teks)
            editor.setObjectName("detail")
            editor.setReadOnly(True)
            editor.setLineWrapMode(QPlainTextEdit.NoWrap)
            editor.setFont(huruf)
            tab.addTab(editor, judul)
        if histogram:
            tab.addTab(self._buat_tab_histogram(histogram), "Histogram")
        tata.addWidget(tab, 1)

        self.label_info = label("", "keterangan")
        baris = QHBoxLayout()
        baris.addWidget(self.label_info, 1)
        baris.addWidget(tombol("Salin semua", "sekunder", self.salin))
        baris.addSpacing(8)
        baris.addWidget(tombol("Simpan .txt", "utama", self.simpan))
        tata.addLayout(baris)

    @staticmethod
    def _buat_tab_histogram(hist):
        halaman = QWidget()
        tata = QVBoxLayout(halaman)
        tata.setContentsMargins(0, 10, 0, 0)
        tata.setSpacing(8)
        tata.addWidget(GrafikHistogram(
            "Histogram Hue (H) pada objek", hist["H"],
            pita=[(35, 85, WARNA_DATA["Hijau"]),
                  (18, 34, WARNA_DATA["Kuning"]),
                  (2, 17, WARNA_DATA["Cokelat"])],
            garis=[(90, "H = 90")],
        ), 1)
        tata.addWidget(GrafikHistogram(
            "Histogram Saturation (S) pada objek", hist["S"],
            garis=[(35, "S = 35")],
        ), 1)
        tata.addWidget(GrafikHistogram(
            "Histogram Value (V) pada objek", hist["V"],
            pita=[(0, AMBANG_GELAP, WARNA_DATA["Hitam"])],
            garis=[(30, "V = 30"), (AMBANG_GELAP, "V = 55")],
        ), 1)
        return halaman

    def salin(self):
        QApplication.clipboard().setText(self._teks_semua)
        self.label_info.setText("Teks disalin ke papan klip.")

    def simpan(self):
        nama, _ = QFileDialog.getSaveFileName(
            self, "Simpan detail citra",
            f"detail_{os.path.splitext(self._nama_berkas)[0]}.txt",
            "Teks (*.txt)",
        )
        if not nama:
            return
        try:
            with open(nama, "w", encoding="utf-8") as f:
                f.write(self._teks_semua)
            self.label_info.setText(f"Disimpan: {os.path.basename(nama)}")
        except OSError as e:
            QMessageBox.warning(self, "Berkas tidak bisa disimpan", str(e))


# ==========================================================
# 5. JENDELA UTAMA
# ==========================================================

class JendelaPisang(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Kematangan Pisang")
        self.resize(1120, 700)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)

        pusat = QWidget()
        self.setCentralWidget(pusat)
        utama = QHBoxLayout(pusat)
        utama.setContentsMargins(0, 0, 0, 0)
        utama.setSpacing(0)
        utama.addWidget(self._buat_panggung(), 1)
        utama.addWidget(self._buat_panel())

        self.statusBar().setSizeGripEnabled(False)
        self.setStyleSheet(STYLE)
        self.reset_hasil()

    def _buat_panggung(self):
        panggung = QFrame()
        panggung.setObjectName("stage")
        tata = QHBoxLayout(panggung)
        tata.setContentsMargins(28, 28, 28, 24)
        tata.setSpacing(24)

        self.area_asli = AreaGambar(
            "Seret foto pisang ke sini,\natau pilih Buka gambar (Ctrl+O)"
        )
        self.area_segmentasi = AreaGambar("Hasil segmentasi tampil di sini")

        for area, keterangan in (
            (self.area_asli, "Asli"), (self.area_segmentasi, "Segmentasi")
        ):
            kolom = QVBoxLayout()
            kolom.setSpacing(10)
            kolom.addWidget(area, 1)
            kolom.addWidget(label(keterangan, "caption", rata=Qt.AlignCenter))
            tata.addLayout(kolom, 1)
        return panggung

    def _buat_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(330)
        tata = QVBoxLayout(panel)
        tata.setContentsMargins(28, 28, 28, 24)
        tata.setSpacing(0)

        self.label_verdict = label("", "verdict", wrap=True)
        self.label_ket = label("", "keterangan", wrap=True)
        self.jalur = JalurWarna()

        tata.addWidget(self.label_verdict)
        tata.addSpacing(8)
        tata.addWidget(self.label_ket)
        tata.addSpacing(32)
        tata.addWidget(self.jalur)
        tata.addSpacing(16)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        self.baris = {
            warna: BarisWarna(grid, i, warna, kode)
            for i, (warna, kode) in enumerate(
                {**WARNA_DATA, "Lainnya": WARNA_LAINNYA}.items()
            )
        }
        tata.addLayout(grid)
        tata.addStretch(1)

        self.tombol_detail = tombol(
            "Detail citra", "sekunder", self.buka_detail, "Ctrl+D"
        )
        for widget in (
            tombol("Buka gambar", "utama", self.pilih_gambar, "Ctrl+O"),
            self.tombol_detail,
            tombol("Bersihkan", "sekunder", self.reset_hasil),
        ):
            tata.addWidget(widget)
            tata.addSpacing(8)
        return panel

    # ---------- Seret dan lepas ----------
    @staticmethod
    def _ambil_path(mime):
        for url in mime.urls():
            path = url.toLocalFile()
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                return path
        return None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self._ambil_path(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._ambil_path(event.mimeData())
        if path:
            self.proses(path)

    # ---------- Aksi ----------
    def pilih_gambar(self):
        nama_file, _ = QFileDialog.getOpenFileName(
            self, "Pilih gambar pisang", "",
            "Gambar (*.jpg *.jpeg *.png *.bmp);;Semua file (*.*)",
        )
        if nama_file:
            self.proses(nama_file)

    def proses(self, nama_file):
        try:
            gambar, segmentasi, persentase, dominan, kematangan, mask = (
                deteksi_pisang(nama_file)
            )
        except Exception as e:
            QMessageBox.warning(self, "Gambar tidak bisa diproses", str(e))
            return

        self.area_asli.set_gambar(numpy_ke_pixmap(gambar))
        self.area_segmentasi.set_gambar(numpy_ke_pixmap(segmentasi))

        for warna, nilai in persentase.items():
            self.baris[warna].isi(f"{angka(nilai, 1)}%", warna == dominan)
        sisa = max(0.0, 100.0 - sum(persentase.values()))
        self.baris["Lainnya"].isi(f"{angka(sisa, 1)}%")
        self.jalur.atur(persentase)

        self.label_verdict.setText(kematangan.capitalize())
        self.label_ket.setText(
            f"Warna dominan {dominan.lower()}, "
            f"{angka(persentase[dominan], 1)}% dari area pisang."
        )

        # Kegagalan membuat detail tidak boleh membatalkan hasil utama
        self.nama_berkas = os.path.basename(nama_file)
        try:
            self.info = info_citra(
                nama_file, gambar, segmentasi, mask,
                persentase, dominan, kematangan
            )
        except Exception as e:
            self.info = (f"Detail citra tidak bisa dibuat: {e}",) * 2 + (None,)
        self.tombol_detail.setEnabled(True)
        self.statusBar().showMessage(
            f"{self.nama_berkas}  (dianalisis pada "
            f"{gambar.shape[1]}×{gambar.shape[0]} piksel)"
        )

    def buka_detail(self):
        if self.info:
            DialogDetail(self, self.nama_berkas, *self.info).exec_()

    def reset_hasil(self):
        self.area_asli.hapus()
        self.area_segmentasi.hapus()
        self.jalur.atur()
        for baris in self.baris.values():
            baris.isi("")
        self.label_verdict.setText("Belum ada gambar")
        self.label_ket.setText(TEKS_KOSONG)
        self.info, self.nama_berkas = None, ""
        self.tombol_detail.setEnabled(False)
        self.statusBar().showMessage(PESAN_AWAL)


# ==========================================================
# 6. MENJALANKAN APLIKASI
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("Bahnschrift")
    if hasattr(font, "setFamilies"):
        font.setFamilies(["Bahnschrift", "Segoe UI", "Helvetica Neue", "Noto Sans"])
    font.setPointSize(10)
    app.setFont(font)

    jendela = JendelaPisang()
    jendela.show()
    sys.exit(app.exec_())