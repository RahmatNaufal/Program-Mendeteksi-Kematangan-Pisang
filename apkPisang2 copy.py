import os
import sys
import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QProgressBar, QFileDialog, QMessageBox,
    QGraphicsDropShadowEffect, QSizePolicy
)


# ==============================================j============
# 1. PROSES DETEKSI KEMATANGAN PISANG (LOGIKA ASLI, TIDAK DIUBAH)
# ==========================================================

def deteksi_pisang(nama_file):
    gambar = cv2.imread(nama_file)

    if gambar is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    gambar = cv2.resize(gambar, (600, 400))

    blur = cv2.GaussianBlur(gambar, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # SEGMENTASI OBJEK PISANG
    h, s, v = cv2.split(hsv)

    # Mengambil objek pisang dan menghilangkan latar putih
    bawah_pisang = np.array([0, 35, 30])
    atas_pisang = np.array([90, 255, 255])

    mask_pisang = cv2.inRange(hsv, bawah_pisang, atas_pisang)

    # Menghilangkan noise kecil
    kernel_kecil = np.ones((3, 3), np.uint8)
    kernel_sedang = np.ones((7, 7), np.uint8)

    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_OPEN, kernel_kecil)

    # Menutup celah kecil pada objek
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
    luas_gambar = tinggi * lebar

    if cv2.contourArea(kontur_terbesar) < luas_gambar * 0.05:
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

    # Rentang HSV untuk setiap warna
    rentang_warna = {
        "Hijau": (np.array([35, 40, 40]), np.array([85, 255, 255])),
        "Kuning": (np.array([18, 50, 50]), np.array([34, 255, 255])),
        "Cokelat": (np.array([2, 40, 20]), np.array([17, 255, 200])),
    }

    persentase = {}

    for warna, (bawah, atas) in rentang_warna.items():
        mask_warna = cv2.inRange(hsv, bawah, atas)
        mask_warna = cv2.bitwise_and(mask_warna, mask_objek)
        jumlah = cv2.countNonZero(mask_warna)
        persentase[warna] = jumlah / total_piksel * 100

    # Menentukan warna dominan
    warna_dominan = max(persentase, key=persentase.get)

    klasifikasi = {
        "Hijau": "Belum Matang",
        "Kuning": "Matang",
        "Cokelat": "Terlalu Matang",
    }

    kematangan = klasifikasi[warna_dominan]

    # Hasil segmentasi
    hasil_segmentasi = cv2.bitwise_and(gambar, gambar, mask=mask_objek)

    return gambar, hasil_segmentasi, persentase, warna_dominan, kematangan


# ==========================================================
# 2. TEMA / WARNA
# ==========================================================

WARNA_TAMPILAN = {
    "Hijau": "#4CAF6A",
    "Kuning": "#F2C230",
    "Cokelat": "#A0694A",
}

WARNA_STATUS = {
    "Belum Matang": ("#E3F4E8", "#2E7D46"),
    "Matang": ("#FFF6D6", "#9A7400"),
    "Terlalu Matang": ("#F3E6DE", "#7A4A2E"),
}

STYLE = """
QMainWindow, QWidget#root {
    background: #F7F4EC;
}
QFrame#header {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #F7C948, stop:1 #F2A93B);
    border-radius: 18px;
}
QLabel#judul {
    color: #3B2A00;
    font-size: 26px;
    font-weight: 800;
}
QLabel#subjudul {
    color: #6B4F0A;
    font-size: 12px;
}
QFrame#kartu {
    background: #FFFFFF;
    border-radius: 16px;
}
QLabel#judulKartu {
    color: #5C4A1A;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#area {
    background: #FAF7EF;
    border: 2px dashed #E3DAC2;
    border-radius: 12px;
    color: #A89F88;
    font-size: 13px;
}
QLabel#judulHasil {
    color: #3B2A00;
    font-size: 16px;
    font-weight: 800;
}
QLabel#kecil {
    color: #8A8167;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#namaWarna {
    color: #3A3A3A;
    font-size: 12px;
}
QLabel#nilaiWarna {
    color: #222222;
    font-size: 12px;
    font-weight: 700;
}
QLabel#dominan {
    color: #3B2A00;
    font-size: 24px;
    font-weight: 800;
}
QLabel#status {
    background: #EEEEEE;
    color: #555555;
    border-radius: 14px;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 700;
}
QFrame#garis {
    background: #EFE9D8;
    max-height: 1px;
    min-height: 1px;
}
QProgressBar {
    background: #F0EBDC;
    border: none;
    border-radius: 6px;
    max-height: 12px;
    min-height: 12px;
}
QPushButton {
    border: none;
    border-radius: 12px;
    padding: 13px 28px;
    font-size: 13px;
    font-weight: 700;
    color: white;
}
QPushButton#pilih {
    background: #3E9B5C;
}
QPushButton#pilih:hover {
    background: #348550;
}
QPushButton#pilih:pressed {
    background: #2A6D41;
}
QPushButton#reset {
    background: #8C8570;
}
QPushButton#reset:hover {
    background: #766F5C;
}
QPushButton#reset:pressed {
    background: #625C4C;
}
QLabel#info {
    color: #8A8167;
    font-size: 11px;
}
"""


def beri_bayangan(widget, blur=30, alpha=40):
    efek = QGraphicsDropShadowEffect()
    efek.setBlurRadius(blur)
    efek.setOffset(0, 4)
    efek.setColor(QColor(38, 32, 46, alpha))
    widget.setGraphicsEffect(efek)


def numpy_ke_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ==========================================================
# 3. WIDGET GAMBAR (otomatis menyesuaikan ukuran jendela)
# ==========================================================

class AreaGambar(QLabel):
    def __init__(self):
        super().__init__("Belum ada gambar")
        self.setObjectName("area")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap = None

    def set_gambar(self, pixmap):
        self._pixmap = pixmap
        self._perbarui()

    def hapus(self):
        self._pixmap = None
        self.clear()
        self.setText("Belum ada gambar")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._perbarui()

    def _perbarui(self):
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(
                    self.width() - 8, self.height() - 8,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


def buat_kartu_gambar(judul):
    kartu = QFrame()
    kartu.setObjectName("kartu")
    beri_bayangan(kartu)

    tata = QVBoxLayout(kartu)
    tata.setContentsMargins(16, 14, 16, 16)
    tata.setSpacing(10)

    label = QLabel(judul)
    label.setObjectName("judulKartu")
    tata.addWidget(label)

    area = AreaGambar()
    tata.addWidget(area, 1)

    return kartu, area


# ==========================================================
# 4. JENDELA UTAMA
# ==========================================================

class JendelaPisang(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Kematangan Pisang")
        self.resize(1050, 760)
        self.setMinimumSize(900, 680)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        utama = QVBoxLayout(root)
        utama.setContentsMargins(24, 20, 24, 20)
        utama.setSpacing(18)

        utama.addWidget(self._buat_header())

        isi = QHBoxLayout()
        isi.setSpacing(18)

        # Kolom kiri: dua gambar
        kolom_gambar = QVBoxLayout()
        kolom_gambar.setSpacing(18)

        kartu_asli, self.area_asli = buat_kartu_gambar("GAMBAR ASLI")
        kartu_seg, self.area_segmentasi = buat_kartu_gambar("HASIL SEGMENTASI")
        kolom_gambar.addWidget(kartu_asli, 1)
        kolom_gambar.addWidget(kartu_seg, 1)

        isi.addLayout(kolom_gambar, 1)
        isi.addWidget(self._buat_panel_hasil())

        utama.addLayout(isi, 1)
        utama.addLayout(self._buat_tombol())

        self.setStyleSheet(STYLE)

    # ---------- Header ----------
    def _buat_header(self):
        header = QFrame()
        header.setObjectName("header")
        beri_bayangan(header, 24, 50)

        tata = QVBoxLayout(header)
        tata.setContentsMargins(20, 18, 20, 18)
        tata.setSpacing(4)

        judul = QLabel("🍌  DETEKSI KEMATANGAN PISANG")
        judul.setObjectName("judul")
        judul.setAlignment(Qt.AlignCenter)

        sub = QLabel("Pengolahan Citra Digital  |  Metode HSV")
        sub.setObjectName("subjudul")
        sub.setAlignment(Qt.AlignCenter)

        tata.addWidget(judul)
        tata.addWidget(sub)
        return header

    # ---------- Panel hasil ----------
    def _buat_panel_hasil(self):
        panel = QFrame()
        panel.setObjectName("kartu")
        panel.setFixedWidth(310)
        beri_bayangan(panel)

        tata = QVBoxLayout(panel)
        tata.setContentsMargins(22, 20, 22, 22)
        tata.setSpacing(6)

        judul = QLabel("HASIL ANALISIS")
        judul.setObjectName("judulHasil")
        tata.addWidget(judul)
        tata.addSpacing(8)

        sub = QLabel("PERSENTASE WARNA")
        sub.setObjectName("kecil")
        tata.addWidget(sub)
        tata.addSpacing(4)

        self.label_persen = {}
        self.batang = {}

        for warna, kode in WARNA_TAMPILAN.items():
            baris = QHBoxLayout()
            nama = QLabel(warna)
            nama.setObjectName("namaWarna")
            nilai = QLabel("0.00%")
            nilai.setObjectName("nilaiWarna")
            nilai.setAlignment(Qt.AlignRight)
            baris.addWidget(nama)
            baris.addWidget(nilai)
            tata.addLayout(baris)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {kode}; border-radius: 6px; }}"
            )
            tata.addWidget(bar)
            tata.addSpacing(8)

            self.label_persen[warna] = nilai
            self.batang[warna] = bar

        garis = QFrame()
        garis.setObjectName("garis")
        tata.addSpacing(4)
        tata.addWidget(garis)
        tata.addSpacing(8)

        k1 = QLabel("WARNA DOMINAN")
        k1.setObjectName("kecil")
        tata.addWidget(k1)

        self.label_dominan = QLabel("-")
        self.label_dominan.setObjectName("dominan")
        tata.addWidget(self.label_dominan)
        tata.addSpacing(12)

        k2 = QLabel("TINGKAT KEMATANGAN")
        k2.setObjectName("kecil")
        tata.addWidget(k2)

        self.label_kematangan = QLabel("Belum dianalisis")
        self.label_kematangan.setObjectName("status")
        self.label_kematangan.setAlignment(Qt.AlignCenter)
        self.label_kematangan.setWordWrap(True)
        tata.addWidget(self.label_kematangan)

        tata.addStretch(1)
        return panel

    # ---------- Tombol ----------
    def _buat_tombol(self):
        kolom = QVBoxLayout()
        kolom.setSpacing(8)

        baris = QHBoxLayout()
        baris.addStretch(1)

        tombol_pilih = QPushButton("📂  PILIH GAMBAR")
        tombol_pilih.setObjectName("pilih")
        tombol_pilih.setCursor(Qt.PointingHandCursor)
        tombol_pilih.clicked.connect(self.pilih_gambar)

        tombol_reset = QPushButton("↺  RESET HASIL")
        tombol_reset.setObjectName("reset")
        tombol_reset.setCursor(Qt.PointingHandCursor)
        tombol_reset.clicked.connect(self.reset_hasil)

        baris.addWidget(tombol_pilih)
        baris.addSpacing(10)
        baris.addWidget(tombol_reset)
        baris.addStretch(1)

        info = QLabel("Pilih gambar lain untuk melakukan analisis berikutnya.")
        info.setObjectName("info")
        info.setAlignment(Qt.AlignCenter)

        kolom.addLayout(baris)
        kolom.addWidget(info)
        return kolom

    # ---------- Aksi ----------
    def pilih_gambar(self):
        nama_file, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih gambar pisang",
            "",
            "Gambar (*.jpg *.jpeg *.png *.bmp);;Semua file (*.*)",
        )

        if not nama_file:
            return

        try:
            gambar, segmentasi, persentase, dominan, kematangan = (
                deteksi_pisang(nama_file)
            )

            self.area_asli.set_gambar(numpy_ke_pixmap(gambar))
            self.area_segmentasi.set_gambar(numpy_ke_pixmap(segmentasi))

            for warna, nilai in persentase.items():
                self.label_persen[warna].setText(f"{nilai:.2f}%")
                self.batang[warna].setValue(int(min(max(nilai, 0), 100) * 10))

            self.label_dominan.setText(dominan)
            self._set_status(kematangan)

        except Exception as e:
            QMessageBox.critical(self, "Gagal memproses gambar", str(e))

    def _set_status(self, kematangan):
        latar, teks = WARNA_STATUS.get(kematangan, ("#EEEEEE", "#555555"))
        self.label_kematangan.setText(kematangan)
        self.label_kematangan.setStyleSheet(
            f"background: {latar}; color: {teks}; border-radius: 14px;"
            "padding: 10px 16px; font-size: 14px; font-weight: 700;"
        )

    def reset_hasil(self):
        self.area_asli.hapus()
        self.area_segmentasi.hapus()

        for warna in self.label_persen:
            self.label_persen[warna].setText("0.00%")
            self.batang[warna].setValue(0)

        self.label_dominan.setText("-")
        self.label_kematangan.setText("Belum dianalisis")
        self.label_kematangan.setStyleSheet("")  # kembali ke gaya bawaan


# ==========================================================
# 5. MENJALANKAN APLIKASI
# ============================import os
import sys
from datetime import datetime
from string import Template

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QRectF, QSize, QVariantAnimation, QEasingCurve, pyqtSignal,
    QPropertyAnimation, QTimer
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPainterPath, QFontDatabase,
    QPen, QIcon, QFontMetrics
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
    QSizePolicy, QDialog, QTabWidget, QPlainTextEdit, QListWidget,
    QListWidgetItem, QStackedWidget, QLineEdit, QCheckBox,
    QGraphicsDropShadowEffect, QGraphicsOpacityEffect
)


# ==========================================================
# 1. PROSES DETEKSI KEMATANGAN PISANG
# ==========================================================

# Piksel dengan V (kecerahan) sampai nilai ini dianggap hitam.
AMBANG_GELAP = 55

# Sisi terpanjang gambar disesuaikan ke ukuran ini (rasio aspek tetap).
# Gambar 3:2 tetap menjadi 600 x 400 seperti sebelumnya.
UKURAN_MAKS = 600

# Batas pemeriksaan "apakah objek ini pisang?". Ubah angkanya bila perlu.
BATAS_WARNA_MIN = 80.0     # % piksel objek yang harus masuk rentang warna pisang
BATAS_ASPEK_MIN = 1.25     # rasio sisi panjang : sisi pendek (bulat = 1,0)
BATAS_EXTENT_MAKS = 0.90   # luas objek : luas kotak miring (persegi = 1,0)
BATAS_SOLIDITAS_MIN = 0.40 # luas objek : luas cembung (makin kecil makin bergerigi)
BATAS_LUAS_MAKS = 0.90     # objek tidak boleh memenuhi hampir seluruh gambar

# Kulit pisang busuk yang menghitam sering tampak hitam pudar: saturasinya
# rendah dan kecerahannya sedikit di atas hitam pekat. Dua batas ini dipakai
# agar piksel seperti itu tetap terhitung sebagai warna "Hitam".
BATAS_HITAM_S = 70
BATAS_HITAM_V = 90

# MobileNetV2 dilatih dengan pisang kuning, sehingga pisang yang hampir hitam
# sering dikenali sebagai benda lain (misalnya hook atau ocarina). Objek tetap
# diterima sebagai pisang busuk bila hitam mendominasi dan bentuknya memanjang.
# Bayangan di luar tubuh pisang tidak ikut dihitung, jadi batas hitam dibuat
# sedikit lebih longgar agar pisang busuk tetap lolos.
BATAS_GELAP_PISANG = 30.0  # % piksel hitam minimal pada objek
BATAS_ASPEK_GELAP = 1.4    # rasio sisi minimal agar tetap dianggap memanjang

# Satu pisang minimal seluas ini (% citra) agar bintik atau noise tidak ikut
# dihitung sebagai pisang tersendiri.
BATAS_LUAS_OBJEK = 2.0

KLASIFIKASI_WARNA = {
    "Hijau": "Belum Matang",
    "Kuning": "Matang",
    "Cokelat": "Terlalu Matang",
    "Hitam": "Busuk",
}

# Rentang HSV untuk setiap warna. Hitam memakai dua rentang: hitam pekat dan
# hitam pudar (banyak pada kulit pisang busuk). Batas bawah V pada warna lain
# berada tepat di atas hitam pekat agar tidak tumpang tindih.
_V_MIN = AMBANG_GELAP + 1
RENTANG_WARNA = {
    "Hijau": [(np.array([35, 40, _V_MIN]), np.array([85, 255, 255]))],
    "Kuning": [(np.array([18, 50, _V_MIN]), np.array([34, 255, 255]))],
    "Cokelat": [(np.array([2, 40, _V_MIN]), np.array([17, 255, 200]))],
    "Hitam": [
        (np.array([0, 0, 0]), np.array([179, 255, AMBANG_GELAP])),
        (np.array([0, 0, 0]), np.array([179, BATAS_HITAM_S, BATAS_HITAM_V])),
    ],
}


class BukanPisang(ValueError):
    """Dilempar bila gambar tidak berisi pisang. Membawa alasan dan tahapan
    yang sudah dijalankan agar GUI tetap bisa menampilkannya."""

    def __init__(self, alasan, gambar, langkah):
        super().__init__(alasan)
        self.alasan = alasan
        self.gambar = gambar
        self.langkah = langkah


# ----------------------------------------------------------
# Model pengenal objek (MobileNetV2 hasil pelatihan ImageNet, format ONNX)
# Dibaca dengan cv2.dnn, jadi tidak perlu pustaka tambahan.
# Siapkan berkasnya dengan siapkan_model.py, lalu letakkan di folder
# yang sama dengan program ini. Bila berkas tidak ada, program tetap
# jalan dan hanya memakai pemeriksaan warna dan bentuk.
# ----------------------------------------------------------
FOLDER = os.path.dirname(os.path.abspath(__file__))
FILE_MODEL = os.path.join(FOLDER, "mobilenetv2-12.onnx")
FILE_LABEL = os.path.join(FOLDER, "imagenet_labels.txt")   # opsional

INDEKS_PISANG = 954     # nomor kelas "banana" pada ImageNet-1k
AMBANG_PELUANG = 0.15   # peluang "banana" di atas ini -> dianggap pisang
AMBANG_TOLAK = 0.05     # peluang "banana" di bawah ini -> dianggap bukan pisang
                        # (di antara keduanya, model dianggap ragu)
TOP_K = 5               # banana masuk lima besar -> dianggap pisang

RATA2 = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

_cache = {}


def muat_model():
    """Memuat model sekali saja. Mengembalikan None bila tidak tersedia."""
    if "net" not in _cache:
        net = None
        if os.path.isfile(FILE_MODEL):
            try:
                net = cv2.dnn.readNetFromONNX(FILE_MODEL)
            except cv2.error:
                net = None
        _cache["net"] = net
    return _cache["net"]


def nama_kelas(indeks):
    """Nama kelas ImageNet (dari imagenet_labels.txt bila ada)."""
    if "label" not in _cache:
        daftar = []
        if os.path.isfile(FILE_LABEL):
            with open(FILE_LABEL, encoding="utf-8") as f:
                daftar = [b.strip() for b in f if b.strip()]
        _cache["label"] = daftar
    if indeks == INDEKS_PISANG:
        return "banana (pisang)"
    daftar = _cache["label"]
    return daftar[indeks] if indeks < len(daftar) else f"kelas #{indeks}"


def kenali_objek(gambar):
    """
    Menjalankan MobileNetV2 pada gambar BGR. Mengembalikan dict berisi
    lima kelas teratas dan peluang "banana", atau None bila model tidak ada.
    """
    net = muat_model()
    if net is None:
        return None
    try:
        # Skala 0..1, urutan RGB, lalu normalisasi ImageNet
        blob = cv2.dnn.blobFromImage(
            gambar, 1 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False
        )
        net.setInput((blob - RATA2) / STD)
        logit = np.asarray(net.forward(), dtype=np.float64).flatten()
    except cv2.error:
        return None

    if logit.size == 1001:           # sebagian model punya kelas "background"
        logit = logit[1:]
    if logit.min() >= 0 and abs(logit.sum() - 1.0) < 1e-3:
        peluang = logit              # keluaran model sudah berupa peluang
    else:
        e = np.exp(logit - logit.max())
        peluang = e / e.sum()        # logit -> peluang (softmax)

    urut = np.argsort(peluang)[::-1][:TOP_K]
    return {
        "top": [(int(i), float(peluang[i])) for i in urut],
        "peluang_pisang": float(peluang[INDEKS_PISANG]),
        "pisang_di_top": INDEKS_PISANG in urut,
        "kelas_top1": int(urut[0]),
        "peluang_top1": float(peluang[urut[0]]),
    }


def putuskan_pisang(pengenalan, alasan_bentuk, hitam=0.0, aspek=0.0):
    """
    Menggabungkan model dan pemeriksaan warna/bentuk.
    Mengembalikan (apakah_pisang, alasan_penolakan). Yang dilihat adalah
    peluang kelas "banana" itu sendiri, bukan seberapa yakin model pada kelas
    lain (peluang bisa terbagi, misalnya cucumber 56% + zucchini 43%).
      1. Peluang banana tinggi                 -> pisang.
      2. Hitam mendominasi dan objek memanjang -> pisang busuk, meskipun model
         salah mengenalinya karena dilatih pada pisang kuning.
      3. Peluang banana sangat rendah          -> bukan pisang.
      4. Di antaranya, atau model tidak ada    -> ikut pemeriksaan warna dan bentuk.
    """
    if pengenalan is not None:
        peluang = pengenalan["peluang_pisang"]
        if peluang >= AMBANG_PELUANG:
            return True, ""
        if (
            not alasan_bentuk
            and hitam >= BATAS_GELAP_PISANG
            and aspek >= BATAS_ASPEK_GELAP
        ):
            return True, ""
        if peluang < AMBANG_TOLAK:
            nama = nama_kelas(pengenalan["kelas_top1"])
            return False, (
                f"Komputer mengenali objek ini sebagai {nama} bukan pisang "
                # f"({angka(pengenalan['peluang_top1'] * 100, 0)}%), bukan pisang "
                # f"(peluang pisang hanya {angka(peluang * 100, 1)}%)."
            )
        awalan = "Model tidak yakin. "
    else:
        awalan = ""

    if alasan_bentuk:
        return False, awalan + " ".join(alasan_bentuk)
    return True, ""


def ambang(kanal, batas, jenis):
    """
    Thresholding satu kanal dengan cv2.threshold, mengembalikan citra biner.
    THRESH_BINARY     : 255 jika nilai > batas, selain itu 0.
    THRESH_BINARY_INV : 255 jika nilai <= batas, selain itu 0.
    """
    _, biner = cv2.threshold(kanal, batas, 255, jenis)
    return biner


def ke_bgr(citra):
    """Citra satu kanal (mask, kanal H/S/V) diubah ke 3 kanal agar bisa ditampilkan."""
    if citra.ndim == 2:
        return cv2.cvtColor(citra, cv2.COLOR_GRAY2BGR)
    return citra.copy()


def info_mask(mask):
    """Teks jumlah dan persentase piksel putih pada sebuah mask."""
    n = cv2.countNonZero(mask)
    return f"Piksel putih: {angka(n, 0)} ({angka(n / mask.size * 100, 1)}% dari citra)."


def mask_dari_kontur(kontur, bentuk):
    """Mask penuh (cv2.FILLED) untuk satu kontur."""
    mask = np.zeros(bentuk, np.uint8)
    cv2.drawContours(mask, [kontur], -1, 255, thickness=cv2.FILLED)
    return mask


def isi_lubang(mask):
    """
    Mengisi lubang (area kosong yang tertutup) pada mask biner.
    Latar di luar objek tetap kosong karena banjir dimulai dari luar gambar.
    """
    tinggi, lebar = mask.shape
    berlapis = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    banjir = berlapis.copy()
    cv2.floodFill(
        banjir, np.zeros((tinggi + 4, lebar + 4), np.uint8), (0, 0), 255
    )
    hasil = cv2.bitwise_or(berlapis, cv2.bitwise_not(banjir))
    return hasil[1:-1, 1:-1]


def bentuk_objek(mask_warna):
    """
    Memperkirakan bentuk utuh objek dari area berwarnanya.

    Celah sempit ditutup (closing 9 × 9), lalu lubang di dalamnya diisi.
    Bagian pisang yang menghitam jadi termasuk ke dalam bentuk, sedangkan
    bayangan yang jatuh di luar tubuh pisang tetap berada di luar bentuk.
    """
    kernel = np.ones((9, 9), np.uint8)
    bentuk = cv2.morphologyEx(mask_warna, cv2.MORPH_CLOSE, kernel)
    return isi_lubang(bentuk)


def persentase_warna(hsv, mask_objek):
    """
    Persentase tiap warna pisang di dalam sebuah mask objek, beserta mask
    per warna. Dipakai untuk objek gabungan maupun untuk tiap pisang.
    """
    total = cv2.countNonZero(mask_objek)
    if total == 0:
        return {warna: 0.0 for warna in RENTANG_WARNA}, {}

    hasil, peta = {}, {}
    tercakup = np.zeros_like(mask_objek)
    for warna, daftar in RENTANG_WARNA.items():
        mask_warna = np.zeros_like(mask_objek)
        for bawah, atas in daftar:
            mask_warna = cv2.bitwise_or(
                mask_warna, cv2.inRange(hsv, bawah, atas)
            )
        if warna == "Hitam":
            # Hitam pudar hanya dihitung pada piksel yang belum masuk warna lain.
            mask_warna = cv2.bitwise_and(mask_warna, cv2.bitwise_not(tercakup))
        mask_warna = cv2.bitwise_and(mask_warna, mask_objek)
        tercakup = cv2.bitwise_or(tercakup, mask_warna)
        hasil[warna] = cv2.countNonZero(mask_warna) / total * 100
        peta[warna] = mask_warna
    return hasil, peta


def ringkas_kematangan(daftar_pisang):
    """Jumlah pisang per tingkat kematangan, mis. '2 matang, 1 busuk'."""
    urutan = ["Belum Matang", "Matang", "Terlalu Matang", "Busuk"]
    jumlah = {}
    for objek in daftar_pisang:
        kelas = objek["kematangan"]
        jumlah[kelas] = jumlah.get(kelas, 0) + 1
    return ", ".join(
        f"{jumlah[kelas]} {kelas.lower()}" for kelas in urutan if kelas in jumlah
    )


def tandai_pisang(gambar, daftar_pisang):
    """Salinan citra dengan kotak dan nomor pada setiap pisang yang terdeteksi."""
    tanda = gambar.copy()
    for objek in daftar_pisang:
        x, y, w, h = objek["bbox"]
        b, g, r = hex_ke_bgr(WARNA_DATA[objek["dominan"]])
        cv2.rectangle(tanda, (x, y), (x + w, y + h), (b, g, r), 2)
        teks = str(objek["nomor"])
        (tw, th), _ = cv2.getTextSize(
            teks, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            tanda, (x, y), (x + tw + 14, y + th + 12), (b, g, r), cv2.FILLED
        )
        cv2.putText(
            tanda, teks, (x + 7, y + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return tanda


def periksa_pisang(kontur, mask_objek, persentase):
    """
    Memeriksa apakah objek hasil segmentasi layak disebut pisang, dari warna
    dan bentuknya. Mengembalikan (daftar_alasan, ringkasan_ukuran, rasio_aspek);
    daftar alasan kosong berarti objek lolos pemeriksaan.
    """
    luas = cv2.contourArea(kontur)
    (_, _), (w, h), _ = cv2.minAreaRect(kontur)
    aspek = max(w, h) / max(min(w, h), 1)
    solid = luas / max(cv2.contourArea(cv2.convexHull(kontur)), 1)
    extent = luas / max(w * h, 1)
    warna_ok = sum(persentase.values())
    rasio_luas = luas / mask_objek.size

    alasan = []
    if rasio_luas > BATAS_LUAS_MAKS:
        alasan.append("Objek memenuhi hampir seluruh gambar, jadi latar tidak terpisah.")
    if warna_ok < BATAS_WARNA_MIN:
        alasan.append(
            f"Hanya {angka(warna_ok, 1)}% piksel objek yang berwarna khas pisang "
            "(hijau, kuning, cokelat, atau hitam)."
        )
    if aspek < BATAS_ASPEK_MIN:
        alasan.append("Bentuknya terlalu bulat atau kompak, sedangkan pisang memanjang.")
    if extent > BATAS_EXTENT_MAKS:
        alasan.append("Bentuknya terlalu lurus atau persegi, sedangkan pisang melengkung.")
    if solid < BATAS_SOLIDITAS_MIN:
        alasan.append("Bentuknya terlalu tidak beraturan untuk pisang.")

    ringkasan = (
        f"Warna pisang {angka(warna_ok, 1)}% (min {angka(BATAS_WARNA_MIN, 0)}%), "
        f"rasio sisi {angka(aspek)} (min {angka(BATAS_ASPEK_MIN)}), "
        f"extent {angka(extent)} (maks {angka(BATAS_EXTENT_MAKS)}), "
        f"solidity {angka(solid)} (min {angka(BATAS_SOLIDITAS_MIN)}), "
        f"luas {angka(rasio_luas * 100, 0)}% dari citra (maks {angka(BATAS_LUAS_MAKS * 100, 0)}%)."
    )
    return alasan, ringkasan, aspek


def deteksi_pisang(nama_file):
    gambar = cv2.imread(nama_file)

    if gambar is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    # Daftar tahapan proses untuk ditampilkan di GUI.
    # Setiap tahap berisi judul, keterangan, dan citra hasil tahap tersebut.
    langkah = []

    def catat(judul, ket, citra):
        langkah.append({"judul": judul, "ket": ket, "citra": ke_bgr(citra)})

    # Ukuran disesuaikan tanpa mengubah rasio aspek, jadi gambar potret,
    # persegi, maupun lanskap tidak gepeng.
    tinggi0, lebar0 = gambar.shape[:2]
    skala = UKURAN_MAKS / max(tinggi0, lebar0)
    gambar = cv2.resize(
        gambar,
        (max(1, round(lebar0 * skala)), max(1, round(tinggi0 * skala))),
    )
    catat(
        "Citra asli",
        f"Berkas dibaca dengan cv2.imread ({angka(lebar0, 0)} × {angka(tinggi0, 0)} piksel), "
        f"lalu diubah ukurannya menjadi {gambar.shape[1]} × {gambar.shape[0]} piksel "
        f"dengan rasio aspek tetap (sisi terpanjang {UKURAN_MAKS} piksel).",
        gambar,
    )

    # Pengenalan objek dengan MobileNetV2 (bila modelnya tersedia).
    # Keputusan akhirnya diambil di tahap "Pemeriksaan pisang" di bawah.
    pengenalan = kenali_objek(gambar)
    if pengenalan is None:
        catat(
            "Pengenalan objek (dilewati)",
            "Objek tidak dikenali oleh komputer karena terbatas pada model yang digunakan."
            f"({os.path.basename(FILE_MODEL)}), jadi hanya pemeriksaan warna "
            "dan bentuk yang dipakai untuk memastikan objek adalah pisang.",
            gambar,
        )
    else:
        urutan = ", ".join(
            f"{nama_kelas(i)} {angka(p * 100, 1)}%" for i, p in pengenalan["top"]
        )
        catat(
            "Pengenalan objek (MobileNetV2)",
            f"Citra diubah ke 224 × 224 piksel, dinormalisasi, lalu dikenali "
            f"MobileNetV2. {TOP_K} kelas teratas: {urutan}. Peluang kelas "
            f"banana: {angka(pengenalan['peluang_pisang'] * 100, 1)}%.",
            gambar,
        )

    blur = cv2.GaussianBlur(gambar, (5, 5), 0)
    catat(
        "Gaussian blur",
        "Penghalusan dengan kernel 5 × 5 untuk meredam noise kecil sebelum "
        "thresholding. Perubahannya halus, terlihat pada tepi dan tekstur kulit.",
        blur,
    )

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    catat(
        "Konversi ke HSV",
        "Citra diubah ke ruang warna HSV dengan cv2.COLOR_BGR2HSV. Ditampilkan "
        "apa adanya, sehingga warnanya tampak tidak natural: H, S, dan V "
        "dibaca sebagai B, G, dan R.",
        hsv,
    )

    # SEGMENTASI OBJEK PISANG (thresholding pada kanal H, S, V)
    # Mengambil objek pisang dan menghilangkan latar putih:
    #   H <= 90  (corak hijau sampai cokelat)
    #   S >= 35  (latar putih hampir tidak berwarna, jadi tersingkir)
    #   V >= 30  (bukan piksel yang terlalu gelap)
    h, s, v = cv2.split(hsv)
    catat(
        "Kanal H (Hue)",
        "Corak warna, bernilai 0 sampai 179, ditampilkan sebagai tingkat "
        "keabuan. Cokelat, kuning, dan hijau berada pada nilai rendah sampai menengah.",
        (h.astype(np.float32) * 255 / 179).astype(np.uint8),
    )
    catat(
        "Kanal S (Saturation)",
        "Kejenuhan warna, bernilai 0 sampai 255. Kulit pisang terang karena "
        "warnanya pekat, sedangkan latar putih gelap karena hampir tidak berwarna.",
        s,
    )
    catat(
        "Kanal V (Value)",
        "Kecerahan, bernilai 0 sampai 255. Bintik, pangkal, dan bagian pisang "
        "yang menghitam bernilai rendah (gelap).",
        v,
    )

    mask_h = ambang(h, 90, cv2.THRESH_BINARY_INV)
    mask_s = ambang(s, 34, cv2.THRESH_BINARY)
    mask_v = ambang(v, 29, cv2.THRESH_BINARY)
    catat(
        "Threshold kanal H",
        "THRESH_BINARY_INV dengan batas 90: piksel dengan H ≤ 90 (corak hijau "
        "sampai cokelat) menjadi putih. " + info_mask(mask_h),
        mask_h,
    )
    catat(
        "Threshold kanal S",
        "THRESH_BINARY dengan batas 34: piksel dengan S ≥ 35 menjadi putih. "
        "Latar putih yang hampir tidak berwarna tersingkir. " + info_mask(mask_s),
        mask_s,
    )
    catat(
        "Threshold kanal V",
        "THRESH_BINARY dengan batas 29: piksel dengan V ≥ 30 menjadi putih. "
        "Membuang piksel yang terlalu gelap. " + info_mask(mask_v),
        mask_v,
    )

    mask_pisang = cv2.bitwise_and(cv2.bitwise_and(mask_h, mask_s), mask_v)
    catat(
        "Gabungan H, S, dan V",
        "bitwise_and ketiga mask: sebuah piksel harus memenuhi syarat H, S, "
        "dan V sekaligus. " + info_mask(mask_pisang),
        mask_pisang,
    )

    # Piksel gelap hampir tidak berwarna (saturasi rendah), jadi lolos dari
    # batas S di atas. Pisang busuk didominasi piksel seperti ini, maka
    # piksel gelap (V <= AMBANG_GELAP) di dekat area pisang ikut dimasukkan.
    # Namun bayangan yang jatuh di meja juga gelap. Karena itu piksel gelap
    # hanya diambil bila berada di dalam bentuk (silhouette) pisang: bagian
    # pisang yang menghitam tetap terhitung, bayangan di luar tubuh pisang
    # tidak. Bentuk pisang diperkirakan dari area berwarnanya, lalu lubang
    # di dalamnya diisi sehingga bagian yang menghitam tetap termasuk.
    mask_warna_pisang = mask_pisang.copy()
    mask_gelap = ambang(v, AMBANG_GELAP, cv2.THRESH_BINARY_INV)
    catat(
        "Mask piksel gelap",
        f"THRESH_BINARY_INV dengan batas {AMBANG_GELAP}: piksel dengan "
        f"V ≤ {AMBANG_GELAP} dianggap gelap. Piksel ini hampir tidak berwarna "
        "sehingga lolos dari batas S, padahal bisa menjadi bagian pisang yang "
        "menghitam. " + info_mask(mask_gelap),
        mask_gelap,
    )

    bentuk_pisang = bentuk_objek(mask_warna_pisang)
    catat(
        "Bentuk pisang",
        "Perkiraan bentuk utuh pisang: area berwarna ditutup dengan kernel "
        "9 × 9 lalu lubang di dalamnya diisi (flood fill dari luar gambar). "
        "Bagian kulit yang menghitam ikut masuk ke bentuk, tetapi bayangan "
        "yang berada di luar tubuh pisang tetap di luar. "
        + info_mask(bentuk_pisang),
        bentuk_pisang,
    )

    kernel_gelap = np.ones((25, 25), np.uint8)
    area_dekat_pisang = cv2.dilate(mask_warna_pisang, kernel_gelap)
    mask_gelap_relevan = cv2.bitwise_and(mask_gelap, bentuk_pisang)
    catat(
        "Piksel gelap di dalam bentuk pisang",
        "bitwise_and antara mask piksel gelap dan bentuk pisang: hanya piksel "
        "gelap yang berada di dalam tubuh pisang yang dipakai, sehingga bagian "
        "pisang yang menghitam tetap terhitung. " + info_mask(mask_gelap_relevan),
        mask_gelap_relevan,
    )

    # Ditampilkan agar terlihat bahwa bayangan di luar pisang memang dibuang.
    mask_bayangan = cv2.bitwise_and(
        cv2.bitwise_and(mask_gelap, area_dekat_pisang),
        cv2.bitwise_not(bentuk_pisang),
    )
    catat(
        "Bayangan diabaikan",
        "Piksel gelap yang berada di dekat pisang (hasil dilasi 25 × 25) "
        "tetapi di luar bentuk pisang, misalnya bayangan yang jatuh di meja, "
        "tidak ikut dihitung. " + info_mask(mask_bayangan),
        mask_bayangan,
    )

    mask_pisang = cv2.bitwise_or(mask_warna_pisang, mask_gelap_relevan)
    catat(
        "Gabungan dengan piksel gelap",
        "bitwise_or antara mask warna pisang dan piksel gelap di dalam bentuk "
        "pisang. " + info_mask(mask_pisang),
        mask_pisang,
    )

    # Menghilangkan noise kecil, lalu menutup celah kecil pada objek
    kernel_kecil = np.ones((3, 3), np.uint8)
    kernel_sedang = np.ones((7, 7), np.uint8)
    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_OPEN, kernel_kecil)
    catat(
        "Morfologi opening",
        "Opening dengan kernel 3 × 3 (erosi lalu dilasi) menghapus noise "
        "berupa titik putih kecil. " + info_mask(mask_pisang),
        mask_pisang,
    )
    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_CLOSE, kernel_sedang)
    catat(
        "Morfologi closing",
        "Closing dengan kernel 7 × 7 (dilasi lalu erosi) menutup celah dan "
        "lubang kecil di dalam objek. " + info_mask(mask_pisang),
        mask_pisang,
    )

    # Mencari kontur objek
    kontur, _ = cv2.findContours(
        mask_pisang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not kontur:
        raise BukanPisang(
            "Tidak ditemukan area berwarna pisang pada gambar.", gambar, langkah
        )

    # Mengambil kontur terbesar
    kontur_terbesar = max(kontur, key=cv2.contourArea)

    tinggi, lebar = mask_pisang.shape
    if cv2.contourArea(kontur_terbesar) < tinggi * lebar * 0.05:
        raise BukanPisang(
            "Objek berwarna pisang terlalu kecil atau tidak terdeteksi.",
            gambar, langkah,
        )

    # Semua kontur yang cukup besar dianggap satu pisang. Kontur kecil
    # (bintik, noise) dibuang, dan kontur tambahan harus punya warna pisang
    # yang wajar agar latar berwarna serupa tidak ikut terhitung.
    luas_min = tinggi * lebar * BATAS_LUAS_OBJEK / 100.0
    kandidat = []
    for k in kontur:
        if cv2.contourArea(k) < luas_min:
            continue
        if k is not kontur_terbesar:
            persen_k, _ = persentase_warna(
                hsv, mask_dari_kontur(k, mask_pisang.shape)
            )
            if sum(persen_k.values()) < 60.0:
                continue
        kandidat.append(k)

    # Urutan baca: baris atas dulu, lalu kiri ke kanan di baris yang sama.
    baris_gambar = max(gambar.shape[0] * 0.30, 1.0)

    def urutan_baca(k):
        x, y, w, h = cv2.boundingRect(k)
        return int(y / baris_gambar), x

    kandidat.sort(key=urutan_baca)

    gambar_kontur = gambar.copy()
    cv2.drawContours(gambar_kontur, kontur, -1, (0, 165, 255), 1)
    cv2.drawContours(gambar_kontur, kandidat, -1, (0, 255, 0), 2)
    catat(
        "Pencarian kontur",
        f"cv2.findContours menemukan {len(kontur)} kontur. "
        f"{len(kandidat)} kontur berukuran cukup besar dipakai sebagai objek "
        f"pisang (garis hijau); kontur kecil (oranye) diabaikan.",
        gambar_kontur,
    )

    # Mask gabungan semua pisang yang terdeteksi
    mask_objek = np.zeros_like(mask_pisang)
    for k in kandidat:
        cv2.drawContours(
            mask_objek, [k], -1, 255, thickness=cv2.FILLED
        )

    # Menghitung total piksel objek
    total_piksel = cv2.countNonZero(mask_objek)

    if total_piksel == 0:
        raise ValueError("Area objek pisang kosong.")

    catat(
        "Mask objek pisang",
        f"{len(kandidat)} kontur pisang diisi penuh (cv2.FILLED) menjadi mask "
        f"objek. Total piksel objek: {angka(total_piksel, 0)} "
        f"({angka(total_piksel / mask_objek.size * 100, 1)}% dari citra).",
        mask_objek,
    )

    # Hasil segmentasi
    hasil_segmentasi = cv2.bitwise_and(gambar, gambar, mask=mask_objek)
    catat(
        "Hasil segmentasi",
        "bitwise_and citra dengan mask objek: hanya piksel pisang yang "
        "dipertahankan, latar menjadi hitam.",
        hasil_segmentasi,
    )

    # Persentase warna dihitung pada gabungan seluruh objek pisang. Hasil ini
    # dipakai untuk ringkasan keseluruhan; tiap pisang juga dihitung sendiri.
    persentase, peta_warna = persentase_warna(hsv, mask_objek)

    for warna, daftar in RENTANG_WARNA.items():
        jumlah = cv2.countNonZero(peta_warna[warna])
        rentang_teks = " atau ".join(
            f"H {bawah[0]}-{atas[0]}, S {bawah[1]}-{atas[1]}, "
            f"V {bawah[2]}-{atas[2]}"
            for bawah, atas in daftar
        )
        catat(
            f"Warna {warna}",
            f"cv2.inRange dengan rentang {rentang_teks}, dibatasi pada objek. "
            f"Piksel {warna.lower()}: {angka(jumlah, 0)} "
            f"({angka(persentase[warna], 1)}% dari objek).",
            cv2.bitwise_and(gambar, gambar, mask=peta_warna[warna]),
        )

    # Pemeriksaan: apakah objek ini benar-benar pisang?
    alasan, ringkasan, aspek = periksa_pisang(
        kontur_terbesar, mask_objek, persentase
    )
    pisang, alasan_tolak = putuskan_pisang(
        pengenalan, alasan, persentase["Hitam"], aspek
    )
    ringkasan_periksa = ("Lolos. " if pisang else "Ditolak. ") + ringkasan
    if (
        pisang
        and pengenalan is not None
        and pengenalan["peluang_pisang"] < AMBANG_PELUANG
        and persentase["Hitam"] >= BATAS_GELAP_PISANG
    ):
        ringkasan_periksa += (
            " Model kurang yakin karena pisang berwarna hitam, tetapi hitam "
            "mendominasi dan bentuknya memanjang, jadi objek tetap diterima "
            "sebagai pisang busuk."
        )
    catat("Pemeriksaan pisang", ringkasan_periksa, gambar_kontur)
    if not pisang:
        raise BukanPisang(alasan_tolak, gambar, langkah)

    # Warna dan kematangan dihitung untuk setiap pisang satu per satu.
    daftar_pisang = []
    for nomor, kontur_objek in enumerate(kandidat, 1):
        mask_satu = mask_dari_kontur(kontur_objek, mask_pisang.shape)
        persen_satu, _ = persentase_warna(hsv, mask_satu)
        dominan_satu = max(persen_satu, key=persen_satu.get)
        daftar_pisang.append({
            "nomor": nomor,
            "bbox": cv2.boundingRect(kontur_objek),
            "persentase": persen_satu,
            "dominan": dominan_satu,
            "kematangan": KLASIFIKASI_WARNA[dominan_satu],
            "luas": cv2.contourArea(kontur_objek),
        })

    gambar_tanda = tandai_pisang(gambar, daftar_pisang)
    rincian = [
        f"Pisang {objek['nomor']}: {objek['kematangan']} "
        f"(dominan {objek['dominan'].lower()}, "
        f"{angka(objek['persentase'][objek['dominan']], 1)}% dari pisang ini, "
        f"luas {angka(objek['luas'] / mask_objek.size * 100, 1)}% citra)."
        for objek in daftar_pisang
    ]
    catat(
        "Rincian tiap pisang",
        f"{len(daftar_pisang)} pisang terdeteksi. Setiap kontur dihitung "
        "warnanya sendiri, lalu dinilai kematangannya. " + " ".join(rincian),
        gambar_tanda,
    )

    # Menentukan warna dominan gabungan dan kematangan keseluruhan.
    warna_dominan = max(persentase, key=persentase.get)

    if len(daftar_pisang) == 1:
        kematangan = daftar_pisang[0]["kematangan"]
        catat(
            "Klasifikasi kematangan",
            f"Warna dominan adalah {warna_dominan.lower()} "
            f"({angka(persentase[warna_dominan], 1)}% dari objek), maka pisang "
            f"tergolong {kematangan.lower()}. Aturan: hijau = belum matang, "
            "kuning = matang, cokelat = terlalu matang, hitam = busuk.",
            hasil_segmentasi,
        )
    else:
        jumlah_kelas = {}
        luas_kelas = {}
        for objek in daftar_pisang:
            kelas = objek["kematangan"]
            jumlah_kelas[kelas] = jumlah_kelas.get(kelas, 0) + 1
            luas_kelas[kelas] = luas_kelas.get(kelas, 0.0) + objek["luas"]
        kematangan = sorted(
            jumlah_kelas, key=lambda k: (-jumlah_kelas[k], -luas_kelas[k])
        )[0]
        sebaran = ", ".join(
            f"{kelas.lower()} {jumlah_kelas[kelas]} pisang"
            for kelas in sorted(
                jumlah_kelas,
                key=lambda k: (-jumlah_kelas[k], -luas_kelas[k]),
            )
        )
        catat(
            "Klasifikasi kematangan",
            f"{len(daftar_pisang)} pisang terdeteksi: {sebaran}. Kematangan "
            f"keseluruhan diambil dari kelas yang paling banyak muncul "
            f"(bila jumlahnya sama, luas area menjadi penentu), yaitu "
            f"{kematangan.lower()}. Aturan: hijau = belum matang, kuning = "
            "matang, cokelat = terlalu matang, hitam = busuk.",
            gambar_tanda,
        )

    return (
        gambar, hasil_segmentasi, persentase, warna_dominan, kematangan,
        mask_objek > 0, langkah, daftar_pisang, gambar_tanda,
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
        kv("Piksel V <= %d" % AMBANG_GELAP, f"{angka(v[:AMBANG_GELAP + 1].sum() / n * 100)}%  (hitam pekat)"),
        kv(
            "Piksel V <= %d" % BATAS_HITAM_V,
            f"{angka(v[:BATAS_HITAM_V + 1].sum() / n * 100)}%  "
            f"(termasuk hitam pudar S <= {BATAS_HITAM_S})",
        ),
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

INK, MUTED, GARIS = "#1F2529", "#6E767B", "#E4E8E9"
PANEL, STAGE, STAGE_TEKS = "#FFFFFF", "#9CCDDB", "#2F5E6C"
LATAR = "#CBC7CE"
BIRU_MUDA = "#EAF6F8"
BIRU_TUA = "#6FAEC2"

WARNA_DATA = {
    "Hijau": "#7FA83A",
    "Kuning": "#EFC31A",
    "Cokelat": "#7B4A24",
    "Hitam": "#2A2522",
}
WARNA_LAINNYA = "#DDE4E5"

PESAN_AWAL = "Mendukung berkas JPG, PNG, dan BMP."
TEKS_KOSONG = "Buka foto pisang untuk melihat tingkat kematangannya."

STYLE = Template("""
QMainWindow { background: $LATAR; }
QDialog { background: $PANEL; }
QWidget#pusat { background: transparent; }

QFrame#kartuBiru { background: $STAGE; border-radius: 26px; }
QFrame#kartuPutih { background: $PANEL; border-radius: 26px; }
QFrame#kartuGambar { background: $BIRU_MUDA; border-radius: 18px; }
QFrame#kepalaDialog { background: $BIRU_MUDA; border-radius: 16px; }
QFrame#kotakPilihan { background: #F7F9F9; border-radius: 16px; }

QLabel#sapaan { color: #4A5257; font-size: 11px; font-weight: 700; }
QLabel#judulAplikasi { color: $INK; font-size: 26px; font-weight: 700; }
QLabel#subjudul { color: #4F575C; font-size: 12px; }
QLabel#chip {
    background: $PANEL; color: $INK; font-size: 11px; font-weight: 700;
    border-radius: 14px; padding: 7px 16px;
}
QLabel#chip[status="ok"] { background: #E1F0E4; color: #2B683D; }
QLabel#chip[status="tolak"] { background: #F7DEDA; color: #8C3A2E; }
QLabel#chipBiru {
    background: rgba(255, 255, 255, 72%); color: #214C59;
    font-size: 10px; font-weight: 700; border-radius: 12px; padding: 5px 12px;
}
QLabel#chipPisang {
    background: #F0F4F4; color: #3C464B; font-size: 11px; font-weight: 600;
    border-radius: 10px; padding: 5px 9px;
}
QLabel#chipPisang[kematangan="Belum Matang"] { background: #E8F0D8; color: #4C6A1C; }
QLabel#chipPisang[kematangan="Matang"] { background: #FBF2CC; color: #7A6413; }
QLabel#chipPisang[kematangan="Terlalu Matang"] { background: #F3E6DC; color: #7B4A24; }
QLabel#chipPisang[kematangan="Busuk"] { background: #E7E6E5; color: #2A2522; }
QLabel#petunjukBiru { color: $STAGE_TEKS; font-size: 11px; }
QLabel#eyebrow { color: #7A8388; font-size: 10px; font-weight: 700; }

QLabel#caption { color: $STAGE_TEKS; font-size: 12px; font-weight: 700; }

QLabel#judulLangkah { color: $INK; font-size: 20px; font-weight: 700; }
QLabel#ketLangkah { color: $STAGE_TEKS; font-size: 13px; }
QListWidget#daftar {
    background: rgba(255, 255, 255, 68%); border: none; border-radius: 16px;
    color: #2C5A67; font-size: 12px; padding: 6px; outline: 0;
}
QListWidget#daftar::item { padding: 7px 8px; border-radius: 10px; }
QListWidget#daftar::item:hover { background: rgba(255, 255, 255, 45%); }
QListWidget#daftar::item:selected { background: #FFFFFF; color: $INK; }
QListWidget#daftar QScrollBar:vertical { background: transparent; width: 8px; }
QListWidget#daftar QScrollBar::handle:vertical {
    background: #7FB6C6; border-radius: 4px; min-height: 24px;
}
QListWidget#daftar QScrollBar::add-line:vertical,
QListWidget#daftar QScrollBar::sub-line:vertical { height: 0; }

QLabel#verdict { color: $INK; font-size: 34px; font-weight: 700; }
QLabel#keterangan { color: $MUTED; font-size: 13px; }
QLabel#namaWarna { color: #4A5257; font-size: 13px; }
QLabel#nilaiWarna { color: $INK; font-size: 13px; font-weight: 600; }
QLabel[tebal="true"] { color: $INK; font-weight: 700; }

QLabel#judulSimpan { color: $INK; font-size: 20px; font-weight: 700; }
QCheckBox { color: $INK; font-size: 13px; spacing: 8px; }
QLineEdit {
    background: #F7F9F9; color: $INK; border: 1px solid $GARIS;
    border-radius: 10px; padding: 8px 11px; font-size: 13px;
    selection-background-color: #CFE7EC; selection-color: $INK;
}
QLineEdit:focus { border-color: $BIRU_TUA; background: #FFFFFF; }
QCheckBox::indicator:focus { outline: none; }

QPushButton#utama, QPushButton#biru, QPushButton#sekunder,
QPushButton#halus {
    border-radius: 20px; padding: 10px 16px; min-height: 20px;
    font-size: 13px; font-weight: 600;
}
QPushButton#utama { background: $INK; color: white; border: 2px solid $INK; }
QPushButton#utama:hover { background: #343C41; border-color: #343C41; }
QPushButton#utama:pressed { background: #0F1214; border-color: #0F1214; }
QPushButton#utama:disabled {
    background: #B9BFC1; border-color: #B9BFC1; color: #F1F3F3;
}
QPushButton#biru { background: $STAGE; color: #173B45; border: 2px solid $STAGE; }
QPushButton#biru:hover { background: #8AC3D4; border-color: #8AC3D4; }
QPushButton#biru:pressed { background: #79B5C7; border-color: #79B5C7; }
QPushButton#biru:disabled {
    background: #DCEBEE; border-color: #DCEBEE; color: #9CB2B7;
}
QPushButton#sekunder {
    background: transparent; color: $INK; border: 2px solid #D8DEDF;
}
QPushButton#sekunder:hover { background: #F0F4F4; }
QPushButton#sekunder:pressed { background: #E6ECEC; }
QPushButton#sekunder:disabled { color: #A9B1B4; border-color: #ECEFEF; }
QPushButton#halus {
    background: transparent; color: $MUTED; border: none;
    padding: 6px 10px; font-weight: 500;
}
QPushButton#halus:hover { color: $INK; }
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
    background: #F7F9F9; color: $INK; border: 1px solid $GARIS;
    border-radius: 12px; padding: 10px;
    selection-background-color: #CFE7EC; selection-color: $INK;
}

QStatusBar {
    background: transparent; color: #4F575C; font-size: 12px;
    border: none;
}
QStatusBar::item { border: none; }

QToolTip {
    background: $INK; color: #FFFFFF; border: none;
    padding: 6px 9px; font-size: 11px;
}
""").substitute(
    INK=INK, MUTED=MUTED, GARIS=GARIS, PANEL=PANEL, LATAR=LATAR,
    STAGE=STAGE, STAGE_TEKS=STAGE_TEKS, BIRU_MUDA=BIRU_MUDA,
    BIRU_TUA=BIRU_TUA,
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
# FITUR SIMPAN HASIL
# ==========================================================

def folder_awal_simpan():
    """Folder awal untuk dialog simpan: Pictures bila ada, jika tidak home."""
    gambar = os.path.join(os.path.expanduser("~"), "Pictures")
    return gambar if os.path.isdir(gambar) else os.path.expanduser("~")


def nama_aman(teks):
    """Bersihkan teks agar aman dipakai sebagai bagian nama berkas."""
    aman = "".join(
        "_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in str(teks)
    )
    aman = aman.strip().rstrip(".").strip()
    return aman or "hasil"


def hex_ke_bgr(kode):
    kode = kode.lstrip("#")
    r, g, b = (int(kode[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def simpan_citra(nama, bgr):
    """
    Menyimpan citra BGR dengan cv2.imencode lalu menulis byte-nya sendiri.
    Cara ini aman untuk folder atau nama berkas berhuruf non-ASCII.
    """
    ext = os.path.splitext(nama)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
        ext = ".png"
    berhasil, buf = cv2.imencode(ext, bgr)
    if not berhasil:
        raise ValueError(f"Citra tidak bisa diubah ke format {ext}.")
    with open(nama, "wb") as f:
        f.write(buf.tobytes())


def laporan_hasil(data):
    """Menyusun laporan hasil analisis sebagai teks siap disimpan."""
    gambar = data["gambar"]
    persentase = data["persentase"]
    dominan = data["dominan"]
    daftar = data.get("daftar_pisang") or []

    baris = [
        "LAPORAN HASIL DETEKSI KEMATANGAN PISANG",
        "=" * 40,
        kv("Berkas", data["nama_berkas"]),
        kv("Waktu analisis", datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
        kv("Dimensi citra", f"{gambar.shape[1]} x {gambar.shape[0]} piksel"),
        kv("Pisang terdeteksi", f"{len(daftar)}"),
        kv("Klasifikasi", data["kematangan"]),
        kv(
            "Warna dominan",
            f"{dominan} ({angka(persentase[dominan], 1)}% dari area pisang)",
        ),
        "",
        "KOMPOSISI WARNA PADA OBJEK PISANG",
        "-" * 33,
    ]
    for warna, nilai in persentase.items():
        tanda = "  <- dominan" if warna == dominan else ""
        baris.append(kv(warna, f"{angka(nilai, 1)}%") + tanda)
    sisa = max(0.0, 100.0 - sum(persentase.values()))
    baris.append(kv("Lainnya", f"{angka(sisa, 1)}%"))

    if daftar:
        baris += ["", "", "RINCIAN TIAP PISANG", "-" * 19]
        for objek in daftar:
            x, y, w, h = objek["bbox"]
            persen = objek["persentase"]
            komposisi = ", ".join(
                f"{warna} {angka(nilai, 1)}%"
                for warna, nilai in persen.items() if nilai >= 1.0
            )
            baris += [
                "",
                f"Pisang {objek['nomor']}  ({objek['kematangan']})",
                kv("  Warna dominan", f"{objek['dominan']} "
                   f"({angka(persen[objek['dominan']], 1)}%)"),
                kv("  Komposisi", komposisi or "-"),
                kv("  Posisi kotak", f"x {x}, y {y}, {w} x {h} piksel"),
                kv("  Luas objek", f"{angka(objek['luas'], 0)} piksel"),
            ]

    langkah = data.get("langkah") or []
    if langkah:
        baris += ["", "", "TAHAPAN PROSES", "-" * 14]
        for i, tahap in enumerate(langkah, 1):
            baris += ["", f"{i}. {tahap['judul']}"]
            baris += [f"   {k}" for k in str(tahap["ket"]).splitlines()]

    info = data.get("info")
    if info:
        sebelum, sesudah = info[0], info[1]
        if sebelum:
            baris += ["", "", sebelum]
        if sesudah:
            baris += ["", "", sesudah]

    return "\n".join(baris) + "\n"


def buat_gambar_ringkasan(data):
    """
    Kartu hasil dalam satu citra: asli dan segmentasi berdampingan,
    dilengkapi klasifikasi dan jalur komposisi warna. Dikembalikan BGR.
    """
    gambar, segmentasi = data["gambar"], data["segmentasi"]
    persentase, dominan = data["persentase"], data["dominan"]
    daftar = data.get("daftar_pisang") or []
    kiri = gambar
    if len(daftar) > 1:
        tanda = data.get("gambar_tanda")
        if tanda is not None and tanda.shape == gambar.shape:
            kiri = tanda

    margin, jarak, kepala, kaki = 22, 18, 64, 118
    w1, h1 = kiri.shape[1], kiri.shape[0]
    w2, h2 = segmentasi.shape[1], segmentasi.shape[0]
    h_gambar = max(h1, h2)

    lebar = margin * 2 + w1 + jarak + w2
    tinggi = margin + kepala + h_gambar + 26 + kaki
    kartu = np.full((tinggi, lebar, 3), 244, np.uint8)

    def tulis(teks, x, y, skala=0.5, tebal=1, warna=(28, 33, 36)):
        cv2.putText(
            kartu, teks, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
            skala, warna, tebal, cv2.LINE_AA,
        )

    def tengah(teks, x, y, skala=0.5, tebal=1, warna=(28, 33, 36)):
        (lebar_teks, _), _ = cv2.getTextSize(
            teks, cv2.FONT_HERSHEY_SIMPLEX, skala, tebal
        )
        tulis(teks, x - lebar_teks / 2, y, skala, tebal, warna)

    redup = (95, 106, 113)
    tulis("Deteksi Kematangan Pisang", margin, margin + 20, 0.72, 2)
    tulis(f"Berkas: {data['nama_berkas']}", margin, margin + 44, 0.46, 1, redup)

    y_gambar = margin + kepala
    kartu[y_gambar:y_gambar + h1, margin:margin + w1] = kiri
    x2 = margin + w1 + jarak
    kartu[y_gambar:y_gambar + h2, x2:x2 + w2] = segmentasi
    y_caption = y_gambar + h_gambar + 20
    tengah(
        "Citra asli (bernomor)" if len(daftar) > 1 else "Citra asli",
        margin + w1 / 2, y_caption, 0.45, 1, redup,
    )
    tengah("Hasil segmentasi", x2 + w2 / 2, y_caption, 0.45, 1, redup)

    y_kaki = y_gambar + h_gambar + 26 + 20
    if len(daftar) > 1:
        tulis(f"{len(daftar)} PISANG TERDETEKSI", margin, y_kaki + 10, 0.78, 2)
        tulis(
            f"{ringkas_kematangan(daftar)} - kematangan keseluruhan "
            f"{data['kematangan'].lower()}.",
            margin + 2, y_kaki + 36, 0.45, 1, redup,
        )
    else:
        tulis(data["kematangan"].upper(), margin, y_kaki + 10, 0.78, 2)
        tulis(
            f"Warna dominan {dominan.lower()}, "
            f"{angka(persentase[dominan], 1)}% dari area pisang.",
            margin + 2, y_kaki + 36, 0.45, 1, redup,
        )

    # Jalur komposisi warna
    urutan = list(persentase.items())
    sisa = max(0.0, 100.0 - sum(persentase.values()))
    urutan.append(("Lainnya", sisa))

    y_bar = y_kaki + 52
    x = margin
    ujung = margin + (lebar - margin * 2)
    for warna, nilai in urutan:
        lebar_warna = int(round((lebar - margin * 2) * max(0.0, nilai) / 100.0))
        if lebar_warna <= 0:
            continue
        cv2.rectangle(
            kartu, (x, y_bar), (x + lebar_warna, y_bar + 12),
            hex_ke_bgr(WARNA_DATA.get(warna, WARNA_LAINNYA)), cv2.FILLED,
        )
        x += lebar_warna
    if x < ujung:
        cv2.rectangle(
            kartu, (x, y_bar), (ujung, y_bar + 12),
            hex_ke_bgr(WARNA_LAINNYA), cv2.FILLED,
        )

    x = margin
    for warna, nilai in urutan:
        teks = f"{warna} {angka(nilai, 1)}%"
        (lebar_teks, _), _ = cv2.getTextSize(
            teks, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        if x + lebar_teks > lebar - margin:
            break
        tulis(teks, x, y_bar + 30, 0.4, 1)
        x += lebar_teks + 20
    return kartu


# ==========================================================
# 4. KOMPONEN
# ==========================================================

class AreaGambar(QWidget):
    """
    Pratinjau gambar dengan sudut membulat. Saat kursor di atasnya muncul
    ajakan untuk mengganti gambar, dan klik membuka dialog pemilihan berkas.
    """

    diklik = pyqtSignal()

    def __init__(self, teks_kosong, bisa_diklik=True):
        super().__init__()
        self._teks_kosong = teks_kosong
        self._pixmap = None
        self._hover = False
        self._sorot = False
        self._bisa_diklik = bisa_diklik
        self.setMinimumSize(220, 160)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCursor(
            Qt.PointingHandCursor if bisa_diklik else Qt.ArrowCursor
        )
        self.setMouseTracking(True)
        self.hapus()

    def set_gambar(self, pixmap):
        self._pixmap = pixmap
        if self._bisa_diklik:
            self.setToolTip("Klik untuk mengganti gambar")
        self.update()

    def hapus(self):
        self._pixmap = None
        if self._bisa_diklik:
            self.setToolTip("Klik untuk memilih gambar")
        self.update()

    def set_sorot(self, aktif):
        """Sorotan saat berkas gambar diseret ke jendela."""
        if self._sorot != aktif:
            self._sorot = aktif
            self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._bisa_diklik:
            self.diklik.emit()
        super().mouseReleaseEvent(event)

    def _pil(self, p, teks, y):
        huruf = p.font()
        huruf.setPointSize(9)
        huruf.setBold(True)
        p.setFont(huruf)
        lebar = p.fontMetrics().horizontalAdvance(teks) + 28
        pil = QRectF((self.width() - lebar) / 2, y, lebar, 28)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 236))
        p.drawRoundedRect(pil, 14, 14)
        p.setPen(QColor("#173B45"))
        p.drawText(pil, Qt.AlignCenter, teks)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        bentuk = QPainterPath()
        bentuk.addRoundedRect(r, 16, 16)
        p.fillPath(bentuk, QColor("#FFFFFF"))

        if self._pixmap is not None:
            p.save()
            p.setClipPath(bentuk)
            skala = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            p.drawPixmap(
                int((self.width() - skala.width()) / 2),
                int((self.height() - skala.height()) / 2),
                skala,
            )
            p.restore()
            if self._hover and self._bisa_diklik:
                p.fillPath(bentuk, QColor(23, 59, 69, 45))
                self._pil(p, "Klik untuk mengganti gambar", self.height() - 44)
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#8FC3D3"), 1.4, Qt.DashLine))
            p.drawRoundedRect(r.adjusted(7, 7, -7, -7), 12, 12)

            huruf = p.font()
            huruf.setPointSize(10)
            p.setFont(huruf)
            p.setPen(QColor("#3E6B7A"))
            p.drawText(
                r.adjusted(26, 16, -26, -44),
                Qt.AlignCenter | Qt.TextWordWrap,
                self._teks_kosong,
            )
            if self._hover and self._bisa_diklik:
                self._pil(p, "Klik untuk memilih gambar", self.height() - 44)

        if self._sorot:
            p.setBrush(QColor(23, 59, 69, 30))
            p.setPen(QPen(QColor("#173B45"), 2.2))
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 16, 16)
            self._pil(p, "Lepaskan di sini", self.height() - 44)


class PanelTahapan(QWidget):
    """
    Menampilkan setiap tahap proses deteksi. Daftar tahap (dengan gambar
    kecil) ada di kiri, tahap yang dipilih tampil besar di kanan beserta
    penjelasannya. Pindah tahap dengan klik atau tombol panah atas/bawah.
    """

    def __init__(self):
        super().__init__()
        self._pixmaps, self._langkah = [], []

        tata = QHBoxLayout(self)
        tata.setContentsMargins(0, 0, 0, 0)
        tata.setSpacing(24)

        self.daftar = QListWidget()
        self.daftar.setObjectName("daftar")
        self.daftar.setFixedWidth(220)
        self.daftar.setIconSize(QSize(76, 52))
        self.daftar.setSpacing(2)
        self.daftar.setWordWrap(True)
        self.daftar.setTextElideMode(Qt.ElideRight)
        self.daftar.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.daftar.currentRowChanged.connect(self._tampil)
        self.daftar.setToolTip("Pilih tahap untuk melihat citra dan penjelasannya")
        tata.addWidget(self.daftar)

        kanan = QVBoxLayout()
        kanan.setSpacing(10)
        self.judul = label("", "judulLangkah", wrap=True)
        self.area = AreaGambar("Belum ada tahapan proses.", bisa_diklik=False)
        self.ket = label("", "ketLangkah", wrap=True)
        self.nomor = label("", "petunjukBiru", rata=Qt.AlignCenter)
        kanan.addWidget(self.judul)
        kanan.addWidget(self.area, 1)
        kanan.addWidget(self.ket)
        kanan.addWidget(self.nomor)
        tata.addLayout(kanan, 1)

    def set_langkah(self, langkah):
        self.daftar.blockSignals(True)
        self.daftar.clear()
        self._langkah = langkah
        self._pixmaps = []
        for i, l in enumerate(langkah):
            pix = numpy_ke_pixmap(l["citra"])
            self._pixmaps.append(pix)
            ikon = QIcon(pix.scaled(
                84, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.daftar.addItem(QListWidgetItem(ikon, f"{i + 1}. {l['judul']}"))
        self.daftar.blockSignals(False)
        self.daftar.setCurrentRow(0)
        self._tampil(0)

    def kosongkan(self):
        self.daftar.blockSignals(True)
        self.daftar.clear()
        self.daftar.blockSignals(False)
        self._langkah, self._pixmaps = [], []
        self.area.hapus()
        self.judul.setText("")
        self.ket.setText("")
        self.nomor.setText("")

    def _tampil(self, baris):
        if not 0 <= baris < len(self._langkah):
            return
        l = self._langkah[baris]
        self.judul.setText(f"{baris + 1}. {l['judul']}")
        self.ket.setText(l["ket"])
        self.area.set_gambar(self._pixmaps[baris])
        self.nomor.setText(
            f"Tahap {baris + 1} dari {len(self._langkah)}  ·  "
            "pakai tombol panah atas/bawah untuk berpindah"
        )


class JalurWarna(QWidget):
    """
    Satu jalur horizontal yang diisi seperti kulit pisang yang menua:
    hijau, kuning, cokelat, hitam. Lebar tiap bagian sama dengan
    persentasenya; sisa piksel yang tidak masuk rentang berwarna abu-abu.
    """

    def __init__(self):
        super().__init__()
        self.setFixedHeight(16)
        self.setToolTip("Komposisi warna pada area pisang")
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
            rincian = "\n".join(
                f"{warna}  {angka(nilai, 1)}%"
                for warna, nilai in self._persen.items()
            )
            self.setToolTip(f"Komposisi warna:\n{rincian}")
            self._anim.start()
        else:
            self.setToolTip("Komposisi warna pada area pisang")
            self._t = 0.0
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        bentuk = QPainterPath()
        bentuk.addRoundedRect(r, 8, 8)
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
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background: {kode}; border-radius: 5px;")
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
            pita=[(0, AMBANG_GELAP, WARNA_DATA["Hitam"]),
                  (AMBANG_GELAP + 1, BATAS_HITAM_V, "#6E6A66")],
            garis=[(30, "V = 30"), (AMBANG_GELAP, "V = 55"),
                   (BATAS_HITAM_V, "V = 90")],
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


class DialogSimpan(QDialog):
    """
    Fitur simpan hasil: pilih bagian yang disimpan (laporan teks, kartu
    ringkasan, citra, tahapan proses) beserta folder dan nama berkasnya.
    """

    def __init__(self, parent, data):
        super().__init__(parent)
        self.setWindowTitle("Simpan hasil analisis")
        self.setMinimumWidth(500)
        self._data = data
        self._folder = folder_awal_simpan()
        self.berkas_tersimpan = []

        tata = QVBoxLayout(self)
        tata.setContentsMargins(24, 20, 24, 20)
        tata.setSpacing(14)

        kepala = QFrame()
        kepala.setObjectName("kepalaDialog")
        isi_kepala = QVBoxLayout(kepala)
        isi_kepala.setContentsMargins(16, 14, 16, 14)
        isi_kepala.setSpacing(3)
        isi_kepala.addWidget(label("Simpan hasil analisis", "judulSimpan"))
        isi_kepala.addWidget(label(
            f"Hasil untuk {data['nama_berkas']} akan disimpan sebagai berkas "
            "baru. Pilih bagian yang ingin disimpan.", "keterangan", wrap=True,
        ))
        tata.addWidget(kepala)

        kisi = QGridLayout()
        kisi.setHorizontalSpacing(10)
        kisi.setVerticalSpacing(10)
        kisi.setColumnStretch(1, 1)
        kisi.addWidget(label("Nama berkas", "namaWarna"), 0, 0)
        self.kolom_nama = QLineEdit(
            os.path.splitext(data["nama_berkas"])[0]
        )
        kisi.addWidget(self.kolom_nama, 0, 1, 1, 2)
        kisi.addWidget(label("Folder", "namaWarna"), 1, 0)
        self.kolom_folder = QLineEdit(self._folder)
        self.kolom_folder.setReadOnly(True)
        kisi.addWidget(self.kolom_folder, 1, 1)
        kisi.addWidget(tombol("Pilih folder", "sekunder", self.pilih_folder), 1, 2)
        tata.addLayout(kisi)

        kotak_pilihan = QFrame()
        kotak_pilihan.setObjectName("kotakPilihan")
        isi_pilihan = QVBoxLayout(kotak_pilihan)
        isi_pilihan.setContentsMargins(14, 12, 14, 12)
        isi_pilihan.setSpacing(8)
        isi_pilihan.addWidget(label("YANG DISIMPAN", "eyebrow"))

        self.kotak = {}
        pilihan_awal = [
            ("laporan", "Laporan analisis (.txt)", True),
            ("ringkasan", "Kartu ringkasan hasil (.png)", True),
            ("segmentasi", "Gambar hasil segmentasi (.png)", True),
            ("asli", "Gambar asli (.png)", False),
        ]
        if len(data.get("daftar_pisang") or []) > 1:
            pilihan_awal.append(
                ("berlabel", "Gambar berlabel: kotak & nomor tiap pisang (.png)",
                 True)
            )
        pilihan_awal.append((
            "tahapan",
            f"Seluruh tahapan proses ({len(data['langkah'])} gambar .png)",
            False,
        ))
        for kunci, teks, aktif in pilihan_awal:
            kotak = QCheckBox(teks)
            kotak.setChecked(aktif)
            self.kotak[kunci] = kotak
            isi_pilihan.addWidget(kotak)
        tata.addWidget(kotak_pilihan)

        self.label_info = label("", "keterangan", wrap=True)
        tata.addWidget(self.label_info)

        baris = QHBoxLayout()
        baris.addStretch(1)
        baris.addWidget(tombol("Batal", "sekunder", self.reject))
        baris.addSpacing(8)
        baris.addWidget(tombol("Simpan", "utama", self.simpan))
        tata.addLayout(baris)

    def pilih_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Pilih folder penyimpanan", self._folder
        )
        if folder:
            self._folder = folder
            self.kolom_folder.setText(folder)

    def _daftar_berkas(self, dasar):
        """(nama berkas, jenis) untuk setiap bagian yang dicentang."""
        pilihan = {k for k, kotak in self.kotak.items() if kotak.isChecked()}
        berkas = []
        if "laporan" in pilihan:
            berkas.append((f"{dasar}_laporan.txt", "laporan"))
        if "ringkasan" in pilihan:
            berkas.append((f"{dasar}_ringkasan.png", "ringkasan"))
        if "segmentasi" in pilihan:
            berkas.append((f"{dasar}_segmentasi.png", "segmentasi"))
        if "asli" in pilihan:
            berkas.append((f"{dasar}_asli.png", "asli"))
        if "berlabel" in pilihan:
            berkas.append((f"{dasar}_berlabel.png", "berlabel"))
        if "tahapan" in pilihan:
            for i, tahap in enumerate(self._data["langkah"], 1):
                judul = nama_aman(tahap["judul"]).replace(" ", "_")[:40]
                berkas.append((
                    f"{dasar}_tahap_{i:02d}_{judul}.png", ("tahap", tahap),
                ))
        return pilihan, berkas

    def simpan(self):
        pilihan = {k for k, kotak in self.kotak.items() if kotak.isChecked()}
        if not pilihan:
            QMessageBox.information(
                self, "Belum ada pilihan",
                "Centang minimal satu bagian yang ingin disimpan.",
            )
            return

        folder = self.kolom_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Folder tidak ditemukan",
                "Pilih folder penyimpanan yang ada terlebih dahulu.",
            )
            return

        dasar = nama_aman(self.kolom_nama.text())
        _, berkas = self._daftar_berkas(dasar)
        tujuan = [
            (os.path.join(folder, nama), jenis) for nama, jenis in berkas
        ]

        sudah_ada = [nama for nama, _ in tujuan if os.path.exists(nama)]
        if sudah_ada:
            jawab = QMessageBox.question(
                self, "Berkas sudah ada",
                f"{len(sudah_ada)} berkas dengan nama itu sudah ada di folder "
                "tersebut. Timpa berkasnya?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if jawab != QMessageBox.Yes:
                return

        tersimpan = []
        try:
            for nama, jenis in tujuan:
                if jenis == "laporan":
                    with open(nama, "w", encoding="utf-8") as f:
                        f.write(laporan_hasil(self._data))
                elif jenis == "ringkasan":
                    simpan_citra(nama, buat_gambar_ringkasan(self._data))
                elif jenis == "segmentasi":
                    simpan_citra(nama, self._data["segmentasi"])
                elif jenis == "asli":
                    simpan_citra(nama, self._data["gambar"])
                elif jenis == "berlabel":
                    simpan_citra(nama, self._data["gambar_tanda"])
                else:
                    simpan_citra(nama, jenis[1]["citra"])
                tersimpan.append(nama)
        except (OSError, ValueError) as e:
            QMessageBox.warning(
                self, "Gagal menyimpan",
                f"{len(tersimpan)} berkas berhasil disimpan, lalu berhenti:\n{e}",
            )
            self.berkas_tersimpan = tersimpan
            return

        self.berkas_tersimpan = tersimpan
        self.label_info.setText(f"{len(tersimpan)} berkas disimpan.")
        QMessageBox.information(
            self, "Hasil disimpan",
            f"{len(tersimpan)} berkas disimpan di:\n{folder}",
        )
        self.accept()


# ==========================================================
# 5. JENDELA UTAMA
# ==========================================================

class JendelaPisang(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Kematangan Pisang")
        layar = QApplication.primaryScreen()
        if layar is not None:
            tersedia = layar.availableGeometry()
            self.resize(min(1200, tersedia.width() - 20),
                        min(780, tersedia.height() - 40))
        else:
            self.resize(1200, 780)
        self.setMinimumSize(1040, 700)
        self.setAcceptDrops(True)

        pusat = QWidget()
        pusat.setObjectName("pusat")
        self.setCentralWidget(pusat)

        luar = QVBoxLayout(pusat)
        luar.setContentsMargins(26, 18, 26, 10)
        luar.setSpacing(14)
        luar.addLayout(self._buat_header())

        badan = QHBoxLayout()
        badan.setSpacing(18)
        self.panggung = self._buat_panggung()
        self.panel_hasil = self._buat_panel()
        beri_bayangan(self.panggung, 40, 45)
        beri_bayangan(self.panel_hasil, 40, 45)
        badan.addWidget(self.panggung, 1)
        badan.addWidget(self.panel_hasil)
        luar.addLayout(badan, 1)

        self.statusBar().setSizeGripEnabled(False)
        self.setStyleSheet(STYLE)
        self.reset_hasil()
        # Ukuran panel belum final saat __init__; hitung ulang setelah tampil.
        QTimer.singleShot(0, self._pas_tinggi_teks)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Lebar panel tetap, tetapi perhitungan diulang bila lebarnya berubah
        # (mis. tampilan dengan skala berbeda), supaya teks tidak terpotong.
        if hasattr(self, "panel_hasil"):
            lebar = max(120, self.panel_hasil.contentsRect().width() - 52)
            if lebar != getattr(self, "_lebar_teks", None):
                self._pas_tinggi_teks()

    def _buat_header(self):
        baris = QHBoxLayout()
        baris.setSpacing(16)

        kiri = QVBoxLayout()
        kiri.setSpacing(3)
        kiri.addWidget(label("HALO, SELAMAT DATANG", "sapaan"))
        kiri.addWidget(label("Deteksi Kematangan Pisang", "judulAplikasi"))
        kiri.addWidget(label(
            "Unggah foto pisang dan lihat tingkat kematangannya dalam sekejap.",
            "subjudul",
        ))
        baris.addLayout(kiri, 1)

        self.chip_status = label("Siap menganalisis", "chip")
        self.chip_status.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        kolom = QVBoxLayout()
        kolom.addStretch(1)
        kolom.addWidget(self.chip_status)
        kolom.addStretch(1)
        baris.addLayout(kolom)
        return baris

    def _set_chip(self, teks, status=None, tooltip=None):
        self.chip_status.setText(teks)
        set_prop(self.chip_status, "status", status or "")
        self.chip_status.setToolTip(tooltip or "")

    def _animasi_verdict(self):
        self._anim_verdict.stop()
        self._anim_verdict.setStartValue(0.0)
        self._anim_verdict.setEndValue(1.0)
        self._anim_verdict.start()

    def _pas_tinggi_teks(self):
        """
        Mengatur tinggi label hasil secara eksplisit.

        Judul hasil (mis. "Belum ada foto") dibuat satu baris: ukuran hurufnya
        dikecilkan sedikit sampai muat di panel, sehingga tidak ada kata yang
        turun ke baris berikutnya lalu menimpa keterangan di bawahnya.
        Keterangan tetap boleh beberapa baris; tingginya dihitung dari lebar
        panel yang sebenarnya.
        """
        lebar = max(120, self.panel_hasil.contentsRect().width() - 52)
        self._lebar_teks = lebar

        # --- Judul hasil: satu baris, huruf menyesuaikan lebar panel ---
        teks = self.label_verdict.text() or " "
        ukuran = 34
        while ukuran > 14:
            huruf = QFont(self.label_verdict.font())
            huruf.setPixelSize(ukuran)
            huruf.setBold(True)
            if QFontMetrics(huruf).horizontalAdvance(teks) <= lebar:
                break
            ukuran -= 1
        huruf = QFont(self.label_verdict.font())
        huruf.setPixelSize(ukuran)
        huruf.setBold(True)
        metrik = QFontMetrics(huruf)
        self.label_verdict.setStyleSheet(f"font-size: {ukuran}px;")
        if metrik.horizontalAdvance(teks) <= lebar:
            self.label_verdict.setWordWrap(False)
            tinggi_judul = metrik.height() + 6
        else:
            # Teks sangat panjang: boleh turun baris, tingginya ikut dihitung.
            self.label_verdict.setWordWrap(True)
            tinggi_judul = metrik.boundingRect(
                0, 0, lebar, 10000, Qt.TextWordWrap, teks
            ).height() + 4
        self.label_verdict.setFixedHeight(tinggi_judul)

        # --- Keterangan: boleh beberapa baris ---
        ket = self.label_ket
        ket.ensurePolished()
        kotak = ket.fontMetrics().boundingRect(
            0, 0, lebar, 10000, Qt.TextWordWrap, ket.text() or " "
        )
        ket.setFixedHeight(kotak.height() + 4)

        # Isi panel bisa bertambah tinggi (mis. banyak pisang terdeteksi).
        # Tinggi minimum jendela ikut menyesuaikan agar isinya tidak saling
        # tumpuk, tetapi tidak melebihi tinggi layar yang tersedia.
        tata_luar = self.centralWidget().layout()
        layar = QApplication.primaryScreen()
        if tata_luar is not None:
            butuh = (tata_luar.totalMinimumSize().height()
                     + self.statusBar().sizeHint().height())
            if layar is not None:
                butuh = min(butuh, layar.availableGeometry().height())
            self.setMinimumHeight(max(700, butuh))

    def _tampil_chip_pisang(self, daftar):
        """Menampilkan kematangan tiap pisang pada panel hasil."""
        while self.grid_pisang.count():
            item = self.grid_pisang.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

        if not daftar or len(daftar) < 2:
            self.bagian_pisang.setVisible(False)
            return

        self.label_pisang.setText(f"PISANG TERDETEKSI ({len(daftar)})")
        for i, objek in enumerate(daftar):
            chip = label(
                f"Pisang {objek['nomor']} · {objek['kematangan']}", "chipPisang"
            )
            set_prop(chip, "kematangan", objek["kematangan"])
            chip.setAlignment(Qt.AlignCenter)
            chip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            chip.setToolTip(
                f"Dominan {objek['dominan'].lower()} "
                f"{angka(objek['persentase'][objek['dominan']], 1)}%"
            )
            baris, kolom = divmod(i, 2)
            if i == len(daftar) - 1 and len(daftar) % 2:
                self.grid_pisang.addWidget(chip, baris, 0, 1, 2)
            else:
                self.grid_pisang.addWidget(chip, baris, kolom)
        self.bagian_pisang.setVisible(True)

    def _buat_panggung(self):
        panggung = QFrame()
        panggung.setObjectName("kartuBiru")
        tata = QVBoxLayout(panggung)
        tata.setContentsMargins(18, 18, 18, 18)
        tata.setSpacing(0)

        # Dua tampilan di kartu biru: hasil (asli + segmentasi) dan tahapan
        self.tumpuk = QStackedWidget()
        self.tumpuk.addWidget(self._halaman_hasil())
        self.tumpuk.addWidget(self._halaman_tahapan())
        tata.addWidget(self.tumpuk)
        return panggung

    def _halaman_hasil(self):
        halaman = QWidget()
        tata = QVBoxLayout(halaman)
        tata.setContentsMargins(0, 0, 0, 0)
        tata.setSpacing(14)

        atas = QHBoxLayout()
        atas.setSpacing(8)
        atas.addWidget(label("PRATINJAU CITRA", "chipBiru"))
        atas.addStretch(1)
        atas.addWidget(label(
            "Seret & lepas foto · klik gambar untuk memilih berkas",
            "petunjukBiru",
        ))
        tata.addLayout(atas)

        baris = QHBoxLayout()
        baris.setSpacing(14)
        self.caption_gambar = {}
        for atribut, judul, kosong in (
            (
                "area_asli", "Citra asli",
                "Seret foto pisang ke sini,\natau klik untuk memilih gambar "
                "(Ctrl+O)",
            ),
            (
                "area_segmentasi", "Hasil segmentasi",
                "Hasil segmentasi tampil di sini setelah foto dianalisis",
            ),
        ):
            kartu = QFrame()
            kartu.setObjectName("kartuGambar")
            isi = QVBoxLayout(kartu)
            isi.setContentsMargins(10, 10, 10, 10)
            isi.setSpacing(8)

            area = AreaGambar(kosong)
            area.diklik.connect(self.pilih_gambar)
            setattr(self, atribut, area)
            isi.addWidget(area, 1)
            self.caption_gambar[atribut] = label(judul, "caption")
            isi.addWidget(self.caption_gambar[atribut], 0, Qt.AlignCenter)
            baris.addWidget(kartu, 1)
        tata.addLayout(baris, 1)
        return halaman

    def _halaman_tahapan(self):
        halaman = QWidget()
        tata = QVBoxLayout(halaman)
        tata.setContentsMargins(0, 0, 0, 0)
        tata.setSpacing(14)

        atas = QHBoxLayout()
        atas.setSpacing(8)
        atas.addWidget(label("TAHAPAN PROSES", "chipBiru"))
        atas.addStretch(1)
        atas.addWidget(label(
            "Pilih tahap di kiri · panah atas/bawah untuk berpindah",
            "petunjukBiru",
        ))
        tata.addLayout(atas)

        self.panel_tahapan = PanelTahapan()
        tata.addWidget(self.panel_tahapan, 1)
        return halaman

    def _buat_panel(self):
        panel = QFrame()
        panel.setObjectName("kartuPutih")
        panel.setFixedWidth(360)
        tata = QVBoxLayout(panel)
        tata.setContentsMargins(26, 24, 26, 22)
        tata.setSpacing(0)

        tata.addWidget(label("HASIL DETEKSI", "eyebrow"))
        tata.addSpacing(6)
        self.label_verdict = label("", "verdict", wrap=True)
        self._efek_verdict = QGraphicsOpacityEffect(self.label_verdict)
        self.label_verdict.setGraphicsEffect(self._efek_verdict)
        self._anim_verdict = QPropertyAnimation(
            self._efek_verdict, b"opacity", self
        )
        self._anim_verdict.setDuration(320)
        self._anim_verdict.setStartValue(0.0)
        self._anim_verdict.setEndValue(1.0)
        self._anim_verdict.setEasingCurve(QEasingCurve.OutCubic)
        self.label_ket = label("", "keterangan", wrap=True)
        tata.addWidget(self.label_verdict)
        tata.addSpacing(6)
        tata.addWidget(self.label_ket)

        # Daftar kematangan tiap pisang (muncul bila lebih dari satu pisang)
        self.bagian_pisang = QWidget()
        isi_pisang = QVBoxLayout(self.bagian_pisang)
        isi_pisang.setContentsMargins(0, 12, 0, 0)
        isi_pisang.setSpacing(7)
        self.label_pisang = label("PISANG TERDETEKSI", "eyebrow")
        isi_pisang.addWidget(self.label_pisang)
        self.grid_pisang = QGridLayout()
        self.grid_pisang.setHorizontalSpacing(6)
        self.grid_pisang.setVerticalSpacing(6)
        self.grid_pisang.setColumnStretch(0, 1)
        self.grid_pisang.setColumnStretch(1, 1)
        isi_pisang.addLayout(self.grid_pisang)
        self.bagian_pisang.setVisible(False)
        tata.addWidget(self.bagian_pisang)
        tata.addSpacing(22)

        tata.addWidget(label("KOMPOSISI WARNA", "eyebrow"))
        tata.addSpacing(10)
        self.jalur = JalurWarna()
        tata.addWidget(self.jalur)
        tata.addSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)
        self.baris = {
            warna: BarisWarna(grid, i, warna, kode)
            for i, (warna, kode) in enumerate(
                {**WARNA_DATA, "Lainnya": WARNA_LAINNYA}.items()
            )
        }
        tata.addLayout(grid)
        tata.addStretch(1)

        self.tombol_tahapan = tombol(
            "Tahapan proses", "sekunder", self.ganti_tampilan, "Ctrl+T"
        )
        self.tombol_tahapan.setToolTip("Lihat seluruh tahap pengolahan citra")
        self.tombol_detail = tombol(
            "Detail citra", "sekunder", self.buka_detail, "Ctrl+D"
        )
        self.tombol_detail.setToolTip("Statistik dan histogram citra")
        self.tombol_simpan = tombol(
            "Simpan hasil", "biru", self.simpan_hasil, "Ctrl+S"
        )
        self.tombol_simpan.setToolTip("Simpan laporan dan gambar hasil analisis")

        tombol_buka = tombol(
            "Buka gambar", "utama", self.pilih_gambar, "Ctrl+O"
        )
        tombol_buka.setToolTip("Pilih berkas gambar (Ctrl+O)")
        tata.addWidget(tombol_buka)
        tata.addSpacing(8)
        tata.addWidget(self.tombol_simpan)
        tata.addSpacing(8)

        baris = QHBoxLayout()
        baris.setSpacing(10)
        baris.addWidget(self.tombol_tahapan, 1)
        baris.addWidget(self.tombol_detail, 1)
        tata.addLayout(baris)
        tata.addSpacing(2)
        tata.addWidget(
            tombol("Bersihkan hasil", "halus", self.reset_hasil),
            0, Qt.AlignCenter,
        )
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
            self.area_asli.set_sorot(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.area_asli.set_sorot(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.area_asli.set_sorot(False)
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
            (
                gambar, segmentasi, persentase, dominan, kematangan, mask,
                langkah, daftar_pisang, gambar_tanda,
            ) = deteksi_pisang(nama_file)
        except BukanPisang as e:
            self.tampilkan_bukan_pisang(e, nama_file)
            return
        except Exception as e:
            self._set_chip("Gagal memproses", "tolak", str(e))
            QMessageBox.warning(self, "Gambar tidak bisa diproses", str(e))
            return

        banyak = len(daftar_pisang) > 1
        self.area_asli.set_gambar(
            numpy_ke_pixmap(gambar_tanda if banyak else gambar)
        )
        self.caption_gambar["area_asli"].setText(
            "Citra asli (bernomor)" if banyak else "Citra asli"
        )
        self.area_segmentasi.set_gambar(numpy_ke_pixmap(segmentasi))

        for warna, nilai in persentase.items():
            self.baris[warna].isi(f"{angka(nilai, 1)}%", warna == dominan)
        sisa = max(0.0, 100.0 - sum(persentase.values()))
        self.baris["Lainnya"].isi(f"{angka(sisa, 1)}%")
        self.jalur.atur(persentase)

        if banyak:
            self.label_verdict.setText(f"{len(daftar_pisang)} pisang terdeteksi")
            self.label_ket.setText(
                f"{ringkas_kematangan(daftar_pisang)}. Warna dominan gabungan "
                f"{dominan.lower()} ({angka(persentase[dominan], 1)}%)."
            )
        else:
            self.label_verdict.setText(kematangan)
            self.label_ket.setText(
                f"Warna dominan {dominan.lower()}, "
                f"{angka(persentase[dominan], 1)}% dari area pisang."
            )
        self._tampil_chip_pisang(daftar_pisang)
        self._pas_tinggi_teks()
        self._animasi_verdict()

        # Tahapan proses ditampilkan di tampilan kedua
        self.panel_tahapan.set_langkah(langkah)
        self.tombol_tahapan.setEnabled(True)

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

        # Data lengkap untuk fitur simpan hasil
        self.hasil = {
            "nama_file": nama_file,
            "nama_berkas": self.nama_berkas,
            "gambar": gambar,
            "gambar_tanda": gambar_tanda,
            "segmentasi": segmentasi,
            "persentase": persentase,
            "dominan": dominan,
            "kematangan": kematangan,
            "mask": mask,
            "langkah": langkah,
            "daftar_pisang": daftar_pisang,
            "info": self.info,
        }
        self.tombol_simpan.setEnabled(True)
        self._set_chip("Selesai dianalisis", "ok", self.nama_berkas)
        self._tampilan_hasil()

        jumlah = f"{len(daftar_pisang)} pisang, " if banyak else ""
        self.statusBar().showMessage(
            f"{self.nama_berkas}  ({jumlah}dianalisis pada "
            f"{gambar.shape[1]}×{gambar.shape[0]} piksel, "
            f"{len(langkah)} tahap proses)"
        )

    def tampilkan_bukan_pisang(self, e, nama_file):
        """Gambar tidak berisi pisang: tampilkan alasannya, hasil lama dibersihkan."""
        self.area_asli.set_gambar(numpy_ke_pixmap(e.gambar))
        self.area_segmentasi.hapus()
        self.jalur.atur()
        for baris in self.baris.values():
            baris.isi("")
        self.label_verdict.setText("Bukan pisang")
        self.label_ket.setText(e.alasan)
        self._tampil_chip_pisang([])
        self.caption_gambar["area_asli"].setText("Citra asli")
        self._pas_tinggi_teks()
        self._animasi_verdict()

        # Tahapan tetap bisa dilihat untuk menelusuri alasan penolakan
        self.panel_tahapan.set_langkah(e.langkah)
        self.tombol_tahapan.setEnabled(True)
        self.info, self.nama_berkas = None, ""
        self.tombol_detail.setEnabled(False)
        self.hasil = None
        self.tombol_simpan.setEnabled(False)
        self._set_chip(
            "Bukan pisang", "tolak", os.path.basename(nama_file)
        )
        self._tampilan_hasil()
        self.statusBar().showMessage(
            f"{os.path.basename(nama_file)}  (tidak dikenali sebagai pisang)"
        )

    def ganti_tampilan(self):
        """Beralih antara tampilan hasil dan tampilan tahapan proses."""
        if self.tumpuk.currentIndex() == 0:
            self.tumpuk.setCurrentIndex(1)
            self.tombol_tahapan.setText("Kembali ke hasil")
            self.panel_tahapan.daftar.setFocus()
        else:
            self._tampilan_hasil()
        self._pas_tinggi_teks()

    def _tampilan_hasil(self):
        self.tumpuk.setCurrentIndex(0)
        self.tombol_tahapan.setText("Tahapan proses")
        self._pas_tinggi_teks()

    def buka_detail(self):
        if self.info:
            DialogDetail(self, self.nama_berkas, *self.info).exec_()

    def simpan_hasil(self):
        """Fitur simpan hasil: laporan, ringkasan, citra, dan tahapan."""
        if not self.hasil:
            return
        dialog = DialogSimpan(self, self.hasil)
        if dialog.exec_() == QDialog.Accepted and dialog.berkas_tersimpan:
            self.statusBar().showMessage(
                f"{len(dialog.berkas_tersimpan)} berkas hasil disimpan di "
                f"{os.path.dirname(dialog.berkas_tersimpan[0])}"
            )

    def reset_hasil(self):
        self.area_asli.hapus()
        self.area_segmentasi.hapus()
        self.panel_tahapan.kosongkan()
        self._tampilan_hasil()
        self.tombol_tahapan.setEnabled(False)
        self.jalur.atur()
        for baris in self.baris.values():
            baris.isi("")
        self.label_verdict.setText("Belum ada foto")
        self.label_ket.setText(TEKS_KOSONG)
        self._tampil_chip_pisang([])
        self.caption_gambar["area_asli"].setText("Citra asli")
        self._pas_tinggi_teks()
        self._efek_verdict.setOpacity(1.0)
        self.info, self.nama_berkas = None, ""
        self.tombol_detail.setEnabled(False)
        self.hasil = None
        self.tombol_simpan.setEnabled(False)
        self._set_chip("Siap menganalisis")
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

# Blok di atas adalah satu-satunya titik masuk aplikasi.