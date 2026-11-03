import os
import sys
from string import Template

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QRectF, QVariantAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
    QSizePolicy
)


# 1. PROSES DETEKSI KEMATANGAN PISANG 

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
# menghitung persentase piksel untuk setiap warna
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


# 2. DESAIN
#
# Antarmuka dibuat netral. Satu-satunya warna di layar adalah
# warna data itu sendiri: hijau, kuning, dan cokelat kulit pisang.
# Panggung foto gelap karena hasil segmentasi berlatar hitam,
# jadi hasilnya menyatu dengan panggung dan bentuk pisang menonjol.


INK = "#1C2124"
MUTED = "#5F6A71"
GARIS = "#D3D8DA"
PANEL = "#F2F4F3"
STAGE = "#1E2225"
STAGE_TEKS = "#8E989E"

WARNA_DATA = {
    "Hijau": "#7FA83A",
    "Kuning": "#EFC31A",
    "Cokelat": "#7B4A24",
}
WARNA_LAINNYA = "#D9DDDE"

STYLE = Template("""
QMainWindow { background: $PANEL; }

QFrame#stage { background: $STAGE; }
QFrame#panel { background: $PANEL; border-left: 1px solid $GARIS; }

QLabel#area { color: $STAGE_TEKS; font-size: 13px; background: transparent; }
QLabel#area[kosong="true"] { border: 1px solid #343A3E; }
QLabel#caption { color: $STAGE_TEKS; font-size: 12px; }

QLabel#verdict { color: $INK; font-size: 30px; font-weight: 700; }
QLabel#keterangan { color: $MUTED; font-size: 13px; }

QLabel#namaWarna, QLabel#nilaiWarna { color: $INK; font-size: 13px; }
QLabel#namaWarna[tebal="true"], QLabel#nilaiWarna[tebal="true"] {
    font-weight: 700;
}

QPushButton#utama, QPushButton#sekunder {
    border-radius: 4px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton#utama {
    background: $INK; color: white; border: 2px solid $INK;
}
QPushButton#utama:hover { background: #343C41; border-color: #343C41; }
QPushButton#utama:pressed { background: #0F1214; border-color: #0F1214; }
QPushButton#sekunder {
    background: transparent; color: $INK; border: 2px solid #C4CACC;
}
QPushButton#sekunder:hover { background: #E6E9E8; }
QPushButton#sekunder:pressed { background: #D9DDDC; }
QPushButton#utama:focus, QPushButton#sekunder:focus {
    border-color: #4A7BD0;
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
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def format_persen(nilai):
    return f"{nilai:.1f}%".replace(".", ",")


# 3. KOMPONEN

class AreaGambar(QLabel):
    """Menampilkan gambar dengan rasio asli, menyesuaikan ukuran jendela."""

    def __init__(self, teks_kosong):
        super().__init__(teks_kosong)
        self._teks_kosong = teks_kosong
        self._pixmap = None
        self.setObjectName("area")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setProperty("kosong", True)

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
            self.setPixmap(
                self._pixmap.scaled(
                    self.width(), self.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


class JalurWarna(QWidget):
    """
    Satu jalur horizontal yang diisi seperti kulit pisang yang menua:
    hijau, kuning, lalu cokelat. Lebar tiap bagian sama dengan
    persentase warnanya. Sisa piksel yang tidak masuk rentang
    ditampilkan abu-abu.
    """

    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self._persen = {}
        self._t = 0.0

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(450)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._ubah)

    def _ubah(self, nilai):
        self._t = float(nilai)
        self.update()

    def tampilkan(self, persen):
        self._persen = dict(persen)
        self._anim.stop()
        self._anim.start()

    def kosongkan(self):
        self._anim.stop()
        self._persen = {}
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
        self.swatch = QLabel()
        self.swatch.setFixedSize(12, 12)
        self.swatch.setStyleSheet(
            f"background: {kode}; border-radius: 2px;"
        )

        self.nama = QLabel(nama)
        self.nama.setObjectName("namaWarna")

        self.nilai = QLabel("")
        self.nilai.setObjectName("nilaiWarna")
        self.nilai.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        grid.addWidget(self.swatch, indeks, 0)
        grid.addWidget(self.nama, indeks, 1)
        grid.addWidget(self.nilai, indeks, 2)

    def isi(self, teks, tebal=False):
        self.nilai.setText(teks)
        set_prop(self.nama, "tebal", tebal)
        set_prop(self.nilai, "tebal", tebal)

    def kosongkan(self):
        self.isi("")


# 4. JENDELA UTAMA

class JendelaPisang(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Kematangan Pisang")
        self.resize(1120, 680)
        self.setMinimumSize(960, 600)
        self.setAcceptDrops(True)

        pusat = QWidget()
        self.setCentralWidget(pusat)
        utama = QHBoxLayout(pusat)
        utama.setContentsMargins(0, 0, 0, 0)
        utama.setSpacing(0)

        utama.addWidget(self._buat_panggung(), 1)
        utama.addWidget(self._buat_panel())

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("Mendukung berkas JPG, PNG, dan BMP.")

        self.setStyleSheet(STYLE)

    # ---------- Panggung foto ----------
    def _buat_panggung(self):
        panggung = QFrame()
        panggung.setObjectName("stage")

        tata = QHBoxLayout(panggung)
        tata.setContentsMargins(28, 28, 28, 24)
        tata.setSpacing(24)

        self.area_asli = AreaGambar(
            "Seret foto pisang ke sini,\natau pilih Buka gambar (Ctrl+O)"
        )
        self.area_segmentasi = AreaGambar(
            "Hasil segmentasi tampil di sini"
        )

        for area, keterangan in (
            (self.area_asli, "Asli"),
            (self.area_segmentasi, "Segmentasi"),
        ):
            kolom = QVBoxLayout()
            kolom.setSpacing(10)
            kolom.addWidget(area, 1)
            cap = QLabel(keterangan)
            cap.setObjectName("caption")
            cap.setAlignment(Qt.AlignCenter)
            kolom.addWidget(cap)
            tata.addLayout(kolom, 1)

        return panggung

    # ---------- Panel hasil ----------
    def _buat_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(330)

        tata = QVBoxLayout(panel)
        tata.setContentsMargins(28, 28, 28, 24)
        tata.setSpacing(0)

        self.label_verdict = QLabel("Belum ada gambar")
        self.label_verdict.setObjectName("verdict")
        self.label_verdict.setWordWrap(True)
        tata.addWidget(self.label_verdict)

        tata.addSpacing(8)
        self.label_ket = QLabel(
            "Buka foto pisang untuk melihat tingkat kematangannya."
        )
        self.label_ket.setObjectName("keterangan")
        self.label_ket.setWordWrap(True)
        tata.addWidget(self.label_ket)

        tata.addSpacing(32)
        self.jalur = JalurWarna()
        tata.addWidget(self.jalur)

        tata.addSpacing(16)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.baris = {}
        for i, (warna, kode) in enumerate(WARNA_DATA.items()):
            self.baris[warna] = BarisWarna(grid, i, warna, kode)
        self.baris["Lainnya"] = BarisWarna(
            grid, len(WARNA_DATA), "Lainnya", WARNA_LAINNYA
        )
        tata.addLayout(grid)

        tata.addStretch(1)

        tombol_pilih = QPushButton("Buka gambar")
        tombol_pilih.setObjectName("utama")
        tombol_pilih.setShortcut("Ctrl+O")
        tombol_pilih.setCursor(Qt.PointingHandCursor)
        tombol_pilih.clicked.connect(self.pilih_gambar)

        tombol_reset = QPushButton("Bersihkan")
        tombol_reset.setObjectName("sekunder")
        tombol_reset.setCursor(Qt.PointingHandCursor)
        tombol_reset.clicked.connect(self.reset_hasil)

        tata.addWidget(tombol_pilih)
        tata.addSpacing(8)
        tata.addWidget(tombol_reset)

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
            self,
            "Pilih gambar pisang",
            "",
            "Gambar (*.jpg *.jpeg *.png *.bmp);;Semua file (*.*)",
        )
        if nama_file:
            self.proses(nama_file)

    def proses(self, nama_file):
        try:
            gambar, segmentasi, persentase, dominan, kematangan = (
                deteksi_pisang(nama_file)
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Gambar tidak bisa diproses", str(e)
            )
            return

        self.area_asli.set_gambar(numpy_ke_pixmap(gambar))
        self.area_segmentasi.set_gambar(numpy_ke_pixmap(segmentasi))

        for warna, nilai in persentase.items():
            self.baris[warna].isi(
                format_persen(nilai), tebal=(warna == dominan)
            )
        sisa = max(0.0, 100.0 - sum(persentase.values()))
        self.baris["Lainnya"].isi(format_persen(sisa))

        self.jalur.tampilkan(persentase)

        self.label_verdict.setText(kematangan.capitalize())
        self.label_ket.setText(
            f"Warna dominan {dominan.lower()}, "
            f"{format_persen(persentase[dominan])} dari area pisang."
        )

        self.statusBar().showMessage(
            f"{os.path.basename(nama_file)}  (dianalisis pada 600×400 piksel)"
        )

    def reset_hasil(self):
        self.area_asli.hapus()
        self.area_segmentasi.hapus()
        self.jalur.kosongkan()

        for baris in self.baris.values():
            baris.kosongkan()

        self.label_verdict.setText("Belum ada gambar")
        self.label_ket.setText(
            "Buka foto pisang untuk melihat tingkat kematangannya."
        )
        self.statusBar().showMessage("Mendukung berkas JPG, PNG, dan BMP.")


# 5. MENJALANKAN APLIKASI

if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("Bahnschrift")
    if hasattr(font, "setFamilies"):
        font.setFamilies(
            ["Bahnschrift", "Segoe UI", "Helvetica Neue", "Noto Sans"]
        )
    font.setPointSize(10)
    app.setFont(font)

    jendela = JendelaPisang()
    jendela.show()
    sys.exit(app.exec_())