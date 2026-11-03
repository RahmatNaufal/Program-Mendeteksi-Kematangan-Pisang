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


# ==========================================================
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
    efek.setColor(QColor(60, 45, 0, alpha))
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
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    jendela = JendelaPisang()
    jendela.show()
    sys.exit(app.exec_())