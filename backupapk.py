import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# ==========================================
# 1. PROSES DETEKSI KEMATANGAN PISANG
# ==========================================
def deteksi_pisang(nama_file):
    gambar = cv2.imread(nama_file)
    if gambar is None:
        raise ValueError("Gambar tidak dapat dibaca.")
    gambar = cv2.resize(gambar, (600, 400))
    blur = cv2.GaussianBlur(gambar, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Segmentasi objek pisang
    saturation = hsv[:, :, 1]
    _, mask_pisang = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    mask_pisang = cv2.morphologyEx(
        mask_pisang, cv2.MORPH_OPEN, kernel
    )
    mask_pisang = cv2.morphologyEx(
        mask_pisang, cv2.MORPH_CLOSE, kernel
    )
    # Mencari kontur terbesar
    # ==========================================
    # MENCARI BEBERAPA AREA OBJEK
    # ==========================================
    kontur, _ = cv2.findContours(
        mask_pisang,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not kontur:
        raise ValueError("Objek pisang tidak ditemukan.")
    # Membuat mask kosong
    mask_objek = np.zeros_like(mask_pisang)
    # Menentukan luas minimum objek
    tinggi, lebar = mask_pisang.shape
    luas_gambar = tinggi * lebar
    luas_minimum = luas_gambar * 0.005
    # Memilih kontur yang cukup besar
    kontur_terpilih = [
        c for c in kontur
        if cv2.contourArea(c) >= luas_minimum
    ]
    if not kontur_terpilih:
        raise ValueError(
            "Tidak ada objek yang memenuhi batas luas."
        )
    # Menggambar semua kontur terpilih
    cv2.drawContours(
        mask_objek,
        kontur_terpilih,
        -1,
        255,
        thickness=cv2.FILLED
    )
    total_piksel = cv2.countNonZero(mask_objek)
    if total_piksel == 0:
        raise ValueError("Area objek pisang kosong.")
    # Rentang HSV untuk setiap warna
    rentang_warna = {
        "Hijau": (
            np.array([35, 60, 40]),
            np.array([85, 255, 255])
        ),
        "Kuning": (
            np.array([20, 80, 100]),
            np.array([34, 255, 255])
        ),
        "Cokelat": (
            np.array([5, 60, 20]),
            np.array([19, 255, 200])
        )
    }
    persentase = {}
    for warna, (bawah, atas) in rentang_warna.items():
        mask_warna = cv2.inRange(hsv, bawah, atas)
        mask_warna = cv2.bitwise_and(
            mask_warna, mask_objek
        )
        jumlah = cv2.countNonZero(mask_warna)
        persentase[warna] = jumlah / total_piksel * 100
    # Menentukan warna dominan
    warna_dominan = max(persentase, key=persentase.get)
    klasifikasi = {
        "Hijau": "Belum Matang",
        "Kuning": "Matang",
        "Cokelat": "Terlalu Matang"
    }
    kematangan = klasifikasi[warna_dominan]
    # Hasil segmentasi
    hasil_segmentasi = cv2.bitwise_and(
        blur, blur, mask=mask_objek
    )
    return gambar, hasil_segmentasi, persentase, warna_dominan, kematangan
# ==========================================
# 2. MEMBUAT JENDELA APLIKASI
# ==========================================
akar = tk.Tk()
akar.title("Deteksi Kematangan Pisang")
akar.geometry("1000x720")
akar.minsize(850, 650)
akar.configure(bg="#F2F5F1")
hijau_tua = "#D7C607"
hijau = "#000202"
putih = "#FFFFFF"
teks = "#000D05"
abu = "#000E04"
gambar_asli_tk = None
gambar_segmentasi_tk = None
# ==========================================
# 3. BAGIAN JUDUL
# ==========================================
header = tk.Frame(akar, bg=hijau_tua, pady=18)
header.pack(fill="x")
tk.Label(
    header,
    text="DETEKSI KEMATANGAN PISANG",
    font=("Arial", 21, "bold"),
    fg=putih,
    bg=hijau_tua
).pack()
tk.Label(
    header,
    text="Pengolahan Citra Digital | Metode HSV",
    font=("Arial", 10),
    fg="#D8E8DC",
    bg=hijau_tua
).pack(pady=(5, 0))
# ==========================================
# 4. BAGIAN GAMBAR
# ==========================================
konten = tk.Frame(akar, bg="#F2F5F1")
konten.pack(fill="both", expand=True, padx=20, pady=15)
panel_gambar = tk.Frame(konten, bg="#F2F5F1")
panel_gambar.pack(side="left", fill="both", expand=True, padx=(0, 12))
panel_hasil = tk.Frame(
    konten,
    bg=putih,
    padx=18,
    pady=15,
    highlightbackground="#DCE5DC",
    highlightthickness=1
)
panel_hasil.pack(side="right", fill="y", padx=(12, 0))
def buat_panel_gambar(parent, judul):
    bingkai = tk.Frame(
        parent,
        bg=putih,
        padx=10,
        pady=10,
        highlightbackground="#DCE5DC",
        highlightthickness=1
    )
    tk.Label(
        bingkai,
        text=judul,
        font=("Arial", 11, "bold"),
        bg=putih,
        fg=teks
    ).pack(pady=(0, 8))
    area = tk.Label(
        bingkai,
        text="Belum ada gambar",
        bg="#E8ECE7",
        fg=abu,
        width=35,
        height=12
    )
    area.pack(fill="both", expand=True)
    return bingkai, area
panel_asli, area_asli = buat_panel_gambar(
    panel_gambar, "GAMBAR ASLI"
)
panel_asli.pack(fill="both", expand=True, pady=(0, 10))
panel_segmentasi, area_segmentasi = buat_panel_gambar(
    panel_gambar, "HASIL SEGMENTASI"
)
panel_segmentasi.pack(fill="both", expand=True, pady=(10, 0))
# ==========================================
# 5. BAGIAN HASIL DETEKSI
# ==========================================
tk.Label(
    panel_hasil,
    text="HASIL ANALISIS",
    font=("Arial", 14, "bold"),
    bg=putih,
    fg=hijau_tua
).pack(anchor="w", pady=(0, 15))
tk.Label(
    panel_hasil,
    text="Persentase warna",
    font=("Arial", 10, "bold"),
    bg=putih,
    fg=teks
).pack(anchor="w", pady=(0, 10))
warna_tampilan = {
    "Hijau": "#45965B",
    "Kuning": "#E5B932",
    "Cokelat": "#956044"
}
label_persen = {}
batang_warna = {}
for warna, kode_warna in warna_tampilan.items():
    tk.Label(
        panel_hasil,
        text=warna,
        font=("Arial", 10),
        bg=putih,
        fg=teks
    ).pack(anchor="w")
    label = tk.Label(
        panel_hasil,
        text="0.00%",
        font=("Arial", 10, "bold"),
        bg=putih,
        fg=teks
    )
    label.pack(anchor="e")
    label_persen[warna] = label
    kanvas = tk.Canvas(
        panel_hasil,
        width=250,
        height=12,
        bg="#E8ECE7",
        highlightthickness=0
    )
    kanvas.pack(fill="x", pady=(2, 12))
    batang_warna[warna] = (kanvas, kode_warna)
garis = tk.Frame(panel_hasil, bg="#DCE5DC", height=1)
garis.pack(fill="x", pady=8)
tk.Label(
    panel_hasil,
    text="WARNA DOMINAN",
    font=("Arial", 9, "bold"),
    bg=putih,
    fg=abu
).pack(anchor="w", pady=(5, 5))
label_dominan = tk.Label(
    panel_hasil,
    text="-",
    font=("Arial", 17, "bold"),
    bg=putih,
    fg=hijau
)
label_dominan.pack(anchor="w")
tk.Label(
    panel_hasil,
    text="TINGKAT KEMATANGAN",
    font=("Arial", 9, "bold"),
    bg=putih,
    fg=abu
).pack(anchor="w", pady=(18, 5))
label_kematangan = tk.Label(
    panel_hasil,
    text="Belum dianalisis",
    font=("Arial", 13, "bold"),
    bg=putih,
    fg=hijau_tua,
    wraplength=240
)
label_kematangan.pack(anchor="w")
# ==========================================
# 6. MEMILIH GAMBAR DAN MENAMPILKAN HASIL
# ==========================================
def pilih_gambar():
    global gambar_asli_tk, gambar_segmentasi_tk
    nama_file = filedialog.askopenfilename(
        title="Pilih gambar pisang",
        filetypes=[
            ("Gambar", "*.jpg *.jpeg *.png *.bmp"),
            ("Semua file", "*.*")
        ]
    )
    if not nama_file:
        return
    try:
        gambar, segmentasi, persentase, dominan, kematangan = (
            deteksi_pisang(nama_file)
        )
        # Mengubah BGR menjadi RGB untuk tampilan
        asli_rgb = cv2.cvtColor(
            gambar, cv2.COLOR_BGR2RGB
        )
        segmentasi_rgb = cv2.cvtColor(
            segmentasi, cv2.COLOR_BGR2RGB
        )
        # Menyesuaikan gambar dengan panel
        foto_asli = Image.fromarray(asli_rgb)
        foto_asli.thumbnail((420, 240))
        foto_segmentasi = Image.fromarray(segmentasi_rgb)
        foto_segmentasi.thumbnail((420, 240))
        gambar_asli_tk = ImageTk.PhotoImage(foto_asli)
        gambar_segmentasi_tk = ImageTk.PhotoImage(
            foto_segmentasi
        )
        area_asli.configure(
            image=gambar_asli_tk,
            text="",
            width=0,
            height=0
        )
        area_segmentasi.configure(
            image=gambar_segmentasi_tk,
            text="",
            width=0,
            height=0
        )
        # Memperbarui persentase dan batang warna
        for warna, nilai in persentase.items():
            label_persen[warna].configure(
                text=f"{nilai:.2f}%"
            )
            kanvas, kode_warna = batang_warna[warna]
            kanvas.delete("all")
            panjang = min(max(nilai, 0), 100) * 2.5
            kanvas.create_rectangle(
                0, 0, panjang, 12,
                fill=kode_warna,
                outline=""
            )
        label_dominan.configure(text=dominan)
        label_kematangan.configure(text=kematangan)
    except Exception as e:
        messagebox.showerror(
            "Gagal memproses gambar",
            str(e)
        )
# ==========================================
# 7. TOMBOL DAN KETERANGAN
# ==========================================
panel_bawah = tk.Frame(akar, bg="#F2F5F1")
panel_bawah.pack(fill="x", padx=20, pady=(0, 15))
# Membuat dua tombol
panel_tombol = tk.Frame(
    panel_bawah,
    bg="#F2F5F1"
)
panel_tombol.pack()
# Tombol untuk memilih gambar pertama atau gambar berikutnya
tombol_pilih = tk.Button(
    panel_tombol,
    text="PILIH GAMBAR",
    command=pilih_gambar,
    font=("Arial", 11, "bold"),
    bg="#367A4D",
    fg="white",
    activebackground="#204B35",
    activeforeground="white",
    relief="flat",
    padx=18,
    pady=12,
    cursor="hand2"
)
tombol_pilih.pack(side="left", padx=5)
# Tombol untuk mengosongkan hasil
def reset_hasil():
    global gambar_asli_tk, gambar_segmentasi_tk
    gambar_asli_tk = None
    gambar_segmentasi_tk = None
    area_asli.configure(
        image="",
        text="Belum ada gambar",
        width=35,
        height=12
    )
    area_segmentasi.configure(
        image="",
        text="Belum ada gambar",
        width=35,
        height=12
    )
    for warna in label_persen:
        label_persen[warna].configure(text="0.00%")
        kanvas, _ = batang_warna[warna]
        kanvas.delete("all")
    label_dominan.configure(text="-")
    label_kematangan.configure(text="Belum dianalisis")
tombol_reset = tk.Button(
    panel_tombol,
    text="RESET HASIL",
    command=reset_hasil,
    font=("Arial", 11, "bold"),
    bg="#68756C",
    fg="white",
    activebackground="#4B554E",
    activeforeground="white",
    relief="flat",
    padx=18,
    pady=12,
    cursor="hand2"
)
tombol_reset.pack(side="left", padx=5)
tk.Label(
    panel_bawah,
    text="Pilih gambar lain untuk melakukan analisis berikutnya.",
    font=("Arial", 9),
    bg="#F2F5F1",
    fg="#68756C"
).pack(pady=(7, 0))
akar.mainloop() 