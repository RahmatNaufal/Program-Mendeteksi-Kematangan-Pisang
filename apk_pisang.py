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

    # ---------------------------------------------------------
    # SEGMENTASI OBJEK PISANG
    # ---------------------------------------------------------
    # 1. Segmentasi Warna Pisang (Hijau, Kuning, Cokelat/Oranye)
    bawah_pisang = np.array([5, 20, 20])
    atas_pisang = np.array([85, 255, 255])

    mask_pisang = cv2.inRange(hsv, bawah_pisang, atas_pisang)

    # Simpan mask asli (sebelum dilebarkan) untuk membuang sisa latar putih
    mask_pisang_asli = mask_pisang.copy()

    # 2. Operasi Morfologi untuk Menyatukan Objek yang Terputus
    kernel_besar = np.ones((15, 15), np.uint8)
    kernel_kecil = np.ones((5, 5), np.uint8)

    # Close: Menutup lubang-lubang kecil di dalam pisang
    mask_pisang = cv2.morphologyEx(mask_pisang, cv2.MORPH_CLOSE, kernel_besar)

    # Dilate: Menyambung bagian kulit pisang yang renggang/terpisah
    mask_pisang = cv2.dilate(mask_pisang, kernel_kecil, iterations=2)

    # 3. Mencari Kontur Terbesar
    kontur, _ = cv2.findContours(
        mask_pisang,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not kontur:
        raise ValueError("Objek pisang tidak ditemukan.")

    mask_objek = np.zeros_like(mask_pisang)

    # Memilih kontur dengan area terbesar (sisir pisang utama)
    kontur_terbesar = max(kontur, key=cv2.contourArea)

    # Memastikan kontur cukup besar (bukan noise)
    tinggi, lebar = mask_pisang.shape
    luas_gambar = tinggi * lebar

    if cv2.contourArea(kontur_terbesar) < (luas_gambar * 0.05):
        raise ValueError("Objek pisang terlalu kecil atau tidak terdeteksi.")

    # Menggambar kontur utama dan mengisi bagian dalamnya secara penuh
    cv2.drawContours(
        mask_objek,
        [kontur_terbesar],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ---------------------------------------------------------
    # PERBAIKAN: MEMBUANG TEPI PUTIH PADA HASIL SEGMENTASI
    # ---------------------------------------------------------
    # a. Mask hasil kontur hanya diambil di area yang benar-benar berwarna
    #    pisang (latar putih/abu yang ikut terlebar akibat dilate dibuang)
    mask_objek = cv2.bitwise_and(mask_objek, mask_pisang_asli)

    # b. Tutup lubang kecil di dalam pisang (bagian terang/mengkilap)
    mask_objek = cv2.morphologyEx(mask_objek, cv2.MORPH_CLOSE, kernel_kecil)

    # c. Isi ulang kontur terbesar agar bagian dalam tetap penuh
    kontur2, _ = cv2.findContours(
        mask_objek,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not kontur2:
        raise ValueError("Objek pisang tidak ditemukan.")

    mask_objek = np.zeros_like(mask_objek)
    cv2.drawContours(
        mask_objek,
        [max(kontur2, key=cv2.contourArea)],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # d. Kikis tepi 2 piksel agar piksel campuran (pisang + putih) hilang
    mask_objek = cv2.erode(mask_objek, np.ones((3, 3), np.uint8), iterations=2)

    # e. Haluskan tepi mask
    mask_objek = cv2.GaussianBlur(mask_objek, (3, 3), 0)
    _, mask_objek = cv2.threshold(mask_objek, 127, 255, cv2.THRESH_BINARY)

    total_piksel = cv2.countNonZero(mask_objek)

    if total_piksel == 0:
        raise ValueError("Area objek pisang kosong.")

    # Rentang HSV untuk setiap warna
    rentang_warna = {
        "Hijau": (
            np.array([35, 40, 40]),
            np.array([85, 255, 255])
        ),
        "Kuning": (
            np.array([18, 50, 50]),
            np.array([34, 255, 255])
        ),
        "Cokelat": (
            np.array([2, 40, 20]),
            np.array([17, 255, 200])
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

    # Hasil segmentasi (memakai gambar asli agar tetap tajam)
    hasil_segmentasi = cv2.bitwise_and(
        gambar, gambar, mask=mask_objek
    )

    return gambar, hasil_segmentasi, persentase, warna_dominan, kematangan


# ==========================================
# 2. MEMBUAT JENDELA APLIKASI
# ==========================================

akar = tk.Tk()
akar.title("Deteksi Kematangan Pisang")
akar.geometry("900x720")
akar.minsize(700, 620)
akar.configure(bg="#F2F5F1")

hijau_tua = "#D7C607"
hijau = "#000202"
putih = "#FFFFFF"
teks = "#000D05"
abu = "#000E04"

gambar_asli_tk = None
gambar_segmentasi_tk = None
hasil_terakhir = None   # menyimpan hasil analisis untuk jendela analisis warna

warna_tampilan = {
    "Hijau": "#45965B",
    "Kuning": "#E5B932",
    "Cokelat": "#956044"
}

# Warna teks hasil kematangan
warna_kematangan = {
    "Belum Matang": "#2E8B4A",
    "Matang": "#C9A100",
    "Terlalu Matang": "#8B4F2B"
}


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
    fg="#FFFFFF",
    bg=hijau_tua
).pack(pady=(5, 0))


# ==========================================
# 4. BAGIAN GAMBAR (HANYA GAMBAR YANG DIMASUKKAN)
# ==========================================

konten = tk.Frame(akar, bg="#F2F5F1")
konten.pack(fill="both", expand=True, padx=20, pady=15)

bingkai_gambar = tk.Frame(
    konten,
    bg=putih,
    padx=10,
    pady=10,
    highlightbackground="#DCE5DC",
    highlightthickness=1
)
bingkai_gambar.pack(fill="both", expand=True)

tk.Label(
    bingkai_gambar,
    text="GAMBAR PISANG",
    font=("Arial", 11, "bold"),
    bg=putih,
    fg=teks
).pack(pady=(0, 8))

area_asli = tk.Label(
    bingkai_gambar,
    text="Belum ada gambar",
    bg="#E8ECE7",
    fg=abu,
    width=35,
    height=12
)
area_asli.pack(fill="both", expand=True)


# ==========================================
# 5. BAGIAN HASIL KEMATANGAN
# ==========================================

panel_hasil = tk.Frame(
    konten,
    bg=putih,
    padx=18,
    pady=12,
    highlightbackground="#DCE5DC",
    highlightthickness=1
)
panel_hasil.pack(fill="x", pady=(12, 0))

tk.Label(
    panel_hasil,
    text="TINGKAT KEMATANGAN",
    font=("Arial", 9, "bold"),
    bg=putih,
    fg=abu
).pack()

label_kematangan = tk.Label(
    panel_hasil,
    text="Belum dianalisis",
    font=("Arial", 22, "bold"),
    bg=putih,
    fg=hijau_tua
)
label_kematangan.pack(pady=(4, 0))


# ==========================================
# 6. JENDELA ANALISIS WARNA
# ==========================================

def tampilkan_analisis():
    if hasil_terakhir is None:
        messagebox.showinfo(
            "Belum ada data",
            "Pilih gambar terlebih dahulu."
        )
        return

    global gambar_segmentasi_tk

    jendela = tk.Toplevel(akar)
    jendela.title("Analisis Warna")
    jendela.configure(bg="#F2F5F1")
    jendela.resizable(False, False)
    jendela.transient(akar)

    isi = tk.Frame(jendela, bg=putih, padx=20, pady=18,
                   highlightbackground="#DCE5DC", highlightthickness=1)
    isi.pack(padx=15, pady=15)

    tk.Label(
        isi, text="ANALISIS WARNA",
        font=("Arial", 14, "bold"), bg=putih, fg=hijau_tua
    ).pack(anchor="w", pady=(0, 10))

    # Hasil segmentasi
    tk.Label(
        isi, text="Hasil Segmentasi",
        font=("Arial", 10, "bold"), bg=putih, fg=teks
    ).pack(anchor="w", pady=(0, 6))

    foto_seg = Image.fromarray(
        cv2.cvtColor(hasil_terakhir["segmentasi"], cv2.COLOR_BGR2RGB)
    )
    foto_seg.thumbnail((420, 280))
    gambar_segmentasi_tk = ImageTk.PhotoImage(foto_seg)

    tk.Label(isi, image=gambar_segmentasi_tk, bg="#E8ECE7").pack(pady=(0, 14))

    # Persentase warna
    tk.Label(
        isi, text="Persentase warna",
        font=("Arial", 10, "bold"), bg=putih, fg=teks
    ).pack(anchor="w", pady=(0, 8))

    for warna, kode_warna in warna_tampilan.items():
        nilai = hasil_terakhir["persentase"][warna]

        baris = tk.Frame(isi, bg=putih)
        baris.pack(fill="x")

        tk.Label(baris, text=warna, font=("Arial", 10),
                 bg=putih, fg=teks).pack(side="left")
        tk.Label(baris, text=f"{nilai:.2f}%", font=("Arial", 10, "bold"),
                 bg=putih, fg=teks).pack(side="right")

        kanvas = tk.Canvas(isi, width=420, height=12,
                           bg="#E8ECE7", highlightthickness=0)
        kanvas.pack(fill="x", pady=(2, 10))

        panjang = min(max(nilai, 0), 100) / 100 * 420
        kanvas.create_rectangle(0, 0, panjang, 12, fill=kode_warna, outline="")

    tk.Frame(isi, bg="#DCE5DC", height=1).pack(fill="x", pady=8)

    tk.Label(
        isi, text="WARNA DOMINAN",
        font=("Arial", 9, "bold"), bg=putih, fg=abu
    ).pack(anchor="w", pady=(0, 4))

    tk.Label(
        isi, text=hasil_terakhir["dominan"],
        font=("Arial", 17, "bold"), bg=putih,
        fg=warna_tampilan[hasil_terakhir["dominan"]]
    ).pack(anchor="w")

    tk.Button(
        isi, text="TUTUP", command=jendela.destroy,
        font=("Arial", 10, "bold"), bg="#68756C", fg="white",
        activebackground="#4B554E", activeforeground="white",
        relief="flat", padx=16, pady=6, cursor="hand2"
    ).pack(pady=(14, 0))


# ==========================================
# 7. MEMILIH GAMBAR DAN MENAMPILKAN HASIL
# ==========================================

def pilih_gambar():
    global gambar_asli_tk, hasil_terakhir

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

        # Menyesuaikan gambar dengan panel
        foto_asli = Image.fromarray(asli_rgb)
        foto_asli.thumbnail((520, 340))

        gambar_asli_tk = ImageTk.PhotoImage(foto_asli)

        area_asli.configure(
            image=gambar_asli_tk,
            text="",
            width=0,
            height=0
        )

        # Simpan hasil untuk jendela analisis warna
        hasil_terakhir = {
            "segmentasi": segmentasi,
            "persentase": persentase,
            "dominan": dominan
        }

        label_kematangan.configure(
            text=kematangan.upper(),
            fg=warna_kematangan[kematangan]
        )

        tombol_analisis.configure(state="normal")

    except Exception as e:
        messagebox.showerror(
            "Gagal memproses gambar",
            str(e)
        )


def reset_hasil():
    global gambar_asli_tk, gambar_segmentasi_tk, hasil_terakhir

    gambar_asli_tk = None
    gambar_segmentasi_tk = None
    hasil_terakhir = None

    area_asli.configure(
        image="",
        text="Belum ada gambar",
        width=35,
        height=12
    )

    label_kematangan.configure(text="Belum dianalisis", fg=hijau_tua)
    tombol_analisis.configure(state="disabled")


# ==========================================
# 8. TOMBOL DAN KETERANGAN
# ==========================================

panel_bawah = tk.Frame(akar, bg="#F2F5F1")
panel_bawah.pack(fill="x", padx=20, pady=(0, 15))

panel_tombol = tk.Frame(
    panel_bawah,
    bg="#F2F5F1"
)
panel_tombol.pack()

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

tombol_analisis = tk.Button(
    panel_tombol,
    text="LIHAT ANALISIS WARNA",
    command=tampilkan_analisis,
    font=("Arial", 11, "bold"),
    bg="#C9A100",
    fg="white",
    activebackground="#9C7D00",
    activeforeground="white",
    disabledforeground="#EFEFEF",
    relief="flat",
    padx=18,
    pady=12,
    cursor="hand2",
    state="disabled"
)
tombol_analisis.pack(side="left", padx=5)

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


# ==========================================
# 9. MENJALANKAN APLIKASI
# ==========================================

akar.mainloop()