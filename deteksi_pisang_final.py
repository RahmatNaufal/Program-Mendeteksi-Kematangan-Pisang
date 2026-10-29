import cv2
import numpy as np

# ==========================================
# 1. MEMASUKKAN NAMA GAMBAR
# ==========================================

nama_file = input("Masukkan nama gambar: ")

gambar = cv2.imread(nama_file)

if gambar is None:
    print("Gambar tidak ditemukan!")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================

gambar = cv2.resize(gambar, (600, 400))

blur = cv2.GaussianBlur(
    gambar,
    (5, 5),
    0
)

# ==========================================
# 3. KONVERSI BGR KE HSV
# ==========================================

hsv = cv2.cvtColor(
    blur,
    cv2.COLOR_BGR2HSV
)

# ==========================================
# 4. SEGMENTASI OBJEK PISANG
# ==========================================

saturation = hsv[:, :, 1]

_, mask_pisang = cv2.threshold(
    saturation,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ==========================================
# 5. OPERASI MORFOLOGI
# ==========================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

mask_pisang = cv2.morphologyEx(
    mask_pisang,
    cv2.MORPH_OPEN,
    kernel
)

mask_pisang = cv2.morphologyEx(
    mask_pisang,
    cv2.MORPH_CLOSE,
    kernel
)

# ==========================================
# 6. MENCARI KONTUR
# ==========================================

kontur, _ = cv2.findContours(
    mask_pisang,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(kontur) == 0:
    print("Objek pisang tidak ditemukan!")
    exit()

kontur_terbesar = max(
    kontur,
    key=cv2.contourArea
)

# ==========================================
# 7. MEMBUAT MASK OBJEK PISANG
# ==========================================

mask_objek = np.zeros_like(mask_pisang)

cv2.drawContours(
    mask_objek,
    [kontur_terbesar],
    -1,
    255,
    thickness=cv2.FILLED
)

# ==========================================
# 8. HASIL SEGMENTASI
# ==========================================

hasil_segmentasi = cv2.bitwise_and(
    blur,
    blur,
    mask=mask_objek
)

# ==========================================
# 9. THRESHOLDING WARNA
# ==========================================

# Warna hijau
lower_hijau = np.array([35, 60, 40])
upper_hijau = np.array([85, 255, 255])

# Warna kuning
lower_kuning = np.array([20, 80, 100])
upper_kuning = np.array([34, 255, 255])

# Warna cokelat
lower_cokelat = np.array([5, 60, 20])
upper_cokelat = np.array([19, 255, 200])

mask_hijau = cv2.inRange(
    hsv,
    lower_hijau,
    upper_hijau
)

mask_kuning = cv2.inRange(
    hsv,
    lower_kuning,
    upper_kuning
)

mask_cokelat = cv2.inRange(
    hsv,
    lower_cokelat,
    upper_cokelat
)

# ==========================================
# 10. MASKING WARNA
# ==========================================

mask_hijau = cv2.bitwise_and(
    mask_hijau,
    mask_objek
)

mask_kuning = cv2.bitwise_and(
    mask_kuning,
    mask_objek
)

mask_cokelat = cv2.bitwise_and(
    mask_cokelat,
    mask_objek
)

# ==========================================
# 11. MENGHITUNG JUMLAH PIKSEL
# ==========================================

total_piksel_pisang = cv2.countNonZero(
    mask_objek
)

jumlah_hijau = cv2.countNonZero(
    mask_hijau
)

jumlah_kuning = cv2.countNonZero(
    mask_kuning
)

jumlah_cokelat = cv2.countNonZero(
    mask_cokelat
)

# ==========================================
# 12. MENGHITUNG PERSENTASE
# ==========================================

persen_hijau = (
    jumlah_hijau / total_piksel_pisang
) * 100

persen_kuning = (
    jumlah_kuning / total_piksel_pisang
) * 100

persen_cokelat = (
    jumlah_cokelat / total_piksel_pisang
) * 100

# ==========================================
# 13. MENENTUKAN WARNA DOMINAN
# ==========================================

if persen_hijau >= persen_kuning and persen_hijau >= persen_cokelat:

    warna_dominan = "Hijau"
    kematangan = "Belum Matang"

elif persen_kuning >= persen_hijau and persen_kuning >= persen_cokelat:

    warna_dominan = "Kuning"
    kematangan = "Matang"

else:

    warna_dominan = "Cokelat"
    kematangan = "Terlalu Matang"

# ==========================================
# 14. MENAMPILKAN HASIL DI TERMINAL
# ==========================================

print()
print("========================================")
print("      DETEKSI KEMATANGAN PISANG")
print("========================================")

print("Nama gambar :", nama_file)

print()
print("Persentase Warna")
print("----------------------------------------")
print(f"Hijau   : {persen_hijau:.2f}%")
print(f"Kuning  : {persen_kuning:.2f}%")
print(f"Cokelat : {persen_cokelat:.2f}%")

print()
print("Hasil Deteksi")
print("----------------------------------------")
print(f"Warna Dominan : {warna_dominan}")
print(f"Kematangan    : {kematangan}")

print("========================================")

# ==========================================
# 15. MENAMPILKAN HASIL SEGMENTASI
# ==========================================

cv2.imshow(
    "Gambar Asli",
    gambar
)

cv2.imshow(
    "Hasil Segmentasi",
    hasil_segmentasi
)

cv2.waitKey(0)
cv2.destroyAllWindows()