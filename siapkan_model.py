"""
Menyiapkan model pengenal objek untuk aplikasi pisang. Cukup dijalankan SEKALI.
TIDAK memerlukan torch/tensorflow: model ONNX diunduh langsung, lalu diperiksa
dengan OpenCV (yang sudah dipakai aplikasi).

Hasilnya dua berkas di folder yang sama dengan skrip ini:
    mobilenetv2-12.onnx     model MobileNetV2 (ImageNet), sekitar 14 MB
    imagenet_labels.txt     nama 1000 kelas ImageNet, satu per baris (opsional)

Butuh internet saat dijalankan.
"""

import os
import sys
import urllib.request

import cv2
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
FILE_MODEL = os.path.join(FOLDER, "mobilenetv2-12.onnx")
FILE_LABEL = os.path.join(FOLDER, "imagenet_labels.txt")

# Alamat github.com/.../raw/ dipakai (bukan raw.githubusercontent.com) karena
# berkas model disimpan dengan Git LFS; alamat itu otomatis mengarah ke
# berkas aslinya, bukan ke berkas penunjuk kecil.
URL_MODEL = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
]
URL_LABEL = [
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "https://raw.githubusercontent.com/pytorch/hub/main/imagenet_classes.txt",
]
UKURAN_MIN_MODEL = 5_000_000      # berkas penunjuk LFS hanya ~130 byte


def valid_model(data):
    return len(data) >= UKURAN_MIN_MODEL


def valid_label(data):
    baris = [b for b in data.decode("utf-8", "ignore").splitlines() if b.strip()]
    return len(baris) == 1000


def unduh(daftar_url, tujuan, valid):
    """Mencoba tiap alamat sampai ada yang berhasil dan isinya sesuai."""
    for url in daftar_url:
        print(f"  mengunduh: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except Exception as e:
            print(f"    gagal: {e}")
            continue
        if not valid(data):
            print(f"    isi tidak sesuai ({len(data)} byte), dilewati")
            continue
        with open(tujuan, "wb") as f:
            f.write(data)
        print(f"    tersimpan: {tujuan} ({len(data) / 1e6:.1f} MB)")
        return True
    return False


def periksa_model():
    """Memastikan OpenCV bisa membaca model dan keluarannya 1000 kelas."""
    net = cv2.dnn.readNetFromONNX(FILE_MODEL)
    net.setInput(np.zeros((1, 3, 224, 224), np.float32))
    keluaran = net.forward().size
    print(f"  keluaran model: {keluaran} nilai")
    return keluaran in (1000, 1001)


def main():
    print("1. Model MobileNetV2")
    if not unduh(URL_MODEL, FILE_MODEL, valid_model):
        print(
            "\nGagal mengunduh otomatis. Unduh manual berkas mobilenetv2-12.onnx "
            "dari\nhttps://github.com/onnx/models (folder validated/vision/"
            "classification/mobilenet/model),\nlalu simpan di folder ini dengan "
            "nama mobilenetv2-12.onnx"
        )
        sys.exit(1)

    print("2. Daftar nama kelas (opsional)")
    if not unduh(URL_LABEL, FILE_LABEL, valid_label):
        print("  dilewati: program tetap jalan, nama kelas tampil sebagai nomor.")

    print("3. Pemeriksaan dengan OpenCV")
    try:
        ok = periksa_model()
    except cv2.error as e:
        print(f"  OpenCV tidak bisa membaca model: {e}")
        sys.exit(1)
    if not ok:
        print("  keluaran model tidak 1000 kelas; berkas ini tidak cocok.")
        sys.exit(1)

    if os.path.isfile(FILE_LABEL):
        with open(FILE_LABEL, encoding="utf-8") as f:
            kelas = [b.strip() for b in f if b.strip()]
        print(f"  kelas nomor 954 = {kelas[954]!r} (harus 'pisang')")
    print("\nSelesai. Jalankan aplikasi_pisang_pyqt.py.")


if __name__ == "__main__":
    main()