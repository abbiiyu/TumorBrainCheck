# Tambahkan print() di berbagai titik
print("--- [1] Aplikasi Mulai, import library ---")
from flask import Flask # dan import lainnya
from tensorflow.keras.models import load_model
import os

# Pastikan Anda menggunakan nama file yang benar
# MODEL_PATH = 'brain_tumor_model_finetuned.keras'
# Jika file ada di folder lain, sesuaikan pathnya. misal: 'models/model.keras'

print("--- [2] Inisialisasi aplikasi Flask ---")
app = Flask(__name__)

print("--- [3] AKAN MEMUAT MODEL. Ini adalah titik kritis. ---")
try:
    # Ganti 'nama_file_model_anda.keras' dengan nama file yang sebenarnya
    model = load_model('brain_tumor_model_finetuned.keras')
    print("--- [4] SELAMAT! Model berhasil dimuat. ---")
except Exception as e:
    print(f"--- [ERROR] GAGAL MEMUAT MODEL: {e} ---")

@app.route('/')
def index():
    print("--- [5] Request masuk ke route '/' ---")
    # Logika untuk halaman utama Anda
    return "Selamat datang di aplikasi Brain Tumor Check!"

# Tambahkan route lain jika ada
# @app.route('/predict', methods=['POST'])
# def predict():
#     print("--- [6] Request masuk ke route '/predict' ---")
#     # Logika prediksi Anda
#     return "Hasil Prediksi"

print("--- [7] Semua definisi route selesai. Aplikasi siap berjalan. ---")
