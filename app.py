from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Import fungsi yang sudah kita perbaiki dari utils.py
from utils import dapatkan_hasil_prediksi

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- MEMUAT MODEL (HANYA DI SINI!) ---
# Ini adalah satu-satunya tempat model dimuat.
model = load_model('brain_tumor_model_finetuned.keras')
print("--- Model berhasil dimuat sekali saja saat aplikasi start ---")

# Route untuk halaman utama
@app.route('/')
def index():
    # Menampilkan halaman utama Anda
    return render_template('main.html') # <-- DIUBAH

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file dari form HTML
    file = request.files.get('file')
    
    if file and file.filename:
        # Panggil fungsi dari utils.py untuk mendapatkan semua hasil sekaligus
        # Kirim 'model' yang sudah kita load dan 'file' yang di-upload
        nama_kelas, skor, deskripsi = dapatkan_hasil_prediksi(model, file)
        
        # Buat teks hasil untuk ditampilkan
        result_text = f'Hasil Prediksi: {nama_kelas} ({skor:.2f}%)'
        
        # Menampilkan halaman HASIL PREDIKSI dengan data yang dikirimkan
        return render_template('prediksi.html',         # <-- DIUBAH
                               prediction_text=result_text,
                               description_text=deskripsi)
    
    # Jika tidak ada file, kembali ke halaman utama
    return render_template('main.html') # <-- DIUBAH

# Untuk menjalankan di komputer lokal
if __name__ == '__main__':
    app.run(debug=True)
