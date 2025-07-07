import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from utils import dapatkan_hasil_prediksi

# Inisialisasi aplikasi dan folder upload
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Memuat model sekali saja
model = load_model('brain_tumor_model_finetuned.keras')
print("--- Model berhasil dimuat ---")

# Route untuk halaman utama (main.html)
@app.route('/')
def index():
    return render_template('main.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    
    if file and file.filename:
        # Mengamankan dan menyimpan file yang di-upload
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Memanggil fungsi prediksi dengan LOKASI FILE
        nama_kelas, skor, deskripsi_tumor = dapatkan_hasil_prediksi(model, filepath)
        
        # Buat teks hasil untuk variabel 'prediction' di HTML
        hasil_prediksi_text = f'{nama_kelas}'
        
        # Menampilkan halaman prediksi.html dan mengirimkan semua variabel yang dibutuhkan
        return render_template('prediksi.html', 
                               prediction=hasil_prediksi_text,  # <-- Sesuai dengan {{ prediction }}
                               description=deskripsi_tumor,    # <-- Sesuai dengan {{ description }}
                               filename=filename)               # <-- Sesuai untuk menampilkan gambar
    
    return render_template('main.html')

# Untuk menjalankan di komputer lokal
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')