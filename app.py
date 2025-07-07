# 1. IMPORT LIBRARY PENTING
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from utils import preprocess_image # Asumsi nama fungsi ini tidak berubah

# 2. INISIALISASI & MEMUAT MODEL
app = Flask(__name__)
model = load_model('brain_tumor_model_finetuned.keras')

# GANTI INI DENGAN KELAS ANDA, SESUAI URUTAN
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# 3. ROUTE UNTUK HALAMAN UTAMA
@app.route('/')
def index():
    # Cukup tampilkan halaman HTML utama
    return render_template('index.html')

# 4. ROUTE UNTUK PREDIKSI
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file dari request
    file = request.files['file']
    
    # Jika ada file, proses
    if file:
        # Proses gambar langsung dari memori (bukan dari path file)
        # Pastikan fungsi preprocess_image Anda mendukung ini
        processed_image = preprocess_image(file)
        
        # Lakukan prediksi
        prediction = model.predict(processed_image)
        
        # Dapatkan hasil
        class_index = np.argmax(prediction)
        class_name = CLASS_NAMES[class_index]
        confidence = np.max(prediction) * 100
        
        result_text = f'Hasil: {class_name} ({confidence:.2f}%)'
        
        # Tampilkan lagi halaman utama dengan hasil prediksi
        return render_template('index.html', prediction_text=result_text)
    
    # Jika tidak ada file, kembali ke halaman utama
    return render_template('index.html')

# (Opsional tapi sangat disarankan) Untuk menjalankan di komputer lokal
if __name__ == '__main__':
    app.run(debug=True)
