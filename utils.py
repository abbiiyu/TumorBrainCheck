import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Muat model Keras
model = load_model('brain_tumor_model_finetuned.keras')

# Mapping indeks ke label, disesuaikan dengan class_indices saat training
label_map = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',    
    3: 'Pituitary'
}

def prediksi_tumor(image_path):
    # Buka dan preprocess gambar
    # img = Image.open(image_path).resize((224, 224))
    img = Image.open(image_path).convert("RGB").resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Lakukan prediksi
    hasil = model.predict(img_array)
    kelas = np.argmax(hasil, axis=1)[0]

    return label_map.get(kelas, 'Tidak Diketahui')

def ambil_deskripsi(label):
    deskripsi_dict = {
        'Glioma': 'Glioma adalah jenis tumor otak yang paling sering terjadi pada orang dewasa. Menurut American Association of Neurological, sekitar 78 persen dari total kasus tumor otak ganas tergolong sebagai glioma.',
        'Meningioma': 'Meningioma merupakan jenis tumor otak yang tumbuh pada meninges, yaitu lapisan pelindung yang menyelimuti otak dan sumsum tulang belakang. Tumor ini dapat muncul di berbagai area otak, namun paling sering ditemukan di sekitar otak besar (cerebrum) dan otak kecil (cerebellum).',
        'No Tumor': 'Tidak ditemukan tumor pada citra MRI.',
        'Pituitary': 'pituitari adalah tumor otak yang berkembang pada kelenjar pituitari, yaitu kelenjar yang mengontrol berbagai fungsi tubuh dan melepaskan hormon ke dalam aliran darah.'
    }
    return deskripsi_dict.get(label, 'Deskripsi tidak tersedia.')
