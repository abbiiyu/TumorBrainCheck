from PIL import Image
import numpy as np

# Mapping label, ini sudah benar.
label_map = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',    
    3: 'Pituitary'
}

def ambil_deskripsi(label):
    # Fungsi ini juga sudah benar, tidak perlu diubah.
    deskripsi_dict = {
        'Glioma': 'Glioma adalah jenis tumor otak yang paling sering terjadi pada orang dewasa...',
        'Meningioma': 'Meningioma merupakan jenis tumor otak yang tumbuh pada meninges...',
        'No Tumor': 'Tidak ditemukan indikasi adanya tumor pada citra MRI yang diunggah.',
        'Pituitary': 'Tumor pituitari adalah tumor otak yang berkembang pada kelenjar pituitari...'
    }
    return deskripsi_dict.get(label, 'Deskripsi tidak tersedia.')

def dapatkan_hasil_prediksi(model, filepath):
    """
    Fungsi utama untuk prediksi.
    Menerima: objek model Keras dan LOKASI FILE GAMBAR (filepath).
    """
    # Membuka gambar dari LOKASI FILE (bukan stream)
    img = Image.open(filepath).convert("RGB").resize((224, 224))

    # Preprocessing gambar
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Lakukan prediksi menggunakan model yang diberikan
    hasil_prediksi = model.predict(img_array)

    # Dapatkan hasilnya
    kelas_index = np.argmax(hasil_prediksi, axis=1)[0]
    skor_kepercayaan = np.max(hasil_prediksi) * 100

    # Ambil nama kelas dan deskripsi
    nama_kelas = label_map.get(kelas_index, 'Tidak Diketahui')
    deskripsi = ambil_deskripsi(nama_kelas)

    return nama_kelas, skor_kepercayaan, deskripsi