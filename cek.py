import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Memuat model yang telah dilatih
model = tf.keras.models.load_model('path_to_your_model.h5')

# Fungsi untuk memproses gambar
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Sesuaikan ukuran dengan model Anda
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array /= 255.0  # Normalisasi
    return img_array

# Fungsi untuk melakukan prediksi
def predict_plant_disease(img_path):
    processed_image = load_and_preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# Contoh penggunaan
img_path = 'path_to_your_image.jpg'  # Ganti dengan path gambar Anda
result = predict_plant_disease(img_path)
print(f'Hasil deteksi: {result}')