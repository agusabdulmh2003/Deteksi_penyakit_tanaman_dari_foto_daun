import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model yang sudah dilatih
model_path = "leaf_disease_model.h5"

if not os.path.exists(model_path):
    st.error("Model tidak ditemukan! Pastikan Anda telah melatih model dan menyimpannya sebagai leaf_disease_model.h5.")
    st.stop()

model = load_model(model_path)

# Definisikan kelas penyakit berdasarkan folder dataset
train_img_dir = "dataset/train/img"
class_names = sorted(os.listdir(train_img_dir))

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.title("🌿 Deteksi Penyakit Daun")
st.write("Unggah gambar daun untuk mengetahui jenis penyakitnya.")

# File uploader
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocess gambar
    image_array = preprocess_image(uploaded_file)
    
    # Prediksi
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Tampilkan hasil
    st.image(uploaded_file, caption=f"Deteksi: {predicted_class}", use_column_width=True)

    # Solusi berdasarkan prediksi
    solutions = {
        "Penyakit A": "Gunakan pestisida X.",
        "Penyakit B": "Cuci daun dengan air sabun.",
        "Penyakit C": "Ganti tanah dan periksa akar.",
        "Sehat": "Tanaman dalam kondisi sehat."
    }

    solusi_text = solutions.get(predicted_class, "Tidak ada solusi tersedia.")
    st.write(f"🩺 Solusi: {solusi_text}")
