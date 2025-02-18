import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Path dataset
train_img_dir = "dataset/train/img"

# Menyiapkan dataset
data = []
labels = []

# Mapping label (jika label ada di nama file)
label_mapping = {
    "grayleafspot": 0,
    "bacterial_leaf_blight": 1,
    "brown_spot": 2
}

# Iterasi gambar langsung dalam folder
for img_name in os.listdir(train_img_dir):
    img_path = os.path.join(train_img_dir, img_name)

    # Ambil label dari nama file (sesuaikan dengan dataset Anda)
    label = None
    for key in label_mapping:
        if key in img_name.lower():  # Cari nama kategori dalam nama file
            label = label_mapping[key]
            break

    if label is None:
        print(f"Warning: Label tidak ditemukan untuk {img_name}, dilewati...")
        continue  # Lewati gambar yang tidak memiliki label

    try:
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        data.append(image)
        labels.append(label)
    except:
        print(f"Error membaca {img_path}, melewati...")

# Konversi ke array numpy dan normalisasi
data = np.array(data) / 255.0
labels = np.array(labels)

# Pastikan dataset tidak kosong
if len(data) == 0:
    raise ValueError("Dataset kosong! Pastikan gambar memiliki label yang dikenali.")

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_mapping), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping untuk mencegah overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stop])

# Simpan model
model.save('leaf_disease_model.h5')

print("Model berhasil disimpan!")
