import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Lokasi dataset otomatis mengikuti lokasi train.py
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "dataset")

print("Lokasi train.py:", base_dir)
print("Lokasi dataset:", data_dir)

# Cek apakah dataset ada
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Folder dataset tidak ditemukan: {data_dir}")

print("Dataset ditemukan di:", data_dir)
print("Kelas yang terbaca:", os.listdir(data_dir))

# Preprocessing + augmentasi
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

# Simpan label kelas
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# Model CNN sederhana
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Simpan model
model.save("model_media_cetak.h5")

print("Training selesai.")
print("Model tersimpan sebagai model_media_cetak.h5")
print("Label tersimpan sebagai class_indices.json")