import cv2
import json
import subprocess
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model_media_cetak.h5")

# =========================
# LOAD LABEL KELAS
# =========================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}
print("Label kelas:", labels)

# Kelas yang boleh ditampilkan dan disuarakan
target_labels = ["koran", "majalah", "novel"]

# =========================
# FUNGSI SUARA WINDOWS
# =========================
def speak(text):
    print("SUARA:", text)

    safe_text = text.replace("'", "''")

    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speak.Rate = 0; "
        "$speak.Volume = 100; "
        f"$speak.Speak('{safe_text}')"
    )

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# =========================
# BUKA KAMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka.")
    exit()

# =========================
# SETTING
# =========================
threshold = 0.75
stable_frame_limit = 8

last_prediction = ""
stable_count = 0
last_announced = ""

# =========================
# LOOP KAMERA
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame kamera tidak terbaca.")
        break

    # Preprocessing gambar kamera
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi
    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction)
    confidence = float(np.max(prediction))

    hasil = labels[class_id]

    # =========================
    # HANYA DETEKSI KORAN / MAJALAH / NOVEL
    # =========================
    if hasil in target_labels and confidence >= threshold:
        display_text = f"{hasil} ({confidence:.2f})"

        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        print("Prediksi:", display_text)

        # Cek stabilitas prediksi
        if hasil == last_prediction:
            stable_count += 1
        else:
            last_prediction = hasil
            stable_count = 1

        # Suara hanya sekali saat objek berubah
        if stable_count >= stable_frame_limit and hasil != last_announced:
            speak(f"Ini adalah {hasil}")
            last_announced = hasil

    else:
        # Kalau bukan koran/majalah/novel, diam saja
        last_prediction = ""
        stable_count = 0

    cv2.imshow("Deteksi Media Cetak", frame)

    # Tekan q atau Esc untuk keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()