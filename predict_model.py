# =========================
# predict.py
# =========================

import cv2
import numpy as np
from keras.models import load_model

# -----------------------------
# 1️⃣ Load saved model
# -----------------------------
model = load_model("currency_detector_final.h5")
class_names = ["real", "fake"]  # Make sure this matches your dataset

# -----------------------------
# 2️⃣ Preprocess function
# -----------------------------
def preprocess_image(img_path, img_size=(224,224)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# 3️⃣ Predict function
# -----------------------------
def predict_currency(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    class_index = np.argmax(pred)
    print(f"Prediction: {class_names[class_index]} (Confidence: {pred[0][class_index]*100:.2f}%)")

# -----------------------------
# 4️⃣ Example usage
# -----------------------------
# Replace with your image path
image_path = "new_currency.jpg"
predict_currency(image_path)
