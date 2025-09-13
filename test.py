import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model("currency_detector_final.h5")
class_names = ["real", "fake"]  # match your dataset

def preprocess_image(img_path, img_size=(224,224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict
img_path = "dataset/test" # change to your test image path
img = preprocess_image(img_path)
pred = model.predict(img)
class_index = np.argmax(pred)
print(f"Prediction: {class_names[class_index]} (Confidence: {pred[0][class_index]*100:.2f}%)")
