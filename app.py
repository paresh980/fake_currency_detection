import os
from flask import Flask, request, render_template, url_for
from keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


model = load_model("currency_detector_final.h5")
class_names = ["real", "fake"]

def preprocess_image(img_path, img_size=(224,224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict
    img = preprocess_image(filepath)
    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = round(float(pred[0][class_index]*100), 2)
    prediction = class_names[class_index]

    
    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_name=filename)

if __name__ == '__main__':
    app.run(debug=True)

