Fake Currency Detection using Deep Learning

Project Overview
This project focuses on detecting fake currency notes using deep learning techniques. A Convolutional Neural Network (CNN) with MobileNetV2 transfer learning is used to classify currency images as real or fake. The project also includes a Flask-based web application for easy deployment and testing.

Features

Preprocessing of currency images (resize, normalization, augmentation)

Model training with MobileNetV2 for high accuracy

Evaluation metrics such as accuracy, loss, F1-score, precision, recall, and confusion matrix

Flask web app for uploading currency images and checking authenticity

User-friendly frontend with HTML, CSS, and Bootstrap

Dataset

Contains images of real and fake currency notes

Images are preprocessed to a fixed size before training

Data is split into training, validation, and test sets

Installation

Clone the repository
git clone https://github.com/username/fake-currency-detection.git

cd fake-currency-detection

Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)

Install dependencies
pip install -r requirements.txt

Model Training

Run train_model.py to train the model
python train_model.py

Training logs include accuracy, loss, validation accuracy, validation loss

Final evaluation gives confusion matrix and F1-score

Deployment (Flask App)

Start the server
python app.py

Open browser and go to
http://127.0.0.1:5000

Upload an image of a currency note to check if it is real or fake

Project Structure

main.py : Entry point for model or preprocessing

train_model.py : Training script

app.py : Flask web application

static/ : Contains CSS and images for frontend

templates/ : HTML files for Flask frontend

dataset/ : Contains training and testing images

saved_model/ : Stores the trained model

Evaluation Metrics

Accuracy: Overall performance of model

Precision: Correct positive predictions / All positive predictions

Recall: Correct positive predictions / All actual positives

F1-Score: Balance between precision and recall

Confusion Matrix: Detailed breakdown of predictions

Future Improvements

Add support for multiple currencies (INR, USD, etc.)

Improve frontend with more responsive design

Deploy on cloud platforms like AWS, Heroku, or Render

Use larger and more diverse dataset for robustness

Contributing

Fork the repo

Create a new branch for your feature/fix

Submit a pull request

License
This project is licensed under the MIT License.
