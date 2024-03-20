import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from skimage import io, transform
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Set a secret key for the Flask application
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Path to the uploaded folder
UPLOAD_FOLDER = r'C:\Users\STUDENT\PycharmProjects\Medical_Image_Analysis\uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to the pre-trained model
MODEL_PATH = r'C:\Users\STUDENT\PycharmProjects\Medical_Image_Analysis\model\pre-trained\model.h5'


# Function to preprocess uploaded images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  # Convert RGBA to RGB format
    img = img.resize((224, 224))  # Resize image to match model input size
    img = np.array(img)  # Convert PIL image to numpy array
    img = img / 255.0  # Normalize pixel values
    return img


# Load the pre-trained model
model = load_model(MODEL_PATH)


# Define the routes
@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded file to the UPLOAD_FOLDER
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Preprocess the uploaded image
            image = preprocess_image(file_path)
            # Make prediction using the pre-trained model
            prediction = model.predict(np.expand_dims(image, axis=0))
            # Get the predicted class
            predicted_class = np.argmax(prediction)
            # Get the class label
            class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
            predicted_label = class_labels[predicted_class]
            return render_template('result.html', predicted_label=predicted_label, file_path=file_path)
    return render_template('upload.html')


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
