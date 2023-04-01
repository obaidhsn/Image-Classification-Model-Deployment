import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
from model import ImagePreprocessor, OnnxModel
from pathlib import Path

# Create object for Image Pre Processor
image_processor = ImagePreprocessor()

# Load the ONNX model
onnx_model = OnnxModel("onnx_model.onnx")

app = Flask(__name__)
app.config['IMAGES_FOLDER'] = 'images'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'JPEG', "PNG"}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Process the image
            image = os.path.join(app.config['IMAGES_FOLDER'], file.filename)
            image = image_processor.preprocess(image)
            # Make the prediction
            prediction = onnx_model.predict(image)
            # Return the result
            return render_template('index.html', predicted_class=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
