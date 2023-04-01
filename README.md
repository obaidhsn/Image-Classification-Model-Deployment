# Image Classification Model Deployment
This project demonstrates the deployment of an image classification model using ONNX. The model is trained on the ImageNet dataset and can classify an input image into one of the 1000 classes in the dataset. The deployment includes the following components:

A PyTorch model implementation (pytorch_model.py)
A script to convert the PyTorch model to ONNX format (convert_to_onnx.py)
An ONNX model implementation (onnx_model.py) for loading and running the ONNX model
A script to test the ONNX model on CPU (test_onnx.py)
A Flask API implementation (main.py) for exposing the model via HTTP requests
Automated test cases to test the deployment (test_cases.py)

## Model Details
The PyTorch model is designed to perform classification on an input image. It accepts an RGB image of size 224x224 and outputs an array with probabilities for each class in the ImageNet dataset. The length of the output array is equal to the number of classes (1000) in the dataset. The PyTorch model is trained on images with specific pre-processing steps, which include converting to RGB format (if needed), resizing to 224x224 using bilinear interpolation, dividing by 255, normalizing using mean and standard deviation values for each channel [RGB][0.485, 0.456, 0.406] and [0.229, 0.224, 0.225].

The pytorch_model.py file contains the PyTorch model implementation. The convert_to_onnx.py script can be used to convert the PyTorch model to ONNX format. The resulting ONNX model can be loaded and run using the onnx_model.py file. The test_onnx.py script can be used to test the ONNX model on CPU.

The main.py file contains a Flask implementation of a web API that exposes the model via HTTP requests. The API accepts an image file via a POST request and returns the class ID predicted by the model. The API runs the ONNX model in the backend.

The test_cases.py file contains automated test cases to test the deployment. These tests cover the PyTorch model implementation, the ONNX model implementation, and the web API implementation.

## Usage
1. To run the web API, simply run the following command:
```python
python3 main.py
```

2. Open your web browser and go to 'http://localhost:5000' to access the app.

## Features
To classify an image, click on the "Choose File" button and select an image from your computer.

Click the "Upload" button to upload the image.

The app will display the predicted class for the image.



