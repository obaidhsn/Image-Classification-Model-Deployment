# Image Classification Model Deployment
This project demonstrates the deployment of an image classification model using ONNX. The model is trained on the ImageNet dataset and can classify an input image into one of the 1000 classes in the dataset. The deployment includes the following components:

A PyTorch model implementation (pytorch_model.py)
A script to convert the PyTorch model to ONNX format (convert_to_onnx.py)
An ONNX model implementation (onnx_model.py) for loading and running the ONNX model
A script to test the ONNX model on CPU (test_onnx.py)
A Flask API implementation (main.py) for exposing the model via HTTP requests
Automated test cases to test the deployment (test_cases.py)

