import torch
from pytorch_model import Classifier

# Create an instance of the PyTorch model class and load the pre-trained weights
model = Classifier()
model.load_state_dict(torch.load("pytorch_model_weights.pth"))

# Define the input shape for the ONNX model
input_shape = (1, 3, 224, 224)

# Create an example input tensor for the PyTorch model
example_input = torch.randn(*input_shape)

# Export the PyTorch model to the ONNX model
torch.onnx.export(model, example_input, "onnx_model.onnx", input_names=["input"], output_names=["output"])
