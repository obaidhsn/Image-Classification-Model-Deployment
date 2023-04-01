import numpy as np
import onnxruntime
from torchvision import transforms
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((224, 224), resample=Image.BILINEAR)
        img_tensor = self.transform(img_resized)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.numpy()

class OnnxModel:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
    
    def predict(self, input_data):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        pred = self.session.run([output_name], {input_name: input_data})
        return np.argmax(pred[0])

if __name__ == '__main__':
    imp = ImagePreprocessor()
    onnx = OnnxModel("onnx_model.onnx")
    image = imp.preprocess("n01667114_mud_turtle.JPEG")
    results = onnx.predict(image)
    print(results)