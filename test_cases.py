import unittest
from model import ImagePreprocessor, OnnxModel

class TestOnnxModel(unittest.TestCase):
    
    def setUp(self):
        # Create object for Image Pre Processor
        self.image_processor = ImagePreprocessor()
        # Load the ONNX model
        self.onnx_model = OnnxModel("onnx_model.onnx")

        return self.image_processor, self.onnx_model
        
    def test_image1(self):
        # Test with image1
        image_path = "images/n01440764_10218.JPEG"
        expected_class = 0
        result = self._classify_image(image_path)
        self.assertEqual(result, expected_class)
    
    def test_image2(self):
        # Test with image2
        image_path = "images/n01514859_10074.JPEG"
        expected_class = 8
        result = self._classify_image(image_path)
        self.assertEqual(result, expected_class)
    
    def test_image3(self):
        # Test with image3
        image_path = "images/n01558993_10351.JPEG"
        expected_class = 15
        result = self._classify_image(image_path)
        self.assertEqual(result, expected_class)
        
    def _classify_image(self, image_path):
        preprocessor, model = self.setUp()
        image = preprocessor.preprocess(image_path)
        pred_class_id = model.predict(image)
        return pred_class_id
        
if __name__ == '__main__':
    unittest.main()
