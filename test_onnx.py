from model import ImagePreprocessor, OnnxModel

# Create object for Image Pre Processor
image_processor = ImagePreprocessor()

# Load the ONNX model
onnx_model = OnnxModel("onnx_model.onnx")

# Load and preprocess the test images
image_files = ["n01440764_tench.JPEG", "n01667114_mud_turtle.JPEG"]
for image_file in image_files:
    image = image_processor.preprocess(image_file)
    pred_class_id = onnx_model.predict(image)
    
    with open('LOC_synset_mapping.txt', 'r') as f:
        class_ = image_file.split("_")[0]
        for line in f:
            if line.startswith(class_):
                line = line.strip()
                class_name = line.split(",")[0].split(" ", 1)[1]

    # Print the predicted class ID
    print(f"{image_file} belongs to class id {pred_class_id} with class name {class_name}")

    # Verify if the predicted class ID and is correct
    if image_file == "n01440764_tench.JPEG":
        assert pred_class_id == 0 and class_name == 'tench'
    elif image_file == "n01667114_mud_turtle.JPEG":
        assert pred_class_id == 35 and class_name == 'mud turtle'
