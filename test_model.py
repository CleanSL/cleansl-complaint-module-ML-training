import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="waste_multiclass_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess test image
img = Image.open("image.jpg").convert("RGB")
img = img.resize((224, 224))
img = np.array(img, dtype=np.float32)

# MobileNetV2 preprocessing
img = (img / 127.5) - 1.0
img = np.expand_dims(img, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(output)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

output = interpreter.get_tensor(output_details[0]['index'])[0]

print("Probabilities:", output)
print("Sum:", np.sum(output))
print("Predicted:", class_names[np.argmax(output)])