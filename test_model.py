import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "waste_multiclass_model.keras"
IMAGE_PATH = "test.jpg"
IMG_SIZE = (224, 224)
GRID_SIZE = 3  # 3x3 patches

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["glass", "metal", "organic", "other", "paper", "plastic"]
# ⚠️ Make sure this matches training order

# -----------------------------
# Load Image
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

patch_predictions = []

# -----------------------------
# Split into patches
# -----------------------------
patch_width = width // GRID_SIZE
patch_height = height // GRID_SIZE

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        left = j * patch_width
        top = i * patch_height
        right = left + patch_width
        bottom = top + patch_height

        patch = image.crop((left, top, right, bottom))
        patch = patch.resize(IMG_SIZE)

        patch_array = np.array(patch, dtype=np.float32)
        patch_array = tf.keras.applications.mobilenet_v2.preprocess_input(patch_array)
        patch_array = np.expand_dims(patch_array, axis=0)

        prediction = model.predict(patch_array, verbose=0)[0]
        top_index = np.argmax(prediction)
        top_class = class_names[top_index]

        patch_predictions.append(top_class)

# -----------------------------
# Majority Vote
# -----------------------------
vote_count = Counter(patch_predictions)
final_prediction = vote_count.most_common(1)[0][0]

print("\nPatch Predictions:", patch_predictions)
print("Vote Distribution:", vote_count)
print("\nFINAL PREDICTION:", final_prediction)