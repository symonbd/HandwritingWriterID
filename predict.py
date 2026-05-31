import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import json




def predict_writer(image_path, model_path, label_map_path=None, top_k=3):
"""Predict the most likely writer(s) for a single handwriting image.
Returns list of (writer_label, confidence) sorted by confidence desc.
"""
model = load_model(model_path)


img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
raise FileNotFoundError(f"Image not found: {image_path}")


# Prepare image
input_shape = model.input_shape[1:3]
img_resized = cv2.resize(img, input_shape)
img_resized = img_resized.astype("float32") / 255.0
if len(model.input_shape) == 4 and model.input_shape[-1] == 1:
img_resized = np.expand_dims(img_resized, axis=-1)
img_resized = np.expand_dims(img_resized, axis=0)


preds = model.predict(img_resized)[0]
top_idx = preds.argsort()[::-1][:top_k]


# Load label map if provided
label_map = None
if label_map_path and os.path.exists(label_map_path):
with open(label_map_path, 'r') as f:
label_map = json.load(f)


results = []
for i in top_idx:
label = label_map.get(str(i), str(i)) if label_map else str(i)
results.append((label, float(preds[i])))


return results