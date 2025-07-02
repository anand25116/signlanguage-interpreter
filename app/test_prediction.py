import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model('ml_model/cnn_model.h5')

# Load class labels
labels = sorted(os.listdir('data/asl_alphabet_train/asl_alphabet_train'))

# Load and preprocess one test image
img_path = 'data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg'  # Change this path
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_label = labels[np.argmax(pred)]

print(f"🔤 Predicted: {predicted_label}")
