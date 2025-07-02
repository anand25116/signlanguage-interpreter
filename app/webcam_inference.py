import cv2
import numpy as np
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model('ml_model/cnn_model.h5')
labels = sorted(os.listdir('data/asl_alphabet_train/asl_alphabet_train'))

# Constants
IMG_SIZE = 64
ROI_BOX = (100, 100, 300, 300)

# Webcam init
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI
    x1, y1, x2, y2 = ROI_BOX
    roi = frame[y1:y2, x1:x2]

    # Show the ROI separately for debugging
    cv2.imshow("ROI", roi)

    # Preprocess ROI to match training format
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    confidence = np.max(preds)
    label = labels[np.argmax(preds)] if confidence > 0.7 else "❓ Uncertain"

    # Debug print
    print(f"🔍 Confidence: {confidence:.2f} | Prediction: {label}")

    # Show prediction on screen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f'Prediction: {label}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Interpreter (CNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
