import tensorflow as tf
import numpy as np
import cv2
import os
import json
from collections import deque
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained models
FACE_MODEL_PATH = "face_recognition_model.h5"
GAIT_MODEL_PATH = "gait_recognition_model.h5"
face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
gait_model = tf.keras.models.load_model(GAIT_MODEL_PATH)

# Load class labels
CLASS_LABELS = {0: "Asmita", 1: "Prajana"}

# Image size (must match training)
IMG_SIZE = 224  

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load metadata
metadata_dict = {}
for folder in os.listdir("images"):
    metadata_path = os.path.join("images", folder, "meta.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
            metadata_dict[folder.lower()] = metadata

# Prediction history (to smooth out fluctuations)
face_prediction_queue = deque(maxlen=5)
gait_prediction_queue = deque(maxlen=5)

def preprocess_frame(frame):
    """Preprocess a single frame for model prediction."""
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    return frame

def predict_face(face_roi):
    """Predicts the class of a detected face."""
    processed_frame = preprocess_frame(face_roi)
    prediction = face_model.predict(processed_frame)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_index, confidence

def predict_gait(frame):
    """Predicts the class based on gait."""
    processed_frame = preprocess_frame(frame)
    prediction = gait_model.predict(processed_frame)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_index, confidence

# Open video stream (Webcam or video file)
video_path = "./videos/sample.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    best_prediction = None
    best_confidence = 0

    # Face recognition
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        class_index, confidence = predict_face(face_roi)

        if confidence > best_confidence:
            best_prediction = (class_index, confidence)
            best_confidence = confidence

    # Gait recognition
    class_index, confidence = predict_gait(frame)
    if confidence > best_confidence:
        best_prediction = (class_index, confidence)
        best_confidence = confidence

    if best_prediction:
        class_index, confidence = best_prediction
        predicted_label = CLASS_LABELS.get(class_index, "Unknown")
        details = metadata_dict.get(predicted_label.lower(), {})

        # Display details
        text_x, text_y = 50, 50
        cv2.putText(frame, f"Name: {details.get('name', predicted_label)}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {details.get('age', 'Unknown')}", (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {details.get('gender', 'Unknown')}", (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Address: {details.get('address', 'Unknown')}", (text_x, text_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Email: {details.get('email', 'Unknown')}", (text_x, text_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Location: {details.get('location', 'Unknown')}", (text_x, text_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (text_x, text_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show video feed with predictions
    cv2.imshow("Missing Person Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
