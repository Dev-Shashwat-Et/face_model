import cv2
import json
import os
import numpy as np
from keras.models import load_model

# Load face detector and classification model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('face_recognition_model.h5')

# Video source
video_path = './videos/asmita01.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

# Load all available metadata
metadata_dict = {}
for folder in os.listdir('images'):
    metadata_path = os.path.join('images', folder, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
            metadata_dict[folder] = metadata

# Helper function for mapping class index to details
def get_details(classNo):
    person_key = list(metadata_dict.keys())[classNo]
    return metadata_dict[person_key]

while True:
    success, imgOriginal = cap.read()
    if not success:
        break
    
    # Ensure the image is correctly oriented (rotate if needed)
    imgOriginal = cv2.rotate(imgOriginal, cv2.ROTATE_90_CLOCKWISE)

    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis=1)[0]
        probabilityValue = np.amax(prediction)

        # Fetch details from metadata
        details = get_details(classIndex)

        # Draw bounding box and add details
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imgOriginal, (x, y - 120), (x + w, y), (0, 255, 0), -1)
        
        # Display information correctly aligned
        text_y = y - 100
        info = [
            f"Name: {details['name']}",
            f"Age: {details['age']}",
            f"Gender: {details['gender']}",
            f"Address: {details['address']}",
            f"Email: {details['email']}",
            f"Location: {details['location']}",
            f"Confidence: {round(probabilityValue * 100, 2)}%"
        ]

        for text in info:
            cv2.putText(imgOriginal, text, (x, text_y), font, 0.5, (255, 255, 255), 1)
            text_y += 15

    frame_resized = cv2.resize(imgOriginal, (1000, 1000))
    cv2.imshow("face recognition", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
