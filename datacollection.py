import cv2
import os
import json

# Video source
video = cv2.VideoCapture(0)
#video_path = './videos/asmita01.mp4'
#video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Load face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
frame_skip = 2  # Process every 2nd frame to balance speed and accuracy
frame_count = 0

# Collect user details
nameID = input("Enter Your Name: ").strip().lower()
ageID = int(input("Enter Your Age: "))
genderID = input("Enter Your Gender (M/F): ").strip().upper()
addressID = input("Enter Your Address: ").strip()
emailID = input("Enter Your Email ID: ").strip()
locationID = input("Enter Your Location: ").strip()

# Ensure unique directory
path = f'images/{nameID}'
while os.path.exists(path):
    print("Name Already Taken")
    nameID = input("Enter Your Name Again: ").strip().lower()
    path = f'images/{nameID}'

os.makedirs(path)

# Create metadata dictionary
metadata = {
    "name": nameID,
    "age": ageID,
    "gender": genderID,
    "address": addressID,
    "email": emailID,
    "location": locationID,
    "images": []
}

# Loop until we capture 250 images
while count < 250:
    ret, frame = video.read()

    # Restart video if it ends
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1

    # Skip frames for efficiency
    if frame_count % frame_skip != 0:
        continue

    # Detect faces
    faces = facedetect.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    for (x, y, w, h) in faces:
        count += 1
        image_name = f"{path}/{count}.jpg"
        print(f"Creating Image: {image_name}")

        # Save cropped face
        face_crop = frame[y:y + h, x:x + w]
        cv2.imwrite(image_name, face_crop)
        metadata["images"].append(image_name)

        # Draw rectangle on detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Stop if we reach 250 images
        if count >=500:
            break

    # Show the frame with face detection
    cv2.imshow("Face Capture", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save metadata to a JSON file
with open(f"{path}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Release resources
video.release()
cv2.destroyAllWindows()

print(f"✅ Data collection complete! {count} images saved in {path}.")
