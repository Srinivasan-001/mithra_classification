import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Load Pre-trained Gender Classification Model
model = load_model("gender_model.h5", compile=False)  # Load your trained model
class_names = ["Female", "Male"]

# Knowledge Representation: Track Unique Faces
unique_men = set()
unique_women = set()

# Initialize Webcam
cap = cv2.VideoCapture(0)

def preprocess_face(face_img):
    """Preprocesses the face image before passing it to the model."""
    face_img = cv2.resize(face_img, (64, 64))  # Resize to model's input size
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Ensure correct color channels
    face_img = np.expand_dims(face_img / 255.0, axis=0)  # Normalize and expand dimensions
    return face_img

def classify_gender(face_img):
    """Predicts gender from face image."""
    face_img = preprocess_face(face_img)
    prediction = model.predict(face_img)

    # Handle output structure
    if isinstance(prediction, list):
        prediction = prediction[0]  # Extract array from list if needed
    prediction = np.array(prediction).flatten()  # Flatten if necessary
    gender = class_names[np.argmax(prediction)]  # Determine gender
    return gender

def is_new_face(existing_faces, new_face_box, threshold=30):
    """Checks if a detected face is new based on bounding box coordinates."""
    x, y, w_box, h_box = new_face_box
    for (ex, ey, ew, eh) in existing_faces:
        if abs(ex - x) < threshold and abs(ey - y) < threshold:
            return False  # Face already detected within proximity
    return True  # New face detected

# Start Webcam Feed
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Extract Face Region
                face_img = frame[y:y + h_box, x:x + w_box]
                if face_img.size == 0 or w_box < 10 or h_box < 10:
                    continue

                # Classify gender
                gender = classify_gender(face_img)

                # Check if it's a new face before updating counts
                if gender == "Male" and is_new_face(unique_men, (x, y, w_box, h_box)):
                    unique_men.add((x, y, w_box, h_box))
                elif gender == "Female" and is_new_face(unique_women, (x, y, w_box, h_box)):
                    unique_women.add((x, y, w_box, h_box))

                # Draw bounding box & label with background
                cv2.rectangle(frame, (x, y - 35), (x + w_box, y), (0, 0, 0), -1)
                cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # Display the unique counts
        cv2.putText(frame, f"Unique Male: {len(unique_men)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Unique Female: {len(unique_women)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow('Gender Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
