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

# Knowledge Representation: Counter
men_count = 0


def classify_gender(face_img):
    """Predicts gender from face image"""
    face_img = cv2.resize(face_img, (64, 64))  # Resize to model's input size
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Ensure correct color channels
    face_img = np.expand_dims(face_img, axis=0)  # Expand dimensions
    face_img = face_img / 255.0  # Normalize

    # Predict gender
    prediction = model.predict(face_img)

    # Print prediction to check its structure
    print("Prediction (before reshaping):", prediction)

    # Handle the nested structure of prediction
    if isinstance(prediction, list) and isinstance(prediction[0], np.ndarray):
        prediction = prediction[0]  # Extract the array from the list

    # Ensure that prediction is a numpy array of the correct shape
    prediction = np.array(prediction)
    print("Prediction Shape (after conversion):", prediction.shape)

    # Flatten the prediction if needed
    if prediction.shape[0] > 1:
        prediction = prediction.flatten()  # Flatten if the shape is (2, 1) or similar

    gender = class_names[np.argmax(prediction)]  # Get the gender label from class_names
    return gender


# Start Webcam Feed
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                    bboxC.height * h)

                # Extract Face Region
                face_img = frame[y:y + h_box, x:x + w_box]
                if face_img.size == 0:
                    continue

                # Classify gender of the face
                gender = classify_gender(face_img)
                if gender == "Male":
                    men_count += 1

                # Draw bounding box and gender label
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the count of men detected
        cv2.putText(frame, f"Men Count: {men_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Gender Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

