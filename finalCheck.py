import cv2
import mediapipe as mp
import requests
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
from threading import Thread, Lock


# Initialize Flask App
app = Flask(__name__)
gender_counts = {"Male": 0, "Female": 0}
gender_counts_lock = Lock()  # Lock for thread safety


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Our ESP32-CAM's IP Address
ESP32_CAM_IP = "192.168.156.198"
ESP32_CAM_STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"
ESP32_CAM_START_STREAM_URL = f"http://{ESP32_CAM_IP}/control?var=enable&val=1"

# To Start the ESP32-CAM stream automatically
print("Starting ESP32-CAM stream...")
requests.get(ESP32_CAM_START_STREAM_URL)
time.sleep(2)  # Allow time for the stream to start

# Loading Pre-trained Gender Classification Model
model = load_model("gender_model.h5", compile=False)  # Loading our trained model
class_names = ["Female", "Male"]

# Tracking Dictionaries
pending_faces = {}  # Temporarily stores detected faces for delay
tracked_faces = {}  # Stores confirmed unique faces

# Initialize Webcam
cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)


def preprocess_face(face_img):
    """Preprocesses the face image before passing it to the model."""
    face_img = cv2.resize(face_img, (64, 64))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = np.expand_dims(face_img / 255.0, axis=0)
    face_img = face_img / 255.0  # Normalize
    return face_img


def classify_gender(face_img):
    """Predicts gender from face image."""
    face_img = preprocess_face(face_img)
    prediction = model.predict(face_img)

    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = np.array(prediction).flatten()
    gender = class_names[np.argmax(prediction)]
    return gender


def is_new_face(face_id, gender, delay_time=0, expiration_time=150):
    """Delays counting a face until it's been detected consistently for a set time."""
    global pending_faces, tracked_faces
    current_time = time.time()

    # Remove expired face IDs
    tracked_faces = {k: v for k, v in tracked_faces.items() if current_time - v["timestamp"] < expiration_time}

    # Store the detected face temporarily
    if face_id not in pending_faces:
        pending_faces[face_id] = {"gender": gender, "timestamp": current_time}
        return False  # Not counted yet (waiting)

    # If face has been in pending state for `delay_time` seconds, move it to tracked_faces
    if current_time - pending_faces[face_id]["timestamp"] >= delay_time:
        tracked_faces[face_id] = pending_faces.pop(face_id)  # Confirm as unique
        return True  # Now counted as a unique face

    return False  # Still waiting to confirm uniqueness


def camera_process():
    global gender_counts
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
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                        bboxC.height * h)

                    # Extract Face Region
                    face_img = frame[y:y + h_box, x:x + w_box]
                    if face_img.size == 0 or w_box < 10 or h_box < 10:
                        continue

                    gender = classify_gender(face_img)
                    face_id = hash((x, y, w_box, h_box))  # Unique identifier for faces

                    if is_new_face(face_id, gender):
                        cv2.putText(frame, f"New {gender} detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                    # Draw bounding box & label with background
                    cv2.rectangle(frame, (x, y - 35), (x + w_box, y), (0, 0, 0), -1)
                    cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            # Count the number of unique males and females
            male_count = sum(1 for g in tracked_faces.values() if g["gender"] == "Male")
            female_count = sum(1 for g in tracked_faces.values() if g["gender"] == "Female")

            # Update the global gender_counts safely using the lock
            with gender_counts_lock:
                gender_counts["Male"] = male_count
                gender_counts["Female"] = female_count

            # Display unique counts
            cv2.putText(frame, f"Unique Male: {male_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Unique Female: {female_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                        2)

            cv2.imshow('Gender Classification', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


@app.route("/gender", methods=["GET"])
def get_gender():
    # Return the gender counts safely using the lock
    with gender_counts_lock:
        return jsonify(gender_counts)


if __name__ == "__main__":
    # Start camera thread
    Thread(target=camera_process, daemon=True).start()

    # Run Flask server
    app.run(host="0.0.0.0", port=5000)
