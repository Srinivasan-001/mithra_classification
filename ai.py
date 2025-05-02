import cv2
from deepface import DeepFace
import time

# Use RetinaFace for accuracy
DETECTOR_BACKEND = 'retinaface'
CONFIDENCE_THRESHOLD = 0.9  # Accept predictions with >90% confidence

# Start video capture
cap = cv2.VideoCapture(0)
print("Improved Gender Detection with Accuracy Filtering Started...")

frame_skip = 5  # Skip frames for better performance
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    start_time = time.time()

    try:
        results = DeepFace.analyze(
            frame,
            actions=["gender"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        continue

    if not isinstance(results, list):
        results = [results]

    men_count = 0
    women_count = 0

    for res in results:
        gender = res['dominant_gender']
        gender_probs = res['gender']

        confidence = gender_probs[gender]
        if confidence < CONFIDENCE_THRESHOLD * 100:
            continue  # Skip low-confidence predictions

        # Correctly extract bounding box coordinates
        x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']

        if gender == 'Man':
            men_count += 1
            color = (255, 0, 0)
        else:
            women_count += 1
            color = (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{gender} ({int(confidence)}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show counts
    cv2.putText(frame, f"Men: {men_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Women: {women_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Accurate Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
