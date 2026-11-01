import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model('best_model.keras')

# Define class labels 
class_labels = ['freshapple', 'rottenapple', 'freshbanana', 'rottenbanana', 'freshorange', 'rottenorange']

def get_camera_source():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Using camera index {i}")
                return i
        cap.release()
    print("⚠️ No camera detected, defaulting to index 0.")
    return 0

camera_index = get_camera_source()
cap = cv2.VideoCapture(camera_index)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize and convert
    frame_resized = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Basic color filtering (adjust as needed)
    lower = np.array([0, 40, 40])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_dict = {label: 0 for label in class_labels}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(contour)
            fruit_roi = frame_resized[y:y+h, x:x+w]
            fruit_resized = cv2.resize(fruit_roi, (224, 224))
            img_array = img_to_array(fruit_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array, verbose=0)
            label = class_labels[np.argmax(preds)]
            confidence = np.max(preds)

            count_dict[label] += 1

            # Bounding box + label
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display counts
    y_offset = 30
    for label, count in count_dict.items():
        if count > 0:
            cv2.putText(frame_resized, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

    cv2.imshow("Fruit Detection & Counting", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
