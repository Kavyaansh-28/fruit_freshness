import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model('best_model.keras')

# Class labels
class_labels = ['freshapple', 'rottenapple', 'freshbanana', 'rottenbanana', 'freshorange', 'rottenorange']

# ---- CAMERA DETECTION ----
def get_iriun_camera():
    print("Detecting Iriun Webcam (iPhone feed)...")
    for i in range(1, 6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width >= 640 and height >= 480:
                    print(f"✅ Iriun feed found at index {i}")
                    return cap, i
            cap.release()
    raise RuntimeError(" Iriun Webcam not detected. Make sure it's open and connected via WiFi.")

try:
    cap, cam_id = get_iriun_camera()
except RuntimeError as e:
    print(f"ERROR: {e}")
    exit()

print("Press 'q' to quit.\n")

# ---- DETECTION LOOP ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Optional: reduce noise and normalize lighting
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # adjust contrast & brightness

    # Convert to HSV for masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 30])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_dict = {label: 0 for label in class_labels}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:
            continue  # skip small blobs

        x, y, w, h = cv2.boundingRect(contour)
        fruit_roi = frame[y:y + h, x:x + w]
        if fruit_roi.size == 0:
            continue

        # Preprocess
        fruit_resized = cv2.resize(fruit_roi, (224, 224))
        img_array = img_to_array(fruit_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array, verbose=0)
        label_idx = np.argmax(preds)
        confidence = float(preds[0][label_idx])
        label = class_labels[label_idx]

        # Skip low-confidence predictions
        if confidence < 0.80:
            continue

        # Count and annotate
        count_dict[label] += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display counts
    y_offset = 30
    for label, count in count_dict.items():
        if count > 0:
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

    cv2.imshow("Fruit Detection (Iriun Feed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Feed closed. Detection complete.")
