from ultralytics import YOLO
import cv2
import time
import os
MODEL_PATH = r"C:\Users\Kavya\OneDrive\Desktop\eai2\runs\detect\fruit_yolo_model\weights\best.pt"

# Verify model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading YOLO model:")
model = YOLO(MODEL_PATH)
print("YOLO model loaded!\n")

def get_iriun_camera():
    print("Detecting Iriun Webcam (iPhone 16 feed)...")
    for i in range(1, 6):  # Start from 1 → skip built-in cam
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if ret:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w >= 640 and h >= 480:  # iPhone feed is high-res
                print(f"IRIUN WEBCAM FOUND @ index {i} | {w}x{h}")
                return cap, i
        cap.release()
    raise RuntimeError("IRIUN WEBCAM NOT DETECTED!\n"
                       "1. Open Iriun Webcam on PC\n"
                       "2. Open Iriun app on iPhone 16\n"
                       "3. Connect to same WiFi or use USB")

try:
    cap, cam_id = get_iriun_camera()
except RuntimeError as e:
    print(e)
    input("Press Enter to exit...")
    exit()
print(f"\nStarting YOLO detection on iPhone 16 feed (index {cam_id})")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Reconnecting...")
        time.sleep(1)
        continue

    start_time = time.time()

    # Run YOLO inference
    results = model(frame, stream=True, verbose=False, imgsz=640)

    fruit_counts = {}

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0].item())
                label = model.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                fruit_counts[label] = fruit_counts.get(label, 0) + 1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display counts
    y_pos = 30
    for label, count in fruit_counts.items():
        cv2.putText(frame, f"{label}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30

    # FPS
    fps = 1.0 / (time.time() - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO Fruit Detection - iPhone 16 via Iriun", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped. Goodbye!")