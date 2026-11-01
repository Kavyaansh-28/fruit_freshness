import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.keras')

# Define class labels 
class_labels = ['freshapples', 'freshbanana', 'freshoranges',
                'rottenapples', 'rottenbanana', 'rottenoranges']

# Start the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame to 224x224 (for MobileNetV2 input)
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display prediction text
    text = f"{predicted_label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow("Fruit Quality Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
