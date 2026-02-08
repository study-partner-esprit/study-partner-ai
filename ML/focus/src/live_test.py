import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from collections import deque, Counter

# Force CPU usage to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Load the trained model ---
model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models', 'focus_model.h5')
model = load_model(model_path)

# Mapping indices to labels
label_map = {0: "Focused", 1: "Drifting", 2: "Lost"}

# Rolling window for Focus Score (last N frames)
ROLLING_WINDOW = 50
focus_scores = deque(maxlen=ROLLING_WINDOW)

# Frame skipping to reduce load
FRAME_SKIP = 5
frame_count = 0
last_pred_idx = 0  # Default to Focused

# --- Start webcam ---
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Set camera properties to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# Flush initial frames to clear buffer
for _ in range(5):
    cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed, attempting to reopen...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        # Re-set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        # Flush frames
        for _ in range(5):
            cap.read()
        continue

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        # Resize frame for model
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0  # Match training preprocessing

        # Predict
        preds = model.predict(img_array, verbose=0)
        print(f"Predictions: {preds[0]}")  # Debug: print probabilities
        pred_idx = np.argmax(preds[0])
        last_pred_idx = pred_idx

        # Add to rolling window
        focus_scores.append(pred_idx)
    else:
        pred_idx = last_pred_idx

    # Use mode of rolling window for stable prediction
    if len(focus_scores) > 0:
        pred_idx = Counter(focus_scores).most_common(1)[0][0]

    pred_label = label_map[pred_idx]

    # Compute rolling Focus Score
    # Example: % of frames in last window that are Focused
    focus_score_percent = (np.array(focus_scores) == 0).mean() * 100

    # Display on frame
    cv2.putText(frame, f"Focus: {pred_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Focus Score: {focus_score_percent:.1f}%", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Show frame
    cv2.imshow('Focus Monitor', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
