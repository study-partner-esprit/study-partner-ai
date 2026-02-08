import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from collections import deque, Counter

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

# --- Check for video path ---
if len(sys.argv) < 2:
    print("Usage: python test_video.py <video_path>")
    exit()

video_path = sys.argv[1]
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
    exit()

# --- Start video ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read failed")
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        # Resize frame for model
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0  # Match training preprocessing

        # Predict
        preds = model.predict(img_array, verbose=0)
        confidence = np.max(preds[0])
        pred_idx = np.argmax(preds[0])

        # Only update if confident enough
        if confidence > 0.7:  # Threshold for confidence
            last_pred_idx = pred_idx
            focus_scores.append(pred_idx)
        else:
            # Keep previous
            focus_scores.append(last_pred_idx)
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