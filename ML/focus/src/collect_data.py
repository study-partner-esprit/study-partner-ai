import cv2
import numpy as np
import os
import time

# --- Start webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# Create directories if not exist
os.makedirs('data/raw/Focused', exist_ok=True)
os.makedirs('data/raw/Drifting', exist_ok=True)
os.makedirs('data/raw/Lost', exist_ok=True)

frame_count = 0

print("Press 'f' to save as Focused, 'd' for Drifting, 'l' for Lost, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    cv2.putText(frame, "Press f/d/l to label, q to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('f'):
        filename = f"data/raw/Focused/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved Focused: {filename}")
        frame_count += 1
    elif key == ord('d'):
        filename = f"data/raw/Drifting/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved Drifting: {filename}")
        frame_count += 1
    elif key == ord('l'):
        filename = f"data/raw/Lost/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved Lost: {filename}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

print("Data collection complete. Now update labels.csv and retrain.")