#!/usr/bin/env python3
"""
Advanced Fatigue Detection System
Combines the best features from both fatigue detection systems:

- MediaPipe face detection (modern, accurate, fast)
- Dlib-style EAR calculations (proven algorithm)
- Yawning detection (additional fatigue indicator)
- Multi-factor fatigue scoring (comprehensive analysis)
- Robust camera handling (production-ready)
- Calibration phase (personalized thresholds)
- Modular architecture (maintainable and extensible)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import time
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(__file__))

from modules.webcam import Webcam
from modules.face_features import FaceFeatures
from modules.fatigue_rules import FatigueRules
from config import (
    SHOW_GUI, WINDOW_NAME, CALIBRATION_FRAMES, CAMERA_TIMEOUT,
    FATIGUE_SCORE_THRESHOLD
)

def main():
    """Main fatigue detection loop."""
    print("=" * 60)
    print("🤖 ADVANCED FATIGUE DETECTION SYSTEM")
    print("=" * 60)
    print("Combining MediaPipe accuracy with robust camera handling")
    print("Features: Eye tracking, yawning detection, multi-factor analysis")
    print("=" * 60)

    try:
        # Initialize modules
        print("\n📹 Initializing camera...")
        webcam = Webcam()

        print("👤 Initializing face features...")
        face_features = FaceFeatures()

        print("🧠 Initializing fatigue analysis...")
        fatigue_rules = FatigueRules()

        # MediaPipe Face Landmarker setup
        print("🎯 Setting up MediaPipe face detection...")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        face_landmarker = FaceLandmarker.create_from_options(options)
        print("✅ All systems initialized successfully!")

        # Calibration phase
        print(f"\n🔧 Starting calibration ({CALIBRATION_FRAMES} frames)...")
        calibration_ear_values = []

        for i in range(CALIBRATION_FRAMES):
            frame_bgr, frame_rgb = webcam.get_frame()
            if frame_bgr is None:
                continue

            # Process frame for calibration
            h, w, _ = frame_bgr.shape
            scale = h / 460  # RESIZE_HEIGHT from config
            frame_resized = cv2.resize(frame_bgr, None, fx=1/scale, fy=1/scale)

            # Simple preprocessing for calibration
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.equalizeHist(gray)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB))
            results = face_landmarker.detect(mp_image)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks[0]

                # Extract eye landmarks for calibration
                left_eye_indices = [33, 159, 158, 133, 153, 144]
                right_eye_indices = [263, 386, 385, 362, 381, 380]

                left_eye = [face_landmarks[i] for i in left_eye_indices]
                right_eye = [face_landmarks[i] for i in right_eye_indices]

                left_ear = face_features.eye_aspect_ratio(left_eye)
                right_ear = face_features.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                calibration_ear_values.append(avg_ear)

            if (i + 1) % 20 == 0:
                print(f"Calibration: {i+1}/{CALIBRATION_FRAMES} frames processed")

        # Calibrate thresholds
        if face_features.calibrate(calibration_ear_values):
            print("✅ Calibration completed - personalized thresholds set")
        else:
            print("⚠️  Calibration failed - using default thresholds")

        # Main detection loop
        print(f"\n🚀 Starting fatigue detection... (Press 'q' to quit)")
        print("-" * 60)

        frame_count = 0
        start_time = time.time()

        while True:
            frame_bgr, frame_rgb = webcam.get_frame()
            if frame_bgr is None:
                print("❌ Camera feed lost, attempting recovery...")
                time.sleep(1)
                continue

            frame_count += 1
            current_time = time.time()

            # Resize frame for processing
            h, w, _ = frame_bgr.shape
            scale = h / 460
            frame_resized = cv2.resize(frame_bgr, None, fx=1/scale, fy=1/scale)

            # Preprocessing
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.equalizeHist(gray)

            # MediaPipe face detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB))
            results = face_landmarker.detect(mp_image)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks[0]

                # Scale landmarks back to original frame size
                scale_x, scale_y = w / adjusted.shape[1], h / adjusted.shape[0]

                # Extract facial features
                # LEFT EYE
                left_eye_indices = [33, 159, 158, 133, 153, 144]
                left_eye = []
                for i in left_eye_indices:
                    landmark = face_landmarks[i]
                    left_eye.append(type('Point', (), {
                        'x': landmark.x * scale_x,
                        'y': landmark.y * scale_y
                    })())

                # RIGHT EYE
                right_eye_indices = [263, 386, 385, 362, 381, 380]
                right_eye = []
                for i in right_eye_indices:
                    landmark = face_landmarks[i]
                    right_eye.append(type('Point', (), {
                        'x': landmark.x * scale_x,
                        'y': landmark.y * scale_y
                    })())

                # MOUTH (for yawning detection)
                # MediaPipe mouth landmarks: outer lips
                mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
                mouth = []
                for i in mouth_indices:
                    if i < len(face_landmarks):
                        landmark = face_landmarks[i]
                        mouth.append(type('Point', (), {
                            'x': landmark.x * scale_x,
                            'y': landmark.y * scale_y
                        })())

                # BROW points
                left_brow = (int(face_landmarks[105].x * w), int(face_landmarks[105].y * h))
                right_brow = (int(face_landmarks[334].x * w), int(face_landmarks[334].y * h))

                # JAW points
                top_jaw = (int(face_landmarks[13].x * w), int(face_landmarks[13].y * h))
                bottom_jaw = (int(face_landmarks[14].x * w), int(face_landmarks[14].y * h))

                # Feature extraction
                left_ear = face_features.eye_aspect_ratio(left_eye)
                right_ear = face_features.eye_aspect_ratio(right_eye)
                blink_rate = face_features.blink_detection(left_ear, right_ear)

                brow_dist = face_features.brow_distance(left_brow, right_brow)
                jaw_open = face_features.jaw_openness(top_jaw, bottom_jaw)

                # Yawning detection
                mar = face_features.mouth_aspect_ratio(mouth)
                yawning = face_features.yawning_detection(mar)

                # Fatigue analysis
                fatigue_prob, suggest_break, fatigue_level, factors = fatigue_rules.compute_fatigue_score(
                    (left_ear + right_ear) / 2, blink_rate, brow_dist, jaw_open, yawning
                )

                # Display results
                if SHOW_GUI:
                    # Status text
                    status_lines = [
                        f"FPS: {frame_count / (current_time - start_time):.1f}",
                        f"Fatigue: {fatigue_prob:.2f} ({fatigue_level})",
                        f"Blinks/min: {blink_rate}",
                        f"Yawns: {yawning}",
                        f"Trend: {fatigue_rules.get_fatigue_trend()}"
                    ]

                    if suggest_break:
                        status_lines.append("⚠️  TAKE A BREAK!")

                    # Draw status
                    y_offset = 30
                    for line in status_lines:
                        cv2.putText(frame_bgr, line, (20, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25

                    # Draw facial landmarks
                    for i in left_eye_indices + right_eye_indices:
                        landmark = face_landmarks[i]
                        cv2.circle(frame_bgr,
                                 (int(landmark.x * w), int(landmark.y * h)),
                                 2, (0, 255, 0), -1)

                    # Alert styling
                    if suggest_break:
                        # Red border for alerts
                        cv2.rectangle(frame_bgr, (0, 0), (w, h), (0, 0, 255), 5)

                    cv2.imshow(WINDOW_NAME, frame_bgr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # Console output mode
                    fps = frame_count / (current_time - start_time)
                    status = f"FPS: {fps:.1f} | Fatigue: {fatigue_prob:.2f} ({fatigue_level}) | Blinks: {blink_rate} | Yawns: {yawning}"

                    if suggest_break:
                        status += " | ⚠️  TAKE A BREAK!"

                    print(f"\r{status}", end="", flush=True)

                    # Small delay to prevent CPU hogging
                    time.sleep(0.01)
            else:
                # No face detected
                if SHOW_GUI:
                    cv2.putText(frame_bgr, "No face detected", (20, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(WINDOW_NAME, frame_bgr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    print("\rNo face detected", end="", flush=True)
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'webcam' in locals():
            webcam.release()
        if SHOW_GUI:
            cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()