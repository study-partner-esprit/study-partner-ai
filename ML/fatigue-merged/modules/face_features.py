from math import hypot
import time
import numpy as np
from scipy.spatial import distance as dist
from config import (
    EYE_AR_THRESH, SMOOTH_BUFFER_SIZE, BLINK_TIME, DROWSY_TIME,
    MOUTH_AR_THRESH, YAWN_LIMIT
)

class FaceFeatures:
    """
    Combined face feature extraction from both systems.
    Uses MediaPipe landmarks but includes dlib-style EAR calculations
    and yawning detection from the original system.
    """

    def __init__(self):
        # Blink detection (from original system)
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.frames_since_last_blink = 0
        self.blinks_per_minute = 0

        # Yawning detection (from original system)
        self.yawning = 0
        self.yawn_threshold = MOUTH_AR_THRESH
        self.yawn_limit = YAWN_LIMIT

        # Smoothing buffers (from fatigue-detection-CV)
        self.ear_buffer = []
        self.jaw_buffer = []
        self.brow_buffer = []
        self.mouth_buffer = []
        self.buffer_size = SMOOTH_BUFFER_SIZE

        # Calibration data
        self.calibrated = False
        self.avg_ear_calibrated = 0.0

    @staticmethod
    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio using MediaPipe landmarks.
        Compatible with both systems' landmark formats.
        """
        # Convert landmarks to the format expected by EAR calculation
        eye_points = []
        for landmark in eye_landmarks:
            eye_points.append((landmark.x, landmark.y))

        if len(eye_points) < 6:
            return 0.0

        # EAR calculation (standard formula)
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)

        # Smooth EAR
        self.ear_buffer.append(ear)
        if len(self.ear_buffer) > self.buffer_size:
            self.ear_buffer.pop(0)
        smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)

        return smooth_ear

    def mouth_aspect_ratio(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio for yawning detection.
        Based on the original dlib system's MAR calculation.
        """
        if len(mouth_landmarks) < 12:
            return 0.0

        # Convert to points
        mouth_points = []
        for landmark in mouth_landmarks:
            mouth_points.append((landmark.x, landmark.y))

        # MAR calculation (vertical/horizontal ratio)
        A = dist.euclidean(mouth_points[1], mouth_points[7])  # Vertical distance
        B = dist.euclidean(mouth_points[0], mouth_points[6])  # Horizontal distance
        mar = A / B if B > 0 else 0.0

        # Smooth MAR
        self.mouth_buffer.append(mar)
        if len(self.mouth_buffer) > self.buffer_size:
            self.mouth_buffer.pop(0)
        smooth_mar = sum(self.mouth_buffer) / len(self.mouth_buffer)

        return smooth_mar

    def brow_distance(self, left_brow, right_brow):
        """
        Calculate distance between eyebrows.
        From fatigue-detection-CV system.
        """
        if not left_brow or not right_brow:
            return 0.0

        distance = self.euclidean_distance(left_brow, right_brow)

        # Smooth brow distance
        self.brow_buffer.append(distance)
        if len(self.brow_buffer) > self.buffer_size:
            self.brow_buffer.pop(0)
        smooth_distance = sum(self.brow_buffer) / len(self.brow_buffer)

        return smooth_distance

    def jaw_openness(self, top_jaw, bottom_jaw):
        """
        Calculate jaw openness (mouth height).
        From fatigue-detection-CV system.
        """
        if not top_jaw or not bottom_jaw:
            return 0.0

        openness = abs(top_jaw[1] - bottom_jaw[1])

        # Smooth jaw openness
        self.jaw_buffer.append(openness)
        if len(self.jaw_buffer) > self.buffer_size:
            self.jaw_buffer.pop(0)
        smooth_openness = sum(self.jaw_buffer) / len(self.jaw_buffer)

        return smooth_openness

    def blink_detection(self, left_ear, right_ear):
        """
        Detect blinks using EAR thresholds.
        Enhanced version combining both systems.
        """
        avg_ear = (left_ear + right_ear) / 2.0
        self.frames_since_last_blink += 1

        # Use calibrated threshold if available, otherwise use config
        threshold = self.avg_ear_calibrated * 0.8 if self.calibrated else EYE_AR_THRESH

        if avg_ear < threshold and self.frames_since_last_blink > 2:
            self.blink_count += 1
            self.frames_since_last_blink = 0

        # Update blinks per minute
        elapsed_time = time.time() - self.blink_start_time
        if elapsed_time > 60:
            self.blinks_per_minute = self.blink_count
            self.blink_count = 0
            self.blink_start_time = time.time()

        return self.blinks_per_minute

    def yawning_detection(self, mar):
        """
        Detect yawning using MAR thresholds.
        From original dlib system.
        """
        if mar > self.yawn_threshold:
            self.yawning += 1
        else:
            if self.yawning >= self.yawn_limit:
                # Yawn detected - could trigger alert here
                pass
            self.yawning = 0

        return self.yawning

    def calibrate(self, ear_values):
        """
        Calibrate EAR threshold based on user's baseline.
        From original dlib system.
        """
        if len(ear_values) > 0:
            self.avg_ear_calibrated = sum(ear_values) / len(ear_values)
            self.calibrated = True
            print(f"✅ Calibrated EAR baseline: {self.avg_ear_calibrated:.3f}")
            return True
        return False

    def reset(self):
        """Reset all detection states."""
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.frames_since_last_blink = 0
        self.blinks_per_minute = 0
        self.yawning = 0
        self.ear_buffer.clear()
        self.jaw_buffer.clear()
        self.brow_buffer.clear()
        self.mouth_buffer.clear()
        self.calibrated = False