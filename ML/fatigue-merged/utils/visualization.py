"""Visualization utilities for fatigue detection display."""

import cv2
import numpy as np

def create_fatigue_gauge(frame, fatigue_score, position=(20, 100), size=(200, 30)):
    """
    Draw a horizontal gauge showing fatigue level.
    Args:
        frame: Image to draw on
        fatigue_score: Fatigue score (0.0-1.0)
        position: Top-left (x, y) position
        size: (width, height) of gauge
    """
    x, y = position
    width, height = size

    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)

    # Filled bar (changes color based on level)
    fill_width = int(width * min(fatigue_score, 1.0))

    if fatigue_score < 0.3:
        color = (0, 255, 0)  # Green
    elif fatigue_score < 0.6:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red

    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

    # Percentage text
    text = f"{int(fatigue_score * 100)}%"
    cv2.putText(frame, text, (x + width + 10, y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_fatigue_meter(frame, fatigue_score, position=(20, 150)):
    """
    Draw a circular meter showing fatigue level.
    Args:
        frame: Image to draw on
        fatigue_score: Fatigue score (0.0-1.0)
        position: Center (x, y) position
    """
    center = position
    radius = 40
    thickness = 8

    # Background circle
    cv2.circle(frame, center, radius, (100, 100, 100), thickness)

    # Fatigue arc
    angle = int(360 * min(fatigue_score, 1.0))

    if fatigue_score < 0.3:
        color = (0, 255, 0)
    elif fatigue_score < 0.6:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, thickness)

    # Center text
    text = f"{int(fatigue_score * 100)}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2

    cv2.putText(frame, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def create_alert_message(frame, message, blink=False, blink_state=[0]):
    """
    Display a prominent alert message.
    Args:
        frame: Image to draw on
        message: Alert text
        blink: Whether to blink the message
        blink_state: List with single element for state tracking
    """
    if blink:
        blink_state[0] = (blink_state[0] + 1) % 20
        if blink_state[0] < 10:
            return

    h, w = frame.shape[:2]

    # Background rectangle
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    box_width = text_size[0] + 40
    box_height = text_size[1] + 30

    box_x = (w - box_width) // 2
    box_y = h - 150

    cv2.rectangle(frame, (box_x, box_y), 
                 (box_x + box_width, box_y + box_height),
                 (0, 0, 255), -1)
    cv2.rectangle(frame, (box_x, box_y),
                 (box_x + box_width, box_y + box_height),
                 (255, 255, 255), 3)

    # Alert text
    text_x = box_x + 20
    text_y = box_y + text_size[1] + 15

    cv2.putText(frame, message, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

def draw_eye_status(frame, left_ear, right_ear, position=(20, 250)):
    """
    Display eye aspect ratio values.
    Args:
        frame: Image to draw on
        left_ear: Left eye aspect ratio
        right_ear: Right eye aspect ratio
        position: Starting (x, y) position
    """
    x, y = position

    lines = [
        f"Left EAR: {left_ear:.3f}",
        f"Right EAR: {right_ear:.3f}",
        f"Avg EAR: {(left_ear + right_ear) / 2:.3f}"
    ]

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def create_dashboard(frame, stats):
    """
    Create a comprehensive dashboard overlay.
    Args:
        frame: Image to draw on
        stats: Dictionary with statistics
            - fps: Frames per second
            - fatigue_score: Current fatigue score
            - fatigue_level: Fatigue level string
            - blink_rate: Blinks per minute
            - yawns: Current yawn count
            - trend: Fatigue trend string
            - ear_avg: Average EAR
    """
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Title
    cv2.putText(frame, "FATIGUE MONITOR", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stats
    y_offset = 70
    stats_lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Fatigue: {stats.get('fatigue_score', 0):.2f}",
        f"Level: {stats.get('fatigue_level', 'Unknown')}",
        f"Blinks/min: {stats.get('blink_rate', 0)}",
        f"Yawns: {stats.get('yawns', 0)}",
        f"Trend: {stats.get('trend', 'stable')}"
    ]

    for line in stats_lines:
        cv2.putText(frame, line, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

def show_calibration_progress(frame, progress, total):
    """
    Display calibration progress bar.
    Args:
        frame: Image to draw on
        progress: Current progress count
        total: Total count needed
    """
    h, w = frame.shape[:2]

    # Progress bar
    bar_width = 400
    bar_height = 30
    bar_x = (w - bar_width) // 2
    bar_y = h // 2

    # Background
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 (100, 100, 100), -1)

    # Fill
    fill_width = int(bar_width * (progress / total))
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x + fill_width, bar_y + bar_height),
                 (0, 255, 0), -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x + bar_width, bar_y + bar_height),
                 (255, 255, 255), 2)

    # Text
    text = f"Calibrating: {progress}/{total}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = bar_y - 20

    cv2.putText(frame, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)