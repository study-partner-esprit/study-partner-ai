"""Helper utility functions for fatigue detection system."""

import cv2
import numpy as np

def histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast.
    Args:
        image: BGR image
    Returns:
        Equalized grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def smooth_values(buffer, new_value, max_size=10):
    """
    Add value to buffer and return smoothed average.
    Args:
        buffer: List to store values
        new_value: New value to add
        max_size: Maximum buffer size
    Returns:
        Smoothed average value
    """
    buffer.append(new_value)
    if len(buffer) > max_size:
        buffer.pop(0)
    return sum(buffer) / len(buffer) if buffer else 0.0

def format_time(seconds):
    """
    Format seconds into readable time string.
    Args:
        seconds: Time in seconds
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def draw_landmarks(frame, landmarks, indices, color=(0, 255, 0), thickness=2):
    """
    Draw facial landmarks on frame.
    Args:
        frame: Image to draw on
        landmarks: List of facial landmarks
        indices: List of landmark indices to draw
        color: BGR color tuple
        thickness: Circle thickness
    """
    h, w = frame.shape[:2]
    for i in indices:
        if i < len(landmarks):
            landmark = landmarks[i]
            cv2.circle(frame, 
                      (int(landmark.x * w), int(landmark.y * h)),
                      thickness, color, -1)

def calculate_fps(frame_count, start_time, current_time):
    """
    Calculate frames per second.
    Args:
        frame_count: Total frames processed
        start_time: Processing start time
        current_time: Current time
    Returns:
        FPS as float
    """
    elapsed = current_time - start_time
    return frame_count / elapsed if elapsed > 0 else 0.0

def create_status_overlay(frame, text_lines, position=(20, 30), 
                         line_spacing=25, font_scale=0.6, 
                         color=(0, 255, 0), thickness=2):
    """
    Create text overlay with multiple lines.
    Args:
        frame: Image to draw on
        text_lines: List of text strings
        position: Starting (x, y) position
        line_spacing: Pixels between lines
        font_scale: Text size
        color: BGR color
        thickness: Text thickness
    """
    x, y = position
    for line in text_lines:
        cv2.putText(frame, line, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += line_spacing

def create_alert_border(frame, color=(0, 0, 255), thickness=5):
    """
    Draw a colored border around the frame for alerts.
    Args:
        frame: Image to draw on
        color: BGR color
        thickness: Border thickness
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), color, thickness)

def resize_maintain_aspect(image, target_height=460):
    """
    Resize image maintaining aspect ratio.
    Args:
        image: Input image
        target_height: Desired height
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    scale = h / target_height
    new_width = int(w / scale)
    return cv2.resize(image, (new_width, target_height))

def calculate_progress_bar(value, max_value):
    """
    Calculate progress bar percentage.
    Args:
        value: Current value
        max_value: Maximum value
    Returns:
        Percentage (0-100)
    """
    return min(int((value / max_value) * 100), 100) if max_value > 0 else 0