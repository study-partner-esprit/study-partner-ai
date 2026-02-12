# Configuration: thresholds, parameters for merged fatigue detection system
# Combines best features from both original systems

# Eye Aspect Ratio threshold for blink detection
EYE_AR_THRESH = 0.25

# Blink rate threshold (blinks per minute)
BLINK_RATE_THRESH = 20

# Brow distance threshold (pixels)
BROW_DISTANCE_THRESH = 10

# Jaw openness threshold (pixels)
JAW_OPEN_THRESH = 15

# Mouth Aspect Ratio threshold for yawning
MOUTH_AR_THRESH = 0.6

# Fatigue score threshold (0.0-1.0)
FATIGUE_SCORE_THRESHOLD = 0.5

# Smoothing buffer size (frames)
SMOOTH_BUFFER_SIZE = 10

# Blink detection parameters
BLINK_TIME = 0.1  # seconds
DROWSY_TIME = 1.0  # seconds

# Yawning detection
YAWN_LIMIT = 2  # consecutive yawns before alert

# Camera settings
CAMERA_TIMEOUT = 5.0  # seconds
CALIBRATION_FRAMES = 100

# Display settings
SHOW_GUI = True
WINDOW_NAME = "Advanced Fatigue Detection System"