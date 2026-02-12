from config import (
    EYE_AR_THRESH, BLINK_RATE_THRESH, BROW_DISTANCE_THRESH,
    JAW_OPEN_THRESH, FATIGUE_SCORE_THRESHOLD, SMOOTH_BUFFER_SIZE
)

class FatigueRules:
    """
    Enhanced fatigue scoring system combining both approaches.
    Includes multi-factor analysis and yawning detection.
    """

    def __init__(self):
        self.score_buffer = []
        self.buffer_size = SMOOTH_BUFFER_SIZE

        # Additional fatigue indicators
        self.consecutive_yawns = 0
        self.drowsy_frames = 0
        self.last_alert_time = 0

    def compute_fatigue_score(self, ear, blink_rate, brow_dist, jaw_open, yawning=0):
        """
        Compute comprehensive fatigue score using multiple indicators.

        Args:
            ear: Eye Aspect Ratio (lower = more tired)
            blink_rate: Blinks per minute (higher = more tired)
            brow_dist: Distance between eyebrows (lower = more tired)
            jaw_open: Jaw openness (higher = more tired)
            yawning: Current yawning count

        Returns:
            tuple: (fatigue_probability, suggest_break, fatigue_level)
        """
        score = 0.0
        factors = []

        # Eye Aspect Ratio factor
        if ear < EYE_AR_THRESH:
            score += 0.3
            factors.append("low_EAR")

        # Blink rate factor
        if blink_rate > BLINK_RATE_THRESH:
            score += 0.25
            factors.append("high_blink_rate")

        # Brow distance factor (frowning/tired expression)
        if brow_dist < BROW_DISTANCE_THRESH:
            score += 0.2
            factors.append("close_brows")

        # Jaw openness factor (yawning or mouth hanging open)
        if jaw_open > JAW_OPEN_THRESH:
            score += 0.15
            factors.append("open_jaw")

        # Yawning factor (strong indicator)
        if yawning > 0:
            yawn_factor = min(yawning * 0.1, 0.2)  # Max 0.2 for yawning
            score += yawn_factor
            factors.append(f"yawning_{yawning}")

        # Cap score at 1.0
        fatigue_prob = min(score, 1.0)

        # Smooth fatigue score over time
        self.score_buffer.append(fatigue_prob)
        if len(self.score_buffer) > self.buffer_size:
            self.score_buffer.pop(0)
        smooth_prob = sum(self.score_buffer) / len(self.score_buffer)

        # Determine fatigue level
        if smooth_prob < 0.3:
            fatigue_level = "alert"
        elif smooth_prob < 0.6:
            fatigue_level = "moderate"
        else:
            fatigue_level = "high"

        # Suggest break based on threshold and consecutive high readings
        suggest_break = smooth_prob > FATIGUE_SCORE_THRESHOLD

        # Track consecutive high fatigue readings
        if smooth_prob > 0.7:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = 0

        # Force break suggestion if too many consecutive drowsy frames
        if self.drowsy_frames > 30:  # ~1 second at 30fps
            suggest_break = True
            fatigue_level = "critical"

        return smooth_prob, suggest_break, fatigue_level, factors

    def get_fatigue_trend(self):
        """
        Analyze fatigue trend over recent buffer.
        Returns: "increasing", "decreasing", "stable"
        """
        if len(self.score_buffer) < 5:
            return "stable"

        recent = self.score_buffer[-5:]
        older = self.score_buffer[-10:-5] if len(self.score_buffer) >= 10 else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg + 0.1:
            return "increasing"
        elif recent_avg < older_avg - 0.1:
            return "decreasing"
        else:
            return "stable"

    def reset(self):
        """Reset fatigue tracking state."""
        self.score_buffer.clear()
        self.consecutive_yawns = 0
        self.drowsy_frames = 0
        self.last_alert_time = 0