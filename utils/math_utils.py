"""Mathematical utility functions for AI processing."""
from typing import List, Optional


def normalize_score(
    value: float,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> float:
    """Normalize a value to a 0-1 scale.
    
    Args:
        value: Value to normalize
        min_value: Minimum value in the range
        max_value: Maximum value in the range
        
    Returns:
        Normalized value between 0 and 1
    """
    if max_value == min_value:
        return 0.5
        
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, normalized))


def calculate_confidence(
    scores: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted confidence score.
    
    Args:
        scores: List of individual scores
        weights: Optional weights for each score
        
    Returns:
        Calculated confidence score
    """
    if not scores:
        return 0.0
        
    if weights is None:
        weights = [1.0] * len(scores)
        
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have the same length")
        
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    if total_weight == 0:
        return 0.0
        
    return weighted_sum / total_weight


def apply_threshold(
    value: float,
    threshold: float,
    above_value: float = 1.0,
    below_value: float = 0.0
) -> float:
    """Apply threshold to a value.
    
    Args:
        value: Value to check
        threshold: Threshold value
        above_value: Return value if above threshold
        below_value: Return value if below threshold
        
    Returns:
        Result based on threshold comparison
    """
    return above_value if value >= threshold else below_value


def calculate_average(values: List[float]) -> float:
    """Calculate average of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Average value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_weighted_average(
    values: List[float],
    weights: List[float]
) -> float:
    """Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted average
    """
    if not values or not weights:
        return 0.0
        
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")
        
    return calculate_confidence(values, weights)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def sigmoid(x: float) -> float:
    """Apply sigmoid activation function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output (0-1)
    """
    import math
    return 1 / (1 + math.exp(-x))


def exponential_decay(
    initial_value: float,
    decay_rate: float,
    time_steps: int
) -> float:
    """Calculate exponential decay.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate (0-1)
        time_steps: Number of time steps
        
    Returns:
        Decayed value
    """
    return initial_value * (decay_rate ** time_steps)
