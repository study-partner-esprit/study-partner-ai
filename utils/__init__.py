"""Utility functions."""
from .math_utils import (
    normalize_score,
    calculate_confidence,
    apply_threshold,
    calculate_average,
    calculate_weighted_average,
    clamp
)

__all__ = [
    "normalize_score",
    "calculate_confidence",
    "apply_threshold",
    "calculate_average",
    "calculate_weighted_average",
    "clamp"
]
