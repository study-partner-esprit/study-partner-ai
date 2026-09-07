"""Maps 7-class facial emotion recognition (FER) probabilities onto the
Coach's 5-value `affective_state` vocabulary (COACH-15).

Pure function, no ML/IO dependency, so it is trivially unit-testable in
isolation from the model/adapter.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

AffectiveState = Literal["engaged", "frustrated", "stressed", "bored", "confident"]

FER_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def map_probabilities_to_affective_state(
    probabilities: Dict[str, float],
) -> Tuple[AffectiveState, float]:
    """
    Map 7 FER class probabilities to one of the 5 CoachInput.affective_state
    values, plus a confidence score in [0, 1].

    Grouping (instantaneous — one video frame in, one state out):

        confident  = P(Happy)
        engaged    = P(Neutral) + P(Surprise)
        frustrated = P(Angry) + P(Disgust)
        stressed   = P(Fear) + P(Sad)

    `affective_state` is the group with the highest summed probability;
    `confidence` is that winning sum itself (already in [0, 1] since it is
    built from softmax probabilities) — mirrors how `focus_confidence`/
    `fatigue_confidence` are already used elsewhere in this service.

    NOTE — "bored" is deliberately never produced by this function. A single
    frame cannot distinguish boredom from calm engagement; boredom is a
    *temporal* pattern (sustained low-intensity, low-confidence readings
    over several minutes), the same way `focus_trend` derives "declining
    focus" from a series rather than a single reading. Deriving "bored"
    belongs in a rule/EMA layer over a window of `EmotionAdapter` outputs,
    not in this per-frame mapping.

    NOTE — classifying `Sad` under `stressed` (rather than treating it as a
    boredom/disengagement signal) is a default assumption pending product
    confirmation. It is isolated to one line below and trivial to move.

    Args:
        probabilities: dict mapping each of FER_EMOTIONS to a probability
            (values need not sum to exactly 1.0; missing keys default to 0).

    Returns:
        (affective_state, confidence)
    """
    p = probabilities
    groups: Dict[AffectiveState, float] = {
        "confident": p.get("Happy", 0.0),
        "engaged": p.get("Neutral", 0.0) + p.get("Surprise", 0.0),
        "frustrated": p.get("Angry", 0.0) + p.get("Disgust", 0.0),
        "stressed": p.get("Fear", 0.0)
        + p.get("Sad", 0.0),  # Sad -> stressed (assumption)
    }

    affective_state: AffectiveState = max(groups, key=groups.get)  # type: ignore[arg-type]
    confidence = max(0.0, min(1.0, groups[affective_state]))
    return affective_state, confidence
