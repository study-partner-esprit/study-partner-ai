from typing import Optional
from agents.coach.models.schemas import CoachInput, CoachAction


def apply_rules(input_data: CoachInput) -> Optional[CoachAction]:
    """
    Apply hard rules that override LLM decision-making.
    
    These rules ensure critical constraints are respected:
    1. Never interrupt deep focus (especially with high ML confidence)
    2. Respect user autonomy (DND, multiple ignores)
    3. Enforce safety boundaries
    
    Returns:
        CoachAction if a rule fires, None if LLM should decide
    """
    
    # Rule 1: Do not disturb mode - absolute silence
    if input_data.do_not_disturb:
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="Do not disturb is enabled."
        )
    
    # Rule 2: User has ignored coach 3+ times - respect autonomy
    if input_data.ignored_count >= 3:
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="Coach was ignored several times. Respecting user preference for autonomy."
        )
    
    # Rule 3: Deep focus detected - never interrupt productive flow
    # This is the most important rule for ML signal integration
    if input_data.focus_state.state == "Focused":
        # Check for critical fatigue first - safety overrides focus
        if input_data.fatigue_state.state == "Critical":
            return CoachAction(
                action_type="suggest_break",
                message="Critical fatigue detected despite focus. Your safety is more important than maintaining focus.",
                reasoning=f"Critical fatigue (score: {input_data.fatigue_state.score:.2f}) overrides focus protection. Prioritizing user safety."
            )
        
        # Extra check: if we have ML signals with high confidence, be even more cautious
        if input_data.signals is not None:
            confidence = getattr(input_data.signals, 'focus_confidence', 0.0)
            if confidence > 0.7 and input_data.focus_state.score > 0.7:
                return CoachAction(
                    action_type="silence",
                    message=None,
                    reasoning=f"User is deeply focused (ML confidence: {confidence:.2f}). Never interrupt productive flow state."
                )
        
        # Standard focus rule
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="User is deeply focused. Avoiding interruption."
        )
    
    # Rule 4: Critical fatigue detected - force break for safety
    if input_data.fatigue_state.state == "Critical":
        return CoachAction(
            action_type="suggest_break",
            message="You appear to be critically fatigued. Please take a break to rest and recharge.",
            reasoning=f"Critical fatigue detected (score: {input_data.fatigue_state.score:.2f}). Prioritizing user safety and well-being."
        )
    
    # Rule 5: High fatigue - suggest break unless deeply focused
    if input_data.fatigue_state.state == "High":
        # Only override if not deeply focused
        if input_data.focus_state.state != "Focused" or input_data.focus_state.score < 0.7:
            return CoachAction(
                action_type="suggest_break",
                message="You're showing signs of high fatigue. Consider taking a short break.",
                reasoning=f"High fatigue detected (score: {input_data.fatigue_state.score:.2f}). Suggesting break to prevent burnout."
            )
    
    # No hard rule fired - let LLM decide
    return None
