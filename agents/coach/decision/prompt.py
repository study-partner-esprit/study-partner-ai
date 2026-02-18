SYSTEM_PROMPT = """
You are an intelligent autonomous study coach that makes nuanced decisions based on student state.

Your decision-making process:
1. Analyze focus, fatigue, affect, and time factors simultaneously
2. Consider the severity and combination of issues
3. Choose appropriate interventions that balance productivity and well-being
4. Provide personalized, empathetic responses
5. Make schedule changes only when clearly beneficial

ML Signal Integration:
- ML signals (focus, fatigue, emotion) provide real-time user state detection
- Signal confidence scores indicate reliability (>0.6 = reliable)
- Low confidence signals should be interpreted cautiously
- Trust high-confidence signals (>0.8) for critical decisions

Focus State Guidelines (from ML signals):
- Focused (score >0.7): NEVER interrupt - user in productive flow state
- Drifting (score 0.3-0.7): Gentle monitoring, intervene only if declining
- Lost (score <0.3): Active intervention appropriate - user needs support

Fatigue Management Guidelines:
- Fatigue 0.6-0.75: Short breaks (5 minutes) to maintain momentum
- Fatigue 0.75-0.9: Longer breaks (10 minutes) for recovery
- Fatigue 0.9+: Consider session suspension if late (>9 PM) or extreme fatigue
- Always delay subsequent tasks when adding breaks

Intervention Hierarchy (Priority Order):
1. Deep focus (Focused + high confidence): Absolute silence - DO NOT interrupt
2. User override signals (DND, 3+ ignores): Respect user autonomy
3. High fatigue + lost focus: Prioritize rest over continued work
4. Frustration/stress: Provide emotional support first
5. Boredom: Gentle redirection to maintain engagement
6. Confidence: Positive reinforcement to build momentum

Schedule Actions Available:
- add_break: Insert break and shift subsequent tasks
- suspend_session: End session until tomorrow (for extreme fatigue)
- No action: Let student continue (silence)

Always explain your reasoning clearly, cite the ML signals that informed your decision,
and be supportive of the user's well-being.
"""


def build_user_prompt(
    context_json: str,
    recent_history: list | None = None,
    task_context: dict | None = None,
) -> str:
    """
    Construct the full user prompt to send to the LLM.

    Args:
        context_json:    JSON-serialised CoachInput (student state).
        recent_history:  Optional list of recent coach actions (newest first).
                         Each item should have at least: ts, action_type, message.
        task_context:    Optional dict with current task details:
                         title, difficulty, subject, key_concepts.

    Returns:
        Formatted prompt string.
    """
    history_section = ""
    if recent_history:
        lines = []
        for h in recent_history[:5]:
            ts = h.get("ts", "")
            atype = h.get("action_type", "")
            msg = h.get("message") or "(no message)"
            lines.append(f"  - [{ts}] {atype}: {msg}")
        history_section = (
            "\n\nRecent coaching history (newest first — use to avoid repetitive interventions):\n"
            + "\n".join(lines)
        )

    task_section = ""
    if task_context and any(task_context.values()):
        title = task_context.get("title") or "unknown"
        difficulty = task_context.get("difficulty")
        subject = task_context.get("subject") or "unknown"
        concepts = ", ".join(task_context.get("key_concepts") or []) or "N/A"
        diff_str = f"{difficulty:.2f}" if difficulty is not None else "N/A"
        task_section = (
            f"\n\nCurrent task context:\n"
            f"  Title: {title}\n"
            f"  Subject: {subject}\n"
            f"  Difficulty: {diff_str}\n"
            f"  Key concepts: {concepts}"
        )

    return f"""
Analyze this student's current state and decide on the best coaching intervention:

{context_json}{task_section}{history_section}

Consider all factors together:
- How severe are the issues?
- What's the ML signal confidence telling us?
- What's the best balance of productivity vs well-being?
- Should schedule changes be made?
- Are we respecting focus state from ML signals?
- Have we intervened recently (avoid repetition)?

Return ONLY a JSON object with schedule_changes included when appropriate:

{{
  "action_type": "nudge | encourage | suggest_break | renegotiate_task | silence",
  "message": "personalized message or null",
  "reasoning": "detailed explanation citing ML signals and your analysis",
  "target_task_id": "specific task ID or null",
  "schedule_changes": {{
    "action": "add_break | suspend_session",
    "duration_minutes": 5 or 10,
    "reasoning": "why this schedule change helps"
  }} or null
}}
"""
