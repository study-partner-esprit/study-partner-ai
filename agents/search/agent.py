"""Search agent public API and Flask app factory.

This file refactors the previous `main.py` into a package-internal agent
module. The Flask app is optional – the module exposes `process_question`
and a `create_app()` helper.
"""

import uuid
import time
import os
from flask import Flask, render_template, request, jsonify
from .retrieval import web_search
from .extraction import extract_text
from .llm import ask_llm
from .services import get_voice_service, VoiceConfig


def process_question(
    question: str,
    use_voice: bool = False,
    user_id: str = "",
    session_id: str = "",
    trace_id: str = "",
):
    """Run the full search pipeline: retrieve → extract → generate → persist.

    Args:
        question:   Natural language question from the user.
        use_voice:  If True, speak the answer asynchronously via VoiceService.
        user_id:    Authenticated user ID (used for MongoDB persistence).
        session_id: Browser / tab session ID (grouped in history).
        trace_id:   Request trace ID for log correlation.

    Returns:
        dict with keys: success, question, answer, sources_count, urls, [error]
    """
    if not trace_id:
        trace_id = str(uuid.uuid4())

    if not question or not question.strip():
        return {"success": False, "error": "No question provided", "answer": ""}

    fallback_answer = (
        "I couldn't retrieve web sources right now, so I can't provide a source-grounded answer at the moment. "
        "Please try again in a bit or rephrase your question with more specific keywords."
    )

    max_pipeline_seconds = int(os.getenv("SEARCH_PIPELINE_TIMEOUT_SECONDS", "70"))
    pipeline_started_at = time.time()

    urls = web_search(question, max_results=5)
    if not urls:
        return {
            "success": True,
            "question": question,
            "answer": fallback_answer,
            "sources_count": 0,
            "urls": [],
            "trace_id": trace_id,
            "degraded": True,
            "reason": "No search results found.",
        }

    content = ""
    for url in urls:
        if time.time() - pipeline_started_at >= max_pipeline_seconds:
            break
        extracted = extract_text(url)
        if extracted:
            content += extracted + "\n\n"

    if not content:
        timed_out = time.time() - pipeline_started_at >= max_pipeline_seconds
        return {
            "success": True,
            "question": question,
            "answer": fallback_answer,
            "sources_count": len(urls),
            "urls": urls,
            "trace_id": trace_id,
            "degraded": True,
            "reason": (
                "Search pipeline time budget exceeded"
                if timed_out
                else "No content extracted from sources"
            ),
        }

    prompt = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{content[:2000]}\n\n"
        f"Provide a clear, comprehensive answer based only on the sources above."
    )
    answer = ask_llm(prompt)

    if not answer:
        return {
            "success": True,
            "question": question,
            "answer": "I found relevant web sources, but answer generation is currently unavailable. Please try again shortly.",
            "sources_count": len(urls),
            "urls": urls,
            "trace_id": trace_id,
            "degraded": True,
            "reason": "LLM unavailable",
        }

    if use_voice and answer:
        get_voice_service().speak_text(answer, async_mode=True)

    # Persist to MongoDB (graceful no-op when DB unavailable or no user_id)
    try:
        from .services.search_repository import SearchRepository

        SearchRepository().save_exchange(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=urls,
            session_id=session_id,
            trace_id=trace_id,
        )
    except Exception:
        pass

    return {
        "success": True,
        "question": question,
        "answer": answer,
        "sources_count": len(urls),
        "urls": urls,
        "trace_id": trace_id,
    }


def create_app():
    app = Flask(__name__, template_folder="templates")
    voice_service = get_voice_service()

    @app.route("/", methods=["GET", "POST"])
    def index():
        question = ""
        answer = ""
        if request.method == "POST":
            question_text = request.form.get("question", "").strip()
            voice_enabled = request.form.get("voice") == "on"
            result = process_question(question_text, use_voice=voice_enabled)
            question = result.get("question", question_text)
            answer = result.get("answer", "")
        return render_template("index.html", question=question, answer=answer)

    @app.route("/api/ask", methods=["POST"])
    def api_ask():
        try:
            data = request.json
            question = data.get("question", "").strip()
            use_voice = data.get("voice", False)
            if not question:
                return jsonify({"success": False, "error": "No question provided"}), 400
            result = process_question(question, use_voice=use_voice)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/voice/config", methods=["GET"])
    def api_voice_config():
        return jsonify(
            {
                "success": True,
                "config": VoiceConfig.get_config(),
                "available_voices": voice_service.get_available_voices(),
            }
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
