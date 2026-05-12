"""
Gradio UI for Socratic Evaluator using Google Gemini.
"""
import os
import logging
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config.settings import GEMINI_API_KEY
from src.evaluator.evaluator_agent import EvaluatorAgent
from src.evaluator.llm_client import GeminiClient

logger = logging.getLogger(__name__)

# Initialize evaluator with Gemini
evaluator = None
try:
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        logger.error("GEMINI_API_KEY not set in environment")
    else:
        evaluator = EvaluatorAgent()
        logger.info("✓ Evaluator initialized with Gemini")
except Exception as e:
    logger.error(f"Failed to initialize evaluator: {e}")

def start_evaluation(
    task_title: str,
    task_description: str,
    task_details: str,
    concepts_input: str,
    state: dict,
) -> tuple[dict, str, str, str, str, str, str]:
    """Start a new evaluation session and initialize session state."""
    
    if not evaluator:
        return state, "", "", "", "", "", "❌ Error: Evaluator not initialized. Check GEMINI_API_KEY environment variable."
    
    if not all([task_title, task_description, task_details]):
        return state, "", "", "", "", "", "❌ Error: Please fill in all fields"
    
    try:
        result = evaluator.start_session(
            task_title=task_title,
            task_description=task_description,
            task_details=task_details,
            max_attempts=5,
        )
        
        session_id = result["session_id"]
        question = result["question"]
        state = {
            "current_question": question,
            "attempt": 1,
            "session_id": session_id,
        }
        session_info = f"Session ID: {session_id} • Attempt: {state['attempt']}"
        
        logger.info(f"Started session: {session_id}")
        return state, question, session_info, "", "", "", ""
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return state, "", "", "", "", "", f"❌ Error: {str(e)}"


def process_answer(answer: str, state: dict) -> tuple[dict, str, str, str, str, str, str, str]:
    """Process the student's answer and return updated state, feedback, next question, final result, score, current question, answer input reset, and session info."""
    
    if not state or not state.get("session_id"):
        return state, "❌ No active session. Start an evaluation first.", "", "", "", state.get("current_question", ""), "", ""
    
    if not answer.strip():
        return state, "❌ Please enter an answer", "", "", "", state.get("current_question", ""), "", ""
    
    try:
        result = evaluator.handle_user_answer(state["session_id"], answer)
        
        if result.get("error"):
            return state, f"❌ Error: {result.get('error')}", "", "", "", state.get("current_question", ""), "", ""
        
        feedback = result.get("feedback", "")
        score = f"{result.get('mastery_score', 0.0):.1%}"
        result_state = result.get("state", "").upper()
        session_info = f"Session ID: {state['session_id']} • Attempt: {state.get('attempt', 1)}"
        
        if result_state == "CONTINUE":
            next_question = result.get("next_question", "")
            state["current_question"] = next_question
            state["attempt"] = state.get("attempt", 1) + 1
            session_info = f"Session ID: {state['session_id']} • Attempt: {state['attempt']}"
            return state, feedback, next_question, "", score, next_question, "", session_info
        
        if result_state in {"FAILED", "COMPLETED"}:
            state["current_question"] = "❌ Evaluation ended. Try again."
            final_result = result.get("message", "Evaluation ended. Try again.")
            return state, feedback, "", final_result, score, state["current_question"], "", session_info
        
        if result_state in {"MASTERY_CONFIRMED", "MASTERED"}:
            state["current_question"] = "✅ Mastery achieved!"
            final_result = result.get("message", "Mastery achieved!")
            return state, feedback, "", final_result, score, state["current_question"], "", session_info
        
        # Fallback: preserve current question while returning feedback
        return state, feedback, "", "", score, state.get("current_question", ""), "", session_info
    except Exception as e:
        logger.error(f"Failed to process answer: {e}")
        return state, f"❌ Error: {str(e)}", "", "", "", state.get("current_question", ""), "", ""


def build_interface():
    """Build the Gradio interface."""
    
    with gr.Blocks(title="Socratic Evaluator with Gemini") as demo:
        gr.Markdown("# 📚 Socratic Evaluator - Powered by Google Gemini")
        gr.Markdown("Engage in Socratic dialogue to demonstrate your understanding of any topic.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Task Setup")
                task_title = gr.Textbox(label="Task Title", placeholder="e.g., Photosynthesis")
                task_description = gr.Textbox(label="Task Description", placeholder="Brief overview", lines=2)
                task_details = gr.Textbox(label="Task Details", lines=3, placeholder="Detailed information")
                concepts_input = gr.Textbox(label="Key Concepts (comma-separated)", placeholder="e.g., chlorophyll, ATP, glucose")
                
                start_btn = gr.Button("🚀 Start Evaluation", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Session Info")
                question_output = gr.Textbox(label="Current Question", lines=4, interactive=False)
                session_info = gr.Textbox(label="Session Info", interactive=False)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Your Answer")
                answer_input = gr.Textbox(label="Your Answer", lines=4, placeholder="Type your response here")
                submit_btn = gr.Button("✅ Submit Answer", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Evaluation")
                feedback_output = gr.Textbox(label="Feedback", lines=6, interactive=False)
                next_question_output = gr.Textbox(label="Next Question", lines=4, interactive=False)
                final_result_output = gr.Textbox(label="Final Result", lines=2, interactive=False)
                score_output = gr.Textbox(label="Score", interactive=False)
        
        session_state = gr.State({
            "current_question": None,
            "attempt": 1,
            "session_id": None,
        })
        
        # Event handlers
        start_btn.click(
            fn=start_evaluation,
            inputs=[task_title, task_description, task_details, concepts_input, session_state],
            outputs=[session_state, question_output, session_info, feedback_output, next_question_output, final_result_output, score_output],
        )
        
        submit_btn.click(
            fn=process_answer,
            inputs=[answer_input, session_state],
            outputs=[session_state, feedback_output, next_question_output, final_result_output, score_output, question_output, answer_input, session_info],
        )
    
    return demo


def main():
    """Launch Gradio interface."""
    app = build_interface()
    print("Launching Gradio on http://127.0.0.1:7860")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
