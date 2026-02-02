"""
Gradio interface for Study Partner AI with JSON input
"""
import gradio as gr
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput


def create_study_plan(goal: str, available_time: int, subject_json: str = "") -> str:
    """
    Create a study plan using the planner agent.

    Args:
        goal: Learning goal
        available_time: Available time in minutes
        subject_json: JSON string with subject information (optional)

    Returns:
        Formatted study plan as string
    """
    # Validate inputs
    if not goal or not goal.strip():
        return "❌ **Error:** Please enter a learning goal."
    
    if available_time < 30:
        return "❌ **Error:** Available time must be at least 30 minutes."

    try:
        # Initialize agent
        agent = PlannerAgent()

        # Process JSON input if provided
        course_documents = None
        
        if subject_json and subject_json.strip():
            import json
            import re
            try:
                # Clean up JSON input
                cleaned_json = subject_json.strip()
                
                # Remove BOM if present
                if cleaned_json.startswith('\ufeff'):
                    cleaned_json = cleaned_json[1:]
                
                # Remove single-line comments (// ...)
                cleaned_json = re.sub(r'//.*', '', cleaned_json)
                
                # Remove multi-line comments (/* ... */)
                cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
                
                # Remove trailing commas before closing brackets
                cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                
                subject_data = json.loads(cleaned_json)
                
                # Extract documents from subject_details
                if 'subject_details' in subject_data:
                    subject_details = subject_data['subject_details']
                    course_documents = []
                    
                    # Extract syllabus topics
                    if 'syllabus' in subject_details:
                        for chapter in subject_details['syllabus']:
                            if 'topics' in chapter and isinstance(chapter['topics'], list):
                                course_documents.extend(chapter['topics'])
                    
                    # Add learning outcomes descriptions
                    if 'learning_outcomes' in subject_details:
                        for outcome in subject_details['learning_outcomes']:
                            if 'description' in outcome:
                                course_documents.append(outcome['description'])
                    
                    # Add general objective
                    if 'general_objective' in subject_details:
                        course_documents.append(subject_details['general_objective'])
                    
                    # Add prerequisites
                    if 'prerequisites' in subject_details and isinstance(subject_details['prerequisites'], list):
                        course_documents.extend(subject_details['prerequisites'])
                    
                    # Filter out empty strings
                    course_documents = [doc for doc in course_documents if doc and doc.strip()]
                    
                    if not course_documents:
                        return '❌ **Error:** No valid content found in JSON.'
                    
                    print(f"✅ Loaded {len(course_documents)} document(s) from subject details")
                else:
                    return '❌ **Error:** JSON must contain a "subject_details" object.'
                    
            except json.JSONDecodeError as e:
                error_msg = f"❌ **Error:** Invalid JSON format.\n\n"
                error_msg += f"**Details:** {str(e)}\n\n"
                error_msg += "**Tips:**\n"
                error_msg += "- Make sure all property names are in double quotes\n"
                error_msg += "- Remove any trailing commas\n"
                error_msg += "- Check for unescaped special characters\n"
                error_msg += "- Verify all brackets {{ }} and [ ] are properly closed"
                return error_msg
        else:
            return '❌ **Error:** Please provide JSON with subject details.'

        # Create request
        from datetime import datetime, timedelta
        deadline = (datetime.now() + timedelta(days=7)).isoformat()

        request = PlannerInput(
            goal=goal,
            deadline_iso=deadline,
            available_minutes=available_time,
            user_id="gradio_user",
            retrieved_concepts=None,
            course_documents=course_documents,
            tokenization_settings=None
        )

        # Generate plan
        output = agent.plan(request)

        # Format output
        result = f"🎯 **Goal:** {output.task_graph.goal}\n\n"
        result += f"📊 **Tasks Generated:** {len(output.task_graph.tasks)}\n"
        result += f"⏱️ **Total Time:** {output.task_graph.total_estimated_minutes} min\n\n"
        
        if course_documents:
            result += f"📚 **Course Documents:** {len(course_documents)} document(s) indexed\n\n"

        if hasattr(output, 'warning') and output.warning:
            result += f"⚠️ **Warning:** {output.warning}\n\n"

        result += "## 📝 Study Tasks:\n\n"

        for i, task in enumerate(output.task_graph.tasks, 1):
            result += f"### {i}. {task.title}\n"
            result += f"- **Description:** {task.description}\n"
            result += f"- **Duration:** {task.estimated_minutes} minutes\n"
            result += f"- **Difficulty:** {task.difficulty:.1f}/1.0\n"
            if task.is_review:
                result += "- **Type:** Review Session ✨\n"
            if task.prerequisites:
                result += f"- **Prerequisites:** {len(task.prerequisites)} task(s)\n"
            result += "\n"

        if output.clarification_required:
            result += "\n❓ **Note:** Some tasks may need clarification. Consider providing more specific learning goals."

        return result

    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease try again with a different goal or contact support if the issue persists."


def main():
    """Main function to run the Gradio interface."""

    # Create Gradio interface
    with gr.Blocks(title="Study Partner AI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎓 Study Partner AI")
        gr.Markdown("Generate personalized study plans using AI-powered task decomposition with optional course document support.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Input")
                
                goal_input = gr.Textbox(
                    label="Learning Goal",
                    placeholder="e.g., Learn Python for machine learning, Master Linear Algebra",
                    lines=3
                )

                time_input = gr.Slider(
                    label="Available Time (minutes)",
                    minimum=30,
                    maximum=480,
                    value=120,
                    step=15
                )
                
                gr.Markdown("### 📚 Subject Information (Optional)")
                gr.Markdown("Paste JSON with course materials and information")
                
                json_input = gr.Textbox(
                    label="Subject JSON",
                    placeholder='{\n  "documents": [\n    "Introduction to...",\n    "Chapter 1..."\n  ]\n}',
                    lines=8,
                    info="Paste JSON containing subject documents"
                )

                generate_btn = gr.Button("🚀 Generate Study Plan", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Study Plan")
                output_display = gr.Markdown(
                    value="""
### Welcome to Study Partner AI! 👋

**Get Started:**
1. Enter your learning goal
2. Set your available time
3. (Optional) Paste JSON with subject information
4. Click "Generate Study Plan"

**JSON Format Example:**
```json
{
  "documents": [
    "Python basics: variables, data types, functions",
    "Object-oriented programming concepts",
    "File I/O and error handling"
  ]
}
```

The AI will analyze your goal and create a personalized study plan!
                    """
                )

        # Connect the function
        generate_btn.click(
            fn=create_study_plan,
            inputs=[goal_input, time_input, json_input],
            outputs=output_display
        )

        # Examples
        gr.Examples(
            examples=[
                ["Learn Python programming fundamentals", 90, ""],
                ["Master Linear Algebra for Machine Learning", 180, ""],
                ["Understand Neural Networks basics", 120, ""],
                ["Learn Data Structures and Algorithms", 150, '{"documents": ["Arrays and linked lists", "Stacks and queues", "Trees and graphs"]}']
            ],
            inputs=[goal_input, time_input, json_input]
        )

        gr.Markdown("""
        ---
        ### 📖 How it works:
        1. **Input your learning goal** - Be specific about what you want to learn
        2. **Set available time** - How many minutes you have for studying
        3. **Paste JSON with subject info** (Optional) - Course materials and documents
        4. **Get personalized tasks** - AI breaks down your goal into manageable study sessions
        5. **Follow the plan** - Each task includes time estimates and difficulty levels

        ### 🔧 Features:
        - ✅ AI-powered task decomposition
        - ✅ JSON-based knowledge input
        - ✅ RAG-based knowledge retrieval
        - ✅ Time-aware planning
        - ✅ Difficulty assessment
        - ✅ Prerequisite tracking
        - ✅ Review session suggestions
        
        ### 📄 JSON Input Format:
        The optional JSON input allows you to provide course materials:
        ```json
        {
          "documents": [
            "First topic or chapter content",
            "Second topic or chapter content",
            "Third topic or chapter content"
          ]
        }
        ```
        Simply paste an array of text strings containing your course materials.
        """)

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()