"""
Task Generator using LM Studio
Generates study tasks from course content
"""

import json
import os
from typing import List, Dict
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# Use LM Studio instead of Google Gemini
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

MODEL_NAME = "lm-studio"  # Using LM Studio


def call_llm_task_generation(prompt: str) -> str:
    """Call LM Studio API for task generation."""
    try:
        messages = [{"role": "user", "content": prompt}]
        
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 3000,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"LM Studio API error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"Error calling LM Studio: {e}")
        return ""

TASK_GENERATION_PROMPT = """
You are an educational task designer. Your job is to create practical, actionable study tasks from course content.

Given a course with topics and subtopics, generate a comprehensive list of study tasks that will help students master the material.

REQUIREMENTS:
1. Create tasks that are specific, actionable, and measurable
2. Include a variety of task types:
   - Reading and comprehension tasks
   - Practice problems and exercises
   - Review and memorization tasks
   - Application and project tasks
3. Estimate realistic time for each task (in minutes)
4. Assign appropriate priority (low, medium, high) based on:
   - Foundational concepts = high priority
   - Advanced topics = medium priority
   - Optional/supplementary = low priority
5. Add relevant tags for organization
6. Create 3-8 tasks per major topic

Return ONLY valid JSON with this exact schema:

{
  "tasks": [
    {
      "title": "Clear, actionable task title",
      "description": "Detailed description of what to do",
      "priority": "low|medium|high",
      "estimatedTime": 30,
      "tags": ["topic-name", "task-type"]
    }
  ]
}

EXAMPLES OF GOOD TASKS:
- "Read and summarize: Introduction to Calculus (Chapter 1)"
- "Practice 10 derivative problems from Section 2.3"
- "Create flashcards for all formulas in Chapter 3"
- "Complete the integration practice worksheet"
- "Watch lecture video and take notes on limits"

AVOID:
- Vague tasks like "Study calculus"
- Tasks without clear outcomes
- Unrealistic time estimates
"""


def generate_tasks_from_course(course_title: str, topics: List[Dict]) -> List[Dict]:
    """
    Generate study tasks from course content using LM Studio
    
    Args:
        course_title: Title of the course
        topics: List of topics from the course, each containing subtopics
        
    Returns:
        List of task dictionaries
    """
    
    # Prepare course content summary for AI
    content_summary = f"Course: {course_title}\n\n"
    
    for topic in topics:
        content_summary += f"## Topic: {topic.get('title', 'Untitled')}\n"
        
        subtopics = topic.get('subtopics', [])
        if subtopics:
            content_summary += "Subtopics:\n"
            for subtopic in subtopics:
                subtopic_title = subtopic.get('title', 'Untitled subtopic')
                content_summary += f"- {subtopic_title}\n"
                
                # Add key concepts if available
                key_concepts = subtopic.get('key_concepts', [])
                if key_concepts:
                    content_summary += f"  Key concepts: {', '.join(key_concepts[:5])}\n"
        
        content_summary += "\n"
    
    # Call LM Studio API
    try:
        full_prompt = f"{TASK_GENERATION_PROMPT}\n\nCOURSE CONTENT:\n{content_summary}\n\nGenerate tasks:"
        
        response_text = call_llm_task_generation(full_prompt)
        
        if not response_text:
            print("Empty response from LM Studio API")
            return []
        
        # Parse JSON response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            # Remove first and last lines (``` markers)
            response_text = '\n'.join(lines[1:-1])
            # Remove language identifier if present
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
        
        try:
            result = json.loads(response_text)
            tasks = result.get('tasks', [])
            
            # Validate and clean tasks
            cleaned_tasks = []
            for task in tasks:
                if task.get('title') and task.get('description'):
                    cleaned_task = {
                        'title': task['title'],
                        'description': task['description'],
                        'priority': task.get('priority', 'medium'),
                        'estimatedTime': task.get('estimatedTime', 30),
                        'tags': task.get('tags', [])
                    }
                    # Ensure tags is a list
                    if not isinstance(cleaned_task['tags'], list):
                        cleaned_task['tags'] = []
                    
                    cleaned_tasks.append(cleaned_task)
            
            print(f"Successfully generated {len(cleaned_tasks)} tasks")
            return cleaned_tasks
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response text: {response_text[:500]}")
            return []
            
    except Exception as e:
        print(f"Error calling LM Studio API: {e}")
        return []


def generate_tasks_simple(course_title: str, topics: List[Dict]) -> List[Dict]:
    """
    Fallback: Generate simple tasks without AI
    Used if AI generation fails
    """
    tasks = []
    
    for topic in topics:
        topic_title = topic.get('title', 'Untitled')
        
        # Reading task
        tasks.append({
            'title': f"Read and understand: {topic_title}",
            'description': f"Read through all materials for {topic_title} and take notes on key concepts",
            'priority': 'high',
            'estimatedTime': 45,
            'tags': [course_title, 'reading', topic_title]
        })
        
        # Practice task
        tasks.append({
            'title': f"Practice exercises: {topic_title}",
            'description': f"Complete practice problems related to {topic_title}",
            'priority': 'medium',
            'estimatedTime': 60,
            'tags': [course_title, 'practice', topic_title]
        })
        
        # Review task
        tasks.append({
            'title': f"Review and summarize: {topic_title}",
            'description': f"Create a summary of {topic_title} in your own words",
            'priority': 'medium',
            'estimatedTime': 30,
            'tags': [course_title, 'review', topic_title]
        })
    
    return tasks


if __name__ == "__main__":
    # Test the task generator
    test_topics = [
        {
            "title": "Introduction to Calculus",
            "subtopics": [
                {
                    "title": "Limits and Continuity",
                    "key_concepts": ["limits", "continuity", "asymptotes"]
                },
                {
                    "title": "Derivatives",
                    "key_concepts": ["derivative rules", "chain rule", "implicit differentiation"]
                }
            ]
        },
        {
            "title": "Integration",
            "subtopics": [
                {
                    "title": "Basic Integration",
                    "key_concepts": ["antiderivatives", "integration rules", "u-substitution"]
                }
            ]
        }
    ]
    
    tasks = generate_tasks_from_course("Calculus 101", test_topics)
    
    print(f"\nGenerated {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. {task['title']}")
        print(f"   Description: {task['description']}")
        print(f"   Priority: {task['priority']}, Time: {task['estimatedTime']} min")
        print(f"   Tags: {', '.join(task['tags'])}")
