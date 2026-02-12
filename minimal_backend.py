from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pydantic import BaseModel
import tempfile
import os
import uuid
import shutil
from datetime import datetime
from agents.course_ingestion.agent import ingest_course

# In-memory storage
subjects_db = {}  # user_id -> [subjects]
courses_db = {}   # subject_id -> [courses]
study_plans_db = {}  # user_id -> [plans]

app = FastAPI(title='Study Partner AI - Full Backend', version='1.0.0')

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Pydantic models for request validation
class StudyPlanRequest(BaseModel):
    user_id: str
    goal: str
    available_time_minutes: int
    course_id: Optional[str] = None
    start_date: str

class CoachDecisionRequest(BaseModel):
    user_id: str
    ignored_count: int = 0
    do_not_disturb: bool = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/api/ai/courses/ingest')
async def ingest_course_endpoint(
    files: List[UploadFile] = File(...),
    course_title: str = Form(...),
    user_id: str = Form(...),
    subject_name: str = Form(default="General"),
    subject_image: Optional[UploadFile] = File(None)
):
    print(f'📚 Course upload started: {course_title} for user {user_id} under subject {subject_name}')
    try:
        # Save uploaded files temporarily
        temp_paths = []
        for uploaded_file in files:
            file_extension = os.path.splitext(uploaded_file.filename)[1] if uploaded_file.filename else '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await uploaded_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_paths.append(temp_path)
        
        print(f'📄 Files saved to: {temp_paths}')
        
        # Process the course (mock implementation for testing)
        print(f'📚 Processing course: {course_title}')
        print(f'📄 File paths: {temp_paths}')
        print(f'👤 User: {user_id}')
        
        # Handle subject image
        image_url = None
        if subject_image:
            image_filename = f"{uuid.uuid4()}_{subject_image.filename}"
            image_path = os.path.join("uploads", image_filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(subject_image.file, buffer)
            image_url = f"http://localhost:8000/uploads/{image_filename}"
            print(f'🖼️ Subject image saved: {image_url}')

        # Get or create subject
        if user_id not in subjects_db:
            subjects_db[user_id] = []
        
        subject = next((s for s in subjects_db[user_id] if s['name'] == subject_name), None)
        if not subject:
            subject_id = str(uuid.uuid4())
            subject = {
                "id": subject_id,
                "name": subject_name,
                "user_id": user_id,
                "image": image_url,
                "created_at": datetime.now().isoformat()
            }
            subjects_db[user_id].append(subject)
            courses_db[subject_id] = []
            print(f'📂 Created new subject: {subject_name} ({subject_id})')
        else:
            subject_id = subject['id']
            # Update image if provided
            if image_url:
                subject['image'] = image_url
            print(f'📂 Using existing subject: {subject_name} ({subject_id})')
        
        # Generate course ID and store course
        course_id = str(uuid.uuid4())
        course_data = {
            "id": course_id,
            "title": course_title,
            "subject_id": subject_id,
            "subject_name": subject_name,
            "user_id": user_id,
            "files": [os.path.basename(path) for path in temp_paths],
            "uploaded_at": datetime.now().isoformat(),
            "status": "processed"
        }
        
        # Store in courses_db under subject
        courses_db[subject_id].append(course_data)
        
        print(f'💾 Course stored: {course_id} under subject {subject_id}')
        
        # Mock successful response
        result = {
            "course_id": course_id,
            "subject_id": subject_id,
            "status": "success",
            "message": f"Course '{course_title}' uploaded successfully under '{subject_name}'"
        }
        
        # Clean up
        for temp_path in temp_paths:
            os.unlink(temp_path)
        
        print(f'✅ Course ingestion completed successfully')
        return result
        
    except Exception as e:
        print(f'❌ Course ingestion failed: {str(e)}')
        raise

@app.get('/health')
async def health():
    return {'status': 'healthy', 'services': {'course_ingestion': True}}

@app.get('/api/ai/subjects')
async def list_subjects(user_id: str):
    """List subjects for a user"""
    print(f'📂 Listing subjects for user: {user_id}')
    user_subjects = subjects_db.get(user_id, [])
    
    # Add course count to each subject
    for subject in user_subjects:
        subject['course_count'] = len(courses_db.get(subject['id'], []))
    
    return {'subjects': user_subjects}

@app.get('/api/ai/courses')
async def list_courses(user_id: str, subject_id: Optional[str] = None):
    """List courses for a user, optionally filtered by subject"""
    print(f'📚 Listing courses for user: {user_id}, subject: {subject_id}')
    
    if subject_id:
        # Get courses for specific subject
        courses = courses_db.get(subject_id, [])
    else:
        # Get all courses for user across all subjects
        courses = []
        user_subjects = subjects_db.get(user_id, [])
        for subject in user_subjects:
            subject_courses = courses_db.get(subject['id'], [])
            courses.extend(subject_courses)
    
    return {'courses': courses}

@app.post('/api/ai/planner/create-plan')
async def create_study_plan(request: StudyPlanRequest):
    """Create a study plan for a user"""
    print(f'📅 Creating study plan for user: {request.user_id}')
    
    # Mock study plan creation
    plan_id = str(uuid.uuid4())
    
    # Generate mock tasks
    time_per_task = max(30, request.available_time_minutes // 3)
    tasks = [
        {
            "id": str(uuid.uuid4()),
            "title": f"Study session: {request.goal[:50]}",
            "description": f"Focus on understanding {request.goal}",
            "duration_minutes": time_per_task,
            "status": "pending",
            "order": 1
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Practice exercises",
            "description": "Apply what you've learned through exercises",
            "duration_minutes": time_per_task,
            "status": "pending",
            "order": 2
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Review and summarize",
            "description": "Review key concepts and create summary notes",
            "duration_minutes": request.available_time_minutes - (2 * time_per_task),
            "status": "pending",
            "order": 3
        }
    ]
    
    study_plan = {
        "id": plan_id,
        "user_id": request.user_id,
        "goal": request.goal,
        "course_id": request.course_id,
        "tasks": tasks,
        "total_duration_minutes": request.available_time_minutes,
        "start_date": request.start_date,
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    # Store plan
    if request.user_id not in study_plans_db:
        study_plans_db[request.user_id] = []
    study_plans_db[request.user_id].append(study_plan)
    
    print(f'✅ Study plan created: {plan_id}')
    return {"plan": study_plan}

@app.post('/api/ai/coach/decision')
async def get_coach_decision(request: CoachDecisionRequest):
    """Get AI coach decision for current study session"""
    print(f'🤖 Getting coach decision for user: {request.user_id}')
    
    # Mock coach decision logic
    import random
    
    if request.do_not_disturb:
        action_type = "silence"
        message = "Do not disturb mode enabled"
    elif request.ignored_count >= 3:
        action_type = "silence"
        message = "You've ignored my suggestions. I'll be quiet for now."
    else:
        # Random coach actions for demo
        actions = [
            {"action_type": "take_break", "message": "You've been studying for a while. Time for a 5-minute break!"},
            {"action_type": "continue", "message": "Great focus! Keep up the good work!"},
            {"action_type": "change_technique", "message": "Try switching to active recall for better retention."},
            {"action_type": "silence", "message": ""}
        ]
        decision = random.choice(actions)
        action_type = decision["action_type"]
        message = decision["message"]
    
    coach_action = {
        "action_type": action_type,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f'🤖 Coach decision: {action_type}')
    return {"coach_action": coach_action}

@app.get('/api/ai/signals/current/{user_id}')
async def get_current_signals(user_id: str):
    """Get current focus and fatigue signals"""
    print(f'📊 Getting signals for user: {user_id}')
    
    # Mock signal data
    import random
    
    signals = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "focus": {
            "score": random.uniform(0.5, 1.0),
            "level": "high" if random.random() > 0.5 else "medium"
        },
        "fatigue": {
            "score": random.uniform(0.0, 0.5),
            "level": "low" if random.random() > 0.5 else "medium"
        }
    }
    
    return signals

@app.get('/debug/database')
async def debug_database():
    """Debug endpoint to inspect the in-memory database"""
    return {
        'total_users': len(subjects_db),
        'total_subjects': sum(len(subjects) for subjects in subjects_db.values()),
        'total_courses': sum(len(courses) for courses in courses_db.values()),
        'total_plans': sum(len(plans) for plans in study_plans_db.values()),
        'subjects': subjects_db,
        'courses': courses_db,
        'study_plans': study_plans_db
    }

if __name__ == "__main__":
    import uvicorn
    print('🚀 Starting minimal AI backend for course uploads...')
    uvicorn.run(app, host="0.0.0.0", port=8000)
