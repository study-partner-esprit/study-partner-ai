"""FastAPI service for AI features: course ingestion, planning, coaching, and signals.

This service provides RESTful endpoints for the frontend to interact with all AI agents.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path
# Ensure project root is on sys.path before importing local packages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.course_ingestion.agent import ingest_course
from agents.course_ingestion.enrichment.task_generator import generate_tasks_from_course, generate_tasks_simple
from agents.course_ingestion.services.database_service import DatabaseService
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput
from services.ai_orchestrator.orchestrator import AIOrchestrator
from services.schedule_orchestrator.orchestrator import ScheduleOrchestrator
# from services.signal_processing_service.service import SignalProcessingService  # Disabled to prevent crashes
from services.signal_processing_service.focus_detector import get_focus_detector
from services.signal_processing_service.fatigue_detector import get_fatigue_detector
from pymongo import MongoClient
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)
 

def convert_objectid_to_str(obj):
    """Recursively convert ObjectId to string in nested dictionaries and lists."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    else:
        return obj

app = FastAPI(title="Study Partner AI API", version="1.0.0")

# MongoDB connection (only for AI-specific data)
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["study_partner"]
signals_collection = db["signals"]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy-load to avoid crashes)
_planner_agent = None
_ai_orchestrator = None
_schedule_orchestrator = None

def get_planner_agent():
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent()
    return _planner_agent

def get_ai_orchestrator():
    global _ai_orchestrator
    if _ai_orchestrator is None:
        _ai_orchestrator = AIOrchestrator()
    return _ai_orchestrator

def get_schedule_orchestrator():
    global _schedule_orchestrator
    if _schedule_orchestrator is None:
        _schedule_orchestrator = ScheduleOrchestrator()
    return _schedule_orchestrator


def get_signal_service():
    """Signal service disabled to prevent crashes."""
    return None


# ==================== Pydantic Models ====================

class CourseIngestionRequest(BaseModel):
    course_title: str
    user_id: str


class TaskGenerationRequest(BaseModel):
    course_id: str
    user_id: str
    course_data: dict  # Contains title and topics


class PlannerRequest(BaseModel):
    user_id: str
    goal: str
    available_time_minutes: int
    start_date: Optional[datetime] = None
    course_id: Optional[str] = None
    calendar_events: Optional[list] = []  # User's blocked time slots from Node.js backend


class CoachRequest(BaseModel):
    user_id: str
    ignored_count: int = 0
    do_not_disturb: bool = False


class SignalProcessingRequest(BaseModel):
    user_id: str


# ==================== Course Ingestion Endpoints ====================

@app.post("/api/ai/courses/ingest")
async def ingest_course_endpoint(
    course_title: str = Form(...),
    user_id: str = Form(...),
    subject_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Process course materials and return structured data.
    
    Args:
        course_title: Name of the course
        user_id: User uploading the course
        subject_id: Subject this course belongs to
        files: List of PDF files to process
    
    Returns:
        Processed course data with topics, subtopics, etc.
    """
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_files.append(tmp.name)
        
        try:
            # Process course ingestion synchronously
            course_id = ingest_course(course_title, temp_files)
            
            # Get the processed course data
            db = DatabaseService()
            course_data = db.get_course_by_id(course_id)
            
            # Return course data with topics
            return {
                "course_id": course_id,
                "user_id": user_id,
                "subject_id": subject_id,
                "files_count": len(temp_files),
                "processed_at": datetime.now().isoformat(),
                "course_title": course_title,
                "topics": course_data.get("topics", []) if course_data else []
            }
            
        finally:
            # Cleanup temp files
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Course processing failed: {str(e)}")


@app.post("/api/ai/courses/generate-tasks")
async def generate_tasks_from_course_endpoint(request: TaskGenerationRequest):
    """
    Generate study tasks from a course using AI.
    
    Args:
        request: TaskGenerationRequest with course_id, user_id, and course_data
    
    Returns:
        List of generated tasks
    """
    try:
        course_title = request.course_data.get('title', 'Untitled Course')
        topics = request.course_data.get('topics', [])
        
        if not topics:
            raise HTTPException(status_code=400, detail="Course has no topics to generate tasks from")
        
        # Generate tasks using AI
        try:
            tasks = generate_tasks_from_course(course_title, topics)
            
            # If AI generation fails, use fallback
            if not tasks:
                logger.warning(f"AI task generation failed, using fallback for course {request.course_id}")
                tasks = generate_tasks_simple(course_title, topics)
        
        except Exception as ai_error:
            logger.error(f"Error in AI task generation: {ai_error}")
            # Fallback to simple task generation
            tasks = generate_tasks_simple(course_title, topics)
        
        return {
            "success": True,
            "course_id": request.course_id,
            "tasks": tasks,
            "count": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"Task generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task generation failed: {str(e)}")


# ==================== Planner Endpoints ====================

@app.post("/api/ai/planner/create-plan")
async def create_study_plan(request: PlannerRequest):
    """
    Create a personalized study plan by generating tasks.
    
    This endpoint creates tasks directly instead of a separate study plan concept.
    Tasks are saved to the database and can be accessed via /api/v1/study/tasks
    
    Args:
        request: PlannerRequest with goal, time, and optional course
    
    Returns:
        List of generated tasks with metadata
    """
    try:
        # Fetch course documents if course_id provided
        course_knowledge = None
        if request.course_id:
            from agents.course_ingestion.services.database_service import DatabaseService
            db_service = DatabaseService()
            course = db_service.get_course_by_id(request.course_id)
            if course:
                # Convert all ObjectIds to strings for JSON serialization
                course_knowledge = convert_objectid_to_str(course)
                print(f"DEBUG: Course knowledge loaded for course_id {request.course_id}")
                print(f"DEBUG: Course has {len(course_knowledge.get('topics', []))} topics")
                if course_knowledge.get('topics') and len(course_knowledge['topics']) > 0:
                    first_topic = course_knowledge['topics'][0]
                    if 'subtopics' in first_topic and len(first_topic['subtopics']) > 0:
                        first_subtopic = first_topic['subtopics'][0]
                        has_tokenized = 'tokenized_chunks' in first_subtopic
                        has_summary = 'summary' in first_subtopic
                        print(f"DEBUG: First subtopic has tokenized_chunks: {has_tokenized}, has summary: {has_summary}")
                        if has_tokenized:
                            print(f"DEBUG: tokenized_chunks length: {len(first_subtopic['tokenized_chunks'])}")
                            print(f"DEBUG: First chunk preview: {first_subtopic['tokenized_chunks'][0][:100]}...")
                        if has_summary:
                            print(f"DEBUG: summary preview: {first_subtopic['summary'][:100]}...")
        
        # Create planner input
        if request.start_date:
            deadline_iso = request.start_date.isoformat()
        else:
            deadline_iso = (datetime.now() + timedelta(days=7)).isoformat()
        
        planner_input = PlannerInput(
            goal=request.goal,
            deadline_iso=deadline_iso,
            available_minutes=request.available_time_minutes,
            user_id=request.user_id,
            course_knowledge=course_knowledge,
        )
        
        print(f"DEBUG: Created planner input with goal: '{request.goal}', course_knowledge: {course_knowledge is not None}")
        if course_knowledge:
            print(f"DEBUG: Course title: {course_knowledge.get('course_title', 'N/A')}")
        
        # Generate plan using planner agent
        planner_agent = get_planner_agent()
        plan_output = planner_agent.plan(planner_input)
        
        # Convert AtomicTasks to Task format for database
        tasks = []
        for atomic_task in plan_output.task_graph.tasks:
            # Map difficulty to priority
            if atomic_task.difficulty < 0.4:
                priority = 'low'
            elif atomic_task.difficulty < 0.7:
                priority = 'medium'
            else:
                priority = 'high'
            
            task = {
                'title': atomic_task.title,
                'description': atomic_task.description,
                'priority': priority,
                'estimatedTime': atomic_task.estimated_minutes,
                'tags': [request.goal[:50]] if request.goal else [],  # Use goal as a tag
            }
            tasks.append(task)
        
        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "total_time": sum(t['estimatedTime'] for t in tasks),
            "warning": plan_output.warning if hasattr(plan_output, 'warning') else None
        }
        
    except Exception as e:
        logger.error(f"Study plan creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create study plan: {str(e)}")


@app.get("/api/ai/planner/plans/{user_id}")
async def get_user_plans(user_id: str):
    """Get all study plans for a user."""
    try:
        from agents.planner.rag.prompt_builder import SchedulingService
        scheduling_service = SchedulingService()
        plans = scheduling_service.get_user_plans(user_id)
        return {"plans": plans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch plans: {str(e)}")


# ==================== Coach Endpoints ====================

@app.post("/api/ai/coach/decision")
async def get_coach_decision(request: CoachRequest):
    """
    Get real-time coaching decision based on current context.
    
    Args:
        request: CoachRequest with user ID and context
    
    Returns:
        CoachAction with decision and optional schedule changes
    """
    try:
        # Run coach through orchestrator
        coach_action = get_ai_orchestrator().run_coach(
            user_id=request.user_id,
            current_time=datetime.now(),
            ignored_count=request.ignored_count,
            do_not_disturb=request.do_not_disturb
        )
        
        # If coach suggests schedule changes, implement them
        if coach_action.schedule_changes:
            schedule_result = get_schedule_orchestrator().process_coach_action(
                coach_action=coach_action,
                user_id=request.user_id,
                current_time=datetime.now()
            )
            return {
                "coach_action": coach_action.model_dump(),
                "schedule_update": schedule_result
            }
        
        return {
            "coach_action": coach_action.model_dump(),
            "schedule_update": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach decision failed: {str(e)}")


@app.get("/api/ai/coach/history/{user_id}")
async def get_coach_history(user_id: str, limit: int = 20):
    """Get coaching history for a user."""
    try:
        # TODO: Implement coach history collection
        return {"history": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


# ==================== Signal Processing Endpoints ====================

@app.get("/api/ai/signals/current/{user_id}")
async def get_current_signals(user_id: str):
    """
    Get the current signal snapshot (focus and fatigue) for a user.
    
    Args:
        user_id: User identifier
    
    Returns:
        Latest signal snapshot with focus and fatigue data
    """
    try:
        signal_service = get_signal_service()
        if signal_service is None:
            raise HTTPException(status_code=503, detail="Signal processing service is disabled")
        
        snapshot = signal_service.get_latest_snapshot(user_id)
        if snapshot is None:
            # Generate a new snapshot if none exists
            snapshot = signal_service.get_current_signal_snapshot(user_id)
        
        return {
            "user_id": user_id,
            "timestamp": snapshot.timestamp,
            "focus": {
                "state": snapshot.focus_state,
                "score": snapshot.focus_score,
                "confidence": snapshot.focus_confidence
            },
            "fatigue": {
                "state": snapshot.fatigue_state,
                "score": snapshot.fatigue_score,
                "confidence": snapshot.fatigue_confidence
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals: {str(e)}")


@app.get("/api/ai/signals/history/{user_id}")
async def get_signal_history(user_id: str, limit: int = 50):
    """Get signal history for a user."""
    try:
        snapshots = get_signal_service().repository.get_signal_history(user_id, limit)
        return {
            "signals": [
                {
                    "timestamp": s.timestamp,
                    "focus": {"state": s.focus_state, "score": s.focus_score},
                    "fatigue": {"state": s.fatigue_state, "score": s.fatigue_score}
                }
                for s in snapshots
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.post("/api/ai/signals/process")
async def process_signals(request: SignalProcessingRequest):
    """
    Manually trigger signal processing for a user.
    
    This endpoint would typically be called by a frontend during an active study session.
    """
    try:
        snapshot = get_signal_service().get_current_signal_snapshot(
            user_id=request.user_id,
            video_features=None,  # Frontend should send video data
            video_frame=None
        )
        
        return {
            "status": "success",
            "snapshot": {
                "focus": {"state": snapshot.focus_state, "score": snapshot.focus_score},
                "fatigue": {"state": snapshot.fatigue_state, "score": snapshot.fatigue_score}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal processing failed: {str(e)}")


@app.post("/api/ai/signals/analyze-frame")
async def analyze_frame(
    user_id: str = Form(...),
    frame: UploadFile = File(...)
):
    """
    Analyze a video frame for focus and fatigue detection.
    
    Args:
        user_id: User ID
        frame: Video frame image file (JPEG/PNG)
    
    Returns:
        Combined focus and fatigue analysis results
    """
    try:
        # Read frame data
        frame_data = await frame.read()
        
        # Run both detectors
        focus_detector = get_focus_detector()
        fatigue_detector = get_fatigue_detector()
        
        focus_result = focus_detector.analyze_frame(frame_data)
        fatigue_result = fatigue_detector.analyze_frame(frame_data)
        
        # Combine results
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "focus": {
                "score": focus_result.get("focus_score", 0),
                "state": focus_result.get("focus_state", "unknown"),
                "confidence": focus_result.get("confidence", 0)
            },
            "fatigue": {
                "score": fatigue_result.get("fatigue_score", 0),
                "state": fatigue_result.get("fatigue_state", "unknown"),
                "indicators": fatigue_result.get("indicators", {}),
                "confidence": fatigue_result.get("confidence", 0)
            }
        }
        
        # Save to MongoDB signals collection
        signals_collection.insert_one(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")


@app.get("/api/ai/signals/latest/{user_id}")
async def get_latest_signals(user_id: str, limit: int = 10):
    """
    Get the most recent signal analysis results for a user.
    
    Args:
        user_id: User ID
        limit: Number of results to return
    
    Returns:
        List of recent signal analyses
    """
    try:
        signals = list(
            signals_collection.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Convert ObjectId to string
        for signal in signals:
            signal['_id'] = str(signal['_id'])
        
        return {"signals": signals}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals: {str(e)}")


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "planner": get_planner_agent() is not None,
            "coach": get_ai_orchestrator() is not None,
            "signals": get_signal_service() is not None,
            "scheduler": get_schedule_orchestrator() is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
