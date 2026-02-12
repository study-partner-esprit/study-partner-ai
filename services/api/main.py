"""FastAPI service for AI features: course ingestion, planning, coaching, and signals.

This service provides RESTful endpoints for the frontend to interact with all AI agents.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import tempfile
import os

from agents.course_ingestion.agent import ingest_course
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput
from services.ai_orchestrator.orchestrator import AIOrchestrator
from services.schedule_orchestrator.orchestrator import ScheduleOrchestrator
# from services.signal_processing_service.service import SignalProcessingService  # Disabled to prevent crashes
from agents.coach.models.schemas import CoachAction

app = FastAPI(title="Study Partner AI API", version="1.0.0")

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


class PlannerRequest(BaseModel):
    user_id: str
    goal: str
    available_time_minutes: int
    start_date: Optional[datetime] = None
    course_id: Optional[str] = None


class CoachRequest(BaseModel):
    user_id: str
    ignored_count: int = 0
    do_not_disturb: bool = False


class SignalProcessingRequest(BaseModel):
    user_id: str


# ==================== Course Ingestion Endpoints ====================

@app.post("/api/ai/courses/ingest")
async def ingest_course_endpoint(
    background_tasks: BackgroundTasks,
    course_title: str = Form(...),
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Ingest course materials (PDF files) and store in MongoDB.
    
    Args:
        course_title: Name of the course
        user_id: User uploading the course
        files: List of PDF files to process
    
    Returns:
        Course ID and processing status
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
        
        # Process course ingestion in background
        def process_ingestion():
            try:
                course_id = ingest_course(course_title, temp_files)
                # Cleanup temp files
                for tmp_file in temp_files:
                    try:
                        os.unlink(tmp_file)
                    except:
                        pass
                return course_id
            except Exception as e:
                print(f"Course ingestion error: {e}")
                for tmp_file in temp_files:
                    try:
                        os.unlink(tmp_file)
                    except:
                        pass
                raise
        
        # Start ingestion asynchronously
        background_tasks.add_task(process_ingestion)
        
        return {
            "status": "processing",
            "message": f"Course '{course_title}' is being processed",
            "files_count": len(temp_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Course ingestion failed: {str(e)}")


@app.get("/api/ai/courses/{course_id}")
async def get_course(course_id: str):
    """Get course details by ID."""
    try:
        from agents.course_ingestion.services.database_service import DatabaseService
        db = DatabaseService()
        course = db.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        return course
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch course: {str(e)}")


@app.get("/api/ai/courses")
async def list_courses(user_id: Optional[str] = None):
    """List all courses, optionally filtered by user."""
    try:
        from agents.course_ingestion.services.database_service import DatabaseService
        db = DatabaseService()
        # For now, return all courses (add user filtering later)
        courses = db.collection.find({}, {"_id": 1, "title": 1, "created_at": 1})
        return {"courses": list(courses)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list courses: {str(e)}")


# ==================== Planner Endpoints ====================

@app.post("/api/ai/planner/create-plan")
async def create_study_plan(request: PlannerRequest):
    """
    Create a personalized study plan using the planner agent.
    
    Args:
        request: PlannerRequest with goal, time, and optional course
    
    Returns:
        Study plan with task graph and scheduling
    """
    try:
        # Fetch course documents if course_id provided
        course_documents = None
        if request.course_id:
            from agents.course_ingestion.services.database_service import DatabaseService
            db = DatabaseService()
            course = db.get_course(request.course_id)
            if course:
                course_documents = course
        
        # Create planner input
        planner_input = PlannerInput(
            goal=request.goal,
            available_time_minutes=request.available_time_minutes,
            start_date=request.start_date or datetime.now(),
            course_documents=course_documents
        )
        
        # Run planner
        plan_output = get_planner_agent().plan(planner_input)
        
        # Save to MongoDB
        from agents.planner.rag.prompt_builder import SchedulingService
        scheduling_service = SchedulingService()
        schedule_id = scheduling_service.save_study_plan(
            user_id=request.user_id,
            study_plan=plan_output.model_dump()
        )
        
        return {
            "status": "success",
            "schedule_id": schedule_id,
            "plan": plan_output.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


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
