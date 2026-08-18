"""FastAPI service for AI features: course ingestion, planning, coaching, and signals.

This service provides RESTful endpoints for the frontend to interact with all AI agents.
"""

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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
from agents.course_ingestion.enrichment.task_generator import (
    generate_tasks_from_course,
    generate_tasks_simple,
)
from agents.course_ingestion.services.database_service import DatabaseService
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput
from agents.coach.models.schemas import CoachAction
from services.ai_orchestrator.orchestrator import AIOrchestrator
from services.schedule_orchestrator.orchestrator import ScheduleOrchestrator
from services.signal_processing_service.focus_detector import get_focus_detector
from services.signal_processing_service.fatigue_detector import get_fatigue_detector
from pymongo import MongoClient
from bson import ObjectId
from utils.logger import get_logger

logger = get_logger(__name__)

# --- Environment validation (fail-fast on missing secrets) ---
_REQUIRED_ENV = ["GEMINI_API_KEY"]
for _key in _REQUIRED_ENV:
    if not os.getenv(_key):
        logger.warning(
            "missing_env_var",
            extra={
                "var": _key,
                "hint": "AI features that require this key will fail at runtime",
            },
        )


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
DB_NAME = os.getenv("DB_NAME", "study_partner")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
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


_signal_service = None


def get_signal_service():
    """Lazy-load the SignalProcessingService to avoid startup crashes."""
    global _signal_service
    if _signal_service is None:
        try:
            from services.signal_processing_service.service import (
                SignalProcessingService,
            )

            _signal_service = SignalProcessingService()
            logger.info(
                "SignalProcessingService initialised (ready=%s)",
                _signal_service.is_ready(),
            )
        except Exception as e:
            logger.warning("SignalProcessingService unavailable: %s", e)
            _signal_service = False  # sentinel: attempted and failed
    return _signal_service if _signal_service is not False else None


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
    calendar_events: Optional[list] = (
        []
    )  # User's blocked time slots from Node.js backend


class CoachRequest(BaseModel):
    user_id: str
    ignored_count: int = 0
    do_not_disturb: bool = False
    # Live signal overrides from frontend webcam pipeline
    focus_score: Optional[float] = None
    focus_state: Optional[str] = None
    fatigue_score: Optional[float] = None
    fatigue_state: Optional[str] = None


class SignalProcessingRequest(BaseModel):
    user_id: str


# ==================== Course Ingestion Endpoints ====================


@app.post("/api/ai/courses/ingest")
async def ingest_course_endpoint(
    course_title: str = Form(...),
    user_id: str = Form(...),
    subject_id: str = Form(...),
    files: List[UploadFile] = File(...),
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
                "topics": course_data.get("topics", []) if course_data else [],
            }

        finally:
            # Cleanup temp files
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Course processing failed: {str(e)}"
        )


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
        course_title = request.course_data.get("title", "Untitled Course")
        topics = request.course_data.get("topics", [])

        if not topics:
            raise HTTPException(
                status_code=400, detail="Course has no topics to generate tasks from"
            )

        # Generate tasks using AI
        try:
            tasks = generate_tasks_from_course(course_title, topics)

            # If AI generation fails, use fallback
            if not tasks:
                logger.warning(
                    f"AI task generation failed, using fallback for course {request.course_id}"
                )
                tasks = generate_tasks_simple(course_title, topics)

        except Exception as ai_error:
            logger.error(f"Error in AI task generation: {ai_error}")
            # Fallback to simple task generation
            tasks = generate_tasks_simple(course_title, topics)

        return {
            "success": True,
            "course_id": request.course_id,
            "tasks": tasks,
            "count": len(tasks),
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
            from agents.course_ingestion.services.database_service import (
                DatabaseService,
            )

            db_service = DatabaseService()
            course = db_service.get_course_by_id(request.course_id)
            if course:
                # Convert all ObjectIds to strings for JSON serialization
                course_knowledge = convert_objectid_to_str(course)
                print(
                    f"DEBUG: Course knowledge loaded for course_id {request.course_id}"
                )
                print(
                    f"DEBUG: Course has {len(course_knowledge.get('topics', []))} topics"
                )
                if (
                    course_knowledge.get("topics")
                    and len(course_knowledge["topics"]) > 0
                ):
                    first_topic = course_knowledge["topics"][0]
                    if "subtopics" in first_topic and len(first_topic["subtopics"]) > 0:
                        first_subtopic = first_topic["subtopics"][0]
                        has_tokenized = "tokenized_chunks" in first_subtopic
                        has_summary = "summary" in first_subtopic
                        print(
                            f"DEBUG: First subtopic has tokenized_chunks: {has_tokenized}, has summary: {has_summary}"
                        )
                        if has_tokenized:
                            print(
                                f"DEBUG: tokenized_chunks length: {len(first_subtopic['tokenized_chunks'])}"
                            )
                            print(
                                f"DEBUG: First chunk preview: {first_subtopic['tokenized_chunks'][0][:100]}..."
                            )
                        if has_summary:
                            print(
                                f"DEBUG: summary preview: {first_subtopic['summary'][:100]}..."
                            )

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

        print(
            f"DEBUG: Created planner input with goal: '{request.goal}', course_knowledge: {course_knowledge is not None}"
        )
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
                priority = "low"
            elif atomic_task.difficulty < 0.7:
                priority = "medium"
            else:
                priority = "high"

            task = {
                "title": atomic_task.title,
                "description": atomic_task.description,
                "priority": priority,
                "estimatedTime": atomic_task.estimated_minutes,
                "tags": (
                    [request.goal[:50]] if request.goal else []
                ),  # Use goal as a tag
            }
            tasks.append(task)

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "total_time": sum(t["estimatedTime"] for t in tasks),
            "warning": plan_output.warning if hasattr(plan_output, "warning") else None,
        }

    except Exception as e:
        logger.error(f"Study plan creation failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create study plan: {str(e)}"
        )


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


# ==================== Scheduler Endpoints ====================


class SchedulerRequest(BaseModel):
    user_id: str
    tasks: List[dict]  # List of task objects from Node.js
    calendar_events: Optional[List[dict]] = []
    max_minutes_per_day: int = 240
    allow_late_night: bool = False


def mix_tasks_intelligently(tasks: List) -> List:
    """
    Mix tasks from different subjects/plans to avoid scheduling all tasks
    from the same subject consecutively.

    Strategy:
    1. Group tasks by tags/subject (first tag indicates subject/course)
    2. Interleave tasks from different groups (round-robin)
    3. Maintain tasks without tags at the end

    Args:
        tasks: List of SchedulerTask objects

    Returns:
        Reordered list with mixed tasks
    """
    from collections import defaultdict

    # Group tasks by their primary tag (subject/course)
    groups = defaultdict(list)
    no_tag_tasks = []

    for task in tasks:
        if task.tags and len(task.tags) > 0:
            # Use first tag as grouping key (usually subject/course)
            primary_tag = task.tags[0]
            groups[primary_tag].append(task)
        else:
            no_tag_tasks.append(task)

    # If there's only one group or no groups, return original order
    if len(groups) <= 1:
        return tasks

    # Mix tasks using round-robin from each group
    mixed_tasks = []
    group_lists = list(groups.values())
    max_length = max(len(group) for group in group_lists)

    for i in range(max_length):
        for group in group_lists:
            if i < len(group):
                mixed_tasks.append(group[i])

    # Add tasks without tags at the end
    mixed_tasks.extend(no_tag_tasks)

    logger.info(
        f"Task mixing: {len(groups)} groups found, mixed {len(mixed_tasks)} tasks"
    )
    for tag, group in groups.items():
        logger.info(f"  Group '{tag}': {len(group)} tasks")

    return mixed_tasks


@app.post("/api/ai/scheduler/schedule")
async def schedule_tasks(request: SchedulerRequest):
    """
    Schedule existing tasks using the AI scheduler agent.

    Args:
        request: SchedulerRequest with tasks and scheduling constraints

    Returns:
        Schedule with sessions, total time, and metadata
    """
    try:
        from agents.scheduler.agent import SchedulerAgent, SchedulingContext
        from models.task import Task as SchedulerTask

        # Convert Node.js tasks to scheduler Task objects
        scheduler_tasks = []
        for task in request.tasks:
            # Map priority to difficulty (0-1 scale)
            priority = task.get("priority", "medium")
            if priority == "low":
                difficulty = 0.3
            elif priority == "high":
                difficulty = 0.8
            else:
                difficulty = 0.5

            scheduler_task = SchedulerTask(
                task_id=str(task.get("_id", task.get("id", ""))),
                user_id=request.user_id,
                title=task.get("title", "Untitled Task"),
                description=task.get("description", ""),
                priority=priority,
                difficulty=str(difficulty),
                estimated_duration=task.get("estimatedTime", 30),
                status=task.get("status", "todo"),
                tags=task.get("tags", []),
                prerequisites=task.get("prerequisites", []),
            )
            scheduler_tasks.append(scheduler_task)

        # Mix tasks from different subjects/courses intelligently
        mixed_tasks = mix_tasks_intelligently(scheduler_tasks)

        # Create  scheduling context
        context = SchedulingContext(
            calendar_events=request.calendar_events,
            max_minutes_per_day=request.max_minutes_per_day,
            allow_late_night=request.allow_late_night,
        )

        # Build schedule with mixed tasks
        scheduler = SchedulerAgent()
        study_plan = scheduler.build_schedule(
            tasks=mixed_tasks,
            context=context,
        )

        # Convert sessions to frontend format
        sessions = []
        for session in study_plan.sessions:
            sessions.append(
                {
                    "taskId": session.task_id,
                    "title": next(
                        (
                            t.title
                            for t in scheduler_tasks
                            if t.task_id == session.task_id
                        ),
                        "Unknown",
                    ),
                    "startTime": session.start_datetime.isoformat(),
                    "endTime": session.end_datetime.isoformat(),
                    "estimatedMinutes": int(
                        (session.end_datetime - session.start_datetime).total_seconds()
                        / 60
                    ),
                    "slotScore": session.slot_score,
                }
            )

        return {
            "success": True,
            "schedule": {
                "sessions": sessions,
                "totalMinutes": study_plan.total_minutes,
                "spanDays": study_plan.span_days,
                "skippedTasks": study_plan.skipped_tasks,
                "fallbackUsed": study_plan.fallback_used,
            },
        }

    except Exception as e:
        logger.error(f"Task scheduling failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule tasks: {str(e)}"
        )


# ==================== Coach Endpoints ====================


@app.post("/api/ai/coach/decision")
async def get_coach_decision(
    request: CoachRequest,
    x_trace_id: Optional[str] = Header(None, alias="x-trace-id"),
):
    """
    Get real-time coaching decision based on current context.

    Args:
        request: CoachRequest with user ID and context
        x_trace_id: Optional request trace ID forwarded from the API gateway

    Returns:
        CoachAction with decision and optional schedule changes
    """
    try:
        # Run coach through orchestrator — propagate trace_id
        coach_action = get_ai_orchestrator().run_coach(
            user_id=request.user_id,
            current_time=datetime.now(),
            ignored_count=request.ignored_count,
            do_not_disturb=request.do_not_disturb,
            trace_id=x_trace_id,
            live_focus_score=request.focus_score,
            live_focus_state=request.focus_state,
            live_fatigue_score=request.fatigue_score,
            live_fatigue_state=request.fatigue_state,
        )

        # If coach suggests schedule changes, implement them
        if coach_action.schedule_changes:
            schedule_result = get_schedule_orchestrator().process_coach_action(
                coach_action=coach_action,
                user_id=request.user_id,
                current_time=datetime.now(),
            )
            return {
                "coach_action": coach_action.model_dump(),
                "schedule_update": schedule_result,
            }

        return {"coach_action": coach_action.model_dump(), "schedule_update": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach decision failed: {str(e)}")


@app.get("/api/ai/coach/history/{user_id}")
async def get_coach_history(user_id: str, limit: int = 20):
    """Get recent coaching action history for a user."""
    try:
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository()
        history = repo.get_recent_actions(user_id, limit=limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch history: {str(e)}"
        )


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
            raise HTTPException(
                status_code=503, detail="Signal processing service is disabled"
            )

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
                "confidence": snapshot.focus_confidence,
            },
            "fatigue": {
                "state": snapshot.fatigue_state,
                "score": snapshot.fatigue_score,
                "confidence": snapshot.fatigue_confidence,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch signals: {str(e)}"
        )


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
                    "fatigue": {"state": s.fatigue_state, "score": s.fatigue_score},
                }
                for s in snapshots
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch history: {str(e)}"
        )


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
            video_frame=None,
        )

        return {
            "status": "success",
            "snapshot": {
                "focus": {"state": snapshot.focus_state, "score": snapshot.focus_score},
                "fatigue": {
                    "state": snapshot.fatigue_state,
                    "score": snapshot.fatigue_score,
                },
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Signal processing failed: {str(e)}"
        )


@app.post("/api/ai/signals/analyze-frame")
async def analyze_frame(user_id: str = Form(...), frame: UploadFile = File(...)):
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

        # Run both detectors (per-user fatigue detector for state persistence)
        focus_detector = get_focus_detector()
        fatigue_detector = get_fatigue_detector(user_id)

        focus_result = focus_detector.analyze_frame(frame_data)
        fatigue_result = fatigue_detector.analyze_frame(frame_data)

        # Combine results
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "focus": {
                "score": focus_result.get("focus_score", 0),
                "state": focus_result.get("focus_state", "unknown"),
                "confidence": focus_result.get("confidence", 0),
            },
            "fatigue": {
                "score": fatigue_result.get("fatigue_score", 0),
                "state": fatigue_result.get("fatigue_state", "unknown"),
                "indicators": fatigue_result.get("indicators", {}),
                "confidence": fatigue_result.get("confidence", 0),
            },
        }

        # Save to MongoDB signals collection
        signals_collection.insert_one(analysis)

        return analysis

    except Exception as e:
        logger.warning("frame_analysis_failed", extra={"error": str(e)})
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
            signal["_id"] = str(signal["_id"])

        return {"signals": signals}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch signals: {str(e)}"
        )


# ==================== New planner / scheduler / signal endpoints ====================


class RecordCompletionRequest(BaseModel):
    user_id: str
    task_id: str
    estimated_minutes: int
    actual_minutes: int
    subject_tag: str = ""


@app.post("/api/ai/planner/record-completion")
async def record_task_completion(request: RecordCompletionRequest):
    """
    Record actual vs estimated task completion time to improve future pacing.
    """
    try:
        from agents.planner.memory.pacing_store import PacingStore

        store = PacingStore()
        store.record_task_completion(
            user_id=request.user_id,
            task_id=request.task_id,
            estimated_minutes=request.estimated_minutes,
            actual_minutes=request.actual_minutes,
            subject_tag=request.subject_tag,
        )
        factor = store.get_user_pacing_factor(request.user_id, request.subject_tag)
        return {"status": "ok", "new_pacing_factor": factor}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Review / Spaced Repetition Endpoints ──────────────────────────────


class ScheduleReviewRequest(BaseModel):
    user_id: str
    task_id: str
    task_title: str
    subject_tag: str = ""
    key_concepts: List[str] = []
    difficulty: str = "medium"


@app.post("/api/ai/reviews/schedule")
async def schedule_review(request: ScheduleReviewRequest):
    """
    Schedule the first spaced repetition review for a completed task.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        result = inserter.schedule_review(
            user_id=request.user_id,
            task_id=request.task_id,
            task_title=request.task_title,
            subject_tag=request.subject_tag,
            key_concepts=request.key_concepts,
            difficulty=request.difficulty,
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to schedule review")
        return {"status": "ok", "review": convert_objectid_to_str(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RecordReviewResultRequest(BaseModel):
    user_id: str
    review_id: str
    quality_score: int  # 0-5


@app.post("/api/ai/reviews/record-result")
async def record_review_result(request: RecordReviewResultRequest):
    """
    Record the result of a review session and schedule the next review.

    Quality scores:
      0 = complete blackout
      1 = incorrect, recognized correct answer
      2 = incorrect, answer seemed easy to recall
      3 = correct with significant difficulty
      4 = correct after hesitation
      5 = perfect, instant recall
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        result = inserter.record_review_result(
            user_id=request.user_id,
            review_id=request.review_id,
            quality_score=request.quality_score,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Review not found")
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/reviews/pending/{user_id}")
async def get_pending_reviews(user_id: str, limit: int = 20):
    """
    Get all pending and overdue reviews for a user.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        reviews = inserter.get_pending_reviews(user_id, limit=limit)
        return {"status": "ok", "reviews": reviews, "count": len(reviews)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/reviews/stats/{user_id}")
async def get_review_stats(user_id: str):
    """
    Get review statistics for a user.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        stats = inserter.get_review_stats(user_id)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RescheduleRequest(BaseModel):
    user_id: str
    reason: str = "manual"


class SchedulerOptimizeRequest(BaseModel):
    user_id: str
    reason: str = "manual_optimize"


class SchedulerApplyCoachActionRequest(BaseModel):
    user_id: str
    coach_action: Dict[str, Any]


class EvaluateSessionRequest(BaseModel):
    user_id: str
    session_duration_minutes: int = 0
    focus_score: float = 0.0
    completed_tasks: int = 0
    skipped_tasks: int = 0


@app.post("/api/ai/vector/rebuild/{course_id}")
async def rebuild_vector_index(course_id: str):
    """Attempt to load/rebuild a course vector index from persisted stores."""
    try:
        from services.vector_store.adapter import get_vector_store

        store = get_vector_store()
        loaded = store.load_course(course_id)
        if not loaded:
            raise HTTPException(status_code=404, detail="Course index not found")

        return {
            "status": "ok",
            "course_id": course_id,
            "loaded": True,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/vector/status/{course_id}")
async def vector_index_status(course_id: str):
    """Get in-memory status of a course vector index."""
    try:
        from services.vector_store.adapter import get_vector_store

        store = get_vector_store()
        loaded = store.load_course(course_id)
        return {
            "status": "ok",
            "course_id": course_id,
            "loaded": loaded,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/scheduler/reschedule")
async def reschedule(request: RescheduleRequest):
    """
    Trigger a re-schedule for a user (e.g. after a long break or plan change).
    Returns updated schedule.
    """
    try:
        db_svc = DatabaseService()
        study_plan = db_svc.get_latest_study_plan(request.user_id)
        if study_plan is None:
            raise HTTPException(status_code=404, detail="No study plan found")
        return {
            "status": "ok",
            "message": "Reschedule queued",
            "user_id": request.user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/scheduler/apply-coach-action")
async def apply_coach_action_to_schedule(request: SchedulerApplyCoachActionRequest):
    """Apply explicit coach action schedule changes for a user."""
    try:
        coach_action = CoachAction(**request.coach_action)
        result = get_schedule_orchestrator().process_coach_action(
            coach_action=coach_action,
            user_id=request.user_id,
            current_time=datetime.now(),
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/scheduler/status/{user_id}")
async def scheduler_status(user_id: str):
    """Get current schedule status for a user."""
    try:
        status = get_schedule_orchestrator().get_schedule_status(user_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/ai/scheduler/optimize")
async def optimize_schedule(request: SchedulerOptimizeRequest):
    """Run schedule optimization for a user and persist snapshot."""
    try:
        result = get_schedule_orchestrator().optimize_schedule(
            request.user_id, reason=request.reason
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/evaluator/session")
async def evaluate_session(request: EvaluateSessionRequest):
    """Evaluate a completed study session and return coaching-quality feedback."""
    try:
        from agents.evaluator.src.evaluator.evaluator_agent import EvaluatorAgent

        evaluator = EvaluatorAgent(require_llm=False)
        evaluation = evaluator.evaluate(
            session_duration_minutes=request.session_duration_minutes,
            focus_score=request.focus_score,
            completed_tasks=request.completed_tasks,
            skipped_tasks=request.skipped_tasks,
        )
        return {
            "status": "ok",
            "user_id": request.user_id,
            "evaluation": evaluation.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CalibrateSignalsRequest(BaseModel):
    user_id: str
    baseline_focus: float = 0.5
    baseline_fatigue: float = 0.2


@app.post("/api/ai/signals/calibrate")
async def calibrate_signals(request: CalibrateSignalsRequest):
    """
    Reset the EMA state for a user (e.g. at the start of a new study session).
    Optionally seed with known baseline values.
    """
    try:
        from services.signal_processing_service.smoothing import get_ema_state

        ema = get_ema_state()
        ema.reset(request.user_id)
        # Seed with baseline
        ema.update(request.user_id, request.baseline_focus, request.baseline_fatigue)
        return {"status": "ok", "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FatigueResetRequest(BaseModel):
    user_id: str


@app.post("/api/ai/signals/fatigue/reset")
async def reset_fatigue(request: FatigueResetRequest):
    """
    Reset the per-user fatigue detector state (blink/yawn counters).
    Call at the start of a new study session.
    """
    try:
        from services.signal_processing_service.fatigue_detector import (
            reset_fatigue_detector,
        )

        reset_fatigue_detector(request.user_id)
        return {"status": "ok", "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Socratic Evaluation Endpoints ====================


class SocraticStartRequest(BaseModel):
    user_id: str
    task_title: str
    task_description: str
    task_details: str
    max_attempts: int = 5


class SocraticAnswerRequest(BaseModel):
    session_id: str
    user_answer: str


@app.post("/api/ai/evaluator/socratic/start")
async def socratic_start(request: SocraticStartRequest):
    """
    Start a new Socratic evaluation session.
    Returns the first question to ask the student.
    """
    try:
        from agents.evaluator.src.evaluator.evaluator_agent import EvaluatorAgent

        evaluator = EvaluatorAgent(require_llm=False)
        result = evaluator.start_session(
            task_title=request.task_title,
            task_description=request.task_description,
            task_details=request.task_details,
            max_attempts=request.max_attempts,
        )
        return {
            "status": "ok",
            "user_id": request.user_id,
            "session_id": result["session_id"],
            "question": result["question"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/evaluator/socratic/answer")
async def socratic_answer(request: SocraticAnswerRequest):
    """
    Submit a student answer to an ongoing Socratic evaluation session.
    Returns the next question or final evaluation result.
    """
    try:
        from agents.evaluator.src.evaluator.evaluator_agent import EvaluatorAgent

        evaluator = EvaluatorAgent(require_llm=False)
        result = evaluator.handle_user_answer(request.session_id, request.user_answer)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Search Agent ====================


class SearchAskRequest(BaseModel):
    question: str
    user_id: Optional[str] = ""
    session_id: Optional[str] = ""


@app.post("/api/ai/search/ask")
async def search_ask(
    req: SearchAskRequest, x_trace_id: Optional[str] = Header(None, alias="x-trace-id")
):
    """Web-search a question using Apify + extract text + answer via LM Studio (Qwen)."""
    import uuid

    trace_id = x_trace_id or str(uuid.uuid4())
    try:
        from agents.search.agent import process_question

        result = process_question(
            req.question,
            user_id=req.user_id,
            session_id=req.session_id,
            trace_id=trace_id,
        )
        if not result.get("success") and result.get("error") == "No question provided":
            raise HTTPException(status_code=400, detail="No question provided")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "search_ask_error", extra={"error": str(exc), "trace_id": trace_id}
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ai/search/history/{user_id}")
async def search_history(user_id: str, limit: int = 20):
    """Return the last N search exchanges for a user."""
    try:
        from agents.search.services.search_repository import SearchRepository

        repo = SearchRepository()
        items = repo.get_history(user_id, limit=limit)
        return {"user_id": user_id, "history": items}
    except Exception as exc:
        logger.warning("search_history_error", extra={"error": str(exc)})
        return {"user_id": user_id, "history": []}


@app.delete("/api/ai/search/history/{user_id}")
async def search_history_clear(user_id: str):
    """Clear all search history for a user."""
    try:
        from agents.search.services.search_repository import SearchRepository

        repo = SearchRepository()
        repo.clear_history(user_id)
        return {"success": True}
    except Exception as exc:
        logger.warning("search_history_clear_error", extra={"error": str(exc)})
        return {"success": False, "error": str(exc)}



# ==================== Health Check ====================


@app.get("/health")
async def health_check():
    """Health check endpoint - fast response for docker healthcheck.
    
    Does NOT initialize services to avoid timeouts during startup.
    Services are lazy-loaded on first actual request.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
