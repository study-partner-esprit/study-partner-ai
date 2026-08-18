"""Course upload and ingestion endpoints."""

import os
import tempfile
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TaskGenerationRequest(BaseModel):
    course_id: str
    user_id: str
    course_data: dict  # Contains title and topics


@router.post("/api/ai/courses/ingest")
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
        from agents.course_ingestion.agent import ingest_course
        from agents.course_ingestion.services.database_service import DatabaseService

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


@router.post("/api/ai/courses/generate-tasks")
async def generate_tasks_from_course_endpoint(request: TaskGenerationRequest):
    """
    Generate study tasks from a course using AI.

    Args:
        request: TaskGenerationRequest with course_id, user_id, and course_data

    Returns:
        List of generated tasks
    """
    try:
        from agents.course_ingestion.enrichment.task_generator import (
            generate_tasks_from_course,
            generate_tasks_simple,
        )

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
