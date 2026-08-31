"""Course catalog reader for the coach (F03 / COACH-14).

Reads the shared course catalog directly from Mongo — the same `study_partner`
DB that `PlannerRepository` and `CoachHistoryRepository` use — so the coach
can ground nudges in the student's enrolled courses and subjects instead of
reasoning from task titles alone:

- `fetch_catalog` — the user's NEWEST ≤ `COURSE_CATALOG_MAX` (10) courses,
  each reduced to subject title + course title + ≤ `KEY_CONCEPTS_MAX` (15)
  key concepts (flattened from `topics[].subtopics[].key_concepts`). No files,
  URLs, descriptions, ids or any other field leaves the DB — nothing but
  subject/title/concepts can reach the prompt, and those are user-supplied
  content wrapped as UNTRUSTED DATA at prompt-build time (COACH-03/12).
- `fetch_current_task` — the live StudySession's in-progress task
  (`taskProgress.tasks[currentTaskIndex]`), mapped to its course subject via
  the session's `courseId` → `courses.subjectId` → `subjects.name`. This is
  the COACH-14 AC#2 "current task → its course/subject" mapping.

Every read is failure-safe: a catalog outage yields `[]`/`None` and the
orchestrator degrades to task-title-only context — the coach job never fails.
Identity (user_id, session_id) always comes from trusted callers, never from
pasted catalog content.
"""

import os
from typing import List, Optional, Dict

from bson import ObjectId
from pymongo import MongoClient

from agents.coach.models.schemas import CourseContext

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "study_partner")
COURSES_COLLECTION = os.getenv("COURSES_COLLECTION", "courses")
SUBJECTS_COLLECTION = os.getenv("SUBJECTS_COLLECTION", "subjects")
STUDY_SESSIONS_COLLECTION = os.getenv("STUDY_SESSIONS_COLLECTION", "studysessions")

COURSE_CATALOG_MAX = 10
KEY_CONCEPTS_MAX = 15
CONCEPT_MAX_CHARS = 100
SUBJECT_MAX_CHARS = 60
TITLE_MAX_CHARS = 100

_COURSE_PROJECTION = {"title": 1, "subjectId": 1, "topics": 1}
_SESSION_PROJECTION = {"courseId": 1, "taskProgress": 1}
_SUBJECT_PROJECTION = {"name": 1}
_COURSE_SUBJECT_PROJECTION = {"subjectId": 1}


class CourseRepository:
    def __init__(self, client: Optional[MongoClient] = None, db=None):
        if db is not None:  # tests inject a fake in-memory DB
            self._db = db
            return
        self._client = client or MongoClient(MONGO_URI)
        self._db = self._client[DB_NAME]

    # ------------------------------------------------------------------ catalog

    def fetch_catalog(self, user_id: str) -> List[CourseContext]:
        """The user's newest ≤ COURSE_CATALOG_MAX courses, bounded to
        subject/title/key_concepts. Any failure → `[]` (task-title-only)."""
        try:
            courses = list(
                self._db[COURSES_COLLECTION]
                .find({COURSES_USER_ID_FIELD: user_id}, _COURSE_PROJECTION)
                .sort([("createdAt", -1)])
                .limit(COURSE_CATALOG_MAX)
            )
        except Exception:
            return []

        subject_names = self._subject_names(courses)
        contexts: List[CourseContext] = []
        for course in courses:
            sid = course.get("subjectId")
            subject = "Unknown"
            if sid:
                subject = subject_names.get(str(sid)) or "Unknown"
            contexts.append(
                CourseContext(
                    subject=subject[:SUBJECT_MAX_CHARS],
                    title=(course.get("title") or "Untitled")[:TITLE_MAX_CHARS],
                    key_concepts=self._key_concepts(course),
                )
            )
        return contexts

    def _subject_names(self, courses: List[dict]) -> Dict[str, str]:
        """Map `str(subjectId)` → subject name for the courses at hand."""
        ids = [self._as_object_id(c.get("subjectId")) for c in courses]
        ids = [i for i in ids if i is not None]
        if not ids:
            return {}
        try:
            docs = list(
                self._db[SUBJECTS_COLLECTION].find(
                    {"_id": {"$in": ids}}, _SUBJECT_PROJECTION
                )
            )
        except Exception:
            return {}
        return {str(d.get("_id")): d.get("name") or "" for d in docs}

    def _key_concepts(self, course: dict) -> List[str]:
        """Flatten + dedupe topics[].subtopics[].key_concepts, capped + trimmed."""
        concepts: List[str] = []
        seen = set()
        for topic in course.get("topics") or []:
            for sub in topic.get("subtopics") or []:
                for kc in sub.get("key_concepts") or []:
                    if not isinstance(kc, str) or not kc.strip():
                        continue
                    k = kc.strip()
                    if k not in seen:
                        seen.add(k)
                        concepts.append(k[:CONCEPT_MAX_CHARS])
                    if len(concepts) >= KEY_CONCEPTS_MAX:
                        return concepts
        return concepts

    # ------------------------------------------------------------ current task

    def fetch_current_task(self, session_id: Optional[str]) -> Optional[dict]:
        """The session's current task mapped to its course subject.

        Returns `{task_id, title, course_id, subject}` or None when the
        session is unreachable/unparseable. `title` may be None when the
        session has no tasks yet — the orchestrator then logs task-title-only.
        """
        if not session_id:
            return None
        doc = self._find_session(session_id)
        if not doc:
            return None

        tp = doc.get("taskProgress") or {}
        tasks = tp.get("tasks") or []
        idx = tp.get("currentTaskIndex") or 0
        task = tasks[idx] if 0 <= idx < len(tasks) else (tasks[0] if tasks else None)

        course_id = doc.get("courseId")
        subject = "Unknown"
        if course_id:
            subject = self._subject_for_course(course_id)
        return {
            "task_id": (task or {}).get("taskId"),
            "title": (task or {}).get("title"),
            "course_id": course_id,
            "subject": subject,
        }

    def _find_session(self, session_id: str) -> Optional[dict]:
        try:
            return self._db[STUDY_SESSIONS_COLLECTION].find_one(
                {"_id": self._as_object_id(session_id)}, _SESSION_PROJECTION
            )
        except Exception:
            return None

    def _subject_for_course(self, course_id) -> str:
        try:
            course = self._db[COURSES_COLLECTION].find_one(
                {"_id": self._as_object_id(course_id)}, _COURSE_SUBJECT_PROJECTION
            )
            if not course or not course.get("subjectId"):
                return "Unknown"
            subj = self._db[SUBJECTS_COLLECTION].find_one(
                {"_id": self._as_object_id(course.get("subjectId"))},
                _SUBJECT_PROJECTION,
            )
            if not subj or not subj.get("name"):
                return "Unknown"
            return str(subj["name"])[:SUBJECT_MAX_CHARS]
        except Exception:
            return "Unknown"

    # ------------------------------------------------------------- id helpers

    @staticmethod
    def _as_object_id(value):
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str):
            s = value.strip()
            if (
                len(s) == 24
                and all(ch in "0123456789abcdefABCDEF" for ch in s)
            ):
                try:
                    return ObjectId(s)
                except Exception:
                    return s
            return s
        return value


COURSES_USER_ID_FIELD = "userId"