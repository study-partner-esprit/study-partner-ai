from agents.planner.planner import PlannerAgent
from agents.scheduler.agent import SchedulerAgent, SchedulingContext


def test_planner_to_scheduler_pipeline():

    planner = PlannerAgent()
    scheduler = SchedulerAgent()

    # minimal fake course context
    course_context = {
        "course_title": "Mini course",
        "topics": [
            {
                "title": "Intro",
                "summary": "Python basics",
                "subtopics": [
                    {"title": "Variables", "summary": "Variables and types"},
                    {"title": "Loops", "summary": "For and while loops"},
                ],
            }
        ],
    }

    tasks = planner.generate_tasks(
        user_goal="Learn Python basics",
        available_time_minutes=120,
        course_context=course_context,
    ) 

    assert len(tasks) > 0

    context = SchedulingContext(
        calendar_events=[]
    )

    plan = scheduler.build_schedule(tasks, context)

    assert plan.sessions
    assert plan.total_minutes > 0
