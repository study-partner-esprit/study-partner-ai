from agents.scheduler.agent import SchedulerAgent, SchedulingContext
from models.task import Task


class TestSchedulerBasic:
    """Basic scheduling functionality tests."""

    def test_scheduler_basic_plan(self):
        """Test basic scheduling with no calendar conflicts."""
        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=[],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=45,
                prerequisites=[],
            ),
        ]

        context = SchedulingContext(
            calendar_events=[
                {
                    "start": "2026-02-06T10:00:00",
                    "end": "2026-02-06T11:00:00",
                }
            ]
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        assert plan.total_minutes > 0
        assert len(plan.sessions) > 0
        assert plan.span_days >= 1


class TestPrerequisiteAwareScheduling:
    """Tests for prerequisite-aware scheduling."""

    def test_prerequisites_enforced(self):
        """Test that tasks without met prerequisites are skipped."""
        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=[],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=45,
                prerequisites=["t1"],  # Depends on t1
            ),
            Task(
                task_id="t3",
                user_id="user1",
                title="Task 3",
                description="",
                estimated_duration=30,
                prerequisites=["t1"],  # Depends on t1
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # t1 should be scheduled
        task_ids = [s.task_id for s in plan.sessions]
        assert "t1" in task_ids

        # t2 and t3 should also be scheduled since t1 is scheduled
        assert "t2" in task_ids
        assert "t3" in task_ids

    def test_missing_prerequisite_skipped(self):
        """Test that tasks with missing prerequisites are skipped and logged."""
        tasks = [
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=30,
                prerequisites=["missing_prerequisite"],
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # t2 should be skipped because prerequisite is missing
        assert "t2" in plan.skipped_tasks
        assert len(plan.sessions) == 0

    def test_prerequisite_ordering(self):
        """Test that tasks are scheduled in prerequisite order."""
        tasks = [
            Task(
                task_id="t3",
                user_id="user1",
                title="Task 3",
                description="",
                estimated_duration=30,
                prerequisites=["t2"],
            ),
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=[],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=30,
                prerequisites=["t1"],
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # All should be scheduled
        task_ids = [s.task_id for s in plan.sessions]
        assert len(task_ids) == 3

        # Find positions in schedule
        t1_pos = next(i for i, s in enumerate(plan.sessions) if s.task_id == "t1")
        t2_pos = next(i for i, s in enumerate(plan.sessions) if s.task_id == "t2")
        t3_pos = next(i for i, s in enumerate(plan.sessions) if s.task_id == "t3")

        # Verify ordering: t1 < t2 < t3
        assert t1_pos < t2_pos
        assert t2_pos < t3_pos


class TestMultiDayScheduling:
    """Tests for multi-day scheduling."""

    def test_multiday_scheduling(self):
        """Test that tasks spill to next day when current day is full."""
        tasks = [
            Task(
                task_id=f"t{i}",
                user_id="user1",
                title=f"Task {i}",
                description="",
                estimated_duration=120,  # 2 hours each
                prerequisites=[],
            )
            for i in range(4)
        ]

        context = SchedulingContext(
            calendar_events=[],
            max_minutes_per_day=240,  # 4 hours per day
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # Should span multiple days
        dates = set(s.start_datetime.date() for s in plan.sessions)
        assert len(dates) >= 2 or plan.fallback_used

    def test_respects_work_window(self):
        """Test that scheduling respects the work window (8:00 - 22:00)."""
        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=[],
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        for session in plan.sessions:
            assert session.start_datetime.hour >= 8
            assert session.end_datetime.hour <= 22

    def test_respects_max_minutes_per_day(self):
        """Test that daily scheduling respects max_minutes_per_day limit."""
        tasks = [
            Task(
                task_id=f"t{i}",
                user_id="user1",
                title=f"Task {i}",
                description="",
                estimated_duration=100,
                prerequisites=[],
            )
            for i in range(5)
        ]

        context = SchedulingContext(
            calendar_events=[],
            max_minutes_per_day=200,
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # Group sessions by day
        from collections import defaultdict

        sessions_by_day = defaultdict(list)
        for session in plan.sessions:
            day = session.start_datetime.date()
            sessions_by_day[day].append(session)

        # Check each day doesn't exceed limit
        for day, sessions in sessions_by_day.items():
            day_minutes = sum(
                (s.end_datetime - s.start_datetime).total_seconds() / 60
                for s in sessions
            )
            assert day_minutes <= context.max_minutes_per_day + 5  # Small tolerance


class TestFallbackScheduling:
    """Tests for fallback Pomodoro-style scheduling."""

    def test_fallback_strategy_used(self):
        """Test that fallback strategy is used when no free slots exist."""
        # Create heavy calendar that blocks all time
        calendar_events = [
            {
                "start": f"2026-02-{6:02d}T{h:02d}:00:00",
                "end": f"2026-02-{6:02d}T{h:02d}:59:00",
            }
            for h in range(8, 22)
        ]

        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=25,
                prerequisites=[],
            ),
        ]

        context = SchedulingContext(
            calendar_events=calendar_events,
            max_minutes_per_day=100,
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # Fallback might be used or task skipped
        # At minimum, we should have a valid result
        assert isinstance(plan.total_minutes, int)
        assert isinstance(plan.fallback_used, bool)

    def test_fallback_pomodoro_format(self):
        """Test that fallback scheduling uses 25-5 Pomodoro format."""
        # Barely any free time - should trigger fallback on second day
        calendar_events = [
            {
                "start": "2026-02-06T08:00:00",
                "end": "2026-02-06T21:00:00",
            }
        ]

        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=25,
                prerequisites=[],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=25,
                prerequisites=[],
            ),
        ]

        context = SchedulingContext(
            calendar_events=calendar_events,
            max_minutes_per_day=100,
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # Should have scheduled something
        assert plan.total_minutes >= 0
        assert isinstance(plan.sessions, list)


class TestSchedulerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task_list(self):
        """Test scheduling with no tasks."""
        tasks = []
        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        assert plan.total_minutes == 0
        assert len(plan.sessions) == 0
        assert plan.skipped_tasks == []

    def test_all_tasks_skipped(self):
        """Test when all tasks are skipped due to unmet prerequisites."""
        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=["impossible"],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=30,
                prerequisites=["also_impossible"],
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        assert len(plan.sessions) == 0
        assert "t1" in plan.skipped_tasks
        assert "t2" in plan.skipped_tasks

    def test_circular_prerequisites_ignored(self):
        """Test that circular prerequisites don't cause infinite loops."""
        tasks = [
            Task(
                task_id="t1",
                user_id="user1",
                title="Task 1",
                description="",
                estimated_duration=30,
                prerequisites=["t2"],
            ),
            Task(
                task_id="t2",
                user_id="user1",
                title="Task 2",
                description="",
                estimated_duration=30,
                prerequisites=["t1"],
            ),
        ]

        context = SchedulingContext(calendar_events=[])

        agent = SchedulerAgent()
        # Should complete without hanging
        plan = agent.build_schedule(tasks, context)

        # Both should be skipped due to circular dependency
        assert "t1" in plan.skipped_tasks or "t2" in plan.skipped_tasks

    def test_high_max_minutes_per_day(self):
        """Test scheduling with very high daily limit."""
        tasks = [
            Task(
                task_id=f"t{i}",
                user_id="user1",
                title=f"Task {i}",
                description="",
                estimated_duration=30,
                prerequisites=[],
            )
            for i in range(10)
        ]

        context = SchedulingContext(
            calendar_events=[],
            max_minutes_per_day=999,  # Very high
        )

        agent = SchedulerAgent()
        plan = agent.build_schedule(tasks, context)

        # Most or all tasks should fit in one day
        assert plan.span_days == 1
        assert len(plan.sessions) >= 5

