import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .calendar_service import CalendarService, CalendarEvent
from ..models.task import Task, TaskManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCalendarSyncService:
    """
    Service for synchronizing tasks with calendar events.
    Maps tasks to calendar events and keeps them in sync.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        calendar_service: CalendarService
    ):
        """
        Initialize the task calendar sync service.

        Args:
            task_manager: Task manager
            calendar_service: Calendar service
        """
        self.task_manager = task_manager
        self.calendar_service = calendar_service

    def sync_task_to_calendar(
        self,
        task: Task,
        provider: str,
        calendar_id: str,
        create_if_missing: bool = True
    ) -> Optional[Task]:
        """
        Sync a task to a calendar event.

        Args:
            task: Task to sync
            provider: Calendar provider (google, microsoft, apple)
            calendar_id: ID of the calendar
            create_if_missing: Whether to create a new event if not found

        Returns:
            Updated task or None if sync failed
        """
        logger.info(
            f"Syncing task {task.id} to calendar {provider}/{calendar_id}")

        # Check if task is already linked to a calendar event
        if task.calendar_id and task.calendar_provider == provider:
            # Update existing event
            try:
                # Convert task to calendar event
                event = self.task_to_event(task, calendar_id)

                # Update the event
                adapter = self.calendar_service.get_adapter(provider)
                updated_event = adapter.update_event(calendar_id, event)

                # Update task with event info
                task.calendar_id = updated_event.id
                task.calendar_provider = provider
                task.calendar_sync = True

                # Save task
                self.task_manager.update_task(task)

                logger.info(
                    f"Updated calendar event {updated_event.id} for task {task.id}")
                return task
            except Exception as e:
                logger.error(
                    f"Error updating calendar event for task {task.id}: {e}")

                # If event not found, clear the calendar link
                task.calendar_id = ""
                task.calendar_sync = False
                self.task_manager.update_task(task)

                # Try to create a new event if requested
                if create_if_missing:
                    return self.create_event_for_task(task, provider, calendar_id)

                return None
        elif create_if_missing:
            # Create new event
            return self.create_event_for_task(task, provider, calendar_id)

        return None

    def create_event_for_task(
        self,
        task: Task,
        provider: str,
        calendar_id: str
    ) -> Optional[Task]:
        """
        Create a calendar event for a task.

        Args:
            task: Task to create an event for
            provider: Calendar provider
            calendar_id: ID of the calendar

        Returns:
            Updated task or None if creation failed
        """
        try:
            # Convert task to calendar event
            event = self.task_to_event(task, calendar_id)

            # Create the event
            adapter = self.calendar_service.get_adapter(provider)
            created_event = adapter.create_event(calendar_id, event)

            # Update task with event info
            task.calendar_id = created_event.id
            task.calendar_provider = provider
            task.calendar_sync = True

            # Save task
            self.task_manager.update_task(task)

            logger.info(
                f"Created calendar event {created_event.id} for task {task.id}")
            return task
        except Exception as e:
            logger.error(
                f"Error creating calendar event for task {task.id}: {e}")
            return None

    def task_to_event(self, task: Task, calendar_id: str) -> CalendarEvent:
        """
        Convert a task to a calendar event.

        Args:
            task: Task to convert
            calendar_id: ID of the calendar

        Returns:
            Calendar event
        """
        # Set start and end times based on due date
        if task.due_date:
            try:
                # Parse due date
                due = datetime.fromisoformat(
                    task.due_date.replace("Z", "+00:00"))

                # Set end time to due date
                end_time = due

                # Set start time to 1 hour before due date
                start_time = due - timedelta(hours=1)
            except (ValueError, TypeError):
                # Default to today if due date is invalid
                start_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(hours=1)
        else:
            # Default to today if no due date
            start_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=1)

        # Create event
        event = CalendarEvent(
            id=task.calendar_id,  # Will be empty for new events
            title=f"{task.title} [{task.status.upper()}]",
            description=f"Task: {task.title}\n\nDetails: {task.description}\n\nStatus: {task.status}\nPriority: {task.priority}\nProject: {task.project}",
            start_time=start_time,
            end_time=end_time,
            location="",
            calendar_id=calendar_id
        )

        return event

    def sync_event_to_task(
        self,
        event: CalendarEvent,
        provider: str,
        create_if_missing: bool = True
    ) -> Optional[Task]:
        """
        Sync a calendar event to a task.

        Args:
            event: Calendar event to sync
            provider: Calendar provider
            create_if_missing: Whether to create a new task if not found

        Returns:
            Updated task or None if sync failed
        """
        logger.info(f"Syncing calendar event {event.id} to task")

        # Check if event is already linked to a task
        task = self.task_manager.get_task_by_calendar_id(event.id, provider)

        if task:
            # Update existing task
            try:
                # Update task fields
                self.update_task_from_event(task, event, provider)

                # Save task
                self.task_manager.update_task(task)

                logger.info(
                    f"Updated task {task.id} from calendar event {event.id}")
                return task
            except Exception as e:
                logger.error(
                    f"Error updating task from calendar event {event.id}: {e}")
                return None
        elif create_if_missing:
            # Create new task
            return self.create_task_from_event(event, provider)

        return None

    def create_task_from_event(
        self,
        event: CalendarEvent,
        provider: str
    ) -> Optional[Task]:
        """
        Create a task from a calendar event.

        Args:
            event: Calendar event to create a task from
            provider: Calendar provider

        Returns:
            Created task or None if creation failed
        """
        try:
            # Create task
            task = Task()

            # Update task fields
            self.update_task_from_event(task, event, provider)

            # Save task
            self.task_manager.add_task(task)

            logger.info(
                f"Created task {task.id} from calendar event {event.id}")
            return task
        except Exception as e:
            logger.error(
                f"Error creating task from calendar event {event.id}: {e}")
            return None

    def update_task_from_event(
        self,
        task: Task,
        event: CalendarEvent,
        provider: str
    ):
        """
        Update a task from a calendar event.

        Args:
            task: Task to update
            event: Calendar event to update from
            provider: Calendar provider
        """
        # Extract title and status from event title
        title = event.title
        status = "pending"

        # Check if the title contains a status tag [STATUS]
        if "[" in title and "]" in title:
            start = title.rfind("[")
            end = title.rfind("]")
            if start < end:
                status_tag = title[start+1:end].lower()
                if status_tag in ["pending", "in-progress", "completed"]:
                    status = status_tag
                    title = title[:start].strip() + title[end+1:].strip()

        # Update task fields
        task.title = title
        task.status = status

        # Extract description, project, and priority from event description
        if event.description:
            # Try to extract project and priority from description
            project = ""
            priority = "medium"

            lines = event.description.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("Project:"):
                    project = line[8:].strip()
                elif line.startswith("Priority:"):
                    priority_str = line[9:].strip().lower()
                    if priority_str in ["low", "medium", "high"]:
                        priority = priority_str

            # Set description (excluding metadata)
            description = event.description
            for prefix in ["Task:", "Details:", "Status:", "Priority:", "Project:"]:
                description = description.replace(f"{prefix} ", "")

            task.description = description.strip()
            task.project = project
            task.priority = priority

        # Set due date from event end time
        if event.end_time:
            task.due_date = event.end_time.isoformat()

        # Set calendar info
        task.calendar_id = event.id
        task.calendar_provider = provider
        task.calendar_sync = True

    def batch_sync_tasks_to_calendar(
        self,
        tasks: List[Task],
        provider: str,
        calendar_id: str
    ) -> Tuple[int, int, List[str]]:
        """
        Sync multiple tasks to calendar events.

        Args:
            tasks: Tasks to sync
            provider: Calendar provider
            calendar_id: ID of the calendar

        Returns:
            Tuple of (success_count, failure_count, error_messages)
        """
        success_count = 0
        failure_count = 0
        error_messages = []

        for task in tasks:
            try:
                if self.sync_task_to_calendar(task, provider, calendar_id):
                    success_count += 1
                else:
                    failure_count += 1
                    error_messages.append(
                        f"Failed to sync task {task.id} ({task.title})")
            except Exception as e:
                failure_count += 1
                error_messages.append(
                    f"Error syncing task {task.id} ({task.title}): {str(e)}")

        return success_count, failure_count, error_messages

    def batch_sync_events_to_tasks(
        self,
        events: List[CalendarEvent],
        provider: str
    ) -> Tuple[int, int, List[str]]:
        """
        Sync multiple calendar events to tasks.

        Args:
            events: Calendar events to sync
            provider: Calendar provider

        Returns:
            Tuple of (success_count, failure_count, error_messages)
        """
        success_count = 0
        failure_count = 0
        error_messages = []

        for event in events:
            try:
                if self.sync_event_to_task(event, provider):
                    success_count += 1
                else:
                    failure_count += 1
                    error_messages.append(
                        f"Failed to sync event {event.id} ({event.title})")
            except Exception as e:
                failure_count += 1
                error_messages.append(
                    f"Error syncing event {event.id} ({event.title}): {str(e)}")

        return success_count, failure_count, error_messages

    def get_syncable_tasks(self) -> List[Task]:
        """
        Get tasks that can be synced to calendar.
        This includes tasks with due dates that aren't completed.

        Returns:
            List of syncable tasks
        """
        all_tasks = self.task_manager.get_all_tasks()
        return [
            task for task in all_tasks
            if task.due_date and task.status != "completed"
        ]

    def get_already_synced_tasks(self, provider: str) -> List[Task]:
        """
        Get tasks that are already synced to a calendar provider.

        Args:
            provider: Calendar provider

        Returns:
            List of synced tasks
        """
        all_tasks = self.task_manager.get_all_tasks()
        return [
            task for task in all_tasks
            if task.calendar_sync and task.calendar_provider == provider
        ]
