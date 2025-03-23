from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import uuid
import json
import os
from datetime import datetime, timedelta


@dataclass
class Task:
    """Task model for both one-time and recurring tasks."""

    id: str = ""
    title: str = ""
    description: str = ""
    status: str = "pending"  # pending, in-progress, completed
    due_date: str = ""  # ISO date or datetime
    priority: str = "medium"  # low, medium, high
    tags: List[str] = field(default_factory=list)
    project: str = ""
    assigned_to: str = ""
    created_at: str = ""  # ISO date or datetime
    updated_at: str = ""  # ISO date or datetime
    completed_at: str = ""  # ISO date or datetime

    # Recurrence fields
    is_recurring: bool = False
    recurrence_frequency: str = ""  # daily, weekly, monthly
    recurrence_start_date: str = ""  # When recurring task starts
    recurrence_last_run: str = ""  # Last time recurring task was executed
    recurrence_next_run: str = ""  # Next scheduled run time
    recurrence_enabled: bool = True  # Whether recurring task is enabled

    # Calendar integration
    calendar_id: str = ""  # ID of the linked calendar event
    calendar_provider: str = ""  # google, microsoft, apple
    calendar_sync: bool = False  # Whether to sync with calendar

    def __post_init__(self):
        """Set default ID and timestamps if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())

        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

        # Set initial recurrence next run date if needed
        if self.is_recurring and not self.recurrence_next_run and self.recurrence_start_date:
            self.recurrence_next_run = self.recurrence_start_date

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary representation."""
        return cls(**data)

    def is_overdue(self) -> bool:
        """Check if the task is overdue."""
        if not self.due_date or self.status == "completed":
            return False

        try:
            due = datetime.fromisoformat(self.due_date.replace("Z", "+00:00"))
            return due < datetime.now()
        except (ValueError, TypeError):
            return False

    def is_due_soon(self, days: int = 3) -> bool:
        """Check if the task is due soon."""
        if not self.due_date or self.status == "completed":
            return False

        try:
            due = datetime.fromisoformat(self.due_date.replace("Z", "+00:00"))
            soon = datetime.now() + timedelta(days=days)
            return datetime.now() <= due <= soon
        except (ValueError, TypeError):
            return False

    def mark_completed(self):
        """Mark the task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def mark_in_progress(self):
        """Mark the task as in progress."""
        self.status = "in-progress"
        self.completed_at = ""
        self.updated_at = datetime.now().isoformat()

    def mark_pending(self):
        """Mark the task as pending."""
        self.status = "pending"
        self.completed_at = ""
        self.updated_at = datetime.now().isoformat()

    def update_due_date(self, due_date: str):
        """Update the due date."""
        self.due_date = due_date
        self.updated_at = datetime.now().isoformat()

    def calculate_next_recurrence(self) -> str:
        """Calculate the next run time for a recurring task."""
        if not self.is_recurring:
            return ""

        # Use last run date or start date
        base_date_str = self.recurrence_last_run or self.recurrence_start_date or datetime.now().isoformat()

        try:
            base_date = datetime.fromisoformat(
                base_date_str.replace("Z", "+00:00"))

            if self.recurrence_frequency == 'daily':
                next_run = base_date + timedelta(days=1)
            elif self.recurrence_frequency == 'weekly':
                next_run = base_date + timedelta(days=7)
            elif self.recurrence_frequency == 'monthly':
                # Simple approximation for a month
                next_run = base_date + timedelta(days=30)
            else:
                next_run = base_date + timedelta(days=1)  # Default to daily

            return next_run.isoformat()
        except (ValueError, TypeError):
            # Default to tomorrow if date parsing fails
            return (datetime.now() + timedelta(days=1)).isoformat()

    def is_recurrence_due(self) -> bool:
        """Check if a recurring task is due for execution."""
        if not self.is_recurring or not self.recurrence_enabled:
            return False

        if not self.recurrence_next_run:
            return False

        try:
            next_run = datetime.fromisoformat(
                self.recurrence_next_run.replace("Z", "+00:00"))
            return next_run <= datetime.now()
        except (ValueError, TypeError):
            return False

    def execute_recurrence(self) -> Dict[str, Any]:
        """
        Execute a recurring task by creating a new regular task instance.

        Returns:
            Dict containing data for the new task instance
        """
        if not self.is_recurring:
            return {}

        # Current date for naming
        today = datetime.now().strftime("%Y-%m-%d")

        # Create new task data based on recurring template
        new_task_data = {
            'title': f"{self.title} ({today})",
            'description': self.description,
            'status': 'pending',
            'due_date': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            'priority': self.priority,
            'project': self.project,
            'tags': self.tags + ['recurring'],
            'source': {
                'type': 'scheduled',
                'recurring_task_id': self.id
            }
        }

        # Update recurrence tracking
        now = datetime.now().isoformat()
        self.recurrence_last_run = now
        self.recurrence_next_run = self.calculate_next_recurrence()
        self.updated_at = now

        return new_task_data


class TaskManager:
    """Manages tasks and their persistence."""

    def __init__(self, storage_path: str = None):
        """
        Initialize task manager.

        Args:
            storage_path: Path to the tasks file
        """
        if storage_path is None:
            # Use default path
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            storage_path = os.path.join(base_dir, "data", "tasks.json")

        self.storage_path = storage_path
        self.tasks: Dict[str, Task] = {}
        self.load_tasks()

    def load_tasks(self):
        """Load tasks from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)

                    # Convert dictionaries to Task objects
                    self.tasks = {
                        task_id: Task.from_dict(task_data)
                        for task_id, task_data in tasks_data.items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading tasks: {e}")
                self.tasks = {}
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self.tasks = {}

    def save_tasks(self):
        """Save tasks to storage."""
        # Convert Task objects to dictionaries
        tasks_data = {
            task_id: task.to_dict()
            for task_id, task in self.tasks.items()
        }

        # Save to file
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2)

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def add_task(self, task: Task) -> Task:
        """Add a new task."""
        if not task.id:
            task.id = str(uuid.uuid4())

        self.tasks[task.id] = task
        self.save_tasks()
        return task

    def update_task(self, task: Task) -> Task:
        """Update an existing task."""
        if task.id not in self.tasks:
            raise ValueError(f"Task with ID {task.id} not found")

        task.updated_at = datetime.now().isoformat()
        self.tasks[task.id] = task
        self.save_tasks()
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id not in self.tasks:
            return False

        del self.tasks[task_id]
        self.save_tasks()
        return True

    def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get tasks by status."""
        return [task for task in self.tasks.values() if task.status == status]

    def get_tasks_by_project(self, project: str) -> List[Task]:
        """Get tasks by project."""
        return [task for task in self.tasks.values() if task.project == project]

    def get_tasks_by_tag(self, tag: str) -> List[Task]:
        """Get tasks by tag."""
        return [task for task in self.tasks.values() if tag in task.tags]

    def get_overdue_tasks(self) -> List[Task]:
        """Get overdue tasks."""
        return [task for task in self.tasks.values() if task.is_overdue()]

    def get_tasks_due_soon(self, days: int = 3) -> List[Task]:
        """Get tasks due soon."""
        return [task for task in self.tasks.values() if task.is_due_soon(days)]

    def get_recurring_tasks(self) -> List[Task]:
        """Get all recurring tasks."""
        return [task for task in self.tasks.values() if task.is_recurring]

    def get_one_time_tasks(self) -> List[Task]:
        """Get all one-time tasks."""
        return [task for task in self.tasks.values() if not task.is_recurring]

    def get_task_by_calendar_id(self, calendar_id: str, provider: str) -> Optional[Task]:
        """Get a task by calendar event ID."""
        for task in self.tasks.values():
            if task.calendar_id == calendar_id and task.calendar_provider == provider:
                return task
        return None

    def process_recurring_tasks(self) -> List[Task]:
        """
        Process all due recurring tasks and create their instances.

        Returns:
            List of created task instances
        """
        created_tasks = []

        for task in self.get_recurring_tasks():
            if task.is_recurrence_due():
                # Create new task instance from the recurring task
                new_task_data = task.execute_recurrence()
                new_task = Task.from_dict(new_task_data)

                # Add the new task
                self.add_task(new_task)
                created_tasks.append(new_task)

                # Update the recurring task
                self.update_task(task)

        return created_tasks
