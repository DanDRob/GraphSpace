import os
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Service for scheduling and managing recurring tasks.

    This service allows:
    - Creating recurring tasks on different schedules (daily, weekly, monthly)
    - Running tasks on schedule
    - Persisting task schedules to disk
    """

    def __init__(self, data_path: str = None, knowledge_graph=None):
        """
        Initialize the task scheduler service.

        Args:
            data_path: Path to store scheduled tasks data
            knowledge_graph: Reference to the knowledge graph instance
        """
        self.knowledge_graph = knowledge_graph

        # Set default data path if not provided
        if data_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "scheduled_tasks.json")

        self.data_path = data_path
        self.scheduled_tasks = self._load_tasks()
        self.running = False
        self.scheduler_thread = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load scheduled tasks from disk."""
        if not os.path.exists(self.data_path):
            return []

        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading scheduled tasks: {e}")
            return []

    def _save_tasks(self) -> None:
        """Save scheduled tasks to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

            with open(self.data_path, 'w') as f:
                json.dump(self.scheduled_tasks, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduled tasks: {e}")

    def add_task(self, task_data: Dict[str, Any]) -> str:
        """
        Schedule a new recurring task.

        Args:
            task_data: Task data including:
                - title: Task title
                - description: Task description
                - frequency: 'daily', 'weekly', or 'monthly'
                - start_date: When to start the recurring task
                - tags: List of tags
                - project: Project name

        Returns:
            Scheduled task ID
        """
        # Validate required fields
        if not task_data.get('title'):
            raise ValueError("Task title is required")

        if task_data.get('frequency') not in ['daily', 'weekly', 'monthly']:
            raise ValueError(
                "Frequency must be 'daily', 'weekly', or 'monthly'")

        # Create scheduled task entry
        task_id = str(len(self.scheduled_tasks) + 1)
        scheduled_task = {
            'id': task_id,
            'title': task_data.get('title'),
            'description': task_data.get('description', ''),
            'frequency': task_data.get('frequency'),
            'start_date': task_data.get('start_date', datetime.now().isoformat()),
            'last_run': None,
            'next_run': task_data.get('start_date', datetime.now().isoformat()),
            'tags': task_data.get('tags', []),
            'project': task_data.get('project', ''),
            'enabled': True
        }

        self.scheduled_tasks.append(scheduled_task)
        self._save_tasks()

        return task_id

    def update_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """
        Update an existing scheduled task.

        Args:
            task_id: ID of the task to update
            task_data: Updated task data

        Returns:
            True if updated successfully, False otherwise
        """
        for i, task in enumerate(self.scheduled_tasks):
            if task['id'] == task_id:
                # Update task fields
                for key, value in task_data.items():
                    if key not in ['id', 'last_run', 'next_run']:
                        task[key] = value

                # Recalculate next run time if frequency changed
                if 'frequency' in task_data:
                    task['next_run'] = self._calculate_next_run(
                        task['last_run'] or datetime.now().isoformat(),
                        task['frequency']
                    )

                self.scheduled_tasks[i] = task
                self._save_tasks()
                return True

        return False

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a scheduled task.

        Args:
            task_id: ID of the task to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        for i, task in enumerate(self.scheduled_tasks):
            if task['id'] == task_id:
                del self.scheduled_tasks[i]
                self._save_tasks()
                return True

        return False

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get all scheduled tasks."""
        return self.scheduled_tasks

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scheduled task by ID."""
        for task in self.scheduled_tasks:
            if task['id'] == task_id:
                return task

        return None

    def _calculate_next_run(self, last_run_str: str, frequency: str) -> str:
        """
        Calculate the next run time for a task.

        Args:
            last_run_str: ISO timestamp of the last run
            frequency: 'daily', 'weekly', or 'monthly'

        Returns:
            ISO timestamp of the next run
        """
        last_run = datetime.fromisoformat(last_run_str)

        if frequency == 'daily':
            next_run = last_run + timedelta(days=1)
        elif frequency == 'weekly':
            next_run = last_run + timedelta(days=7)
        elif frequency == 'monthly':
            # Simple approximation for a month
            next_run = last_run + timedelta(days=30)
        else:
            next_run = last_run + timedelta(days=1)  # Default to daily

        return next_run.isoformat()

    def _create_actual_task(self, scheduled_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an actual task from a scheduled task.

        Args:
            scheduled_task: The scheduled task data

        Returns:
            The task data to be added to the knowledge graph
        """
        # Generate a title with date
        today = datetime.now().strftime("%Y-%m-%d")
        title = f"{scheduled_task['title']} ({today})"

        task_data = {
            'title': title,
            'description': scheduled_task['description'],
            'status': 'pending',
            'due_date': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            'project': scheduled_task['project'],
            'tags': scheduled_task['tags'] + ['recurring'],
            'source': {
                'type': 'scheduled',
                'scheduled_task_id': scheduled_task['id']
            }
        }

        return task_data

    def _process_due_tasks(self) -> None:
        """Process all due scheduled tasks."""
        now = datetime.now()

        for i, task in enumerate(self.scheduled_tasks):
            if not task['enabled']:
                continue

            next_run = datetime.fromisoformat(task['next_run'])

            if next_run <= now:
                # This task is due for execution
                logger.info(f"Running scheduled task: {task['title']}")

                try:
                    # Create an actual task in the knowledge graph
                    if self.knowledge_graph:
                        task_data = self._create_actual_task(task)

                        # Add task to knowledge graph
                        self.knowledge_graph.data["tasks"].append(task_data)

                        # Rebuild graph
                        self.knowledge_graph.build_graph()

                        # Save data
                        self.knowledge_graph.save_data()

                        logger.info(
                            f"Created task from scheduled task: {task['title']}")

                    # Update last_run and next_run
                    now_str = now.isoformat()
                    task['last_run'] = now_str
                    task['next_run'] = self._calculate_next_run(
                        now_str, task['frequency'])

                    # Update in list
                    self.scheduled_tasks[i] = task

                    # Save changes
                    self._save_tasks()
                except Exception as e:
                    logger.error(
                        f"Error processing scheduled task {task['id']}: {e}")

    def start(self) -> None:
        """Start the task scheduler."""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        logger.info("Task scheduler started")

    def stop(self) -> None:
        """Stop the task scheduler."""
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Task scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs in a separate thread."""
        while self.running:
            try:
                # Process due tasks
                self._process_due_tasks()

                # Sleep for a while (check every 15 minutes)
                # In a production environment, this could be more sophisticated
                time.sleep(15 * 60)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Sleep and retry on error
