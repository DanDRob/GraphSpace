import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_scheduled_tasks_to_unified_model(knowledge_graph, task_scheduler=None):
    """
    Migrate existing scheduled tasks to the new unified task model.

    Args:
        knowledge_graph: The knowledge graph instance
        task_scheduler: Optional task scheduler instance

    Returns:
        Dict containing migration statistics
    """
    if task_scheduler is None:
        # Load tasks from the default scheduled tasks file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scheduled_tasks_path = os.path.join(
            base_dir, "data", "scheduled_tasks.json")

        if not os.path.exists(scheduled_tasks_path):
            logger.info("No scheduled tasks file found, nothing to migrate")
            return {"success": True, "migrated": 0}

        try:
            with open(scheduled_tasks_path, 'r') as f:
                scheduled_tasks = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scheduled tasks: {e}")
            return {"success": False, "error": str(e)}
    else:
        # Use the task scheduler's loaded tasks
        scheduled_tasks = task_scheduler.get_tasks()

    # Statistics
    stats = {
        "success": True,
        "total": len(scheduled_tasks),
        "migrated": 0,
        "errors": 0
    }

    # Process each scheduled task
    for scheduled_task in scheduled_tasks:
        try:
            # Create a unified task model entry
            task_data = {
                'title': scheduled_task.get('title', 'Unnamed Task'),
                'description': scheduled_task.get('description', ''),
                'status': 'pending',
                'project': scheduled_task.get('project', ''),
                'tags': scheduled_task.get('tags', []) + ['recurring', 'migrated'],
                'created_at': scheduled_task.get('start_date', datetime.now().isoformat()),
                'updated_at': datetime.now().isoformat(),

                # Add recurrence fields
                'is_recurring': True,
                'recurrence_frequency': scheduled_task.get('frequency', 'daily'),
                'recurrence_start_date': scheduled_task.get('start_date', datetime.now().isoformat()),
                'recurrence_last_run': scheduled_task.get('last_run'),
                'recurrence_next_run': scheduled_task.get('next_run', datetime.now().isoformat()),
                'recurrence_enabled': scheduled_task.get('enabled', True)
            }

            # Add to knowledge graph
            knowledge_graph.data.setdefault("tasks", []).append(task_data)
            stats["migrated"] += 1

        except Exception as e:
            logger.error(f"Error migrating scheduled task: {e}")
            stats["errors"] += 1

    # Save the updated knowledge graph
    if stats["migrated"] > 0:
        try:
            # Rebuild the graph
            knowledge_graph.build_graph()

            # Save the data
            knowledge_graph.save_data()

            # If migration was successful, rename the old scheduled tasks file
            if task_scheduler is None and stats["errors"] == 0:
                backup_path = scheduled_tasks_path + ".bak"
                try:
                    Path(scheduled_tasks_path).rename(backup_path)
                    logger.info(
                        f"Renamed old scheduled tasks file to {backup_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not rename old scheduled tasks file: {e}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph after migration: {e}")
            stats["success"] = False
            stats["error"] = str(e)

    logger.info(
        f"Migration complete: {stats['migrated']} tasks migrated, {stats['errors']} errors")
    return stats


def is_migration_needed():
    """
    Check if migration is needed by looking for the scheduled tasks file.

    Returns:
        bool: True if migration is needed, False otherwise
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scheduled_tasks_path = os.path.join(
        base_dir, "data", "scheduled_tasks.json")
    backup_path = scheduled_tasks_path + ".bak"

    # If original file exists and .bak doesn't, we need migration
    return os.path.exists(scheduled_tasks_path) and not os.path.exists(backup_path)
