from models.llm_module import LLMModule
from models.gnn_module import GNNModule
from models.knowledge_graph import KnowledgeGraph
from models.task import Task, TaskManager
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import json
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
from services.document_processing.pipeline import DocumentProcessor
from utils.migration_utils import migrate_scheduled_tasks_to_unified_model, is_migration_needed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "uploads")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the GraphSpace instance
graphspace = None

# Initialize components
data_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "user_data.json")
knowledge_graph = None
gnn_module = None
llm_module = None
document_processor = None
task_manager = None


def initialize_components():
    """Initialize all components of the GraphSpace system."""
    global knowledge_graph, gnn_module, llm_module, document_processor, task_manager

    try:
        # Initialize Knowledge Graph
        knowledge_graph = KnowledgeGraph(data_path=data_path)

        # Initialize GNN Module
        gnn_module = GNNModule(input_dim=64, hidden_dim=128, output_dim=64)

        # Initialize LLM Module
        llm_module = LLMModule(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            language_model_name="distilgpt2"
        )

        # Initialize Document Processor
        document_processor = DocumentProcessor(llm_module=llm_module)

        # Initialize Task Manager with knowledge graph directory
        tasks_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "tasks.json")
        task_manager = TaskManager(storage_path=tasks_path)

        # Check if migration of legacy scheduled tasks is needed
        if is_migration_needed():
            print("Migrating legacy scheduled tasks to unified model...")
            stats = migrate_scheduled_tasks_to_unified_model(knowledge_graph)
            print(f"Migration complete: {stats['migrated']} tasks migrated")

        # Train GNN if there are nodes in the graph
        if len(knowledge_graph.graph.nodes()) > 0:
            stats = gnn_module.train(knowledge_graph.graph, epochs=50)

            # Update node embeddings in knowledge graph
            knowledge_graph.update_embeddings(gnn_module.get_node_embeddings())

        return True
    except Exception as e:
        print(f"Error initializing components: {e}")
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle user queries using RAG."""
    try:
        data = request.json
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        # Use RAG to answer the query
        result = graphspace.query(user_query)

        return jsonify(result)
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/add_note', methods=['POST'])
def add_note():
    """Add a new note to the knowledge graph."""
    try:
        data = request.json

        # Validate required fields
        if not data.get('content', '').strip():
            return jsonify({'error': 'Note content is required'}), 400

        # Prepare note data
        note_data = {
            'title': data.get('title', 'Untitled Note'),
            'content': data.get('content', ''),
            'tags': data.get('tags', []),
            'created': data.get('created', datetime.now().isoformat()),
            'updated': data.get('updated', datetime.now().isoformat())
        }

        # Add to knowledge graph
        node_id = knowledge_graph.add_note(note_data)

        # Re-train GNN if we have enough nodes
        if len(knowledge_graph.graph.nodes()) > 1:
            stats = gnn_module.train(knowledge_graph.graph, epochs=50)
            knowledge_graph.update_embeddings(gnn_module.get_node_embeddings())

        return jsonify({'success': True, 'node_id': node_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks (both one-time and recurring)."""
    try:
        # Get from knowledge graph for backward compatibility
        kg_tasks = knowledge_graph.data.get("tasks", [])

        # Get from task manager for new tasks
        tm_tasks = [task.to_dict() for task in task_manager.get_all_tasks()]

        # Combine both sources
        all_tasks = kg_tasks + tm_tasks

        # Add type flag for UI differentiation
        for task in all_tasks:
            task['is_recurring'] = task.get('is_recurring', False)

        return jsonify({'tasks': all_tasks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task by ID."""
    try:
        # Try to get from task manager first
        task = task_manager.get_task(task_id)
        if task:
            return jsonify(task.to_dict())

        # Fall back to knowledge graph for backward compatibility
        for task in knowledge_graph.data.get("tasks", []):
            if task.get("id") == task_id:
                return jsonify(task)

        return jsonify({'error': 'Task not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks', methods=['POST'])
def add_task():
    """Add a new task (both one-time and recurring)."""
    try:
        data = request.json

        # Validate required fields
        if not data.get('title', '').strip():
            return jsonify({'error': 'Task title is required'}), 400

        # Check if this is a recurring task
        is_recurring = data.get('is_recurring', False)

        if is_recurring and data.get('recurrence_frequency') not in ['daily', 'weekly', 'monthly']:
            return jsonify({'error': 'For recurring tasks, frequency must be daily, weekly, or monthly'}), 400

        # Create a Task object
        task_dict = {
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'status': data.get('status', 'pending'),
            'due_date': data.get('due_date', ''),
            'priority': data.get('priority', 'medium'),
            'tags': data.get('tags', []),
            'project': data.get('project', ''),

            # Recurrence fields
            'is_recurring': is_recurring,
            'recurrence_frequency': data.get('recurrence_frequency', ''),
            'recurrence_start_date': data.get('recurrence_start_date', datetime.now().isoformat()),
            'recurrence_enabled': True,

            # Calendar integration
            'calendar_sync': data.get('calendar_sync', False),
            'calendar_id': data.get('calendar_id', ''),
            'calendar_provider': data.get('calendar_provider', '')
        }

        # Create the Task object
        task = Task.from_dict(task_dict)

        # For recurring tasks, calculate next run date
        if is_recurring:
            task.recurrence_next_run = task.calculate_next_recurrence()

        # Add to task manager
        task = task_manager.add_task(task)

        # If task is linked to a calendar event and calendar sync is enabled,
        # add it to the calendar (to be implemented in calendar integration)

        # Also add to knowledge graph for backward compatibility
        knowledge_graph.data.setdefault("tasks", []).append(task.to_dict())
        knowledge_graph.build_graph()
        knowledge_graph.save_data()

        # Re-train GNN if we have enough nodes
        if len(knowledge_graph.graph.nodes()) > 1:
            stats = gnn_module.train(knowledge_graph.graph, epochs=50)
            knowledge_graph.update_embeddings(gnn_module.get_node_embeddings())

        return jsonify({'success': True, 'task_id': task.id, 'task': task.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """Update an existing task."""
    try:
        data = request.json

        # Try to get from task manager first
        task = task_manager.get_task(task_id)
        if task:
            # Update task fields
            if 'title' in data:
                task.title = data['title']
            if 'description' in data:
                task.description = data['description']
            if 'status' in data:
                task.status = data['status']
            if 'due_date' in data:
                task.due_date = data['due_date']
            if 'priority' in data:
                task.priority = data['priority']
            if 'tags' in data:
                task.tags = data['tags']
            if 'project' in data:
                task.project = data['project']

            # Recurrence fields
            if 'is_recurring' in data:
                task.is_recurring = data['is_recurring']
            if 'recurrence_frequency' in data:
                task.recurrence_frequency = data['recurrence_frequency']
            if 'recurrence_start_date' in data:
                task.recurrence_start_date = data['recurrence_start_date']
            if 'recurrence_enabled' in data:
                task.recurrence_enabled = data['recurrence_enabled']

            # Calendar integration
            if 'calendar_sync' in data:
                task.calendar_sync = data['calendar_sync']
            if 'calendar_id' in data:
                task.calendar_id = data['calendar_id']
            if 'calendar_provider' in data:
                task.calendar_provider = data['calendar_provider']

            # Recalculate next run time if it's a recurring task and relevant fields changed
            if task.is_recurring and any(field in data for field in [
                'recurrence_frequency', 'recurrence_start_date'
            ]):
                task.recurrence_next_run = task.calculate_next_recurrence()

            # Update task in the task manager
            task = task_manager.update_task(task)

            # Also update in knowledge graph for backward compatibility
            for i, kg_task in enumerate(knowledge_graph.data.get("tasks", [])):
                if kg_task.get("id") == task_id:
                    knowledge_graph.data["tasks"][i] = task.to_dict()
                    knowledge_graph.build_graph()
                    knowledge_graph.save_data()
                    break

            return jsonify({'success': True, 'task': task.to_dict()})

        # Fall back to knowledge graph for backward compatibility
        for i, kg_task in enumerate(knowledge_graph.data.get("tasks", [])):
            if kg_task.get("id") == task_id:
                # Update task fields
                for key, value in data.items():
                    kg_task[key] = value

                # Always update the updated timestamp
                kg_task['updated_at'] = datetime.now().isoformat()

                # Save changes
                knowledge_graph.data["tasks"][i] = kg_task
                knowledge_graph.build_graph()
                knowledge_graph.save_data()

                return jsonify({'success': True, 'task': kg_task})

        return jsonify({'error': 'Task not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task."""
    try:
        deleted = False

        # Try to delete from task manager first
        if task_manager.delete_task(task_id):
            deleted = True

        # Also try to delete from knowledge graph for backward compatibility
        for i, task in enumerate(knowledge_graph.data.get("tasks", [])):
            if task.get("id") == task_id:
                del knowledge_graph.data["tasks"][i]
                knowledge_graph.build_graph()
                knowledge_graph.save_data()
                deleted = True
                break

        if deleted:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Task not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/process_recurring', methods=['POST'])
def process_recurring_tasks():
    """Process all due recurring tasks manually."""
    try:
        created_tasks = task_manager.process_recurring_tasks()

        # Update knowledge graph with the new tasks for backward compatibility
        if created_tasks:
            for task in created_tasks:
                knowledge_graph.data.setdefault(
                    "tasks", []).append(task.to_dict())

            knowledge_graph.build_graph()
            knowledge_graph.save_data()

            # Re-train GNN if we have enough nodes
            if len(knowledge_graph.graph.nodes()) > 1:
                stats = gnn_module.train(knowledge_graph.graph, epochs=50)
                knowledge_graph.update_embeddings(
                    gnn_module.get_node_embeddings())

        return jsonify({
            'success': True,
            'created_count': len(created_tasks),
            'created_tasks': [task.to_dict() for task in created_tasks]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calendars', methods=['GET'])
def get_calendars():
    """Get all available calendars for the authenticated user."""
    try:
        # This would be implemented when calendar service is connected
        # For now, return a placeholder
        return jsonify({
            'success': True,
            'calendars': [
                {
                    'id': 'primary',
                    'name': 'Primary Calendar',
                    'provider': 'google',
                    'connected': False,
                    'description': 'Your main Google Calendar. Connect to sync tasks with it.'
                },
                {
                    'id': 'outlook_primary',
                    'name': 'Outlook Calendar',
                    'provider': 'microsoft',
                    'connected': False,
                    'description': 'Your Outlook/Microsoft Calendar. Connect to sync tasks with it.'
                }
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calendar/connect', methods=['POST'])
def connect_calendar():
    """Connect to a calendar service (OAuth flow)."""
    try:
        data = request.json
        provider = data.get('provider')

        if not provider:
            return jsonify({'error': 'Calendar provider is required'}), 400

        # This would begin the OAuth flow for the selected provider
        # For now, return a placeholder response with mock auth URL
        auth_url = f"/mock-auth/{provider}?redirect_uri=/calendar/callback"

        return jsonify({
            'success': True,
            'auth_url': auth_url,
            'message': f"Please authorize access to your {provider} calendar"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calendar/sync', methods=['POST'])
def sync_calendar():
    """Sync tasks with connected calendar."""
    try:
        data = request.json
        provider = data.get('provider')
        calendar_id = data.get('calendar_id')

        if not provider or not calendar_id:
            return jsonify({'error': 'Provider and calendar ID are required'}), 400

        # This would sync tasks with the calendar
        # For now, return a placeholder success message
        return jsonify({
            'success': True,
            'synced_tasks': 0,
            'message': f"Calendar sync with {provider} completed"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/add_contact', methods=['POST'])
def add_contact():
    """Add a new contact to the knowledge graph."""
    try:
        data = request.json

        # Validate required fields
        if not data.get('name', '').strip():
            return jsonify({'error': 'Contact name is required'}), 400

        # Prepare contact data
        contact_data = {
            'id': str(len(knowledge_graph.data.get("contacts", [])) + 1),
            'name': data.get('name', ''),
            'email': data.get('email', ''),
            'phone': data.get('phone', ''),
            'organization': data.get('organization', ''),
            'tags': data.get('tags', [])
        }

        # Add to data
        knowledge_graph.data.setdefault("contacts", []).append(contact_data)

        # Rebuild graph
        knowledge_graph.build_graph()

        # Save data
        knowledge_graph.save_data()

        # Re-train GNN if we have enough nodes
        if len(knowledge_graph.graph.nodes()) > 1:
            stats = gnn_module.train(knowledge_graph.graph, epochs=50)
            knowledge_graph.update_embeddings(gnn_module.get_node_embeddings())

        return jsonify({'success': True, 'contact_id': contact_data['id']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/graph_data', methods=['GET'])
def graph_data():
    """Get graph data for visualization."""
    try:
        # Convert NetworkX graph to a format suitable for visualization
        nodes = []
        edges = []

        # Add nodes
        for node_id, attrs in knowledge_graph.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            label = ''

            if node_type == 'note':
                label = attrs.get('title', 'Untitled Note')
            elif node_type == 'task':
                label = attrs.get('title', 'Untitled Task')
            elif node_type == 'contact':
                label = attrs.get('name', 'Unnamed Contact')

            nodes.append({
                'id': node_id,
                'label': label,
                'type': node_type,
                'tags': attrs.get('tags', [])
            })

        # Add edges
        for source, target, attrs in knowledge_graph.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'relationship': attrs.get('relationship', 'unknown'),
                'weight': attrs.get('weight', 1.0)
            })

        return jsonify({
            'nodes': nodes,
            'edges': edges
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/similar_nodes', methods=['GET'])
def similar_nodes():
    """Find similar nodes to a given node using GNN embeddings."""
    try:
        node_id = request.args.get('node_id')

        if not node_id:
            return jsonify({'error': 'No node ID provided'}), 400

        # Ensure node exists
        if node_id not in knowledge_graph.graph:
            return jsonify({'error': 'Node not found'}), 404

        # Find similar nodes
        similar = gnn_module.find_similar_nodes(node_id, k=5)

        # Format response
        results = []
        for similar_id, score in similar:
            node_data = knowledge_graph.get_node_attributes(similar_id)
            node_type = node_data.get('type', 'unknown')

            item = {
                'id': similar_id,
                'type': node_type,
                'similarity_score': score
            }

            if node_type == 'note':
                item['title'] = node_data.get('title', 'Untitled Note')
                item['content'] = node_data.get('content', '')[:100] + '...' if len(
                    node_data.get('content', '')) > 100 else node_data.get('content', '')
            elif node_type == 'task':
                item['title'] = node_data.get('title', 'Untitled Task')
                item['status'] = node_data.get('status', 'unknown')
            elif node_type == 'contact':
                item['name'] = node_data.get('name', 'Unnamed Contact')
                item['organization'] = node_data.get('organization', '')

            results.append(item)

        return jsonify({'similar_nodes': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    """Upload and process a document."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the document
            doc_info, summary_data = document_processor.process_file(file_path)

            # Add to knowledge graph as a note
            note_data = {
                'title': doc_info.title or f"Document: {filename}",
                'content': f"Summary: {summary_data.get('summary', '')}\n\nContent: {doc_info.content[:1000]}...",
                'tags': summary_data.get('topics', []) + ['document'],
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat(),
                'source': {
                    'type': 'document',
                    'filename': filename,
                    'doc_id': summary_data.get('doc_id', ''),
                    'metadata': doc_info.metadata
                }
            }

            node_id = knowledge_graph.add_note(note_data)

            # Re-train GNN if we have enough nodes
            if len(knowledge_graph.graph.nodes()) > 1:
                stats = gnn_module.train(knowledge_graph.graph, epochs=50)
                knowledge_graph.update_embeddings(
                    gnn_module.get_node_embeddings())

            return jsonify({
                'success': True,
                'node_id': node_id,
                'doc_id': summary_data.get('doc_id', ''),
                'title': doc_info.title,
                'summary': summary_data.get('summary', ''),
                'topics': summary_data.get('topics', []),
                'entities': summary_data.get('entities', {})
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get all processed documents."""
    try:
        documents = document_processor.get_all_documents()
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get a specific document by ID."""
    try:
        document = document_processor.get_document_info(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        return jsonify(document)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_app(graphspace_instance):
    """Run the Flask web application with the provided GraphSpace instance."""
    global graphspace
    graphspace = graphspace_instance

    print(f"Starting web server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
