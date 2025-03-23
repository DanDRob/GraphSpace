#!/usr/bin/env python3
from services.document_processor import DocumentProcessor
from services.google_drive_service import GoogleDriveService
from models.task import TaskManager, Task
from models.note import NoteManager, Note
from models.advanced_llm_module import AdvancedLLMModule
from models.embedding_module import EmbeddingModule
from models.knowledge_graph import KnowledgeGraph
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Get DeepSeek API key from environment
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# Initialize client with DeepSeek configuration
client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)


class EnhancedGraphSpace:
    """
    Enhanced GraphSpace integrating all advanced features:
    - Improved embeddings
    - Advanced LLM with API integration
    - Batch document processing
    - Google Drive integration
    - Enhanced knowledge graph
    """

    def __init__(
        self,
        data_path: str = "data/user_data.json",
        config_path: str = "config/config.json",
        use_api: bool = True,
        api_key: Optional[str] = None,
        use_google_drive: bool = False,
        google_credentials_file: Optional[str] = None
    ):
        """
        Initialize EnhancedGraphSpace.

        Args:
            data_path: Path to user data file
            config_path: Path to configuration file
            use_api: Whether to use API for LLM
            api_key: API key for LLM service
            use_google_drive: Whether to enable Google Drive integration
            google_credentials_file: Path to Google credentials file
        """
        self.data_path = data_path
        self.config = self._load_config(config_path)

        # Extract API key from environment if not provided
        if api_key is None and "DEEPSEEK_API_KEY" in os.environ:
            api_key = os.environ["DEEPSEEK_API_KEY"]

        # Initialize core components
        self.knowledge_graph = self._init_knowledge_graph(data_path)
        self.embedding_module = self._init_embedding_module()
        self.llm_module = self._init_llm_module(use_api, api_key)
        self.document_processor = self._init_document_processor()

        # Initialize managers
        self.note_manager = NoteManager()
        self.task_manager = TaskManager()

        # Initialize Google Drive service if enabled
        self.google_drive_service = None
        if use_google_drive:
            self.google_drive_service = self._init_google_drive_service(
                google_credentials_file)

        # Synchronize components
        self._sync_components()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file or create a default one."""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")

        # Default configuration
        default_config = {
            "embedding": {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "dimension": 768
            },
            "llm": {
                "api_enabled": True,
                "model": "deepseek-ai/deepseek-chat-v1",
                "fallback_model": "meta-llama/Llama-3-8B-Instruct"
            },
            "document_processing": {
                "max_workers": 4,
                "chunk_size": 1000
            }
        }

        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _init_knowledge_graph(self, data_path: str) -> KnowledgeGraph:
        """Initialize knowledge graph from data file."""
        return KnowledgeGraph(data_path=data_path)

    def _init_embedding_module(self) -> EmbeddingModule:
        """Initialize embedding module based on configuration."""
        model_name = self.config["embedding"]["model"]
        dimension = self.config["embedding"]["dimension"]
        return EmbeddingModule(model_name=model_name, embedding_dimension=dimension)

    def _init_llm_module(self, use_api: bool, api_key: Optional[str]) -> AdvancedLLMModule:
        """Initialize LLM module based on configuration."""
        model_name = self.config["llm"]["model"]
        fallback_model = self.config["llm"]["fallback_model"]
        return AdvancedLLMModule(
            api_key=api_key,
            model_name=model_name,
            fallback_model_name=fallback_model,
            embedding_module=self.embedding_module,
            use_api=use_api
        )

    def _init_document_processor(self) -> DocumentProcessor:
        """Initialize document processor based on configuration."""
        max_workers = self.config["document_processing"]["max_workers"]
        chunk_size = self.config["document_processing"]["chunk_size"]
        return DocumentProcessor(
            llm_module=self.llm_module,
            embedding_module=self.embedding_module,
            max_workers=max_workers,
            chunk_size=chunk_size
        )

    def _init_google_drive_service(self, credentials_file: Optional[str]) -> GoogleDriveService:
        """Initialize Google Drive service."""
        return GoogleDriveService(
            credentials_file=credentials_file,
            document_processor=self.document_processor
        )

    def _sync_components(self):
        """Synchronize components after initialization."""
        # Train GNN if there are nodes in the graph
        if hasattr(self.embedding_module, 'train_on_graph') and len(self.knowledge_graph.graph.nodes()) > 1:
            print("Training embedding module on graph...")
            self.embedding_module.train_on_graph(self.knowledge_graph.graph)
            self.knowledge_graph.update_embeddings(
                self.embedding_module.get_node_embeddings())

    def add_note(self, note_data: Dict[str, Any]) -> str:
        """
        Add a new note to the system.

        Args:
            note_data: Dictionary with note data

        Returns:
            ID of the new note
        """
        # Check if we need to generate title and tags
        if not note_data.get("title") and note_data.get("content"):
            title = self.llm_module.generate_title(note_data["content"])
            note_data["title"] = title or "Untitled Note"

        if not note_data.get("tags") and note_data.get("content"):
            tags = self.llm_module.extract_tags(note_data["content"])
            note_data["tags"] = tags

        # Set timestamps if not provided
        now = datetime.now().isoformat()
        if not note_data.get("created_at"):
            note_data["created_at"] = now
        if not note_data.get("updated_at"):
            note_data["updated_at"] = now

        # Add to knowledge graph
        node_id = self.knowledge_graph.add_note(note_data)

        # Create embeddings
        if self.embedding_module:
            try:
                chunks = self.embedding_module.chunk_document(note_data)
                self.embedding_module.store_chunks(chunks)
            except Exception as e:
                print(f"Error creating embeddings: {e}")

        # Create note object
        note = Note.from_dict(note_data)
        self.note_manager.add_note(note)

        return node_id

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with processing results
        """
        result = self.document_processor.process_single_file(file_path)

        # Add to knowledge graph if successful
        if result.get("success", False) and "document" in result:
            doc = result["document"]
            note_data = {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "tags": doc.get("tags", []),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "source": {
                    "type": "document",
                    "filename": os.path.basename(file_path),
                    "doc_id": "",
                    "metadata": {
                        "file_type": doc.get("file_type"),
                        "file_size": doc.get("file_size")
                    }
                }
            }

            # Add summary if available
            if "summary" in doc:
                note_data["summary"] = doc["summary"]

            # Add to knowledge graph
            node_id = self.knowledge_graph.add_note(note_data)
            result["node_id"] = node_id

        return result

    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories

        Returns:
            List of processing results
        """
        # Process all files in the directory
        results = self.document_processor.process_files_in_directory(
            directory_path,
            recursive=recursive
        )

        # Add to knowledge graph
        for result in results:
            if result.get("success", False) and "document" in result:
                doc = result["document"]
                note_data = {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "tags": doc.get("tags", []),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "source": {
                        "type": "document",
                        "filename": os.path.basename(result.get("file_path", "")),
                        "doc_id": "",
                        "metadata": {
                            "file_type": doc.get("file_type"),
                            "file_size": doc.get("file_size")
                        }
                    }
                }

                # Add summary if available
                if "summary" in doc:
                    note_data["summary"] = doc["summary"]

                # Add to knowledge graph
                node_id = self.knowledge_graph.add_note(note_data)
                result["node_id"] = node_id

        return results

    def sync_google_drive_folder(
        self,
        folder_id: str = "root",
        recursive: bool = True,
        file_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sync a Google Drive folder to the knowledge graph.

        Args:
            folder_id: ID of the folder to sync
            recursive: Whether to process subfolders
            file_types: Optional list of MIME types to filter by

        Returns:
            List of notes created
        """
        if not self.google_drive_service:
            raise ValueError("Google Drive integration is not enabled")

        # Convert notes from the Google Drive folder
        notes = self.google_drive_service.convert_folder_to_notes(
            folder_id=folder_id,
            recursive=recursive,
            file_types=file_types,
            llm_module=self.llm_module
        )

        results = []

        # Add each note to the system
        for note_data in notes:
            node_id = self.add_note(note_data)

            results.append({
                "success": True,
                "node_id": node_id,
                "note": note_data
            })

        return results

    def check_google_drive_changes(self) -> List[Dict[str, Any]]:
        """
        Check for changes in Google Drive and update the knowledge graph.

        Returns:
            List of updated files/notes
        """
        if not self.google_drive_service:
            raise ValueError("Google Drive integration is not enabled")

        # Check for changes
        changed_files = self.google_drive_service.check_for_changes()

        results = []

        # Process each changed file
        for file_data in changed_files:
            file_id = file_data.get("id")
            mime_type = file_data.get("mimeType", "")

            # Skip folders
            if mime_type == "application/vnd.google-apps.folder":
                continue

            # Convert to note
            note_data = self.google_drive_service.convert_to_note(
                file_id=file_id,
                llm_module=self.llm_module
            )

            if note_data:
                # Check if note already exists with this file ID
                existing_notes = self.knowledge_graph.search_nodes(file_id)

                if existing_notes:
                    # Update existing note
                    for node_id in existing_notes:
                        node_data = self.knowledge_graph.get_node_attributes(
                            node_id)
                        if node_data.get("type") == "note":
                            # Update note data
                            update_data = node_data.get("data", {}).copy()
                            update_data.update(note_data)

                            # Update in knowledge graph
                            self.knowledge_graph.data["notes"][node_data["data"]
                                                               ["id"]] = update_data
                            self.knowledge_graph.build_graph()
                            self.knowledge_graph.save_data()

                            results.append({
                                "success": True,
                                "node_id": node_id,
                                "note": update_data,
                                "action": "updated"
                            })
                else:
                    # Add new note
                    node_id = self.add_note(note_data)

                    results.append({
                        "success": True,
                        "node_id": node_id,
                        "note": note_data,
                        "action": "created"
                    })

        return results

    def query(self, query: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Query the knowledge graph with natural language.

        Args:
            query: Natural language query
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with query, response, and context
        """
        # Handle complex queries by decomposing them
        if len(query) > 50:
            sub_queries = self.llm_module.decompose_query(query)

            # If decomposition was successful
            if len(sub_queries) > 1 and sub_queries[0] != query:
                # Execute each sub-query
                context = []
                for sub_query in sub_queries:
                    # Get context for this sub-query
                    sub_context = self.knowledge_graph.get_context_for_query(
                        sub_query)

                    # Add non-duplicate context
                    for item in sub_context:
                        if not any(c.get("id") == item.get("id") for c in context):
                            context.append(item)

                # Generate response with combined context
                response = self.llm_module.generate_response(
                    query, context, max_tokens)

                return {
                    "query": query,
                    "response": response,
                    "context": context,
                    "sub_queries": sub_queries
                }

        # For simple queries or if decomposition failed
        result = self.llm_module.rag_query(
            query=query,
            knowledge_graph=self.knowledge_graph,
            embedding_module=self.embedding_module,
            max_tokens=max_tokens
        )

        return result

    def search(self, query: str, level: int = 2, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content using vector search.

        Args:
            query: Search query
            level: Chunk level to search at (0-3)
            top_k: Number of results to return

        Returns:
            List of search results
        """
        if not self.embedding_module:
            # Fallback to knowledge graph search
            context_items = self.knowledge_graph.get_context_for_query(
                query, max_results=top_k)
            return context_items

        # Search using embedding module
        results = self.embedding_module.search(query, level=level, top_k=top_k)
        return results

    def save(self):
        """Save all data to disk."""
        # Save knowledge graph
        self.knowledge_graph.save_data()

        # Save note manager
        self.note_manager.save_notes()

        # Save task manager
        self.task_manager.save_tasks()

        print("All data saved successfully")
