"""GraphSpace models module."""
from .knowledge_graph import KnowledgeGraph
from .advanced_llm_module import AdvancedLLMModule
from .embedding_module import EmbeddingModule
from .note import Note, NoteManager
from .task import Task, TaskManager
from .gnn_module import GNNModule
# Keep llm_module import for backward compatibility
from .llm_module import LLMModule
