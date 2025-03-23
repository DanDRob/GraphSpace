from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import json
from dataclasses import dataclass, field
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import lancedb
from lancedb.embeddings import get_registry
import pyarrow as pa


@dataclass
class TextChunk:
    """Represents a chunk of text with hierarchical relationships."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    source_id: str = ""  # ID of the source document
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    level: int = 0  # 0=document, 1=section, 2=paragraph, 3=sentence
    position: int = 0  # Position within parent


class EmbeddingModule:
    """
    Enhanced embedding module with tiered architecture and hierarchical chunking.
    Supports both local models and API-based embeddings.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        embedding_dimension: int = 1024,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        use_api: bool = False,
        device: Optional[str] = None,
        storage_path: str = "data/vector_db",
        chunk_sizes: Dict[int, int] = None
    ):
        """
        Initialize the embedding module.

        Args:
            model_name: Name or path of the embedding model
            embedding_dimension: Dimension of embeddings
            api_key: API key for external embedding service
            api_endpoint: Endpoint URL for external embedding service
            use_api: Whether to use the API instead of local model
            device: Device to run models on ('cpu' or 'cuda')
            storage_path: Path to store vector database
            chunk_sizes: Dictionary mapping level to chunk size
        """
        # Determine device for local model
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # API settings
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.use_api = use_api

        # Model configuration
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension

        # Default chunk sizes if not provided
        if chunk_sizes is None:
            self.chunk_sizes = {
                0: 100000,  # Document
                1: 10000,   # Section
                2: 2000,    # Paragraph
                3: 500      # Sentence
            }
        else:
            self.chunk_sizes = chunk_sizes

        # Initialize local model if not using API exclusively
        if not self.use_api:
            try:
                self.model = SentenceTransformer(
                    model_name, device=self.device)
                print(f"Loaded embedding model: {model_name} on {self.device}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Using a randomly initialized embedding function instead.")
                self.model = None
        else:
            self.model = None

        # Initialize text splitters for hierarchical chunking
        self.splitters = {
            0: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_sizes[0],
                chunk_overlap=200,
                separators=["\n\n\n", "\n\n", "\n",
                            ".", "!", "?", ",", " ", ""]
            ),
            1: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_sizes[1],
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            ),
            2: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_sizes[2],
                chunk_overlap=50,
                separators=["\n", ".", "!", "?", ",", " ", ""]
            ),
            3: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_sizes[3],
                chunk_overlap=20,
                separators=[".", "!", "?", ",", " ", ""]
            )
        }

        # Initialize vector database
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Initialize LanceDB with the BGE embeddings
        self.vector_db = lancedb.connect(storage_path)
        self.embedding_model = get_registry().get(
            "sentence-transformers").create(name=model_name)

        # Create tables if they don't exist
        self._init_tables()

    def _init_tables(self):
        """Initialize vector database tables for each level."""
        # Define schema using pyarrow
        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
            ("source_id", pa.string()),
            ("parent_id", pa.string()),
            ("child_ids", pa.list_(pa.string())),
            ("metadata", pa.string()),  # JSON as string
            ("level", pa.int32()),
            ("position", pa.int32()),
            ("vector", pa.list_(pa.float32(), self.embedding_dimension))
        ])

        # Create a table for each level if it doesn't exist
        for level in range(4):
            table_name = f"chunks_level_{level}"
            if table_name not in self.vector_db.table_names():
                self.vector_db.create_table(
                    table_name,
                    schema=schema
                )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not text.strip():
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        if self.use_api and self.api_endpoint and self.api_key:
            return self._get_api_embedding(text)
        elif self.model is not None:
            return self._get_local_embedding(text)
        else:
            # Fallback: random embedding (for testing)
            return np.random.randn(self.embedding_dimension).astype(np.float32)

    def _get_local_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy().astype(np.float32)

    def _get_api_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json={"text": text}
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"API embedding error: {e}")
            # Fallback to local model if available
            if self.model is not None:
                return self._get_local_embedding(text)
            # Last resort fallback
            return np.random.randn(self.embedding_dimension).astype(np.float32)

    def chunk_document(self, document: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk document hierarchically into multiple levels.

        Args:
            document: Document with text content

        Returns:
            List of TextChunk objects
        """
        doc_id = document.get("id", str(uuid.uuid4()))
        text = document.get("content", "")
        metadata = {
            "title": document.get("title", ""),
            "tags": document.get("tags", []),
            "created_at": document.get("created_at", ""),
            "updated_at": document.get("updated_at", ""),
            "source": document.get("source", {})
        }

        # Create document-level chunk (level 0)
        doc_chunk = TextChunk(
            text=text,
            source_id=doc_id,
            parent_id=None,
            level=0,
            position=0,
            metadata=metadata
        )

        all_chunks = [doc_chunk]
        level_chunks = {0: [doc_chunk]}

        # Create hierarchical chunks for each level
        for level in range(1, 4):
            level_chunks[level] = []
            parent_level = level - 1

            # For each parent chunk, create child chunks
            for parent_idx, parent_chunk in enumerate(level_chunks[parent_level]):
                child_texts = self.splitters[level].split_text(
                    parent_chunk.text)

                # Create chunk objects for this level
                for i, child_text in enumerate(child_texts):
                    child_chunk = TextChunk(
                        text=child_text,
                        source_id=doc_id,
                        parent_id=parent_chunk.id,
                        level=level,
                        position=i,
                        metadata=metadata.copy()  # Copy parent metadata
                    )
                    parent_chunk.child_ids.append(child_chunk.id)
                    level_chunks[level].append(child_chunk)
                    all_chunks.append(child_chunk)

        # Generate embeddings for all chunks
        for chunk in all_chunks:
            chunk.embedding = self.embed_text(chunk.text)

        return all_chunks

    def store_chunks(self, chunks: List[TextChunk]):
        """
        Store chunks in the vector database.

        Args:
            chunks: List of TextChunk objects
        """
        # Group chunks by level
        level_chunks = {0: [], 1: [], 2: [], 3: []}
        for chunk in chunks:
            level_chunks[chunk.level].append(chunk)

        # Store chunks for each level
        for level, chunks_at_level in level_chunks.items():
            if not chunks_at_level:
                continue

            table = self.vector_db.open_table(f"chunks_level_{level}")

            # Convert chunks to records
            records = []
            for chunk in chunks_at_level:
                record = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source_id": chunk.source_id,
                    "parent_id": chunk.parent_id or "",
                    "child_ids": chunk.child_ids,
                    "metadata": json.dumps(chunk.metadata),
                    "level": chunk.level,
                    "position": chunk.position,
                    "vector": chunk.embedding
                }
                records.append(record)

            # Add to vector database
            table.add(records)

    def search(self, query: str, level: int = 2, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to query.

        Args:
            query: Search query
            level: Chunk level to search (0-3)
            top_k: Number of results to return

        Returns:
            List of search results
        """
        table = self.vector_db.open_table(f"chunks_level_{level}")

        # Search using the query
        results = table.search(query).limit(top_k).to_list()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "text": result["text"],
                "source_id": result["source_id"],
                "parent_id": result["parent_id"],
                "child_ids": result["child_ids"],
                "metadata": json.loads(result["metadata"]),
                "level": result["level"],
                "position": result["position"],
                "score": float(result["_distance"])
            })

        return formatted_results

    def get_children(self, chunk_id: str, level: int) -> List[Dict[str, Any]]:
        """
        Get child chunks for a given chunk.

        Args:
            chunk_id: ID of the parent chunk
            level: Level of the parent chunk

        Returns:
            List of child chunks
        """
        child_level = level + 1
        if child_level > 3:
            return []

        table = self.vector_db.open_table(f"chunks_level_{child_level}")
        results = table.search(f"parent_id:{chunk_id}").to_list()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "text": result["text"],
                "source_id": result["source_id"],
                "parent_id": result["parent_id"],
                "child_ids": result["child_ids"],
                "metadata": json.loads(result["metadata"]),
                "level": result["level"],
                "position": result["position"]
            })

        # Sort by position
        formatted_results.sort(key=lambda x: x["position"])
        return formatted_results

    def get_parent(self, chunk_id: str, level: int) -> Optional[Dict[str, Any]]:
        """
        Get parent chunk for a given chunk.

        Args:
            chunk_id: ID of the child chunk
            level: Level of the child chunk

        Returns:
            Parent chunk or None
        """
        if level <= 0:
            return None

        parent_level = level - 1

        # Get the chunk to find its parent ID
        table = self.vector_db.open_table(f"chunks_level_{level}")
        chunk_results = table.search(f"id:{chunk_id}").to_list()

        if not chunk_results:
            return None

        parent_id = chunk_results[0]["parent_id"]

        # Get the parent chunk
        parent_table = self.vector_db.open_table(
            f"chunks_level_{parent_level}")
        parent_results = parent_table.search(f"id:{parent_id}").to_list()

        if not parent_results:
            return None

        parent = parent_results[0]
        return {
            "id": parent["id"],
            "text": parent["text"],
            "source_id": parent["source_id"],
            "parent_id": parent["parent_id"],
            "child_ids": parent["child_ids"],
            "metadata": json.loads(parent["metadata"]),
            "level": parent["level"],
            "position": parent["position"]
        }

    def get_document_chunks(self, source_id: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all chunks for a document.

        Args:
            source_id: ID of the source document

        Returns:
            Dictionary mapping level to list of chunks
        """
        document_chunks = {}

        for level in range(4):
            table = self.vector_db.open_table(f"chunks_level_{level}")
            results = table.search(f"source_id:{source_id}").to_list()

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "text": result["text"],
                    "source_id": result["source_id"],
                    "parent_id": result["parent_id"],
                    "child_ids": result["child_ids"],
                    "metadata": json.loads(result["metadata"]),
                    "level": result["level"],
                    "position": result["position"]
                })

            # Sort by position
            formatted_results.sort(key=lambda x: x["position"])
            document_chunks[level] = formatted_results

        return document_chunks
