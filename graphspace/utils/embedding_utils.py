import numpy as np
from typing import List, Dict, Any, Optional
import os
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingUtils:
    """Utility class for handling text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding utilities.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using a randomly initialized embedding function instead.")
            self.model = None
            self.dimension = 384  # Default dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using the sentence transformer model.

        Args:
            text: Input text

        Returns:
            Embedding vector (numpy array)
        """
        if self.model is None:
            # Fallback: generate random embedding (for testing only)
            return np.random.randn(self.dimension).astype(np.float32)

        # Generate embedding using the model
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy().astype(np.float32)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of input texts

        Returns:
            Matrix of embeddings (numpy array)
        """
        if self.model is None:
            # Fallback: generate random embeddings (for testing only)
            return np.random.randn(len(texts), self.dimension).astype(np.float32)

        # Generate embeddings using the model
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy().astype(np.float32)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (float)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length.

        Args:
            embedding: Input embedding vector

        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
