import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import os


class LLMModule:
    """
    LLM module that implements Retrieval-Augmented Generation (RAG).
    Uses FAISS for embedding similarity search and a pretrained language model for generation.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        language_model_name: str = "distilgpt2",
        device: Optional[str] = None,
        embedding_dimension: int = 384,
        max_context_length: int = 512
    ):
        """
        Initialize the LLM module.

        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
            language_model_name: Name of the language model for generation
            device: Device to run models on ('cpu' or 'cuda')
            embedding_dimension: Dimension of the embedding vectors
            max_context_length: Maximum context length for language model
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model_name, device=self.device)
            self.embedding_dimension = embedding_dimension
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using a randomly initialized embedding function instead.")
            self.embedding_model = None
            self.embedding_dimension = embedding_dimension

        # Initialize language model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name)

            # Move model to device if cuda is available
            if self.device == "cuda":
                self.language_model = self.language_model.to("cuda")

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.language_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading language model: {e}")
            print("Using a simple placeholder response generator instead.")
            self.tokenizer = None
            self.language_model = None
            self.generator = None

        # Initialize FAISS index for context retrieval
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index_mapping = {}  # Maps FAISS index positions to context IDs
        self.context_store = {}  # Stores context info by ID

        # Settings
        self.max_context_length = max_context_length

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using the sentence transformer model.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            # Fallback: generate random embedding (for testing only)
            return np.random.randn(self.embedding_dimension).astype(np.float32)

        # Generate embedding using the model
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy().astype(np.float32)

    def add_context(self, context_items: List[Dict[str, Any]]):
        """
        Add context items to the FAISS index for retrieval.

        Args:
            context_items: List of dictionaries with 'id', 'title', 'content', 'type'
        """
        if not context_items:
            return

        # Extract embeddings and update mappings
        embeddings = []
        for i, item in enumerate(context_items):
            context_id = item["id"]
            content = f"{item.get('title', '')} {item.get('content', '')}"

            # Skip if content is empty
            if not content.strip():
                continue

            # Get embedding and normalize
            embedding = self._get_embedding(content)
            faiss.normalize_L2(np.reshape(embedding, (1, -1)))

            # Add to batch
            embeddings.append(embedding)

            # Update mappings
            next_index = self.faiss_index.ntotal + len(embeddings) - 1
            self.index_mapping[next_index] = context_id
            self.context_store[context_id] = item

        # Add embeddings to index if any were processed
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            self.faiss_index.add(embeddings_matrix)

    def clear_context(self):
        """Clear all context from the FAISS index."""
        # Reset FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index_mapping = {}
        self.context_store = {}

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant context items for a query.

        Args:
            query: User query
            k: Number of context items to retrieve

        Returns:
            List of retrieved context items
        """
        # Handle empty index
        if self.faiss_index.ntotal == 0:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)
        faiss.normalize_L2(np.reshape(query_embedding, (1, -1)))

        # Search FAISS index
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]),
            min(k, self.faiss_index.ntotal)
        )

        # Map indices to context items
        results = []
        for i, idx in enumerate(indices[0]):
            context_id = self.index_mapping.get(int(idx))
            if context_id and context_id in self.context_store:
                item = self.context_store[context_id].copy()
                item["score"] = float(distances[0][i])
                results.append(item)

        return results

    def generate_response(self, query: str, retrieved_context: List[Dict[str, Any]], max_length: int = 150) -> str:
        """
        Generate a response based on the query and retrieved context.

        Args:
            query: User query
            retrieved_context: Context items retrieved from FAISS
            max_length: Maximum length of generated response

        Returns:
            Generated response
        """
        if not self.generator:
            # Fallback response if no language model is available
            if not retrieved_context:
                return "I don't have enough information to answer that question."

            # Simple response based on first retrieved context
            context = retrieved_context[0]
            return f"Based on my knowledge about {context.get('title', 'this topic')}, I can tell you that {context.get('content', 'no specific information is available')}."

        # Format context for prompt
        context_text = ""
        for item in retrieved_context:
            item_type = item.get("type", "information")
            title = item.get("title", "")
            content = item.get("content", "")

            if title and content:
                context_text += f"- {item_type.capitalize()}: {title}\n{content}\n\n"
            elif content:
                context_text += f"- {item_type.capitalize()}: {content}\n\n"

        # Construct prompt with context and query
        prompt = f"Context information:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        try:
            outputs = self.generator(
                prompt,
                max_length=len(self.tokenizer(prompt)[
                               "input_ids"]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            # Extract generated text after prompt
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt):].strip()

            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while trying to generate a response."

    def rag_query(self, query: str, knowledge_graph, k: int = 5, max_length: int = 150) -> Dict[str, Any]:
        """
        Perform full RAG query: retrieve context and generate response.

        Args:
            query: User query
            knowledge_graph: Knowledge graph to retrieve context from
            k: Number of context items to retrieve
            max_length: Maximum length of generated response

        Returns:
            Dictionary with response and retrieved context
        """
        # Get context from knowledge graph
        context_items = knowledge_graph.get_context_for_query(
            query, max_results=k)

        # Add to FAISS index (temporary)
        temp_index = faiss.IndexFlatIP(self.embedding_dimension)
        temp_mapping = {}
        context_store = {}

        # Get embeddings for context items
        embeddings = []
        for i, item in enumerate(context_items):
            content = f"{item.get('title', '')} {item.get('content', '')}"
            if not content.strip():
                continue

            embedding = self._get_embedding(content)
            faiss.normalize_L2(np.reshape(embedding, (1, -1)))

            embeddings.append(embedding)
            temp_mapping[i] = item["id"]
            context_store[item["id"]] = item

        # Search for relevant context if any embeddings were processed
        retrieved_context = []
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            temp_index.add(embeddings_matrix)

            # Get query embedding
            query_embedding = self._get_embedding(query)
            faiss.normalize_L2(np.reshape(query_embedding, (1, -1)))

            # Search
            distances, indices = temp_index.search(
                np.array([query_embedding]),
                min(k, temp_index.ntotal)
            )

            # Map indices to context items
            for i, idx in enumerate(indices[0]):
                context_id = temp_mapping.get(int(idx))
                if context_id and context_id in context_store:
                    item = context_store[context_id].copy()
                    item["score"] = float(distances[0][i])
                    retrieved_context.append(item)

        # Generate response
        response = self.generate_response(query, retrieved_context, max_length)

        return {
            "query": query,
            "response": response,
            "context": retrieved_context
        }

    def save_index(self, path: str):
        """
        Save the FAISS index to a file.

        Args:
            path: Path to save the index
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{path}.index")

        # Save mappings
        torch.save({
            "index_mapping": self.index_mapping,
            "context_store": self.context_store
        }, f"{path}.mappings")

    def load_index(self, path: str):
        """
        Load the FAISS index from a file.

        Args:
            path: Path to load the index from
        """
        if os.path.exists(f"{path}.index") and os.path.exists(f"{path}.mappings"):
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{path}.index")

            # Load mappings
            data = torch.load(f"{path}.mappings")
            self.index_mapping = data["index_mapping"]
            self.context_store = data["context_store"]

            return True
        return False
