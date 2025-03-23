from typing import List, Dict, Any, Optional, Tuple, Union
import os
import json
import requests
import numpy as np
import torch
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from .embedding_module import EmbeddingModule


class AdvancedLLMModule:
    """
    Advanced LLM module that integrates with API-based models and provides
    specialized pipelines for summarization, title generation, and RAG.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: str = "https://api.deepseek.com/v1",
        model_name: str = "deepseek-ai/deepseek-chat-v1",
        fallback_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        embedding_module: Optional[EmbeddingModule] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        use_api: bool = True,
        device: Optional[str] = None,
        max_context_length: int = 16384,
        cache_dir: str = "data/model_cache"
    ):
        """
        Initialize the advanced LLM module.

        Args:
            api_key: API key for external LLM service
            api_base_url: Base URL for API
            model_name: Name of the remote model
            fallback_model_name: Name of local model for fallback
            embedding_module: EmbeddingModule instance for retrieval
            embedding_model_name: Name of sentence transformer model for embeddings
            embedding_dimension: Dimension of embedding vectors when using internal embeddings
            use_api: Whether to use API (True) or local model (False)
            device: Device to use for local model ('cuda' or 'cpu')
            max_context_length: Maximum context length for the model
            cache_dir: Directory to cache model files
        """
        # API settings
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.use_api = use_api
        self.max_context_length = max_context_length

        # Determine device for local model
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize embedding module if not provided
        self.embedding_module = embedding_module
        self.embedding_dimension = embedding_dimension
        self.embedding_model_name = embedding_model_name
        self._init_embedding_model()

        # Initialize local model for fallback
        self.fallback_model_name = fallback_model_name
        self.local_model = None
        self.local_tokenizer = None
        self.generator = None

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize FAISS index for context retrieval (for compatibility with basic LLMModule)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index_mapping = {}  # Maps FAISS index positions to context IDs
        self.context_store = {}  # Stores context info by ID

        # Initialize local model if API is not used or for fallback purposes
        if not use_api:
            self._initialize_local_model()

    def _init_embedding_model(self):
        """Initialize embedding model for embeddings if embedding_module is not provided."""
        if self.embedding_module is not None:
            return

        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name, device=self.device)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using a randomly initialized embedding function instead.")
            self.embedding_model = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using the sentence transformer model.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.embedding_module is not None:
            return self.embedding_module.embed_text(text)

        if self.embedding_model is None:
            # Fallback: generate random embedding (for testing only)
            return np.random.randn(self.embedding_dimension).astype(np.float32)

        # Generate embedding using the model
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy().astype(np.float32)

    def _initialize_local_model(self):
        """Initialize local LLM for generation."""
        try:
            print(f"Loading local model: {self.fallback_model_name}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model_name,
                cache_dir=self.cache_dir
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            # Create text generation pipeline for compatibility with basic LLMModule
            self.generator = pipeline(
                "text-generation",
                model=self.local_model,
                tokenizer=self.local_tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            print(f"Local model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("No fallback model available. Will use API exclusively.")
            self.local_model = None
            self.local_tokenizer = None
            self.generator = None

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

    def _api_generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_message: str = "You are a helpful assistant."
    ) -> str:
        """
        Generate text using the API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            system_message: System message to set context

        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("API key is required for API generation")

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"API generation error: {e}")
            # Fall back to local model if available
            if self.local_model and self.local_tokenizer:
                return self._local_generate(prompt, max_tokens, temperature, system_message)
            return f"Error generating response: {str(e)}"

    def _local_generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_message: str = "You are a helpful assistant."
    ) -> str:
        """
        Generate text using the local model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            system_message: System message to set context

        Returns:
            Generated text
        """
        if not self.local_model or not self.local_tokenizer:
            raise ValueError("Local model not initialized")

        try:
            # Format prompt with system message for instruction-tuned models
            formatted_prompt = f"<s>[INST] {system_message}\n\n{prompt} [/INST]"

            inputs = self.local_tokenizer(
                formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )

            # Decode the generated tokens, skipping the prompt
            prompt_length = inputs.input_ids.shape[1]
            generated_text = self.local_tokenizer.decode(
                outputs[0][prompt_length:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            print(f"Local generation error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_response(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response based on prompt and optional context.

        Args:
            prompt: Input prompt
            context: List of context items
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated response
        """
        if context:
            # Format context
            formatted_context = self._format_context(context)
            # Construct prompt with context
            full_prompt = f"Context information:\n{formatted_context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt

        # Generate response
        if self.use_api and self.api_key:
            return self._api_generate(full_prompt, max_tokens, temperature)
        elif self.local_model and self.local_tokenizer:
            return self._local_generate(full_prompt, max_tokens, temperature)
        else:
            return "No generation model available."

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context items for prompt.

        Args:
            context: List of context items

        Returns:
            Formatted context string
        """
        formatted_context = ""

        for i, item in enumerate(context):
            item_type = item.get("type", "information")
            title = item.get("title", "")
            content = item.get("text", item.get("content", ""))
            metadata = item.get("metadata", {})

            if metadata:
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                # Extract metadata fields if available
                if not title and "title" in metadata:
                    title = metadata["title"]

            # Format based on type and available information
            if title and content:
                formatted_context += f"[{i+1}] {item_type.capitalize()}: {title}\n{content}\n\n"
            elif content:
                formatted_context += f"[{i+1}] {item_type.capitalize()}: {content}\n\n"

        return formatted_context

    def summarize_document(
        self,
        document: Dict[str, Any],
        max_length: int = 200,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a summary for a document.

        Args:
            document: Document to summarize
            max_length: Maximum length of summary
            temperature: Temperature for generation

        Returns:
            Generated summary
        """
        content = document.get("content", "")
        title = document.get("title", "")

        if not content:
            return "No content to summarize."

        # Construct prompt
        prompt = f"""Please provide a comprehensive summary of the following document. Focus on the main points, key findings, and important details. Keep your summary concise but complete.

Document title: {title}

Document content:
{content[:50000]}  # Limit for very large documents

Summary:"""

        system_message = "You are an expert summarizer. Your summaries are comprehensive, accurate, and concise."

        # Generate summary
        if self.use_api and self.api_key:
            return self._api_generate(prompt, max_length, temperature, system_message)
        elif self.local_model and self.local_tokenizer:
            return self._local_generate(prompt, max_length, temperature, system_message)
        else:
            return "No generation model available."

    def generate_title(
        self,
        content: str,
        max_length: int = 50,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a title for content.

        Args:
            content: Content to generate title for
            max_length: Maximum length of title
            temperature: Temperature for generation

        Returns:
            Generated title
        """
        if not content:
            return "Untitled"

        # Truncate content if too long
        truncated_content = content[:10000] if len(
            content) > 10000 else content

        # Construct prompt
        prompt = f"""Based on the following content, generate a concise, descriptive, and informative title that accurately reflects the main topic or theme. The title should be clear, engaging, and generally less than 10 words.

Content:
{truncated_content}

Title:"""

        system_message = "You are a professional editor who excels at creating accurate, concise, and descriptive titles."

        # Generate title
        if self.use_api and self.api_key:
            return self._api_generate(prompt, max_length, temperature, system_message)
        elif self.local_model and self.local_tokenizer:
            return self._local_generate(prompt, max_length, temperature, system_message)
        else:
            return "Untitled"

    def extract_tags(
        self,
        content: str,
        max_tags: int = 5,
        temperature: float = 0.3
    ) -> List[str]:
        """
        Extract relevant tags from content.

        Args:
            content: Content to extract tags from
            max_tags: Maximum number of tags
            temperature: Temperature for generation

        Returns:
            List of tags
        """
        if not content:
            return []

        # Truncate content if too long
        truncated_content = content[:10000] if len(
            content) > 10000 else content

        # Construct prompt
        prompt = f"""Extract up to {max_tags} relevant tags or keywords from the following content. The tags should accurately represent the main topics, themes, or entities discussed. Return the tags as a comma-separated list.

Content:
{truncated_content}

Tags:"""

        system_message = "You are an expert at extracting relevant keywords and tags from content."

        # Generate tags
        if self.use_api and self.api_key:
            tags_text = self._api_generate(
                prompt, 100, temperature, system_message)
        elif self.local_model and self.local_tokenizer:
            tags_text = self._local_generate(
                prompt, 100, temperature, system_message)
        else:
            return []

        # Process tags
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        return tags[:max_tags]  # Limit to max_tags

    def rag_query(
        self,
        query: str,
        knowledge_graph=None,
        embedding_module=None,
        search_level: int = 2,
        top_k: int = 5,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform full RAG query.

        Args:
            query: User query
            knowledge_graph: Knowledge graph to query
            embedding_module: EmbeddingModule to use for search
            search_level: Chunk level to search
            top_k: Number of results to retrieve
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response and retrieved context
        """
        context = []

        # Get context from knowledge graph if available
        if knowledge_graph:
            context = knowledge_graph.get_context_for_query(
                query, max_results=top_k)

        # Get context from embedding module if available and no knowledge graph results
        elif embedding_module or self.embedding_module:
            emb_module = embedding_module or self.embedding_module
            context = emb_module.search(query, level=search_level, top_k=top_k)

        # Fallback to internal FAISS index if needed
        elif self.faiss_index.ntotal > 0:
            context = self.retrieve_context(query, top_k)

        # Generate response
        response = self.generate_response(query, context, max_tokens)

        return {
            "query": query,
            "response": response,
            "context": context
        }

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.

        Args:
            query: Complex user query

        Returns:
            List of sub-queries
        """
        if not query or len(query) < 50:
            return [query]  # Don't decompose short queries

        prompt = f"""Analyze the following complex query and break it down into a series of simpler, logical sub-queries that together would answer the original question. Return only the list of sub-queries as a numbered list.

Complex query: {query}

Sub-queries:"""

        system_message = "You are an expert at analyzing complex questions and breaking them down into logical components."

        # Generate decomposition
        if self.use_api and self.api_key:
            decomposition = self._api_generate(
                prompt, 500, 0.3, system_message)
        elif self.local_model and self.local_tokenizer:
            decomposition = self._local_generate(
                prompt, 500, 0.3, system_message)
        else:
            return [query]

        # Process decomposition
        lines = decomposition.strip().split("\n")
        sub_queries = []

        for line in lines:
            # Extract sub-queries from numbered or bulleted list
            line = line.strip()
            if not line:
                continue

            # Remove numbering/bullets and any extra formatting
            if line[0].isdigit():
                # Handle numbered lists like "1. Query"
                parts = line.split(".", 1)
                if len(parts) > 1:
                    sub_queries.append(parts[1].strip())
            elif line[0] in ['-', '*', '•']:
                # Handle bullet lists
                sub_queries.append(line[1:].strip())
            else:
                # Handle any remaining text
                sub_queries.append(line)

        # If parsing failed, return original query
        if not sub_queries:
            return [query]

        return sub_queries

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
