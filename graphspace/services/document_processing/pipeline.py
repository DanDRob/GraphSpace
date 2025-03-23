import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, BinaryIO, Set
import concurrent.futures
from pathlib import Path
import uuid
import json

from .extractors import DocumentInfo, ExtractorFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents by extracting text, generating summaries,
    and extracting metadata.
    """

    def __init__(self, llm_module=None, storage_dir: str = None):
        """
        Initialize the document processor.

        Args:
            llm_module: Module for generating summaries and analyzing content
            storage_dir: Directory to store processed documents
        """
        self.llm_module = llm_module

        if storage_dir is None:
            # Use a default storage directory
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))
            storage_dir = os.path.join(base_dir, "data", "documents")

        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def process_file(self, file_path: str) -> Tuple[DocumentInfo, Dict[str, Any]]:
        """
        Process a file on disk.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of DocumentInfo and additional metadata
        """
        logger.info(f"Processing file: {file_path}")

        # Extract text and metadata from the file
        doc_info = ExtractorFactory.extract_from_file(file_path)

        # Generate a summary and additional metadata
        summary_data = self.generate_summary_and_metadata(doc_info)

        # Store the processed document
        self.store_document(doc_info, summary_data)

        return doc_info, summary_data

    def process_bytes(self, file_bytes: bytes, file_name: str) -> Tuple[DocumentInfo, Dict[str, Any]]:
        """
        Process a file from bytes.

        Args:
            file_bytes: Bytes containing the file content
            file_name: Name of the file (used to determine the file type)

        Returns:
            Tuple of DocumentInfo and additional metadata
        """
        logger.info(f"Processing file bytes for: {file_name}")

        # Extract text and metadata from the file bytes
        doc_info = ExtractorFactory.extract_from_bytes(file_bytes, file_name)

        # Generate a summary and additional metadata
        summary_data = self.generate_summary_and_metadata(doc_info)

        # Store the processed document
        self.store_document(doc_info, summary_data)

        return doc_info, summary_data

    def generate_summary_and_metadata(self, doc_info: DocumentInfo) -> Dict[str, Any]:
        """
        Generate a summary and additional metadata for a document.

        Args:
            doc_info: Document information

        Returns:
            Dictionary containing summary and metadata
        """
        # Extract topics and entities
        topics = self.extract_topics(doc_info.content)
        entities = self.extract_entities(doc_info.content)

        # Generate summary
        summary = self.generate_summary(doc_info)

        # Collect additional metadata
        metadata = {
            "summary": summary,
            "topics": topics,
            "entities": entities,
            "wordcount": len(doc_info.content.split()),
            "charcount": len(doc_info.content),
            "timestamp": doc_info.metadata.get("created", "")
        }

        return metadata

    def extract_topics(self, content: str) -> List[str]:
        """
        Extract main topics from the document content.

        Args:
            content: Document content

        Returns:
            List of topics
        """
        # If LLM module is available, use it to extract topics
        if self.llm_module:
            try:
                # Simple prompt for topic extraction
                prompt = f"Extract the main topics from the following text. Return only a list of topics separated by commas with no additional text:\n\n{content[:5000]}"

                response = self.llm_module.generate_response(
                    prompt, [], max_length=100)

                # Parse the comma-separated list
                topics = [topic.strip()
                          for topic in response.split(',') if topic.strip()]
                return topics[:5]  # Limit to top 5 topics
            except Exception as e:
                logger.error(f"Error extracting topics with LLM: {e}")

        # Fallback: Simple frequency-based extraction
        # This is a very basic approach and should be replaced with a better method
        words = content.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
                      'by', 'about', 'as', 'of', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
        filtered_words = [
            w for w in words if w not in stop_words and len(w) > 3]

        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(),
                              key=lambda x: x[1], reverse=True)

        # Return top words as topics
        return [word for word, _ in sorted_words[:5]]

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the document content.

        Args:
            content: Document content

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        # If LLM module is available, use it to extract entities
        if self.llm_module:
            try:
                # Simple prompt for entity extraction
                prompt = f"Extract named entities from the following text. Categorize them as PERSON, ORGANIZATION, LOCATION, or DATE. Return the results in this format: 'CATEGORY: entity1, entity2, etc.' with each category on a new line:\n\n{content[:5000]}"

                response = self.llm_module.generate_response(
                    prompt, [], max_length=200)

                # Parse the response
                entities = {}
                for line in response.split('\n'):
                    if ':' in line:
                        category, values = line.split(':', 1)
                        category = category.strip().upper()
                        entity_list = [e.strip()
                                       for e in values.split(',') if e.strip()]
                        entities[category] = entity_list

                return entities
            except Exception as e:
                logger.error(f"Error extracting entities with LLM: {e}")

        # Fallback: Return empty result
        return {}

    def generate_summary(self, doc_info: DocumentInfo) -> str:
        """
        Generate a summary of the document.

        Args:
            doc_info: Document information

        Returns:
            Summary text
        """
        # If LLM module is available, use it to generate a summary
        if self.llm_module:
            try:
                # Simple prompt for summary generation
                prompt = f"Generate a concise summary (2-3 sentences) of the following document:\nTitle: {doc_info.title}\n\nContent: {doc_info.content[:5000]}"

                summary = self.llm_module.generate_response(
                    prompt, [], max_length=200)
                return summary
            except Exception as e:
                logger.error(f"Error generating summary with LLM: {e}")

        # Fallback: Create a basic summary using the first paragraph
        paragraphs = doc_info.content.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) > 200:
                return first_para[:197] + "..."
            return first_para

        return "No summary available."

    def store_document(self, doc_info: DocumentInfo, summary_data: Dict[str, Any]) -> str:
        """
        Store the processed document.

        Args:
            doc_info: Document information
            summary_data: Summary and metadata

        Returns:
            Path to the stored document
        """
        # Create a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Create a storage directory for this document
        doc_dir = os.path.join(self.storage_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        # Store the document information
        doc_data = doc_info.to_dict()
        doc_data.update({
            "summary": summary_data.get("summary", ""),
            "topics": summary_data.get("topics", []),
            "entities": summary_data.get("entities", {}),
            "process_date": doc_info.metadata.get("process_date", "")
        })

        # Save the document metadata
        metadata_path = os.path.join(doc_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2)

        # Save the content to a text file
        content_path = os.path.join(doc_dir, "content.txt")
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(doc_info.content)

        return doc_dir

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a stored document.

        Args:
            doc_id: ID of the document

        Returns:
            Document information or None if not found
        """
        doc_dir = os.path.join(self.storage_dir, doc_id)
        if not os.path.exists(doc_dir):
            return None

        metadata_path = os.path.join(doc_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get information about all stored documents.

        Returns:
            List of document information dictionaries
        """
        documents = []

        for doc_id in os.listdir(self.storage_dir):
            doc_dir = os.path.join(self.storage_dir, doc_id)
            if os.path.isdir(doc_dir):
                metadata_path = os.path.join(doc_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            doc_data["id"] = doc_id
                            documents.append(doc_data)
                    except Exception as e:
                        logger.error(f"Error loading document {doc_id}: {e}")

        return documents


class BatchDocumentProcessor:
    """
    Processes multiple documents in parallel.
    """

    def __init__(self, processor: DocumentProcessor, max_workers: int = 4):
        """
        Initialize the batch processor.

        Args:
            processor: Document processor
            max_workers: Maximum number of worker processes
        """
        self.processor = processor
        self.max_workers = max_workers

    def process_files(self, file_paths: List[str]) -> List[Tuple[DocumentInfo, Dict[str, Any]]]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of (DocumentInfo, metadata) tuples
        """
        logger.info(f"Processing {len(file_paths)} files in batch")

        results = []

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self.processor.process_file, path): path
                for path in file_paths
            }

            # Process the results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    doc_info, summary_data = future.result()
                    results.append((doc_info, summary_data))
                    logger.info(f"Successfully processed {path}")
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")

        return results

    async def process_files_async(self, file_paths: List[str]) -> List[Tuple[DocumentInfo, Dict[str, Any]]]:
        """
        Process multiple files asynchronously.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of (DocumentInfo, metadata) tuples
        """
        logger.info(f"Processing {len(file_paths)} files asynchronously")

        # Create a thread pool executor
        loop = asyncio.get_event_loop()

        # Process the files in parallel
        tasks = []
        for path in file_paths:
            # Create a task for each file
            task = loop.run_in_executor(
                None, self.processor.process_file, path)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {file_paths[i]}: {result}")
            else:
                valid_results.append(result)

        return valid_results
