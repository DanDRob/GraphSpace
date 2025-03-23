import os
import glob
import time
import filetype
import concurrent.futures
import threading
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from tqdm import tqdm
import traceback
from datetime import datetime
import json

# Import file type handlers
from PyPDF2 import PdfReader
import docx
import pandas as pd
import markdown
from bs4 import BeautifulSoup


class DocumentProcessor:
    """
    Document processor that handles batch document processing,
    including multiple files and folder uploads with parallel processing.
    """

    def __init__(
        self,
        upload_dir: str = "data/uploads",
        max_workers: int = 4,
        llm_module=None,
        embedding_module=None,
        file_extensions: List[str] = None,
        chunk_size: int = 10
    ):
        """
        Initialize the document processor.

        Args:
            upload_dir: Directory to store uploaded files
            max_workers: Maximum number of parallel workers
            llm_module: Advanced LLM module for document enhancement
            embedding_module: EmbeddingModule for vectorizing content
            file_extensions: List of allowed file extensions
            chunk_size: Number of documents to process at once
        """
        self.upload_dir = upload_dir
        self.max_workers = max_workers
        self.llm_module = llm_module
        self.embedding_module = embedding_module
        self.chunk_size = chunk_size

        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)

        # Default allowed file extensions if not provided
        if file_extensions is None:
            self.file_extensions = [
                ".txt", ".md", ".pdf", ".docx", ".doc",
                ".xlsx", ".xls", ".csv", ".html", ".htm"
            ]
        else:
            self.file_extensions = file_extensions

        # Cache for processed file metadata
        self.processed_files = {}

        # Lock for thread safety
        self.lock = threading.Lock()

    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported for processing.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is supported, False otherwise
        """
        # Check if the file exists
        if not os.path.isfile(file_path):
            return False

        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.file_extensions:
            return False

        # Check if it's a binary file
        try:
            kind = filetype.guess(file_path)
            if kind is None:  # Text files return None
                return ext in ['.txt', '.md', '.csv']

            # Check for supported MIME types
            supported_mimes = [
                'text/plain', 'text/markdown', 'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-excel',
                'text/csv',
                'text/html'
            ]
            return kind.mime in supported_mimes
        except:
            # If we can't determine MIME type, check by extension
            return ext in self.file_extensions

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            return ""

        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            # Text files
            if file_ext in ['.txt']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()

            # Markdown files
            elif file_ext in ['.md']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    md_text = f.read()
                    # Convert markdown to HTML and then to plain text
                    html = markdown.markdown(md_text)
                    soup = BeautifulSoup(html, 'html.parser')
                    return soup.get_text()

            # PDF files
            elif file_ext in ['.pdf']:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text

            # Word documents
            elif file_ext in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])

            # Excel files
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                return df.to_string()

            # CSV files
            elif file_ext in ['.csv']:
                df = pd.read_csv(file_path)
                return df.to_string()

            # HTML files
            elif file_ext in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    html_text = f.read()
                    soup = BeautifulSoup(html_text, 'html.parser')
                    return soup.get_text()

            # Unsupported file type
            else:
                return f"Unsupported file type: {file_ext}"

        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            traceback.print_exc()
            return f"Error processing file: {str(e)}"

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and extract relevant information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with extracted information
        """
        if not self.is_supported_file(file_path):
            return {
                "success": False,
                "error": "Unsupported file type",
                "file_path": file_path
            }

        try:
            # Extract basic metadata
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path)

            # Extract text content
            content = self.extract_text_from_file(file_path)

            # Create basic document
            doc = {
                "id": str(int(time.time())) + "_" + file_name.replace(".", "_"),
                "title": file_name,
                "content": content,
                "file_path": file_path,
                "file_size": file_size,
                "file_type": file_ext[1:] if file_ext.startswith('.') else file_ext,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "tags": []
            }

            # Use LLM to enhance the document if available
            if self.llm_module and content and len(content) > 100:
                try:
                    # Generate a better title
                    new_title = self.llm_module.generate_title(content[:10000])
                    if new_title and len(new_title) > 5:
                        doc["title"] = new_title

                    # Generate summary if content is long
                    if len(content) > 1000:
                        summary = self.llm_module.summarize_document(
                            {"content": content, "title": doc["title"]})
                        if summary and len(summary) > 100:
                            doc["summary"] = summary

                    # Extract tags
                    tags = self.llm_module.extract_tags(content[:10000])
                    if tags:
                        doc["tags"] = tags
                except Exception as e:
                    print(f"Error enhancing document with LLM: {e}")

            # Use embedding module to create embeddings if available
            if self.embedding_module and content:
                try:
                    # Create and store document chunks
                    chunks = self.embedding_module.chunk_document(doc)
                    self.embedding_module.store_chunks(chunks)
                    doc["num_chunks"] = len(chunks)
                except Exception as e:
                    print(f"Error creating embeddings: {e}")

            return {
                "success": True,
                "document": doc,
                "file_path": file_path
            }

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def process_files_in_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            callback: Optional callback function to call after each file is processed

        Returns:
            List of processed document results
        """
        if not os.path.isdir(directory_path):
            return [{
                "success": False,
                "error": f"Directory does not exist: {directory_path}"
            }]

        # Find all files in the directory
        pattern = os.path.join(directory_path, "**" if recursive else "", "*")
        all_files = glob.glob(pattern, recursive=recursive)

        # Filter for supported files
        supported_files = [f for f in all_files if os.path.isfile(
            f) and self.is_supported_file(f)]

        if not supported_files:
            return [{
                "success": False,
                "error": f"No supported files found in directory: {directory_path}"
            }]

        # Process files in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in supported_files
            }

            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                               total=len(supported_files),
                               desc="Processing files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()

                    # Store result
                    with self.lock:
                        results.append(result)
                        self.processed_files[file_path] = result

                    # Call callback if provided
                    if callback and callable(callback):
                        callback(result)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    traceback.print_exc()
                    result = {
                        "success": False,
                        "error": str(e),
                        "file_path": file_path
                    }
                    results.append(result)

        return results

    def process_uploaded_files(
        self,
        file_paths: List[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a list of uploaded files.

        Args:
            file_paths: List of paths to uploaded files
            callback: Optional callback function to call after each file is processed

        Returns:
            List of processed document results
        """
        # Filter for existing and supported files
        supported_files = [f for f in file_paths if os.path.isfile(
            f) and self.is_supported_file(f)]

        if not supported_files:
            return [{
                "success": False,
                "error": "No valid files to process"
            }]

        # Process in chunks to avoid memory issues with large batches
        results = []
        for i in range(0, len(supported_files), self.chunk_size):
            chunk = supported_files[i:i+self.chunk_size]

            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path
                    for file_path in chunk
                }

                # Process as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_file),
                                   total=len(chunk),
                                   desc=f"Processing chunk {i//self.chunk_size + 1}/{(len(supported_files)-1)//self.chunk_size + 1}"):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()

                        # Store result
                        with self.lock:
                            results.append(result)
                            self.processed_files[file_path] = result

                        # Call callback if provided
                        if callback and callable(callback):
                            callback(result)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        traceback.print_exc()
                        result = {
                            "success": False,
                            "error": str(e),
                            "file_path": file_path
                        }
                        results.append(result)

        return results

    def extract_document_relationships(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships between documents based on content similarity.

        Args:
            documents: List of document dictionaries

        Returns:
            Dictionary mapping document IDs to lists of related documents
        """
        # This requires embedding_module to be available
        if not self.embedding_module:
            return {}

        relationships = {}

        # For each document, find related documents
        for doc in documents:
            if not doc.get("success", False) or "document" not in doc:
                continue

            document = doc["document"]
            doc_id = document["id"]

            # Skip if no content
            if not document.get("content"):
                relationships[doc_id] = []
                continue

            # Use the content as a query to find similar documents
            search_results = self.embedding_module.search(
                # Use first 5000 chars as query
                document.get("content", "")[:5000],
                level=0,  # Search at document level
                top_k=5    # Find 5 similar documents
            )

            # Filter out self-matches
            related_docs = []
            for result in search_results:
                source_id = result.get("source_id")
                if source_id and source_id != doc_id:
                    related_docs.append({
                        "id": source_id,
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {})
                    })

            relationships[doc_id] = related_docs

        return relationships

    def save_processed_files_metadata(self, output_path: str = "data/processed_files_metadata.json"):
        """
        Save metadata of processed files to a JSON file.

        Args:
            output_path: Path to save the metadata
        """
        # Prepare metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "num_files": len(self.processed_files),
            "files": {}
        }

        # Add file metadata
        for file_path, result in self.processed_files.items():
            if result.get("success", False) and "document" in result:
                doc = result["document"]
                metadata["files"][file_path] = {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "file_size": doc.get("file_size"),
                    "file_type": doc.get("file_type"),
                    "num_chunks": doc.get("num_chunks", 0),
                    "has_summary": "summary" in doc,
                    "num_tags": len(doc.get("tags", [])),
                    "tags": doc.get("tags", []),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at")
                }
            else:
                metadata["files"][file_path] = {
                    "error": result.get("error", "Unknown error"),
                    "success": False
                }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata
