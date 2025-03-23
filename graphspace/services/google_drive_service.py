import os
import io
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


class GoogleDriveService:
    """
    Google Drive integration service that provides access to files and folders
    without duplicating them locally. Supports real-time synchronization.
    """

    def __init__(
        self,
        credentials_file: str = "client_secrets.json",
        token_file: str = "token.pickle",
        scopes: List[str] = None,
        cache_dir: str = "data/gdrive_cache",
        cache_expiry: int = 3600  # Cache expiry in seconds (1 hour)
    ):
        """
        Initialize the Google Drive service.

        Args:
            credentials_file: Path to the client secrets JSON file
            token_file: Path to save/load the auth token
            scopes: OAuth scopes to request
            cache_dir: Directory to store metadata cache
            cache_expiry: Cache expiry time in seconds
        """
        # Default scopes if not provided
        if scopes is None:
            self.scopes = [
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/drive.metadata.readonly'
            ]
        else:
            self.scopes = scopes

        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.credentials = None

        # Setup cache
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)

        # File/folder metadata cache
        self.metadata_cache = {}
        self.metadata_cache_timestamp = {}

        # List of change tokens for real-time sync
        self.last_change_token = None

        # Load cached metadata if available
        self._load_metadata_cache()

    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API.

        Returns:
            True if authentication was successful, False otherwise
        """
        creds = None

        # Check if token file exists
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)

        # If credentials don't exist or are invalid, refresh or get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # If credentials file doesn't exist, return False
                if not os.path.exists(self.credentials_file):
                    print(
                        f"Credentials file {self.credentials_file} not found.")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

        # Build the service
        try:
            self.service = build('drive', 'v3', credentials=creds)
            self.credentials = creds

            # Get the current change token for future updates
            self._update_change_token()

            return True
        except Exception as e:
            print(f"Error building Drive service: {e}")
            return False

    def _update_change_token(self):
        """Get the latest change token for tracking changes."""
        try:
            results = self.service.changes().getStartPageToken().execute()
            self.last_change_token = results.get('startPageToken')
        except HttpError as error:
            print(f"Error getting change token: {error}")

    def _load_metadata_cache(self):
        """Load metadata cache from disk."""
        cache_file = os.path.join(self.cache_dir, "metadata_cache.json")
        timestamp_file = os.path.join(self.cache_dir, "cache_timestamp.json")

        if os.path.exists(cache_file) and os.path.exists(timestamp_file):
            try:
                with open(cache_file, 'r') as f:
                    self.metadata_cache = json.load(f)
                with open(timestamp_file, 'r') as f:
                    self.metadata_cache_timestamp = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.metadata_cache = {}
                self.metadata_cache_timestamp = {}

    def _save_metadata_cache(self):
        """Save metadata cache to disk."""
        cache_file = os.path.join(self.cache_dir, "metadata_cache.json")
        timestamp_file = os.path.join(self.cache_dir, "cache_timestamp.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(self.metadata_cache, f)
            with open(timestamp_file, 'w') as f:
                json.dump(self.metadata_cache_timestamp, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _is_cache_valid(self, file_id: str) -> bool:
        """Check if cache for a file/folder is still valid."""
        if file_id not in self.metadata_cache_timestamp:
            return False

        timestamp = self.metadata_cache_timestamp.get(file_id, 0)
        current_time = time.time()

        return (current_time - timestamp) < self.cache_expiry

    def list_files(
        self,
        folder_id: str = "root",
        file_types: List[str] = None,
        recursive: bool = False,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List files and folders in the specified folder.

        Args:
            folder_id: ID of the folder to list
            file_types: Optional list of MIME types to filter by
            recursive: Whether to list files recursively
            use_cache: Whether to use cached results if available

        Returns:
            List of file/folder metadata
        """
        if not self.service:
            if not self.authenticate():
                return []

        # Check cache first if enabled
        cache_key = f"folder_{folder_id}"
        if use_cache and cache_key in self.metadata_cache and self._is_cache_valid(cache_key):
            files = self.metadata_cache[cache_key]

            # Apply MIME type filter if specified
            if file_types:
                files = [f for f in files if f.get('mimeType') in file_types]

            return files

        # Prepare the query
        query = f"'{folder_id}' in parents and trashed=false"

        try:
            # Execute the query
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink)',
                pageSize=1000
            ).execute()

            files = results.get('files', [])

            # Cache the results
            self.metadata_cache[cache_key] = files
            self.metadata_cache_timestamp[cache_key] = time.time()
            self._save_metadata_cache()

            # If recursive, get files from subfolders
            if recursive:
                # Find folders
                folders = [f for f in files if f.get(
                    'mimeType') == 'application/vnd.google-apps.folder']

                # Recursively get files from each folder
                for folder in folders:
                    subfiles = self.list_files(
                        folder_id=folder['id'],
                        file_types=file_types,
                        recursive=True,
                        use_cache=use_cache
                    )
                    files.extend(subfiles)

            # Apply MIME type filter if specified
            if file_types:
                files = [f for f in files if f.get('mimeType') in file_types]

            return files

        except HttpError as error:
            print(f"Error listing files: {error}")
            return []

    def get_file_metadata(
        self,
        file_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file.

        Args:
            file_id: ID of the file
            use_cache: Whether to use cached metadata if available

        Returns:
            File metadata or None if not found
        """
        if not self.service:
            if not self.authenticate():
                return None

        # Check cache first if enabled
        if use_cache and file_id in self.metadata_cache and self._is_cache_valid(file_id):
            return self.metadata_cache[file_id]

        try:
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink'
            ).execute()

            # Cache the metadata
            self.metadata_cache[file_id] = file_metadata
            self.metadata_cache_timestamp[file_id] = time.time()
            self._save_metadata_cache()

            return file_metadata

        except HttpError as error:
            print(f"Error getting file metadata: {error}")
            return None

    def stream_file_content(
        self,
        file_id: str,
        chunk_size: int = 1024 * 1024
    ) -> Optional[io.BytesIO]:
        """
        Stream a file's content without saving it locally.

        Args:
            file_id: ID of the file to stream
            chunk_size: Size of chunks to download

        Returns:
            BytesIO object containing the file content, or None if error
        """
        if not self.service:
            if not self.authenticate():
                return None

        try:
            # Get the file metadata to check if it's a Google Docs file
            file_metadata = self.get_file_metadata(file_id, use_cache=True)
            if not file_metadata:
                return None

            mime_type = file_metadata.get('mimeType', '')

            # For Google Docs, Sheets, etc., export as PDF
            if mime_type.startswith('application/vnd.google-apps'):
                if mime_type == 'application/vnd.google-apps.document':
                    export_mime = 'application/pdf'
                elif mime_type == 'application/vnd.google-apps.spreadsheet':
                    export_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                elif mime_type == 'application/vnd.google-apps.presentation':
                    export_mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                else:
                    export_mime = 'application/pdf'

                response = self.service.files().export(
                    fileId=file_id,
                    mimeType=export_mime
                )
            else:
                # For regular files, download directly
                response = self.service.files().get_media(fileId=file_id)

            # Stream the content
            content = io.BytesIO()
            downloader = MediaIoBaseDownload(
                content, response, chunksize=chunk_size)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            content.seek(0)
            return content

        except HttpError as error:
            print(f"Error streaming file: {error}")
            return None

    def get_file_content_text(
        self,
        file_id: str
    ) -> Optional[str]:
        """
        Get the textual content of a file.

        Args:
            file_id: ID of the file

        Returns:
            Text content of the file, or None if not possible
        """
        # Get file metadata to determine type
        metadata = self.get_file_metadata(file_id)
        if not metadata:
            return None

        # Get file content
        content = self.stream_file_content(file_id)
        if not content:
            return None

        mime_type = metadata.get('mimeType', '')

        # Handle text files directly
        if mime_type in ['text/plain', 'text/markdown', 'text/csv']:
            try:
                return content.read().decode('utf-8')
            except UnicodeDecodeError:
                # Try another common encoding
                content.seek(0)
                return content.read().decode('latin-1')

        # Handle PDF files
        elif mime_type == 'application/pdf' or mime_type == 'application/vnd.google-apps.document':
            try:
                # Import here to avoid dependency if not used
                from PyPDF2 import PdfReader

                reader = PdfReader(content)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e:
                print(f"Error extracting PDF text: {e}")
                return None

        # Handle DOCX files
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                           'application/vnd.google-apps.document']:
            try:
                # Import here to avoid dependency if not used
                import docx

                doc = docx.Document(content)
                text = "\n".join([p.text for p in doc.paragraphs])
                return text
            except Exception as e:
                print(f"Error extracting DOCX text: {e}")
                return None

        # Handle spreadsheets
        elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           'application/vnd.google-apps.spreadsheet']:
            try:
                # Import here to avoid dependency if not used
                import pandas as pd

                df = pd.read_excel(content)
                return df.to_string()
            except Exception as e:
                print(f"Error extracting Excel text: {e}")
                return None

        # Handle other file types
        else:
            print(f"Unsupported file type for text extraction: {mime_type}")
            return None

    def search_files(
        self,
        query: str,
        file_types: List[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for files and folders by name or content.

        Args:
            query: Search query
            file_types: Optional list of MIME types to filter by
            max_results: Maximum number of results to return

        Returns:
            List of file/folder metadata
        """
        if not self.service:
            if not self.authenticate():
                return []

        # Prepare the search query
        search_query = f"name contains '{query}' and trashed=false"

        # Add file type filter if specified
        if file_types:
            mime_type_conditions = " or ".join(
                [f"mimeType='{mime}'" for mime in file_types])
            search_query += f" and ({mime_type_conditions})"

        try:
            # Execute the search
            results = self.service.files().list(
                q=search_query,
                spaces='drive',
                fields='files(id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink)',
                pageSize=max_results
            ).execute()

            files = results.get('files', [])
            return files

        except HttpError as error:
            print(f"Error searching files: {error}")
            return []

    def check_for_changes(self) -> List[Dict[str, Any]]:
        """
        Check for changes since the last check.

        Returns:
            List of changed files/folders metadata
        """
        if not self.service or not self.last_change_token:
            if not self.authenticate():
                return []

        try:
            changed_files = []
            page_token = self.last_change_token

            while page_token:
                response = self.service.changes().list(
                    pageToken=page_token,
                    spaces='drive',
                    fields='nextPageToken, newStartPageToken, changes(fileId, file(id, name, mimeType, createdTime, modifiedTime, size, parents, webViewLink), removed, time)'
                ).execute()

                # Process changes
                for change in response.get('changes', []):
                    # If the file was deleted or we don't have permission
                    if change.get('removed') or 'file' not in change:
                        # Remove from cache if present
                        file_id = change.get('fileId')
                        if file_id in self.metadata_cache:
                            del self.metadata_cache[file_id]
                            if file_id in self.metadata_cache_timestamp:
                                del self.metadata_cache_timestamp[file_id]
                    else:
                        # Update cache with new metadata
                        file_data = change.get('file', {})
                        file_id = file_data.get('id')

                        if file_id:
                            self.metadata_cache[file_id] = file_data
                            self.metadata_cache_timestamp[file_id] = time.time(
                            )

                            # Add to changed files list
                            changed_files.append(file_data)

                            # If it's a folder, invalidate its listing cache
                            if file_data.get('mimeType') == 'application/vnd.google-apps.folder':
                                folder_cache_key = f"folder_{file_id}"
                                if folder_cache_key in self.metadata_cache:
                                    del self.metadata_cache[folder_cache_key]
                                    if folder_cache_key in self.metadata_cache_timestamp:
                                        del self.metadata_cache_timestamp[folder_cache_key]

                # Update tokens for next iteration
                if 'nextPageToken' in response:
                    page_token = response['nextPageToken']
                else:
                    # Save the new start token for the next polling interval
                    if 'newStartPageToken' in response:
                        self.last_change_token = response['newStartPageToken']
                    page_token = None

            # Save updated cache
            self._save_metadata_cache()

            return changed_files

        except HttpError as error:
            print(f"Error checking for changes: {error}")
            return []

    def convert_to_note(
        self,
        file_id: str,
        llm_module=None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a Google Drive file to a note.

        Args:
            file_id: ID of the file to convert
            llm_module: Advanced LLM module for summarization (optional)

        Returns:
            Note dictionary or None if conversion failed
        """
        # Get file metadata
        metadata = self.get_file_metadata(file_id)
        if not metadata:
            return None

        # Get file content as text
        content = self.get_file_content_text(file_id)
        if not content:
            # If we couldn't extract text
            content = f"This file could not be converted to text. View it at {metadata.get('webViewLink', '')}."

        # Create basic note
        note = {
            "id": str(int(time.time())),
            "title": metadata.get('name', 'Untitled'),
            "content": content[:100000],  # Limit content size
            "tags": ["google_drive"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "source": {
                "type": "google_drive",
                "file_id": file_id,
                "mime_type": metadata.get('mimeType', ''),
                "web_link": metadata.get('webViewLink', ''),
                "metadata": metadata
            }
        }

        # Use LLM to enhance the note if available
        if llm_module:
            try:
                # Generate a better title
                new_title = llm_module.generate_title(content[:10000])
                if new_title and len(new_title) > 5:
                    note["title"] = new_title

                # Generate summary if content is long
                if len(content) > 1000:
                    summary = llm_module.summarize_document(
                        {"content": content, "title": note["title"]})
                    if summary and len(summary) > 100:
                        note["summary"] = summary

                # Extract tags
                tags = llm_module.extract_tags(content[:10000])
                if tags:
                    note["tags"].extend(tags)
            except Exception as e:
                print(f"Error enhancing note with LLM: {e}")

        return note

    def convert_folder_to_notes(
        self,
        folder_id: str,
        recursive: bool = True,
        file_types: List[str] = None,
        llm_module=None
    ) -> List[Dict[str, Any]]:
        """
        Convert all files in a folder to notes.

        Args:
            folder_id: ID of the folder
            recursive: Whether to process subfolders
            file_types: Optional list of MIME types to filter by
            llm_module: Advanced LLM module for summarization (optional)

        Returns:
            List of note dictionaries
        """
        # Default file types to convert if not specified
        if file_types is None:
            file_types = [
                'text/plain',
                'text/markdown',
                'application/pdf',
                'application/vnd.google-apps.document',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.google-apps.spreadsheet',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ]

        # List files in the folder
        files = self.list_files(
            folder_id=folder_id,
            file_types=file_types,
            recursive=recursive
        )

        notes = []

        # Convert each file to a note
        for file in files:
            note = self.convert_to_note(file['id'], llm_module)
            if note:
                notes.append(note)

        return notes

    def setup_webhook(self, webhook_url: str) -> bool:
        """
        Set up a webhook for real-time notifications of changes.

        Args:
            webhook_url: URL to send notifications to

        Returns:
            True if setup was successful, False otherwise
        """
        # Note: Google Drive API doesn't have native webhook support
        # This method would typically use Google Cloud Pub/Sub instead
        # For now, we'll use polling as implemented in check_for_changes()
        print("Webhook setup not implemented, using polling mechanism instead.")
        return False
