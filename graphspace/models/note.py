from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import uuid
import json
import os
from datetime import datetime
import shutil


@dataclass
class FileAttachment:
    """File attachment model."""

    id: str = ""
    filename: str = ""
    file_path: str = ""
    file_type: str = ""
    size: int = 0
    created_at: str = ""
    title: str = ""
    summary: str = ""
    extracted_text: str = ""

    def __post_init__(self):
        """Set default ID and timestamp if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())

        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileAttachment':
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class Note:
    """Note model with support for file attachments."""

    id: str = ""
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    attachments: List[FileAttachment] = field(default_factory=list)
    source: str = ""  # 'manual', 'upload', 'import'

    def __post_init__(self):
        """Set default ID and timestamps if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())

        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert attachments to dictionaries
        result["attachments"] = [attachment.to_dict()
                                 for attachment in self.attachments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        """Create from dictionary representation."""
        # Handle attachment objects
        if "attachments" in data:
            attachments = [FileAttachment.from_dict(
                attachment) for attachment in data["attachments"]]
            data_copy = data.copy()
            data_copy["attachments"] = attachments
            return cls(**data_copy)
        return cls(**data)

    def add_attachment(self, attachment: FileAttachment) -> None:
        """Add an attachment to the note."""
        self.attachments.append(attachment)
        self.updated_at = datetime.now().isoformat()

    def remove_attachment(self, attachment_id: str) -> bool:
        """
        Remove an attachment from the note.

        Args:
            attachment_id: ID of the attachment to remove

        Returns:
            True if the attachment was removed, False otherwise
        """
        for i, attachment in enumerate(self.attachments):
            if attachment.id == attachment_id:
                del self.attachments[i]
                self.updated_at = datetime.now().isoformat()
                return True
        return False

    def get_attachment(self, attachment_id: str) -> Optional[FileAttachment]:
        """
        Get an attachment by ID.

        Args:
            attachment_id: ID of the attachment to get

        Returns:
            The attachment or None if not found
        """
        for attachment in self.attachments:
            if attachment.id == attachment_id:
                return attachment
        return None


class NoteManager:
    """Manages notes and their persistence."""

    def __init__(self, storage_path: str = None, attachments_dir: str = None):
        """
        Initialize note manager.

        Args:
            storage_path: Path to the notes file
            attachments_dir: Directory to store attachments
        """
        if storage_path is None:
            # Use default path
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            storage_path = os.path.join(base_dir, "data", "notes.json")

        if attachments_dir is None:
            # Use default path
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            attachments_dir = os.path.join(base_dir, "data", "attachments")

        self.storage_path = storage_path
        self.attachments_dir = attachments_dir
        self.notes: Dict[str, Note] = {}

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        os.makedirs(attachments_dir, exist_ok=True)

        self.load_notes()

    def load_notes(self):
        """Load notes from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)

                    # Convert dictionaries to Note objects
                    self.notes = {
                        note_id: Note.from_dict(note_data)
                        for note_id, note_data in notes_data.items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading notes: {e}")
                self.notes = {}
        else:
            self.notes = {}

    def save_notes(self):
        """Save notes to storage."""
        # Convert Note objects to dictionaries
        notes_data = {
            note_id: note.to_dict()
            for note_id, note in self.notes.items()
        }

        # Save to file
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(notes_data, f, indent=2)

    def get_all_notes(self) -> List[Note]:
        """Get all notes."""
        return list(self.notes.values())

    def get_note(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        return self.notes.get(note_id)

    def add_note(self, note: Note) -> Note:
        """Add a new note."""
        if not note.id:
            note.id = str(uuid.uuid4())

        self.notes[note.id] = note
        self.save_notes()
        return note

    def update_note(self, note: Note) -> Note:
        """Update an existing note."""
        if note.id not in self.notes:
            raise ValueError(f"Note with ID {note.id} not found")

        note.updated_at = datetime.now().isoformat()
        self.notes[note.id] = note
        self.save_notes()
        return note

    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note and its attachments.

        Args:
            note_id: ID of the note to delete

        Returns:
            True if the note was deleted, False otherwise
        """
        if note_id not in self.notes:
            return False

        # Get the note to delete its attachments
        note = self.notes[note_id]

        # Delete attachment files
        for attachment in note.attachments:
            self.delete_attachment_file(attachment)

        # Delete the note
        del self.notes[note_id]
        self.save_notes()
        return True

    def get_notes_by_tag(self, tag: str) -> List[Note]:
        """Get notes by tag."""
        return [note for note in self.notes.values() if tag in note.tags]

    def search_notes(self, query: str) -> List[Note]:
        """
        Search notes by query string.

        Args:
            query: Query string to search for

        Returns:
            List of matching notes
        """
        query = query.lower()
        matching_notes = []

        for note in self.notes.values():
            # Search in title and content
            if query in note.title.lower() or query in note.content.lower():
                matching_notes.append(note)
                continue

            # Search in attachment titles and summaries
            for attachment in note.attachments:
                if (query in attachment.title.lower() or
                    query in attachment.summary.lower() or
                        query in attachment.extracted_text.lower()):
                    matching_notes.append(note)
                    break

        return matching_notes

    def add_attachment_to_note(
        self,
        note_id: str,
        file_path: str,
        title: str = "",
        summary: str = "",
        extracted_text: str = ""
    ) -> Optional[FileAttachment]:
        """
        Add an attachment to a note.

        Args:
            note_id: ID of the note to add the attachment to
            file_path: Path to the file to attach
            title: Title of the attachment
            summary: Summary of the attachment
            extracted_text: Extracted text from the attachment

        Returns:
            The created attachment or None if the note was not found
        """
        note = self.get_note(note_id)
        if not note:
            return None

        # Create attachment directory for this note if it doesn't exist
        note_attachments_dir = os.path.join(self.attachments_dir, note_id)
        os.makedirs(note_attachments_dir, exist_ok=True)

        # Get file info
        filename = os.path.basename(file_path)
        file_type = os.path.splitext(filename)[1].lstrip('.').lower()
        size = os.path.getsize(file_path)

        # Generate a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        dest_path = os.path.join(note_attachments_dir, unique_filename)

        # Copy the file to the attachments directory
        shutil.copy2(file_path, dest_path)

        # Create the attachment
        attachment = FileAttachment(
            filename=filename,
            file_path=dest_path,
            file_type=file_type,
            size=size,
            title=title or filename,
            summary=summary,
            extracted_text=extracted_text
        )

        # Add the attachment to the note
        note.add_attachment(attachment)

        # Save the note
        self.update_note(note)

        return attachment

    def remove_attachment_from_note(self, note_id: str, attachment_id: str) -> bool:
        """
        Remove an attachment from a note.

        Args:
            note_id: ID of the note to remove the attachment from
            attachment_id: ID of the attachment to remove

        Returns:
            True if the attachment was removed, False otherwise
        """
        note = self.get_note(note_id)
        if not note:
            return False

        # Find the attachment
        attachment = note.get_attachment(attachment_id)
        if not attachment:
            return False

        # Delete the attachment file
        self.delete_attachment_file(attachment)

        # Remove the attachment from the note
        if note.remove_attachment(attachment_id):
            # Save the note
            self.update_note(note)
            return True

        return False

    def delete_attachment_file(self, attachment: FileAttachment) -> bool:
        """
        Delete an attachment file.

        Args:
            attachment: Attachment to delete

        Returns:
            True if the file was deleted, False otherwise
        """
        if os.path.exists(attachment.file_path):
            try:
                os.remove(attachment.file_path)
                return True
            except OSError:
                return False
        return False

    def get_attachment_file_path(self, note_id: str, attachment_id: str) -> Optional[str]:
        """
        Get the file path for an attachment.

        Args:
            note_id: ID of the note containing the attachment
            attachment_id: ID of the attachment

        Returns:
            The file path or None if not found
        """
        note = self.get_note(note_id)
        if not note:
            return None

        attachment = note.get_attachment(attachment_id)
        if not attachment:
            return None

        return attachment.file_path
