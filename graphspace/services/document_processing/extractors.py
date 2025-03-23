import os
import io
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, BinaryIO, TextIO, Tuple

# Import format-specific libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class DocumentInfo:
    """Holds extracted information from a document."""

    def __init__(
        self,
        title: str = "",
        content: str = "",
        metadata: Dict[str, Any] = None,
        file_path: str = "",
        file_type: str = "",
        pages: int = 0,
        detected_language: str = "en"
    ):
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.file_path = file_path
        self.file_type = file_type
        self.pages = pages
        self.detected_language = detected_language

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "pages": self.pages,
            "detected_language": self.detected_language
        }

    def get_summary(self) -> str:
        """Get a summary of document information."""
        return (
            f"Document: {self.title}\n"
            f"Type: {self.file_type}\n"
            f"Pages: {self.pages}\n"
            f"Size: {len(self.content)} characters\n"
            f"Language: {self.detected_language}"
        )


class DocumentExtractor(ABC):
    """Base class for all document extractors."""

    @abstractmethod
    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """
        Extract text and metadata from a document.

        Args:
            file_obj: File-like object containing the document data
            file_path: Path to the original file

        Returns:
            DocumentInfo object containing extracted information
        """
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean up extracted text."""
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        # Remove leading and trailing whitespace
        return text.strip()

    @staticmethod
    def extract_title_from_text(text: str, default_title: str = "Untitled Document") -> str:
        """Try to extract a title from the text content."""
        # Try to find the first non-empty line
        if not text:
            return default_title

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # Assume title is not too long
                return line

        return default_title


class PDFExtractor(DocumentExtractor):
    """Extractor for PDF documents."""

    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """Extract text and metadata from a PDF document."""
        if not PYPDF2_AVAILABLE:
            raise ImportError(
                "PyPDF2 is required for PDF extraction but is not installed.")

        # Extract the filename from the path as a fallback title
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file_obj)

            # Extract metadata
            metadata = {}
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if key.startswith('/'):
                        clean_key = key[1:]  # Remove leading slash
                        metadata[clean_key] = value

            # Extract title from metadata or use filename
            title = metadata.get('Title', base_filename)

            # Extract text from all pages
            content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content += page.extract_text() + "\n\n"

            # Clean the text
            content = self.clean_text(content)

            # If no title in metadata, try to extract from content
            if title == base_filename:
                extracted_title = self.extract_title_from_text(content)
                if extracted_title != "Untitled Document":
                    title = extracted_title

            return DocumentInfo(
                title=title,
                content=content,
                metadata=metadata,
                file_path=file_path,
                file_type="pdf",
                pages=len(pdf_reader.pages)
            )
        except Exception as e:
            # If extraction fails, return a basic DocumentInfo
            return DocumentInfo(
                title=base_filename,
                content=f"PDF extraction failed: {str(e)}",
                file_path=file_path,
                file_type="pdf",
                pages=0
            )


class DocxExtractor(DocumentExtractor):
    """Extractor for Microsoft Word (.docx) documents."""

    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """Extract text and metadata from a DOCX document."""
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for DOCX extraction but is not installed.")

        # Extract the filename from the path as a fallback title
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Save the file temporarily because python-docx needs a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(file_obj.read())
                temp_path = temp_file.name

            # Open the temporary file with python-docx
            doc = docx.Document(temp_path)

            # Clean up the temporary file
            os.unlink(temp_path)

            # Extract metadata
            metadata = {
                "author": doc.core_properties.author or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
                "last_modified_by": doc.core_properties.last_modified_by or "",
                "modified": str(doc.core_properties.modified) if doc.core_properties.modified else "",
                "title": doc.core_properties.title or "",
                "subject": doc.core_properties.subject or "",
                "keywords": doc.core_properties.keywords or "",
                "category": doc.core_properties.category or "",
                "comments": doc.core_properties.comments or ""
            }

            # Extract title from metadata or use filename
            title = metadata.get('title') or base_filename

            # Extract text from all paragraphs
            content = "\n".join([para.text for para in doc.paragraphs])

            # Count pages (approximate)
            # DOCX doesn't have fixed pages like PDF, so this is an approximation
            # Assuming ~3000 characters per page
            pages = max(1, len(content) // 3000)

            # Clean the text
            content = self.clean_text(content)

            # If no title in metadata, try to extract from content
            if not title or title == base_filename:
                extracted_title = self.extract_title_from_text(content)
                if extracted_title != "Untitled Document":
                    title = extracted_title

            return DocumentInfo(
                title=title,
                content=content,
                metadata=metadata,
                file_path=file_path,
                file_type="docx",
                pages=pages
            )
        except Exception as e:
            # If extraction fails, return a basic DocumentInfo
            return DocumentInfo(
                title=base_filename,
                content=f"DOCX extraction failed: {str(e)}",
                file_path=file_path,
                file_type="docx",
                pages=0
            )


class TxtExtractor(DocumentExtractor):
    """Extractor for plain text (.txt) files."""

    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """Extract text from a plain text file."""
        # Extract the filename from the path as a fallback title
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Read the file content
            content = file_obj.read().decode('utf-8', errors='replace')

            # Clean the text
            content = self.clean_text(content)

            # Try to extract title from content
            title = self.extract_title_from_text(content, base_filename)

            # Count pages (approximate)
            # Assuming ~3000 characters per page
            pages = max(1, len(content) // 3000)

            return DocumentInfo(
                title=title,
                content=content,
                metadata={"encoding": "utf-8"},
                file_path=file_path,
                file_type="txt",
                pages=pages
            )
        except Exception as e:
            # If extraction fails, return a basic DocumentInfo
            return DocumentInfo(
                title=base_filename,
                content=f"Text extraction failed: {str(e)}",
                file_path=file_path,
                file_type="txt",
                pages=0
            )


class MarkdownExtractor(DocumentExtractor):
    """Extractor for Markdown (.md) files."""

    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """Extract text from a Markdown file."""
        # Extract the filename from the path as a fallback title
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Read the file content
            content = file_obj.read().decode('utf-8', errors='replace')

            # Try to extract headers (usually the first # header is the title)
            title = base_filename
            headers = re.findall(r'^#\s+(.+)$', content, re.MULTILINE)
            if headers:
                title = headers[0]

            # Convert markdown to plain text for better processing if the library is available
            plain_content = content
            if MARKDOWN_AVAILABLE:
                html_content = markdown.markdown(content)
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    plain_content = soup.get_text("\n")

            # Clean the text
            content = self.clean_text(content)
            plain_content = self.clean_text(plain_content)

            # Count pages (approximate)
            # Assuming ~3000 characters per page
            pages = max(1, len(plain_content) // 3000)

            return DocumentInfo(
                title=title,
                content=plain_content,
                metadata={
                    "raw_markdown": content,
                    "headers": headers
                },
                file_path=file_path,
                file_type="md",
                pages=pages
            )
        except Exception as e:
            # If extraction fails, return a basic DocumentInfo
            return DocumentInfo(
                title=base_filename,
                content=f"Markdown extraction failed: {str(e)}",
                file_path=file_path,
                file_type="md",
                pages=0
            )


class HtmlExtractor(DocumentExtractor):
    """Extractor for HTML (.html) files."""

    def extract(self, file_obj: BinaryIO, file_path: str) -> DocumentInfo:
        """Extract text from an HTML file."""
        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup is required for HTML extraction but is not installed.")

        # Extract the filename from the path as a fallback title
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]

        try:
            # Read the file content
            content = file_obj.read().decode('utf-8', errors='replace')

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Extract title
            title = base_filename
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            # Extract metadata from meta tags
            metadata = {}
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    metadata[meta['name']] = meta['content']

            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Extract text
            plain_content = soup.get_text("\n")

            # Clean the text
            plain_content = self.clean_text(plain_content)

            # Count pages (approximate)
            # Assuming ~3000 characters per page
            pages = max(1, len(plain_content) // 3000)

            return DocumentInfo(
                title=title,
                content=plain_content,
                metadata=metadata,
                file_path=file_path,
                file_type="html",
                pages=pages
            )
        except Exception as e:
            # If extraction fails, return a basic DocumentInfo
            return DocumentInfo(
                title=base_filename,
                content=f"HTML extraction failed: {str(e)}",
                file_path=file_path,
                file_type="html",
                pages=0
            )


class ExtractorFactory:
    """Factory for creating document extractors based on file extension."""

    @staticmethod
    def get_extractor(file_extension: str) -> DocumentExtractor:
        """Get the appropriate extractor for the given file extension."""
        file_extension = file_extension.lower().lstrip('.')

        if file_extension == 'pdf':
            return PDFExtractor()
        elif file_extension == 'docx':
            return DocxExtractor()
        elif file_extension == 'txt':
            return TxtExtractor()
        elif file_extension == 'md':
            return MarkdownExtractor()
        elif file_extension in ('html', 'htm'):
            return HtmlExtractor()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    @staticmethod
    def extract_from_file(file_path: str) -> DocumentInfo:
        """Extract document information from a file on disk."""
        _, file_extension = os.path.splitext(file_path)
        extractor = ExtractorFactory.get_extractor(file_extension)

        with open(file_path, 'rb') as file_obj:
            return extractor.extract(file_obj, file_path)

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, file_name: str) -> DocumentInfo:
        """Extract document information from a bytes object."""
        _, file_extension = os.path.splitext(file_name)
        extractor = ExtractorFactory.get_extractor(file_extension)

        file_obj = io.BytesIO(file_bytes)
        return extractor.extract(file_obj, file_name)
