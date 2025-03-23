from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import uuid
import json
import os


@dataclass
class ContactAddress:
    street: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    type: str = "home"  # home, work, other

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContactAddress':
        return cls(**data)


@dataclass
class Contact:
    id: str = ""
    name: str = ""
    # [{"value": "email@example.com", "type": "home"}]
    email: List[Dict[str, str]] = field(default_factory=list)
    # [{"value": "555-1234", "type": "mobile"}]
    phone: List[Dict[str, str]] = field(default_factory=list)
    organization: str = ""
    job_title: str = ""
    addresses: List[ContactAddress] = field(default_factory=list)
    photo_url: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    provider: str = ""  # google, microsoft, apple
    provider_id: str = ""  # ID from the provider
    updated_at: str = ""  # ISO timestamp
    created_at: str = ""  # ISO timestamp

    def __post_init__(self):
        """Set default ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Convert addresses to dictionaries
        result["addresses"] = [addr.to_dict() for addr in self.addresses]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contact':
        """Create from dictionary representation."""
        # Handle address objects
        if "addresses" in data:
            addresses = [ContactAddress.from_dict(
                addr) for addr in data["addresses"]]
            data_copy = data.copy()
            data_copy["addresses"] = addresses
            return cls(**data_copy)
        return cls(**data)

    def get_primary_email(self) -> str:
        """Get the primary email address."""
        if not self.email:
            return ""

        # Try to find an email marked as primary
        for email in self.email:
            if email.get("primary", False):
                return email["value"]

        # Otherwise return the first one
        return self.email[0]["value"]

    def get_primary_phone(self) -> str:
        """Get the primary phone number."""
        if not self.phone:
            return ""

        # Try to find a phone marked as primary
        for phone in self.phone:
            if phone.get("primary", False):
                return phone["value"]

        # Otherwise return the first one
        return self.phone[0]["value"]

    def get_display_name(self) -> str:
        """Get a display name for the contact."""
        if self.name:
            return self.name
        elif self.email:
            return self.get_primary_email()
        elif self.organization:
            return self.organization
        return "Unnamed Contact"


class ContactAdapter(ABC):
    """Abstract base class for contact adapters."""

    @abstractmethod
    def get_contacts(self) -> List[Contact]:
        """Get all contacts from the provider."""
        pass

    @abstractmethod
    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact with the provider."""
        pass

    @abstractmethod
    def update_contact(self, contact: Contact) -> Contact:
        """Update an existing contact with the provider."""
        pass

    @abstractmethod
    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from the provider."""
        pass


class LocalContactStorage:
    """Stores contacts locally."""

    def __init__(self, storage_path: str = None):
        """
        Initialize local contact storage.

        Args:
            storage_path: Path to the contacts file
        """
        if storage_path is None:
            # Use default path
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))
            storage_path = os.path.join(base_dir, "data", "contacts.json")

        self.storage_path = storage_path
        self.contacts: Dict[str, Contact] = {}
        self.load_contacts()

    def load_contacts(self):
        """Load contacts from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    contacts_data = json.load(f)

                    # Convert dictionaries to Contact objects
                    self.contacts = {
                        contact_id: Contact.from_dict(contact_data)
                        for contact_id, contact_data in contacts_data.items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading contacts: {e}")
                self.contacts = {}
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self.contacts = {}

    def save_contacts(self):
        """Save contacts to storage."""
        # Convert Contact objects to dictionaries
        contacts_data = {
            contact_id: contact.to_dict()
            for contact_id, contact in self.contacts.items()
        }

        # Save to file
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(contacts_data, f, indent=2)

    def get_all_contacts(self) -> List[Contact]:
        """Get all contacts."""
        return list(self.contacts.values())

    def get_contact(self, contact_id: str) -> Optional[Contact]:
        """Get a contact by ID."""
        return self.contacts.get(contact_id)

    def add_contact(self, contact: Contact) -> Contact:
        """Add a new contact."""
        if not contact.id:
            contact.id = str(uuid.uuid4())

        self.contacts[contact.id] = contact
        self.save_contacts()
        return contact

    def update_contact(self, contact: Contact) -> Contact:
        """Update an existing contact."""
        if contact.id not in self.contacts:
            raise ValueError(f"Contact with ID {contact.id} not found")

        self.contacts[contact.id] = contact
        self.save_contacts()
        return contact

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact."""
        if contact_id not in self.contacts:
            return False

        del self.contacts[contact_id]
        self.save_contacts()
        return True

    def find_contacts_by_email(self, email: str) -> List[Contact]:
        """Find contacts by email address."""
        matching_contacts = []

        for contact in self.contacts.values():
            for contact_email in contact.email:
                if contact_email["value"].lower() == email.lower():
                    matching_contacts.append(contact)
                    break

        return matching_contacts

    def find_contacts_by_name(self, name: str) -> List[Contact]:
        """Find contacts by name (partial match)."""
        name = name.lower()

        matching_contacts = []
        for contact in self.contacts.values():
            if name in contact.name.lower():
                matching_contacts.append(contact)

        return matching_contacts

    def find_contacts_by_tag(self, tag: str) -> List[Contact]:
        """Find contacts by tag."""
        matching_contacts = []

        for contact in self.contacts.values():
            if tag in contact.tags:
                matching_contacts.append(contact)

        return matching_contacts
