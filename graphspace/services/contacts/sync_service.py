import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime

from .base import Contact, ContactAdapter, LocalContactStorage
from .adapters import GoogleContactAdapter, MicrosoftContactAdapter, AppleContactAdapter
from graphspace.utils.auth_service import OAuth2Service, OAuth2TokenStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContactSyncResult:
    """Result of a contact synchronization operation."""

    def __init__(self):
        self.added = []
        self.updated = []
        self.deleted = []
        self.skipped = []
        self.errors = []
        self.total_processed = 0

    def add_result(self, contact: Contact, operation: str, error: Optional[Exception] = None):
        """Add a result for a contact."""
        if operation == "added":
            self.added.append(contact)
        elif operation == "updated":
            self.updated.append(contact)
        elif operation == "deleted":
            self.deleted.append(contact)
        elif operation == "skipped":
            self.skipped.append(contact)

        if error:
            self.errors.append((contact, error))

        self.total_processed += 1

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of the sync results."""
        return {
            "added": len(self.added),
            "updated": len(self.updated),
            "deleted": len(self.deleted),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "total_processed": self.total_processed
        }

    def __str__(self) -> str:
        """Get a string representation of the sync results."""
        summary = self.get_summary()
        return (
            f"Contact Sync Results:\n"
            f"  Added: {summary['added']}\n"
            f"  Updated: {summary['updated']}\n"
            f"  Deleted: {summary['deleted']}\n"
            f"  Skipped: {summary['skipped']}\n"
            f"  Errors: {summary['errors']}\n"
            f"  Total Processed: {summary['total_processed']}"
        )


class ContactSyncService:
    """
    Service for synchronizing contacts between providers and local storage.
    """

    def __init__(
        self,
        oauth_service: OAuth2Service,
        token_storage: OAuth2TokenStorage,
        local_storage: LocalContactStorage,
        user_id: str = "default"
    ):
        """
        Initialize the contact sync service.

        Args:
            oauth_service: OAuth2 service for authentication
            token_storage: Storage for OAuth2 tokens
            local_storage: Local contact storage
            user_id: User ID for OAuth2 tokens
        """
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.local_storage = local_storage
        self.user_id = user_id

        # Map of provider names to adapters
        self.adapters: Dict[str, ContactAdapter] = {}

        # Map of provider IDs to local IDs
        self.provider_to_local_id: Dict[str, Dict[str, str]] = {
            "google": {},
            "microsoft": {},
            "apple": {}
        }

        # Track last sync time for each provider
        self.last_sync_time: Dict[str, float] = {}

    def get_adapter(self, provider: str) -> ContactAdapter:
        """
        Get or create an adapter for the specified provider.

        Args:
            provider: Provider name ('google', 'microsoft', 'apple')

        Returns:
            Contact adapter for the provider
        """
        if provider not in self.adapters:
            if provider == "google":
                self.adapters[provider] = GoogleContactAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            elif provider == "microsoft":
                self.adapters[provider] = MicrosoftContactAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            elif provider == "apple":
                self.adapters[provider] = AppleContactAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            else:
                raise ValueError(f"Unsupported contact provider: {provider}")

        return self.adapters[provider]

    def build_id_mappings(self):
        """Build mappings of provider IDs to local IDs."""
        # Get all local contacts
        local_contacts = self.local_storage.get_all_contacts()

        # Create mappings by provider
        for provider in self.provider_to_local_id:
            self.provider_to_local_id[provider] = {}

        # Add mappings for each contact
        for contact in local_contacts:
            if contact.provider and contact.provider_id:
                provider = contact.provider
                provider_id = contact.provider_id
                local_id = contact.id

                if provider in self.provider_to_local_id:
                    self.provider_to_local_id[provider][provider_id] = local_id

    def sync_from_provider(
        self,
        provider: str,
        merge_strategy: str = "provider_wins"
    ) -> ContactSyncResult:
        """
        Sync contacts from a provider to local storage.

        Args:
            provider: Provider name ('google', 'microsoft', 'apple')
            merge_strategy: Strategy for merging conflicts ('provider_wins', 'local_wins', 'newer_wins')

        Returns:
            Sync result
        """
        logger.info(f"Syncing contacts from {provider} to local storage")

        # Initialize result
        result = ContactSyncResult()

        # Ensure we have a mapping of provider IDs to local IDs
        self.build_id_mappings()

        try:
            # Get adapter for the provider
            adapter = self.get_adapter(provider)

            # Get all contacts from the provider
            provider_contacts = adapter.get_contacts()
            logger.info(
                f"Retrieved {len(provider_contacts)} contacts from {provider}")

            # Process each contact
            for contact in provider_contacts:
                try:
                    # Check if we already have this contact locally
                    provider_id = contact.provider_id
                    if provider_id in self.provider_to_local_id.get(provider, {}):
                        # Get the local ID for this contact
                        local_id = self.provider_to_local_id[provider][provider_id]

                        # Get the local contact
                        local_contact = self.local_storage.get_contact(
                            local_id)

                        if local_contact:
                            # Check if we need to update
                            if self._should_update_local(local_contact, contact, merge_strategy):
                                # Update the local ID and save to local storage
                                contact.id = local_id
                                self.local_storage.update_contact(contact)
                                result.add_result(contact, "updated")
                            else:
                                # Skip this contact
                                result.add_result(contact, "skipped")
                        else:
                            # Local contact not found, add it
                            contact.id = local_id
                            self.local_storage.add_contact(contact)
                            result.add_result(contact, "added")
                    else:
                        # Add as a new contact
                        self.local_storage.add_contact(contact)

                        # Update the ID mapping
                        self.provider_to_local_id.setdefault(
                            provider, {})[provider_id] = contact.id

                        result.add_result(contact, "added")
                except Exception as e:
                    logger.error(f"Error processing contact {contact.id}: {e}")
                    result.add_result(contact, "error", e)

            # Update last sync time
            self.last_sync_time[provider] = time.time()

            return result
        except Exception as e:
            logger.error(f"Error syncing contacts from {provider}: {e}")
            raise

    def _should_update_local(
        self,
        local_contact: Contact,
        provider_contact: Contact,
        merge_strategy: str
    ) -> bool:
        """
        Determine if a local contact should be updated with provider data.

        Args:
            local_contact: Local contact
            provider_contact: Provider contact
            merge_strategy: Merge strategy

        Returns:
            True if the local contact should be updated
        """
        if merge_strategy == "provider_wins":
            return True
        elif merge_strategy == "local_wins":
            return False
        elif merge_strategy == "newer_wins":
            # Parse timestamps (if available)
            local_time = self._parse_timestamp(local_contact.updated_at)
            provider_time = self._parse_timestamp(provider_contact.updated_at)

            # If we have both timestamps, compare them
            if local_time and provider_time:
                return provider_time > local_time

            # If we only have one timestamp, use that
            return bool(provider_time)
        else:
            raise ValueError(f"Invalid merge strategy: {merge_strategy}")

    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        """
        Parse a timestamp string to a datetime object.

        Args:
            timestamp: Timestamp string

        Returns:
            Datetime object or None if parsing fails
        """
        if not timestamp:
            return None

        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def sync_to_provider(
        self,
        provider: str,
        merge_strategy: str = "local_wins"
    ) -> ContactSyncResult:
        """
        Sync contacts from local storage to a provider.

        Args:
            provider: Provider name ('google', 'microsoft', 'apple')
            merge_strategy: Strategy for merging conflicts ('provider_wins', 'local_wins', 'newer_wins')

        Returns:
            Sync result
        """
        logger.info(f"Syncing contacts from local storage to {provider}")

        # Initialize result
        result = ContactSyncResult()

        # Ensure we have a mapping of provider IDs to local IDs
        self.build_id_mappings()

        try:
            # Get adapter for the provider
            adapter = self.get_adapter(provider)

            # Get all local contacts
            local_contacts = self.local_storage.get_all_contacts()
            logger.info(
                f"Retrieved {len(local_contacts)} contacts from local storage")

            # Get all provider contacts (for checking existing contacts)
            provider_contacts = adapter.get_contacts()
            provider_contact_dict = {
                contact.provider_id: contact for contact in provider_contacts}

            # Create a map of local IDs to provider IDs
            local_to_provider_id = {
                local_id: provider_id
                for provider_id, local_id in self.provider_to_local_id.get(provider, {}).items()
            }

            # Process each local contact
            for local_contact in local_contacts:
                try:
                    # Skip contacts that came from other providers
                    if local_contact.provider and local_contact.provider != provider:
                        result.add_result(local_contact, "skipped")
                        continue

                    # Check if this contact already exists in the provider
                    if local_contact.id in local_to_provider_id:
                        provider_id = local_to_provider_id[local_contact.id]

                        # Get the provider contact
                        provider_contact = provider_contact_dict.get(
                            provider_id)

                        if provider_contact:
                            # Check if we need to update
                            if self._should_update_provider(local_contact, provider_contact, merge_strategy):
                                # Set the provider ID and update
                                local_contact.provider = provider
                                local_contact.provider_id = provider_id

                                adapter.update_contact(local_contact)
                                result.add_result(local_contact, "updated")
                            else:
                                # Skip this contact
                                result.add_result(local_contact, "skipped")
                        else:
                            # Provider contact not found, add it
                            local_contact.provider = provider
                            local_contact.provider_id = ""  # Will be set by create_contact

                            created_contact = adapter.create_contact(
                                local_contact)

                            # Update the local contact with the provider ID
                            local_contact.provider_id = created_contact.provider_id
                            self.local_storage.update_contact(local_contact)

                            # Update the ID mapping
                            self.provider_to_local_id.setdefault(
                                provider, {})[local_contact.provider_id] = local_contact.id

                            result.add_result(local_contact, "added")
                    else:
                        # Add as a new contact
                        local_contact.provider = provider
                        local_contact.provider_id = ""  # Will be set by create_contact

                        created_contact = adapter.create_contact(local_contact)

                        # Update the local contact with the provider ID
                        local_contact.provider_id = created_contact.provider_id
                        self.local_storage.update_contact(local_contact)

                        # Update the ID mapping
                        self.provider_to_local_id.setdefault(
                            provider, {})[local_contact.provider_id] = local_contact.id

                        result.add_result(local_contact, "added")
                except Exception as e:
                    logger.error(
                        f"Error processing contact {local_contact.id}: {e}")
                    result.add_result(local_contact, "error", e)

            # Update last sync time
            self.last_sync_time[provider] = time.time()

            return result
        except Exception as e:
            logger.error(f"Error syncing contacts to {provider}: {e}")
            raise

    def _should_update_provider(
        self,
        local_contact: Contact,
        provider_contact: Contact,
        merge_strategy: str
    ) -> bool:
        """
        Determine if a provider contact should be updated with local data.

        Args:
            local_contact: Local contact
            provider_contact: Provider contact
            merge_strategy: Merge strategy

        Returns:
            True if the provider contact should be updated
        """
        if merge_strategy == "local_wins":
            return True
        elif merge_strategy == "provider_wins":
            return False
        elif merge_strategy == "newer_wins":
            # Parse timestamps (if available)
            local_time = self._parse_timestamp(local_contact.updated_at)
            provider_time = self._parse_timestamp(provider_contact.updated_at)

            # If we have both timestamps, compare them
            if local_time and provider_time:
                return local_time > provider_time

            # If we only have one timestamp, use that
            return bool(local_time)
        else:
            raise ValueError(f"Invalid merge strategy: {merge_strategy}")

    def do_bidirectional_sync(
        self,
        provider: str,
        conflict_resolution: str = "newer_wins"
    ) -> Tuple[ContactSyncResult, ContactSyncResult]:
        """
        Perform a bidirectional sync between provider and local storage.

        Args:
            provider: Provider name ('google', 'microsoft', 'apple')
            conflict_resolution: Strategy for resolving conflicts ('provider_wins', 'local_wins', 'newer_wins')

        Returns:
            Tuple of (provider_to_local_result, local_to_provider_result)
        """
        logger.info(f"Starting bidirectional sync with {provider}")

        # First sync from provider to local
        from_provider_result = self.sync_from_provider(
            provider, conflict_resolution)

        # Then sync from local to provider
        to_provider_result = self.sync_to_provider(
            provider, conflict_resolution)

        return from_provider_result, to_provider_result

    def find_duplicates(self) -> List[List[Contact]]:
        """
        Find duplicate contacts in local storage.

        Returns:
            List of groups of duplicate contacts
        """
        # Get all local contacts
        contacts = self.local_storage.get_all_contacts()

        # Create groups of potential duplicates
        potential_duplicates: Dict[str, List[Contact]] = {}

        # Group by email (if any email address matches)
        for contact in contacts:
            for email in contact.email:
                email_val = email["value"].lower()
                if email_val:
                    if email_val not in potential_duplicates:
                        potential_duplicates[email_val] = []
                    potential_duplicates[email_val].append(contact)

        # Find groups with more than one contact
        duplicate_groups = [
            group for group in potential_duplicates.values()
            if len(group) > 1
        ]

        return duplicate_groups

    def merge_duplicates(
        self,
        duplicate_groups: Optional[List[List[Contact]]] = None,
        merge_strategy: str = "newer_wins"
    ) -> List[Contact]:
        """
        Merge duplicate contacts.

        Args:
            duplicate_groups: Groups of duplicate contacts, or None to find duplicates
            merge_strategy: Strategy for merging ('newer_wins', 'most_complete')

        Returns:
            List of merged contacts
        """
        if duplicate_groups is None:
            duplicate_groups = self.find_duplicates()

        merged_contacts = []

        for group in duplicate_groups:
            if not group:
                continue

            # Determine which contact to keep
            if merge_strategy == "newer_wins":
                # Find the newest contact
                sorted_contacts = sorted(
                    group,
                    key=lambda c: self._parse_timestamp(
                        c.updated_at) or datetime.min,
                    reverse=True
                )
                base_contact = sorted_contacts[0]
                other_contacts = sorted_contacts[1:]
            elif merge_strategy == "most_complete":
                # Find the most complete contact (most non-empty fields)
                def count_fields(c: Contact) -> int:
                    count = 0
                    if c.name:
                        count += 1
                    count += len(c.email)
                    count += len(c.phone)
                    if c.organization:
                        count += 1
                    if c.job_title:
                        count += 1
                    count += len(c.addresses)
                    if c.notes:
                        count += 1
                    count += len(c.tags)
                    return count

                sorted_contacts = sorted(
                    group,
                    key=count_fields,
                    reverse=True
                )
                base_contact = sorted_contacts[0]
                other_contacts = sorted_contacts[1:]
            else:
                raise ValueError(f"Invalid merge strategy: {merge_strategy}")

            # Merge the other contacts into the base contact
            merged_contact = self._merge_contacts(base_contact, other_contacts)

            # Update the local storage
            self.local_storage.update_contact(merged_contact)

            # Delete the other contacts
            for contact in other_contacts:
                self.local_storage.delete_contact(contact.id)

            merged_contacts.append(merged_contact)

        return merged_contacts

    def _merge_contacts(self, base_contact: Contact, other_contacts: List[Contact]) -> Contact:
        """
        Merge multiple contacts into a base contact.

        Args:
            base_contact: Base contact to merge into
            other_contacts: Other contacts to merge from

        Returns:
            Merged contact
        """
        # Make a copy of the base contact
        merged = Contact(
            id=base_contact.id,
            name=base_contact.name,
            email=base_contact.email.copy(),
            phone=base_contact.phone.copy(),
            organization=base_contact.organization,
            job_title=base_contact.job_title,
            addresses=base_contact.addresses.copy(),
            photo_url=base_contact.photo_url,
            notes=base_contact.notes,
            tags=base_contact.tags.copy(),
            provider=base_contact.provider,
            provider_id=base_contact.provider_id,
            updated_at=base_contact.updated_at,
            created_at=base_contact.created_at
        )

        # Sets to track what we already have
        email_values = {email["value"].lower() for email in merged.email}
        phone_values = {phone["value"] for phone in merged.phone}
        address_values = {
            (addr.street, addr.city, addr.state, addr.postal_code, addr.country)
            for addr in merged.addresses
        }
        tag_values = set(merged.tags)

        # Merge in data from other contacts
        for contact in other_contacts:
            # Add emails that don't already exist
            for email in contact.email:
                if email["value"].lower() not in email_values:
                    merged.email.append(email)
                    email_values.add(email["value"].lower())

            # Add phones that don't already exist
            for phone in contact.phone:
                if phone["value"] not in phone_values:
                    merged.phone.append(phone)
                    phone_values.add(phone["value"])

            # Add organization if not already set
            if not merged.organization and contact.organization:
                merged.organization = contact.organization

            # Add job title if not already set
            if not merged.job_title and contact.job_title:
                merged.job_title = contact.job_title

            # Add addresses that don't already exist
            for addr in contact.addresses:
                addr_key = (addr.street, addr.city, addr.state,
                            addr.postal_code, addr.country)
                if addr_key not in address_values:
                    merged.addresses.append(addr)
                    address_values.add(addr_key)

            # Add photo URL if not already set
            if not merged.photo_url and contact.photo_url:
                merged.photo_url = contact.photo_url

            # Add notes if not already set
            if not merged.notes and contact.notes:
                merged.notes = contact.notes

            # Add tags that don't already exist
            for tag in contact.tags:
                if tag not in tag_values:
                    merged.tags.append(tag)
                    tag_values.add(tag)

        return merged

    def get_sync_status(self, provider: str) -> Dict[str, Any]:
        """
        Get sync status for a provider.

        Args:
            provider: Provider name

        Returns:
            Status information
        """
        local_count = 0
        provider_count = 0
        last_sync = self.last_sync_time.get(provider, 0)

        # Count local contacts from this provider
        for contact in self.local_storage.get_all_contacts():
            if contact.provider == provider:
                local_count += 1

        # Count provider contacts
        try:
            adapter = self.get_adapter(provider)
            provider_contacts = adapter.get_contacts()
            provider_count = len(provider_contacts)
        except Exception as e:
            logger.error(f"Error getting contacts from {provider}: {e}")

        return {
            "provider": provider,
            "local_count": local_count,
            "provider_count": provider_count,
            "last_sync": last_sync,
            "last_sync_formatted": datetime.fromtimestamp(last_sync).isoformat() if last_sync else None,
            "is_connected": bool(self.token_storage.load_token(provider, self.user_id))
        }
