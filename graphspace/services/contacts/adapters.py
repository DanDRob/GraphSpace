import requests
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


from graphspace.utils.auth_service import OAuth2Service, OAuth2TokenStorage
from .base import Contact, ContactAddress, ContactAdapter


class GoogleContactAdapter(ContactAdapter):
    """Adapter for Google People API."""

    def __init__(self, oauth_service: OAuth2Service, token_storage: OAuth2TokenStorage, user_id: str):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id
        self.api_base_url = "https://people.googleapis.com/v1"

    def _get_headers(self) -> Dict[str, str]:
        """Get authenticated headers for API requests."""
        token_data = self.token_storage.load_token("google", self.user_id)

        if not token_data:
            raise Exception("No Google authentication token found")

        if self.oauth_service.is_token_expired(token_data):
            if 'refresh_token' not in token_data:
                raise Exception(
                    "Google token expired and no refresh token available")

            token_data = self.oauth_service.refresh_token(
                "google", token_data["refresh_token"])
            token_data = self.oauth_service.update_token_expiry(token_data)
            self.token_storage.save_token("google", self.user_id, token_data)

        return self.oauth_service.get_authenticated_header("google", token_data)

    def get_contacts(self) -> List[Contact]:
        """Get all contacts from Google."""
        headers = self._get_headers()

        # Request fields we want to retrieve
        person_fields = [
            "names", "emailAddresses", "phoneNumbers", "organizations",
            "addresses", "photos", "biographies", "userDefined"
        ]

        params = {
            "personFields": ",".join(person_fields),
            "pageSize": 100  # Max page size
        }

        url = f"{self.api_base_url}/people/me/connections"

        all_contacts = []
        next_page_token = None

        # Paginate through all contacts
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token

            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                raise Exception(
                    f"Failed to get Google contacts: {response.text}")

            data = response.json()

            # Process contacts in this page
            connections = data.get("connections", [])
            for person in connections:
                contact = self._convert_google_person_to_contact(person)
                all_contacts.append(contact)

            # Check if there are more pages
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        return all_contacts

    def _convert_google_person_to_contact(self, person: Dict[str, Any]) -> Contact:
        """Convert a Google Person resource to a Contact object."""
        # Extract the resource name (ID)
        resource_name = person.get("resourceName", "")

        # Extract name
        name = ""
        if "names" in person and person["names"]:
            name = person["names"][0].get("displayName", "")

        # Extract emails
        emails = []
        if "emailAddresses" in person and person["emailAddresses"]:
            for email in person["emailAddresses"]:
                emails.append({
                    "value": email.get("value", ""),
                    "type": email.get("type", "").lower(),
                    "primary": email.get("metadata", {}).get("primary", False)
                })

        # Extract phone numbers
        phones = []
        if "phoneNumbers" in person and person["phoneNumbers"]:
            for phone in person["phoneNumbers"]:
                phones.append({
                    "value": phone.get("value", ""),
                    "type": phone.get("type", "").lower(),
                    "primary": phone.get("metadata", {}).get("primary", False)
                })

        # Extract organization
        organization = ""
        job_title = ""
        if "organizations" in person and person["organizations"]:
            org = person["organizations"][0]
            organization = org.get("name", "")
            job_title = org.get("title", "")

        # Extract addresses
        addresses = []
        if "addresses" in person and person["addresses"]:
            for addr in person["addresses"]:
                addresses.append(ContactAddress(
                    street=addr.get("streetAddress", ""),
                    city=addr.get("city", ""),
                    state=addr.get("region", ""),
                    postal_code=addr.get("postalCode", ""),
                    country=addr.get("country", ""),
                    type=addr.get("type", "").lower()
                ))

        # Extract photo URL
        photo_url = ""
        if "photos" in person and person["photos"]:
            for photo in person["photos"]:
                if not photo.get("default", True):
                    photo_url = photo.get("url", "")
                    break

        # Extract notes
        notes = ""
        if "biographies" in person and person["biographies"]:
            notes = person["biographies"][0].get("value", "")

        # Extract tags (user-defined data)
        tags = []
        if "userDefined" in person and person["userDefined"]:
            for user_data in person["userDefined"]:
                if user_data.get("key") == "tags":
                    tags = [tag.strip()
                            for tag in user_data.get("value", "").split(",")]

        # Create contact object
        contact = Contact(
            id=resource_name,
            name=name,
            email=emails,
            phone=phones,
            organization=organization,
            job_title=job_title,
            addresses=addresses,
            photo_url=photo_url,
            notes=notes,
            tags=tags,
            provider="google",
            provider_id=resource_name
        )

        return contact

    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact in Google."""
        headers = self._get_headers()

        # Convert Contact to Google Person resource
        person = self._convert_contact_to_google_person(contact)

        url = f"{self.api_base_url}/people:createContact"
        response = requests.post(url, headers=headers, json=person)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to create Google contact: {response.text}")

        # Get the created contact data
        created_person = response.json()

        # Convert back to Contact object
        created_contact = self._convert_google_person_to_contact(
            created_person)

        return created_contact

    def _convert_contact_to_google_person(self, contact: Contact) -> Dict[str, Any]:
        """Convert a Contact object to a Google Person resource."""
        person = {}

        # Add name
        if contact.name:
            person["names"] = [
                {
                    "displayName": contact.name,
                    "unstructuredName": contact.name
                }
            ]

        # Add emails
        if contact.email:
            person["emailAddresses"] = []
            for email in contact.email:
                person["emailAddresses"].append({
                    "value": email["value"],
                    "type": email.get("type", "home").upper(),
                    "metadata": {
                        "primary": email.get("primary", False)
                    }
                })

        # Add phone numbers
        if contact.phone:
            person["phoneNumbers"] = []
            for phone in contact.phone:
                person["phoneNumbers"].append({
                    "value": phone["value"],
                    "type": phone.get("type", "home").upper(),
                    "metadata": {
                        "primary": phone.get("primary", False)
                    }
                })

        # Add organization
        if contact.organization or contact.job_title:
            person["organizations"] = [
                {
                    "name": contact.organization,
                    "title": contact.job_title
                }
            ]

        # Add addresses
        if contact.addresses:
            person["addresses"] = []
            for addr in contact.addresses:
                person["addresses"].append({
                    "streetAddress": addr.street,
                    "city": addr.city,
                    "region": addr.state,
                    "postalCode": addr.postal_code,
                    "country": addr.country,
                    "type": addr.type.upper()
                })

        # Add notes
        if contact.notes:
            person["biographies"] = [
                {
                    "value": contact.notes,
                    "contentType": "TEXT_PLAIN"
                }
            ]

        # Add tags
        if contact.tags:
            person["userDefined"] = [
                {
                    "key": "tags",
                    "value": ",".join(contact.tags)
                }
            ]

        return person

    def update_contact(self, contact: Contact) -> Contact:
        """Update an existing contact in Google."""
        headers = self._get_headers()

        # Ensure we have a resource name (Google ID)
        if not contact.provider_id:
            raise ValueError(
                "Cannot update contact without Google resource name")

        # Convert Contact to Google Person resource
        person = self._convert_contact_to_google_person(contact)

        # Add update mask (fields to update)
        update_mask = [
            "names", "emailAddresses", "phoneNumbers", "organizations",
            "addresses", "biographies", "userDefined"
        ]

        url = f"{self.api_base_url}/{contact.provider_id}:updateContact"
        params = {
            "updatePersonFields": ",".join(update_mask)
        }

        response = requests.patch(
            url, headers=headers, params=params, json=person)

        if response.status_code != 200:
            raise Exception(
                f"Failed to update Google contact: {response.text}")

        # Get the updated contact data
        updated_person = response.json()

        # Convert back to Contact object
        updated_contact = self._convert_google_person_to_contact(
            updated_person)

        return updated_contact

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from Google."""
        headers = self._get_headers()

        url = f"{self.api_base_url}/{contact_id}:deleteContact"
        response = requests.delete(url, headers=headers)

        if response.status_code not in (200, 204):
            raise Exception(
                f"Failed to delete Google contact: {response.text}")

        return True


class MicrosoftContactAdapter(ContactAdapter):
    """Adapter for Microsoft Graph API for contacts."""

    def __init__(self, oauth_service: OAuth2Service, token_storage: OAuth2TokenStorage, user_id: str):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id
        self.api_base_url = "https://graph.microsoft.com/v1.0"

    def _get_headers(self) -> Dict[str, str]:
        """Get authenticated headers for API requests."""
        token_data = self.token_storage.load_token("microsoft", self.user_id)

        if not token_data:
            raise Exception("No Microsoft authentication token found")

        if self.oauth_service.is_token_expired(token_data):
            if 'refresh_token' not in token_data:
                raise Exception(
                    "Microsoft token expired and no refresh token available")

            token_data = self.oauth_service.refresh_token(
                "microsoft", token_data["refresh_token"])
            token_data = self.oauth_service.update_token_expiry(token_data)
            self.token_storage.save_token(
                "microsoft", self.user_id, token_data)

        return self.oauth_service.get_authenticated_header("microsoft", token_data)

    def get_contacts(self) -> List[Contact]:
        """Get all contacts from Microsoft."""
        headers = self._get_headers()

        url = f"{self.api_base_url}/me/contacts"
        params = {
            "$top": 100  # Page size
        }

        all_contacts = []
        next_link = None

        # Paginate through all contacts
        while True:
            # If we have a next link, use it directly
            if next_link:
                response = requests.get(next_link, headers=headers)
            else:
                response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                raise Exception(
                    f"Failed to get Microsoft contacts: {response.text}")

            data = response.json()

            # Process contacts in this page
            contacts = data.get("value", [])
            for contact_data in contacts:
                contact = self._convert_microsoft_contact_to_contact(
                    contact_data)
                all_contacts.append(contact)

            # Check if there are more pages
            next_link = data.get("@odata.nextLink")
            if not next_link:
                break

        return all_contacts

    def _convert_microsoft_contact_to_contact(self, ms_contact: Dict[str, Any]) -> Contact:
        """Convert a Microsoft contact resource to a Contact object."""
        # Extract ID
        contact_id = ms_contact.get("id", "")

        # Extract name
        display_name = ms_contact.get("displayName", "")

        # Extract emails
        emails = []
        for email_type in ["homeEmails", "businessEmails", "otherEmails"]:
            if email_type in ms_contact and ms_contact[email_type]:
                for email in ms_contact[email_type]:
                    email_type_simplified = email_type.replace("Emails", "")
                    emails.append({
                        "value": email,
                        "type": email_type_simplified,
                        "primary": email_type == "homeEmails"  # Assume home email is primary
                    })

        # Extract phone numbers
        phones = []
        for phone_type in ["homePhones", "businessPhones", "mobilePhone"]:
            if phone_type in ms_contact and ms_contact[phone_type]:
                # Handle mobilePhone differently (it's a single string)
                if phone_type == "mobilePhone":
                    phones.append({
                        "value": ms_contact[phone_type],
                        "type": "mobile",
                        "primary": True  # Assume mobile is primary
                    })
                else:
                    for phone in ms_contact[phone_type]:
                        phone_type_simplified = phone_type.replace(
                            "Phones", "")
                        phones.append({
                            "value": phone,
                            "type": phone_type_simplified,
                            "primary": phone_type == "mobilePhone"
                        })

        # Extract organization and job title
        company_name = ms_contact.get("companyName", "")
        job_title = ms_contact.get("jobTitle", "")

        # Extract addresses
        addresses = []
        if "homeAddress" in ms_contact and any(ms_contact["homeAddress"].values()):
            home_addr = ms_contact["homeAddress"]
            addresses.append(ContactAddress(
                street=home_addr.get("street", ""),
                city=home_addr.get("city", ""),
                state=home_addr.get("state", ""),
                postal_code=home_addr.get("postalCode", ""),
                country=home_addr.get("countryOrRegion", ""),
                type="home"
            ))

        if "businessAddress" in ms_contact and any(ms_contact["businessAddress"].values()):
            business_addr = ms_contact["businessAddress"]
            addresses.append(ContactAddress(
                street=business_addr.get("street", ""),
                city=business_addr.get("city", ""),
                state=business_addr.get("state", ""),
                postal_code=business_addr.get("postalCode", ""),
                country=business_addr.get("countryOrRegion", ""),
                type="work"
            ))

        # Extract notes
        notes = ms_contact.get("personalNotes", "")

        # There's no direct tags field in Microsoft Contacts, but we could use categories
        tags = []
        if "categories" in ms_contact:
            tags = ms_contact["categories"]

        # Extract created and updated timestamps
        created_at = ms_contact.get("createdDateTime", "")
        updated_at = ms_contact.get("lastModifiedDateTime", "")

        # Create contact object
        contact = Contact(
            id=contact_id,
            name=display_name,
            email=emails,
            phone=phones,
            organization=company_name,
            job_title=job_title,
            addresses=addresses,
            notes=notes,
            tags=tags,
            provider="microsoft",
            provider_id=contact_id,
            created_at=created_at,
            updated_at=updated_at
        )

        return contact

    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact in Microsoft."""
        headers = self._get_headers()

        # Convert Contact to Microsoft contact resource
        ms_contact = self._convert_contact_to_microsoft_contact(contact)

        url = f"{self.api_base_url}/me/contacts"
        response = requests.post(url, headers=headers, json=ms_contact)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to create Microsoft contact: {response.text}")

        # Get the created contact data
        created_contact_data = response.json()

        # Convert back to Contact object
        created_contact = self._convert_microsoft_contact_to_contact(
            created_contact_data)

        return created_contact

    def _convert_contact_to_microsoft_contact(self, contact: Contact) -> Dict[str, Any]:
        """Convert a Contact object to a Microsoft contact resource."""
        ms_contact = {
            "displayName": contact.name
        }

        # Add emails by type
        home_emails = []
        business_emails = []
        other_emails = []

        for email in contact.email:
            email_type = email.get("type", "home").lower()
            if email_type == "home":
                home_emails.append(email["value"])
            elif email_type in ("work", "business"):
                business_emails.append(email["value"])
            else:
                other_emails.append(email["value"])

        if home_emails:
            ms_contact["homeEmails"] = home_emails
        if business_emails:
            ms_contact["businessEmails"] = business_emails
        if other_emails:
            ms_contact["otherEmails"] = other_emails

        # Add phone numbers by type
        home_phones = []
        business_phones = []
        mobile_phone = None

        for phone in contact.phone:
            phone_type = phone.get("type", "home").lower()
            if phone_type == "mobile":
                mobile_phone = phone["value"]
            elif phone_type == "home":
                home_phones.append(phone["value"])
            elif phone_type in ("work", "business"):
                business_phones.append(phone["value"])
            else:
                # Default to home
                home_phones.append(phone["value"])

        if home_phones:
            ms_contact["homePhones"] = home_phones
        if business_phones:
            ms_contact["businessPhones"] = business_phones
        if mobile_phone:
            ms_contact["mobilePhone"] = mobile_phone

        # Add organization and job title
        if contact.organization:
            ms_contact["companyName"] = contact.organization
        if contact.job_title:
            ms_contact["jobTitle"] = contact.job_title

        # Add addresses
        for addr in contact.addresses:
            addr_dict = {
                "street": addr.street,
                "city": addr.city,
                "state": addr.state,
                "postalCode": addr.postal_code,
                "countryOrRegion": addr.country
            }

            if addr.type.lower() == "home":
                ms_contact["homeAddress"] = addr_dict
            elif addr.type.lower() in ("work", "business"):
                ms_contact["businessAddress"] = addr_dict

        # Add notes
        if contact.notes:
            ms_contact["personalNotes"] = contact.notes

        # Add tags as categories
        if contact.tags:
            ms_contact["categories"] = contact.tags

        return ms_contact

    def update_contact(self, contact: Contact) -> Contact:
        """Update an existing contact in Microsoft."""
        headers = self._get_headers()

        # Ensure we have a contact ID
        if not contact.provider_id:
            raise ValueError(
                "Cannot update contact without Microsoft contact ID")

        # Convert Contact to Microsoft contact resource
        ms_contact = self._convert_contact_to_microsoft_contact(contact)

        url = f"{self.api_base_url}/me/contacts/{contact.provider_id}"
        response = requests.patch(url, headers=headers, json=ms_contact)

        if response.status_code != 200:
            raise Exception(
                f"Failed to update Microsoft contact: {response.text}")

        # Get the updated contact data
        updated_contact_data = response.json()

        # Convert back to Contact object
        updated_contact = self._convert_microsoft_contact_to_contact(
            updated_contact_data)

        return updated_contact

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact from Microsoft."""
        headers = self._get_headers()

        url = f"{self.api_base_url}/me/contacts/{contact_id}"
        response = requests.delete(url, headers=headers)

        if response.status_code not in (200, 204):
            raise Exception(
                f"Failed to delete Microsoft contact: {response.text}")

        return True


class AppleContactAdapter(ContactAdapter):
    """
    Adapter for Apple Contacts.

    Note: This is a stub implementation as Apple doesn't provide a direct API for contacts.
    A real implementation would use CardDAV protocol.
    """

    def __init__(self, oauth_service: OAuth2Service, token_storage: OAuth2TokenStorage, user_id: str):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id

        # For demo/testing purposes, use an in-memory contact store
        self.contacts = []

    def get_contacts(self) -> List[Contact]:
        """Get all contacts (stub implementation)."""
        return self.contacts

    def create_contact(self, contact: Contact) -> Contact:
        """Create a new contact (stub implementation)."""
        # Set provider
        contact.provider = "apple"
        contact.provider_id = str(len(self.contacts) + 1)

        # Add to our test store
        self.contacts.append(contact)

        return contact

    def update_contact(self, contact: Contact) -> Contact:
        """Update an existing contact (stub implementation)."""
        # Find and update the contact in our test store
        for i, existing_contact in enumerate(self.contacts):
            if existing_contact.provider_id == contact.provider_id:
                self.contacts[i] = contact
                return contact

        raise ValueError(f"Contact with ID {contact.provider_id} not found")

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact (stub implementation)."""
        # Find and delete the contact from our test store
        for i, contact in enumerate(self.contacts):
            if contact.provider_id == contact_id:
                del self.contacts[i]
                return True

        return False
