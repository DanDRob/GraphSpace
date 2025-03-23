import os
import json
import time
import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests
from abc import ABC, abstractmethod
import uuid
import icalendar
import recurring_ical_events
from dateutil import parser

from ..utils.auth_service import OAuth2Service, OAuth2TokenStorage


class CalendarEvent:
    def __init__(
        self,
        id: str = None,
        title: str = "",
        description: str = "",
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        location: str = "",
        attendees: List[str] = None,
        provider_data: Dict[str, Any] = None,
        calendar_id: str = None,
        provider: str = None,
        is_all_day: bool = False,
        is_recurring: bool = False,
        recurrence_rule: str = None
    ):
        self.id = id or str(uuid.uuid4())
        self.title = title
        self.description = description
        self.start_time = start_time or datetime.datetime.now()
        self.end_time = end_time or (
            self.start_time + datetime.timedelta(hours=1))
        self.location = location
        self.attendees = attendees or []
        self.provider_data = provider_data or {}
        self.calendar_id = calendar_id
        self.provider = provider
        self.is_all_day = is_all_day
        self.is_recurring = is_recurring
        self.recurrence_rule = recurrence_rule

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "location": self.location,
            "attendees": self.attendees,
            "calendar_id": self.calendar_id,
            "provider": self.provider,
            "is_all_day": self.is_all_day,
            "is_recurring": self.is_recurring,
            "recurrence_rule": self.recurrence_rule
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalendarEvent':
        # Convert ISO format strings back to datetime objects
        start_time = parser.parse(data["start_time"]) if data.get(
            "start_time") else None
        end_time = parser.parse(data["end_time"]) if data.get(
            "end_time") else None

        return cls(
            id=data.get("id"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            start_time=start_time,
            end_time=end_time,
            location=data.get("location", ""),
            attendees=data.get("attendees", []),
            provider_data=data.get("provider_data", {}),
            calendar_id=data.get("calendar_id"),
            provider=data.get("provider"),
            is_all_day=data.get("is_all_day", False),
            is_recurring=data.get("is_recurring", False),
            recurrence_rule=data.get("recurrence_rule")
        )


class Calendar:
    def __init__(
        self,
        id: str,
        name: str,
        description: str = "",
        provider: str = "",
        provider_data: Dict[str, Any] = None,
        color: str = "#3498db"
    ):
        self.id = id
        self.name = name
        self.description = description
        self.provider = provider
        self.provider_data = provider_data or {}
        self.color = color

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Calendar':
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            provider=data.get("provider", ""),
            provider_data=data.get("provider_data", {}),
            color=data.get("color", "#3498db")
        )


class CalendarAdapter(ABC):
    """Abstract base class for calendar service adapters."""

    @abstractmethod
    def get_calendars(self) -> List[Calendar]:
        """Get a list of available calendars."""
        pass

    @abstractmethod
    def get_events(
        self,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime
    ) -> List[CalendarEvent]:
        """Get events from a calendar within a date range."""
        pass

    @abstractmethod
    def create_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Create a new event in the specified calendar."""
        pass

    @abstractmethod
    def update_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Update an existing event."""
        pass

    @abstractmethod
    def delete_event(self, calendar_id: str, event_id: str) -> bool:
        """Delete an event from the calendar."""
        pass


class GoogleCalendarAdapter(CalendarAdapter):
    """Adapter for Google Calendar API."""

    def __init__(self, oauth_service: OAuth2Service, token_storage: OAuth2TokenStorage, user_id: str):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id
        self.api_base_url = "https://www.googleapis.com/calendar/v3"

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

    def get_calendars(self) -> List[Calendar]:
        """Get list of Google calendars."""
        headers = self._get_headers()
        response = requests.get(
            f"{self.api_base_url}/users/me/calendarList", headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to get Google calendars: {response.text}")

        calendars = []
        for item in response.json().get("items", []):
            calendars.append(Calendar(
                id=item["id"],
                name=item["summary"],
                description=item.get("description", ""),
                provider="google",
                provider_data=item,
                color=item.get("backgroundColor", "#3498db")
            ))

        return calendars

    def get_events(
        self,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime
    ) -> List[CalendarEvent]:
        """Get events from Google Calendar within a date range."""
        headers = self._get_headers()

        # Format dates for Google API
        time_min = start_date.isoformat() + "Z"  # 'Z' indicates UTC time
        time_max = end_date.isoformat() + "Z"

        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",  # Expand recurring events
            "maxResults": 2500  # Maximum allowed by Google
        }

        url = f"{self.api_base_url}/calendars/{calendar_id}/events"
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(
                f"Failed to get Google calendar events: {response.text}")

        events = []
        for item in response.json().get("items", []):
            # Parse start and end times
            start = item.get("start", {})
            end = item.get("end", {})

            is_all_day = "date" in start and "date" in end

            if is_all_day:
                start_time = datetime.datetime.fromisoformat(start["date"])
                end_time = datetime.datetime.fromisoformat(end["date"])
            else:
                start_time = datetime.datetime.fromisoformat(
                    start.get("dateTime", "").replace("Z", "+00:00"))
                end_time = datetime.datetime.fromisoformat(
                    end.get("dateTime", "").replace("Z", "+00:00"))

            # Get attendees
            attendees = []
            for attendee in item.get("attendees", []):
                if "email" in attendee:
                    attendees.append(attendee["email"])

            events.append(CalendarEvent(
                id=item["id"],
                title=item["summary"],
                description=item.get("description", ""),
                start_time=start_time,
                end_time=end_time,
                location=item.get("location", ""),
                attendees=attendees,
                provider_data=item,
                calendar_id=calendar_id,
                provider="google",
                is_all_day=is_all_day,
                is_recurring="recurrence" in item,
                recurrence_rule=item.get("recurrence", [None])[0]
            ))

        return events

    def create_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Create a new event in Google Calendar."""
        headers = self._get_headers()

        # Prepare event data for Google API
        event_data = {
            "summary": event.title,
            "description": event.description,
            "location": event.location
        }

        # Handle all-day events differently
        if event.is_all_day:
            event_data["start"] = {
                "date": event.start_time.date().isoformat()
            }
            event_data["end"] = {
                "date": event.end_time.date().isoformat()
            }
        else:
            event_data["start"] = {
                "dateTime": event.start_time.isoformat()
            }
            event_data["end"] = {
                "dateTime": event.end_time.isoformat()
            }

        # Add attendees if present
        if event.attendees:
            event_data["attendees"] = [{"email": email}
                                       for email in event.attendees]

        # Add recurrence if specified
        if event.recurrence_rule:
            event_data["recurrence"] = [event.recurrence_rule]

        url = f"{self.api_base_url}/calendars/{calendar_id}/events"
        response = requests.post(url, headers=headers, json=event_data)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to create Google calendar event: {response.text}")

        created_event = response.json()

        # Update event with created data
        event.id = created_event["id"]
        event.provider_data = created_event

        return event

    def update_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Update an existing event in Google Calendar."""
        if not event.id:
            raise ValueError("Event ID is required for update")

        headers = self._get_headers()

        # Prepare event data for Google API
        event_data = {
            "summary": event.title,
            "description": event.description,
            "location": event.location
        }

        # Handle all-day events differently
        if event.is_all_day:
            event_data["start"] = {
                "date": event.start_time.date().isoformat()
            }
            event_data["end"] = {
                "date": event.end_time.date().isoformat()
            }
        else:
            event_data["start"] = {
                "dateTime": event.start_time.isoformat()
            }
            event_data["end"] = {
                "dateTime": event.end_time.isoformat()
            }

        # Add attendees if present
        if event.attendees:
            event_data["attendees"] = [{"email": email}
                                       for email in event.attendees]

        # Add recurrence if specified
        if event.recurrence_rule:
            event_data["recurrence"] = [event.recurrence_rule]

        url = f"{self.api_base_url}/calendars/{calendar_id}/events/{event.id}"
        response = requests.put(url, headers=headers, json=event_data)

        if response.status_code != 200:
            raise Exception(
                f"Failed to update Google calendar event: {response.text}")

        updated_event = response.json()

        # Update event with latest data
        event.provider_data = updated_event

        return event

    def delete_event(self, calendar_id: str, event_id: str) -> bool:
        """Delete an event from Google Calendar."""
        headers = self._get_headers()

        url = f"{self.api_base_url}/calendars/{calendar_id}/events/{event_id}"
        response = requests.delete(url, headers=headers)

        if response.status_code not in (200, 204):
            raise Exception(
                f"Failed to delete Google calendar event: {response.text}")

        return True


class MicrosoftCalendarAdapter(CalendarAdapter):
    """Adapter for Microsoft Graph API for calendar access."""

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

    def get_calendars(self) -> List[Calendar]:
        """Get list of Microsoft calendars."""
        headers = self._get_headers()
        response = requests.get(
            f"{self.api_base_url}/me/calendars", headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Failed to get Microsoft calendars: {response.text}")

        calendars = []
        for item in response.json().get("value", []):
            calendars.append(Calendar(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                provider="microsoft",
                provider_data=item,
                color="#0078d4"  # Microsoft blue
            ))

        return calendars

    def get_events(
        self,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime
    ) -> List[CalendarEvent]:
        """Get events from Microsoft Calendar within a date range."""
        headers = self._get_headers()

        # Format dates for Microsoft API
        start_datetime = start_date.isoformat()
        end_datetime = end_date.isoformat()

        # Use query parameters to filter events by date range
        filter_query = f"startsWith(subject, ') and start/dateTime ge '{start_datetime}' and end/dateTime le '{end_datetime}'"
        params = {
            "$filter": filter_query,
            "$top": 100  # Limit number of results
        }

        url = f"{self.api_base_url}/me/calendars/{calendar_id}/events"
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(
                f"Failed to get Microsoft calendar events: {response.text}")

        events = []
        for item in response.json().get("value", []):
            # Parse start and end times
            start = item.get("start", {})
            end = item.get("end", {})

            start_time = datetime.datetime.fromisoformat(
                start.get("dateTime", "").replace("Z", "+00:00"))
            end_time = datetime.datetime.fromisoformat(
                end.get("dateTime", "").replace("Z", "+00:00"))

            # Check if it's an all-day event
            is_all_day = item.get("isAllDay", False)

            # Get attendees
            attendees = []
            for attendee in item.get("attendees", []):
                if "emailAddress" in attendee:
                    attendees.append(
                        attendee["emailAddress"].get("address", ""))

            events.append(CalendarEvent(
                id=item["id"],
                title=item["subject"],
                description=item.get("bodyPreview", ""),
                start_time=start_time,
                end_time=end_time,
                location=item.get("location", {}).get("displayName", ""),
                attendees=attendees,
                provider_data=item,
                calendar_id=calendar_id,
                provider="microsoft",
                is_all_day=is_all_day,
                is_recurring=item.get("recurrence") is not None,
                recurrence_rule=json.dumps(
                    item.get("recurrence")) if item.get("recurrence") else None
            ))

        return events

    def create_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Create a new event in Microsoft Calendar."""
        headers = self._get_headers()

        # Prepare event data for Microsoft API
        event_data = {
            "subject": event.title,
            "body": {
                "contentType": "text",
                "content": event.description
            },
            "start": {
                "dateTime": event.start_time.isoformat(),
                "timeZone": "UTC"
            },
            "end": {
                "dateTime": event.end_time.isoformat(),
                "timeZone": "UTC"
            },
            "location": {
                "displayName": event.location
            },
            "isAllDay": event.is_all_day
        }

        # Add attendees if present
        if event.attendees:
            event_data["attendees"] = [
                {
                    "emailAddress": {
                        "address": email
                    },
                    "type": "required"
                }
                for email in event.attendees
            ]

        # Add recurrence if specified
        if event.recurrence_rule:
            try:
                recurrence = json.loads(event.recurrence_rule)
                event_data["recurrence"] = recurrence
            except (json.JSONDecodeError, TypeError):
                # If recurrence rule isn't valid JSON, ignore it
                pass

        url = f"{self.api_base_url}/me/calendars/{calendar_id}/events"
        response = requests.post(url, headers=headers, json=event_data)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to create Microsoft calendar event: {response.text}")

        created_event = response.json()

        # Update event with created data
        event.id = created_event["id"]
        event.provider_data = created_event

        return event

    def update_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """Update an existing event in Microsoft Calendar."""
        if not event.id:
            raise ValueError("Event ID is required for update")

        headers = self._get_headers()

        # Prepare event data for Microsoft API
        event_data = {
            "subject": event.title,
            "body": {
                "contentType": "text",
                "content": event.description
            },
            "start": {
                "dateTime": event.start_time.isoformat(),
                "timeZone": "UTC"
            },
            "end": {
                "dateTime": event.end_time.isoformat(),
                "timeZone": "UTC"
            },
            "location": {
                "displayName": event.location
            },
            "isAllDay": event.is_all_day
        }

        # Add attendees if present
        if event.attendees:
            event_data["attendees"] = [
                {
                    "emailAddress": {
                        "address": email
                    },
                    "type": "required"
                }
                for email in event.attendees
            ]

        # Add recurrence if specified
        if event.recurrence_rule:
            try:
                recurrence = json.loads(event.recurrence_rule)
                event_data["recurrence"] = recurrence
            except (json.JSONDecodeError, TypeError):
                # If recurrence rule isn't valid JSON, ignore it
                pass

        url = f"{self.api_base_url}/me/calendars/{calendar_id}/events/{event.id}"
        response = requests.patch(url, headers=headers, json=event_data)

        if response.status_code != 200:
            raise Exception(
                f"Failed to update Microsoft calendar event: {response.text}")

        updated_event = response.json()

        # Update event with latest data
        event.provider_data = updated_event

        return event

    def delete_event(self, calendar_id: str, event_id: str) -> bool:
        """Delete an event from Microsoft Calendar."""
        headers = self._get_headers()

        url = f"{self.api_base_url}/me/calendars/{calendar_id}/events/{event_id}"
        response = requests.delete(url, headers=headers)

        if response.status_code not in (200, 204):
            raise Exception(
                f"Failed to delete Microsoft calendar event: {response.text}")

        return True


class AppleCalendarAdapter(CalendarAdapter):
    """Adapter for Apple Calendar using CalDAV protocol."""

    def __init__(self, oauth_service: OAuth2Service, token_storage: OAuth2TokenStorage, user_id: str):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id

        # For Apple integration, we'd need to use CalDAV, which is complex
        # This is a simplified implementation that wouldn't work in production
        # A real implementation would use a CalDAV client library
        self.calendars_data = []
        self.events_data = {}

    def get_calendars(self) -> List[Calendar]:
        """
        Get list of Apple calendars.

        Note: This is a stub implementation.
        Real implementation would use CalDAV.
        """
        # In a real implementation, we'd query CalDAV here
        # For now, return a mock calendar
        if not self.calendars_data:
            self.calendars_data.append(
                Calendar(
                    id="apple-default",
                    name="Apple Calendar",
                    description="Default Apple Calendar",
                    provider="apple",
                    color="#FF2D55"  # Apple red
                )
            )

        return self.calendars_data

    def get_events(
        self,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime
    ) -> List[CalendarEvent]:
        """
        Get events from Apple Calendar within a date range.

        Note: This is a stub implementation.
        Real implementation would use CalDAV.
        """
        # In a real implementation, we'd query CalDAV here
        # For now, return mock events if any were created previously

        if calendar_id not in self.events_data:
            return []

        events = []
        for event in self.events_data[calendar_id]:
            if (event.start_time <= end_date and event.end_time >= start_date):
                events.append(event)

        return events

    def create_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """
        Create a new event in Apple Calendar.

        Note: This is a stub implementation.
        Real implementation would use CalDAV.
        """
        # Generate a unique ID for the event
        event.id = str(uuid.uuid4())
        event.calendar_id = calendar_id
        event.provider = "apple"

        # Store the event in our mock data
        if calendar_id not in self.events_data:
            self.events_data[calendar_id] = []

        self.events_data[calendar_id].append(event)

        return event

    def update_event(self, calendar_id: str, event: CalendarEvent) -> CalendarEvent:
        """
        Update an existing event in Apple Calendar.

        Note: This is a stub implementation.
        Real implementation would use CalDAV.
        """
        if not event.id:
            raise ValueError("Event ID is required for update")

        # Find and update the event in our mock data
        if calendar_id in self.events_data:
            for i, existing_event in enumerate(self.events_data[calendar_id]):
                if existing_event.id == event.id:
                    self.events_data[calendar_id][i] = event
                    return event

        raise Exception(
            f"Event with ID {event.id} not found in calendar {calendar_id}")

    def delete_event(self, calendar_id: str, event_id: str) -> bool:
        """
        Delete an event from Apple Calendar.

        Note: This is a stub implementation.
        Real implementation would use CalDAV.
        """
        # Find and delete the event from our mock data
        if calendar_id in self.events_data:
            for i, existing_event in enumerate(self.events_data[calendar_id]):
                if existing_event.id == event_id:
                    del self.events_data[calendar_id][i]
                    return True

        raise Exception(
            f"Event with ID {event_id} not found in calendar {calendar_id}")


class CalendarService:
    """
    Service for interacting with various calendar providers.
    Uses provider-specific adapters to handle the differences.
    """

    def __init__(
        self,
        oauth_service: OAuth2Service,
        token_storage: OAuth2TokenStorage,
        user_id: str = "default"
    ):
        self.oauth_service = oauth_service
        self.token_storage = token_storage
        self.user_id = user_id
        self.adapters = {}

        # Initialize empty events cache
        self.events_cache = {}
        self.last_sync_time = {}

    def get_adapter(self, provider: str) -> CalendarAdapter:
        """Get or create an adapter for the specified provider."""
        if provider not in self.adapters:
            if provider == "google":
                self.adapters[provider] = GoogleCalendarAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            elif provider == "microsoft":
                self.adapters[provider] = MicrosoftCalendarAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            elif provider == "apple":
                self.adapters[provider] = AppleCalendarAdapter(
                    self.oauth_service,
                    self.token_storage,
                    self.user_id
                )
            else:
                raise ValueError(f"Unsupported calendar provider: {provider}")

        return self.adapters[provider]

    def get_calendars(self, provider: str) -> List[Calendar]:
        """Get a list of calendars for the specified provider."""
        adapter = self.get_adapter(provider)
        return adapter.get_calendars()

    def get_events(
        self,
        provider: str,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        use_cache: bool = True
    ) -> List[CalendarEvent]:
        """
        Get events for a calendar within a date range.
        Optionally use cached results for better performance.
        """
        adapter = self.get_adapter(provider)

        cache_key = f"{provider}_{calendar_id}_{start_date.date()}_{end_date.date()}"

        # Check if we have a recent cache and use_cache is True
        if use_cache and cache_key in self.events_cache:
            cache_time, events = self.events_cache[cache_key]

            # Use cache if it's less than 5 minutes old
            if (time.time() - cache_time) < 300:
                return events

        # Get fresh events from the provider
        events = adapter.get_events(calendar_id, start_date, end_date)

        # Update the cache
        self.events_cache[cache_key] = (time.time(), events)

        return events

    def create_event(
        self,
        provider: str,
        calendar_id: str,
        title: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        description: str = "",
        location: str = "",
        attendees: List[str] = None,
        is_all_day: bool = False
    ) -> CalendarEvent:
        """Create a new event in the specified calendar."""
        event = CalendarEvent(
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            attendees=attendees or [],
            calendar_id=calendar_id,
            provider=provider,
            is_all_day=is_all_day
        )

        adapter = self.get_adapter(provider)
        return adapter.create_event(calendar_id, event)

    def update_event(self, event: CalendarEvent) -> CalendarEvent:
        """Update an existing event."""
        if not event.provider or not event.calendar_id:
            raise ValueError("Event must have provider and calendar_id")

        adapter = self.get_adapter(event.provider)
        return adapter.update_event(event.calendar_id, event)

    def delete_event(self, provider: str, calendar_id: str, event_id: str) -> bool:
        """Delete an event from a calendar."""
        adapter = self.get_adapter(provider)
        return adapter.delete_event(calendar_id, event_id)

    def sync_events(
        self,
        provider: str,
        calendar_id: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        handle_added: callable = None,
        handle_updated: callable = None,
        handle_deleted: callable = None
    ) -> Dict[str, int]:
        """
        Sync events with a calendar provider, detecting changes.
        Calls the provided handlers when events are added, updated or deleted.

        Returns a summary of changes (added, updated, deleted counts).
        """
        adapter = self.get_adapter(provider)

        # Get the current events from the provider
        current_events = adapter.get_events(calendar_id, start_date, end_date)

        # Create dictionaries for efficient lookups
        current_events_dict = {event.id: event for event in current_events}

        # Check if we've synced before
        cache_key = f"{provider}_{calendar_id}"

        if cache_key in self.last_sync_time:
            last_sync_time, previous_events_dict = self.last_sync_time[cache_key]

            # Find events that were added or updated
            added_events = []
            updated_events = []

            for event_id, event in current_events_dict.items():
                if event_id not in previous_events_dict:
                    added_events.append(event)
                elif self._event_was_updated(event, previous_events_dict[event_id]):
                    updated_events.append(event)

            # Find events that were deleted
            deleted_event_ids = set(
                previous_events_dict.keys()) - set(current_events_dict.keys())
            deleted_events = [previous_events_dict[event_id]
                              for event_id in deleted_event_ids]

            # Call the handlers if provided
            if handle_added and added_events:
                for event in added_events:
                    handle_added(event)

            if handle_updated and updated_events:
                for event in updated_events:
                    handle_updated(event)

            if handle_deleted and deleted_events:
                for event in deleted_events:
                    handle_deleted(event)

            # Update the last sync information
            self.last_sync_time[cache_key] = (time.time(), current_events_dict)

            # Return a summary of changes
            return {
                "added": len(added_events),
                "updated": len(updated_events),
                "deleted": len(deleted_events),
                "total": len(current_events)
            }
        else:
            # First sync, just store the events
            self.last_sync_time[cache_key] = (time.time(), current_events_dict)

            # Call the handler for all events as "added"
            if handle_added:
                for event in current_events:
                    handle_added(event)

            # Return a summary
            return {
                "added": len(current_events),
                "updated": 0,
                "deleted": 0,
                "total": len(current_events)
            }

    def _event_was_updated(self, current_event: CalendarEvent, previous_event: CalendarEvent) -> bool:
        """Compare events to determine if one was updated."""
        # Simple comparison based on fields that matter for updates
        return (
            current_event.title != previous_event.title or
            current_event.description != previous_event.description or
            current_event.start_time != previous_event.start_time or
            current_event.end_time != previous_event.end_time or
            current_event.location != previous_event.location or
            set(current_event.attendees) != set(previous_event.attendees) or
            current_event.is_all_day != previous_event.is_all_day
        )

    def clear_cache(self):
        """Clear the events cache."""
        self.events_cache = {}

    def clear_sync_state(self):
        """Clear the sync state and start fresh."""
        self.last_sync_time = {}
