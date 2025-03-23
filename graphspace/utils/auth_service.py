import os
import json
import time
import requests
import base64
import secrets
from typing import Dict, Optional, List, Tuple, Any
from urllib.parse import urlencode, quote
from flask import url_for, session, redirect, request


class OAuth2Provider:
    def __init__(
        self,
        name: str,
        client_id: str,
        client_secret: str,
        auth_url: str,
        token_url: str,
        redirect_uri: str,
        scopes: List[str]
    ):
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self.scopes = scopes


class OAuth2Service:
    def __init__(self):
        self.providers = {}

    def register_provider(self, provider: OAuth2Provider):
        self.providers[provider.name] = provider

    def get_provider(self, name: str) -> Optional[OAuth2Provider]:
        return self.providers.get(name)

    def get_authorization_url(self, provider_name: str) -> Tuple[str, str]:
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not registered")

        # Generate and store state parameter to prevent CSRF
        state = secrets.token_urlsafe(32)

        params = {
            'client_id': provider.client_id,
            'redirect_uri': provider.redirect_uri,
            'scope': ' '.join(provider.scopes),
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',  # To get refresh token
            'prompt': 'consent'  # Force consent screen to ensure refresh token
        }

        auth_url = f"{provider.auth_url}?{urlencode(params)}"
        return auth_url, state

    def exchange_code_for_token(self, provider_name: str, code: str) -> Dict[str, Any]:
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not registered")

        payload = {
            'client_id': provider.client_id,
            'client_secret': provider.client_secret,
            'code': code,
            'redirect_uri': provider.redirect_uri,
            'grant_type': 'authorization_code'
        }

        response = requests.post(provider.token_url, data=payload)
        if response.status_code != 200:
            raise Exception(
                f"Failed to exchange code for token: {response.text}")

        return response.json()

    def refresh_token(self, provider_name: str, refresh_token: str) -> Dict[str, Any]:
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not registered")

        payload = {
            'client_id': provider.client_id,
            'client_secret': provider.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }

        response = requests.post(provider.token_url, data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to refresh token: {response.text}")

        return response.json()

    def get_authenticated_header(self, provider_name: str, token_data: Dict[str, Any]) -> Dict[str, str]:
        if 'access_token' not in token_data:
            raise ValueError("Invalid token data")

        return {
            'Authorization': f"Bearer {token_data['access_token']}",
            'Content-Type': 'application/json'
        }

    def is_token_expired(self, token_data: Dict[str, Any]) -> bool:
        if 'expires_at' not in token_data:
            # If we don't have an expiry time, assume it's expired
            return True

        # Check if the token is expired, with 60-second buffer
        return token_data['expires_at'] - 60 < time.time()

    def update_token_expiry(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'expires_in' in token_data:
            token_data['expires_at'] = time.time(
            ) + int(token_data['expires_in'])
        return token_data


class OAuth2TokenStorage:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(
                os.path.dirname(__file__)), 'data', 'tokens')

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_token_path(self, provider: str, user_id: str) -> str:
        return os.path.join(self.storage_dir, f"{provider}_{user_id}.json")

    def save_token(self, provider: str, user_id: str, token_data: Dict[str, Any]) -> None:
        with open(self._get_token_path(provider, user_id), 'w') as f:
            json.dump(token_data, f)

    def load_token(self, provider: str, user_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_token_path(provider, user_id)
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def delete_token(self, provider: str, user_id: str) -> bool:
        path = self._get_token_path(provider, user_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

# Initialize common providers


def configure_oauth_service(base_url: str) -> OAuth2Service:
    service = OAuth2Service()

    # Google provider
    google_provider = OAuth2Provider(
        name="google",
        client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri=f"{base_url}/auth/google/callback",
        scopes=[
            "https://www.googleapis.com/auth/calendar",  # Calendar access
            "https://www.googleapis.com/auth/contacts.readonly",  # Contacts access
            "profile", "email"
        ]
    )
    service.register_provider(google_provider)

    # Microsoft provider
    microsoft_provider = OAuth2Provider(
        name="microsoft",
        client_id=os.environ.get("MICROSOFT_CLIENT_ID", ""),
        client_secret=os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        redirect_uri=f"{base_url}/auth/microsoft/callback",
        scopes=[
            "offline_access",
            "Calendars.ReadWrite",
            "Contacts.Read",
            "User.Read"
        ]
    )
    service.register_provider(microsoft_provider)

    # Apple provider (limited OAuth2 support, more complex in practice)
    apple_provider = OAuth2Provider(
        name="apple",
        client_id=os.environ.get("APPLE_CLIENT_ID", ""),
        client_secret=os.environ.get("APPLE_CLIENT_SECRET", ""),
        auth_url="https://appleid.apple.com/auth/authorize",
        token_url="https://appleid.apple.com/auth/token",
        redirect_uri=f"{base_url}/auth/apple/callback",
        scopes=["name", "email"]
    )
    service.register_provider(apple_provider)

    return service
