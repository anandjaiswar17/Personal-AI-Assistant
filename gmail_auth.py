"""
gmail_auth.py
─────────────────────────────────────────
Handles Google OAuth2 authentication for Gmail API + Google Calendar API.

SETUP STEPS:
1. Go to https://console.cloud.google.com
2. Create a new project
3. Enable Gmail API AND Google Calendar API
4. Create OAuth 2.0 credentials (Desktop App)
5. Download as credentials.json and place in this folder
6. Run this file once: python gmail_auth.py
   → It will open a browser to authorize your account
   → A token.json will be saved for future use
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

load_dotenv()

# Scopes needed for Gmail + Google Calendar
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",      # Read emails
    "https://www.googleapis.com/auth/gmail.compose",       # Create drafts
    "https://www.googleapis.com/auth/gmail.modify",        # Modify labels
    "https://www.googleapis.com/auth/calendar",            # Full calendar access
    "https://www.googleapis.com/auth/calendar.events",     # Create/edit events
]

CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")


def get_gmail_credentials() -> Credentials:
    """
    Returns valid Gmail credentials.
    - Uses saved token.json if it exists and is valid
    - Refreshes token if expired
    - Runs OAuth flow if no token exists
    """
    creds = None

    # Load existing token
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Refresh or re-authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"\n❌ '{CREDENTIALS_FILE}' not found!\n"
                    "Please download OAuth credentials from Google Cloud Console.\n"
                    "See: https://console.cloud.google.com/apis/credentials"
                )
            print("🔐 Starting OAuth flow... (browser will open)")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for next run
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"✅ Token saved to {TOKEN_FILE}")

    return creds


if __name__ == "__main__":
    """Run this file directly to authenticate for the first time."""
    print("🚀 Gmail Authentication Setup")
    print("─" * 40)
    creds = get_gmail_credentials()
    print("\n✅ Authentication successful!")
    print(f"   Token saved to: {TOKEN_FILE}")
    print("\nYou can now run the email agent.")