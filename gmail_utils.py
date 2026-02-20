"""
gmail_utils.py
─────────────────────────────────────────
Utility functions for reading emails and saving drafts via Gmail API.
"""

import base64
import re
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gmail_auth import get_gmail_credentials


def get_gmail_service():
    """Build and return an authenticated Gmail service client."""
    creds = get_gmail_credentials()
    return build("gmail", "v1", credentials=creds)


def extract_email_body(payload: dict) -> str:
    """
    Recursively extract plain text body from email payload.
    Handles both simple and multipart emails.
    """
    body = ""

    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    break
            elif part["mimeType"].startswith("multipart"):
                # Recurse into nested multipart
                body = extract_email_body(part)
                if body:
                    break
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    return body.strip()


def clean_email_body(body: str) -> str:
    """
    Remove excessive whitespace, quoted reply chains, and email signatures.
    Keeps only the most relevant content for summarization.
    """
    # Remove excessive blank lines
    body = re.sub(r'\n{3,}', '\n\n', body)

    # Cut off common reply chain markers
    cut_markers = [
        "On ", "-----Original Message-----",
        "From:", "________________________________",
        "Sent from my iPhone", "Sent from my Galaxy",
        "Get Outlook for"
    ]
    for marker in cut_markers:
        idx = body.find(f"\n{marker}")
        if idx != -1:
            body = body[:idx]

    return body.strip()


def fetch_unread_emails(max_results: int = 10) -> list[dict]:
    """
    Fetch unread emails from Gmail inbox.

    Returns a list of email dicts with:
    - id, thread_id, sender, subject, body, snippet, date
    """
    service = get_gmail_service()
    emails = []

    try:
        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX", "UNREAD"],
            maxResults=max_results
        ).execute()

        messages = results.get("messages", [])

        if not messages:
            print("📭 No unread emails found.")
            return []

        print(f"📬 Found {len(messages)} unread email(s). Processing...")

        for msg_ref in messages:
            try:
                msg = service.users().messages().get(
                    userId="me",
                    id=msg_ref["id"],
                    format="full"
                ).execute()

                # Parse headers
                headers = {
                    h["name"]: h["value"]
                    for h in msg["payload"]["headers"]
                }

                # Extract and clean body
                raw_body = extract_email_body(msg["payload"])
                clean_body = clean_email_body(raw_body)

                emails.append({
                    "id": msg["id"],
                    "thread_id": msg["threadId"],
                    "sender": headers.get("From", "Unknown"),
                    "sender_email": extract_sender_email(headers.get("From", "")),
                    "subject": headers.get("Subject", "(No Subject)"),
                    "date": headers.get("Date", ""),
                    "body": clean_body or msg.get("snippet", ""),
                    "snippet": msg.get("snippet", ""),
                })

            except Exception as e:
                print(f"⚠️  Could not parse email {msg_ref['id']}: {e}")
                continue

    except HttpError as e:
        print(f"❌ Gmail API error: {e}")

    return emails


def extract_sender_email(from_header: str) -> str:
    """Extract just the email address from a 'From' header like 'John Doe <john@example.com>'"""
    match = re.search(r'<(.+?)>', from_header)
    if match:
        return match.group(1)
    # If no angle brackets, the whole thing is likely just the email
    return from_header.strip()


def save_draft(
    to: str,
    subject: str,
    body: str,
    thread_id: Optional[str] = None
) -> Optional[str]:
    """
    Save an email as a Gmail Draft (does NOT send it).

    Args:
        to: Recipient email address
        subject: Email subject
        body: Plain text email body
        thread_id: Optional thread ID to keep reply in the same thread

    Returns:
        Draft ID if successful, None otherwise
    """
    service = get_gmail_service()

    try:
        # Build the raw email message
        raw_message = (
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Content-Type: text/plain; charset=utf-8\n\n"
            f"{body}"
        )
        encoded = base64.urlsafe_b64encode(
            raw_message.encode("utf-8")
        ).decode("utf-8")

        draft_body = {"message": {"raw": encoded}}

        # Attach to existing thread if it's a reply
        if thread_id:
            draft_body["message"]["threadId"] = thread_id

        draft = service.users().drafts().create(
            userId="me",
            body=draft_body
        ).execute()

        return draft["id"]

    except HttpError as e:
        print(f"❌ Failed to save draft: {e}")
        return None


def mark_as_read(message_id: str):
    """Mark an email as read by removing the UNREAD label."""
    service = get_gmail_service()
    try:
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
    except HttpError as e:
        print(f"⚠️  Could not mark email as read: {e}")