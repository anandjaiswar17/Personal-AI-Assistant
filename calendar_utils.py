"""
calendar_utils.py
─────────────────────────────────────────
Utility functions for Google Calendar API.
Handles creating events, reminders, and fetching upcoming schedule.
"""

from datetime import datetime, timedelta
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gmail_auth import get_gmail_credentials


def get_calendar_service():
    """Build and return an authenticated Google Calendar service client."""
    creds = get_gmail_credentials()
    return build("calendar", "v3", credentials=creds)


def create_event(
    title: str,
    start_time: str,          # ISO format: "2024-03-15T14:00:00"
    end_time: str,            # ISO format: "2024-03-15T15:00:00"
    description: str = "",
    attendee_email: str = "",
    reminder_minutes: int = 30,
    timezone: str = "Asia/Kolkata"
) -> Optional[dict]:
    """
    Create a Google Calendar event.

    Args:
        title: Event title
        start_time: Start time in ISO format
        end_time: End time in ISO format
        description: Event description / notes
        attendee_email: Optional attendee to invite
        reminder_minutes: Minutes before event to send reminder
        timezone: Timezone for the event

    Returns:
        Created event dict or None if failed
    """
    service = get_calendar_service()

    event_body = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_time,
            "timeZone": timezone,
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email",  "minutes": reminder_minutes},
                {"method": "popup",  "minutes": 10},
            ],
        },
    }

    # Add attendee if provided
    if attendee_email:
        event_body["attendees"] = [{"email": attendee_email}]

    try:
        event = service.events().insert(
            calendarId="primary",
            body=event_body,
            sendUpdates="all" if attendee_email else "none"
        ).execute()

        print(f"\n📅 Calendar event created!")
        print(f"   Title    : {title}")
        print(f"   Start    : {start_time}")
        print(f"   End      : {end_time}")
        print(f"   Link     : {event.get('htmlLink', 'N/A')}")

        return event

    except HttpError as e:
        print(f"❌ Failed to create calendar event: {e}")
        return None


def create_reminder(
    title: str,
    remind_at: str,           # ISO format datetime
    description: str = "",
    timezone: str = "Asia/Kolkata"
) -> Optional[dict]:
    """
    Create a reminder as a 15-minute calendar block.

    Args:
        title: Reminder title
        remind_at: When to remind in ISO format
        description: What to remember
        timezone: Timezone

    Returns:
        Created event dict or None
    """
    # Reminder = 15 min block
    start = datetime.fromisoformat(remind_at)
    end = start + timedelta(minutes=15)

    return create_event(
        title=f"🔔 Reminder: {title}",
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        description=description,
        reminder_minutes=0,   # Remind exactly at the time
        timezone=timezone
    )


def get_upcoming_events(days_ahead: int = 7, max_results: int = 10) -> list[dict]:
    """
    Fetch upcoming calendar events for the next N days.

    Returns:
        List of event dicts with title, start, end, description
    """
    service = get_calendar_service()

    now = datetime.utcnow().isoformat() + "Z"
    future = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + "Z"

    try:
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            timeMax=future,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        parsed = []

        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date", ""))
            end = e["end"].get("dateTime", e["end"].get("date", ""))
            parsed.append({
                "id": e["id"],
                "title": e.get("summary", "No Title"),
                "start": start,
                "end": end,
                "description": e.get("description", ""),
                "attendees": [
                    a["email"] for a in e.get("attendees", [])
                ],
            })

        return parsed

    except HttpError as e:
        print(f"❌ Failed to fetch calendar events: {e}")
        return []


def check_conflicts(start_time: str, end_time: str) -> list[dict]:
    """
    Check if a proposed time slot conflicts with existing events.

    Returns:
        List of conflicting events (empty = no conflicts)
    """
    service = get_calendar_service()

    try:
        body = {
            "timeMin": start_time if start_time.endswith("Z") else start_time + "Z",
            "timeMax": end_time if end_time.endswith("Z") else end_time + "Z",
            "items": [{"id": "primary"}]
        }
        result = service.freebusy().query(body=body).execute()
        busy_slots = result.get("calendars", {}).get("primary", {}).get("busy", [])
        return busy_slots

    except HttpError as e:
        print(f"❌ Could not check calendar conflicts: {e}")
        return []