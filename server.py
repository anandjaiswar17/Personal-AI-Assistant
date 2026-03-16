"""
server.py
─────────────────────────────────────────
FastAPI backend for the Email + Calendar AI Assistant web app.

ENDPOINTS:
  GET  /                     → Serve the frontend
  POST /api/run              → Run the email agent
  POST /api/confirm-calendar → Confirm and create calendar events
  GET  /api/status           → Check if agent is running
"""

import os
import json
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from gmail_utils import fetch_unread_emails, save_draft
from calendar_utils import create_event, create_reminder, check_conflicts
from gmail_auth import get_gmail_credentials

load_dotenv()

app = FastAPI(title="Email + Calendar AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
)

TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
DEFAULT_DURATION = int(os.getenv("DEFAULT_MEETING_DURATION_MINS", "60"))


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────
class RunAgentRequest(BaseModel):
    name: str
    email: str
    num_emails: int = 10
    email_type: str = "unread"   # "unread" or "latest"
    tone: str = "professional"

class CalendarConfirmRequest(BaseModel):
    action: str                  # "meeting" or "reminder"
    title: str
    date: str                    # YYYY-MM-DD
    time: str                    # HH:MM
    duration_minutes: int = 60
    description: str = ""
    attendee_email: str = ""
    email_id: str = ""


# ─────────────────────────────────────────
# HELPER: Analyze a single email
# ─────────────────────────────────────────
def analyze_email(email: dict, your_name: str, tone: str) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    body_preview = email["body"][:3000]

    system_prompt = """You are an expert executive assistant managing emails and schedules.
Analyze emails and extract communication needs and scheduling information.
Always respond with valid JSON only — no extra text, no markdown fences."""

    user_prompt = f"""Analyze this email and respond with JSON only.

TODAY: {today}
FROM: {email['sender']}
SUBJECT: {email['subject']}
DATE: {email['date']}
BODY:
{body_preview}

Respond ONLY with this JSON:
{{
  "summary": "2-3 sentence summary",
  "sender_intent": "What does the sender want?",
  "key_points": ["point 1", "point 2", "point 3"],
  "action_required": true,
  "urgency": "LOW",
  "reply_needed": true,
  "reply_reason": "Why a reply is or is not needed",
  "draft_reply": "Full professional email reply body signed as {your_name}. Use [PLACEHOLDER] for unknown details.",
  "calendar_action": "none",
  "calendar_details": {{
    "title": "",
    "date": "",
    "time": "",
    "duration_minutes": 60,
    "description": "",
    "attendee_email": ""
  }}
}}

calendar_action rules:
- "meeting" if email requests/confirms a meeting, call, or interview
- "reminder" if email mentions a deadline or follow-up with a date
- "none" if no scheduling needed

calendar_details rules:
- date: YYYY-MM-DD, empty string if unknown
- time: HH:MM 24hr format, empty string if unknown
- attendee_email: sender email for meetings, empty for reminders"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except Exception as e:
        data = {
            "summary": "Could not parse email summary.",
            "sender_intent": "",
            "key_points": [],
            "action_required": False,
            "urgency": "LOW",
            "reply_needed": False,
            "reply_reason": "Parse error",
            "draft_reply": "",
            "calendar_action": "none",
            "calendar_details": {}
        }

    return data


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    with open("index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/run")
async def run_agent(request: RunAgentRequest):
    """
    Main endpoint — fetch and analyze emails.
    Returns summaries, draft replies, and calendar suggestions.
    User must separately confirm calendar events.
    """
    try:
        # Fetch emails based on type
        if request.email_type == "latest":
            # Fetch latest emails regardless of read status
            from googleapiclient.discovery import build
            creds = get_gmail_credentials()
            service = build("gmail", "v1", credentials=creds)

            results = service.users().messages().list(
                userId="me",
                labelIds=["INBOX"],
                maxResults=request.num_emails
            ).execute()

            messages = results.get("messages", [])
            emails = []

            from gmail_utils import extract_email_body, clean_email_body, extract_sender_email
            import base64

            for msg_ref in messages:
                try:
                    msg = service.users().messages().get(
                        userId="me", id=msg_ref["id"], format="full"
                    ).execute()
                    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
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
                except Exception:
                    continue
        else:
            # Fetch unread emails
            emails = fetch_unread_emails(max_results=request.num_emails)

        if not emails:
            return JSONResponse(content={
                "success": True,
                "message": "No emails found.",
                "results": []
            })

        # Analyze each email
        results = []
        for email in emails:
            analysis = analyze_email(email, request.name, request.tone)

            # Save draft if reply needed
            draft_id = None
            if analysis.get("reply_needed") and analysis.get("draft_reply"):
                draft_id = save_draft(
                    to=email["sender_email"],
                    subject=f"Re: {email['subject']}",
                    body=analysis["draft_reply"],
                    thread_id=email["thread_id"]
                )

            # Check calendar conflicts if meeting detected
            conflict = False
            cal = analysis.get("calendar_details", {})
            cal_date = cal.get("date", "")
            cal_time = cal.get("time", "")
            if (
                analysis.get("calendar_action") == "meeting"
                and cal_date and "[" not in cal_date
                and cal_time and "[" not in cal_time
            ):
                try:
                    start_dt = f"{cal_date}T{cal_time}:00"
                    duration = cal.get("duration_minutes", DEFAULT_DURATION)
                    end_dt = (datetime.fromisoformat(start_dt) + timedelta(minutes=duration)).isoformat()
                    conflicts = check_conflicts(start_dt, end_dt)
                    conflict = len(conflicts) > 0
                except ValueError:
                    conflict = False

            results.append({
                "email_id": email["id"],
                "thread_id": email["thread_id"],
                "sender": email["sender"],
                "sender_email": email["sender_email"],
                "subject": email["subject"],
                "date": email["date"],
                "summary": analysis.get("summary", ""),
                "sender_intent": analysis.get("sender_intent", ""),
                "key_points": analysis.get("key_points", []),
                "urgency": analysis.get("urgency", "LOW"),
                "action_required": analysis.get("action_required", False),
                "reply_needed": analysis.get("reply_needed", False),
                "reply_reason": analysis.get("reply_reason", ""),
                "draft_reply": analysis.get("draft_reply", ""),
                "draft_id": draft_id,
                "calendar_action": analysis.get("calendar_action", "none"),
                "calendar_details": analysis.get("calendar_details", {}),
                "conflict_detected": conflict,
            })

        return JSONResponse(content={
            "success": True,
            "total": len(results),
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/confirm-calendar")
async def confirm_calendar(request: CalendarConfirmRequest):
    """
    Called when user clicks 'Confirm' on a calendar suggestion.
    Creates the actual Google Calendar event.
    """
    try:
        # ── Validate date ──
        if not request.date or "[" in request.date or not request.date.strip():
            raise HTTPException(
                status_code=400,
                detail="Please enter a valid date (YYYY-MM-DD) before confirming."
            )

        # ── Validate time ──
        if not request.time or "[" in request.time or not request.time.strip():
            raise HTTPException(
                status_code=400,
                detail="Please enter a valid time (HH:MM) before confirming."
            )

        # ── Validate date format ──
        try:
            datetime.strptime(request.date.strip(), "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format '{request.date}'. Please use YYYY-MM-DD (e.g. 2026-03-15)."
            )

        # ── Validate time format ──
        try:
            datetime.strptime(request.time.strip(), "%H:%M")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time format '{request.time}'. Please use HH:MM (e.g. 14:30)."
            )

        start_dt = f"{request.date.strip()}T{request.time.strip()}:00"
        end_dt = (
            datetime.fromisoformat(start_dt) +
            timedelta(minutes=request.duration_minutes)
        ).isoformat()

        if request.action == "meeting":
            event = create_event(
                title=request.title,
                start_time=start_dt,
                end_time=end_dt,
                description=request.description,
                attendee_email=request.attendee_email,
                reminder_minutes=30,
                timezone=TIMEZONE
            )
        elif request.action == "reminder":
            event = create_reminder(
                title=request.title,
                remind_at=start_dt,
                description=request.description,
                timezone=TIMEZONE
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid action type")

        if not event:
            raise HTTPException(status_code=500, detail="Failed to create calendar event")

        return JSONResponse(content={
            "success": True,
            "event_id": event.get("id"),
            "event_link": event.get("htmlLink", ""),
            "title": request.title,
            "start": start_dt,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/skip-calendar")
async def skip_calendar(data: dict):
    """User chose to skip a calendar event."""
    return JSONResponse(content={"success": True, "message": "Calendar event skipped."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)