"""
email_agent.py
─────────────────────────────────────────
LangGraph Email + Calendar Assistant using Groq (llama-3.3-70b-versatile)

WHAT IT DOES:
  1. Fetches unread emails from Gmail
  2. Summarizes each email (sender, intent, urgency, action items)
  3. Detects if a meeting/event/reminder is mentioned
  4. Creates Google Calendar events or reminders automatically
  5. Checks for scheduling conflicts before creating events
  6. Decides if a draft reply is needed
  7. Drafts a professional reply if needed
  8. Saves the draft to Gmail Drafts (NEVER sends automatically)
  9. Prints a final digest of all processed emails + calendar actions

WHAT IT DOES NOT DO:
  ✗ Auto-send any email
  ✗ Delete or archive emails
  ✗ Create events without extracting real info from the email

USAGE:
  python email_agent.py
"""

import os
import json
from typing import TypedDict, Annotated, Optional
import operator
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from gmail_utils import fetch_unread_emails, save_draft
from calendar_utils import create_event, create_reminder, get_upcoming_events, check_conflicts

load_dotenv()

# ─────────────────────────────────────────
# LLM Setup — Groq with Llama 3.3 70B
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
)

# ─────────────────────────────────────────
# Config from .env
# ─────────────────────────────────────────
YOUR_NAME                = os.getenv("YOUR_NAME", "Your Name")
YOUR_EMAIL               = os.getenv("YOUR_EMAIL", "your@email.com")
EMAIL_TONE               = os.getenv("EMAIL_TONE", "professional")
MAX_EMAILS               = int(os.getenv("MAX_EMAILS_TO_PROCESS", "10"))
TIMEZONE                 = os.getenv("TIMEZONE", "Asia/Kolkata")
DEFAULT_MEETING_DURATION = int(os.getenv("DEFAULT_MEETING_DURATION_MINS", "60"))


# ─────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────
class AgentState(TypedDict):
    # Email list
    emails: list[dict]
    current_index: int
    current_email: Optional[dict]

    # Email processing
    summary: str
    needs_reply: bool
    reply_reason: str
    draft_body: str
    draft_id: Optional[str]

    # Calendar processing
    calendar_action: str           # "meeting" | "reminder" | "none"
    calendar_details: dict         # Extracted event details from LLM
    calendar_event_id: Optional[str]
    conflict_detected: bool

    # Accumulated results
    processed_results: Annotated[list, operator.add]


# ─────────────────────────────────────────
# NODE 1: Fetch Emails
# ─────────────────────────────────────────
def fetch_emails_node(state: AgentState) -> dict:
    """Fetch all unread emails from Gmail and show upcoming schedule."""
    print("\n" + "═" * 55)
    print("🤖  EMAIL + CALENDAR AGENT STARTING")
    print("═" * 55)

    # Show upcoming schedule as context
    print("\n📅 Your upcoming schedule (next 7 days):")
    upcoming = get_upcoming_events(days_ahead=7, max_results=5)
    if upcoming:
        for e in upcoming:
            print(f"   • {e['title']} → {e['start']}")
    else:
        print("   No upcoming events found.")

    print(f"\n📥 Fetching up to {MAX_EMAILS} unread emails...")
    emails = fetch_unread_emails(max_results=MAX_EMAILS)

    if not emails:
        print("✅ Inbox is clean — no unread emails!")
        return {"emails": [], "current_index": 0, "processed_results": []}

    print(f"✅ Fetched {len(emails)} email(s)\n")
    return {
        "emails": emails,
        "current_index": 0,
        "processed_results": []
    }


# ─────────────────────────────────────────
# NODE 2: Load Next Email
# ─────────────────────────────────────────
def load_next_email_node(state: AgentState) -> dict:
    """Load the next email from the list."""
    idx = state.get("current_index", 0)
    emails = state.get("emails", [])

    if idx >= len(emails):
        return {"current_email": None}

    email = emails[idx]
    print(f"\n{'─' * 55}")
    print(f"📧 Processing Email {idx + 1}/{len(emails)}")
    print(f"   From    : {email['sender']}")
    print(f"   Subject : {email['subject']}")
    print(f"   Date    : {email['date']}")
    print(f"{'─' * 55}")

    return {"current_email": email}


# ─────────────────────────────────────────
# NODE 3: Summarize + Detect Calendar Intent
# ─────────────────────────────────────────
def summarize_email_node(state: AgentState) -> dict:
    """
    Summarize the email AND detect if it mentions a meeting,
    event, or reminder that should be added to Google Calendar.
    """
    email = state["current_email"]
    if not email:
        return {
            "summary": "", "needs_reply": False,
            "reply_reason": "", "calendar_action": "none",
            "calendar_details": {}
        }

    today = datetime.now().strftime("%Y-%m-%d")
    body_preview = email["body"][:3000]

    system_prompt = """You are an expert executive assistant managing emails and schedules.
Analyze emails carefully and extract both communication needs and scheduling information.
Always respond with valid JSON only — no extra text, no markdown."""

    user_prompt = f"""Analyze this email and respond with a JSON object only.

TODAY'S DATE: {today}
FROM: {email['sender']}
SUBJECT: {email['subject']}
DATE RECEIVED: {email['date']}
BODY:
{body_preview}

Respond ONLY with this JSON:
{{
  "summary": "2-3 sentence summary",
  "sender_intent": "What does the sender want?",
  "key_points": ["point 1", "point 2"],
  "action_required": true,
  "urgency": "LOW",
  "reply_needed": true,
  "reply_reason": "Why a reply is or is not needed",
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

Rules for calendar_action:
- Use "meeting" if the email requests or confirms a meeting/call/interview
- Use "reminder" if the email mentions a deadline, follow-up, or task with a date
- Use "none" if no scheduling is needed

Rules for calendar_details:
- date: YYYY-MM-DD format, empty string if unknown
- time: HH:MM in 24hr format, empty string if unknown
- attendee_email: sender's email for meetings, empty for reminders"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # Parse JSON response safely
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except Exception as e:
        print(f"⚠️  JSON parse error: {e}")
        data = {
            "summary": response.content[:300],
            "reply_needed": False,
            "reply_reason": "Parse error",
            "calendar_action": "none",
            "calendar_details": {}
        }

    summary_text = f"""SUMMARY: {data.get('summary', '')}
SENDER_INTENT: {data.get('sender_intent', '')}
KEY_POINTS: {', '.join(data.get('key_points', []))}
ACTION_REQUIRED: {'YES' if data.get('action_required') else 'NO'}
URGENCY: {data.get('urgency', 'LOW')}
REPLY_NEEDED: {'YES' if data.get('reply_needed') else 'NO'}
REPLY_REASON: {data.get('reply_reason', '')}
CALENDAR_ACTION: {data.get('calendar_action', 'none').upper()}"""

    print(f"\n📋 ANALYSIS:\n{summary_text}")

    cal_action = data.get("calendar_action", "none").lower()
    if cal_action not in ["meeting", "reminder"]:
        cal_action = "none"

    return {
        "summary": summary_text,
        "needs_reply": data.get("reply_needed", False),
        "reply_reason": data.get("reply_reason", ""),
        "calendar_action": cal_action,
        "calendar_details": data.get("calendar_details", {}),
    }


# ─────────────────────────────────────────
# NODE 4: Handle Calendar Action
# ─────────────────────────────────────────
def calendar_action_node(state: AgentState) -> dict:
    """
    Create a Google Calendar event or reminder.
    Checks for conflicts before creating meetings.
    """
    action  = state.get("calendar_action", "none")
    details = state.get("calendar_details", {})
    email   = state["current_email"]

    if action == "none" or not details:
        return {"calendar_event_id": None, "conflict_detected": False}

    # Extract details
    date_str    = details.get("date", "")
    time_str    = details.get("time", "")
    title       = details.get("title", "") or email["subject"]
    description = details.get("description", "") or f"From: {email['sender']}\nSubject: {email['subject']}"
    duration    = details.get("duration_minutes", DEFAULT_MEETING_DURATION)
    attendee    = details.get("attendee_email", "")

    # Default to tomorrow 10am if date/time not found
    if not date_str or not time_str:
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        time_str = "10:00"
        print(f"\n⚠️  No specific date/time in email. Defaulting to tomorrow at 10:00 AM.")
        print(f"   Please review and update the calendar event if needed.")

    start_dt = f"{date_str}T{time_str}:00"
    end_dt   = (datetime.fromisoformat(start_dt) + timedelta(minutes=duration)).isoformat()

    print(f"\n📅 Creating {action.upper()} on Calendar:")
    print(f"   Title    : {title}")
    print(f"   Start    : {start_dt}")
    print(f"   Duration : {duration} minutes")
    if attendee:
        print(f"   Attendee : {attendee}")

    event = None
    conflict_detected = False

    if action == "meeting":
        # Conflict check
        conflicts = check_conflicts(start_dt, end_dt)
        if conflicts:
            conflict_detected = True
            print(f"\n⚠️  CONFLICT: You already have {len(conflicts)} event(s) at this time!")

        event = create_event(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            description=description,
            attendee_email=attendee,
            reminder_minutes=30,
            timezone=TIMEZONE
        )

    elif action == "reminder":
        event = create_reminder(
            title=title,
            remind_at=start_dt,
            description=description,
            timezone=TIMEZONE
        )

    return {
        "calendar_event_id": event.get("id") if event else None,
        "conflict_detected": conflict_detected
    }


# ─────────────────────────────────────────
# NODE 5: Draft Reply
# ─────────────────────────────────────────
def draft_reply_node(state: AgentState) -> dict:
    """Draft a professional email reply using Groq."""
    email           = state["current_email"]
    summary         = state["summary"]
    calendar_action = state.get("calendar_action", "none")
    event_id        = state.get("calendar_event_id")

    tone_map = {
        "professional": "professional, clear, and concise",
        "casual":       "friendly, warm, and conversational",
        "formal":       "formal, polished, and respectful",
    }
    tone = tone_map.get(EMAIL_TONE, "professional, clear, and concise")

    calendar_note = ""
    if event_id and calendar_action == "meeting":
        calendar_note = "Note: A calendar invite has been created for this meeting."
    elif event_id and calendar_action == "reminder":
        calendar_note = "Note: A reminder has been set in the calendar."

    system_prompt = f"""You are drafting email replies on behalf of {YOUR_NAME}.
Tone: {tone}.
Never invent facts or commitments. Use [PLACEHOLDER] where specific info is needed.
Write only the email body — no subject line."""

    user_prompt = f"""Draft a reply to this email.

FROM: {email['sender']}
SUBJECT: {email['subject']}

ANALYSIS:
{summary}

{calendar_note}

INSTRUCTIONS:
- Address the sender's request directly
- If a meeting was scheduled, mention it naturally
- Keep under 200 words unless necessary
- Use [DATE], [TIME], [DETAIL] as placeholders where needed
- Sign off as: {YOUR_NAME}

Write the email body:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    draft_body = response.content.strip()
    print(f"\n✍️  DRAFT REPLY:\n{draft_body}")

    return {"draft_body": draft_body}


# ─────────────────────────────────────────
# NODE 6: Save Draft to Gmail
# ─────────────────────────────────────────
def save_draft_node(state: AgentState) -> dict:
    """Save the drafted reply to Gmail Drafts. Does NOT send."""
    email      = state["current_email"]
    draft_body = state.get("draft_body", "")

    if not draft_body:
        return {"draft_id": None}

    draft_id = save_draft(
        to=email["sender_email"],
        subject=f"Re: {email['subject']}",
        body=draft_body,
        thread_id=email["thread_id"]
    )

    if draft_id:
        print(f"\n💾 Draft saved to Gmail Drafts!")
        print(f"   To: {email['sender_email']}")
        print(f"   ⚠️  Review and send manually.")
    else:
        print("\n⚠️  Could not save draft.")

    return {"draft_id": draft_id}


# ─────────────────────────────────────────
# NODE 7: Record Result & Advance
# ─────────────────────────────────────────
def record_result_node(state: AgentState) -> dict:
    """Store result of current email and advance index."""
    email = state.get("current_email", {})
    idx   = state.get("current_index", 0)

    result = {
        "index":            idx + 1,
        "sender":           email.get("sender", ""),
        "subject":          email.get("subject", ""),
        "needs_reply":      state.get("needs_reply", False),
        "reply_reason":     state.get("reply_reason", ""),
        "draft_id":         state.get("draft_id"),
        "calendar_action":  state.get("calendar_action", "none"),
        "calendar_event_id":state.get("calendar_event_id"),
        "conflict_detected":state.get("conflict_detected", False),
    }

    return {
        "processed_results":  [result],
        "current_index":      idx + 1,
        "current_email":      None,
        "summary":            "",
        "needs_reply":        False,
        "reply_reason":       "",
        "draft_body":         "",
        "draft_id":           None,
        "calendar_action":    "none",
        "calendar_details":   {},
        "calendar_event_id":  None,
        "conflict_detected":  False,
    }


# ─────────────────────────────────────────
# NODE 8: Final Digest
# ─────────────────────────────────────────
def print_digest_node(state: AgentState) -> dict:
    """Print a clean summary of everything the agent did."""
    results = state.get("processed_results", [])

    print("\n" + "═" * 55)
    print("📊  FINAL DIGEST")
    print("═" * 55)

    if not results:
        print("No emails were processed.")
        return {}

    drafts_saved   = [r for r in results if r.get("draft_id")]
    events_created = [r for r in results if r.get("calendar_event_id")]
    conflicts      = [r for r in results if r.get("conflict_detected")]

    print(f"  Emails Processed : {len(results)}")
    print(f"  Drafts Saved     : {len(drafts_saved)}")
    print(f"  Calendar Events  : {len(events_created)}")
    print(f"  Conflicts Found  : {len(conflicts)}")

    print("\n📝 BREAKDOWN:")
    for r in results:
        email_status = "✅ Draft Saved" if r.get("draft_id") else (
            "⏭️  No Reply Needed" if not r["needs_reply"] else "⚠️  Draft Failed"
        )
        cal_status = ""
        if r.get("calendar_event_id"):
            action   = r.get("calendar_action", "event").capitalize()
            conflict = " ⚠️ CONFLICT" if r.get("conflict_detected") else ""
            cal_status = f" | 📅 {action} Created{conflict}"

        print(f"\n  [{r['index']}] {email_status}{cal_status}")
        print(f"      From    : {r['sender']}")
        print(f"      Subject : {r['subject']}")
        if r.get("reply_reason"):
            print(f"      Reason  : {r['reply_reason']}")

    print("\n" + "═" * 55)
    print("✅  Done! Check Gmail Drafts + Google Calendar.")
    print("═" * 55 + "\n")

    return {}


# ─────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────
def route_after_fetch(state: AgentState) -> str:
    return "load_next" if state.get("emails") else "digest"

def route_after_load(state: AgentState) -> str:
    return "summarize" if state.get("current_email") else "digest"

def route_after_summarize(state: AgentState) -> str:
    if state.get("calendar_action", "none") != "none":
        return "calendar"
    if state.get("needs_reply"):
        return "draft"
    return "record"

def route_after_calendar(state: AgentState) -> str:
    return "draft" if state.get("needs_reply") else "record"

def route_after_record(state: AgentState) -> str:
    idx   = state.get("current_index", 0)
    total = len(state.get("emails", []))
    return "load_next" if idx < total else "digest"


# ─────────────────────────────────────────
# BUILD THE LANGGRAPH
# ─────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("fetch",      fetch_emails_node)
    graph.add_node("load_next",  load_next_email_node)
    graph.add_node("summarize",  summarize_email_node)
    graph.add_node("calendar",   calendar_action_node)
    graph.add_node("draft",      draft_reply_node)
    graph.add_node("save_draft", save_draft_node)
    graph.add_node("record",     record_result_node)
    graph.add_node("digest",     print_digest_node)

    graph.set_entry_point("fetch")

    graph.add_conditional_edges("fetch",     route_after_fetch,     {"load_next": "load_next", "digest": "digest"})
    graph.add_conditional_edges("load_next", route_after_load,      {"summarize": "summarize", "digest": "digest"})
    graph.add_conditional_edges("summarize", route_after_summarize, {"calendar": "calendar", "draft": "draft", "record": "record"})
    graph.add_conditional_edges("calendar",  route_after_calendar,  {"draft": "draft", "record": "record"})
    graph.add_edge("draft",      "save_draft")
    graph.add_edge("save_draft", "record")
    graph.add_conditional_edges("record",    route_after_record,    {"load_next": "load_next", "digest": "digest"})
    graph.add_edge("digest", END)

    return graph.compile()


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent()

    final_state = agent.invoke({
        "emails":            [],
        "current_index":     0,
        "current_email":     None,
        "summary":           "",
        "needs_reply":       False,
        "reply_reason":      "",
        "draft_body":        "",
        "draft_id":          None,
        "calendar_action":   "none",
        "calendar_details":  {},
        "calendar_event_id": None,
        "conflict_detected": False,
        "processed_results": [],
    })