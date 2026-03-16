"""
email_agent.py
─────────────────────────────────────────
LangGraph Email Assistant using Groq (llama-3.3-70b-versatile)

WHAT IT DOES:
  1. Fetches unread emails from Gmail
  2. Summarizes each email (sender, intent, urgency, action items)
  3. Decides if a reply is needed
  4. Drafts a professional reply if needed
  5. Saves the draft to Gmail Drafts (NEVER sends automatically)
  6. Prints a final digest of all processed emails

WHAT IT DOES NOT DO:
  ✗ Auto-send any email
  ✗ Delete or archive emails
  ✗ Access any other data

USAGE:
  python email_agent.py
"""

import os
from typing import TypedDict, Annotated, Optional
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from gmail_utils import fetch_unread_emails, save_draft

load_dotenv() 

# ─────────────────────────────────────────
# LLM Setup — Groq with Llama 3.3 70B
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,   # Slightly creative but mostly consistent
    max_tokens=1024,
)

# ─────────────────────────────────────────
# Config from .env
# ─────────────────────────────────────────
YOUR_NAME = os.getenv("YOUR_NAME", "Your Name")
YOUR_EMAIL = os.getenv("YOUR_EMAIL", "your@email.com")
EMAIL_TONE = os.getenv("EMAIL_TONE", "professional")
MAX_EMAILS = int(os.getenv("MAX_EMAILS_TO_PROCESS", "10"))


# ─────────────────────────────────────────
# State Definition
# Each email goes through its own state cycle
# ─────────────────────────────────────────
class EmailState(TypedDict):
    # Input
    emails: list[dict]                        # All fetched emails
    current_index: int                        # Which email we're processing now
    current_email: Optional[dict]             # Current email being processed

    # Processing results
    summary: str                              # LLM-generated summary
    needs_reply: bool                         # Does this email need a reply?
    reply_reason: str                         # Why it does/doesn't need a reply
    draft_body: str                           # Drafted reply text
    draft_id: Optional[str]                   # Gmail draft ID after saving

    # Accumulated results across all emails
    processed_results: Annotated[list, operator.add]  # All results combined


# ─────────────────────────────────────────
# NODE 1: Fetch Emails
# ─────────────────────────────────────────
def fetch_emails_node(state: EmailState) -> dict:
    """Fetch all unread emails from Gmail inbox."""
    print("\n" + "═" * 50)
    print("📬  EMAIL AGENT STARTING")
    print("═" * 50)
    print(f"📥 Fetching up to {MAX_EMAILS} unread emails...")

    emails = fetch_unread_emails(max_results=MAX_EMAILS)

    if not emails:
        print("✅ Inbox is clean — no unread emails!")
        return {
            "emails": [],
            "current_index": 0,
            "processed_results": []
        }

    print(f"✅ Fetched {len(emails)} email(s)\n")
    return {
        "emails": emails,
        "current_index": 0,
        "processed_results": []
    }


# ─────────────────────────────────────────
# NODE 2: Load Next Email
# ─────────────────────────────────────────
def load_next_email_node(state: EmailState) -> dict:
    """Load the next email to process from the list."""
    idx = state.get("current_index", 0)
    emails = state.get("emails", [])

    if idx >= len(emails):
        return {"current_email": None}

    email = emails[idx]
    print(f"\n{'─' * 50}")
    print(f"📧 Processing Email {idx + 1}/{len(emails)}")
    print(f"   From:    {email['sender']}")
    print(f"   Subject: {email['subject']}")
    print(f"   Date:    {email['date']}")
    print(f"{'─' * 50}")

    return {"current_email": email}


# ─────────────────────────────────────────
# NODE 3: Summarize Email
# ─────────────────────────────────────────
def summarize_email_node(state: EmailState) -> dict:
    """
    Use Groq (Llama 3.3 70B) to summarize the current email.
    Extracts: intent, urgency, action items, and whether a reply is needed.
    """
    email = state["current_email"]

    if not email:
        return {"summary": "", "needs_reply": False, "reply_reason": ""}

    # Truncate very long emails to avoid token limits
    body_preview = email["body"][:3000] if len(email["body"]) > 3000 else email["body"]

    system_prompt = """You are an expert executive assistant helping to manage emails efficiently.
Your job is to read emails and provide clear, actionable summaries.
Always respond in the exact structured format requested — no extra text."""

    user_prompt = f"""Analyze this email and provide a structured summary.

FROM: {email['sender']}
SUBJECT: {email['subject']}
DATE: {email['date']}
BODY:
{body_preview}

Respond ONLY in this exact format (no extra text):

SUMMARY: [2-3 sentence summary of what this email is about]
SENDER_INTENT: [What does the sender want or need?]
KEY_POINTS: [Bullet points of the most important information]
ACTION_REQUIRED: [YES or NO — does this email require a response or action?]
URGENCY: [LOW / MEDIUM / HIGH]
REPLY_NEEDED: [YES or NO]
REPLY_REASON: [Brief reason why a reply is or isn't needed]"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    summary = response.content.strip()

    # Parse whether a reply is needed from the structured response
    needs_reply = "REPLY_NEEDED: YES" in summary.upper()

    # Extract the reply reason
    reply_reason = ""
    for line in summary.split("\n"):
        if line.upper().startswith("REPLY_REASON:"):
            reply_reason = line.split(":", 1)[1].strip()
            break

    print(f"\n📋 SUMMARY:\n{summary}")

    return {
        "summary": summary,
        "needs_reply": needs_reply,
        "reply_reason": reply_reason,
    }


# ─────────────────────────────────────────
# NODE 4: Draft Reply
# ─────────────────────────────────────────
def draft_reply_node(state: EmailState) -> dict:
    """
    Use Groq (Llama 3.3 70B) to draft a professional email reply.
    Only runs if needs_reply is True.
    Uses placeholders where facts cannot be assumed.
    """
    email = state["current_email"]
    summary = state["summary"]

    tone_instructions = {
        "professional": "professional, clear, and concise",
        "casual": "friendly, warm, and conversational",
        "formal": "formal, polished, and respectful",
    }
    tone = tone_instructions.get(EMAIL_TONE, "professional, clear, and concise")

    system_prompt = f"""You are drafting email replies on behalf of {YOUR_NAME}.
Write in a {tone} tone.
Never invent facts, dates, or commitments.
Use [PLACEHOLDER] format when specific info is needed from {YOUR_NAME}.
Write only the email body — no subject line, no headers."""

    user_prompt = f"""Draft a reply to this email.

ORIGINAL EMAIL:
From: {email['sender']}
Subject: {email['subject']}

EMAIL SUMMARY:
{summary}

INSTRUCTIONS:
- Address the sender's core request or question directly
- Be {tone}
- Keep it focused and under 200 words unless the topic demands more
- Use [DATE], [TIME], [SPECIFIC DETAIL], etc. as placeholders where needed
- End with a clear next step or closing
- Sign off as: {YOUR_NAME}

Write the email body now:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    draft_body = response.content.strip()
    print(f"\n✍️  DRAFT REPLY:\n{draft_body}")

    return {"draft_body": draft_body}


# ─────────────────────────────────────────
# NODE 5: Save Draft to Gmail
# ─────────────────────────────────────────
def save_draft_node(state: EmailState) -> dict:
    """
    Save the drafted reply to Gmail Drafts folder.
    The email is NOT sent — it sits in Drafts for your review.
    """
    email = state["current_email"]
    draft_body = state.get("draft_body", "")

    if not draft_body:
        return {"draft_id": None}

    subject = f"Re: {email['subject']}"
    to = email["sender_email"]
    thread_id = email["thread_id"]

    draft_id = save_draft(
        to=to,
        subject=subject,
        body=draft_body,
        thread_id=thread_id
    )

    if draft_id:
        print(f"\n💾 Draft saved to Gmail!")
        print(f"   Draft ID: {draft_id}")
        print(f"   To: {to}")
        print(f"   Subject: {subject}")
        print(f"   ⚠️  Review and send manually from Gmail Drafts.")
    else:
        print("\n⚠️  Could not save draft to Gmail.")

    return {"draft_id": draft_id}


# ─────────────────────────────────────────
# NODE 6: Record Result & Advance
# ─────────────────────────────────────────
def record_result_node(state: EmailState) -> dict:
    """Store the result of the current email and advance the index."""
    email = state.get("current_email", {})
    idx = state.get("current_index", 0)

    result = {
        "index": idx + 1,
        "sender": email.get("sender", ""),
        "subject": email.get("subject", ""),
        "summary": state.get("summary", ""),
        "needs_reply": state.get("needs_reply", False),
        "reply_reason": state.get("reply_reason", ""),
        "draft_id": state.get("draft_id"),
    }

    return {
        "processed_results": [result],
        "current_index": idx + 1,
        "current_email": None,
        "summary": "",
        "needs_reply": False,
        "reply_reason": "",
        "draft_body": "",
        "draft_id": None,
    }


# ─────────────────────────────────────────
# NODE 7: Print Final Digest
# ─────────────────────────────────────────
def print_digest_node(state: EmailState) -> dict:
    """Print a clean summary digest of all processed emails."""
    results = state.get("processed_results", [])

    print("\n" + "═" * 50)
    print("📊  FINAL EMAIL DIGEST")
    print("═" * 50)

    if not results:
        print("No emails were processed.")
        return {}

    drafts_saved = [r for r in results if r.get("draft_id")]
    no_reply_needed = [r for r in results if not r.get("needs_reply")]

    print(f"  Total Emails Processed : {len(results)}")
    print(f"  Drafts Saved           : {len(drafts_saved)}")
    print(f"  No Reply Needed        : {len(no_reply_needed)}")

    print("\n📝 BREAKDOWN:")
    for r in results:
        status = "✅ Draft Saved" if r.get("draft_id") else (
            "⏭️  No Reply Needed" if not r["needs_reply"] else "⚠️  Draft Failed"
        )
        print(f"\n  [{r['index']}] {status}")
        print(f"      From    : {r['sender']}")
        print(f"      Subject : {r['subject']}")
        print(f"      Reason  : {r['reply_reason']}")
        if r.get("draft_id"):
            print(f"      Draft ID: {r['draft_id']}")

    print("\n" + "═" * 50)
    print("✅  Agent finished. Check Gmail Drafts to review replies.")
    print("═" * 50 + "\n")

    return {}


# ─────────────────────────────────────────
# ROUTING FUNCTIONS (Conditional Edges)
# ─────────────────────────────────────────
def route_after_fetch(state: EmailState) -> str:
    """After fetching, go to load_next if there are emails, else end."""
    if state.get("emails"):
        return "load_next"
    return "digest"


def route_after_load(state: EmailState) -> str:
    """After loading, if there's a current email, summarize it. Else digest."""
    if state.get("current_email"):
        return "summarize"
    return "digest"


def route_after_summarize(state: EmailState) -> str:
    """After summarizing, draft a reply if needed, else record result."""
    if state.get("needs_reply"):
        return "draft"
    return "record"


def route_after_draft(state: EmailState) -> str:
    """After drafting, always try to save it."""
    return "save_draft"


def route_after_record(state: EmailState) -> str:
    """After recording, check if there are more emails to process."""
    idx = state.get("current_index", 0)
    total = len(state.get("emails", []))
    if idx < total:
        return "load_next"
    return "digest"


# ─────────────────────────────────────────
# BUILD THE LANGGRAPH
# ─────────────────────────────────────────
def build_email_agent():
    graph = StateGraph(EmailState)

    # Add all nodes
    graph.add_node("fetch",         fetch_emails_node)
    graph.add_node("load_next",     load_next_email_node)
    graph.add_node("summarize",     summarize_email_node)
    graph.add_node("draft",         draft_reply_node)
    graph.add_node("save_draft",    save_draft_node)
    graph.add_node("record",        record_result_node)
    graph.add_node("digest",        print_digest_node)

    # Entry point
    graph.set_entry_point("fetch")

    # Edges with routing logic
    graph.add_conditional_edges("fetch", route_after_fetch, {
        "load_next": "load_next",
        "digest": "digest"
    })

    graph.add_conditional_edges("load_next", route_after_load, {
        "summarize": "summarize",
        "digest": "digest"
    })

    graph.add_conditional_edges("summarize", route_after_summarize, {
        "draft": "draft",
        "record": "record"
    })

    graph.add_edge("draft", "save_draft")
    graph.add_edge("save_draft", "record")

    graph.add_conditional_edges("record", route_after_record, {
        "load_next": "load_next",
        "digest": "digest"
    })

    graph.add_edge("digest", END)

    return graph.compile()


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    agent = build_email_agent()

    # Run the agent with empty initial state
    final_state = agent.invoke({
        "emails": [],
        "current_index": 0,
        "current_email": None,
        "summary": "",
        "needs_reply": False,
        "reply_reason": "",
        "draft_body": "",
        "draft_id": None,
        "processed_results": [],
    })
