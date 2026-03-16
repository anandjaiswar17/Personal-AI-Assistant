# 📧 AI Email Agent — LangGraph + Groq (Llama 3.3 70B)

An intelligent email assistant that reads your Gmail, summarizes each email,
and saves draft replies — **without ever sending anything automatically**.

---

## ✅ What It Does

| Step | Action |
|------|--------|
| 1 | Fetches unread emails from your Gmail inbox |
| 2 | Summarizes each email (sender intent, urgency, key points) |
| 3 | Decides if a reply is needed |
| 4 | Drafts a professional reply using Llama 3.3 70B via Groq |
| 5 | Saves the draft to **Gmail Drafts** (never auto-sends) |
| 6 | Prints a final digest of everything processed |

---

## 🗂 Project Structure

```
email_agent/
├── email_agent.py     ← Main LangGraph agent (run this)
├── gmail_auth.py      ← Google OAuth2 authentication
├── gmail_utils.py     ← Gmail API helper functions
├── requirements.txt   ← Python dependencies
├── .env.example       ← Environment variable template
├── credentials.json   ← (you add this) Google OAuth credentials
└── token.json         ← (auto-generated) Your Gmail auth token
```

---

## ⚙️ Setup Guide

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Get Your Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Create an API key
4. Copy it for the next step

### Step 3 — Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
GROQ_API_KEY=gsk_your_actual_key_here
YOUR_NAME=John Doe
YOUR_EMAIL=john@example.com
EMAIL_TONE=professional        # professional | casual | formal
MAX_EMAILS_TO_PROCESS=10
```

### Step 4 — Set Up Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (e.g., "Email Agent")
3. Go to **APIs & Services → Library**
4. Search for and enable **Gmail API**
5. Go to **APIs & Services → Credentials**
6. Click **Create Credentials → OAuth 2.0 Client ID**
7. Choose **Desktop App** as the application type
8. Download the JSON file and rename it to `credentials.json`
9. Place `credentials.json` in this folder

### Step 5 — Authenticate Gmail (One-Time Setup)

```bash
python gmail_auth.py
```

A browser window will open. Log in with your Google account and grant permission.
A `token.json` file will be saved — this is your auth token for future runs.

### Step 6 — Run the Agent

```bash
python email_agent.py
```

---

## 📊 Example Output

```
══════════════════════════════════════════════════
📬  EMAIL AGENT STARTING
══════════════════════════════════════════════════
📥 Fetching up to 10 unread emails...
✅ Fetched 3 email(s)

──────────────────────────────────────────────────
📧 Processing Email 1/3
   From:    John Smith <john@company.com>
   Subject: Q3 Budget Review Meeting
──────────────────────────────────────────────────

📋 SUMMARY:
SUMMARY: John is requesting a meeting to review Q3 budget...
ACTION_REQUIRED: YES
URGENCY: MEDIUM
REPLY_NEEDED: YES
REPLY_REASON: Meeting request requires confirmation

✍️  DRAFT REPLY:
Hi John,

Thank you for reaching out. I'd be happy to meet for the Q3 budget review.
[DATE] at [TIME] works for me — please confirm if that suits your schedule.

Looking forward to it.

Best,
Your Name

💾 Draft saved to Gmail!
   Draft ID: r123456789
   ⚠️  Review and send manually from Gmail Drafts.

══════════════════════════════════════════════════
📊  FINAL EMAIL DIGEST
══════════════════════════════════════════════════
  Total Emails Processed : 3
  Drafts Saved           : 2
  No Reply Needed        : 1
══════════════════════════════════════════════════
✅  Agent finished. Check Gmail Drafts to review replies.
```

---

## 🔧 Customization

### Change Email Tone
Set `EMAIL_TONE` in `.env` to:
- `professional` — Clear, business-appropriate
- `casual` — Friendly and conversational
- `formal` — Formal and polished

### Process More/Fewer Emails
Set `MAX_EMAILS_TO_PROCESS` in `.env` (default: 10)

### Add Your Writing Style
In `email_agent.py`, update the `system_prompt` in `draft_reply_node()` with examples of your past emails to match your personal style.

---

## 🔒 Privacy & Security

- `token.json` contains your Gmail access token — **never commit this to git**
- `credentials.json` contains your OAuth secrets — **never commit this to git**
- The agent only has **read** and **draft** permissions (no delete, no auto-send)
- Add both files to `.gitignore`

```
# .gitignore
token.json
credentials.json
.env
``` 
