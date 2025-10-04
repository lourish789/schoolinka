# schoolinka

# 🚀 Render Deployment Checklist

Follow this checklist to ensure smooth deployment of your AI Coach Bot to Render.

## ✅ Pre-Deployment (Local Setup)

### 1. API Keys & Accounts Setup
- [ ] Created [Pinecone](https://www.pinecone.io/) account
- [ ] Got Pinecone API key
- [ ] Created [Google Gemini API](https://aistudio.google.com/) key  
- [ ] Created [Green API](https://green-api.com/) account
- [ ] Got Green API Instance ID and Token
- [ ] Have WhatsApp number connected to Green API

### 2. Google Sheets Setup
- [ ] Created Google Cloud Project
- [ ] Enabled Google Sheets API
- [ ] Created Service Account with Editor role
- [ ] Downloaded `credentials.json` file
- [ ] Created new Google Spreadsheet
- [ ] Copied Spreadsheet ID from URL
- [ ] Shared spreadsheet with service account email (Editor access)

### 3. Repository Setup
- [ ] Forked/cloned the repository
- [ ] Added `credentials.json` to root folder
- [ ] Created `data/` folder
- [ ] Added PDF teaching resources to `data/`
- [ ] Verified all files are present:
  - `ai_coach_bot.py`
  - `vector_storage.py`
  - `requirements.txt`
  - `render.yaml`
  - `.env.example`
  - `credentials.json`
  - `runtime.txt`
  - `.gitignore`
  - `README.md`

### 4. Local Vector Storage Setup
- [ ] Installed Python 3.9+ locally
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Updated API keys in `vector_storage.py` (lines 12-13)
- [ ] Ran `python vector_storage.py` successfully
- [ ] Verified Pinecone index created in dashboard
- [ ] Confirmed vectors uploaded (check count in Pinecone)

**Expected Output:**
```
✅ Total vectors in index: 245
✅ Index dimension: 768
✅ VECTOR STORAGE COMPLETE!
```

### 5. Git Commit (Optional but Recommended)
```bash
git add .
git commit -m "Initial setup with credentials and PDF resources"
git push origin main
```

**Note:** Ensure `credentials.json` is in `.gitignore` if making repo public!

---

## 🌐 Render Deployment

### 1. Create Web Service
- [ ] Logged into [Render Dashboard](https://dashboard.render.com/)
- [ ] Clicked "New +" → "Web Service"
- [ ] Connected GitHub repository
- [ ] Selected correct repository and branch (`main`)

### 2. Configure Service
- [ ] **Name:** Set to `ai-coach-bot` (or custom name)
- [ ] **Region:** Selected `Oregon (US West)`
- [ ] **Branch:** `main`
- [ ] **Runtime:** Auto-detected as `Python 3`
- [ ] **Build Command:** `pip install -r requirements.txt`
- [ ] **Start Command:** `gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 ai_coach_bot:app`
- [ ] **Plan:** Selected `Free` (or paid plan)

### 3. Environment Variables Setup
Click "Advanced" → Add all these environment variables:

- [ ] `PINECONE_API_KEY` = `your_actual_pinecone_key`
- [ ] `GOOGLE_API_KEY` = `your_actual_google_key`
- [ ] `GREEN_API_ID_INSTANCE` = `your_instance_id`
- [ ] `GREEN_API_TOKEN` = `your_green_api_token`
- [ ] `SPREADSHEET_ID` = `your_spreadsheet_id`

**Double-check:** No extra spaces, quotes, or line breaks in values!

### 4. Deploy
- [ ] Clicked "Create Web Service"
- [ ] Waited for deployment (5-10 minutes)
- [ ] Checked logs for errors
- [ ] Deployment status shows "Live" with green indicator
- [ ] Noted the public URL: `https://your-app-name.onrender.com`

**Expected Log Output:**
```
Starting AI Coach - Schoolinka
Pinecone Index: Connected
Google Sheets: Connected
========================================
 * Running on http://0.0.0.0:5000
```

---

## 🔗 WhatsApp Integration

### 1. Configure Green API Webhook
- [ ] Logged into [Green API Console](https://console.green-api.com/)
- [ ] Selected your instance
- [ ] Went to Settings → Webhooks
- [ ] Set Webhook URL: `https://your-app-name.onrender.com/webhook`
- [ ] Enabled **only** "Incoming Message Received"
- [ ] Saved settings
- [ ] Verified webhook URL is correct (no typos!)

### 2. Test Webhook Connection
```bash
# From terminal, test health endpoint
curl https://your-app-name.onrender.com/

# Expected response:
{
  "status": "healthy",
  "service": "AI Coach - Schoolinka",
  "pinecone_connected": true,
  "sheets_connected": true
}
```

- [ ] Health check returns 200 OK
- [ ] `pinecone_connected: true`
- [ ] `sheets_connected: true`

---

## 🧪 Testing

### 1. Test Registration Flow
Send to WhatsApp bot:
```
Name: Test User
Email: test@email.com
Phone: 08012345678
Location: Lagos
Class: Primary 4
```

- [ ] Received welcome message
- [ ] User appears in Google Sheets "Users" tab
- [ ] Timestamp is correct

### 2. Test Teaching Question
Send to WhatsApp bot:
```
How do I teach multiplication to young learners?
```

- [ ] Received relevant response within 10 seconds
- [ ] Response is concise (2-4 sentences)
- [ ] No asterisks or formatting issues
- [ ] Conversation logged in Google Sheets "Conversations" tab

### 3. Test Context Awareness
Send follow-up:
```
What about visual aids?
```

- [ ] Bot remembers previous context (multiplication teaching)
- [ ] Response is relevant to previous question
- [ ] Conversation flows naturally

### 4. Test Edge Cases
- [ ] Send very long message (500+ characters) - should respond
- [ ] Send gibberish - should give polite error
- [ ] Send incomplete registration - should prompt for correct format
- [ ] Send voice note - should respond: "I can only respond to text messages"
- [ ] Send image - should respond: "I can only respond to text messages"

### 5. Verify Logging
Check Google Sheets:
- [ ] **Users sheet:** Has test user with all details
- [ ] **Conversations sheet:** Shows all test messages
- [ ] Timestamps are accurate
- [ ] Intent classification is working
- [ ] No duplicate entries

---

## 📊 Monitoring Setup

### 1. Render Dashboard
- [ ] Enabled email notifications for errors
- [ ] Bookmarked logs page
- [ ] Set up uptime monitoring (optional)

### 2. Check Initial Stats
```bash
curl https://your-app-name.onrender.com/stats
```

- [ ] Returns user count
- [ ] Returns message count
- [ ] Shows connected services

### 3. Set Up Alerts (Optional)
- [ ] Set up UptimeRobot or similar service
- [ ] Configure ping every 5 minutes
- [ ] Add alert email/SMS

---

## 🎉 Post-Deployment

### 1. Documentation
- [ ] Updated README with your actual Render URL
- [ ] Documented any custom configurations
- [ ] Created internal usage guide for teachers

### 2. Share with Team
- [ ] Sent WhatsApp number to test users
- [ ] Created user guide with registration format
- [ ] Set up support channel for questions

### 3. Monitor First 24 Hours
- [ ] Check Render logs every few hours
- [ ] Monitor Google Sheets for user registrations
- [ ] Review conversation quality
- [ ] Note any error patterns

### 4. Performance Baseline
Record initial metrics:
- [ ] Average response time: _____ seconds
- [ ] First registration completed: _____ (timestamp)
- [ ] Total users after 24h: _____
- [ ] Total messages after 24h: _____

---

## 🔧 Common Issues & Fixes

| Issue | Check | Fix |
|-------|-------|-----|
| "Pinecone index not found" | Did you run vector_storage.py? | Run locally first |
| "Sheets permission denied" | Service account email shared? | Re-share with Editor access |
| "Webhook not receiving" | Webhook URL correct? | Double-check URL in Green API |
| "Slow responses" | Free tier cold start? | Wait 30s or upgrade plan |
| "Empty responses" | Vectors in Pinecone? | Check dashboard, re-run storage |

---

## 📞 Emergency Contacts

- **Render Support:** support@render.com
- **Green API Support:** support@green-api.com
- **Pinecone Support:** support@pinecone.io
- **Your Team:** _____________

---

## ✨ Success Criteria

Your deployment is successful when:
- ✅ Health endpoint returns 200 with all services connected
- ✅ Registration flow works end-to-end
- ✅ Bot responds to teaching questions within 10 seconds
- ✅ Responses are relevant and concise
- ✅ Google Sheets logs all activity correctly
- ✅ No errors in Render logs for 24
