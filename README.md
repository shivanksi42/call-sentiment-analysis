# 🎧 AI-Powered Call Quality Auditor

An AI-driven solution for automated call quality auditing and customer sentiment analysis. This POC uses **OpenAI Whisper** for speech-to-text transcription and **GPT-3.5 Turbo** for sentiment analysis and quality scoring.

## ✨ Features

- **Audio Transcription**: Upload call recordings and get accurate transcriptions using OpenAI Whisper API
- **Sentiment Analysis**: Analyze customer emotions, urgency levels, and escalation risk
- **Agent Scoring**: Questionnaire-based evaluation of agent performance across multiple categories
- **Compliance Checking**: Automated fraud detection and compliance risk assessment
- **Cloud Storage**: Audio recordings stored in Supabase Storage (auto-deleted after 3 days)
- **Interactive Dashboard**: Beautiful visualizations with Plotly charts
  - Sentiment distribution pie chart
  - Urgency level donut chart
  - Agent performance bar chart
  - Daily trends line chart
  - Category score breakdown
  - Escalation risk histogram

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **Transcription**: OpenAI Whisper API
- **AI Analysis**: GPT-3.5 Turbo
- **Charts**: Plotly.js
- **Database**: PostgreSQL (Supabase)
- **Storage**: Supabase Storage (for audio files)
- **Deployment**: Vercel (Serverless)
- **Frontend**: HTML/CSS/JavaScript with Jinja2 templates

## 📋 Prerequisites

- Python 3.11+
- OpenAI API Key
- Supabase Account (free tier works)
- Vercel Account (for deployment)

## 🚀 Quick Start (Local Development)

1. **Clone and navigate to the POC directory:**
   ```bash
   cd "Ai Sentiment Anaalysis/poc"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp sample.env .env
   # Edit .env and add your API keys
   ```

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8000`

---

## 🌐 Deploy to Vercel

### Step 1: Set Up Supabase

1. **Create a Supabase Project** at [supabase.com](https://supabase.com)

2. **Get your credentials** from Project Settings > API:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`

3. **Get your database URL** from Project Settings > Database:
   - Use the "URI" connection string

4. **Create a Storage Bucket** (optional - the app creates it automatically):
   - Go to Storage > Create bucket
   - Name: `call-recordings`
   - Make it private

### Step 2: Deploy to Vercel

1. **Install Vercel CLI** (optional):
   ```bash
   npm i -g vercel
   ```

2. **Push to GitHub** and connect to Vercel, or deploy via CLI:
   ```bash
   cd poc
   vercel
   ```

3. **Configure Environment Variables** in Vercel Dashboard:
   
   | Variable | Description |
   |----------|-------------|
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `DATABASE_URL` | Supabase PostgreSQL connection string |
   | `SUPABASE_URL` | Supabase project URL |
   | `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (keep secret!) |
   | `CRON_SECRET` | Random secret for cron job auth (generate with `openssl rand -hex 32`) |

4. **Enable Cron Jobs** (Vercel Pro/Enterprise required for custom crons):
   - The `vercel.json` includes a cron job that runs daily at 3 AM UTC
   - This automatically deletes recordings older than 3 days

### Step 3: Verify Deployment

1. Visit your Vercel deployment URL
2. Upload a test audio file
3. Check that recordings appear and can be played back
4. Verify the auto-delete functionality

---

## 📊 Dashboard Screenshots

The dashboard provides:

| Feature | Description |
|---------|-------------|
| **Summary Cards** | Total calls, average score, positive sentiment rate, escalation risk |
| **Sentiment Pie Chart** | Distribution of Positive/Neutral/Negative/Mixed sentiments |
| **Urgency Donut Chart** | High/Medium/Low urgency level distribution |
| **Agent Performance** | Bar chart showing agent scores comparison |
| **Daily Trends** | Line chart showing call volume and sentiment trends |
| **Category Scores** | Horizontal bar chart of scores by question category |

## 📝 API Endpoints

### Calls
- `POST /api/calls/upload` - Upload and analyze a call recording
- `GET /api/calls/` - List all analyzed calls
- `GET /api/calls/{call_id}` - Get detailed analysis for a specific call
- `GET /api/calls/{call_id}/audio-url` - Get signed URL for audio playback
- `DELETE /api/calls/{call_id}` - Delete a call and its recording
- `DELETE /api/calls/{call_id}/recording` - Delete only the recording, keep analysis
- `POST /api/calls/cleanup-expired` - Clean up expired recordings (cron job)

### Dashboard
- `GET /api/dashboard/metrics` - Get aggregated dashboard metrics
- `GET /api/dashboard/charts/sentiment-pie` - Sentiment distribution data
- `GET /api/dashboard/charts/agent-performance` - Agent performance data
- `GET /api/dashboard/charts/daily-trends` - Daily trends data
- `GET /api/dashboard/charts/category-scores` - Category scores data
- `GET /api/dashboard/charts/urgency-distribution` - Urgency distribution data
- `GET /api/dashboard/charts/escalation-risk` - Escalation risk data

### Health
- `GET /health` - Health check endpoint
- `GET /api/health` - API health check endpoint

## 🎯 Evaluation Categories

The system evaluates calls across these categories:

1. **Call Opening** (10 points)
   - Customer name verification
   - Script adherence
   - Response timing
   - Language greeting

2. **Soft Skills** (16 points)
   - Helpfulness
   - Grammar & communication
   - Confidence
   - Empathy
   - Professional tone

3. **Probing & Understanding** (10 points)
   - Effective questioning
   - First-instance understanding
   - Diagnostic questions

4. **Problem Resolution** (14 points)
   - Information accuracy
   - Solution appropriateness
   - Objection handling

5. **Call Closing** (8 points)
   - Closing format
   - Call summarization
   - Further assistance offer

6. **Critical Parameters** (15 points)
   - No premature disconnection
   - Correct categorization

## 📁 Project Structure

```
poc/
├── api/
│   └── index.py              # Vercel serverless entry point
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py       # SQLAlchemy models
│   │   └── schemas.py        # Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── calls.py          # Call upload & analysis endpoints
│   │   └── dashboard.py      # Dashboard & charts endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py  # Whisper API integration
│   │   ├── sentiment_analysis.py  # GPT-3.5 analysis
│   │   └── storage.py        # Supabase Storage service
│   ├── static/
│   │   └── css/
│   │       └── style.css     # Application styles
│   └── templates/
│       ├── base.html         # Base template
│       ├── dashboard.html    # Main dashboard
│       ├── upload.html       # Call upload page
│       └── call_detail.html  # Call analysis detail
├── sample.env                # Environment variables template
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for Vercel
├── vercel.json               # Vercel configuration
└── README.md
```

## 🔒 Supported Audio Formats

- MP3
- WAV
- M4A
- WebM
- OGG

Maximum file size: 25MB (Whisper API limit)

## ⏰ Recording Retention

- **Recordings are automatically deleted after 3 days** to save storage
- Analysis data (transcription, scores, sentiment) is preserved permanently
- Users can manually delete recordings from the call detail page
- A daily cron job runs at 3 AM UTC to clean up expired recordings

## 💡 Tips for Best Results

1. **Audio Quality**: Use clear audio with minimal background noise
2. **File Size**: Keep recordings under 25MB
3. **Language**: The system auto-detects language, but English works best
4. **Duration**: Shorter calls (< 10 minutes) process faster

## 🔧 Troubleshooting

### Vercel Deployment Issues

1. **Build Fails**: Ensure Python 3.11 is specified in `runtime.txt`
2. **Storage Errors**: Check Supabase credentials and bucket permissions
3. **Timeout Errors**: Audio processing may exceed free tier limits (10s)
   - Consider upgrading to Vercel Pro for 60s function duration
4. **Database Connection**: Verify DATABASE_URL uses the correct pooler URL

### Local Development Issues

1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **API Key Errors**: Ensure `.env` file has correct keys
3. **Port Conflict**: Change port with `--port 8001`

## 📄 License

This is a Proof of Concept (POC) for demonstration purposes.

## 🤝 Contact

For questions or support, refer to the original requirements document: `AI Powered Call Auditor.pdf`
