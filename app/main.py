import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file in the poc directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from .routers import calls_router, dashboard_router

# Initialize FastAPI app
app = FastAPI(
    title="AI Call Quality Auditor",
    description="AI-powered call quality auditing and customer sentiment analysis system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
templates_path = os.path.join(os.path.dirname(__file__), "templates")

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Include routers
app.include_router(calls_router)
app.include_router(dashboard_router)


@app.get("/")
async def home(request: Request):
    """Render the main dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/upload")
async def upload_page(request: Request):
    """Render the upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/calls")
async def calls_list_page(request: Request):
    """Render the calls list page"""
    return templates.TemplateResponse("calls_list.html", {"request": request})


@app.get("/calls/{call_id}")
async def call_detail_page(request: Request, call_id: str):
    """Render the call detail page"""
    return templates.TemplateResponse("call_detail.html", {"request": request, "call_id": call_id})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Call Quality Auditor"}


@app.get("/api/health")
async def api_health_check():
    """API health check endpoint for Vercel"""
    return {"status": "healthy", "service": "AI Call Quality Auditor", "version": "1.0.0"}

