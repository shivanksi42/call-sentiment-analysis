"""
Vercel Serverless Function Entry Point
Vercel auto-detects FastAPI apps when you export 'app'
"""
import os
import sys
from pathlib import Path

# Add the poc directory to Python path for imports
poc_dir = Path(__file__).parent.parent
sys.path.insert(0, str(poc_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the FastAPI app - Vercel auto-detects this
from app.main import app
