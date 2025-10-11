"""
Render deployment entry point for PriceOptima backend.
This file imports the FastAPI app from backend/api_backend.py to resolve Render's module import issue.
"""

from backend.api_backend import app

# Export the app for Render to use
__all__ = ["app"]
