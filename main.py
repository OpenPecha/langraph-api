"""Main entry point for the Tibetan Buddhist Translation API."""

import asyncio
import uvicorn
from src.translation_api.api import create_app
from src.translation_api.config import get_settings


def main():
    """Run the translation API server."""
    settings = get_settings()
    
    print(f"Starting Tibetan Buddhist Translation API...")
    print(f"Server will run on http://{settings.api_host}:{settings.api_port}")
    print(f"API Documentation available at http://{settings.api_host}:{settings.api_port}/docs")
    
    uvicorn.run(
        "src.translation_api.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()