"""Main FastAPI application."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..models.model_router import get_model_router
from ..routers import (
    system,
    web,
    translation,
    glossary,
    standardization,
    pipeline,
    editor,
    dharmamitra,
    ucca,
    gloss,
    workflow,
    test
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting Tibetan Buddhist Translation API...")
    print(f"Available models: {list(get_model_router().get_available_models().keys())}")
    yield
    # Shutdown
    print("Shutting down Tibetan Buddhist Translation API...")


# Initialize FastAPI app
app = FastAPI(
    title="Lang-Graph Translation and UCCA API",
    description="""
An advanced API for translating Buddhist texts using a configurable, streaming-first pipeline.
It supports multi-stage processing including translation, glossary extraction, and terminology standardization.
It also provides endpoints for generating UCCA (Universal Conceptual Cognitive Annotation) graphs from text.

**Key Features:**
- Real-time streaming of results via Server-Sent Events (SSE).
- Batch processing capabilities.
- Configurable language models (e.g., Claude, Gemini).
- UCCA graph generation with optional commentaries.
- Cache management for improved performance.
    """,
    version="1.1.0",
    contact={
        "name": "API Support",
        "url": "http://example.com/contact",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# Calculate path to project root: go up 4 levels from src/translation_api/api/main.py
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    # Fallback: try relative to current working directory
    static_dir_fallback = os.path.join(os.getcwd(), "static")
    if os.path.exists(static_dir_fallback):
        app.mount("/static", StaticFiles(directory=static_dir_fallback), name="static")

# Include all routers
app.include_router(system.router)
app.include_router(web.router)
app.include_router(translation.router)
app.include_router(glossary.router)
app.include_router(standardization.router)
app.include_router(pipeline.router)
app.include_router(editor.router)
app.include_router(dharmamitra.router)
app.include_router(ucca.router)
app.include_router(gloss.router)
app.include_router(workflow.router)


# test router
app.include_router(test.router)


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app

