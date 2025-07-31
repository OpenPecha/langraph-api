"""FastAPI application for Tibetan Buddhist text translation."""

import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import uvicorn

from .workflows.translation_state import TranslationRequest, TranslationResult
from .workflows.streaming import stream_translation_progress, stream_single_translation_progress
from .models.model_router import get_model_router
from .config import get_settings

# Import from root level graph module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from graph import run_translation_workflow


class TranslationAPIRequest(BaseModel):
    """API request model for translation."""
    texts: List[str] = Field(..., description="List of texts to translate", min_items=1, max_items=100)
    target_language: str = Field(..., description="Target language for translation")
    model_name: str = Field("claude", description="Model to use for translation")
    text_type: str = Field("Buddhist text", description="Type of Buddhist text")
    batch_size: int = Field(5, description="Number of texts to process per batch", ge=1, le=50)
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules/instructions")


class TranslationAPIResponse(BaseModel):
    """API response model for translation."""
    success: bool = Field(..., description="Whether the translation was successful")
    results: List[TranslationResult] = Field(..., description="Translation results")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Any errors that occurred")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    available_models: Dict[str, Dict[str, Any]] = Field(..., description="Available translation models")




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
    title="Buddhist Text Translation API",
    description="Specialized API for translating Tibetan Buddhist texts with real-time streaming, multiple AI models, and custom user rules",
    version="1.0.0",
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
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global settings
settings = get_settings()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API status and available translation models."""
    model_router = get_model_router()
    available_models = model_router.get_available_models()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_models=available_models
    )




@app.post("/translate", response_model=TranslationAPIResponse, tags=["Translation"])
async def translate_texts(request: TranslationAPIRequest):
    """
    Translate multiple Buddhist texts with batch processing.
    
    Features:
    - Multiple AI model support (Claude, Gemini)
    - Custom user rules for translation preferences
    - Domain-aware prompting for Buddhist texts
    - Efficient batch processing
    """
    try:
        # Validate model availability
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' is not available. Available models: {available_models}"
            )
        
        # Validate batch size
        if request.batch_size > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {request.batch_size} exceeds maximum allowed {settings.max_batch_size}"
            )
        
        # Create workflow request
        workflow_request = TranslationRequest(
            texts=request.texts,
            target_language=request.target_language,
            model_name=request.model_name,
            text_type=request.text_type,
            batch_size=request.batch_size,
            model_params=request.model_params,
            user_rules=request.user_rules
        )
        
        # Run the translation workflow
        final_state = await run_translation_workflow(workflow_request)
        
        # Prepare response
        return TranslationAPIResponse(
            success=final_state["workflow_status"] == "completed",
            results=final_state["final_results"],
            metadata=final_state["metadata"],
            errors=final_state["errors"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )




class SingleTranslationRequest(BaseModel):
    """Request model for single text translation."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language for translation")
    model_name: str = Field("claude", description="Model to use for translation")
    text_type: str = Field("Buddhist text", description="Type of Buddhist text")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules/instructions")


@app.post("/translate/single", response_model=TranslationAPIResponse, tags=["Translation"])
async def translate_single_text(request: SingleTranslationRequest):
    """
    Translate a single Buddhist text.
    
    Convenience endpoint for translating individual texts with the same features as batch translation.
    """
    batch_request = TranslationAPIRequest(
        texts=[request.text],
        target_language=request.target_language,
        model_name=request.model_name,
        text_type=request.text_type,
        batch_size=1,
        model_params=request.model_params,
        user_rules=request.user_rules
    )
    
    return await translate_texts(batch_request)




@app.post("/translate/stream", tags=["Streaming"])
async def stream_translate_texts(request: TranslationAPIRequest):
    """
    Stream translation progress with real-time updates (RECOMMENDED).
    
    Returns Server-Sent Events with:
    - Real-time progress updates
    - Results delivered as batches complete
    - Detailed processing statistics
    - Error handling and recovery
    
    Perfect for building responsive UIs with live translation updates.
    """
    try:
        # Validate model availability
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' is not available. Available models: {available_models}"
            )
        
        # Validate batch size
        if request.batch_size > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {request.batch_size} exceeds maximum allowed {settings.max_batch_size}"
            )
        
        # Create workflow request
        workflow_request = TranslationRequest(
            texts=request.texts,
            target_language=request.target_language,
            model_name=request.model_name,
            text_type=request.text_type,
            batch_size=request.batch_size,
            model_params=request.model_params,
            user_rules=request.user_rules
        )
        
        # Return SSE stream
        return EventSourceResponse(
            stream_translation_progress(workflow_request),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/translate/single/stream", tags=["Streaming"])
async def stream_translate_single_text(request: SingleTranslationRequest):
    """
    Stream single text translation with real-time progress.
    
    Returns Server-Sent Events for individual text translation with the same
    real-time updates as the batch streaming endpoint.
    """
    try:
        # Validate model availability
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' is not available. Available models: {available_models}"
            )
        
        # Return SSE stream for single text
        return EventSourceResponse(
            stream_single_translation_progress(
                text=request.text,
                target_language=request.target_language,
                model_name=request.model_name,
                text_type=request.text_type,
                model_params=request.model_params,
                user_rules=request.user_rules
            ),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/", tags=["Web UI"])
async def root():
    """Access the built-in web interface for testing translations."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/ui", tags=["Web UI"])
async def web_ui():
    """Access the built-in web interface for testing translations."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")




def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "src.translation_api.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )