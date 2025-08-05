"""FastAPI application for Tibetan Buddhist text translation."""

import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import uvicorn

from .workflows.translation_state import TranslationRequest, TranslationResult
from .models.glossary import Glossary, GlossaryTerm
from .workflows.streaming import stream_translation_progress, stream_single_translation_progress
from .models.model_router import get_model_router
from .config import get_settings
from .models.standardization import (
    AnalysisRequest, 
    AnalysisResponse, 
    StandardizationRequest, 
    StandardizationResponse,
    RetranslationResponse,
    StandardizationInputItem
)
from .prompts.tibetan_buddhist import RETRANSLATION_PROMPT

# Import from root level graph module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from graph import run_translation_workflow
from .cache import get_cache


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
    glossary: Optional[Glossary] = Field(None, description="A consolidated glossary of key terms extracted from all the translated texts.")


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


@app.post("/system/clear-cache", tags=["System"])
async def clear_cache():
    """
    Clear the in-memory cache for translations and glossaries.
    
    This is useful for ensuring fresh results from the language model,
    especially after changing a model or prompt.
    """
    cache = get_cache()
    cache.clear_all()
    return {"status": "success", "message": "Cache cleared."}


@app.post(
    "/translate", 
    response_model=TranslationAPIResponse, 
    tags=["Translation"],
    responses={
        200: {
            "description": "Successful translation.",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "results": [
                            {
                                "original_text": "OM MANI PADME HUM",
                                "translated_text": "Om Mani Padme Hum (Hail the jewel in the lotus)",
                                "metadata": {"batch_id": "...", "model_used": "claude", "text_type": "mantra"}
                            }
                        ],
                        "metadata": {
                            "initialized_at": "2023-10-27T10:00:00Z",
                            "completed_at": "2023-10-27T10:00:05Z",
                            "total_processing_time": 5.0,
                            "successful_batches": 1,
                            "failed_batches": 0,
                            "total_translations": 1
                        },
                        "errors": [],
                        "glossary": {
                            "terms": [
                                {"source_term": "MANI", "translated_term": "Jewel"},
                                {"source_term": "PADME", "translated_term": "Lotus"}
                            ]
                        }
                    }
                }
            }
        },
        400: {"description": "Bad Request, e.g., invalid model name."},
        500: {"description": "Internal Server Error."}
    }
)
async def translate_texts(request: TranslationAPIRequest):
    """
    Translate multiple Buddhist texts with batch processing.
    
    Features:
    - Multiple AI model support (Claude, Gemini)
    - Custom user rules for translation preferences
    - Domain-aware prompting for Buddhist texts
    - Efficient batch processing
    The response includes the full list of translations and a consolidated glossary of all unique terms found. Results are cached to accelerate repeated requests.
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
            errors=final_state["errors"],
            glossary=final_state.get("glossary")
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




@app.post(
    "/translate/stream", 
    tags=["Streaming"],
    responses={
        200: {
            "description": """
A stream of Server-Sent Events (SSE). Each event is a JSON object prefixed with `data: `.
Below are examples of the key event types.

**Event Type: `initialization`**
```json
{
  "timestamp": "...",
  "type": "initialization",
  "status": "starting",
  "total_texts": 5
}
```

**Event Type: `batch_completed`** (sent after each batch of translations)
```json
{
  "timestamp": "...",
  "type": "batch_completed",
  "status": "batch_completed",
  "batch_results": [
    {
      "original_text": "...",
      "translated_text": "...",
      "metadata": {}
    }
  ]
}
```

**Event Type: `glossary_extraction_start`** (sent after all translations are done)
```json
{
  "timestamp": "...",
  "type": "glossary_extraction_start",
  "status": "extracting_glossary",
  "message": "..."
}
```

**Event Type: `glossary_extraction_completed`** (sent after all glossaries are extracted)
```json
{
  "timestamp": "...",
  "type": "glossary_extraction_completed",
  "status": "glossary_extracted",
  "glossary": {
    "terms": [{"source_term": "...", "translated_term": "..."}]
  }
}
```

**Event Type: `completion`** (the final event)
```json
{
  "timestamp": "...",
  "type": "completion",
  "status": "completed",
  "total_texts": 5,
  "results": [...],
  "glossary": {...}
}
```
            """,
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string"
                    }
                }
            }
        }
    }
)
async def stream_translate_texts(request: TranslationAPIRequest):
    """
    Stream translation progress with real-time updates (RECOMMENDED).
    
    Returns Server-Sent Events with:
    - Real-time progress updates
    - Results delivered as batches complete
    - Detailed processing statistics
    - Error handling and recovery
    
    Perfect for building responsive UIs with live translation updates.
    This is ideal for building responsive user interfaces.
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


# --- Standardization Endpoints ---

@app.post("/standardize/analyze", response_model=AnalysisResponse, tags=["Standardization"])
async def analyze_consistency(request: AnalysisRequest):
    """
    Analyze a list of translations to find inconsistent glossary terms.
    
    This endpoint takes a list of translation results (each with its own glossary)
    and returns a dictionary of source terms that have more than one unique translation,
    making it easy to identify inconsistencies.
    """
    term_map = defaultdict(set)

    for item in request.items:
        if item.glossary:
            for term in item.glossary:
                term_map[term.source_term].add(term.translated_term)
    
    inconsistent_terms = {
        term: list(translations)
        for term, translations in term_map.items()
        if len(translations) > 1
    }
    
    return AnalysisResponse(inconsistent_terms=inconsistent_terms)


@app.post("/standardize/apply", response_model=StandardizationResponse, tags=["Standardization"])
async def apply_standardization(request: StandardizationRequest):
    """
    Apply a set of standardization rules to a list of translations.

    This endpoint intelligently re-translates only the necessary texts
    to enforce the provided standardization rules, while making minimal
    changes to the rest of the text.
    """
    model_router = get_model_router()
    try:
        model = model_router.get_model(request.model_name)
        structured_model = model.with_structured_output(RetranslationResponse)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prepare standardization rules for faster lookup and for the prompt
    source_words_to_standardize = {pair.source_word for pair in request.standardization_pairs}
    rules_block = ""
    for pair in request.standardization_pairs:
        rules_block += f'- For the source term "{pair.source_word}", use the exact translation "{pair.standardized_translation}".\n'

    updated_items = []
    for item in request.items:
        # Check if this item's original text contains any of the words to be standardized
        if any(word in item.original_text for word in source_words_to_standardize):
            # This item needs to be re-translated
            prompt = RETRANSLATION_PROMPT.format(
                user_rules=request.user_rules or "No specific user rules provided.",
                standardization_rules_block=rules_block,
                original_text=item.original_text,
                original_translation=item.translated_text
            )
            
            try:
                # Get the new translation
                response = await structured_model.ainvoke(prompt)
                new_translation = response.new_translation
                
                # Update the item's translation
                item.translated_text = new_translation

                # Update the glossary for this item
                # A simple approach: re-extract the glossary for the new pair
                glossary_terms = []
                glossary_model = model.with_structured_output(Glossary)
                glossary_prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                    text_pairs=f"Source: {item.original_text}\\nTranslated: {new_translation}\\n\\n"
                )
                glossary_result = await glossary_model.ainvoke(glossary_prompt)
                if glossary_result and glossary_result.terms:
                    item.glossary = glossary_result.terms
                else:
                    item.glossary = []

            except Exception as e:
                # If re-translation fails, you could either keep the original,
                # or mark it as failed. For now, we'll just log and keep original.
                # In a real app, you might add an error flag to the item.
                print(f"Failed to re-translate item: {e}")
        
        updated_items.append(item)

    return StandardizationResponse(updated_items=updated_items)


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