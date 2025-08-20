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
from .workflows.streaming import (
    stream_translation_progress, 
    stream_single_translation_progress, 
    stream_glossary_progress,
    stream_standardization_progress
)
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
from .prompts.tibetan_buddhist import RETRANSLATION_PROMPT, GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from .models.pipeline import PipelineRequest, PipelineResponse

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
    results: List[TranslationResult] = Field(..., description="A list of the translation results.")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata, including timing and batch information.")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="A list of any errors that occurred during the process.")


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
    description="""
A specialized, multi-stage API for translating Tibetan Buddhist texts. The workflow is designed to be modular:

### 1. Translation
- Use the `/translate` or `/translate/stream` endpoints to get high-quality translations for your source texts.

### 2. Glossary Extraction
- After translating, send the results to the `/glossary/extract` or `/glossary/extract/stream` endpoints to generate a detailed glossary of key terms.

### 3. Standardization
- Use the `/standardize/analyze` endpoint to identify inconsistencies in your translations.
- Use the `/standardize/apply` or `/standardize/apply/stream` endpoints to enforce a consistent terminology across all your translations.

### Additional Features:
- **Multiple AI Models**: Supports various models from Anthropic, OpenAI, and Google.
- **Intelligent Caching**: In-memory cache for translations and glossaries to boost performance.
- **Custom Rules**: Allows users to provide custom translation instructions.
    """,
    version="2.0.0",
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
                        "errors": []
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
    
    This endpoint's sole purpose is to return translations. For glossary extraction or standardization, please use the dedicated endpoints after receiving the translation results from this one. Results are cached to accelerate repeated requests.
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




@app.post(
    "/translate/stream", 
    tags=["Streaming"],
    responses={
        200: {
            "description": """
A stream of Server-Sent Events (SSE) for the translation process.

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

**Event Type: `completion`** (the final event)
```json
{
  "timestamp": "...",
  "type": "completion",
  "status": "completed",
  "results": [...]
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

    This endpoint returns a stream of events as texts are translated in batches. It does not perform glossary extraction. Use the `/glossary/extract/stream` endpoint for that functionality after the translation is complete.
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


# --- Glossary Endpoints ---

class GlossaryExtractionRequest(BaseModel):
    items: List[TranslationResult]
    model_name: str = Field("claude", description="The model to use for the extraction task.")
    batch_size: int = Field(5, description="Number of items to process per parallel batch request.")

@app.post(
    "/glossary/extract", 
    response_model=Glossary, 
    tags=["Glossary"],
    responses={
        200: {
            "description": "A consolidated glossary of all unique terms.",
            "content": {
                "application/json": {
                    "example": {
                        "terms": [
                            {"source_term": "bodhicitta", "translated_term": "mind of enlightenment"},
                            {"source_term": "shunyata", "translated_term": "emptiness"}
                        ]
                    }
                }
            }
        }
    }
)
async def extract_glossary(request: GlossaryExtractionRequest):
    """
    Extract a consolidated glossary from a list of translated texts.

    This endpoint uses a parallel, batched approach to efficiently extract
    key terms from a large number of translation pairs.
    """
    model_router = get_model_router()
    try:
        model = model_router.get_model(request.model_name)
        structured_model = model.with_structured_output(Glossary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompts = [
        GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
            text_pairs=f"Source: {item.original_text}\\nTranslated: {item.translated_text}\\n\\n"
        ) for item in request.items
    ]

    all_terms = []
    try:
        for i in range(0, len(prompts), request.batch_size):
            batch_prompts = prompts[i:i + request.batch_size]
            glossary_results = await structured_model.abatch(batch_prompts)
            for gloss in glossary_results:
                if gloss and gloss.terms:
                    all_terms.extend(gloss.terms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Glossary extraction failed: {str(e)}")

    return Glossary(terms=all_terms)


@app.post(
    "/glossary/extract/stream", 
    tags=["Glossary"],
    responses={
        200: {
            "description": """
A stream of Server-Sent Events (SSE) for glossary extraction.

**Event Type: `glossary_batch_completed`**
```json
{
  "timestamp": "...",
  "type": "glossary_batch_completed",
  "status": "batch_complete",
  "terms": [{"source_term": "...", "translated_term": "..."}]
}
```

**Event Type: `completion`**
```json
{
  "timestamp": "...",
  "type": "completion",
  "status": "completed",
  "glossary": {"terms": [...]}
}
```
            """,
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"}
                }
            }
        }
    }
)
async def stream_extract_glossary(request: GlossaryExtractionRequest):
    """
    Extract a glossary from translated texts with real-time streaming progress.

    This endpoint provides a stream of events as batches of glossary terms
    are extracted in parallel, allowing the UI to display results incrementally.
    """
    return EventSourceResponse(
        stream_glossary_progress(request),
        media_type="text/event-stream"
    )


# --- Standardization Endpoints ---

@app.post(
    "/standardize/analyze", 
    response_model=AnalysisResponse, 
    tags=["Standardization"],
    responses={
        200: {
            "description": "A report of inconsistent terms.",
            "content": {
                "application/json": {
                    "example": {
                        "inconsistent_terms": {
                            "bodhicitta": ["mind of enlightenment", "enlightenment mind"]
                        }
                    }
                }
            }
        }
    }
)
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


@app.post(
    "/standardize/apply", 
    response_model=StandardizationResponse, 
    tags=["Standardization"],
    responses={
        200: {
            "description": "The full list of translations with standardizations applied.",
            "content": {
                "application/json": {
                    "example": {
                        "updated_items": [
                            {
                                "original_text": "Develop bodhicitta...",
                                "translated_text": "Develop mind of enlightenment...",
                                "glossary": [{"source_term": "bodhicitta", "translated_term": "mind of enlightenment"}]
                            }
                        ]
                    }
                }
            }
        }
    }
)
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


@app.post(
    "/standardize/apply/stream", 
    tags=["Standardization"],
    responses={
        200: {
            "description": """
A stream of Server-Sent Events (SSE) for the standardization process.

**Event Type: `retranslation_completed`**
```json
{
  "timestamp": "...",
  "type": "retranslation_completed",
  "status": "item_updated",
  "index": 1,
  "updated_item": {
      "original_text": "...",
      "translated_text": "...",
      "glossary": [...]
  }
}
```
            """,
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"}
                }
            }
        }
    }
)
async def stream_apply_standardization(request: StandardizationRequest):
    """
    Apply standardization rules and stream the updated translations in real-time.
    """
    return EventSourceResponse(
        stream_standardization_progress(request),
        media_type="text/event-stream"
    )


@app.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Run a customizable workflow by selecting stages and providing inputs.

    Stages (in order):
    - translate: uses existing translation workflow
    - extract_glossary: runs glossary extraction on results/items
    - analyze: runs standardization analysis
    - apply_standardization: applies standardization pairs with minimal-change retranslation
    """
    aggregate: PipelineResponse = PipelineResponse(metadata={"stages": request.stages})

    # Stage: translate
    if "translate" in request.stages:
        if not request.texts or not request.target_language:
            raise HTTPException(status_code=400, detail="texts and target_language are required for translate stage")
        workflow_request = TranslationRequest(
            texts=request.texts,
            target_language=request.target_language,
            model_name=request.model_name,
            text_type=request.text_type,
            batch_size=request.batch_size,
            model_params=request.model_params,
            user_rules=request.user_rules,
        )
        final_state = await run_translation_workflow(workflow_request)
        aggregate.results = final_state["final_results"]
        aggregate.metadata.update({"translation": final_state.get("metadata", {})})

    # Build items for downstream stages
    items_source = request.items
    if aggregate.results:
        # Convert TranslationResult list into StandardizationInputItem-like dicts
        items_source = [
            {
                "original_text": r.original_text,  # type: ignore[attr-defined]
                "translated_text": r.translated_text,  # type: ignore[attr-defined]
                "glossary": getattr(r, "glossary", []),
            }
            for r in aggregate.results
        ]

    # Stage: extract_glossary
    if "extract_glossary" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for glossary extraction")
        glossary_req = GlossaryExtractionRequest(items=items_source, model_name=request.model_name, batch_size=request.batch_size)  # type: ignore[arg-type]
        aggregate.glossary = await extract_glossary(glossary_req)  # reuse handler logic

    # Stage: analyze
    if "analyze" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for analysis")
        # If we have a new glossary, enrich items with glossary terms
        if aggregate.glossary:
            enriched = []
            for item in items_source:
                terms = [t for t in aggregate.glossary.terms if (t.source_term in item["original_text"] or t.translated_term in item["translated_text"]) ]
                enriched.append({**item, "glossary": terms})
            items_source = enriched
        analysis_req = AnalysisRequest(items=items_source)  # type: ignore[arg-type]
        analysis_resp = await analyze_consistency(analysis_req)
        aggregate.inconsistent_terms = analysis_resp.inconsistent_terms

    # Stage: apply_standardization
    if "apply_standardization" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for standardization")
        if not request.standardization_pairs:
            raise HTTPException(status_code=400, detail="standardization_pairs is required for apply_standardization stage")
        std_req = StandardizationRequest(
            items=items_source,  # type: ignore[arg-type]
            standardization_pairs=request.standardization_pairs,
            model_name=request.model_name,
            user_rules=request.user_rules,
        )
        std_resp = await apply_standardization(std_req)
        aggregate.updated_items = std_resp.updated_items

    return aggregate


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