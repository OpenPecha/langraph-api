"""FastAPI application for Tibetan Buddhist text translation."""

import asyncio
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException
import httpx
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
from .models.ucca import (
    UCCARequest,
    UCCAResponse,
    UCCABatchRequest,
    UCCABatchItemResult,
)
from .workflows.ucca import generate_ucca_graph, stream_ucca_generation
from .models.gloss import (
    GlossRequest,
    GlossResponse,
    GlossBatchRequest,
    GlossBatchItemResult,
)
from .workflows.gloss import generate_gloss, stream_gloss_generation
from .models.workflow import (
    WorkflowInput,
    WorkflowResponse,
    WorkflowLLMResult,
    WorkflowRunRequest,
    WorkflowBatchRequest,
    WorkflowBatchItemResult,
)
from .models.comment import (
    EditorCommentRequest,
    EditorCommentResponse,
    EditorCommentLLMOutput,
)
from langchain_core.messages import HumanMessage

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


@app.get("/models", tags=["System"], summary="Get All Supported Models")
async def get_models():
    """
    Get a list of all supported translation models.
    
    Returns:
        Dictionary of available models with their capabilities and configurations.
        Each model includes:
        - provider: The AI provider (Anthropic, OpenAI, Google)
        - description: Model description
        - capabilities: List of model capabilities
        - context_window: Maximum context window size
    
    Example response:
    ```json
    {
        "claude-sonnet-4-20250514": {
            "provider": "Anthropic",
            "description": "Claude Sonnet 4.0 (2025-05-14)",
            "capabilities": ["text", "reasoning", "translation"],
            "context_window": 200000
        },
        ...
    }
    ```
    """
    model_router = get_model_router()
    available_models = model_router.get_available_models()
    return available_models


@app.post("/system/clear-cache", tags=["System"], summary="Clear Server-Side Cache")
async def clear_cache():
    """
    Clears the server-side cache for all memoized functions. This is useful
    when you want to ensure you are getting fresh results from the language models,
    bypassing any previously cached translations or analyses.
    """
    cache = get_cache()
    cleared_count = cache.clear()
    return {"status": "cache_cleared", "cleared_items": cleared_count}


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


# --- Editor Comment Endpoints ---

import re as _re


def _extract_mentions(messages: list[dict], scope: str = "last", max_mentions: int = 5) -> list[str]:
    pattern = _re.compile(r"@[A-Za-z0-9_\-]+")
    contents: list[str] = []
    if scope == "thread":
        contents = [m.content for m in messages if hasattr(m, "content")]
    else:
        if messages:
            contents = [messages[-1].content]
    seen = set()
    mentions: list[str] = []
    for c in contents:
        for h in pattern.findall(c or ""):
            if h not in seen:
                mentions.append(h)
                seen.add(h)
                if len(mentions) >= max_mentions:
                    return mentions
    return mentions


def _enumerate_references(refs: list[dict]) -> tuple[str, list[str]]:
    lines: list[str] = []
    ids: list[str] = []
    for i, r in enumerate(refs, start=1):
        t = (getattr(r, "type", None) or "ref").strip().lower()
        slug = _re.sub(r"[^a-z0-9\-]", "-", t) or "ref"
        rid = f"ref-{slug}-{i}"
        ids.append(rid)
        content = getattr(r, "content", "")
        lines.append(f"- [{rid}] (type={t}) {content}")
    section = "\n".join(lines) if lines else "None."
    return section, ids


def _build_thread(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_mentions_section(mentions: list[str]) -> str:
    if not mentions:
        return "None."
    return "\n".join([f"- {h}" for h in mentions])


def _build_editor_prompt(thread_text: str, refs_section: str, mentions_section: str) -> str:
    return (
        "You are a Tibetan Buddhist translation reviewer. Produce a concise, actionable commentary grounded ONLY in the provided references.\n\n"
        "TRIGGER\n"
        "- Proceed ONLY if the last message contains \"@Comment\".\n"
        "- If absent, return exactly:\n"
        '{"skipped": true, "reason": "No @Comment trigger"}\n'
        "and stop.\n\n"
        "THREAD (most recent last)\n"
        f"{thread_text}\n\n"
        "REFERENCES (use these IDs for citations)\n"
        f"{refs_section}\n\n"
        "MENTIONS\n"
        f"{mentions_section}\n\n"
        "TASK\n"
        "- Evaluate terminology, doctrinal accuracy, register, grammar, and consistency.\n"
        "- Make minimal, high-impact suggestions.\n"
        "- Every sentence MUST be supported by at least one reference and MUST end with bracketed citations using the exact IDs above (e.g., \"... [ref-commentary-1]\" or \"... [ref-scan-2;ref-lexicon-3]\").\n"
        "- If evidence is insufficient, do not make the claim. If critical context is missing, end with a short sentence requesting the needed references and cite [ref-needed].\n"
        "- If MENTIONS is non-empty, begin the comment with all handles in order, space-separated (e.g., \"@Kun @Tenzin \"), using the handles exactly.\n\n"
        "OUTPUT (JSON ONLY; single object)\n"
        "{\n"
        "  \"comment_text\": \"The full commentary with inline bracketed citations at the end of each sentence.\",\n"
        "  \"citations_used\": [\"ref-...\"]\n"
        "}\n\n"
        "RULES\n"
        "- Only output the JSON object above; no extra fields.\n"
        "- citations_used must be the unique set of IDs actually cited in comment_text.\n"
        "- Do not invent references or handles.\n"
    )


@app.post("/editor/comment", response_model=EditorCommentResponse, tags=["Editor"], summary="Generate grounded commentary for a translation thread")
async def editor_comment(request: EditorCommentRequest):
    # Trigger check (@Comment in last message)
    if not request.messages or "@Comment" not in (request.messages[-1].content or ""):
        return {"mentions": [], "comment_text": "", "citations_used": [], "metadata": {"skipped": True, "reason": "No @Comment trigger"}}

    opts = request.options or {}
    mention_scope = getattr(opts, "mention_scope", "last")
    max_mentions = getattr(opts, "max_mentions", 5)

    # Mentions and references enumeration
    mentions = _extract_mentions(request.messages, scope=mention_scope, max_mentions=max_mentions)
    refs_section, _all_ids = _enumerate_references(request.references)
    thread_text = _build_thread(request.messages)
    mentions_section = _build_mentions_section(mentions)
    prompt = _build_editor_prompt(thread_text, refs_section, mentions_section)

    # Choose model (disallow 'dharamitra')
    model_router = get_model_router()
    selected_model = getattr(opts, "model_name", None)
    if selected_model and selected_model.lower() == "dharamitra":
        raise HTTPException(status_code=400, detail="'dharamitra' is translation-only and not supported for editor comments")
    # Fallback preference: gemini-2.5-pro if available else default
    model_name = selected_model or ("gemini-2.5-pro" if model_router.validate_model_availability("gemini-2.5-pro") else get_settings().default_model)
    if not model_router.validate_model_availability(model_name):
        available = list(model_router.get_available_models().keys())
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available. Available models: {available}")

    model = model_router.get_model(model_name)
    structured = model.with_structured_output(EditorCommentLLMOutput)

    try:
        resp = await structured.ainvoke(prompt)
        comment_text = resp.comment_text
        citations_used = resp.citations_used or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Editor comment generation failed: {str(e)}")

    return EditorCommentResponse(
        mentions=mentions,
        comment_text=comment_text,
        citations_used=citations_used,
        metadata={
            "triggered_by": "@Comment",
            "model_used": model_name,
        },
    )


@app.post("/editor/comment/stream", tags=["Editor"], summary="Generate grounded commentary (SSE)")
async def editor_comment_stream(request: EditorCommentRequest):
    async def _gen():
        # Trigger check
        if not request.messages or "@Comment" not in (request.messages[-1].content or ""):
            yield f"data: {{\"skipped\": true, \"reason\": \"No @Comment trigger\"}}\n\n"
            return

        opts = request.options or {}
        mention_scope = getattr(opts, "mention_scope", "last")
        max_mentions = getattr(opts, "max_mentions", 5)

        mentions = _extract_mentions(request.messages, scope=mention_scope, max_mentions=max_mentions)
        refs_section, _all_ids = _enumerate_references(request.references)
        thread_text = _build_thread(request.messages)
        mentions_section = _build_mentions_section(mentions)
        prompt = _build_editor_prompt(thread_text, refs_section, mentions_section)

        # Model selection
        model_router = get_model_router()
        selected_model = getattr(opts, "model_name", None)
        if selected_model and selected_model.lower() == "dharamitra":
            yield f"data: {{\"error\": \"'dharamitra' not supported for editor comments\"}}\n\n"
            return
        model_name = selected_model or ("gemini-2.5-pro" if model_router.validate_model_availability("gemini-2.5-pro") else get_settings().default_model)
        if not model_router.validate_model_availability(model_name):
            avail = list(model_router.get_available_models().keys())
            yield f"data: {{\"error\": \"Model '{model_name}' not available.\", \"available\": {json.dumps(avail)} }}\n\n"
            return

        yield f"data: {{\"type\": \"initialization\", \"mentions\": {json.dumps(mentions)}, \"model_used\": {json.dumps(model_name)} }}\n\n"

        # Build a streaming-friendly prompt (plain text output only)
        def _build_editor_stream_prompt(thread_text: str, refs_section: str, mentions_section: str) -> str:
            return (
                "You are a Tibetan Buddhist translation reviewer. Produce a concise, actionable commentary grounded ONLY in the provided references.\n\n"
                "TRIGGER\n- Proceed ONLY if the last message contains \"@Comment\". If absent, output exactly: SKIP and stop.\n\n"
                "THREAD (most recent last)\n" + thread_text + "\n\n"
                "REFERENCES (use these IDs for citations)\n" + refs_section + "\n\n"
                "MENTIONS\n" + mentions_section + "\n\n"
                "TASK\n"
                "- Begin the comment with all handles in MENTIONS (space-separated) if any.\n"
                "- Make minimal, high-impact suggestions only.\n"
                "- Every sentence MUST end with bracketed citations using the exact IDs from REFERENCES, e.g., [ref-...;ref-...].\n"
                "- Do not add any preface or headers; output ONLY the final comment text.\n"
            )

        stream_prompt = _build_editor_stream_prompt(thread_text, refs_section, mentions_section)

        # If trigger missing, short-circuit
        if "@Comment" not in (request.messages[-1].content or ""):
            yield "data: {\"skipped\": true, \"reason\": \"No @Comment trigger\"}\n\n"
            return

        model = model_router.get_model(model_name)
        full_text = ""
        try:
            # Prefer token-level events when available
            try:
                # Only pass generation_config for Gemini models
                if model_name.startswith("gemini"):
                    stream_kwargs = {"generation_config": {"response_mime_type": "text/plain"}}
                else:
                    stream_kwargs = {}
                
                async for event in model.astream_events([HumanMessage(content=stream_prompt)], **stream_kwargs):
                    et = event.get("event")
                    if et in ("on_chat_model_stream", "on_llm_stream"):
                        chunk = event.get("data", {}).get("chunk")
                        piece = getattr(chunk, "content", None)
                        if isinstance(piece, list):
                            piece = "".join([str(p) for p in piece])
                        if isinstance(piece, str) and piece:
                            full_text += piece
                            yield f"data: {{\"type\": \"comment_delta\", \"text\": {json.dumps(piece)} }}\n\n"
            except AttributeError:
                # Fallback: no native streaming; use single shot but send once
                if model_name.startswith("gemini"):
                    resp = await model.ainvoke([HumanMessage(content=stream_prompt)], generation_config={"response_mime_type": "text/plain"})
                else:
                    resp = await model.ainvoke([HumanMessage(content=stream_prompt)])
                text = getattr(resp, "content", "") or ""
                if isinstance(text, list):
                    text = "".join([str(p) for p in text])
                full_text = str(text)
                if full_text:
                    yield f"data: {{\"type\": \"comment_delta\", \"text\": {json.dumps(full_text)} }}\n\n"

            # Derive citations_used by scanning bracketed IDs
            ids = []
            try:
                import re as _rx
                for m in _rx.finditer(r"\[(.*?)\]", full_text):
                    inside = m.group(1) or ""
                    for tok in inside.split(";"):
                        t = tok.strip()
                        if t and t not in ids:
                            ids.append(t)
            except Exception:
                pass

            yield f"data: {{\"type\": \"completion\", \"comment_text\": {json.dumps(full_text)}, \"citations_used\": {json.dumps(ids)}, \"mentions\": {json.dumps(mentions)} }}\n\n"
        except Exception as e:
            yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))} }}\n\n"

    return EventSourceResponse(_gen(), media_type="text/event-stream")
# --- Dharmamitra Proxy Endpoints ---

class DharmamitraKnnRequest(BaseModel):
    query: str
    language: str
    password: Optional[str] = None
    do_grammar: Optional[bool] = False  # Ignored; always forced False


@app.post("/dharmamitra/knn-translate-mitra", tags=["Dharmamitra"], summary="Proxy: KNN Translate Mitra (Streaming)")
async def dharmamitra_knn_translate_mitra(request: DharmamitraKnnRequest):
    """Proxy to Dharmamitra KNN Translate Mitra SSE endpoint.

    Upstreams to https://dharmamitra.org/api-search/knn-translate-mitra/
    Returns text/event-stream forwarding 'data:' chunks.
    """
    url = "https://dharmamitra.org/api-search/knn-translate-mitra/"
    pwd = get_settings().dharmamitra_password or request.password
    if not pwd:
        raise HTTPException(status_code=400, detail="DHARMAMITRA_PASSWORD not set and no password provided")

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(url, json={
                "query": request.query,
                "language": (request.language or "").lower(),
                "password": pwd,
                "do_grammar": False,
            })
            # Dharmamitra returns SSE in the body as text; forward lines with data:
            text = resp.text
            # Normalize CRLF
            buf = text.replace("\r\n", "\n")
            for line in buf.split("\n"):
                if line.startswith("data: "):
                    yield f"data: {line[6:].strip()}\n\n"

    return EventSourceResponse(event_stream(), media_type="text/event-stream")


class DharmamitraGeminiRequest(BaseModel):
    query: str
    language: str
    password: Optional[str] = None
    do_grammar: Optional[bool] = None  # Ignored; forced False
    use_pro_model: Optional[bool] = False  # Ignored; forced False


@app.post("/dharmamitra/knn-translate-gemini-no-stream", tags=["Dharmamitra"], summary="Proxy: KNN Translate Gemini (Non-stream)")
async def dharmamitra_knn_translate_gemini(request: DharmamitraGeminiRequest):
    """Proxy to Dharmamitra Gemini non-stream endpoint.

    Upstreams to https://dharmamitra.org/api-search/knn-translate-gemini-no-stream1/
    """
    url = "https://dharmamitra.org/api-search/knn-translate-gemini-no-stream1/"
    pwd = get_settings().dharmamitra_password or request.password
    if not pwd:
        raise HTTPException(status_code=400, detail="DHARMAMITRA_PASSWORD not set and no password provided")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "query": request.query,
                "language": (request.language or "").lower(),
                "password": pwd,
                "do_grammar": False,
                "use_pro_model": False,
            }

            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dharmamitra Gemini proxy failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.translation_api.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


# --- UCCA Endpoints ---

class UCCAErrorResponse(BaseModel):
    error: str


@app.post("/ucca/generate", response_model=UCCAResponse, tags=["UCCA"], summary="Generate a Single UCCA Graph", responses={
    200: {"description": "Generated UCCA graph"},
    400: {"description": "Invalid model name"},
    500: {"description": "UCCA generation failed", "model": UCCAErrorResponse},
})
async def ucca_generate(request: UCCARequest) -> UCCAResponse:
    """
    Generates a UCCA (Universal Conceptual Cognitive Annotation) graph for a single input text.

    This endpoint invokes a language model with a specialized prompt to parse the text
    and return a structured UCCA graph. Optional commentaries can be provided for context.
    """
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)
        raw_json, graph = generate_ucca_graph(
            model,
            request.input_text,
            commentary_1=request.commentary_1,
            commentary_2=request.commentary_2,
            commentary_3=request.commentary_3,
            sanskrit_text=request.sanskrit_text,
        )
        return UCCAResponse(ucca_graph=graph, raw_json=raw_json)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        return UCCAResponse(error=f"Failed to parse LLM output as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UCCA generation failed: {str(e)}")


@app.post("/ucca/generate/batch", response_model=list[UCCABatchItemResult], tags=["UCCA"], summary="Generate UCCA Graphs in Batch")
async def ucca_generate_batch(request: UCCABatchRequest) -> list[UCCABatchItemResult]:
    """
    Generates UCCA graphs for a batch of input texts.

    This endpoint processes multiple texts in parallel (up to the specified `batch_size`)
    and returns a list of results once all items have been processed.
    """
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)

        results: list[UCCABatchItemResult] = []
        for idx, item in enumerate(request.items):
            try:
                _, graph = generate_ucca_graph(
                    model,
                    item.input_text,
                    commentary_1=item.commentary_1,
                    commentary_2=item.commentary_2,
                    commentary_3=item.commentary_3,
                    sanskrit_text=item.sanskrit_text,
                )
                results.append(UCCABatchItemResult(index=idx, ucca_graph=graph))
            except Exception as e:
                results.append(UCCABatchItemResult(index=idx, error=str(e)))

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch UCCA generation failed: {str(e)}")


@app.post("/ucca/generate/stream", tags=["UCCA"], summary="Generate UCCA Graphs via SSE Stream")
async def ucca_generate_stream(request: UCCABatchRequest):
    """
    Generates UCCA graphs for a batch of input texts and streams the results via SSE.

    This is the recommended endpoint for generating UCCA for multiple items in an interactive
    application. It provides real-time feedback as each item is processed and sends
    a final completion event with all results.
    """
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)
        # Convert Pydantic items to dict for the streamer
        items = [
            {
                "input_text": it.input_text,
                "commentary_1": it.commentary_1,
                "commentary_2": it.commentary_2,
                "commentary_3": it.commentary_3,
                "sanskrit_text": it.sanskrit_text,
            }
            for it in request.items
        ]

        return EventSourceResponse(
            stream_ucca_generation(model, items, batch_size=request.batch_size),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming UCCA generation failed: {str(e)}")


# --- Gloss Endpoints ---

@app.post("/gloss/generate", response_model=GlossResponse, tags=["Gloss"], summary="Generate Gloss for a Single Text")
async def gloss_generate(request: GlossRequest) -> GlossResponse:
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)
        raw, data = generate_gloss(
            model,
            request.input_text,
            ucca_interpretation=request.ucca_interpretation,
            commentary_1=request.commentary_1,
            commentary_2=request.commentary_2,
            commentary_3=request.commentary_3,
            sanskrit_text=request.sanskrit_text,
        )

        std_text = data.get("StandardizedText", {}).get("standardized_text")
        note = data.get("StandardizedText", {}).get("note")
        analysis = json.dumps(data.get("analysis", []), ensure_ascii=False)
        glossary = data.get("Glossary", {}).get("glossary")
        return GlossResponse(standardized_text=std_text, note=note, analysis=analysis, glossary=glossary, raw_output=raw)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        return GlossResponse(error=f"Failed to parse LLM output as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gloss generation failed: {str(e)}")


@app.post("/gloss/generate/batch", response_model=list[GlossBatchItemResult], tags=["Gloss"], summary="Generate Gloss in Batch")
async def gloss_generate_batch(request: GlossBatchRequest) -> list[GlossBatchItemResult]:
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)
        results: list[GlossBatchItemResult] = []
        for idx, item in enumerate(request.items):
            try:
                _, data = generate_gloss(
                    model,
                    item.input_text,
                    ucca_interpretation=item.ucca_interpretation,
                    commentary_1=item.commentary_1,
                    commentary_2=item.commentary_2,
                    commentary_3=item.commentary_3,
                    sanskrit_text=item.sanskrit_text,
                )
                results.append(GlossBatchItemResult(
                    index=idx,
                    standardized_text=data.get("StandardizedText", {}).get("standardized_text"),
                    note=data.get("StandardizedText", {}).get("note"),
                    analysis=json.dumps(data.get("analysis", []), ensure_ascii=False),
                    glossary=data.get("Glossary", {}).get("glossary"),
                ))
            except Exception as e:
                results.append(GlossBatchItemResult(index=idx, error=str(e)))
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch gloss generation failed: {str(e)}")


@app.post("/gloss/generate/stream", tags=["Gloss"], summary="Generate Gloss via SSE Stream")
async def gloss_generate_stream(request: GlossBatchRequest):
    try:
        model_router = get_model_router()
        if not model_router.validate_model_availability(request.model_name):
            available_models = list(model_router.get_available_models().keys())
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' is not available. Available models: {available_models}")

        model = model_router.get_model(request.model_name, **request.model_params)
        items = [
            {
                "input_text": it.input_text,
                "ucca_interpretation": it.ucca_interpretation,
                "commentary_1": it.commentary_1,
                "commentary_2": it.commentary_2,
                "commentary_3": it.commentary_3,
                "sanskrit_text": it.sanskrit_text,
            }
            for it in request.items
        ]
        return EventSourceResponse(stream_gloss_generation(model, items, batch_size=request.batch_size), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming gloss generation failed: {str(e)}")


# --- Workflow Endpoint (Echo-based, prompt selection by combo key) ---

def _canonicalize_combo_key(combo_key: str) -> str:
    """Make combo_key order-independent by sorting tokens.

    Recognized tokens: source, ucca, gloss, sanskrit, commentariesK (K in {0,1,2,3,>3}).
    Any unknown tokens are preserved and sorted as well.
    """
    tokens = [t for t in combo_key.strip().split("+") if t]
    # Normalize 'commentaries' tokens to a count bucket if present
    normalized = []
    for t in tokens:
        if t.startswith("commentaries"):
            normalized.append(t)
        else:
            normalized.append(t)
    # Always include 'source' by default
    if "source" not in normalized:
        normalized.append("source")
    # Ensure unique and sorted
    normalized = sorted(set(normalized))
    return "+".join(normalized)


def _derive_commentaries_token(comments: list[str] | None) -> str | None:
    if comments is None:
        return None
    n = len(comments)
    if n > 3:
        return "commentaries>3"
    return f"commentaries{n}"


@app.post("/workflow/run", response_model=WorkflowResponse, tags=["Workflow"], summary="Run workflow by combo key")
async def workflow_run(request: WorkflowRunRequest) -> WorkflowResponse:
    """Echo-based workflow runner.

    - Selects prompt based on the provided path combo_key (order-independent).
    - Canonicalizes the provided combo_key regardless of which inputs are present.
    - Returns the inputs for UI testing.
    """
    inputs = request.input
    combo_key = request.combo_key
    model_name = request.model_name or "claude-sonnet-4-20250514"

    if not inputs.source:
        raise HTTPException(status_code=400, detail="'source' is required")

    provided_key = _canonicalize_combo_key(combo_key)

    # Always prefer the provided path key for prompt selection (order-independent)
    final_key = provided_key

    # Validate presence and bounds based on tokens
    tokens = set(final_key.split('+'))
    # commentaries length validation
    if inputs.commentaries is not None and len(inputs.commentaries) > 3:
        raise HTTPException(status_code=400, detail="At most 3 commentaries are allowed")
    # If combo specifies commentariesK where K>0, ensure provided
    requires_commentaries = any(t.startswith('commentaries') and t not in ['commentaries0'] for t in tokens)
    if requires_commentaries and (not inputs.commentaries or len(inputs.commentaries) == 0):
        raise HTTPException(status_code=400, detail="This combination requires commentaries, but none were provided")
    # If combo includes ucca/gloss/sanskrit ensure corresponding input present
    if 'ucca' in tokens and inputs.ucca is None:
        raise HTTPException(status_code=400, detail="Combo includes 'ucca' but no UCCA JSON was provided")
    if 'gloss' in tokens and inputs.gloss is None:
        raise HTTPException(status_code=400, detail="Combo includes 'gloss' but no Gloss JSON was provided")
    if 'sanskrit' in tokens and not inputs.sanskrit:
        raise HTTPException(status_code=400, detail="Combo includes 'sanskrit' but no Sanskrit text was provided")

    # Build translation instructions: no additions beyond source; obey target language if provided
    target_line = f"Translate into {inputs.target_language}." if inputs.target_language else "Translate into the requested target language."
    guidelines: list[str] = [
        target_line,
        "Do not add content beyond the source. No examples, adaptations, or expansions.",
        "Preserve meaning, nuance, and accuracy; avoid extraneous explanation.",
    ]
    if 'ucca' in tokens:
        guidelines.append("Use the UCCA structure to disambiguate roles, participants, and processes; do not include UCCA in the output.")
    if 'gloss' in tokens:
        guidelines.append("Use Gloss to prefer standardized term choices and respect any provided notes.")
    if inputs.commentaries and len(inputs.commentaries) > 0:
        guidelines.append("Leverage commentaries to resolve ambiguity; do not quote or cite them explicitly.")
    if 'sanskrit' in tokens:
        guidelines.append("Use Sanskrit to validate terms and transliterations where applicable; do not add Sanskrit unless necessary.")
    # No output structure restriction; return plain text translation only.

    prompt_text = "\n- " + "\n- ".join(guidelines)

    # Invoke LLM with a default generic prompt using the selected model
    model_router = get_model_router()
    if not model_router.validate_model_availability(model_name):
        available_models = list(model_router.get_available_models().keys())
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available. Available models: {available_models}")

    # Build a simple combined prompt with inputs for a generic echo/summary
    def block(title: str, body: str) -> str:
        return f"\n\n### {title}\n{body.strip()}" if body else ""

    # Render inputs sections
    commentary_text = "\n\n".join((inputs.commentaries or [])[:3]) if inputs.commentaries else ""
    # Formatted commentaries block for {commentaries}/{commenteries} placeholder
    commentaries_block = ""
    if inputs.commentaries:
        lines: list[str] = []
        for idx, c in enumerate(inputs.commentaries[:3]):
            c_str = (c or "").strip()
            if c_str:
                lines.append(f"commentary {idx+1}: {c_str}")
        commentaries_block = "\n".join(lines)
    ucca_text = json.dumps(inputs.ucca, ensure_ascii=False, indent=2) if isinstance(inputs.ucca, dict) else (inputs.ucca or "")
    gloss_text = json.dumps(inputs.gloss, ensure_ascii=False, indent=2) if isinstance(inputs.gloss, dict) else (inputs.gloss or "")
    sanskrit_text = inputs.sanskrit or ""

    # Support custom prompt if provided. It must include {source}; other placeholders optional.
    if request.custom_prompt:
        tmpl = request.custom_prompt
        if "{source}" not in tmpl:
            raise HTTPException(status_code=400, detail="custom_prompt must include {source}")
        # Build substitution map
        subs = {
            "source": inputs.source,
            "ucca": ucca_text,
            "gloss": gloss_text,
            "commentary1": (inputs.commentaries[0] if inputs.commentaries and len(inputs.commentaries) > 0 else ""),
            "commentary2": (inputs.commentaries[1] if inputs.commentaries and len(inputs.commentaries) > 1 else ""),
            "commentary3": (inputs.commentaries[2] if inputs.commentaries and len(inputs.commentaries) > 2 else ""),
            # Support a single block placeholder for all commentaries
            "commentaries": commentaries_block,
            # Common misspelling alias
            "commenteries": commentaries_block,
            "sanskrit": sanskrit_text,
            "target_language": (inputs.target_language or "").strip(),
        }
        # Protect allowed placeholders, escape all other braces, then restore
        allowed_placeholders = [
            "source","ucca","gloss","commentary1","commentary2","commentary3","commentaries","commenteries","sanskrit","target_language"
        ]
        sentinel_map = {name: f"<<PH_{name.upper()}>>" for name in allowed_placeholders}
        protected = tmpl
        for name, token in sentinel_map.items():
            protected = protected.replace(f"{{{name}}}", token)
        # Escape any remaining single braces to avoid format errors
        protected = protected.replace("{", "{{").replace("}", "}}")
        # Restore placeholders
        for name, token in sentinel_map.items():
            protected = protected.replace(token, f"{{{name}}}")
        combined_prompt = protected.format(**subs)
    else:
        combined_prompt = (
            f"You are a professional translator for Buddhist literature.\n"
            + block("Instructions", "\n- " + "\n- ".join(guidelines))
            + block("Source", inputs.source)
            + block("Commentaries (up to 3)", commentary_text)
            + block("UCCA", ucca_text)
            + block("Gloss", gloss_text)
            + block("Sanskrit", sanskrit_text)
            + "\n\nReturn only the translated text with no additional commentary."
        )

    llm_output: str = ""
    translation_text: str | None = None
    try:
        model = model_router.get_model(model_name, **(request.model_params or {}))
        # Single plain-text response; no structured schema
        # Force plain text only for Gemini models; other providers don't use this flag
        if (model_name or "").startswith("gemini"):
            resp = await model.ainvoke(
                combined_prompt,
                generation_config={"response_mime_type": "text/plain"}
            )
        else:
            resp = await model.ainvoke(combined_prompt)
        llm_output = getattr(resp, "content", str(resp)) or ""
        if isinstance(llm_output, list):
            try:
                llm_output = "\n".join([str(p) for p in llm_output])
            except Exception:
                llm_output = str(llm_output)
        translation_text = llm_output
    except Exception as e:
        # Return the error inside the payload for easier UI debugging instead of failing the request entirely
        llm_output = f"LLM invocation error: {str(e)}"
        translation_text = None

    return WorkflowResponse(
        combo_key=final_key,
        translation=translation_text,
    )


@app.post("/workflow/run/batch", response_model=list[WorkflowBatchItemResult], tags=["Workflow"], summary="Run workflow in batch")
async def workflow_run_batch(request: WorkflowBatchRequest) -> list[WorkflowBatchItemResult]:
    # Reuse the single-run logic in a loop
    results: list[WorkflowBatchItemResult] = []
    for idx, item in enumerate(request.items):
        try:
            single = WorkflowRunRequest(
                combo_key=request.combo_key,
                input=item,
                model_name=request.model_name,
                model_params=request.model_params,
            )
            resp = await workflow_run(single)
            results.append(WorkflowBatchItemResult(index=idx, translation=resp.translation))
        except HTTPException as he:
            results.append(WorkflowBatchItemResult(index=idx, error=str(he.detail)))
        except Exception as e:
            results.append(WorkflowBatchItemResult(index=idx, error=str(e)))
    return results