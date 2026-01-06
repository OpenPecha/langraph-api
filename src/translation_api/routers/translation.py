"""Translation endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from ..schemas.translation import TranslationAPIRequest, TranslationAPIResponse, SingleTranslationRequest
from ..workflows.translation_state import TranslationRequest
from ..workflows.streaming import stream_translation_progress, stream_single_translation_progress
from ..models.model_router import get_model_router
from ..config import get_settings
from ..api.dependencies import router_limiter
import sys
import os

# Import from root level graph module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph import run_translation_workflow

router = APIRouter(prefix="/translate", tags=["Translation"])
settings = get_settings()


@router.post(
    "", 
    response_model=TranslationAPIResponse,
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
            user_rules=request.user_rules,
            context=request.context
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


@router.post("/single", response_model=TranslationAPIResponse, dependencies=[Depends(router_limiter)])
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


@router.post(
    "/stream", 
    tags=["Streaming"],
    dependencies=[Depends(router_limiter)],
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
            user_rules=request.user_rules,
            context=request.context
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


@router.post("/single/stream", tags=["Streaming"], dependencies=[Depends(router_limiter)])
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

