"""System endpoints: health checks, models, cache management."""

from fastapi import APIRouter, HTTPException
from ..schemas.system import HealthResponse
from ..models.model_router import get_model_router
from ..cache import get_cache

router = APIRouter(prefix="", tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API status and available translation models."""
    model_router = get_model_router()
    available_models = model_router.get_available_models()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_models=available_models
    )


@router.get("/models", summary="Get All Supported Models")
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


@router.post("/system/clear-cache", summary="Clear Server-Side Cache")
async def clear_cache():
    """
    Clears the server-side cache for all memoized functions. This is useful
    when you want to ensure you are getting fresh results from the language models,
    bypassing any previously cached translations or analyses.
    """
    cache = get_cache()
    cleared_count = cache.clear()
    return {"status": "cache_cleared", "cleared_items": cleared_count}

