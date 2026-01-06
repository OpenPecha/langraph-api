"""Glossary extraction endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from ..schemas.glossary import GlossaryExtractionRequest
from ..models.glossary import Glossary
from ..models.model_router import get_model_router
from ..prompts.tibetan_buddhist import GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from ..workflows.streaming import stream_glossary_progress
from ..api.dependencies import router_limiter

router = APIRouter(prefix="/glossary", tags=["Glossary"])


@router.post(
    "/extract", 
    response_model=Glossary,
    dependencies=[Depends(router_limiter)],
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


@router.post(
    "/extract/stream", 
    dependencies=[Depends(router_limiter)],
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

