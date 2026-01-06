"""Standardization endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from collections import defaultdict
from ..models.standardization import (
    AnalysisRequest,
    AnalysisResponse,
    StandardizationRequest,
    StandardizationResponse,
    RetranslationResponse,
)
from ..models.glossary import Glossary
from ..models.model_router import get_model_router
from ..prompts.tibetan_buddhist import RETRANSLATION_PROMPT, GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from ..workflows.streaming import stream_standardization_progress
from ..api.dependencies import router_limiter

router = APIRouter(prefix="/standardize", tags=["Standardization"])


@router.post(
    "/analyze", 
    response_model=AnalysisResponse,
    dependencies=[Depends(router_limiter)],
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


@router.post(
    "/apply", 
    response_model=StandardizationResponse,
    dependencies=[Depends(router_limiter)],
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


@router.post(
    "/apply/stream",
    dependencies=[Depends(router_limiter)],
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

