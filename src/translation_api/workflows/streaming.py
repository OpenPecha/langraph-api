"""Streaming workflow with real-time progress updates via Server-Sent Events."""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List
from datetime import datetime

from .translation_state import TranslationRequest, TranslationResult
from ..models.model_router import get_model_router
from ..models.glossary import Glossary
from ..prompts.tibetan_buddhist import get_translation_prompt, GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from ..utils.helpers import clean_translation_text, parse_batch_translation_response
from langchain_core.messages import HumanMessage
from ..cache import get_cache
from ..models.standardization import StandardizationRequest, RetranslationResponse
from ..prompts.tibetan_buddhist import RETRANSLATION_PROMPT

async def _extract_glossary_for_pair_async(model: Any, source_text: str, translated_text: str) -> List:
    """Async helper to extract glossary for a single source/translation pair."""
    try:
        structured_model = model.with_structured_output(Glossary)
        prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
            text_pairs=f"Source: {source_text}\\nTranslated: {translated_text}\\n\\n"
        )
        glossary_result = await structured_model.ainvoke(prompt)
        return glossary_result.terms if glossary_result else []
    except Exception:
        return []

class ProgressEvent:
    """Progress event for SSE streaming."""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now().isoformat()
    
    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format."""
        event_data = {
            "timestamp": self.timestamp,
            "type": self.event_type,
            **self.data
        }
        return f"data: {json.dumps(event_data)}\n\n"


async def stream_translation_progress(request: TranslationRequest) -> AsyncGenerator[str, None]:
    """
    Stream translation progress with real-time updates.
    
    Args:
        request: Translation request
        
    Yields:
        SSE formatted progress events
    """
    start_time = time.time()
    total_texts = len(request.texts)
    cache = get_cache()
    
    # Emit initialization event
    yield ProgressEvent("initialization", {
        "status": "starting",
        "total_texts": total_texts,
        "target_language": request.target_language,
        "model": request.model_name,
        "batch_size": request.batch_size
    }).to_sse_format()
    
    # Yield control to event loop
    await asyncio.sleep(0)
    
    # Create batches
    batch_size = min(request.batch_size, 50)  # Max batch size
    batches = []
    
    for i in range(0, len(request.texts), batch_size):
        batch_texts = request.texts[i:i + batch_size]
        batches.append({
            "batch_id": f"batch_{i//batch_size + 1}",
            "texts": batch_texts,
            "start_index": i
        })
    
    # Emit batch planning event
    yield ProgressEvent("planning", {
        "status": "batches_created",
        "total_batches": len(batches),
        "batch_size": batch_size
    }).to_sse_format()
    
    # Yield control to event loop
    await asyncio.sleep(0)
    
    # Process each batch
    all_results = []
    processed_texts = 0
    
    try:
        model_router = get_model_router()
        model = model_router.get_model(request.model_name, **request.model_params)
        
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            
            # Emit batch start event
            yield ProgressEvent("batch_start", {
                "status": "processing_batch",
                "batch_number": batch_idx + 1,
                "batch_id": batch["batch_id"],
                "texts_in_batch": len(batch["texts"]),
                "progress_percent": int((processed_texts / total_texts) * 100)
            }).to_sse_format()
            
            await asyncio.sleep(0)
            
            batch_results = []
            try:
                # Use a single prompt for the entire batch
                model_router = get_model_router()
                model = model_router.get_model(request.model_name, **request.model_params)

                for i, text_to_translate in enumerate(batch["texts"]):
                    # 1. Check translation cache
                    translation_key = cache.get_translation_cache_key(
                        text_to_translate, request.target_language, request.text_type, request.model_name, request.user_rules
                    )
                    cached_translation = cache.get_translation(translation_key)

                    if cached_translation:
                        clean_translation = cached_translation
                    else:
                        prompt = get_translation_prompt(
                            source_text=text_to_translate,
                            target_language=request.target_language,
                            text_type=request.text_type,
                            user_rules=request.user_rules
                        )
                        message = HumanMessage(content=prompt)
                        response = await model.ainvoke([message])
                        clean_translation = clean_translation_text(response.content)
                        cache.set_translation(translation_key, clean_translation)

                    # 2. Check glossary cache
                    glossary_key = cache.get_glossary_cache_key(text_to_translate, clean_translation, request.model_name)
                    cached_glossary = cache.get_glossary(glossary_key)
                    glossary_terms = []

                    if cached_glossary is not None:
                        glossary_terms = cached_glossary
                    else:
                        try:
                            glossary_terms = await _extract_glossary_for_pair_async(model, text_to_translate, clean_translation)
                            cache.set_glossary(glossary_key, glossary_terms)
                        except Exception:
                            pass # Fail silently for the streamer

                    result = TranslationResult(
                        original_text=text_to_translate,
                        translated_text=clean_translation,
                        glossary=glossary_terms,
                        metadata={
                            "batch_id": batch["batch_id"],
                            "model_used": request.model_name,
                            "text_type": request.text_type,
                            "batch_index": i
                        }
                    )
                    batch_results.append(result)
                    all_results.append(result)
                    processed_texts += 1
                
                # Emit batch completion with all results for this batch
                yield ProgressEvent("batch_completed", {
                    "status": "batch_completed",
                    "batch_number": batch_idx + 1,
                    "batch_id": batch["batch_id"],
                    "processing_time": time.time() - batch_start_time,
                    "texts_processed": len(batch["texts"]),
                    "cumulative_progress": int((processed_texts / total_texts) * 100),
                    "batch_results": [res.dict() for res in batch_results]
                }).to_sse_format()
                
                await asyncio.sleep(0)
                
            except Exception as e:
                # Emit batch error
                yield ProgressEvent("batch_error", {
                    "status": "batch_failed",
                    "batch_number": batch_idx + 1,
                    "batch_id": batch["batch_id"],
                    "error": str(e)
                }).to_sse_format()
                
                await asyncio.sleep(0)
                
                # Skip failed texts in progress count
                processed_texts += len(batch["texts"])
        
        # Emit final completion
        total_time = time.time() - start_time
        yield ProgressEvent("completion", {
            "status": "completed",
            "total_texts": total_texts,
            "successful_translations": len(all_results),
            "total_processing_time": round(total_time, 2),
            "average_time_per_text": round(total_time / total_texts, 2) if total_texts > 0 else 0,
            "results": [res.dict() for res in all_results]
        }).to_sse_format()
        
        # Yield control to event loop
        await asyncio.sleep(0)
        
    except Exception as e:
        # Emit global error
        yield ProgressEvent("error", {
            "status": "failed",
            "error": str(e),
            "processed_texts": processed_texts,
            "total_texts": total_texts
        }).to_sse_format()
        
        # Yield control to event loop
        await asyncio.sleep(0)


async def stream_single_translation_progress(
    text: str, 
    target_language: str, 
    model_name: str = "claude",
    text_type: str = "Buddhist text",
    model_params: Dict[str, Any] = None,
    user_rules: str = None
) -> AsyncGenerator[str, None]:
    """
    Stream progress for a single text translation.
    
    Args:
        text: Text to translate
        target_language: Target language
        model_name: Model to use
        text_type: Type of Buddhist text
        model_params: Additional model parameters
        
    Yields:
        SSE formatted progress events
    """
    if model_params is None:
        model_params = {}
    
    request = TranslationRequest(
        texts=[text],
        target_language=target_language,
        model_name=model_name,
        text_type=text_type,
        batch_size=1,
        model_params=model_params,
        user_rules=user_rules
    )
    
    async for event in stream_translation_progress(request):
        yield event
        
        # Yield control to event loop
        await asyncio.sleep(0)


async def stream_glossary_progress(request: "GlossaryExtractionRequest") -> AsyncGenerator[str, None]:
    """Generator for streaming glossary extraction progress."""
    
    yield ProgressEvent("glossary_extraction_start", {
        "status": "starting",
        "message": f"Starting glossary extraction for {len(request.items)} items..."
    }).to_sse_format()
    await asyncio.sleep(0)

    try:
        model_router = get_model_router()
        model = model_router.get_model(request.model_name)
        structured_model = model.with_structured_output(Glossary)

        prompts = [
            GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                text_pairs=f"Source: {item.original_text}\\nTranslated: {item.translated_text}\\n\\n"
            ) for item in request.items
        ]

        all_terms = []
        for i in range(0, len(prompts), request.batch_size):
            batch_prompts = prompts[i:i + request.batch_size]
            
            glossary_results = await structured_model.abatch(batch_prompts)
            
            batch_terms = []
            for gloss in glossary_results:
                if gloss and gloss.terms:
                    batch_terms.extend(gloss.terms)
                    all_terms.extend(gloss.terms)

            yield ProgressEvent("glossary_batch_completed", {
                "status": "batch_complete",
                "terms": [term.dict() for term in batch_terms]
            }).to_sse_format()
            await asyncio.sleep(0)

        yield ProgressEvent("completion", {
            "status": "completed",
            "glossary": {"terms": [term.dict() for term in all_terms]}
        }).to_sse_format()
        await asyncio.sleep(0)

    except Exception as e:
        yield ProgressEvent("error", {
            "status": "failed",
            "error": str(e)
        }).to_sse_format()
        await asyncio.sleep(0)


async def stream_standardization_progress(request: "StandardizationRequest") -> AsyncGenerator[str, None]:
    """Generator for streaming standardization progress."""
    
    yield ProgressEvent("standardization_start", {
        "status": "starting",
        "message": "Starting standardization process..."
    }).to_sse_format()
    await asyncio.sleep(0)

    try:
        model_router = get_model_router()
        model = model_router.get_model(request.model_name)
        retranslation_model = model.with_structured_output(RetranslationResponse)
        glossary_model = model.with_structured_output(Glossary)

        source_words_to_standardize = {pair.source_word for pair in request.standardization_pairs}
        rules_block = ""
        for pair in request.standardization_pairs:
            rules_block += f'- For the source term "{pair.source_word}", use the exact translation "{pair.standardized_translation}".\\n'

        updated_count = 0
        for i, item in enumerate(request.items):
            if any(word in item.original_text for word in source_words_to_standardize):
                yield ProgressEvent("retranslation_start", {
                    "status": "retranslating_item",
                    "index": i
                }).to_sse_format()
                await asyncio.sleep(0)

                prompt = RETRANSLATION_PROMPT.format(
                    user_rules=request.user_rules or "No specific user rules provided.",
                    standardization_rules_block=rules_block,
                    original_text=item.original_text,
                    original_translation=item.translated_text
                )
                
                response = await retranslation_model.ainvoke(prompt)
                new_translation = response.new_translation
                item.translated_text = new_translation

                glossary_prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                    text_pairs=f"Source: {item.original_text}\\nTranslated: {new_translation}\\n\\n"
                )
                glossary_result = await glossary_model.ainvoke(glossary_prompt)
                item.glossary = glossary_result.terms if glossary_result else []
                
                updated_count += 1
                yield ProgressEvent("retranslation_completed", {
                    "status": "item_updated",
                    "index": i,
                    "updated_item": item.dict()
                }).to_sse_format()
                await asyncio.sleep(0)
        
        yield ProgressEvent("completion", {
            "status": "completed",
            "message": f"Standardization complete. {updated_count} items updated."
        }).to_sse_format()
        await asyncio.sleep(0)

    except Exception as e:
        yield ProgressEvent("error", {
            "status": "failed",
            "error": str(e)
        }).to_sse_format()
        await asyncio.sleep(0)