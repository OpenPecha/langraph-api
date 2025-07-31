"""Streaming workflow with real-time progress updates via Server-Sent Events."""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from .translation_state import TranslationRequest, TranslationResult
from ..models.model_router import get_model_router
from ..prompts.tibetan_buddhist import get_translation_prompt
from ..utils.helpers import clean_translation_text, parse_batch_translation_response

from langchain_core.messages import HumanMessage


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
    
    # Emit initialization event
    yield ProgressEvent("initialization", {
        "status": "starting",
        "total_texts": total_texts,
        "target_language": request.target_language,
        "model": request.model_name,
        "batch_size": request.batch_size
    }).to_sse_format()
    
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
            
            try:
                if len(batch["texts"]) == 1:
                    # Single text translation
                    prompt = get_translation_prompt(
                        source_text=batch["texts"][0],
                        target_language=request.target_language,
                        text_type=request.text_type,
                        user_rules=request.user_rules
                    )
                    
                    # Emit translation start for single text
                    yield ProgressEvent("translation_start", {
                        "status": "translating_single_text",
                        "text_preview": batch["texts"][0][:50] + "..." if len(batch["texts"][0]) > 50 else batch["texts"][0]
                    }).to_sse_format()
                    
                    message = HumanMessage(content=prompt)
                    response = model.invoke([message])
                    clean_translation = clean_translation_text(response.content)
                    
                    result = TranslationResult(
                        original_text=batch["texts"][0],
                        translated_text=clean_translation,
                        metadata={
                            "batch_id": batch["batch_id"],
                            "model_used": request.model_name,
                            "text_type": request.text_type
                        }
                    )
                    all_results.append(result)
                    
                else:
                    # Batch translation
                    prompt = get_translation_prompt(
                        source_text="",
                        target_language=request.target_language,
                        text_type=request.text_type,
                        batch_texts=batch["texts"],
                        user_rules=request.user_rules
                    )
                    
                    # Emit batch translation start
                    yield ProgressEvent("translation_start", {
                        "status": "translating_batch",
                        "batch_size": len(batch["texts"])
                    }).to_sse_format()
                    
                    message = HumanMessage(content=prompt)
                    response = model.invoke([message])
                    translated_texts = parse_batch_translation_response(response.content)
                    
                    # Process each translation in the batch
                    for i, original_text in enumerate(batch["texts"]):
                        if i < len(translated_texts):
                            clean_translation = clean_translation_text(translated_texts[i])
                        else:
                            clean_translation = "Translation failed"
                        
                        result = TranslationResult(
                            original_text=original_text,
                            translated_text=clean_translation,
                            metadata={
                                "batch_id": batch["batch_id"],
                                "model_used": request.model_name,
                                "text_type": request.text_type,
                                "batch_index": i
                            }
                        )
                        all_results.append(result)
                        processed_texts += 1
                        
                        # Emit individual text completion
                        yield ProgressEvent("text_completed", {
                            "status": "text_translated",
                            "text_number": processed_texts,
                            "total_texts": total_texts,
                            "progress_percent": int((processed_texts / total_texts) * 100),
                            "translation_preview": clean_translation[:100] + "..." if len(clean_translation) > 100 else clean_translation
                        }).to_sse_format()
                
                # Update processed count for single text
                if len(batch["texts"]) == 1:
                    processed_texts += 1
                
                # Emit batch completion with results
                batch_time = time.time() - batch_start_time
                
                # Send batch results immediately
                batch_results = []
                if len(batch["texts"]) == 1:
                    batch_results = [all_results[-1]]  # Last added result
                else:
                    # Get the results for this batch
                    batch_results = all_results[-len(batch["texts"]):]
                
                yield ProgressEvent("batch_completed", {
                    "status": "batch_completed",
                    "batch_number": batch_idx + 1,
                    "batch_id": batch["batch_id"],
                    "processing_time": round(batch_time, 2),
                    "texts_processed": len(batch["texts"]),
                    "cumulative_progress": int((processed_texts / total_texts) * 100),
                    "batch_results": [
                        {
                            "original_text": result.original_text,
                            "translated_text": result.translated_text,
                            "metadata": result.metadata
                        }
                        for result in batch_results
                    ]
                }).to_sse_format()
                
            except Exception as e:
                # Emit batch error
                yield ProgressEvent("batch_error", {
                    "status": "batch_failed",
                    "batch_number": batch_idx + 1,
                    "batch_id": batch["batch_id"],
                    "error": str(e)
                }).to_sse_format()
                
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
            "results": [
                {
                    "original_text": result.original_text,
                    "translated_text": result.translated_text,
                    "metadata": result.metadata
                }
                for result in all_results
            ]
        }).to_sse_format()
        
    except Exception as e:
        # Emit global error
        yield ProgressEvent("error", {
            "status": "failed",
            "error": str(e),
            "processed_texts": processed_texts,
            "total_texts": total_texts
        }).to_sse_format()


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