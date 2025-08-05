"""
LangGraph workflow for Tibetan Buddhist text translation.

This module implements a flexible, extensible workflow for translating
Tibetan Buddhist texts with support for batch processing and multiple models.
"""

import time
import uuid
from typing import List, Dict, Any
import asyncio
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from src.translation_api.workflows.translation_state import (
    TranslationWorkflowState,
    TranslationRequest,
    TranslationBatch,
    TranslationResult,
    BatchResult
)
from src.translation_api.models.model_router import get_model_router
from src.translation_api.models.glossary import Glossary
from src.translation_api.prompts.tibetan_buddhist import get_translation_prompt, GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from src.translation_api.config import get_settings
from src.translation_api.utils.helpers import clean_translation_text, parse_batch_translation_response
from src.translation_api.cache import get_cache


def initialize_workflow(state: TranslationWorkflowState) -> TranslationWorkflowState:
    """
    Initialize the translation workflow state.
    
    This node prepares the workflow by:
    - Setting up initial state
    - Creating batches from input texts
    - Initializing counters and metadata
    """
    request = state["original_request"]
    
    # Create batches
    batches = []
    texts = request.texts
    batch_size = min(request.batch_size, get_settings().max_batch_size)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch = TranslationBatch(
            batch_id=str(uuid.uuid4()),
            texts=batch_texts,
            target_language=request.target_language,
            text_type=request.text_type,
            model_name=request.model_name,
            model_params=request.model_params,
            user_rules=request.user_rules
        )
        batches.append(batch)
    
    # Update state
    state.update({
        "batches": batches,
        "current_batch_index": 0,
        "batch_results": [],
        "final_results": [],
        "total_texts": len(texts),
        "processed_texts": 0,
        "workflow_start_time": time.time(),
        "workflow_status": "running",
        "errors": [],
        "retry_count": 0,
        "model_name": request.model_name,
        "model_params": request.model_params,
        "custom_steps": {},
        "metadata": {
            "initialized_at": datetime.now().isoformat(),
            "total_batches": len(batches)
        }
    })
    
    return state

def _extract_glossary_for_pair(model: Any, source_text: str, translated_text: str) -> List:
    """Helper to extract glossary for a single source/translation pair."""
    try:
        structured_model = model.with_structured_output(Glossary)
        prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
            text_pairs=f"Source: {source_text}\\nTranslated: {translated_text}\\n\\n"
        )
        glossary_result = structured_model.invoke(prompt)
        return glossary_result.terms if glossary_result else []
    except Exception:
        # On failure, return an empty list. The error is logged in the main function.
        return []


def process_batch(state: TranslationWorkflowState) -> TranslationWorkflowState:
    """
    Process a single batch of translations.
    
    This is the core translation node that:
    - Takes the current batch
    - Uses the selected model to translate texts
    - Handles errors and retries
    - Updates processing state
    """
    current_index = state["current_batch_index"]
    batches = state["batches"]
    
    if current_index >= len(batches):
        state["workflow_status"] = "completed"
        return state
    
    current_batch = batches[current_index]
    batch_start_time = time.time()
    
    try:
        # Get the model
        model_router = get_model_router()
        model = model_router.get_model(
            current_batch.model_name,
            **current_batch.model_params
        )
        
        cache = get_cache()
        translation_results = []

        # Process one by one to allow for caching
        for text_to_translate in current_batch.texts:
            # 1. Check translation cache
            translation_key = cache.get_translation_cache_key(
                text_to_translate, 
                current_batch.target_language, 
                current_batch.text_type, 
                current_batch.model_name, 
                current_batch.user_rules
            )
            cached_translation = cache.get_translation(translation_key)
            
            if cached_translation:
                clean_translation = cached_translation
            else:
                # If not cached, call the model
                prompt = get_translation_prompt(
                    source_text=text_to_translate,
                    target_language=current_batch.target_language,
                    text_type=current_batch.text_type,
                    user_rules=current_batch.user_rules
                )
                message = HumanMessage(content=prompt)
                response = model.invoke([message])
                clean_translation = clean_translation_text(response.content)
                cache.set_translation(translation_key, clean_translation)

            # 2. Check glossary cache
            glossary_key = cache.get_glossary_cache_key(text_to_translate, clean_translation, current_batch.model_name)
            cached_glossary = cache.get_glossary(glossary_key)
            glossary_terms = []

            if cached_glossary is not None:
                glossary_terms = cached_glossary
            else:
                # If not cached, extract glossary
                try:
                    structured_model = model.with_structured_output(Glossary)
                    prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                        text_pairs=f"Source: {text_to_translate}\\nTranslated: {clean_translation}\\n\\n"
                    )
                    glossary_result = structured_model.invoke(prompt)
                    if glossary_result and glossary_result.terms:
                        glossary_terms = glossary_result.terms
                    cache.set_glossary(glossary_key, glossary_terms)
                except Exception as e:
                    # Log error but don't fail, just return no terms
                    pass
            
            # 3. Assemble the result
            result = TranslationResult(
                original_text=text_to_translate,
                translated_text=clean_translation,
                metadata={
                    "batch_id": current_batch.batch_id,
                    "model_used": current_batch.model_name,
                    "text_type": current_batch.text_type
                }
            )
            translation_results.append(result)

        # Create batch result
        processing_time = time.time() - batch_start_time
        batch_result = BatchResult(
            batch_id=current_batch.batch_id,
            results=translation_results,
            processing_time=processing_time,
            model_used=current_batch.model_name,
            success=True
        )
        
        # Update state
        state["batch_results"].append(batch_result)
        state["final_results"].extend(translation_results)
        state["processed_texts"] += len(current_batch.texts)
        state["current_batch_index"] += 1
        
    except Exception as e:
        # Handle batch processing error
        processing_time = time.time() - batch_start_time
        error_result = BatchResult(
            batch_id=current_batch.batch_id,
            results=[],
            processing_time=processing_time,
            model_used=current_batch.model_name,
            success=False,
            error_message=str(e)
        )
        
        state["batch_results"].append(error_result)
        state["errors"].append({
            "batch_id": current_batch.batch_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["current_batch_index"] += 1
    
    return state


def extract_glossaries_in_parallel(state: TranslationWorkflowState) -> TranslationWorkflowState:
    """
    After all translations are done, extract glossaries for all pairs in parallel.
    """
    try:
        model_router = get_model_router()
        model = model_router.get_model(state["model_name"] or "claude")
        structured_model = model.with_structured_output(Glossary)

        # Create a list of prompts, one for each translation pair
        prompts = []
        for result in state["final_results"]:
            prompt_text = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                text_pairs=f"Source: {result.original_text}\\nTranslated: {result.translated_text}\\n\\n"
            )
            prompts.append(HumanMessage(content=prompt_text))

        # Batch the prompts to avoid sending too many at once
        batch_size = state["original_request"].batch_size
        all_terms = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Use the batch method to process each chunk of prompts in parallel
            glossary_results = structured_model.batch(batch_prompts)
            
            for gloss in glossary_results:
                if gloss and gloss.terms:
                    all_terms.extend(gloss.terms)
        
        state["glossary"] = Glossary(terms=all_terms)

    except Exception as e:
        state["errors"].append({
            "step": "extract_glossaries_in_parallel",
            "error": f"Failed to generate glossaries in parallel: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        state["glossary"] = Glossary(terms=[])

    return state


def check_completion(state: TranslationWorkflowState) -> str:
    """
    Check if workflow is complete or should continue processing.
    
    Returns:
        "continue" if more batches to process
        "extract_glossaries_in_parallel" if all batches are complete
    """
    current_index = state["current_batch_index"]
    total_batches = len(state["batches"])
    
    if current_index >= total_batches:
        return "extract_glossaries_in_parallel"
    else:
        return "continue"


def finalize_workflow(state: TranslationWorkflowState) -> TranslationWorkflowState:
    """
    Finalize the translation workflow.
    
    This node:
    - Calculates final statistics
    - Sets completion status
    - Prepares final output
    """
    total_time = time.time() - state["workflow_start_time"]
    successful_batches = len([r for r in state["batch_results"] if r.success])
    failed_batches = len([r for r in state["batch_results"] if not r.success])
    
    state["workflow_status"] = "completed"
    state["metadata"].update({
        "completed_at": datetime.now().isoformat(),
        "total_processing_time": total_time,
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "total_translations": len(state["final_results"]),
        "success_rate": len(state["final_results"]) / state["total_texts"] if state["total_texts"] > 0 else 0
    })
    
    return state


def create_translation_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for translation.
    
    This workflow is designed to be easily extensible. New nodes can be added
    for features like:
    - Quality review
    - Post-editing
    - Content filtering
    - Custom validation steps
    
    Returns:
        Configured StateGraph for translation workflow
    """
    # Create the workflow graph
    workflow = StateGraph(TranslationWorkflowState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("process_batch", process_batch)
    workflow.add_node("extract_glossaries_in_parallel", extract_glossaries_in_parallel)
    workflow.add_node("finalize", finalize_workflow)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "process_batch")
    
    # Conditional edge for batch processing loop
    workflow.add_conditional_edges(
        "process_batch",
        check_completion,
        {
            "continue": "process_batch",  # Loop back for next batch
            "extract_glossaries_in_parallel": "extract_glossaries_in_parallel"
        }
    )
    
    workflow.add_edge("extract_glossaries_in_parallel", "finalize")
    # End the workflow
    workflow.add_edge("finalize", END)
    
    return workflow


# Convenience function for easy workflow execution
async def run_translation_workflow(request: TranslationRequest) -> TranslationWorkflowState:
    """
    Execute the translation workflow with the given request.
    
    Args:
        request: Translation request with texts and configuration
        
    Returns:
        Final workflow state with translation results
    """
    workflow = create_translation_workflow()
    app = workflow.compile()
    
    # Initialize state
    initial_state: TranslationWorkflowState = {
        "original_request": request,
        "batches": [],
        "current_batch_index": 0,
        "batch_results": [],
        "final_results": [],
        "total_texts": 0,
        "processed_texts": 0,
        "workflow_start_time": 0.0,
        "workflow_status": "initializing",
        "errors": [],
        "retry_count": 0,
        "model_name": request.model_name,
        "model_params": request.model_params,
        "custom_steps": {},
        "metadata": {}
    }
    
    # Run the workflow
    final_state = await app.ainvoke(initial_state)
    return final_state