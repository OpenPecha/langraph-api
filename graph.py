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
from src.translation_api.prompts.tibetan_buddhist import get_translation_prompt
from src.translation_api.config import get_settings
from src.translation_api.utils.helpers import clean_translation_text, parse_batch_translation_response


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
        
        # Prepare translation results
        translation_results = []
        
        if len(current_batch.texts) == 1:
            # Single text translation
            prompt = get_translation_prompt(
                source_text=current_batch.texts[0],
                target_language=current_batch.target_language,
                text_type=current_batch.text_type,
                user_rules=current_batch.user_rules
            )
            
            message = HumanMessage(content=prompt)
            response = model.invoke([message])
            
            # Clean the translation text
            clean_translation = clean_translation_text(response.content)
            
            result = TranslationResult(
                original_text=current_batch.texts[0],
                translated_text=clean_translation,
                metadata={
                    "batch_id": current_batch.batch_id,
                    "model_used": current_batch.model_name,
                    "text_type": current_batch.text_type
                }
            )
            translation_results.append(result)
            
        else:
            # Batch translation
            prompt = get_translation_prompt(
                source_text="",  # Not used for batch
                target_language=current_batch.target_language,
                text_type=current_batch.text_type,
                batch_texts=current_batch.texts,
                user_rules=current_batch.user_rules
            )
            
            message = HumanMessage(content=prompt)
            response = model.invoke([message])
            
            # Parse batch response using helper function
            translated_texts = parse_batch_translation_response(response.content)
            
            for i, original_text in enumerate(current_batch.texts):
                if i < len(translated_texts):
                    clean_translation = clean_translation_text(translated_texts[i])
                else:
                    clean_translation = "Translation failed"
                
                result = TranslationResult(
                    original_text=original_text,
                    translated_text=clean_translation,
                    metadata={
                        "batch_id": current_batch.batch_id,
                        "model_used": current_batch.model_name,
                        "text_type": current_batch.text_type,
                        "batch_index": i
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


def check_completion(state: TranslationWorkflowState) -> str:
    """
    Check if workflow is complete or should continue processing.
    
    Returns:
        "continue" if more batches to process
        "finalize" if all batches are complete
    """
    current_index = state["current_batch_index"]
    total_batches = len(state["batches"])
    
    if current_index >= total_batches:
        return "finalize"
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
            "finalize": "finalize"        # Move to finalization
        }
    )
    
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