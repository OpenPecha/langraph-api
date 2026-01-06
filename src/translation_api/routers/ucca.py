"""UCCA graph generation endpoints."""

import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from ..models.ucca import UCCARequest, UCCAResponse, UCCABatchRequest, UCCABatchItemResult
from ..schemas.ucca import UCCAErrorResponse
from ..models.model_router import get_model_router
from ..workflows.ucca import generate_ucca_graph, stream_ucca_generation

router = APIRouter(prefix="/ucca", tags=["UCCA"])


@router.post("/generate", response_model=UCCAResponse, summary="Generate a Single UCCA Graph", responses={
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


@router.post("/generate/batch", response_model=list[UCCABatchItemResult], summary="Generate UCCA Graphs in Batch")
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


@router.post("/generate/stream", summary="Generate UCCA Graphs via SSE Stream")
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

