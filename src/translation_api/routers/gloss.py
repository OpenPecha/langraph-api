"""Gloss generation endpoints."""

import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from ..models.gloss import GlossRequest, GlossResponse, GlossBatchRequest, GlossBatchItemResult
from ..models.model_router import get_model_router
from ..workflows.gloss import generate_gloss, stream_gloss_generation

router = APIRouter(prefix="/gloss", tags=["Gloss"])


@router.post("/generate", response_model=GlossResponse, summary="Generate Gloss for a Single Text")
async def gloss_generate(request: GlossRequest) -> GlossResponse:
    """Generate gloss for a single text."""
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


@router.post("/generate/batch", response_model=list[GlossBatchItemResult], summary="Generate Gloss in Batch")
async def gloss_generate_batch(request: GlossBatchRequest) -> list[GlossBatchItemResult]:
    """Generate gloss for multiple texts in batch."""
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


@router.post("/generate/stream", summary="Generate Gloss via SSE Stream")
async def gloss_generate_stream(request: GlossBatchRequest):
    """Generate gloss for multiple texts with SSE streaming."""
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

