"""Dharmamitra proxy endpoints."""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import httpx
from ..schemas.dharmamitra import DharmamitraKnnRequest, DharmamitraGeminiRequest
from ..config import get_settings

router = APIRouter(prefix="/dharmamitra", tags=["Dharmamitra"])


@router.post("/knn-translate-mitra", summary="Proxy: KNN Translate Mitra (Streaming)")
async def dharmamitra_knn_translate_mitra(request: DharmamitraKnnRequest):
    """Proxy to Dharmamitra KNN Translate Mitra SSE endpoint.

    Upstreams to https://dharmamitra.org/api-search/knn-translate-mitra/
    Returns text/event-stream forwarding 'data:' chunks.
    """
    url = "https://dharmamitra.org/api-search/knn-translate-mitra/"
    pwd = get_settings().dharmamitra_password or request.password
    if not pwd:
        raise HTTPException(status_code=400, detail="DHARMAMITRA_PASSWORD not set and no password provided")

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(url, json={
                "query": request.query,
                "language": (request.language or "").lower(),
                "password": pwd,
                "do_grammar": False,
            })
            # Dharmamitra returns SSE in the body as text; forward lines with data:
            text = resp.text
            # Normalize CRLF
            buf = text.replace("\r\n", "\n")
            for line in buf.split("\n"):
                if line.startswith("data: "):
                    yield f"data: {line[6:].strip()}\n\n"

    return EventSourceResponse(event_stream(), media_type="text/event-stream")


@router.post("/knn-translate-gemini-no-stream", summary="Proxy: KNN Translate Gemini (Non-stream)")
async def dharmamitra_knn_translate_gemini(request: DharmamitraGeminiRequest):
    """Proxy to Dharmamitra Gemini non-stream endpoint.

    Upstreams to https://dharmamitra.org/api-search/knn-translate-gemini-no-stream1/
    """
    url = "https://dharmamitra.org/api-search/knn-translate-gemini-no-stream1/"
    pwd = get_settings().dharmamitra_password or request.password
    if not pwd:
        raise HTTPException(status_code=400, detail="DHARMAMITRA_PASSWORD not set and no password provided")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "query": request.query,
                "language": (request.language or "").lower(),
                "password": pwd,
                "do_grammar": False,
                "use_pro_model": False,
            }

            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dharmamitra Gemini proxy failed: {str(e)}")

