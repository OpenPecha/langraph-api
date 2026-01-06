"""Editor comment endpoints."""

import json
import re as _rx
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from ..models.comment import EditorCommentRequest, EditorCommentResponse, EditorCommentLLMOutput
from ..models.model_router import get_model_router
from ..config import get_settings
from ..utils.editor_helpers import (
    extract_mentions,
    enumerate_references,
    build_thread,
    build_mentions_section,
    build_editor_prompt,
    build_editor_stream_prompt,
)

router = APIRouter(prefix="/editor", tags=["Editor"])


@router.post("/comment", response_model=EditorCommentResponse, summary="Generate grounded commentary for a translation thread")
async def editor_comment(request: EditorCommentRequest):
    """Generate grounded commentary for a translation thread."""
    # Trigger check (@Comment in last message)
    if not request.messages or "@Comment" not in (request.messages[-1].content or ""):
        return {"mentions": [], "comment_text": "", "citations_used": [], "metadata": {"skipped": True, "reason": "No @Comment trigger"}}

    opts = request.options or {}
    mention_scope = getattr(opts, "mention_scope", "last")
    max_mentions = getattr(opts, "max_mentions", 5)

    # Mentions and references enumeration
    mentions = extract_mentions(request.messages, scope=mention_scope, max_mentions=max_mentions)
    refs_section, _all_ids = enumerate_references(request.references)
    thread_text = build_thread(request.messages)
    mentions_section = build_mentions_section(mentions)
    prompt = build_editor_prompt(thread_text, refs_section, mentions_section)

    # Choose model (disallow 'dharamitra')
    model_router = get_model_router()
    selected_model = getattr(opts, "model_name", None)
    if selected_model and selected_model.lower() == "dharamitra":
        raise HTTPException(status_code=400, detail="'dharamitra' is translation-only and not supported for editor comments")
    # Fallback preference: gemini-2.5-pro if available else default
    model_name = selected_model or ("gemini-2.5-pro" if model_router.validate_model_availability("gemini-2.5-pro") else get_settings().default_model)
    if not model_router.validate_model_availability(model_name):
        available = list(model_router.get_available_models().keys())
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available. Available models: {available}")

    model = model_router.get_model(model_name)
    structured = model.with_structured_output(EditorCommentLLMOutput)

    try:
        resp = await structured.ainvoke(prompt)
        comment_text = resp.comment_text
        citations_used = resp.citations_used or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Editor comment generation failed: {str(e)}")

    return EditorCommentResponse(
        mentions=mentions,
        comment_text=comment_text,
        citations_used=citations_used,
        metadata={
            "triggered_by": "@Comment",
            "model_used": model_name,
        },
    )


@router.post("/comment/stream", summary="Generate grounded commentary (SSE)")
async def editor_comment_stream(request: EditorCommentRequest):
    """Generate grounded commentary with SSE streaming."""
    async def _gen():
        # Trigger check
        if not request.messages or "@Comment" not in (request.messages[-1].content or ""):
            yield f"data: {{\"skipped\": true, \"reason\": \"No @Comment trigger\"}}\n\n"
            return

        opts = request.options or {}
        mention_scope = getattr(opts, "mention_scope", "last")
        max_mentions = getattr(opts, "max_mentions", 5)

        mentions = extract_mentions(request.messages, scope=mention_scope, max_mentions=max_mentions)
        refs_section, _all_ids = enumerate_references(request.references)
        thread_text = build_thread(request.messages)
        mentions_section = build_mentions_section(mentions)
        prompt = build_editor_prompt(thread_text, refs_section, mentions_section)

        # Model selection
        model_router = get_model_router()
        selected_model = getattr(opts, "model_name", None)
        if selected_model and selected_model.lower() == "dharamitra":
            yield f"data: {{\"error\": \"'dharamitra' not supported for editor comments\"}}\n\n"
            return
        model_name = selected_model or ("gemini-2.5-pro" if model_router.validate_model_availability("gemini-2.5-pro") else get_settings().default_model)
        if not model_router.validate_model_availability(model_name):
            avail = list(model_router.get_available_models().keys())
            yield f"data: {{\"error\": \"Model '{model_name}' not available.\", \"available\": {json.dumps(avail)} }}\n\n"
            return

        yield f"data: {{\"type\": \"initialization\", \"mentions\": {json.dumps(mentions)}, \"model_used\": {json.dumps(model_name)} }}\n\n"

        stream_prompt = build_editor_stream_prompt(thread_text, refs_section, mentions_section)

        # If trigger missing, short-circuit
        if "@Comment" not in (request.messages[-1].content or ""):
            yield "data: {\"skipped\": true, \"reason\": \"No @Comment trigger\"}\n\n"
            return

        model = model_router.get_model(model_name)
        full_text = ""
        try:
            # Prefer token-level events when available
            try:
                # Only pass generation_config for Gemini models
                if model_name.startswith("gemini"):
                    stream_kwargs = {"generation_config": {"response_mime_type": "text/plain"}}
                else:
                    stream_kwargs = {}
                
                async for event in model.astream_events([HumanMessage(content=stream_prompt)], **stream_kwargs):
                    et = event.get("event")
                    if et in ("on_chat_model_stream", "on_llm_stream"):
                        chunk = event.get("data", {}).get("chunk")
                        piece = getattr(chunk, "content", None)
                        if isinstance(piece, list):
                            piece = "".join([str(p) for p in piece])
                        if isinstance(piece, str) and piece:
                            full_text += piece
                            yield f"data: {{\"type\": \"comment_delta\", \"text\": {json.dumps(piece)} }}\n\n"
            except AttributeError:
                # Fallback: no native streaming; use single shot but send once
                if model_name.startswith("gemini"):
                    resp = await model.ainvoke([HumanMessage(content=stream_prompt)], generation_config={"response_mime_type": "text/plain"})
                else:
                    resp = await model.ainvoke([HumanMessage(content=stream_prompt)])
                text = getattr(resp, "content", "") or ""
                if isinstance(text, list):
                    text = "".join([str(p) for p in text])
                full_text = str(text)
                if full_text:
                    yield f"data: {{\"type\": \"comment_delta\", \"text\": {json.dumps(full_text)} }}\n\n"

            # Derive citations_used by scanning bracketed IDs
            ids = []
            try:
                for m in _rx.finditer(r"\[(.*?)\]", full_text):
                    inside = m.group(1) or ""
                    for tok in inside.split(";"):
                        t = tok.strip()
                        if t and t not in ids:
                            ids.append(t)
            except Exception:
                pass

            yield f"data: {{\"type\": \"completion\", \"comment_text\": {json.dumps(full_text)}, \"citations_used\": {json.dumps(ids)}, \"mentions\": {json.dumps(mentions)} }}\n\n"
        except Exception as e:
            yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))} }}\n\n"

    return EventSourceResponse(_gen(), media_type="text/event-stream")

