"""Workflow endpoints for echo-based translation workflows."""

import json
from fastapi import APIRouter, HTTPException
from ..models.workflow import WorkflowRunRequest, WorkflowResponse, WorkflowBatchRequest, WorkflowBatchItemResult
from ..models.model_router import get_model_router
from ..utils.workflow_helpers import canonicalize_combo_key
from ..config import get_settings

router = APIRouter(prefix="/workflow", tags=["Workflow"])


@router.post("/run", response_model=WorkflowResponse, summary="Run workflow by combo key")
async def workflow_run(request: WorkflowRunRequest) -> WorkflowResponse:
    """Echo-based workflow runner.

    - Selects prompt based on the provided path combo_key (order-independent).
    - Canonicalizes the provided combo_key regardless of which inputs are present.
    - Returns the inputs for UI testing.
    """
    inputs = request.input
    combo_key = request.combo_key
    model_name = request.model_name or "claude-sonnet-4-20250514"

    if not inputs.source:
        raise HTTPException(status_code=400, detail="'source' is required")

    provided_key = canonicalize_combo_key(combo_key)

    # Always prefer the provided path key for prompt selection (order-independent)
    final_key = provided_key

    # Validate presence and bounds based on tokens
    tokens = set(final_key.split('+'))
    # commentaries length validation
    if inputs.commentaries is not None and len(inputs.commentaries) > 3:
        raise HTTPException(status_code=400, detail="At most 3 commentaries are allowed")
    # If combo specifies commentariesK where K>0, ensure provided
    requires_commentaries = any(t.startswith('commentaries') and t not in ['commentaries0'] for t in tokens)
    if requires_commentaries and (not inputs.commentaries or len(inputs.commentaries) == 0):
        raise HTTPException(status_code=400, detail="This combination requires commentaries, but none were provided")
    # If combo includes ucca/gloss/sanskrit ensure corresponding input present
    if 'ucca' in tokens and inputs.ucca is None:
        raise HTTPException(status_code=400, detail="Combo includes 'ucca' but no UCCA JSON was provided")
    if 'gloss' in tokens and inputs.gloss is None:
        raise HTTPException(status_code=400, detail="Combo includes 'gloss' but no Gloss JSON was provided")
    if 'sanskrit' in tokens and not inputs.sanskrit:
        raise HTTPException(status_code=400, detail="Combo includes 'sanskrit' but no Sanskrit text was provided")

    # Build translation instructions: no additions beyond source; obey target language if provided
    target_line = f"Translate into {inputs.target_language}." if inputs.target_language else "Translate into the requested target language."
    guidelines: list[str] = [
        target_line,
        "Do not add content beyond the source. No examples, adaptations, or expansions.",
        "Preserve meaning, nuance, and accuracy; avoid extraneous explanation.",
    ]
    if 'ucca' in tokens:
        guidelines.append("Use the UCCA structure to disambiguate roles, participants, and processes; do not include UCCA in the output.")
    if 'gloss' in tokens:
        guidelines.append("Use Gloss to prefer standardized term choices and respect any provided notes.")
    if inputs.commentaries and len(inputs.commentaries) > 0:
        guidelines.append("Leverage commentaries to resolve ambiguity; do not quote or cite them explicitly.")
    if 'sanskrit' in tokens:
        guidelines.append("Use Sanskrit to validate terms and transliterations where applicable; do not add Sanskrit unless necessary.")
    # No output structure restriction; return plain text translation only.

    prompt_text = "\n- " + "\n- ".join(guidelines)

    # Invoke LLM with a default generic prompt using the selected model
    model_router = get_model_router()
    if not model_router.validate_model_availability(model_name):
        available_models = list(model_router.get_available_models().keys())
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available. Available models: {available_models}")

    # Build a simple combined prompt with inputs for a generic echo/summary
    def block(title: str, body: str) -> str:
        return f"\n\n### {title}\n{body.strip()}" if body else ""

    # Render inputs sections
    commentary_text = "\n\n".join((inputs.commentaries or [])[:3]) if inputs.commentaries else ""
    # Formatted commentaries block for {commentaries}/{commenteries} placeholder
    commentaries_block = ""
    if inputs.commentaries:
        lines: list[str] = []
        for idx, c in enumerate(inputs.commentaries[:3]):
            c_str = (c or "").strip()
            if c_str:
                lines.append(f"commentary {idx+1}: {c_str}")
        commentaries_block = "\n".join(lines)
    ucca_text = json.dumps(inputs.ucca, ensure_ascii=False, indent=2) if isinstance(inputs.ucca, dict) else (inputs.ucca or "")
    gloss_text = json.dumps(inputs.gloss, ensure_ascii=False, indent=2) if isinstance(inputs.gloss, dict) else (inputs.gloss or "")
    sanskrit_text = inputs.sanskrit or ""

    # Support custom prompt if provided. It must include {source}; other placeholders optional.
    if request.custom_prompt:
        tmpl = request.custom_prompt
        if "{source}" not in tmpl:
            raise HTTPException(status_code=400, detail="custom_prompt must include {source}")
        # Build substitution map
        subs = {
            "source": inputs.source,
            "ucca": ucca_text,
            "gloss": gloss_text,
            "commentary1": (inputs.commentaries[0] if inputs.commentaries and len(inputs.commentaries) > 0 else ""),
            "commentary2": (inputs.commentaries[1] if inputs.commentaries and len(inputs.commentaries) > 1 else ""),
            "commentary3": (inputs.commentaries[2] if inputs.commentaries and len(inputs.commentaries) > 2 else ""),
            # Support a single block placeholder for all commentaries
            "commentaries": commentaries_block,
            # Common misspelling alias
            "commenteries": commentaries_block,
            "sanskrit": sanskrit_text,
            "target_language": (inputs.target_language or "").strip(),
        }
        # Protect allowed placeholders, escape all other braces, then restore
        allowed_placeholders = [
            "source","ucca","gloss","commentary1","commentary2","commentary3","commentaries","commenteries","sanskrit","target_language"
        ]
        sentinel_map = {name: f"<<PH_{name.upper()}>>" for name in allowed_placeholders}
        protected = tmpl
        for name, token in sentinel_map.items():
            protected = protected.replace(f"{{{name}}}", token)
        # Escape any remaining single braces to avoid format errors
        protected = protected.replace("{", "{{").replace("}", "}}")
        # Restore placeholders
        for name, token in sentinel_map.items():
            protected = protected.replace(token, f"{{{name}}}")
        combined_prompt = protected.format(**subs)
    else:
        combined_prompt = (
            f"You are a professional translator for Buddhist literature.\n"
            + block("Instructions", "\n- " + "\n- ".join(guidelines))
            + block("Source", inputs.source)
            + block("Commentaries (up to 3)", commentary_text)
            + block("UCCA", ucca_text)
            + block("Gloss", gloss_text)
            + block("Sanskrit", sanskrit_text)
            + "\n\nReturn only the translated text with no additional commentary."
        )

    llm_output: str = ""
    translation_text: str | None = None
    try:
        model = model_router.get_model(model_name, **(request.model_params or {}))
        # Single plain-text response; no structured schema
        # Force plain text only for Gemini models; other providers don't use this flag
        if (model_name or "").startswith("gemini"):
            resp = await model.ainvoke(
                combined_prompt,
                generation_config={"response_mime_type": "text/plain"}
            )
        else:
            resp = await model.ainvoke(combined_prompt)
        llm_output = getattr(resp, "content", str(resp)) or ""
        if isinstance(llm_output, list):
            try:
                llm_output = "\n".join([str(p) for p in llm_output])
            except Exception:
                llm_output = str(llm_output)
        translation_text = llm_output
    except Exception as e:
        # Return the error inside the payload for easier UI debugging instead of failing the request entirely
        llm_output = f"LLM invocation error: {str(e)}"
        translation_text = None

    return WorkflowResponse(
        combo_key=final_key,
        translation=translation_text,
    )


@router.post("/run/batch", response_model=list[WorkflowBatchItemResult], summary="Run workflow in batch")
async def workflow_run_batch(request: WorkflowBatchRequest) -> list[WorkflowBatchItemResult]:
    """Run workflow for multiple items in batch."""
    # Reuse the single-run logic in a loop
    results: list[WorkflowBatchItemResult] = []
    for idx, item in enumerate(request.items):
        try:
            single = WorkflowRunRequest(
                combo_key=request.combo_key,
                input=item,
                model_name=request.model_name,
                model_params=request.model_params,
            )
            resp = await workflow_run(single)
            results.append(WorkflowBatchItemResult(index=idx, translation=resp.translation))
        except HTTPException as he:
            results.append(WorkflowBatchItemResult(index=idx, error=str(he.detail)))
        except Exception as e:
            results.append(WorkflowBatchItemResult(index=idx, error=str(e)))
    return results

