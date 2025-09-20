"""
Demo script to probe UCCA/Gloss behavior with thinking models (Gemini 2.5),
and to diagnose Empty/None structured responses by using production prompts
and robust fallbacks with detailed debug output.

Run:
  python examples/thinking_models_demo.py
"""

import json
import sys
from typing import Optional

# Ensure project root is on sys.path when running from repository root
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.translation_api.models.model_router import get_model_router
from src.translation_api.workflows.ucca import (
    UCCAGraph,
    UCCA_PROMPT_TEMPLATE,
    UCCA_GRAPH_SCHEMA_JSON,
    UCCA_NODE_SCHEMA_JSON,
)
from src.translation_api.models.gloss import GlossFullOutput
from src.translation_api.workflows.gloss import GLOSS_PROMPT_TEMPLATE


SOURCE_TEXT = (
    "སྔོན་ཆད་མ་བྱུང་བ་ཡང་འདིར་བརྗོད་མེད། །"
    "སྡེབ་སྦྱོར་མཁས་པའང་བདག་ལ་ཡོད་མིན་ཏེ། །"
    "དེ་ཕྱིར་གཞན་དོན་བསམ་པ་བདག་ལ་མེད། །"
    "རང་གི་ཡིད་ལ་བསྒོམ་ཕྱིར་ངས་འདི་བརྩམས། །"
)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")]
    return cleaned.strip()


def _debug_print_response(label: str, resp) -> None:
    try:
        meta = getattr(resp, "response_metadata", None)
        if meta is not None:
            print(f"[{label}] response_metadata:")
            try:
                print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
            except Exception:
                print(meta)
        content = getattr(resp, "content", None)
        if content is None:
            print(f"[{label}] content: <None>")
        else:
            text = str(content)
            preview = text if len(text) <= 500 else text[:500] + "... [truncated]"
            print(f"[{label}] content preview ({len(text)} chars):\n{preview}")
    except Exception as e:
        print(f"[{label}] debug print failed: {e}")


def _format_ucca_prompt(input_text: str,
                        commentary_1: Optional[str] = None,
                        commentary_2: Optional[str] = None,
                        commentary_3: Optional[str] = None,
                        sanskrit_text: Optional[str] = None) -> str:
    # Reuse UCCA prod template directly
    # We don't need to inject JSON schema placeholders here because the template
    # references the schema variables; we must provide them here.
    return UCCA_PROMPT_TEMPLATE.format(
        input_text=input_text,
        UCCA_GRAPH_SCHEMA_JSON=UCCA_GRAPH_SCHEMA_JSON,
        UCCA_NODE_SCHEMA_JSON=UCCA_NODE_SCHEMA_JSON,
        commentary_1=commentary_1 or "",
        commentary_2=commentary_2 or "",
        commentary_3=commentary_3 or "",
        sanskrit_text=sanskrit_text or "",
    )


def run_ucca(model_name: str, input_text: str,
             commentary_1: Optional[str] = None,
             commentary_2: Optional[str] = None,
             commentary_3: Optional[str] = None,
             sanskrit_text: Optional[str] = None) -> dict:
    router = get_model_router()
    model = router.get_model(model_name)
    prompt = _format_ucca_prompt(
        input_text=input_text,
        commentary_1=commentary_1,
        commentary_2=commentary_2,
        commentary_3=commentary_3,
        sanskrit_text=sanskrit_text,
    )
    # Try structured invoke first
    try:
        structured = model.with_structured_output(UCCAGraph)
        parsed = structured.invoke(prompt)
        if parsed is None:
            raise ValueError("Structured invoke returned None")
        return parsed.model_dump()
    except Exception as e:
        # Fallback: raw invoke + fence strip + JSON load
        print(f"[UCCA] Structured parse failed: {e}")
        try:
            raw_resp = model.invoke(prompt)
            _debug_print_response("UCCA raw", raw_resp)
            raw_content = getattr(raw_resp, "content", str(raw_resp))
            if not raw_content or raw_content.strip().lower() == "none":
                raise ValueError("Raw invoke returned empty/None content")
            cleaned = _strip_code_fences(raw_content)
            data = json.loads(cleaned)
            return UCCAGraph(**data).model_dump()
        except Exception as e2:
            # Final: return debug info
            return {"error": f"UCCA failed: {e2}"}


def _format_gloss_prompt(input_text: str,
                         ucca_interpretation: Optional[str] = None,
                         commentary_1: Optional[str] = None,
                         commentary_2: Optional[str] = None,
                         commentary_3: Optional[str] = None,
                         sanskrit_text: Optional[str] = None) -> str:
    return GLOSS_PROMPT_TEMPLATE.format(
        source_text=input_text,
        ucca_interpretation=ucca_interpretation or "",
        commentary_1=commentary_1 or "",
        commentary_2=commentary_2 or "",
        commentary_3=commentary_3 or "",
        sanskrit_text=sanskrit_text or "",
    )


def run_gloss(model_name: str, input_text: str,
              ucca_interpretation: Optional[str] = None,
              commentary_1: Optional[str] = None,
              commentary_2: Optional[str] = None,
              commentary_3: Optional[str] = None,
              sanskrit_text: Optional[str] = None) -> dict:
    router = get_model_router()
    model = router.get_model(model_name)
    structured = model.with_structured_output(GlossFullOutput)
    prompt = _format_gloss_prompt(
        input_text,
        ucca_interpretation=ucca_interpretation,
        commentary_1=commentary_1,
        commentary_2=commentary_2,
        commentary_3=commentary_3,
        sanskrit_text=sanskrit_text,
    )
    try:
        output = structured.invoke(prompt)
        if output is None:
            raise ValueError("Structured invoke returned None")
        return output.model_dump()
    except Exception as e:
        print(f"[Gloss] Structured parse failed: {e}")
        try:
            raw_resp = model.invoke(prompt)
            _debug_print_response("Gloss raw", raw_resp)
            raw_content = getattr(raw_resp, "content", str(raw_resp))
            if not raw_content or raw_content.strip().lower() == "none":
                raise ValueError("Raw invoke returned empty/None content")
            cleaned = _strip_code_fences(raw_content)
            data = json.loads(cleaned)
            return data
        except Exception as e2:
            return {"error": f"Gloss failed: {e2}"}


def main():
    # Models to compare. Both are configured with thinking enabled by router defaults.
    models = ["gemini-2.5-pro", "gemini-2.5-flash-thinking"]

    for m in models:
        print("\n==============================")
        print(f"Model: {m}")
        print("==============================\n")

        ucca_graph = run_ucca(m, SOURCE_TEXT)
        if "error" in ucca_graph:
            print("UCCA generation error:", ucca_graph["error"]) 
        else:
            print("UCCA Graph:")
            print(json.dumps(ucca_graph, ensure_ascii=False, indent=2))

        gloss = run_gloss(m, SOURCE_TEXT)
        if "error" in gloss:
            print("Gloss generation error:", gloss["error"]) 
        else:
            print("\nGloss Output:")
            print(json.dumps(gloss, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


