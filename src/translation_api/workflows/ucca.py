import json
import logging
from typing import Tuple, Optional, AsyncGenerator, List, Dict, Any

from langchain_core.language_models import BaseChatModel
from datetime import datetime

from ..models.ucca import UCCAGraph, UCCANode


logger = logging.getLogger("translation_api.ucca")


UCCA_NODE_SCHEMA_JSON = json.dumps(UCCANode.model_json_schema(), indent=2)
UCCA_GRAPH_SCHEMA_JSON = json.dumps(UCCAGraph.model_json_schema(), indent=2)


UCCA_PROMPT_TEMPLATE = """
You are an expert in UCCA (Universal Conceptual Cognitive Annotation) parsing.
Your task is to parse the given input text and generate a UCCA graph in JSON format.
The input text may contain Tibetan script and its English translation, or just Tibetan script.
If only Tibetan is provided, infer the English translation for the 'english_text' field in each UCCANode.

The output JSON MUST strictly adhere to the following Pydantic model schemas:

UCCAGraph Schema:
{UCCA_GRAPH_SCHEMA_JSON}

UCCANode Schema (referenced within UCCAGraph nodes list):
{UCCA_NODE_SCHEMA_JSON}

Key instructions for UCCANode fields:
- 'id': A unique string identifier for each node (e.g., "N1", "N2").
- 'type': The semantic type of the node (e.g., "Parallel Scene", "Participant", "Process", "State", "Adverbial", "Center", "Elaborator").
- 'text': The exact Tibetan text span this node covers. If not applicable or if the node is purely structural/implicit, use an empty string or a conceptual placeholder.
- 'english_text': A literal English translation of the 'text'. Words not in the source text (e.g., implied elements) should be in square brackets [ ].
- 'implicit': Clarify implied or contextually understood content not explicitly in the text but necessary for comprehension. Use an empty string if content is explicit.
- 'parent_id': The 'id' of the parent node. The root node of the graph should have an empty string for 'parent_id'.
- 'children': A list of 'id's of child nodes. This should be consistent with 'parent_id' relationships.
- 'descriptor': A brief, human-readable descriptor or label for the node, summarizing its role or content.

Input Text:
{input_text}

Optional Contexts (may be empty):
- Commentary 1: {commentary_1}
- Commentary 2: {commentary_2}
- Commentary 3: {commentary_3}
- Sanskrit: {sanskrit_text}

Generate the UCCA graph as a single JSON object conforming to the UCCAGraph Schema provided above.
Ensure all node 'id's are unique. Ensure 'root_id' in UCCAGraph points to the main root node's 'id'.
Focus on creating a coherent and valid UCCA structure based on the input.
Return ONLY the JSON (no markdown fences or explanations).
"""


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


def _format_prompt(
    input_text: str,
    commentary_1: Optional[str] = None,
    commentary_2: Optional[str] = None,
    commentary_3: Optional[str] = None,
    sanskrit_text: Optional[str] = None,
) -> str:
    return UCCA_PROMPT_TEMPLATE.format(
        input_text=input_text,
        UCCA_GRAPH_SCHEMA_JSON=UCCA_GRAPH_SCHEMA_JSON,
        UCCA_NODE_SCHEMA_JSON=UCCA_NODE_SCHEMA_JSON,
        commentary_1=commentary_1 or "",
        commentary_2=commentary_2 or "",
        commentary_3=commentary_3 or "",
        sanskrit_text=sanskrit_text or "",
    )


def generate_ucca_graph(
    model: BaseChatModel,
    input_text: str,
    commentary_1: Optional[str] = None,
    commentary_2: Optional[str] = None,
    commentary_3: Optional[str] = None,
    sanskrit_text: Optional[str] = None,
) -> Tuple[str, UCCAGraph]:
    prompt = _format_prompt(
        input_text=input_text,
        commentary_1=commentary_1,
        commentary_2=commentary_2,
        commentary_3=commentary_3,
        sanskrit_text=sanskrit_text,
    )
    response = model.invoke(prompt)
    raw = getattr(response, "content", str(response))
    cleaned = _strip_code_fences(raw)
    data = json.loads(cleaned)
    graph = UCCAGraph(**data)
    return cleaned, graph


async def stream_ucca_generation(
    model: BaseChatModel,
    items: List[Dict[str, Any]],
    batch_size: int = 5,
) -> AsyncGenerator[str, None]:
    # SSE helper
    def sse(event: Dict[str, Any]) -> str:
        event_with_ts = {"timestamp": datetime.now().isoformat(), **event}
        return f"data: {json.dumps(event_with_ts)}\n\n"

    yield sse({"type": "ucca_start", "status": "starting", "total_items": len(items)})
    aggregated_results: list[dict] = []

    for start in range(0, len(items), max(1, batch_size)):
        chunk = items[start:start + batch_size]
        # Emit item_start events for the chunk
        for i, _ in enumerate(chunk, start=start):
            yield sse({"type": "ucca_item_start", "index": i, "status": "processing"})

        prompts = [
            _format_prompt(
                input_text=it.get("input_text", ""),
                commentary_1=it.get("commentary_1"),
                commentary_2=it.get("commentary_2"),
                commentary_3=it.get("commentary_3"),
                sanskrit_text=it.get("sanskrit_text"),
            )
            for it in chunk
        ]

        try:
            structured = model.with_structured_output(UCCAGraph)
            responses = await structured.abatch(prompts)
        except Exception:
            # Fallbacks: raw abatch then sequential
            try:
                responses = await model.abatch(prompts)
            except Exception:
                responses = [model.invoke(p) for p in prompts]

        for offset, resp in enumerate(responses):
            idx = start + offset
            try:
                if resp is None:
                    raise ValueError("Empty response from LLM")
                if hasattr(resp, "model_dump"):
                    dumped = resp.model_dump() if resp is not None else None
                    if not dumped:
                        raise ValueError("Empty structured response")
                    graph = UCCAGraph(**dumped)
                else:
                    raw = getattr(resp, "content", str(resp))
                    if not raw or raw.strip().lower() == "none":
                        raise ValueError("Empty raw response")
                    cleaned = _strip_code_fences(raw)
                    data = json.loads(cleaned)
                    graph = UCCAGraph(**data)
                item_result = {"index": idx, "ucca_graph": graph.model_dump()}
                aggregated_results.append(item_result)
                yield sse({
                    "type": "ucca_item_completed",
                    "index": idx,
                    "status": "completed",
                    "ucca_graph": item_result["ucca_graph"],
                })
            except Exception as e:
                # One retry with structured invoke per item
                try:
                    structured_single = model.with_structured_output(UCCAGraph)
                    parsed = structured_single.invoke(prompts[offset])
                    if parsed is None:
                        raise ValueError("Empty retry response from LLM")
                    if hasattr(parsed, "model_dump"):
                        dumped = parsed.model_dump()
                        graph = UCCAGraph(**dumped)
                    else:
                        raw2 = getattr(parsed, "content", str(parsed))
                        if not raw2 or raw2.strip().lower() == "none":
                            raise ValueError("Empty retry raw response")
                        cleaned2 = _strip_code_fences(raw2)
                        data2 = json.loads(cleaned2)
                        graph = UCCAGraph(**data2)
                    item_result = {"index": idx, "ucca_graph": graph.model_dump()}
                    aggregated_results.append(item_result)
                    yield sse({
                        "type": "ucca_item_completed",
                        "index": idx,
                        "status": "completed",
                        "ucca_graph": item_result["ucca_graph"],
                    })
                except Exception as e2:
                    aggregated_results.append({"index": idx, "error": str(e2)})
                    yield sse({
                        "type": "ucca_item_error",
                        "index": idx,
                        "status": "failed",
                        "error": str(e2),
                    })

    yield sse({"type": "completion", "status": "completed", "results": aggregated_results})


