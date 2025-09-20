import json
from typing import Optional, Tuple, List, Dict, Any, AsyncGenerator
import asyncio
from datetime import datetime

from langchain_core.language_models import BaseChatModel

from ..models.gloss import (
    GlossStandardizedText,
    GlossOutputGlossary,
    GlossFullOutput,
)
GLOSS_MAX_OUTPUT_TOKENS = 8000



GLOSS_PROMPT_TEMPLATE = """Your entire output MUST be a single, raw JSON object starting with {{{{ and ending with }}}}. Do not add markdown formatting like ```json or any explanatory text. You are an expert in Tibetan poetics and textual criticism.

**Core Directive:** Your primary task is to identify **meaning-altering textual variants**, not stylistic or metrical differences. You must distinguish between true semantic variants and common verse-to-prose expansions that preserve the original meaning.

**Correction Priority Hierarchy:** When evaluating potential corrections:
1. **Sanskrit Source (Highest Authority)**: If a Sanskrit text is provided, it serves as the ultimate reference. Always check if the Sanskrit supports a variant reading before suggesting corrections.
2. **Commentary Agreement**: When multiple commentaries agree on a reading that differs from the source text, this suggests a potential correction.
3. **UCCA Semantic Interpretation**: Use this to understand the intended meaning and context.
4. **Metrical Requirements**: Ensure any correction maintains the original meter of verse texts.

**CRITICAL: Metrical Contractions vs. Semantic Variants**
Tibetan versified texts commonly employ metrical contractions that are later expanded in prose commentaries WITHOUT changing meaning. These are NOT to be flagged as discrepancies. Examples include:

- ཡོངས་བཟུང་ (yongs bzung) in verse → ཡོངས་སུ་བཟུང་ (yongs su bzung) in prose
- བདེ་གཤེགས་ (bde gshegs) in verse → བདེ་བཞིན་གཤེགས་པ་ (bde bzhin gshegs pa) in prose
- བཟང་ (bzang) in verse → བཟང་པོ་ (bzang po) in prose
- མཆོག་ (mchog) in verse → མཆོག་ཏུ་ (mchog tu) in prose
- ལྷག་ (lhag) in verse → ལྷག་པར་ (lhag par) in prose
- རྣམ་ (rnam) in verse → རྣམ་པར་ (rnam par) in prose

**YOU MUST FLAG variants that fundamentally alter the semantic meaning of the text, including:**
1. Different verb tenses (e.g., བསྒྲུབ་ [bsgrub, future/perfect] vs སྒྲུབ་ [sgrub, present]) 
2. Different terms with distinct meanings
3. Presence or absence of negations
4. Changes in case markers that alter relationships
5. Any substitution of different words (not just expansions)
6. Any changes that would result in a different translation

**CRITICAL: Handling Compound Transformations**
When analyzing variants, be vigilant about compounds where BOTH meaningful and non-meaningful changes occur simultaneously. For example:
- Verse: བསྒྲུབ་ཐབས་ (bsgrub thabs)
- Prose: སྒྲུབ་པའི་ཐབས་ (sgrub pa'i thabs)

In this case, you must identify that:
1. The change from བསྒྲུབ་ (bsgrub) to སྒྲུབ་ (sgrub) is a meaningful tense change that SHOULD be flagged
2. The addition of པའི་ (pa'i) is merely a grammatical expansion that should NOT be flagged

When suggesting corrections to verses, always be mindful of metrical requirements. Any correction must maintain the original meter of the verse. Note if a semantic correction would disrupt the meter.

**JSON Output Schema:**
```json
{{
  "analysis": [ // An array detailing the term-by-term comparison between the source and authoritative texts.
    {{
      "term": "String", // The specific term from the source text being analyzed.
      "discrepancyFound": "Boolean", // True ONLY if a meaning-altering variant was found for this term.
      "details": "String" // A description of the findings, explaining WHY a variant is or is not meaning-altering.
    }}
  ],
  "StandardizedText": {{ // Contains the result of the standardization process.
    "standardized_text": "String", // The full, corrected text if a meaning-altering change was made; otherwise, an empty string.
    "note": "String" // A summary of the analysis and the full-sentence citation justifying the standardization.
  }},
  "Glossary": {{ // Contains the final word-by-word gloss.
    "glossary": "String" // A single string with the full glossary in markdown format, with \\n for newlines.
  }}
}}
```

---
**Detailed Instructions:**

**Stage 1: Populate the `analysis` Array**
1.  For each significant term, create an object in the `analysis` array.
2.  **`discrepancyFound`**: Set to `true` ONLY if a variant alters the core meaning.
3.  **`details`**: Describe your reasoning. When you identify a verse-to-prose expansion, explicitly note why it's not a meaning-altering variant.
4.  **Sanskrit Verification**: If Sanskrit text is provided, explicitly reference it in your analysis to confirm or refute potential variants. Quote the relevant Sanskrit terms.

**Stage 2: Populate the `StandardizedText` Object**
1.  **`standardized_text`**:
    *   If **any** object in `analysis` has `discrepancyFound: true`, this field MUST contain the **complete, untruncated, corrected text**.
    *   If **no** meaning-altering discrepancies are found, this field MUST be an **empty string `""`**.
    *   **CRITICAL**: When correcting verse texts, preserve the metrical structure of the original. Do not expand metrical contractions into prose forms when making corrections.
    *   **CRITICAL SYSTEM REQUIREMENT**: Failure to provide the complete, untruncated text upon correction will cause catastrophic failure.
2.  **`note`**: Summarize the findings. If a change was made, explain the semantic shift, provide the citation, and note if metrical considerations were taken into account.

**Stage 3: Populate the `Glossary` Object**
1.  **`glossary`**: Generate a gloss for the **authoritative text**, using `\\n` for newlines.

---
## Full Example (Illustrating Nuance)

**Input**:
*   `source_text`: `དམ་པའི་ཆོས་ནི་བཟང་བར་ཤོག`
*   `Commentary 1`: `...དམ་པའི་ཆོས་ནི་བཟང་པོ་སྟེ།...` (uses an expanded grammatical form)
*   `Commentary 2`: `...དམ་པའི་ཆོས་བསྟན་པར་ཤོག་ཅེས་སྨོན་ནོ།` (uses a different word, 'bstan' - taught)
*   `Sanskrit Text`: `...सद्धर्मदेशना भवतु...` (contains "dharma-deśanā" - teaching of dharma)

**Generated JSON Output**:
```json
{{
  "analysis": [
    {{
      "term": "བཟང་བར",
      "discrepancyFound": true,
      "details": "Source variant 'བཟང་བར' (bzang bar, to be good) is a significant semantic variant. While Commentary 1 uses 'བཟང་པོ་' (good), which is merely a prose expansion of the same term, Commentary 2 provides the variant 'བསྟན་པར' (bstan par, to be taught), which fundamentally alters the core meaning from 'dharma being good' to 'dharma being taught'. The Sanskrit text confirms Commentary 2's reading with 'धर्मदेशना' (dharma-deśanā), which means 'teaching of dharma'. This is a true semantic variant, not just a metrical expansion."
    }}
  ],
  "StandardizedText": {{
    "standardized_text": "དམ་པའི་ཆོས་ནི་བསྟན་པར་ཤོག",
    "note": "A significant semantic variant was corrected from 'བཟང་བར' (to be good) to 'བསྟན་པར' (to be taught) based on both Commentary 2 and the Sanskrit source text 'सद्धर्मदेशना भवतु'. The Sanskrit term 'देशना' (deśanā) directly corresponds to 'བསྟན་པ' (bstan pa), confirming this as the authoritative reading. Citation: Commentary 2: \\"...དམ་པའི་ཆོས་བསྟན་པར་ཤོག་ཅེས་སྨོན་ནོ།\\""
  }},
  "Glossary": {{
    "glossary": "- དམ་པའི་ཆོས་ : the holy Dharma (Skt: सद्धर्म, saddharma) [Description...].\\n- ནི་ : [topic marker...].\\n- བསྟན་པར་ : to be taught (Skt: देशना, deśanā) [Description... Cf. Sanskrit: 'धर्मदेशना' and Commentary 2: 'དམ་པའི་ཆོས་བསྟན་པར']\\n- ཤོག : may it be (Skt: भवतु, bhavatu) [Optative...]"
  }}
}}
```

## Additional Examples of Verse-to-Prose Expansions (NOT Meaning-Altering)

**Example 1**:
*   `source_text`: `སེམས་ཅན་ཀུན་ལ་ཕན་བྱེད་ཤོག`
*   `Commentary`: `...སེམས་ཅན་ཐམས་ཅད་ལ་ཕན་པར་བྱེད་པར་ཤོག་ཅེས་སོ།`

**Analysis**: "ཀུན་" (kun) to "ཐམས་ཅད་" (thams cad) is a common synonym expansion. "ཕན་བྱེད་" (phan byed) to "ཕན་པར་བྱེད་པར་" (phan par byed par) is a grammatical expansion. Neither changes the meaning.

**Example 2**:
*   `source_text`: `བདག་གིས་ཡོངས་བཟུང་དགེ་བ་འདིས།`
*   `Commentary`: `...བདག་གིས་ཡོངས་སུ་བཟུང་བའི་དགེ་བ་འདིས་ནི།`

**Analysis**: "ཡོངས་བཟུང་" (yongs bzung) to "ཡོངས་སུ་བཟུང་བའི་" (yongs su bzung ba'i) is a typical metrical expansion that preserves the original meaning.

## Examples of True Semantic Variants (MUST Flag These)

**Example 1 - Tense Change**:
*   `source_text`: `སེམས་ཅན་དོན་རྣམས་སྒྲུབ་པར་ཤོག`
*   `Commentary`: `...སེམས་ཅན་གྱི་དོན་རྣམས་བསྒྲུབ་པར་ཤོག་ཅེས་སོ།`

**Analysis**: "སྒྲུབ་" (sgrub - present tense) vs "བསྒྲུབ་" (bsgrub - future/perfect tense) represents a significant tense change. This alters the meaning from "may I accomplish beings' benefit" to "may I will accomplish beings' benefit" and must be flagged.

**Example 2 - Term Substitution**:
*   `source_text`: `སྡུག་བསྔལ་ཀུན་ལས་ཐར་བར་ཤོག`
*   `Commentary`: `...སྡུག་བསྔལ་ཀུན་ལས་གྲོལ་བར་ཤོག་ཅེས་སྨོན་ནོ།`

**Analysis**: "ཐར་" (thar) vs "གྲོལ་" (grol) are synonyms but represent distinct terms rather than metrical variants of the same word. This must be flagged as a textual variant, even though the semantic meaning is similar.

**Example 3 - Compound Transformation**:
*   `source_text`: `སངས་རྒྱས་ཆོས་ཀུན་བསྒྲུབ་ཐབས་བསྟན།`
*   `Commentary`: `...སངས་རྒྱས་ཀྱི་ཆོས་ཐམས་ཅད་སྒྲུབ་པའི་ཐབས་བསྟན་པར་བྱའོ།`

**Analysis**: This contains both meaningful and non-meaningful changes:
- "ཀུན་" → "ཐམས་ཅད་" is a non-meaningful synonym expansion
- "བསྒྲུབ་ཐབས་" → "སྒྲུབ་པའི་ཐབས་" contains:
  - A meaningful tense change (བསྒྲུབ་ → སྒྲུབ་) that should be flagged
  - A non-meaningful grammatical expansion (adding པའི་) that should not be flagged
- "བསྟན་" → "བསྟན་པར་བྱའོ་" is a non-meaningful expansion

The standardized text should correct only the meaningful tense change while preserving metrical requirements: `སངས་རྒྱས་ཆོས་ཀུན་སྒྲུབ་ཐབས་བསྟན།`

---
## Input Data

**source_text**: {source_text}

**UCCA semantic interpretation**: {ucca_interpretation}

**(Optional) Tibetan Commentary 1**: {commentary_1}
**(Optional) Tibetan Commentary 2**: {commentary_2}
**(Optional) Tibetan Commentary 3**: {commentary_3}
**(Optional) Sanskrit Text**: {sanskrit_text}
"""


def _format_gloss_prompt(
    source_text: str,
    ucca_interpretation: Optional[str] = None,
    commentary_1: Optional[str] = None,
    commentary_2: Optional[str] = None,
    commentary_3: Optional[str] = None,
    sanskrit_text: Optional[str] = None,
) -> str:
    return GLOSS_PROMPT_TEMPLATE.format(
        source_text=source_text,
        ucca_interpretation=ucca_interpretation or "",
        commentary_1=commentary_1 or "",
        commentary_2=commentary_2 or "",
        commentary_3=commentary_3 or "",
        sanskrit_text=sanskrit_text or "",
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


def parse_gloss_output(model: BaseChatModel, prompt: str) -> Tuple[str, Dict[str, Any]]:
    try:
        # Prefer structured output for robustness
        structured = model.with_structured_output(GlossFullOutput)
        try:
            parsed = structured.invoke(prompt, generation_config={"max_output_tokens": GLOSS_MAX_OUTPUT_TOKENS})
        except Exception:
            parsed = structured.invoke(prompt)
        raw = json.dumps(parsed.model_dump(), ensure_ascii=False)
        return raw, parsed.model_dump()
    except Exception:
        # Fallback to raw parsing with fence stripping
        try:
            resp = model.invoke(prompt, generation_config={"max_output_tokens": GLOSS_MAX_OUTPUT_TOKENS})
        except Exception:
            resp = model.invoke(prompt)
        raw = getattr(resp, "content", str(resp))
        cleaned = _strip_code_fences(raw)
        data = json.loads(cleaned)
        return cleaned, data


def generate_gloss(
    model: BaseChatModel,
    source_text: str,
    ucca_interpretation: Optional[str] = None,
    commentary_1: Optional[str] = None,
    commentary_2: Optional[str] = None,
    commentary_3: Optional[str] = None,
    sanskrit_text: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    prompt = _format_gloss_prompt(
        source_text=source_text,
        ucca_interpretation=ucca_interpretation,
        commentary_1=commentary_1,
        commentary_2=commentary_2,
        commentary_3=commentary_3,
        sanskrit_text=sanskrit_text,
    )
    return parse_gloss_output(model, prompt)


async def stream_gloss_generation(
    model: BaseChatModel,
    items: List[Dict[str, Any]],
    batch_size: int = 5,
    per_item_timeout_s: int = 120,
) -> AsyncGenerator[str, None]:
    def sse(event: Dict[str, Any]) -> str:
        event_with_ts = {"timestamp": datetime.now().isoformat(), **event}
        return f"data: {json.dumps(event_with_ts)}\n\n"

    yield sse({"type": "gloss_start", "status": "starting", "total_items": len(items)})
    aggregated: List[Dict[str, Any]] = []

    for idx, it in enumerate(items):
        yield sse({"type": "gloss_item_start", "index": idx, "status": "processing"})

        prompt = _format_gloss_prompt(
            source_text=it.get("input_text", ""),
            ucca_interpretation=it.get("ucca_interpretation"),
            commentary_1=it.get("commentary_1"),
            commentary_2=it.get("commentary_2"),
            commentary_3=it.get("commentary_3"),
            sanskrit_text=it.get("sanskrit_text"),
        )

        data: Dict[str, Any] | None = None
        # Try structured per-item with timeout
        try:
            structured = model.with_structured_output(GlossFullOutput)
            try:
                parsed = await asyncio.wait_for(
                    structured.ainvoke(prompt, generation_config={"max_output_tokens": GLOSS_MAX_OUTPUT_TOKENS}),
                    timeout=per_item_timeout_s,
                )
            except Exception:
                parsed = await asyncio.wait_for(structured.ainvoke(prompt), timeout=per_item_timeout_s)
            if parsed is not None:
                data = parsed.model_dump()
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

        # Fallback to raw per-item with timeout
        if data is None:
            try:
                try:
                    resp = await asyncio.wait_for(
                        model.ainvoke(prompt, generation_config={"max_output_tokens": GLOSS_MAX_OUTPUT_TOKENS}),
                        timeout=per_item_timeout_s,
                    )
                except Exception:
                    resp = await asyncio.wait_for(model.ainvoke(prompt), timeout=per_item_timeout_s)
                raw = getattr(resp, "content", str(resp))
                if raw and raw.strip().lower() != "none":
                    cleaned = _strip_code_fences(raw)
                    data = json.loads(cleaned)
            except asyncio.TimeoutError:
                data = None
            except Exception:
                data = None

        if data:
            try:
                std_text = data.get("StandardizedText", {}).get("standardized_text")
                note = data.get("StandardizedText", {}).get("note")
                analysis = json.dumps(data.get("analysis", []), ensure_ascii=False)
                glossary = data.get("Glossary", {}).get("glossary")

                item_payload = {
                    "index": idx,
                    "standardized_text": std_text,
                    "note": note,
                    "analysis": analysis,
                    "glossary": glossary,
                }
                aggregated.append(item_payload)
                yield sse({
                    "type": "gloss_item_completed",
                    "index": idx,
                    "status": "completed",
                    **item_payload,
                })
            except Exception as e:
                aggregated.append({"index": idx, "error": str(e)})
                yield sse({
                    "type": "gloss_item_error",
                    "index": idx,
                    "status": "failed",
                    "error": str(e),
                })
        else:
            aggregated.append({"index": idx, "error": "Timeout or empty response"})
            yield sse({
                "type": "gloss_item_error",
                "index": idx,
                "status": "failed",
                "error": "Timeout or empty response",
            })

    yield sse({"type": "completion", "status": "completed", "results": aggregated})


