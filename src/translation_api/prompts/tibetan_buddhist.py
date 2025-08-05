"""Specialized prompts for translating Tibetan Buddhist texts."""

from typing import Dict, Any


TIBETAN_BUDDHIST_TRANSLATION_PROMPT = """You are an expert translator specializing in Tibetan Buddhist texts. Translate the provided text into {target_language} while maintaining doctrinal accuracy and spiritual integrity.

REQUIREMENTS:
- Preserve Buddhist terminology and concepts precisely
- Maintain the spiritual context and meaning
- Use appropriate register for {text_type} texts
- Produce fluent, natural {target_language}
- Keep proper names and sacred terms intact when appropriate

{user_rules_section}

OUTPUT FORMAT - PROVIDE ONLY THE TRANSLATION:
Example input: "May all beings be free from suffering"
Correct output: "May all sentient beings be liberated from suffering"
Wrong output: "Translation: May all sentient beings be liberated from suffering"
Wrong output: "TEXT 1: May all sentient beings be liberated from suffering"

TEXT TYPE: {text_type}
SOURCE TEXT:
{source_text}

Translation:"""


BATCH_PROCESSING_INSTRUCTION = """
You will be translating {batch_size} Tibetan Buddhist texts. Provide ONLY the clean translations separated by "---TRANSLATION_SEPARATOR---".

CORRECT FORMAT EXAMPLE:
Input texts: "OM MANI PADME HUM", "May all beings be happy", "Compassion is the root of virtue"

CORRECT output:
Om Mani Padme Hum (Hail to the jewel in the lotus)
---TRANSLATION_SEPARATOR---
May all sentient beings find happiness
---TRANSLATION_SEPARATOR---
Compassion is the foundation of all virtuous qualities

WRONG output:
TEXT 1: Om Mani Padme Hum (Hail to the jewel in the lotus)
---TRANSLATION_SEPARATOR---
TEXT 2: May all sentient beings find happiness
---TRANSLATION_SEPARATOR---
TEXT 3: Compassion is the foundation of all virtuous qualities

IMPORTANT: 
- Do NOT include "TEXT 1:", "TEXT 2:", etc. prefixes
- Start directly with the translation
- Follow any user-specified formatting rules

Texts to translate:
{texts}
"""


def get_translation_prompt(
    source_text: str,
    target_language: str,
    text_type: str = "Buddhist text",
    batch_texts: list = None,
    user_rules: str = None
) -> str:
    """
    Generate a translation prompt for Tibetan Buddhist texts.
    
    Args:
        source_text: The text to translate (or first text if batch)
        target_language: Target language for translation
        text_type: Type of Buddhist text (sutra, commentary, etc.)
        batch_texts: List of texts for batch processing
        user_rules: Optional custom rules/instructions from user
    
    Returns:
        Formatted prompt string
    """
    # Create user rules section
    user_rules_section = ""
    if user_rules and user_rules.strip():
        user_rules_section = f"ADDITIONAL USER RULES:\n{user_rules.strip()}\n"
    
    if batch_texts:
        # Format texts for batch processing
        formatted_texts = ""
        for i, text in enumerate(batch_texts, 1):
            formatted_texts += f"\nTEXT {i}:\n{text}\n"
        
        batch_prompt = BATCH_PROCESSING_INSTRUCTION.format(
            batch_size=len(batch_texts),
            texts=formatted_texts
        )
        return TIBETAN_BUDDHIST_TRANSLATION_PROMPT.format(
            target_language=target_language,
            text_type=text_type,
            source_text=batch_prompt,
            user_rules_section=user_rules_section
        )
    
    return TIBETAN_BUDDHIST_TRANSLATION_PROMPT.format(
        target_language=target_language,
        text_type=text_type,
        source_text=source_text,
        user_rules_section=user_rules_section
    )


GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT = """
You are a linguistic expert specializing in Tibetan Buddhist texts.
Your task is to analyze the following pairs of source and translated texts to create a glossary of key terms.

**CRITICAL REQUIREMENTS**:
1.  **Exact Match**: The `source_term` and `translated_term` you provide must be *exactly* as they appear in the texts. Do not paraphrase or change them in any way. This is essential for word-finding later.
2.  **Relevance**: Only extract important doctrinal terms, proper names, or specialized concepts. Do not include common words.
3.  **Format**: You MUST return the output as a single, valid JSON object that follows the provided structure.

**JSON OUTPUT EXAMPLE**:
Your output must conform to this structure.

```json
{{
  "terms": [
    {{
      "source_term": "bodhicitta",
      "translated_term": "the mind of enlightenment"
    }},
    {{
      "source_term": "Tathāgata",
      "translated_term": "the Thus-Gone One"
    }},
    {{
      "source_term": "śūnyatā",
      "translated_term": "emptiness"
    }}
  ]
}}
```

**TEXT PAIRS FOR ANALYSIS**:
{text_pairs}
"""


def get_specialized_prompts() -> Dict[str, Any]:
    """Get specialized prompts for different types of Buddhist texts."""
    return {
        "sutra": {
            "context": "This is a sutra - a discourse attributed to the Buddha. Maintain the formal, sacred tone and preserve the traditional opening and closing formulas.",
            "terminology_focus": "Focus on accurate translation of core Buddhist concepts like dharma, sangha, nirvana, etc."
        },
        "commentary": {
            "context": "This is a commentary or explanation of Buddhist teachings. Preserve the analytical and scholarly tone while maintaining accessibility.",
            "terminology_focus": "Maintain consistency with standard Buddhist philosophical terminology."
        },
        "practice_manual": {
            "context": "This is a practice or meditation manual. Ensure instructions remain clear and actionable while preserving spiritual significance.",
            "terminology_focus": "Focus on meditation and practice-related terminology accuracy."
        },
        "philosophical_treatise": {
            "context": "This is a philosophical treatise. Maintain the rigorous analytical approach and complex argumentation structure.",
            "terminology_focus": "Preserve precise philosophical distinctions and technical terminology."
        }
    }