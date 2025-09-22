from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Structured outputs parsed from the LLM result text
class GlossStandardizedText(BaseModel):
    standardized_text: str = Field(description="The full corrected text if a meaning-altering change was made, else empty string")
    note: str = Field(description="Summary of analysis and citation justifying the standardization")
    analysis: str = Field(description="The analysis section as JSON string or serialized content")


class GlossOutputGlossary(BaseModel):
    glossary: str = Field(description="Full word-by-word glossary as a single markdown string with \\n for newlines")


class GlossRequest(BaseModel):
    input_text: str = Field(..., description="Primary input (Tibetan, optionally with English)")
    model_name: str = Field("claude", description="Model to use for generation")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    ucca_interpretation: Optional[str] = Field(None, description="Optional UCCA semantic interpretation to guide glossing")
    commentary_1: Optional[str] = Field(None, description="Optional commentary input 1")
    commentary_2: Optional[str] = Field(None, description="Optional commentary input 2")
    commentary_3: Optional[str] = Field(None, description="Optional commentary input 3")
    sanskrit_text: Optional[str] = Field(None, description="Optional Sanskrit parallel text")


class GlossResponse(BaseModel):
    standardized_text: Optional[str] = None
    note: Optional[str] = None
    analysis: Optional[str] = None
    glossary: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None


# Structured output schema for robust parsing from the LLM
class GlossStandardizedTextLite(BaseModel):
    standardized_text: str = Field(description="The full corrected text if a meaning-altering change was made, else empty string")
    note: str = Field(description="Summary of analysis and citation justifying the standardization")


class GlossFullOutput(BaseModel):
    analysis: List[Dict[str, Any]] = Field(default_factory=list)
    StandardizedText: GlossStandardizedTextLite
    Glossary: GlossOutputGlossary


class GlossItem(BaseModel):
    input_text: str
    ucca_interpretation: Optional[str] = None
    commentary_1: Optional[str] = None
    commentary_2: Optional[str] = None
    commentary_3: Optional[str] = None
    sanskrit_text: Optional[str] = None


class GlossBatchRequest(BaseModel):
    items: List[GlossItem]
    model_name: str = Field("claude", description="Model to use for generation")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(5, ge=1, le=50)


class GlossBatchItemResult(BaseModel):
    index: int
    standardized_text: Optional[str] = None
    note: Optional[str] = None
    analysis: Optional[str] = None
    glossary: Optional[str] = None
    error: Optional[str] = None


