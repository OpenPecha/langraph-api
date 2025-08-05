from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .glossary import GlossaryTerm

class StandardizationInputItem(BaseModel):
    """Represents a single translation with its glossary for analysis."""
    original_text: str
    translated_text: str
    glossary: List[GlossaryTerm]

class AnalysisRequest(BaseModel):
    items: List[StandardizationInputItem]

class AnalysisResponse(BaseModel):
    """Shows which source terms have multiple translations across the dataset."""
    inconsistent_terms: Dict[str, List[str]]

class StandardizationTermPair(BaseModel):
    """A single source term and its official translation."""
    source_word: str
    standardized_translation: str

class StandardizationRequest(BaseModel):
    items: List[StandardizationInputItem]
    standardization_pairs: List[StandardizationTermPair]
    model_name: str = Field("claude", description="The model to use for the re-translation task.")
    user_rules: Optional[str] = Field(None, description="Optional custom rules to guide the re-translation.")

class StandardizationResponse(BaseModel):
    updated_items: List[StandardizationInputItem]

class RetranslationResponse(BaseModel):
    """The structured output for a re-translation task."""
    new_translation: str = Field(..., description="The new, re-translated text with all rules applied.") 