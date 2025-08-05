from pydantic import BaseModel, Field
from typing import List

class GlossaryTerm(BaseModel):
    """A single term with its exact translation."""
    source_term: str = Field(..., description="The exact key term as it appears in the source text.")
    translated_term: str = Field(..., description="The corresponding exact translation as it appears in the translated text.")

class Glossary(BaseModel):
    """A structured glossary of key terms from the text."""
    terms: List[GlossaryTerm] = Field(..., description="A list of key terms and their exact translations.") 