"""Glossary API schemas."""

from typing import List
from pydantic import BaseModel, Field
from ..workflows.translation_state import TranslationResult


class GlossaryExtractionRequest(BaseModel):
    """Request model for glossary extraction."""
    items: List[TranslationResult]
    model_name: str = Field("claude", description="The model to use for the extraction task.")
    batch_size: int = Field(5, description="Number of items to process per parallel batch request.")

