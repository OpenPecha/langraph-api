from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from .glossary import Glossary
from .standardization import (
    StandardizationInputItem,
    StandardizationTermPair,
)


class PipelineRequest(BaseModel):
    """Configure a custom workflow pipeline by selecting stages and providing their inputs."""

    # Stages to run, in order
    stages: List[str] = Field(
        ..., description="Ordered list of stages to run. Allowed: translate, extract_glossary, analyze, apply_standardization"
    )

    # Translation inputs
    texts: Optional[List[str]] = Field(
        None, description="Texts to translate (required if 'translate' stage is included)"
    )
    target_language: Optional[str] = Field(
        None, description="Target language (required if 'translate' stage is included)"
    )
    model_name: str = Field("claude", description="Model to use for LLM-backed stages")
    text_type: str = Field("Buddhist text", description="Type of text for prompting context")
    batch_size: int = Field(5, ge=1, le=50, description="Batch size for translation and glossary")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model params")
    user_rules: Optional[str] = Field(None, description="Optional global user rules")

    # If skipping translation, user can supply already translated items
    items: Optional[List[StandardizationInputItem]] = Field(
        None, description="Existing translated items with optional glossaries"
    )

    # Standardization
    standardization_pairs: Optional[List[StandardizationTermPair]] = Field(
        None, description="Pairs of source_word and standardized_translation (required for apply_standardization)"
    )

    @validator("stages")
    def validate_stages(cls, v: List[str]) -> List[str]:
        allowed = {"translate", "extract_glossary", "analyze", "apply_standardization"}
        for s in v:
            if s not in allowed:
                raise ValueError(f"Invalid stage '{s}'. Allowed: {sorted(list(allowed))}")
        return v


class PipelineResponse(BaseModel):
    """Aggregated outputs of the selected pipeline stages."""

    # Translation results if produced
    results: Optional[List[Dict[str, Any]]] = None  # keep generic to avoid circular types

    # Glossary if produced
    glossary: Optional[Glossary] = None

    # Analysis report if produced
    inconsistent_terms: Optional[Dict[str, List[str]]] = None

    # Updated items if standardization applied
    updated_items: Optional[List[StandardizationInputItem]] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
