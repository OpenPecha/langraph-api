"""Translation API schemas."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from ..workflows.translation_state import TranslationResult


class TranslationAPIRequest(BaseModel):
    """API request model for translation."""
    texts: List[str] = Field(..., description="List of texts to translate", min_items=1, max_items=100)
    target_language: str = Field(..., description="Target language for translation")
    model_name: str = Field("claude", description="Model to use for translation")
    text_type: str = Field("Buddhist text", description="Type of Buddhist text")
    batch_size: int = Field(5, description="Number of texts to process per batch", ge=1, le=50)
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules/instructions")
    context: Optional[str] = Field(None, description="Optional context for translation")


class TranslationAPIResponse(BaseModel):
    """API response model for translation."""
    success: bool = Field(..., description="Whether the translation was successful")
    results: List[TranslationResult] = Field(..., description="A list of the translation results.")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata, including timing and batch information.")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="A list of any errors that occurred during the process.")


class SingleTranslationRequest(BaseModel):
    """Request model for single text translation."""
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language for translation")
    model_name: str = Field("claude", description="Model to use for translation")
    text_type: str = Field("Buddhist text", description="Type of Buddhist text")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules/instructions")

