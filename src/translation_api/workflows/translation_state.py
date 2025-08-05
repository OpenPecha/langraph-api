"""State management for translation workflows."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from ..models.glossary import Glossary
from ..models.glossary import GlossaryTerm


class TranslationRequest(BaseModel):
    """Input request for translation."""
    texts: List[str] = Field(..., description="List of texts to translate")
    target_language: str = Field(..., description="Target language for translation")
    model_name: str = Field("claude", description="Model to use for translation")
    text_type: str = Field("Buddhist text", description="Type of Buddhist text")
    batch_size: int = Field(5, description="Number of texts to process per batch")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules/instructions")


class TranslationBatch(BaseModel):
    """A batch of texts for processing."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    texts: List[str] = Field(..., description="Texts in this batch")
    target_language: str = Field(..., description="Target language")
    text_type: str = Field(..., description="Type of Buddhist text")
    model_name: str = Field(..., description="Model to use")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    user_rules: Optional[str] = Field(None, description="Optional custom translation rules")


class TranslationResult(BaseModel):
    """Result for a single text translation."""
    original_text: str
    translated_text: str
    metadata: Dict[str, Any]


class BatchResult(BaseModel):
    """Result of processing a batch."""
    batch_id: str = Field(..., description="Batch identifier")
    results: List[TranslationResult] = Field(..., description="Translation results")
    processing_time: float = Field(..., description="Time taken to process batch in seconds")
    model_used: str = Field(..., description="Model used for translation")
    success: bool = Field(True, description="Whether batch processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class TranslationWorkflowState(TypedDict):
    """Represents the state of the translation workflow."""
    original_request: TranslationRequest
    batches: List[TranslationBatch]
    current_batch_index: int
    batch_results: List[BatchResult]
    final_results: List[TranslationResult]
    total_texts: int
    processed_texts: int
    workflow_start_time: float
    workflow_status: str
    errors: List[Dict[str, Any]]
    retry_count: int
    model_name: str
    model_params: Dict[str, Any]
    custom_steps: Dict[str, Any]
    metadata: Dict[str, Any]
    glossary: Optional[Glossary]