from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class WorkflowInput(BaseModel):
    """Unified input for workflow runs.

    All fields except `source` are optional. `ucca` and `gloss` are kept as
    arbitrary JSON objects (dict) to allow UI to pass through whatever
    structure they currently have without strict validation.
    """

    source: str = Field(..., description="Root/source Tibetan text")
    commentaries: Optional[List[str]] = Field(
        default=None, description="Up to three optional commentaries"
    )
    ucca: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Optional UCCA: JSON object or raw string"
    )
    gloss: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Optional Gloss: JSON object or raw string"
    )
    sanskrit: Optional[str] = Field(
        default=None, description="Optional Sanskrit source text"
    )
    target_language: Optional[str] = Field(
        default=None, description="Target language (name or ISO code). If absent, model may infer."
    )


class WorkflowResponse(BaseModel):
    combo_key: str
    translation: Optional[str] = None


class WorkflowLLMResult(BaseModel):
    translation: str = Field(..., description="Translation of the source text into English, concise and faithful.")


class WorkflowRunRequest(BaseModel):
    combo_key: str = Field(..., description="Order-independent combo key, e.g., 'source+ucca+gloss'")
    input: WorkflowInput = Field(..., description="Single workflow input")
    model_name: Optional[str] = Field(None, description="Model to use; overrides input.model if provided")
    model_params: Dict[str, Any] = Field(default_factory=dict)
<<<<<<< HEAD
=======
    api: Optional[str] = Field(
        default=None,
        description="Optional custom base API endpoint for the selected model (Claude/Gemini)."
    )
>>>>>>> 258f501 (Enhance API key handling in ModelRouter)
    custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional custom prompt template. Must include {source}. Optional placeholders: "
            "{ucca}, {gloss}, {commentary1}, {commentary2}, {commentary3}, {sanskrit}."
        ),
    )


class WorkflowBatchItemResult(BaseModel):
    index: int
    translation: Optional[str] = None
    error: Optional[str] = None


class WorkflowBatchRequest(BaseModel):
    combo_key: str = Field(..., description="Order-independent combo key for all items")
    items: List[WorkflowInput]
    model_name: str = Field("claude")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(3, ge=1, le=50)


