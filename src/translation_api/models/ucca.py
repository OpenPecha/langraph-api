from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UCCANode(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    type: str = Field(description="Node type (e.g., Parallel Scenes, Participants, Process, etc.)")
    text: str = Field(description="Text span covered by this node")
    english_text: str = Field(description="Literal English translation of the node; words not in the source text should be in square brackets [ ]")
    implicit: str = Field(description="Clarifies implied or contextually understood content; use empty string if explicit")
    parent_id: str = Field(description="ID of parent node", default="")
    children: List[str] = Field(description="IDs of child nodes", default_factory=list)
    descriptor: str = Field(description="Descriptor of the node")


class UCCAGraph(BaseModel):
    nodes: List[UCCANode] = Field(description="List of UCCA nodes in the graph")
    root_id: str = Field(description="ID of the root node")


class UCCARequest(BaseModel):
    input_text: str = Field(..., description="Input text to analyze (Tibetan, optionally with English)")
    model_name: str = Field("claude", description="Model to use for generation")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    commentary_1: Optional[str] = Field(None, description="Optional commentary input 1")
    commentary_2: Optional[str] = Field(None, description="Optional commentary input 2")
    commentary_3: Optional[str] = Field(None, description="Optional commentary input 3")
    sanskrit_text: Optional[str] = Field(None, description="Optional Sanskrit parallel text")


class UCCAResponse(BaseModel):
    ucca_graph: Optional[UCCAGraph] = None
    raw_json: Optional[str] = None
    error: Optional[str] = None


class UCCAItem(BaseModel):
    input_text: str
    commentary_1: Optional[str] = None
    commentary_2: Optional[str] = None
    commentary_3: Optional[str] = None
    sanskrit_text: Optional[str] = None


class UCCABatchRequest(BaseModel):
    items: List[UCCAItem]
    model_name: str = Field("claude", description="Model to use for generation")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(5, ge=1, le=50)


class UCCABatchItemResult(BaseModel):
    index: int
    ucca_graph: Optional[UCCAGraph] = None
    error: Optional[str] = None


"""Batch endpoint returns a plain list[UCCABatchItemResult] for consistency with array-of-dicts requirement."""


