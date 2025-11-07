from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class DynamicReferenceInput(BaseModel):
    type: str = Field(..., description="Arbitrary reference type (e.g., commentary, sanskrit, scan, lexicon)")
    content: str = Field(..., description="Reference content/excerpt")


class EditorCommentOptions(BaseModel):
    model_name: Optional[str] = Field(None, description="Optional model override; must not be 'dharamitra'")
    target_language: Optional[str] = Field(None, description="Optional target language for the comment (informational)")
    require_full_justification: bool = Field(True, description="Require all sentences to be justified by citations")
    mention_scope: Literal["last", "thread"] = Field("last", description="Where to extract @mentions from")
    max_mentions: int = Field(5, ge=0, le=20, description="Maximum mentions to include at the head of the comment")


class EditorCommentRequest(BaseModel):
    messages: List[ChatMessageInput]
    references: List[DynamicReferenceInput] = Field(default_factory=list)
    options: Optional[EditorCommentOptions] = None


class EditorCommentLLMOutput(BaseModel):
    comment_text: str = Field(..., description="Full commentary with inline bracketed citations at the end of each sentence")
    citations_used: List[str] = Field(default_factory=list, description="Unique list of citation IDs used in the comment_text")


class EditorCommentResponse(BaseModel):
    mentions: List[str] = Field(default_factory=list, description="Mentions included at the beginning of the comment")
    comment_text: str
    citations_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Optional[str]] = Field(default_factory=dict)


