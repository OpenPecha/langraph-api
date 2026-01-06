"""Dharmamitra proxy API schemas."""

from typing import Optional
from pydantic import BaseModel


class DharmamitraKnnRequest(BaseModel):
    """Request model for Dharmamitra KNN translation."""
    query: str
    language: str
    password: Optional[str] = None
    do_grammar: Optional[bool] = False  # Ignored; always forced False


class DharmamitraGeminiRequest(BaseModel):
    """Request model for Dharmamitra Gemini translation."""
    query: str
    language: str
    password: Optional[str] = None
    do_grammar: Optional[bool] = None  # Ignored; forced False
    use_pro_model: Optional[bool] = False  # Ignored; forced False

