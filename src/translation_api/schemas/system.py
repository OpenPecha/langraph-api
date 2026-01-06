"""System API schemas."""

from typing import Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    available_models: Dict[str, Dict[str, Any]] = Field(..., description="Available translation models")

