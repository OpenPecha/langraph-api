"""UCCA API schemas."""

from pydantic import BaseModel


class UCCAErrorResponse(BaseModel):
    """Error response model for UCCA endpoints."""
    error: str

