# app/schemas/response.py

from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, Union


class ErrorDetail(BaseModel):
    """Enhanced error detail model."""
    code: str = Field(..., description="Error code for frontend handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    timestamp: str = Field(..., description="Error timestamp in ISO format")

    class Config:
        schema_extra = {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": [...]},
                "timestamp": "2025-01-31T15:35:38.485514Z"
            }
        }


class APIResponse(BaseModel):
    """Standardized API response wrapper."""
    success: bool = Field(..., description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[Union[ErrorDetail, Dict[str, Any]]] = Field(None, description="Error information if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
