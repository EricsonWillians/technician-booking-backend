# app/routers/bookings.py

"""
==========================================================
               BOOKINGS ROUTER MODULE
==========================================================

Enhanced FastAPI router for the Technician Booking System with
comprehensive error handling and frontend-friendly responses.

Features:
- Consistent JSON response structure
- Detailed error information
- NLP processing with confidence scores
- Comprehensive request validation
- Rich metadata in responses

Author : Ericson Willians  
Email  : ericsonwillians@protonmail.com  
Date   : January 2025  

==========================================================
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services import booking_service
from app.schemas.booking import BookingCreate, BookingResponse
from app.services.nlp_service import nlp_service
from app.schemas.response import APIResponse, ErrorDetail  # Import from the centralized module

# Configure router
router = APIRouter()
logger = logging.getLogger("bookings_router")

# Enhanced response models
class AnalysisScore(BaseModel):
    """Model for intent classification scores."""
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(..., description="Confidence score for the intent")
    assessment: str = Field(..., description="Qualitative assessment of the confidence")

class ErrorDetail(BaseModel):
    """Enhanced error detail model."""
    code: str = Field(..., description="Error code for frontend handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    timestamp: str = Field(..., description="Error timestamp in ISO format")

class CommandRequest(BaseModel):
    """Enhanced command request model."""
    message: str = Field(..., min_length=1, description="User command text")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class CommandResult(BaseModel):
    """Enhanced command result model."""
    success: bool = Field(..., description="Whether the command was successful")
    intent: str = Field(..., description="Classified intent")
    message: str = Field(..., description="Response message")
    analysis: Optional[List[AnalysisScore]] = Field(None, description="Intent classification scores")
    booking: Optional[BookingResponse] = Field(None, description="Created/retrieved booking")
    bookings: Optional[List[BookingResponse]] = Field(None, description="List of bookings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

def create_error_response(status_code: int, code: str, message: str, details: Optional[Dict] = None) -> JSONResponse:
    """Create a standardized error response with ISO-formatted timestamp."""
    error_detail = ErrorDetail(
        code=code,
        message=message,
        details=details,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

    # Convert `ErrorDetail` explicitly to a dictionary before passing it
    api_response = APIResponse(
        success=False,
        error=error_detail.model_dump(),
        metadata={}
    )

    return JSONResponse(
        status_code=status_code,
        content=api_response.model_dump() 
    )

def create_success_response(data: Any, metadata: Optional[Dict] = None) -> APIResponse:
    """Create a standardized success response with ISO-formatted metadata."""
    # Convert any datetime objects in metadata to ISO-formatted strings
    if metadata:
        metadata = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in metadata.items()
        }
    api_response = APIResponse(
        success=True,
        data=data,
        metadata=metadata or {}
    )
    return api_response

def get_confidence_assessment(score: float) -> str:
    """Get qualitative assessment of confidence score."""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    return "low"

@router.get(
    "/",
    response_model=APIResponse,
    summary="List all bookings",
    response_description="List of all bookings with metadata"
)
async def list_bookings():
    """Retrieve all bookings with enhanced error handling."""
    try:
        bookings = booking_service.get_all_bookings()
        
        # Manually construct BookingResponse instances with ISO-formatted datetime fields
        booking_responses = []
        for b in bookings:
            booking_data = {
                "id": b.id,
                "customer_name": b.customer_name,
                "technician_name": b.technician_name,
                "profession": b.profession,
                "start_time": b.start_time.isoformat(),
                "end_time": b.end_time.isoformat()
            }
            booking_responses.append(BookingResponse(**booking_data))
        
        return create_success_response(
            data=booking_responses,
            metadata={
                "total_count": len(bookings),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Failed to list bookings: {str(e)}", exc_info=True)
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "BOOKING_LIST_FAILED",
            "Failed to retrieve bookings"
        )

@router.get(
    "/{booking_id}",
    response_model=APIResponse,
    summary="Retrieve booking details"
)
async def retrieve_booking(booking_id: str):
    """Retrieve a specific booking with enhanced error handling."""
    try:
        booking = booking_service.get_booking_by_id(booking_id)
        if not booking:
            return create_error_response(
                status.HTTP_404_NOT_FOUND,
                "BOOKING_NOT_FOUND",
                f"Booking {booking_id} not found"
            )
        
        # Manually construct BookingResponse with ISO-formatted datetime fields
        booking_data = {
            "id": booking.id,
            "customer_name": booking.customer_name,
            "technician_name": booking.technician_name,
            "profession": booking.profession,
            "start_time": booking.start_time.isoformat(),
            "end_time": booking.end_time.isoformat()
        }
        booking_response = BookingResponse(**booking_data)
        
        return create_success_response(
            data=booking_response,
            metadata={"retrieved_at": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error(f"Error retrieving booking {booking_id}: {str(e)}", exc_info=True)
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "BOOKING_RETRIEVAL_FAILED",
            "Failed to retrieve booking"
        )

@router.post(
    "/",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_booking(booking_data: BookingCreate):
    """Create a new booking with enhanced validation."""
    try:
        new_booking = booking_service.create_booking(booking_data)
        
        # Manually construct BookingResponse with ISO-formatted datetime fields
        booking_response_data = {
            "id": new_booking.id,
            "customer_name": new_booking.customer_name or "Anonymous Customer",
            "technician_name": new_booking.technician_name,
            "profession": new_booking.profession,
            "start_time": new_booking.start_time.isoformat(), 
            "end_time": new_booking.end_time.isoformat()  
        }

        booking_response = BookingResponse(**booking_response_data) 

        
        return create_success_response(
            data=booking_response,
            metadata={
                "created_at": datetime.utcnow().isoformat() + "Z",
                "booking_duration": "1 hour"
            }
        )
    except ValueError as ve:
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            "BOOKING_CREATION_FAILED",
            str(ve),
            details={"provided_data": booking_data.dict()}
        )
    except Exception as e:
        logger.error(f"Error creating booking: {str(e)}", exc_info=True)
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "Failed to create booking"
        )

@router.delete(
    "/{booking_id}",
    response_model=APIResponse,
    summary="Cancel booking"
)
async def cancel_booking(booking_id: str):
    """Cancel a booking with enhanced error handling."""
    try:
        if not booking_service.cancel_booking(booking_id):
            return create_error_response(
                status.HTTP_404_NOT_FOUND,
                "BOOKING_NOT_FOUND",
                f"Booking {booking_id} not found"
            )
        
        return create_success_response(
            data={"booking_id": booking_id, "status": "cancelled"},
            metadata={"cancelled_at": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error(f"Error cancelling booking {booking_id}: {str(e)}", exc_info=True)
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "CANCELLATION_FAILED",
            "Failed to cancel booking"
        )

@router.post(
    "/commands",
    response_model=APIResponse,
    summary="Process natural language command"
)
async def process_command(command: CommandRequest):
    """Process natural language commands with enhanced analysis."""
    try:
        # Get intent classification with confidence scores
        intent, scores = nlp_service.classify_intent(command.message)
        
        # Create analysis scores
        analysis = [
            AnalysisScore(
                intent=intent_name,
                confidence=score,
                assessment=get_confidence_assessment(score)
            )
            for intent_name, score in scores.items()
        ]

        # Process the command
        response = nlp_service.handle_message(command.message)
        
        # Manually serialize booking datetime fields if present
        booking = response.booking
        if booking:
            booking = booking.dict()
            booking["start_time"] = booking["start_time"].isoformat()
            booking["end_time"] = booking["end_time"].isoformat()
            booking_response = BookingResponse(**booking)
        else:
            booking_response = None
        
        return create_success_response(
            data=CommandResult(
                success=True,
                intent=intent,
                message=response.response,
                analysis=analysis,
                booking=booking_response,
                metadata={
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_ms": 0  # You could add actual processing time here
                }
            ).dict(),
            metadata={}
        )
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}", exc_info=True)
        return create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "COMMAND_PROCESSING_FAILED",
            "Failed to process command",
            details={"original_command": command.message}
        )
