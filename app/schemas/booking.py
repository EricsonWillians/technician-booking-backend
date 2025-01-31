# app/schemas/booking.py

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from app.models.professions import ProfessionEnum


class BookingBase(BaseModel):
    """
    Base schema for Booking-related data.
    
    Provides core fields and validation shared across booking operations.
    """
    customer_name: Optional[str] = Field(
        "Anonymous Customer",  # Default value
        example="Nicolas Woollett",
        description="Name of the customer requesting the service"
    )
    technician_name: str = Field(
        ...,
        example="Nicolas Woollett",
        description="Name of the technician performing the service"
    )
    profession: ProfessionEnum = Field(
        ...,
        example="Plumber",
        description="Professional qualification of the technician"
    )
    start_time: datetime = Field(
        ...,
        example="2022-10-15T10:00:00",
        description="Scheduled start time of the service"
    )

    model_config = ConfigDict(
        extra='forbid'  # Forbid extra fields not defined in the model
    )


class BookingCreate(BookingBase):
    """
    Schema for creating new bookings.
    Used by both the API and internal service layer for booking creation.
    """
    class Config:
        schema_extra = {
            "example": {
                "customer_name": "Nicolas Woollett",
                "technician_name": "Nicolas Woollett",
                "profession": "Plumber",
                "start_time": "2022-10-15T10:00:00"
            }
        }


class BookingResponse(BookingBase):
    """
    Schema for booking responses.
    Extends the base schema with system-generated fields.
    """
    id: str = Field(
        ...,
        example="123e4567-e89b-12d3-a456-426614174000",
        description="Unique identifier for the booking"
    )
    end_time: datetime = Field(
        ...,
        example="2022-10-15T11:00:00",
        description="Scheduled end time (automatically set to 1 hour after start_time)"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "customer_name": "Nicolas Woollett",
                "technician_name": "Nicolas Woollett",
                "profession": "Plumber",
                "start_time": "2022-10-15T10:00:00",
                "end_time": "2022-10-15T11:00:00"
            }
        }
