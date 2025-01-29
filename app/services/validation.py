"""
validation.py

Provides validation utilities for the Technician Booking System, supporting both
regular operations and system initialization scenarios.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from app.config.settings import settings
from app.models.booking import Booking


def validate_booking_time(
    start_time: datetime,
    end_time: Optional[datetime] = None,
    technician_name: Optional[str] = None,
    existing_bookings: Optional[Dict[str, Booking]] = None,
    system_init: bool = False
) -> None:
    """
    Validates booking time constraints according to business rules.
    
    Args:
        start_time: The proposed booking start time
        end_time: The proposed booking end time
        technician_name: The name of the technician for conflict checking
        existing_bookings: Dictionary of current bookings to check for conflicts
        system_init: When True, bypasses the past-time validation for system initialization
    
    Raises:
        ValueError: If any time-related constraint is violated
    """
    if not system_init and start_time < datetime.now():
        raise ValueError("Cannot book a technician in the past.")

    # Enforce one-hour duration
    calculated_end_time = start_time + timedelta(hours=1)
    if end_time and end_time != calculated_end_time:
        raise ValueError("All bookings must be exactly one hour long.")
        
    # If no end time provided, use calculated one-hour duration
    end_time = end_time or calculated_end_time

    # Check for technician scheduling conflicts
    if technician_name and existing_bookings:
        for booking in existing_bookings.values():
            if booking.technician_name.lower() == technician_name.lower():
                if not (end_time <= booking.start_time or start_time >= booking.end_time):
                    raise ValueError(
                        f"Time conflict: {technician_name} is already booked "
                        f"from {booking.start_time.strftime('%Y-%m-%d %I:%M %p')} "
                        f"to {booking.end_time.strftime('%Y-%m-%d %I:%M %p')}."
                    )


def validate_profession(profession: str) -> None:
    """
    Validates that the provided profession is supported by the system.
    
    Args:
        profession: The profession to validate
    
    Raises:
        ValueError: If the profession is not recognized or not allowed
    """
    allowed_professions = {
        prof.lower() for prof in settings.PROFESSION_KEYWORDS.keys()
    }

    if profession.lower() not in allowed_professions:
        raise ValueError(
            f"Unsupported profession '{profession}'. "
            f"Valid options: {', '.join(sorted(allowed_professions))}."
        )


def validate_booking_request(
    start_time: datetime,
    end_time: Optional[datetime],
    technician_name: str,
    profession: str,
    existing_bookings: Dict[str, Booking],
    system_init: bool = False
) -> None:
    """
    Performs comprehensive validation of a booking request.
    
    Args:
        start_time: The proposed booking start time
        end_time: The proposed booking end time
        technician_name: The name of the technician
        profession: The technician's profession
        existing_bookings: Dictionary of current bookings in the system
        system_init: When True, bypasses past-time validation for system initialization
    
    Raises:
        ValueError: If any validation constraint is violated
    """
    validate_profession(profession)
    validate_booking_time(
        start_time=start_time,
        end_time=end_time,
        technician_name=technician_name,
        existing_bookings=existing_bookings,
        system_init=system_init
    )