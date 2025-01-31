"""
==========================================================
                TECHNICIAN BOOKING SYSTEM - SERVICE
==========================================================

Implements the core business logic (CRUD operations) 
for managing bookings within the Technician Booking System.

- Uses an in-memory data store by default
- Can be extended to connect with a real database in production

Author : Ericson Willians  
Email  : ericsonwillians@protonmail.com  
Date   : January 2025  

==========================================================
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.models.booking import Booking
from app.schemas.booking import BookingCreate, BookingResponse
from app.services.validation import validate_booking_request
from app.models.professions import ProfessionEnum
from uuid import UUID
import logging

# Configure logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory storage for Bookings
# ---------------------------------------------------------------------------
in_memory_bookings_db: Dict[str, Booking] = {}  # Keyed by booking.id


def get_all_bookings() -> List[BookingResponse]:
    """
    Retrieve a list of all existing bookings.

    Returns:
        A list of BookingResponse objects in the system.
    """
    bookings = list(in_memory_bookings_db.values())
    return [BookingResponse(
        id=booking.id,
        customer_name=booking.customer_name,
        technician_name=booking.technician_name,
        profession=booking.profession,
        start_time=booking.start_time,
        end_time=booking.end_time
    ) for booking in bookings]


def get_booking_by_id(booking_id: str) -> Optional[BookingResponse]:
    """
    Retrieve a specific booking by its unique ID.

    Args:
        booking_id: The UUID string of the desired booking.

    Returns:
        BookingResponse if found, otherwise None.
    """
    booking = in_memory_bookings_db.get(booking_id)
    if booking:
        return BookingResponse(
            id=booking.id,
            customer_name=booking.customer_name,
            technician_name=booking.technician_name,
            profession=booking.profession,
            start_time=booking.start_time,
            end_time=booking.end_time
        )
    logger.warning(f"Booking with ID {booking_id} not found.")
    return None


def delete_booking_by_id(booking_id: str) -> bool:
    """
    Remove a booking from the system by ID.

    Args:
        booking_id: The UUID of the booking to remove.

    Returns:
        True if the booking was found and removed; False otherwise.
    """
    if booking_id in in_memory_bookings_db:
        del in_memory_bookings_db[booking_id]
        logger.info(f"Booking with ID {booking_id} has been deleted.")
        return True
    logger.warning(f"Attempted to delete non-existent booking with ID {booking_id}.")
    return False


def create_booking(booking_data: BookingCreate, system_init: bool = False) -> BookingResponse:
    """
    Create a new booking, ensuring all business rules and constraints are satisfied.

    Args:
        booking_data: Pydantic schema containing booking details.
        system_init: When True, bypasses time validation for system initialization.

    Returns:
        The newly created BookingResponse (schema model).

    Raises:
        ValueError: If the booking violates any business rules or constraints.
    """
    start_time = booking_data.start_time
    end_time = start_time + timedelta(hours=1)

    # Validate the booking request against business rules
    validate_booking_request(
        start_time=start_time,
        end_time=end_time,
        technician_name=booking_data.technician_name,
        profession=booking_data.profession,
        existing_bookings=in_memory_bookings_db,
        system_init=system_init
    )

    # Create a new Booking instance
    new_booking = Booking(
        customer_name=booking_data.customer_name,
        technician_name=booking_data.technician_name,
        profession=booking_data.profession.value,  # Extract string from enum
        start_time=start_time,
        end_time=end_time
    )

    # Add the new booking to the in-memory database
    in_memory_bookings_db[new_booking.id] = new_booking
    logger.info(f"New booking created with ID: {new_booking.id}")

    # Return the BookingResponse schema
    return BookingResponse(
        id=new_booking.id,
        customer_name=new_booking.customer_name,
        technician_name=new_booking.technician_name,
        profession=new_booking.profession,
        start_time=new_booking.start_time,
        end_time=new_booking.end_time
    )


def cancel_booking(booking_id: str) -> bool:
    """
    Cancel (delete) a booking by its ID.

    Args:
        booking_id: The booking ID to remove.

    Returns:
        True if the booking was successfully removed, otherwise False.
    """
    return delete_booking_by_id(booking_id)

def is_overlapping(technician_name: str, start_time: datetime) -> bool:
    """
    Checks if the technician is already booked at the given start_time.
    
    Args:
        technician_name: Name of the technician
        start_time: Proposed booking start time
        
    Returns:
        bool: True if there is an overlap, False otherwise
    """
    end_time = start_time + timedelta(hours=1)
    
    # Convert dict values to list for iteration
    bookings = list(in_memory_bookings_db.values())
    
    for booking in bookings:
        if booking.technician_name == technician_name:
            if max(start_time, booking.start_time) < min(end_time, booking.end_time):
                return True
    return False