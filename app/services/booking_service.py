"""
==========================================================
                BOOKING SERVICE MODULE
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

from datetime import timedelta
from typing import Dict, List, Optional

from app.models.booking import Booking
from app.schemas.booking import BookingCreate
from app.services.validation import validate_booking_request, validate_booking_time
from uuid import UUID

# ---------------------------------------------------------------------------
# In-memory storage for Bookings
# ---------------------------------------------------------------------------
in_memory_bookings_db: Dict[str, Booking] = {}  # Keyed by booking.id


def get_all_bookings() -> List[Booking]:
    """
    Retrieve a list of all existing bookings.

    Returns:
        A list of Booking objects in the system.
    """
    return list(in_memory_bookings_db.values())


def get_booking_by_id(booking_id: str) -> Optional[Booking]:
    """
    Retrieve a specific booking by its unique ID.

    Args:
        booking_id: The UUID string of the desired booking.

    Returns:
        Booking if found, otherwise None.
    """
    return in_memory_bookings_db.get(booking_id)


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
        return True
    return False


def create_booking(booking_data: BookingCreate, system_init: bool = False) -> Booking:
    """
    Create a new booking, ensuring all business rules and constraints are satisfied.

    Args:
        booking_data: Pydantic schema containing booking details
        system_init: When True, bypasses time validation for system initialization

    Returns:
        The newly created Booking (domain model)

    Raises:
        ValueError: If the booking violates any business rules or constraints
    """
    start_time = booking_data.start_time
    end_time = start_time + timedelta(hours=1)

    validate_booking_request(
        start_time=start_time,
        end_time=end_time,
        technician_name=booking_data.technician_name,
        profession=booking_data.profession,
        existing_bookings=in_memory_bookings_db,
        system_init=system_init
    )

    new_booking = Booking(
        customer_name=booking_data.customer_name,
        technician_name=booking_data.technician_name,
        profession=booking_data.profession,
        start_time=start_time,
        end_time=end_time
    )
    in_memory_bookings_db[new_booking.id] = new_booking
    return new_booking


def create_booking_from_llm(parsed_data: dict) -> Booking:
    """
    Create a booking from partially structured data coming from the LLM.

    Args:
        parsed_data: NLP-extracted data containing booking details

    Returns:
        The newly created booking object

    Raises:
        ValueError: If the data is invalid or violates business rules
    """
    required_keys = ["customer_name", "technician_name", "profession", "start_time"]
    missing_keys = [key for key in required_keys if key not in parsed_data]
    
    if missing_keys:
        raise ValueError(f"Missing required booking fields: {', '.join(missing_keys)}")

    booking_schema = BookingCreate(
        customer_name=parsed_data["customer_name"],
        technician_name=parsed_data["technician_name"],
        profession=parsed_data["profession"],
        start_time=parsed_data["start_time"]
    )
    return create_booking(booking_schema)


def cancel_booking(booking_id: str) -> bool:
    """
    Cancel (delete) a booking by its ID.

    Args:
        booking_id: The booking ID to remove

    Returns:
        True if the booking was successfully removed, otherwise False
    """
    return delete_booking_by_id(booking_id)