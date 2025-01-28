"""
booking_service.py

Implements the core business logic (CRUD operations) for managing Bookings
within the Technician Booking System. Uses an in-memory data store by default,
but can be extended to connect with a real database in production.

Features:
  1. List all bookings
  2. Retrieve a booking by ID
  3. Delete a booking by ID
  4. Schedule a new booking (ensuring no overlap and one-hour duration)
Constraints:
  - A booking is one hour long
  - A technician cannot be booked twice at the same time
  - Uses validation checks from validation.py
"""

from datetime import timedelta
from typing import Dict, List, Optional

from app.models.booking import Booking
from app.schemas.booking import BookingCreate
from app.services.validation import validate_booking_time, validate_profession
from uuid import UUID

# ---------------------------------------------------------------------------
# In-memory storage for Bookings
# ---------------------------------------------------------------------------
in_memory_bookings_db: Dict[str, Booking] = {}  # Keyed by booking.id


# ---------------------------------------------------------------------------
# Core Service Methods
# ---------------------------------------------------------------------------
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
        booking_id (str): The UUID string of the desired booking.

    Returns:
        Booking if found, otherwise None.
    """
    return in_memory_bookings_db.get(booking_id)


def delete_booking_by_id(booking_id: str) -> bool:
    """
    Remove a booking from the system by ID.

    Args:
        booking_id (str): The UUID of the booking to remove.

    Returns:
        bool: True if the booking was found and removed; False otherwise.
    """
    if booking_id in in_memory_bookings_db:
        del in_memory_bookings_db[booking_id]
        return True
    return False


def create_booking(booking_data: BookingCreate) -> Booking:
    """
    Create a new booking, ensuring it does not conflict with an existing booking
    for the same technician. Automatically sets a one-hour duration.

    Args:
        booking_data (BookingCreate): Pydantic schema containing
            customer_name, technician_name, profession, and start_time.

    Raises:
        ValueError: If the requested slot overlaps with an existing booking
            for the same technician, the profession is not allowed,
            or the time is invalid (e.g., in the past).

    Returns:
        The newly created Booking (domain model).
    """
    # Validate profession and times (start_time, end_time)
    validate_profession(booking_data.profession)

    start_time = booking_data.start_time
    end_time = start_time + timedelta(hours=1)
    validate_booking_time(start_time, end_time)

    # Check for technician conflicts
    for existing_booking in in_memory_bookings_db.values():
        if existing_booking.technician_name.lower() == booking_data.technician_name.lower():
            # Overlap occurs if (start < existing.end) and (end > existing.start)
            if not (end_time <= existing_booking.start_time or start_time >= existing_booking.end_time):
                raise ValueError(
                    f"Time conflict: Technician '{booking_data.technician_name}' is already booked "
                    f"from {existing_booking.start_time} to {existing_booking.end_time}."
                )

    # Create the domain booking object
    new_booking = Booking(
        customer_name=booking_data.customer_name,
        technician_name=booking_data.technician_name,
        profession=booking_data.profession,
        start_time=start_time,
        end_time=end_time
    )
    in_memory_bookings_db[new_booking.id] = new_booking
    return new_booking


# ---------------------------------------------------------------------------
# Extended Methods for NLP-based CLI (optional, if you parse data differently)
# ---------------------------------------------------------------------------
def create_booking_from_llm(parsed_data: dict) -> Booking:
    """
    Create a booking from partially structured data coming from the LLM.

    This method assumes that `parsed_data` contains keys like:
      - 'customer_name' (str)
      - 'technician_name' (str)
      - 'profession' (str)
      - 'start_time' (datetime)
    and converts them to a BookingCreate schema. If any are missing,
    you may need additional checks or defaults.

    Args:
        parsed_data (dict): NLP-extracted data containing booking details.

    Returns:
        Booking: The newly created booking object.

    Raises:
        ValueError: If required keys are missing, a time conflict is detected,
                    or the profession is invalid.
    """
    # Example required fields; adapt as needed
    required_keys = ["customer_name", "technician_name", "profession", "start_time"]
    for key in required_keys:
        if key not in parsed_data:
            raise ValueError(f"Missing required booking field: '{key}'")

    booking_schema = BookingCreate(
        customer_name=parsed_data["customer_name"],
        technician_name=parsed_data["technician_name"],
        profession=parsed_data["profession"],
        start_time=parsed_data["start_time"],
    )
    return create_booking(booking_schema)


def cancel_booking(booking_id: str) -> bool:
    """
    Cancel (delete) a booking by its ID. Useful for NLP/LLM-based
    commands like "cancel booking 123".

    Args:
        booking_id (str): The booking ID to remove.

    Returns:
        True if the booking was successfully removed, otherwise False.
    """
    return delete_booking_by_id(booking_id)
