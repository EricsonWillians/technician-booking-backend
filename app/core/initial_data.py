"""
initial_data.py

This module provides functionality for seeding initial data into the
Technician Booking System using the established schema format.
"""

from typing import NoReturn
from datetime import datetime
from app.services.booking_service import in_memory_bookings_db, create_booking
from app.schemas.booking import BookingCreate


async def load_initial_data() -> NoReturn:
    """
    Initialize the system with required initial bookings using the standard
    BookingCreate schema. This function populates the global in_memory_bookings_db
    and should be called once during application startup.
    """
    if in_memory_bookings_db:
        return

    initial_bookings = [
        BookingCreate(
            customer_name="Nicolas Woollett",
            technician_name="Nicolas Woollett",
            profession="Plumber",
            start_time=datetime(2022, 10, 15, 10, 0)
        ),
        BookingCreate(
            customer_name="Franky Flay",
            technician_name="Franky Flay",
            profession="Electrician",
            start_time=datetime(2022, 10, 16, 18, 0)
        ),
        BookingCreate(
            customer_name="Griselda Dickson",
            technician_name="Griselda Dickson",
            profession="Welder",
            start_time=datetime(2022, 10, 18, 11, 0)
        )
    ]

    try:
        for booking_data in initial_bookings:
            # Pass system_init=True to bypass time validation for initial data
            create_booking(booking_data, system_init=True)
    except Exception as e:
        in_memory_bookings_db.clear()
        raise RuntimeError(f"Failed to initialize booking system: {str(e)}")