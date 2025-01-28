"""
initial_data.py

This module provides functionality for seeding initial data into the
Technician Booking System, particularly for development or testing
purposes. It ensures that when the application starts, the system has
some default bookings available.
"""

from typing import NoReturn
from datetime import datetime
from dateutil.parser import parse
from app.services.booking_service import in_memory_bookings_db, create_booking
from app.schemas.booking import BookingCreate


async def load_initial_data() -> NoReturn:
    """
    Seed the system with a predefined set of bookings. This function is
    intended to be called once on application startup.

    If the system already contains data (in_memory_bookings_db is not empty),
    the function returns immediately to avoid re-inserting the same entries.
    """
    if in_memory_bookings_db: 
        return

    # Example initial data
    initial_bookings = [
        {
            "customer_name": "Nicolas Woollett",
            "technician_name": "Nicolas Woollett", 
            "profession": "Plumber",
            "start_time_str": "15/10/2022 10:00 AM"
        },
        {
            "customer_name": "Franky Flay",
            "technician_name": "Franky Flay", 
            "profession": "Electrician",
            "start_time_str": "16/10/2022 06:00 PM"
        },
        {
            "customer_name": "Griselda Dickson",
            "technician_name": "Griselda Dickson",
            "profession": "Welder",
            "start_time_str": "18/10/2022 11:00 AM"
        }
    ]

    for item in initial_bookings:
        # Parse time
        start_dt = parse(item["start_time_str"])
        booking_data = BookingCreate(
            customer_name=item["customer_name"],
            technician_name=item["technician_name"],
            profession=item["profession"],
            start_time=start_dt
        )
        # Note: create_booking handles one-hour duration logic
        create_booking(booking_data)