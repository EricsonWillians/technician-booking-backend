"""
validation.py

Provides additional validation utilities for Booking creation and updates.
These helper functions can be used by the service layer to ensure data
integrity beyond basic Pydantic checks, especially when you need more
complex or multi-field rules.
"""

from datetime import datetime
from typing import Optional


def validate_booking_time(
    start_time: datetime,
    end_time: Optional[datetime] = None
) -> None:
    """
    Checks whether the proposed booking times are valid.

    This function can be extended to enforce domain-specific constraints
    such as a minimum advance booking time, closing hours, etc.

    Args:
        start_time (datetime): The scheduled start time.
        end_time (datetime): Optionally, the scheduled end time.

    Raises:
        ValueError: If the start or end times fail the validation rules.
    """
    if start_time < datetime.now():
        raise ValueError("Cannot book a technician in the past.")

    # If end_time is provided, verify it is after start_time
    if end_time and end_time <= start_time:
        raise ValueError("The end time must be strictly after the start time.")


def validate_profession(profession: str) -> None:
    """
    Placeholder for additional profession-related checks.

    For example, if you only allow certain predefined professions in your system
    (like "Plumber", "Electrician", "Welder"), you can enforce that here.

    Args:
        profession (str): The profession to validate.

    Raises:
        ValueError: If the profession is not recognized or not allowed.
    """
    allowed_professions = {"plumber", "electrician", "welder"}

    if profession.lower() not in allowed_professions:
        raise ValueError(
            f"Unsupported profession '{profession}'. "
            f"Valid options: {', '.join(allowed_professions)}."
        )
