"""
==========================================================
            TECHNICIAN BOOKING SYSTEM - MODEL
==========================================================

  Defines the core Booking domain model used within the  
  Technician Booking System.  

  - Implemented as a simple dataclass for in-memory usage  
  - Can be extended to support a real database (e.g., SQLAlchemy)  
  - Automatically generates unique booking IDs (UUID)  

  Author : Ericson Willians  
  Email  : ericsonwillians@protonmail.com  
  Date   : January 2025  

==========================================================
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Booking:
    """
    Represents a single technician booking instance.

    Attributes:
        id (str): Unique identifier for the booking, auto-generated as a UUID.
        customer_name (str): Name of the customer requesting the service.
        technician_name (str): Name of the technician assigned to this booking.
        profession (str): The profession/specialization of the technician (e.g. 'Plumber').
        start_time (datetime): The start time of the scheduled service.
        end_time (datetime): The end time of the scheduled service.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_name: str = "Anonymous Customer"  # Default value
    technician_name: str = ""
    profession: str = ""
    start_time: datetime = datetime.now()
    end_time: datetime = datetime.now()
