"""
bookings.py

This module defines a FastAPI router for CRUD operations on bookings.
It relies on the booking_service for core business logic. Endpoints
ensure that:
  1. A technician cannot be booked twice at the same time.
  2. Bookings are one hour long by default.
  3. Clients can create, list, retrieve, and delete bookings.

Endpoints:
  - GET     /bookings          -> list_all_bookings
  - GET     /bookings/{id}     -> retrieve_booking
  - POST    /bookings          -> create_new_booking
  - DELETE  /bookings/{id}     -> delete_booking
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from app.schemas.booking import BookingCreate, BookingResponse
from app.services import booking_service

router = APIRouter()


@router.get("/", response_model=List[BookingResponse], status_code=status.HTTP_200_OK)
def list_all_bookings() -> List[BookingResponse]:
    """
    List all technician bookings currently in the system.

    Returns:
        A list of BookingResponse objects representing all bookings.
    """
    bookings = booking_service.get_all_bookings()
    return [
        BookingResponse(
            id=b.id,
            customer_name=b.customer_name,
            technician_name=b.technician_name,
            profession=b.profession,
            start_time=b.start_time,
            end_time=b.end_time
        )
        for b in bookings
    ]


@router.get("/{booking_id}", response_model=BookingResponse, status_code=status.HTTP_200_OK)
def retrieve_booking(booking_id: str) -> BookingResponse:
    """
    Retrieve a single booking by its unique ID.

    Args:
        booking_id (str): The UUID string of the booking to retrieve.

    Raises:
        HTTPException(404): If the booking does not exist.

    Returns:
        The booking's details as a BookingResponse object.
    """
    booking = booking_service.get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID '{booking_id}' not found."
        )
    return BookingResponse(
        id=booking.id,
        customer_name=booking.customer_name,
        technician_name=booking.technician_name,
        profession=booking.profession,
        start_time=booking.start_time,
        end_time=booking.end_time
    )


@router.post("/", response_model=BookingResponse, status_code=status.HTTP_201_CREATED)
def create_new_booking(data: BookingCreate) -> BookingResponse:
    """
    Create a new booking. Ensures that:
      - The requested start time does not overlap with
        an existing booking for the same technician.
      - The booking will last exactly one hour.

    Args:
        data (BookingCreate): Request body including
            customer_name, technician_name, profession, and start_time.

    Raises:
        HTTPException(400): If there is a time overlap or invalid data.

    Returns:
        The newly created booking as a BookingResponse.
    """
    try:
        new_booking = booking_service.create_booking(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    return BookingResponse(
        id=new_booking.id,
        customer_name=new_booking.customer_name,
        technician_name=new_booking.technician_name,
        profession=new_booking.profession,
        start_time=new_booking.start_time,
        end_time=new_booking.end_time
    )


@router.delete("/{booking_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_booking(booking_id: str) -> None:
    """
    Delete a booking by its UUID.

    Args:
        booking_id (str): The UUID of the booking to remove.

    Raises:
        HTTPException(404): If the booking does not exist.

    Returns:
        None
    """
    success = booking_service.delete_booking_by_id(booking_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID '{booking_id}' not found."
        )
