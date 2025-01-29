"""
bookings.py

A FastAPI router dedicated to Booking CRUD operations within the Technician Booking System.

Endpoints:
  1. GET    /bookings
     - List all existing bookings in the system.
  2. GET    /bookings/{booking_id}
     - Retrieve details of a specific booking by its unique ID (UUID or numeric).
  3. POST   /bookings
     - Create a new booking with validated request data (customer, technician, etc.).
  4. DELETE /bookings/{booking_id}
     - Cancel or remove a booking by ID.

All endpoints return appropriate HTTP status codes and responses,
with standardized error handling (e.g., 404 if a booking is not found).
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from app.services import booking_service
from app.schemas.booking import BookingCreate, BookingResponse

router = APIRouter()

@router.get(
    "/",
    response_model=List[BookingResponse],
    summary="List all bookings",
    response_description="A list of all stored bookings in the system."
)
def list_bookings():
    """
    Retrieve all existing bookings from the in-memory store (or future database).

    **OpenAPI:**
    - **OperationID**: "listBookings"
    - **Tags**: ["Bookings"]
    - **Responses**:
      - 200: Successful retrieval of a (possibly empty) list of bookings.
      - 500: Internal server error if something unexpected happens.

    **Business Logic:**
    Simply enumerates all currently known Bookings without pagination.
    This endpoint is read-only; no modifications occur.

    Returns:
        List[BookingResponse]: Zero or more Bookings as stored in the system.
    """
    bookings = booking_service.get_all_bookings()
    # Convert domain Booking objects to Pydantic `BookingResponse`.
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


@router.get(
    "/{booking_id}",
    response_model=BookingResponse,
    summary="Retrieve a specific booking by ID",
    response_description="The details of the requested booking if found."
)
def retrieve_booking(booking_id: str):
    """
    Retrieve a single booking by its unique identifier.

    **OpenAPI:**
    - **OperationID**: "retrieveBooking"
    - **Tags**: ["Bookings"]
    - **Parameters**:
      - booking_id (str): The unique booking identifier (UUID string or numeric).
    - **Responses**:
      - 200: Returns the booking record if located.
      - 404: Booking not found with the given ID.
      - 500: Internal server error if something unexpected happens.

    **Business Logic:**
    Looks up an existing booking in the in-memory dictionary (or future DB).
    If not found, raises a 404 Not Found error.

    Args:
        booking_id (str):
            A string representing either a UUID or numeric ID as recognized 
            by the system.

    Returns:
        BookingResponse: The booking record matching the given ID.

    Raises:
        HTTPException(404): If no matching booking is found.
    """
    booking = booking_service.get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID {booking_id} not found."
        )
    return BookingResponse(
        id=booking.id,
        customer_name=booking.customer_name,
        technician_name=booking.technician_name,
        profession=booking.profession,
        start_time=booking.start_time,
        end_time=booking.end_time
    )


@router.post(
    "/",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new booking",
    response_description="The newly created booking record."
)
def create_new_booking(booking_data: BookingCreate):
    """
    Create a new booking in the system.

    **OpenAPI:**
    - **OperationID**: "createBooking"
    - **Tags**: ["Bookings"]
    - **Request Body**:
      - A JSON object conforming to the `BookingCreate` schema.
    - **Responses**:
      - 201: The booking has been successfully created.
      - 400: Bad request if validation fails (e.g. missing fields or invalid profession).
      - 500: Internal server error if something unexpected happens.

    **Business Logic:**
    1. Validates the incoming data (customer_name, technician_name, etc.).
    2. Delegates to `create_booking` in booking_service for the 
       actual creation, which also sets an ID and end_time.

    Args:
        booking_data (BookingCreate):
            The validated request body containing all necessary
            booking fields (name, profession, start_time, etc.).

    Returns:
        BookingResponse: The newly persisted booking, complete with
        generated ID and computed end_time.

    Raises:
        HTTPException(400): If the booking logic raises a ValueError
                            (e.g. invalid profession).
    """
    try:
        new_booking = booking_service.create_booking(booking_data)
        return BookingResponse(
            id=new_booking.id,
            customer_name=new_booking.customer_name,
            technician_name=new_booking.technician_name,
            profession=new_booking.profession,
            start_time=new_booking.start_time,
            end_time=new_booking.end_time
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )


@router.delete(
    "/{booking_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete/Cancel a booking",
    response_description="No content if deletion is successful."
)
def delete_booking(booking_id: str):
    """
    Cancel or delete a booking by its unique identifier.

    **OpenAPI:**
    - **OperationID**: "deleteBooking"
    - **Tags**: ["Bookings"]
    - **Parameters**:
      - booking_id (str): The unique booking identifier to remove.
    - **Responses**:
      - 204: The booking was successfully deleted (no content returned).
      - 404: Booking was not found with the provided ID.
      - 500: Internal server error if something unexpected happens.

    **Business Logic:**
    If the booking exists, it is removed from the system. 
    Otherwise, a 404 is returned.

    Args:
        booking_id (str): The booking's unique ID in string (UUID or numeric).

    Raises:
        HTTPException(404): If the booking is not found in the store.
    """
    success = booking_service.cancel_booking(booking_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID {booking_id} not found."
        )
    # status_code=204 → no content in successful response
