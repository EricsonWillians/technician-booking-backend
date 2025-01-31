# tests/unit/test_booking_service.py

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from uuid import UUID

from app.services import booking_service
from app.schemas.booking import BookingCreate, BookingResponse
from app.models.professions import ProfessionEnum
from app.models.booking import Booking
from app.config.settings import settings

@pytest.fixture(autouse=True)
def clear_bookings():
    """Clear the in-memory booking database before and after each test."""
    booking_service.in_memory_bookings_db.clear()
    yield
    booking_service.in_memory_bookings_db.clear()

@pytest.fixture
def sample_booking_data():
    """Create sample booking data for tests."""
    return {
        "customer_name": "John Doe",
        "technician_name": "Bob Smith",
        "profession": ProfessionEnum.PLUMBER,
        "start_time": datetime.now(ZoneInfo(settings.TIMEZONE)) + timedelta(days=1)
    }

@pytest.fixture
def create_sample_booking(sample_booking_data):
    """Create a sample booking and return its response."""
    booking_create = BookingCreate(**sample_booking_data)
    return booking_service.create_booking(booking_create)

class TestBookingCreation:
    """Test suite for booking creation functionality."""

    def test_successful_booking_creation(self, sample_booking_data):
        """Test creating a valid booking."""
        booking_create = BookingCreate(**sample_booking_data)
        response = booking_service.create_booking(booking_create)
        
        assert isinstance(response, BookingResponse)
        assert isinstance(UUID(response.id), UUID)  # Verify valid UUID
        assert response.customer_name == sample_booking_data["customer_name"]
        assert response.technician_name == sample_booking_data["technician_name"]
        assert response.profession == sample_booking_data["profession"]
        assert response.start_time == sample_booking_data["start_time"]
        assert response.end_time == sample_booking_data["start_time"] + timedelta(hours=1)

    def test_booking_in_past(self, sample_booking_data):
        """Test that booking in the past raises an error."""
        sample_booking_data["start_time"] = datetime.now(ZoneInfo(settings.TIMEZONE)) - timedelta(days=1)
        booking_create = BookingCreate(**sample_booking_data)
        
        with pytest.raises(ValueError, match="Cannot book a technician in the past"):  # Matches exact message
            booking_service.create_booking(booking_create)
            booking_service.create_booking(booking_create)

    def test_overlapping_bookings(self, sample_booking_data, create_sample_booking):
        """Test that overlapping bookings are not allowed."""
        # Try to create another booking for the same technician at the same time
        booking_create = BookingCreate(**sample_booking_data)
        
        with pytest.raises(ValueError, match="Time conflict"):  # Just match the start of the error message
            booking_service.create_booking(booking_create)

    def test_invalid_profession(self, sample_booking_data):
        """Test that invalid professions are rejected."""
        sample_booking_data["profession"] = "InvalidProfession"
        with pytest.raises(ValueError):
            BookingCreate(**sample_booking_data)

    @pytest.mark.parametrize("field", ["customer_name", "technician_name", "profession", "start_time"])
    def test_missing_required_fields(self, sample_booking_data, field):
        """Test that missing required fields raise appropriate errors."""
        del sample_booking_data[field]
        with pytest.raises(ValueError):
            BookingCreate(**sample_booking_data)

class TestBookingRetrieval:
    """Test suite for booking retrieval functionality."""

    def test_get_all_bookings_empty(self):
        """Test retrieving bookings when none exist."""
        bookings = booking_service.get_all_bookings()
        assert isinstance(bookings, list)
        assert len(bookings) == 0

    def test_get_all_bookings(self, create_sample_booking):
        """Test retrieving all bookings."""
        bookings = booking_service.get_all_bookings()
        assert len(bookings) == 1
        assert isinstance(bookings[0], BookingResponse)
        assert bookings[0].id == create_sample_booking.id

    def test_get_booking_by_id_exists(self, create_sample_booking):
        """Test retrieving an existing booking by ID."""
        booking = booking_service.get_booking_by_id(create_sample_booking.id)
        assert booking is not None
        assert booking.id == create_sample_booking.id

    def test_get_booking_by_id_not_exists(self):
        """Test retrieving a non-existent booking."""
        booking = booking_service.get_booking_by_id("non-existent-id")
        assert booking is None

class TestBookingCancellation:
    """Test suite for booking cancellation functionality."""

    def test_cancel_existing_booking(self, create_sample_booking):
        """Test cancelling an existing booking."""
        result = booking_service.cancel_booking(create_sample_booking.id)
        assert result is True
        assert booking_service.get_booking_by_id(create_sample_booking.id) is None

    def test_cancel_non_existent_booking(self):
        """Test attempting to cancel a non-existent booking."""
        result = booking_service.cancel_booking("non-existent-id")
        assert result is False

class TestOverlappingBookings:
    """Test suite for booking overlap detection."""

    def test_overlapping_same_time(self, sample_booking_data, create_sample_booking):
        """Test detecting overlap for same time slot."""
        is_overlapping = booking_service.is_overlapping(
            sample_booking_data["technician_name"],
            sample_booking_data["start_time"]
        )
        assert is_overlapping is True

    def test_overlapping_partial(self, sample_booking_data, create_sample_booking):
        """Test detecting partial time overlap."""
        start_time = sample_booking_data["start_time"] + timedelta(minutes=30)
        is_overlapping = booking_service.is_overlapping(
            sample_booking_data["technician_name"],
            start_time
        )
        assert is_overlapping is True

    def test_non_overlapping_different_time(self, sample_booking_data, create_sample_booking):
        """Test non-overlapping times."""
        start_time = sample_booking_data["start_time"] + timedelta(hours=2)
        is_overlapping = booking_service.is_overlapping(
            sample_booking_data["technician_name"],
            start_time
        )
        assert is_overlapping is False

    def test_non_overlapping_different_technician(self, sample_booking_data, create_sample_booking):
        """Test same time but different technician."""
        is_overlapping = booking_service.is_overlapping(
            "Different Technician",
            sample_booking_data["start_time"]
        )
        assert is_overlapping is False

@pytest.mark.parametrize("profession", [prof for prof in ProfessionEnum if prof != ProfessionEnum.UNKNOWN])
def test_all_professions(sample_booking_data, profession):
    """Test booking creation with all valid professions."""
    sample_booking_data["profession"] = profession
    booking_create = BookingCreate(**sample_booking_data)
    response = booking_service.create_booking(booking_create)
    assert response.profession == profession

def test_system_init_bypass_validation(sample_booking_data):
    """Test that system_init flag bypasses time validation."""
    sample_booking_data["start_time"] = datetime.now(ZoneInfo(settings.TIMEZONE)) - timedelta(days=1)
    booking_create = BookingCreate(**sample_booking_data)
    
    # Should succeed with system_init=True despite being in the past
    response = booking_service.create_booking(booking_create, system_init=True)
    assert isinstance(response, BookingResponse)

def test_booking_end_time_calculation(sample_booking_data):
    """Test that end_time is correctly calculated as start_time + 1 hour."""
    booking_create = BookingCreate(**sample_booking_data)
    response = booking_service.create_booking(booking_create)
    
    expected_end_time = sample_booking_data["start_time"] + timedelta(hours=1)
    assert response.end_time == expected_end_time
    
def test_unknown_profession_rejection(sample_booking_data):
    """Test that UNKNOWN profession is rejected."""
    sample_booking_data["profession"] = ProfessionEnum.UNKNOWN
    booking_create = BookingCreate(**sample_booking_data)
    with pytest.raises(ValueError, match="Unsupported profession"):
        booking_service.create_booking(booking_create)