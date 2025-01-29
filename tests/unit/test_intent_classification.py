import unittest
from datetime import datetime, timedelta
from app.services.nlp_service import llm_processor

class TestIntentClassification(unittest.TestCase):
    def test_create_booking_intent(self):
        command = "I need a plumber tomorrow at 2pm"
        parsed = llm_processor.parse_user_input(command)
        self.assertEqual(parsed.intent, "create_booking")
        self.assertIn("profession", parsed.data)
        self.assertEqual(parsed.data["profession"], "plumber")
        self.assertGreaterEqual(parsed.data.get("confidence", 0), 0.4)

    def test_cancel_booking_intent(self):
        command = "Cancel booking #123"  # Updated to include explicit booking ID
        parsed = llm_processor.parse_user_input(command)
        self.assertEqual(parsed.intent, "cancel_booking")
        self.assertIn("booking_id", parsed.data)
        self.assertEqual(parsed.data["booking_id"], "123")
        self.assertGreaterEqual(parsed.data.get("confidence", 0), 0.4)

    def test_list_bookings_intent(self):
        command = "List all bookings"
        parsed = llm_processor.parse_user_input(command)
        self.assertEqual(parsed.intent, "list_bookings")
        self.assertGreaterEqual(parsed.data.get("confidence", 0), 0.4)

    def test_retrieve_booking_intent(self):
        command = "Get details for booking #456"  # Updated to include explicit booking ID
        parsed = llm_processor.parse_user_input(command)
        self.assertEqual(parsed.intent, "retrieve_booking")
        self.assertIn("booking_id", parsed.data)
        self.assertEqual(parsed.data["booking_id"], "456")
        self.assertGreaterEqual(parsed.data.get("confidence", 0), 0.4)

    def test_unknown_command_intent(self):
        command = "asdfqwerty"  # Using gibberish to ensure it's truly unrecognizable
        with self.assertRaises(ValueError):
            parsed = llm_processor.parse_user_input(command)

    def test_invalid_booking_command(self):
        """Test that commands without required data raise appropriate errors"""
        command = "Cancel my booking"  # Missing booking ID
        with self.assertRaises(ValueError) as context:
            llm_processor.parse_user_input(command)
        self.assertIn("Booking ID is required", str(context.exception))

if __name__ == '__main__':
    unittest.main()