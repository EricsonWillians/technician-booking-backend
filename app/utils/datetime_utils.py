# app/utils/datetime_utils.py

import re
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
from dateutil import parser
from dateutil.relativedelta import relativedelta

from app.config.settings import settings

logger = logging.getLogger(__name__)

class DateTimeExtractionError(Exception):
    """Base exception for datetime extraction errors."""
    pass

class DateTimeExtractor:
    """Enhanced datetime extraction with better relative date handling."""
    
    WEEKDAYS = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6
    }

    RELATIVE_DAYS = {
        "today": 0,
        "tonight": 0,
        "tomorrow": 1,
        "tmr": 1,
        "tmrw": 1,
        "day after tomorrow": 2,
        "next day": 1,
        "coming": 1,
        "following": 1,
    }

    TIME_PERIODS = {
        "morning": (9, 0),
        "afternoon": (14, 0),
        "evening": (17, 0),
        "night": (20, 0),
        "noon": (12, 0),
        "midnight": (0, 0),
        "midday": (12, 0),
    }

    RELATIVE_TIME_PATTERNS = [
        (r'in\s+(\d+)\s+hour(?:s)?', lambda m: timedelta(hours=int(m.group(1)))),
        (r'in\s+(\d+)\s+minute(?:s)?', lambda m: timedelta(minutes=int(m.group(1)))),
        (r'in\s+(\d+)\s+day(?:s)?', lambda m: timedelta(days=int(m.group(1)))),
        (r'after\s+(\d+)\s+hour(?:s)?', lambda m: timedelta(hours=int(m.group(1)))),
        (r'next\s+week', lambda m: timedelta(weeks=1)),
        (r'next\s+month', lambda m: relativedelta(months=1)),
    ]

    def __init__(self):
        """Initialize with timezone and current time."""
        self.timezone = settings.TIMEZONE or "UTC"
        try:
            self.timezone_obj = ZoneInfo(self.timezone)
            self.current_time = datetime.now(self.timezone_obj)
            logger.info(f"DateTimeExtractor initialized with timezone: {self.timezone}")
        except Exception as e:
            logger.error(f"Failed to set timezone {self.timezone}. Defaulting to UTC. Error: {e}")
            self.timezone = "UTC"
            self.timezone_obj = ZoneInfo("UTC")
            self.current_time = datetime.now(self.timezone_obj)

    def extract_datetime_entities(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Enhanced datetime extraction with better relative date handling."""
        try:
            # Try explicit date/time first
            date_str = entities.get("date", "")
            time_str = entities.get("time", "")
            datetime_str = f"{date_str} {time_str}".strip()

            extracted_time = None
            if datetime_str:
                try:
                    extracted_time = self._parse_datetime(datetime_str)
                except DateTimeExtractionError:
                    pass

            # If no explicit datetime, try relative patterns
            if not extracted_time:
                extracted_time = self._extract_relative_datetime(text)

            # If still no datetime, try fuzzy parsing
            if not extracted_time:
                try:
                    extracted_time = self._fuzzy_parse_datetime(text)
                except DateTimeExtractionError:
                    extracted_time = self._default_booking_time()

            # Ensure timezone awareness and business hours
            if extracted_time and not extracted_time.tzinfo:
                extracted_time = extracted_time.replace(tzinfo=self.timezone_obj)

            extracted_time = BusinessHours.adjust_to_business_hours(extracted_time)
            
            entities["start_time"] = extracted_time
            logger.info(f"Final extracted datetime: {extracted_time}")
            return entities

        except Exception as e:
            logger.error(f"Failed to extract datetime: {e}")
            entities["start_time"] = self._default_booking_time()
            return entities

    def _extract_relative_datetime(self, text: str) -> Optional[datetime]:
        """Extract datetime from relative expressions."""
        text_lower = text.lower()
        
        # Check for relative days
        for rel_day, days_ahead in self.RELATIVE_DAYS.items():
            if rel_day in text_lower:
                base_date = self.current_time + timedelta(days=days_ahead)
                
                # Check for time period in the same phrase
                for period, (hour, minute) in self.TIME_PERIODS.items():
                    if period in text_lower:
                        return base_date.replace(
                            hour=hour,
                            minute=minute,
                            second=0,
                            microsecond=0
                        )
                
                # No time specified, use default booking hour
                return base_date.replace(
                    hour=settings.DEFAULT_BOOKING_HOUR,
                    minute=0,
                    second=0,
                    microsecond=0
                )

        # Check for weekdays
        for weekday, day_num in self.WEEKDAYS.items():
            if weekday in text_lower:
                target_date = self._next_weekday(self.current_time, day_num)
                return target_date.replace(
                    hour=settings.DEFAULT_BOOKING_HOUR,
                    minute=0,
                    second=0,
                    microsecond=0
                )

        # Check for relative time patterns
        for pattern, delta_func in self.RELATIVE_TIME_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                delta = delta_func(match)
                return self.current_time + delta

        return None

    def _next_weekday(self, ref_date: datetime, weekday: int) -> datetime:
        """Get the next occurrence of a weekday."""
        days_ahead = weekday - ref_date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return ref_date + timedelta(days=days_ahead)

    def _fuzzy_parse_datetime(self, text: str) -> datetime:
        """Fuzzy parse datetime with better handling of relative terms."""
        try:
            parsed = parser.parse(text, fuzzy=True, default=self.current_time)
            
            # Handle past dates
            if parsed < self.current_time:
                if any(rel in text.lower() for rel in self.RELATIVE_DAYS.keys()):
                    # Relative date reference, adjust forward
                    days_ahead = 1  # Default to tomorrow
                    for rel_day, days in self.RELATIVE_DAYS.items():
                        if rel_day in text.lower():
                            days_ahead = days
                            break
                    parsed = self.current_time + timedelta(days=days_ahead)
                
                elif parsed.date() == self.current_time.date():
                    # Same day but past time, try to interpret as next occurrence
                    if parsed.time() < self.current_time.time():
                        parsed = parsed + timedelta(days=1)

            return parsed
                
        except Exception as e:
            logger.error(f"Fuzzy parsing failed: {e}")
            raise DateTimeExtractionError(f"Could not parse datetime from: {text}")

    def _default_booking_time(self) -> datetime:
        """Get default booking time (next business day)."""
        next_day = self.current_time + timedelta(days=1)
        return next_day.replace(
            hour=settings.DEFAULT_BOOKING_HOUR,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=self.timezone_obj
        )

class BusinessHours:
    """Business hours handling with improved validation."""
    
    OPEN_HOUR = settings.DEFAULT_BOOKING_HOUR
    CLOSE_HOUR = settings.LAST_BOOKING_HOUR

    @classmethod
    def is_within_hours(cls, dt: datetime) -> bool:
        """Check if time is within business hours."""
        return cls.OPEN_HOUR <= dt.hour < cls.CLOSE_HOUR

    @classmethod
    def adjust_to_business_hours(cls, dt: datetime) -> datetime:
        """Adjust datetime to valid business hours."""
        if not dt:
            return cls._next_business_day()

        if cls.is_within_hours(dt):
            return dt

        if dt.hour < cls.OPEN_HOUR:
            return dt.replace(hour=cls.OPEN_HOUR, minute=0, second=0, microsecond=0)
        
        # After business hours, schedule for next day
        next_day = dt + timedelta(days=1)
        return next_day.replace(hour=cls.OPEN_HOUR, minute=0, second=0, microsecond=0)

    @classmethod
    def _next_business_day(cls) -> datetime:
        """Get next business day with proper timezone."""
        now = datetime.now(ZoneInfo(settings.TIMEZONE or "UTC"))
        next_day = now + timedelta(days=1)
        return next_day.replace(
            hour=cls.OPEN_HOUR,
            minute=0,
            second=0,
            microsecond=0
        )