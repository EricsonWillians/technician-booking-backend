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
            self.business_hours = BusinessHours()
            logger.info(f"DateTimeExtractor initialized with timezone: {self.timezone}")
        except Exception as e:
            logger.error(f"Failed to set timezone {self.timezone}. Defaulting to UTC. Error: {e}")
            self.timezone = "UTC"
            self.timezone_obj = ZoneInfo("UTC")
            self.current_time = datetime.now(self.timezone_obj)

    def extract_datetime_entities(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Main entry point for datetime extraction."""
        try:
            # Try explicit time pattern first
            time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)'
            time_match = re.search(time_pattern, text.lower())
            
            base_date = self._extract_date_component(text)
            if not base_date:
                base_date = self.current_time + timedelta(days=1)
            
            if time_match:
                # Extract hour and minute from the match
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                meridiem = time_match.group(3)
                
                # Convert to 24-hour format
                if meridiem == 'pm' and hour < 12:
                    hour += 12
                elif meridiem == 'am' and hour == 12:
                    hour = 0
                
                # Combine date and time
                extracted_time = base_date.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0,
                    tzinfo=self.timezone_obj
                )
            else:
                # No explicit time found, use default booking hour
                extracted_time = base_date.replace(
                    hour=settings.DEFAULT_BOOKING_HOUR,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=self.timezone_obj
                )

            # Adjust to business hours if needed
            final_time = self.business_hours.adjust_to_business_hours(extracted_time)
            entities["start_time"] = final_time
            logger.info(f"Extracted and adjusted datetime: {final_time}")
            return entities

        except Exception as e:
            logger.error(f"Failed to extract datetime: {str(e)}")
            # Fallback to next business day
            entities["start_time"] = self.business_hours._next_business_day()
            return entities
        
    def _extract_date_component(self, text: str) -> Optional[datetime]:
        """Extract date component from text."""
        text_lower = text.lower()
        
        # Check for weekdays
        for weekday, day_num in self.WEEKDAYS.items():
            if weekday in text_lower:
                return self._next_weekday(self.current_time, day_num)
                
        # Try fuzzy parsing as last resort
        try:
            parsed_date = parser.parse(text, fuzzy=True, default=self.current_time)
            if parsed_date < self.current_time:
                return self.current_time + timedelta(days=1)
            return parsed_date
        except:
            return None

    def _extract_relative_datetime(self, text: str) -> Optional[datetime]:
        """Extract datetime from relative expressions with preserved time specifications."""
        text_lower = text.lower()
        
        # First try to extract any explicit time
        explicit_time = None
        time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)'
        time_match = re.search(time_pattern, text_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            meridiem = time_match.group(3)
            
            # Convert to 24-hour format
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
                
            explicit_time = time(hour=hour, minute=minute)

        # Check for weekdays
        for weekday, day_num in self.WEEKDAYS.items():
            if weekday in text_lower:
                target_date = self._next_weekday(self.current_time, day_num)
                if explicit_time:
                    return target_date.replace(
                        hour=explicit_time.hour,
                        minute=explicit_time.minute,
                        second=0,
                        microsecond=0
                    )
                else:
                    # Only use default hour if no time was specified
                    return target_date.replace(
                        hour=settings.DEFAULT_BOOKING_HOUR,
                        minute=0,
                        second=0,
                        microsecond=0
                    )

        # Check for relative days
        for rel_day, days_ahead in self.RELATIVE_DAYS.items():
            if rel_day in text_lower:
                base_date = self.current_time + timedelta(days=days_ahead)
                
                if explicit_time:
                    return base_date.replace(
                        hour=explicit_time.hour,
                        minute=explicit_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
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

        return None

    def _next_weekday(self, ref_date: datetime, weekday: int) -> datetime:
        """Get the next occurrence of a weekday."""
        days_ahead = weekday - ref_date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_date = ref_date + timedelta(days=days_ahead)
        return next_date

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
    """Business hours handling with proper instance methods."""
    
    def __init__(self):
        self.open_hour = settings.DEFAULT_BOOKING_HOUR
        self.close_hour = settings.LAST_BOOKING_HOUR
        self.timezone = settings.TIMEZONE or "UTC"
        self.timezone_obj = ZoneInfo(self.timezone)

    def is_within_hours(self, dt: datetime) -> bool:
        """Check if time is within business hours."""
        return self.open_hour <= dt.hour < self.close_hour

    def adjust_to_business_hours(self, dt: datetime) -> datetime:
        """Adjust datetime to valid business hours while preserving user-specified times when possible."""
        if not dt:
            return self._next_business_day()

        # Ensure datetime is timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone_obj)

        # If within business hours, respect the original time
        if self.is_within_hours(dt):
            return dt
            
        # If before business hours on the same day
        if dt.hour < self.open_hour:
            logger.warning(f"Requested time {dt.strftime('%I:%M %p')} is before business hours, adjusting to opening time")
            return dt.replace(hour=self.open_hour, minute=0, second=0, microsecond=0)
        
        # If after business hours, move to next day
        next_day = dt + timedelta(days=1)
        logger.warning(f"Requested time {dt.strftime('%I:%M %p')} is after business hours, moving to next day")
        return next_day.replace(hour=self.open_hour, minute=0, second=0, microsecond=0)

    def _next_business_day(self) -> datetime:
        """Get next business day starting time."""
        now = datetime.now(self.timezone_obj)
        next_day = now + timedelta(days=1)
        return next_day.replace(
            hour=self.open_hour,
            minute=0,
            second=0,
            microsecond=0
        )