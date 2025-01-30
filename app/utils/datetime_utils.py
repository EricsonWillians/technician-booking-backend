"""
==========================================================
                 DATETIME UTILITIES MODULE
==========================================================

A supremely precise datetime interpretation system for the 
Technician Booking System. This module provides sophisticated 
natural language processing capabilities with Swiss-watch 
precision, handling complex temporal expressions while 
enforcing strict business rules.

Key Capabilities:
    • Precise weekday calculations
    • Relative date interpretations
    • Business hours enforcement
    • Robust validation
    • Comprehensive logging

  Author : Ericson Willians  
  Email  : ericsonwillians@protonmail.com  
  Date   : January 2025  

==========================================================
"""

import re
import logging
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Union, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from transformers import Pipeline
from dateutil.parser import parse

from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


class DateTimeExtractionError(Exception):
    """Base exception for datetime extraction errors."""
    pass


class InvalidTimeFormatError(DateTimeExtractionError):
    """Raised when time format doesn't match expected patterns."""
    pass


class BusinessHoursError(DateTimeExtractionError):
    """Raised when time falls outside business hours."""
    pass


class TimeReference(Enum):
    """
    Classification of temporal references in natural language.

    Attributes:
        THIS: References like "this Wednesday"
        NEXT: References like "next Wednesday"
        AFTER: References like "Wednesday after next"
        SPECIFIC: Explicit dates like "February 5th"
        RELATIVE: Relative dates like "tomorrow"
    """
    THIS = auto()
    NEXT = auto()
    AFTER = auto()
    SPECIFIC = auto()
    RELATIVE = auto()


@dataclass(frozen=True)
class TimePoint:
    """
    Immutable representation of a specific point in time with metadata.

    Attributes:
        year (int): The year component
        month (int): The month component (1-12)
        day (int): The day component (1-31)
        hour (int): The hour component (0-23)
        minute (int): The minute component (0-59)
        reference_type (TimeReference): How this time was referenced
        timezone (str): Timezone of the time point
        confidence (float): Confidence score of the interpretation
    """
    year: int
    month: int
    day: int
    hour: int
    minute: int
    reference_type: TimeReference
    timezone: str = field(default='UTC', metadata={"description": "Timezone of the time point"})
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate all components after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Ensure all time components are within acceptable ranges.

        Raises:
            ValueError: If any component is out of range
        """
        if not (2024 <= self.year <= 2030):
            raise ValueError(f"Year {self.year} out of reasonable range")
        if not (1 <= self.month <= 12):
            raise ValueError(f"Month {self.month} invalid")
        if not (1 <= self.day <= 31):
            raise ValueError(f"Day {self.day} invalid")
        if not (0 <= self.hour <= 23):
            raise ValueError(f"Hour {self.hour} invalid")
        if not (0 <= self.minute <= 59):
            raise ValueError(f"Minute {self.minute} invalid")
        # Validate timezone
        try:
            ZoneInfo(self.timezone)
        except Exception:
            raise ValueError(f"Invalid timezone: {self.timezone}")

    def to_datetime(self) -> datetime:
        """Convert to timezone-aware datetime object with validation."""
        return datetime(
            self.year, self.month, self.day,
            self.hour, self.minute, 0, 0,
            tzinfo=ZoneInfo(self.timezone)
        )


class BusinessHours:
    """
    Business hours configuration and validation.

    This class manages the definition and enforcement of business hours,
    ensuring all appointments fall within acceptable time ranges.
    """
    OPEN_HOUR: int = 9
    CLOSE_HOUR: int = 17
    DEFAULT_DURATION: timedelta = timedelta(hours=1)

    @classmethod
    def is_within_hours(cls, dt: datetime) -> bool:
        """
        Check if time falls within business hours.

        Args:
            dt: Datetime to validate

        Returns:
            bool: True if within business hours
        """
        return cls.OPEN_HOUR <= dt.hour < cls.CLOSE_HOUR

    @classmethod
    def adjust_to_business_hours(cls, dt: datetime, strict: bool = True) -> datetime:
        """
        Adjust datetime to fall within business hours.

        Args:
            dt: Datetime to adjust
            strict: If True, raises error instead of adjusting

        Returns:
            datetime: Adjusted datetime

        Raises:
            BusinessHoursError: If strict=True and time is outside hours
        """
        if cls.is_within_hours(dt):
            return dt

        if strict:
            raise BusinessHoursError(f"Time {dt} is outside business hours")

        if dt.hour < cls.OPEN_HOUR:
            return dt.replace(hour=cls.OPEN_HOUR, minute=0)

        # Move to next day opening time
        next_day = dt + timedelta(days=1)
        return next_day.replace(hour=cls.OPEN_HOUR, minute=0)


class DateTimeExtractor:
    """
    Supremely precise datetime extraction from natural language.

    This class implements sophisticated datetime interpretation with
    extreme attention to detail and robust error handling.
    """

    WEEKDAYS = {
        'monday': 0, 'mon': 0,
        'tuesday': 1, 'tue': 1,
        'wednesday': 2, 'wed': 2,
        'thursday': 3, 'thu': 3,
        'friday': 4, 'fri': 4,
        'saturday': 5, 'sat': 5,
        'sunday': 6, 'sun': 6
    }

    TIME_DEFAULTS = {
        'morning': time(9, 0),
        'afternoon': time(14, 0),
        'evening': time(17, 0),
    }

    def __init__(self, text_interpreter: Pipeline):
        """
        Initialize with required LLM pipeline.

        Args:
            text_interpreter: Transform pipeline for text interpretation
        """
        self.timezone = settings.TIMEZONE
        self.current_time = datetime.now(ZoneInfo(self.timezone)).replace(microsecond=0)
        self.current_weekday = self.current_time.weekday()
        self.text_interpreter = text_interpreter
        logger.debug(f"Initialized DateTimeExtractor with reference time: {self.current_time} ({self.timezone})")

    def _calculate_next_occurrence(self, weekday_target: int) -> datetime:
        """
        Calculate the next occurrence of the specified weekday.

        Args:
            weekday_target (int): Target weekday as an integer (Monday=0, Sunday=6).

        Returns:
            datetime: The datetime of the next occurrence of the target weekday.
        """
        now = self.current_time
        current_weekday = now.weekday()
        days_ahead = (weekday_target - current_weekday + 7) % 7
        if days_ahead == 0:
            days_ahead = 7  # Ensure it's the next occurrence, not today
        next_occurrence = now + timedelta(days=days_ahead)
        logger.debug(f"Next occurrence of weekday {weekday_target} is on {next_occurrence.strftime('%Y-%m-%d')}")
        return next_occurrence

    def _generate_prompt(self, text: str) -> str:
        """Generate a supreme prompt for precise datetime interpretation."""
        reference_time = self.current_time.replace(microsecond=0)
        current_weekday = reference_time.strftime('%A')  # e.g., 'Monday'
        target_wednesday = self._calculate_next_occurrence(2)  # Wednesday = 2
        next_monday = self._calculate_next_occurrence(0)  # Monday = 0

        prompt = f"""You are an exceptionally accurate datetime interpreter for the Technician Booking System. Your task is to convert natural language time expressions into precise datetime values strictly adhering to the specified format.

    **Reference Information:**
    - **Current Time:** {reference_time.strftime('%Y-%m-%d %H:%M')} ({self.timezone})
    - **Current Day:** {current_weekday}
    - **Next Wednesday:** {target_wednesday.strftime('%Y-%m-%d')}
    - **Next Monday:** {next_monday.strftime('%Y-%m-%d')}

    **Output Requirements:**
    1. **Format:** `DATE: YYYY-MM-DD | TIME: HH:MM`
    2. **Time Format:** 24-hour with leading zeros (e.g., `14:00` for 2 PM).
    3. **No AM/PM Indicators:** Only numeric time representations.
    4. **Future Date:** The date must be in the future relative to the Current Time.
    5. **Timezone Consistency:** Assume all times are in the `{self.timezone}` timezone.

    **Examples:**
    - **Input:** "tomorrow at 2 PM"
    - **Output:** `DATE: { (reference_time + timedelta(days=1)).strftime('%Y-%m-%d') } | TIME: 14:00`

    - **Input:** "this Wednesday at 3:30 PM"
    - **Output:** `DATE: { target_wednesday.strftime('%Y-%m-%d') } | TIME: 15:30`

    - **Input:** "next Monday morning"
    - **Output:** `DATE: { next_monday.strftime('%Y-%m-%d') } | TIME: 09:00`

    - **Input:** "in two weeks at 10 AM"
    - **Output:** `DATE: { (self.current_time + timedelta(weeks=2)).strftime('%Y-%m-%d') } | TIME: 10:00`

    - **Input:** "Friday evening at 6:45 PM"
    - **Output:** `DATE: { self._calculate_next_occurrence(4).strftime('%Y-%m-%d') } | TIME: 18:45`

    **Task:**
    Analyze the following input and provide the output strictly adhering to the requirements above.

    **Input:** {text}

    **Output:**"""

        return prompt.strip()

    def _validate_llm_output(self, output: str) -> bool:
        """
        Validates the LLM output against the expected format.

        Returns:
            bool: True if valid, False otherwise.
        """
        pattern = r'^DATE:\s\d{4}-\d{2}-\d{2}\s\|\sTIME:\s\d{2}:\d{2}$'
        return bool(re.match(pattern, output))

    def _parse_llm_output(self, output: str, original_text: str) -> TimePoint:
        """Parse the LLM output with strict validation."""
        try:
            if not self._validate_llm_output(output):
                raise DateTimeExtractionError(f"LLM output does not match the required format: '{output}'")

            date_part, time_part = map(str.strip, output.split('|'))

            date_match = re.match(r'^DATE:\s*(\d{4})-(\d{2})-(\d{2})$', date_part)
            time_match = re.match(r'^TIME:\s*(\d{2}):(\d{2})$', time_part)

            if not date_match:
                raise DateTimeExtractionError(f"Invalid date format in output: '{date_part}'")
            if not time_match:
                raise DateTimeExtractionError(f"Invalid time format in output: '{time_part}'")

            year, month, day = map(int, date_match.groups())
            hour, minute = map(int, time_match.groups())

            # Additional validation can be added here if necessary

            time_point = TimePoint(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                reference_type=TimeReference.SPECIFIC,
                timezone=self.timezone
            )

            logger.debug(f"Successfully parsed datetime: {time_point.to_datetime().strftime('%Y-%m-%d %H:%M %Z')}")
            return time_point

        except Exception as e:
            logger.error(f"Failed to parse LLM output '{output}' for input '{original_text}': {e}")
            raise DateTimeExtractionError(f"Failed to parse datetime: {e}")

    def _fallback_datetime_extraction(self, text: str) -> datetime:
        """
        Alternative method to extract datetime using dateutil if LLM fails.
        """
        try:
            parsed_dt = parse(text, fuzzy=True, default=self.current_time)
            if parsed_dt <= self.current_time:
                parsed_dt += timedelta(days=1)
            adjusted_dt = BusinessHours.adjust_to_business_hours(parsed_dt, strict=False)
            return adjusted_dt
        except Exception as e:
            raise DateTimeExtractionError(f"Fallback parsing failed: {e}")

    def extract_datetime(self, text: str, max_retries: int = 3) -> datetime:
        """Extract datetime with supreme precision."""
        for attempt in range(max_retries):
            try:
                prompt = self._generate_prompt(text)
                logger.debug(f"Generated prompt: {prompt}")
                
                interpretation = self.text_interpreter(prompt)[0]["generated_text"].strip()
                logger.debug(f"LLM raw interpretation: {interpretation}")

                if not self._validate_llm_output(interpretation):
                    raise DateTimeExtractionError(f"LLM output does not match the required format: '{interpretation}'")

                time_point = self._parse_llm_output(interpretation, text)
                result = time_point.to_datetime()

                # Validation 1: Future date
                if result <= self.current_time:
                    logger.info(f"Adjusting past date: {result}")
                    days_to_add = 7 if result.date() == self.current_time.date() else 1
                    result += timedelta(days=days_to_add)

                # Validation 2: Business hours (9:00-17:00)
                if not BusinessHours.is_within_hours(result):
                    logger.info(f"Adjusting to business hours: {result}")
                    result = BusinessHours.adjust_to_business_hours(result, strict=False)

                logger.info(f"Final datetime: {result.strftime('%Y-%m-%d %H:%M %Z')}")
                return result

            except DateTimeExtractionError as e:
                logger.warning(f"DateTime extraction failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying datetime extraction with adjusted prompt.")
                    continue
                else:
                    # Fallback to alternative parsing
                    try:
                        fallback_dt = self._fallback_datetime_extraction(text)
                        logger.info(f"Fallback datetime extraction successful: {fallback_dt}")
                        return fallback_dt
                    except Exception as fe:
                        logger.error(f"Fallback datetime extraction failed: {fe}")
                        raise DateTimeExtractionError(f"Failed to extract datetime after {max_retries} attempts: {e} | Fallback failed: {fe}")


def create_datetime_extractor(text_interpreter: Pipeline) -> DateTimeExtractor:
    """Factory function to create a configured DateTimeExtractor."""
    return DateTimeExtractor(text_interpreter=text_interpreter)
