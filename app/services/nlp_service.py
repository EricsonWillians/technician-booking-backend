"""
==========================================================
                  NLP SERVICE MODULE
==========================================================

  A professional-grade multi-pipeline NLP service using 
  Hugging Face Transformers.

  Features:
  - Zero-shot classification for user intent detection
  - Named Entity Recognition (NER) for extracting names 
    (technician, customer) and partial time expressions
  - Text-to-text LLM for advanced date/time interpretation

  Author : Ericson Willians  
  Email  : ericsonwillians@protonmail.com  
  Date   : January 2025  

==========================================================
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import dateutil
from dateutil.parser import parse
from transformers import pipeline, Pipeline
from dataclasses import dataclass
from pydantic import BaseModel, Field

from app.config.settings import settings

from app.utils.datetime_utils import (
    create_datetime_extractor,
    DateTimeExtractionError,
    BusinessHours
)

from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class PipelineException(Exception):
    """Custom exception for pipeline initialization failures."""

    pass


class PipelineInitError(Exception):
    """Raised when a pipeline fails to initialize properly."""

    pass


class TextInterpretationError(Exception):
    """Raised when the LLM fails to interpret text correctly."""

    pass


class EntityExtractionError(Exception):
    """Raised when entity extraction fails critically."""

    pass


class ExtractedNames(BaseModel):
    """
    Data class for storing extracted name entities.

    Attributes:
        customer_name (Optional[str]): The identified customer's name, if found
        technician_name (Optional[str]): The identified technician's name, if found
    """

    customer_name: Optional[str] = Field(None, description="Customer's full name")
    technician_name: Optional[str] = Field(None, description="Technician's full name")


@dataclass(frozen=True)
class ParsedCommand:
    """
    Immutable container for parsed user input.

    Attributes:
        intent (str): The identified intent (e.g., "create_booking")
        data (Dict[str, Any]): Key/value data extracted from user input
    """

    intent: str
    data: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.intent:
            raise ValueError("Intent cannot be empty")
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")


class NLPService:
    """
    Multi-pipeline NLP service for booking system text interpretation.

    This service orchestrates multiple machine learning models to understand and
    extract structured information from natural language booking requests. It combines:
        1. Zero-shot classification for intent detection
        2. Named Entity Recognition for profession identification
        3. Large Language Model for complex text interpretation

    Attributes:
        intent_classifier (Pipeline): Zero-shot classification pipeline
        ner_pipeline (Pipeline): Named Entity Recognition pipeline
        text_interpreter (Pipeline): Text interpretation LLM pipeline
        device (torch.device): The device (CPU/GPU) used for inference
        device_name (str): Human-readable device name

    Example:
        >>> service = NLPService()
        >>> text = "I'm John Doe and I need plumber Mike Johnson tomorrow at 2 PM"
        >>> entities = service._extract_entities(text)
        >>> print(entities)
        {
            'customer_name': 'John Doe',
            'technician_name': 'Mike Johnson',
            'profession': 'plumber',
            'start_time': datetime(2024, 1, 31, 14, 0)
        }
    """
    
    INTENT_LABEL_MAPPING = {
        "Create a booking": "create_booking",
        "Cancel a booking": "cancel_booking",
        "Retrieve booking details": "retrieve_booking",
        "List all bookings": "list_bookings",
    }

    def __init__(self) -> None:
        """
        Initialize the NLP service with required ML pipelines.

        The initialization process includes:
        1. Validating configuration settings
        2. Optimizing hardware selection (CPU/GPU)
        3. Initializing all required ML pipelines

        Raises:
            PipelineInitError: If any pipeline fails to initialize
            ValueError: If critical settings are invalid
        """
        self._validate_settings()
        self.device, self.device_name = self._optimize_hardware()

        try:
            # Initialize core pipelines
            self.intent_classifier = self._init_pipeline_with_retry(
                "zero-shot-classification",
                settings.ZERO_SHOT_MODEL_NAME,
                {"multi_label": False},
            )

            self.ner_pipeline = self._init_pipeline_with_retry(
                "token-classification",
                settings.NER_MODEL_NAME,
                {"aggregation_strategy": "simple"},
            )

            # Initialize text interpretation LLM
            self.text_interpreter = self._init_pipeline_with_retry(
                "text2text-generation",
                settings.TEXT_TO_TEXT_MODEL_NAME,
                {"max_length": 128, "do_sample": False},
            )

        except Exception as e:
            raise PipelineInitError(f"Failed to initialize NLP pipelines: {e}")

        # Initialize datetime extractor
        self.datetime_extractor = create_datetime_extractor(self.text_interpreter)

    def _interpret_text(self, prompt: str) -> str:
        """
        Generic text interpretation method using the LLM.

        Args:
            prompt (str): The formatted prompt for the LLM

        Returns:
            str: The LLM's interpretation result

        Raises:
            ValueError: If no LLM is configured or the result is invalid
        """
        if not self.text_interpreter:
            raise ValueError("No text interpretation LLM configured")

        result = self.text_interpreter(prompt)
        if not result or not isinstance(result, list):
            raise ValueError(f"Invalid LLM interpretation result")

        return result[0]["generated_text"].strip()

    # -------------------------------------------------------------------------
    # Initialization & Configuration
    # -------------------------------------------------------------------------
    def _validate_settings(self) -> None:
        """
        Validate critical configuration settings.

        Checks:
            - Required model names are configured
            - Intent confidence threshold is valid
            - Candidate intents list is not empty

        Raises:
            ValueError: If any validation fails
        """
        if not settings.CANDIDATE_INTENTS:
            raise ValueError("CANDIDATE_INTENTS must be non-empty list")

        if not (0 <= settings.INTENT_CONFIDENCE_THRESHOLD <= 1):
            raise ValueError("INTENT_CONFIDENCE_THRESHOLD must be between 0 and 1")

        required_models = {
            "ZERO_SHOT_MODEL_NAME": settings.ZERO_SHOT_MODEL_NAME,
            "NER_MODEL_NAME": settings.NER_MODEL_NAME,
            "TEXT_TO_TEXT_MODEL_NAME": settings.TEXT_TO_TEXT_MODEL_NAME,
        }

        missing = [k for k, v in required_models.items() if not v]
        if missing:
            raise ValueError(f"Missing required model names: {', '.join(missing)}")

    def _optimize_hardware(self) -> Tuple[torch.device, str]:
        """
        Select optimal hardware configuration for inference.

        Returns:
            Tuple[torch.device, str]: Device object and human-readable name

        Note:
            Enables GPU if available and configured, with optional half-precision
        """
        if torch.cuda.is_available() and settings.USE_GPU:
            device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)

            if settings.USE_HALF_PRECISION:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled half-precision inference")
        else:
            device = torch.device("cpu")
            device_name = "CPU"

        logger.info(f"Using device: {device_name}")
        return device, device_name

    def _model_cache_status(self, model_name: str) -> Tuple[bool, str]:
        """Check if model is locally cached."""
        cache_root = os.path.expanduser(settings.HF_CACHE_DIR)
        model_dir = f"models--{model_name.replace('/', '--')}"
        cache_path = os.path.join(cache_root, model_dir)
        return os.path.exists(cache_path), cache_path

    def _init_pipeline_with_retry(
        self, task: str, model_name: str, task_args: Dict[str, Any]
    ) -> Pipeline:
        """
        Create a pipeline with robust retries. If cache fails, attempt fresh download.
        """
        if not model_name:
            logger.warning(
                f"No model name provided for task={task}; skipping pipeline init."
            )
            return None

        is_cached, cache_path = self._model_cache_status(model_name)
        logger.info(f"Initializing {task} pipeline (Cached: {is_cached})...")

        for attempt in range(settings.MODEL_LOAD_RETRIES):
            try:
                return pipeline(
                    task=task,
                    model=model_name,
                    device=self.device,
                    model_kwargs={"local_files_only": is_cached},
                    **task_args,
                )
            except (OSError, PipelineException) as e:
                if is_cached and attempt == 0:
                    logger.warning(f"Model load failed from cache {cache_path}: {e}")
                    is_cached = False
                    continue
                logger.error(f"Pipeline init failed (attempt {attempt+1}): {e}")
                if attempt == settings.MODEL_LOAD_RETRIES - 1:
                    raise RuntimeError(
                        f"Failed to init {task} after {settings.MODEL_LOAD_RETRIES} tries"
                    ) from e

        # Should never reach here
        raise RuntimeError("Unexpected pipeline initialization error.")

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    def parse_user_input(self, text: str) -> ParsedCommand:
        """
        Process natural language user input into structured command.
        
        Args:
            text: Raw user input text
            
        Returns:
            ParsedCommand with intent and extracted data
            
        Raises:
            ValueError: If input is invalid or intent cannot be determined
        """
        self._validate_input(text)
        cleaned_text = self._preprocess_text(text)

        try:
            # Determine intent
            intent_label, confidence = self._classify_intent(cleaned_text)
            internal_intent = self.INTENT_LABEL_MAPPING.get(intent_label)
            if not internal_intent:
                raise ValueError(f"Unknown intent: {intent_label}")

            # Extract entities
            entities = self._extract_entities(cleaned_text)
            entities["confidence"] = confidence

            # Validate required fields
            self._validate_intent_requirements(internal_intent, entities)

            return ParsedCommand(
                intent=internal_intent,
                data=entities
            )
        except Exception as e:
            logger.error(f"Failed to parse input '{text}': {e}")
            raise

    # -------------------------------------------------------------------------
    # Input Validation & Preprocessing
    # -------------------------------------------------------------------------
    def _validate_input(self, text: str) -> None:
        """Raise if text is empty, too short, too long, or not a string."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty or whitespace only")

        if len(cleaned_text) < settings.MIN_INPUT_LENGTH:
            raise ValueError(
                f"Input must be at least {settings.MIN_INPUT_LENGTH} characters"
            )

        if len(cleaned_text) > settings.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input exceeds max length of {settings.MAX_INPUT_LENGTH} characters"
            )

        if not any(c.isalnum() for c in cleaned_text):
            raise ValueError("Input must contain at least one alphanumeric character")

    def _preprocess_text(self, text: str) -> str:
        """Remove extra spacing, clip at max length."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text[: settings.MAX_INPUT_LENGTH]

    # -------------------------------------------------------------------------
    # Intent Classification
    # -------------------------------------------------------------------------
    def _classify_intent(self, text: str) -> Tuple[str, float]:
        """
        1) Attempt explicit regex patterns
        2) Otherwise fallback to zero-shot classification
        """
        text_lower = text.lower()
        # Very simple rules to short-circuit
        intent_patterns = {
            "Cancel a booking": [
                # e.g., "I want to cancel my booking", "Cancel booking", "Please remove my reservation"
                r"(?i)\b(?:cancel|remove|delete|terminate|abort|discard|void)\s+(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b",
                # e.g., "I need to undo my booking", "Revoke reservation"
                r"(?i)\b(?:undo|revoke)\s+(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b",
                # e.g., "Cancel my booking for tomorrow", "Delete booking on Friday"
                r"(?i)\b(?:cancel|remove|delete|terminate|abort|discard|void)\s+(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\s+(?:for\s+.+)$",
            ],
            "List all bookings": [
                # e.g., "List my bookings", "Show all reservations", "Display my appointments"
                r"(?i)\b(?:list|show|display|view|see|get|fetch|present|give\s+me)\s+(?:all\s+)?(?:my\s+)?(?:bookings|reservations|appointments|schedules|sessions)\b",
                # e.g., "Can you list my bookings?", "Please show all reservations"
                r"(?i)\b(?:can\s+you\s+|please\s+)?(?:list|show|display|view|see|get|fetch|present|give\s+me)\s+(?:all\s+)?(?:my\s+)?(?:bookings|reservations|appointments|schedules|sessions)\b",
            ],
            "Retrieve booking details": [
                # e.g., "Get details of my booking", "Find my reservation", "Retrieve appointment information"
                r"(?i)\b(?:get|find|retrieve|access|look\s+up)\s+(?:the\s+)?(?:details\s+of\s+)?(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b",
                # e.g., "Can you get booking details for me?", "Retrieve my reservation information"
                r"(?i)\b(?:can\s+you\s+)?(?:get|find|retrieve|access|look\s+up)\s+(?:the\s+)?(?:details\s+of\s+)?(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b",
            ],
            "Create a booking": [
                # e.g., "I want to book a plumber", "Schedule an electrician", "Need a technician"
                r"(?i)\b(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\b",
                # e.g., "Can you book a plumber for me?", "Please schedule an electrician"
                r"(?i)\b(?:can\s+you\s+|please\s+)?(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\b",
                # e.g., "I need to book a technician tomorrow", "Schedule a plumber on Monday"
                r"(?i)\b(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\s+(?:for\s+.+)$",
            ],
        }
        for intent_label, patterns in intent_patterns.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    logger.debug(f"Pattern matched intent: {intent_label}")
                    return intent_label, 0.95

        # Fall back to the zero-shot pipeline
        if self.intent_classifier is None:
            raise ValueError("No zero-shot pipeline configured to classify intent.")

        classification = self.intent_classifier(
            text, candidate_labels=settings.CANDIDATE_INTENTS, multi_label=False
        )
        top_label = classification["labels"][0]
        top_conf = classification["scores"][0]

        if top_conf < settings.INTENT_CONFIDENCE_THRESHOLD:
            raise ValueError(f"Intent confidence too low: {top_conf}")

        logger.debug(f"Zero-shot predicted '{top_label}' ({top_conf:.2f})")
        return top_label, top_conf

    def _validate_intent_requirements(self, intent: str, data: Dict[str, Any]) -> None:
        """Ensure required fields exist for each intent."""
        missing_booking_id = (
            intent in ["cancel_booking", "retrieve_booking"]
            and "booking_id" not in data
        )
        if missing_booking_id:
            raise ValueError(f"Booking ID is required for {intent}")

        missing_profession = (intent == "create_booking") and ("profession" not in data)
        if missing_profession:
            raise ValueError("Profession is required for booking creation")

    def _interpret_text(self, prompt: str) -> str:
        """
        Process a prompt using the text interpretation LLM.

        This is the core method for extracting structured information from
        natural language text. It provides a consistent interface for all
        LLM-based text interpretation tasks.

        Args:
            prompt (str): Formatted prompt for the LLM

        Returns:
            str: The LLM's interpretation result

        Raises:
            TextInterpretationError: If interpretation fails
            ValueError: If no LLM is configured
        """
        if not self.text_interpreter:
            raise ValueError("No text interpretation LLM configured")

        try:
            result = self.text_interpreter(prompt)
            if not result or not isinstance(result, list):
                raise TextInterpretationError("Invalid LLM output format")

            return result[0]["generated_text"].strip()
        except Exception as e:
            raise TextInterpretationError(f"Text interpretation failed: {e}")

    def _extract_names(self, text: str) -> ExtractedNames:
        """
        Extract customer and technician names using the text interpreter.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            ExtractedNames: Dataclass containing extracted names
        """
        prompt = f"""
    Extract the customer and technician names from this booking request. 
    If a name isn't found, output 'None'.

    Example Input: "I'm John Smith and I need electrician Mike Brown"
    Example Output: CUSTOMER: John Smith | TECHNICIAN: Mike Brown

    Example Input: "Book a plumber for tomorrow"
    Example Output: CUSTOMER: None | TECHNICIAN: None

    Input: {text}
    Output:""".strip()

        try:
            interpretation = self._interpret_text(prompt)
            customer_part, technician_part = interpretation.split("|")
            
            # Clean up the extracted names
            customer_name = customer_part.split("CUSTOMER:")[1].strip()
            technician_name = technician_part.split("TECHNICIAN:")[1].strip()
            
            return ExtractedNames(
                customer_name=None if customer_name == "None" else customer_name,
                technician_name=None if technician_name == "None" else technician_name
            )
        except Exception as e:
            logger.error(f"Name extraction failed: {e}")
            return ExtractedNames()

    def _process_ner_results(self, ner_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process named entity recognition results to extract names and roles.
        Combines direct text analysis with NER results for robust entity extraction.
        """
        extracted = {}
        text_markers = []
        text_lower = " ".join(word.lower() for item in ner_results for word in item["word"].split())
        
        # Step 1: First pass to collect person entities
        for item in ner_results:
            if item["entity_group"] == "PER":
                # Check for self-introduction at the start
                is_self_intro = item["start"] <= 5
                if is_self_intro:
                    extracted["customer_name"] = item["word"]
                else:
                    text_markers.append({
                        "type": "person",
                        "word": item["word"],
                        "position": item["start"]
                    })
        
        # Step 2: Direct profession detection from text using settings keywords
        for profession, keywords in settings.PROFESSION_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                extracted["profession"] = profession
                # Find position of matched keyword
                keyword_pos = next(
                    text_lower.find(keyword) 
                    for keyword in keywords 
                    if keyword in text_lower
                )
                
                # Look for technician after profession mention
                for marker in text_markers:
                    if marker["position"] > keyword_pos:
                        extracted["technician_name"] = marker["word"]
                        break
                break
        
        # Step 3: If no technician was found but we have other names
        if "technician_name" not in extracted and text_markers:
            # If customer isn't set, first person is customer
            if "customer_name" not in extracted:
                extracted["customer_name"] = text_markers[0]["word"]
                # If there's a second person, they're the technician
                if len(text_markers) > 1:
                    extracted["technician_name"] = text_markers[1]["word"]
            else:
                # If customer is set, other person is technician
                extracted["technician_name"] = text_markers[0]["word"]
        
        logger.debug(f"Processed NER results: {extracted}")
        return extracted

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all booking-related entities from input text.
        Combines NER, profession detection, and datetime parsing.
        """
        entities: Dict[str, Any] = {}

        try:
            # Extract booking ID first
            booking_id = self._extract_booking_id(text)
            if booking_id:
                entities["booking_id"] = booking_id

            # Process NER results for names and professions
            if self.ner_pipeline:
                ner_results = self.ner_pipeline(text)
                extracted = self._process_ner_results(ner_results)
                entities.update(extracted)
                
                # Directly check text for profession if not found through NER
                if "profession" not in entities:
                    text_lower = text.lower()
                    for profession, keywords in settings.PROFESSION_KEYWORDS.items():
                        if any(keyword in text_lower for keyword in keywords):
                            entities["profession"] = profession
                            logger.debug(f"Found profession through keyword match: {profession}")
                            break

            # Extract datetime
            try:
                dt = self.datetime_extractor.extract_datetime(text)
                if dt:
                    entities["start_time"] = dt
                    entities["duration"] = timedelta(hours=1)
            except DateTimeExtractionError as e:
                logger.warning(f"DateTime extraction failed: {e}")
                next_hour = BusinessHours.adjust_to_business_hours(datetime.now())
                entities["start_time"] = next_hour
                entities["duration"] = timedelta(hours=1)

            logger.debug(f"Extracted entities: {entities}")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise ValueError(f"Failed to extract entities: {e}")

    def _extract_booking_id(self, text: str) -> Optional[str]:
        """
        Extract booking ID from the text.

        1) Try matching UUIDs (e.g. 'b56a726a-4ec2-4681-8fd4-b51ff2c19c19').
        2) Fall back to numeric patterns (e.g., 'booking 123').
        3) If still not found, check if user typed 'cancel ...', 'retrieve ...', etc.
        and forcibly parse any standalone number.
        """
        text_lower = text.lower()

        # 1) Match typical UUID
        uuid_pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        match_uuid = re.search(uuid_pattern, text, re.IGNORECASE)
        if match_uuid:
            return match_uuid.group(0)  # entire UUID

        # 2) Additional numeric patterns
        patterns = [
            r"booking\s*#?\s*(\d+)",
            r"id\s*#?\s*(\d+)",
            r"#\s*(\d+)",
            r"\bid\s*(\d+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1)

        # 3) If user said e.g. "retrieve booking 123", forcibly parse any standalone number
        if any(
            k in text_lower
            for k in ["cancel", "booking", "retrieve", "find", "details"]
        ):
            m2 = re.search(r"\b(\d+)\b", text)
            if m2:
                return m2.group(1)

        return None

    # -------------------------------------------------------------------------
    # Date/Time Parsing
    # -------------------------------------------------------------------------
    def _assign_time(self, time_str: str, entities: Dict[str, Any]) -> None:
        """
        Attempt to parse time_str via:
          1) LLM pipeline -> ISO string -> datetime
          2) Fallback to local _parse_time
        """
        if self.date_time_llm:
            try:
                iso_result = self._interpret_datetime_with_llm(time_str)
                dt_val = datetime.strptime(iso_result, "%Y-%m-%d %H:%M")
                if dt_val < datetime.now():
                    dt_val += timedelta(days=7)
                entities["start_time"] = dt_val
                return
            except Exception as e:
                logger.warning(f"LLM parse failed for '{time_str}': {e}")

        # fallback
        try:
            dt_val = self._parse_time(time_str)
            entities["start_time"] = dt_val
        except ValueError as ve:
            logger.debug(f"Fallback parse_time failed: {ve}")

    def _interpret_datetime_with_llm(self, date_expression: str) -> str:
        """
        text2text-generation call to convert user expression -> 'YYYY-MM-DD HH:MM'
        """
        if not self.date_time_llm:
            raise ValueError("No date_time_llm pipeline configured")

        prompt = f"""
Convert the following date/time expression into a future date/time in format YYYY-MM-DD HH:MM.
Expression: {date_expression}
""".strip()

        result = self.date_time_llm(prompt)
        if not result or not isinstance(result, list):
            raise ValueError(f"Invalid LLM result for '{date_expression}'")
        return result[0].get("generated_text", "").strip()

    def _parse_time(self, time_str: str) -> datetime:
        """
        Parse a partial time expression like "Friday at 3 PM" into a guaranteed *future* datetime:

        1) Identify if user typed a weekday (mon, fri...), then compute the next occurrence from 'now'.
        2) Otherwise rely on dateutil parsing, anchored to the current date/time.
        3) Force minutes/seconds to 0 if the user didn't specify them (i.e., "3 PM" => 15:00 not 15:27).
        4) If final_dt is still < now, add 1 day or raise an error as needed.
        """

        now = datetime.now()
        day_idx = self._extract_weekday(time_str)

        # Step A: Let dateutil parse hour/minute if present, fallback to now
        try:
            dt_approx = parse(time_str, fuzzy=True, default=now)
        except Exception as ex:
            raise ValueError(f"Could not parse time from '{time_str}': {ex}")

        # Step B: If a weekday is mentioned, override the date portion
        if day_idx is not None:
            future_day = self._next_weekday(now, day_idx)
            final_dt = future_day.replace(
                hour=dt_approx.hour,
                minute=dt_approx.minute,
                second=dt_approx.second,
                microsecond=0,
            )
        else:
            final_dt = dt_approx

        # Step C: If user typed "3 pm" or "2am" with no colon,
        #         forcibly set minute=0, second=0 to avoid "3:27 pm" or "2:58 am".
        #         We'll detect if there's no colon (:) or dot (.) in the expression.
        normalized = time_str.lower()
        if re.search(r"\b(\d{1,2})\s*[ap]\.?m\.?\b", normalized):
            # This means user typed something like "3 pm" (with NO :xx in it).
            if not re.search(r"[:.]\d{1,2}", normalized):
                # No mention of minutes => default minute=0
                final_dt = final_dt.replace(minute=0, second=0, microsecond=0)

        # Step D: If final_dt < now, we push it forward by a day (or 7 days for next occurrence).
        if final_dt < now:
            final_dt += timedelta(days=1)
            if final_dt < now:
                raise ValueError(f"Parsed time '{final_dt}' is still in the past.")

        logger.debug(f"_parse_time => '{time_str}' => {final_dt}")
        return final_dt

    def _extract_weekday(self, text: str) -> Optional[int]:
        """
        Return int for Monday=0..Sunday=6 if matched, else None.
        """
        mapping = {
            "mon": 0,
            "monday": 0,
            "tue": 1,
            "tues": 1,
            "tuesday": 1,
            "wed": 2,
            "weds": 2,
            "wednesday": 2,
            "thu": 3,
            "thur": 3,
            "thurs": 3,
            "thursday": 3,
            "fri": 4,
            "friday": 4,
            "sat": 5,
            "saturday": 5,
            "sun": 6,
            "sunday": 6,
        }
        lowered = text.lower()
        for k, idx in mapping.items():
            if re.search(rf"\b{k}\b", lowered):
                return idx
        return None

    def _next_weekday(self, now: datetime, weekday_target: int) -> datetime:
        """
        Return the next future occurrence of e.g. weekday=4 (Friday).
        If 'now' is that weekday, skip to next week.
        """
        current_wday = now.weekday()
        diff = weekday_target - current_wday
        if diff <= 0:
            diff += 7
        return now + timedelta(days=diff)

    # -------------------------------------------------------------------------
    # Data Enrichment
    # -------------------------------------------------------------------------
    def _enrich_data(
        self, intent: str, entities: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """Add defaults (profession/time) for 'create_booking' or other logic."""
        enriched = dict(entities)  # shallow copy
        enriched.setdefault("timestamp", datetime.now().isoformat())
        enriched.setdefault("source", settings.DATA_SOURCE)

        if intent == "create_booking":
            # Possibly fill with default time if none
            if "start_time" not in enriched:
                enriched["start_time"] = self._default_booking_time()

        return enriched

    def _ensure_profession(self, data: Dict[str, Any], text: str) -> None:
        """If no 'profession' field, attempt to detect or set default."""
        if "profession" not in data or not data["profession"]:
            guess = self._detect_profession(text)
            data["profession"] = guess if guess else settings.DEFAULT_PROFESSION

    def _default_booking_time(self) -> datetime:
        """
        If no time extracted, schedule for next hour or next day if it's too late.
        """
        now = datetime.now()
        if now.hour >= settings.LAST_BOOKING_HOUR:
            return now.replace(
                hour=settings.DEFAULT_BOOKING_HOUR, minute=0, second=0
            ) + timedelta(days=1)
        return now.replace(hour=now.hour + 1, minute=0, second=0)

    def _detect_profession(self, text: str) -> Optional[str]:
        """
        Scan the text for known profession keywords if none found in NER.
        """
        lower_text = text.lower()
        for profession, keywords in settings.PROFESSION_KEYWORDS.items():
            if any(k.lower() in lower_text for k in keywords):
                logger.debug(f"Detected profession: {profession}")
                return profession
        logger.debug("No profession found in text.")
        return None

    def __repr__(self) -> str:
        """For debugging/logging introspection."""
        return (
            f"NLPService("
            f"device={self.device_name}, "
            f"intent_model={settings.ZERO_SHOT_MODEL_NAME}, "
            f"ner_model={settings.NER_MODEL_NAME}, "
            f"date_time_model={settings.TEXT_TO_TEXT_MODEL_NAME})"
        )


# Initialize a global (singleton) instance if desired
try:
    nlp_service = NLPService()
except Exception as e:
    logger.critical(f"Failed to initialize NLPService: {e}")
    raise
