"""
nlp_service.py

A professional-grade implementation of a multi-pipeline NLP service
using Hugging Face Transformers. Manages:
  • Zero-shot classification for user intent
  • NER for names (technician, customer) and partial times
  • Text2text LLM for advanced date/time interpretation

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

from app.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class PipelineException(Exception):
    """Custom exception for pipeline initialization failures."""
    pass


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
    Multi-pipeline NLP service. Orchestrates:
      1) Zero-shot for intent classification
      2) NER for entity extraction
      3) Text2text LLM for complex date/time parsing

    Key Features:
      - Model caching & retries
      - Deterministic or fallback date/time resolution
      - Basic name & profession extraction
      - Error handling & logging
    """

    INTENT_LABEL_MAPPING = {
        "Create a booking": "create_booking",
        "Cancel a booking": "cancel_booking",
        "Retrieve booking details": "retrieve_booking",
        "List all bookings": "list_bookings",
    }

    def __init__(self) -> None:
        """Initialize the NLP service with all required pipelines."""
        self._validate_settings()
        self.device, self.device_name = self._optimize_hardware()
        self.intent_classifier = self._init_pipeline_with_retry(
            "zero-shot-classification",
            settings.ZERO_SHOT_MODEL_NAME,
            {"multi_label": False}
        )
        self.ner_pipeline = self._init_pipeline_with_retry(
            "token-classification",
            settings.NER_MODEL_NAME,
            {"aggregation_strategy": "simple"}
        )
        self.date_time_llm = self._init_pipeline_with_retry(
            "text2text-generation",
            settings.DATE_TIME_MODEL_NAME,
            {
                "max_length": 64,
                "temperature": 0.0
            }
        )

    # -------------------------------------------------------------------------
    # Initialization & Configuration
    # -------------------------------------------------------------------------
    def _validate_settings(self) -> None:
        """Check critical config values."""
        if not settings.CANDIDATE_INTENTS:
            raise ValueError("CANDIDATE_INTENTS must be non-empty list")
        if not (0 <= settings.INTENT_CONFIDENCE_THRESHOLD <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not settings.ZERO_SHOT_MODEL_NAME:
            raise ValueError("Zero-shot model name must be configured")
        if not settings.NER_MODEL_NAME:
            raise ValueError("NER model name must be configured")
        if not settings.DATE_TIME_MODEL_NAME:
            logger.warning("DATE_TIME_MODEL_NAME not set; advanced date/time LLM unavailable")

    def _optimize_hardware(self) -> Tuple[int, str]:
        """Pick GPU vs. CPU, optionally enable half-precision."""
        if torch.cuda.is_available() and settings.USE_GPU:
            device = 0
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
        else:
            device = -1
            device_name = "CPU"
            logger.info("Using CPU for processing")

        if settings.USE_HALF_PRECISION and device >= 0:
            torch.backends.cudnn.benchmark = True
            logger.debug("Enabled mixed precision training")

        return device, device_name

    def _model_cache_status(self, model_name: str) -> Tuple[bool, str]:
        """Check if model is locally cached."""
        cache_root = os.path.expanduser(settings.HF_CACHE_DIR)
        model_dir = f"models--{model_name.replace('/', '--')}"
        cache_path = os.path.join(cache_root, model_dir)
        return os.path.exists(cache_path), cache_path

    def _init_pipeline_with_retry(self, task: str, model_name: str, task_args: Dict[str, Any]) -> Pipeline:
        """
        Create a pipeline with robust retries. If cache fails, attempt fresh download.
        """
        if not model_name:
            logger.warning(f"No model name provided for task={task}; skipping pipeline init.")
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
                    **task_args
                )
            except (OSError, PipelineException) as e:
                if is_cached and attempt == 0:
                    logger.warning(f"Model load failed from cache {cache_path}: {e}")
                    is_cached = False
                    continue
                logger.error(f"Pipeline init failed (attempt {attempt+1}): {e}")
                if attempt == settings.MODEL_LOAD_RETRIES - 1:
                    raise RuntimeError(f"Failed to init {task} after {settings.MODEL_LOAD_RETRIES} tries") from e

        # Should never reach here
        raise RuntimeError("Unexpected pipeline initialization error.")

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    def parse_user_input(self, text: str) -> ParsedCommand:
        """
        Main entry point:
         1) Validate input,
         2) Determine intent,
         3) Extract relevant entities,
         4) Return final structured data in a ParsedCommand.
        """
        self._validate_input(text)
        cleaned_text = self._preprocess_text(text)

        try:
            intent_label, confidence = self._classify_intent(cleaned_text)
            internal_intent = self.INTENT_LABEL_MAPPING.get(intent_label)
            if not internal_intent:
                raise ValueError(f"Unknown intent: {intent_label}")

            # Extract entity data
            entities = self._extract_entities(cleaned_text)
            # Possibly enrich with defaults
            enriched_data = self._enrich_data(internal_intent, entities, cleaned_text)
            enriched_data["confidence"] = confidence

            # If it's a booking, ensure we have a profession
            if internal_intent == "create_booking":
                self._ensure_profession(enriched_data, cleaned_text)

            # Check required fields for the chosen intent
            self._validate_intent_requirements(internal_intent, enriched_data)

            return ParsedCommand(intent=internal_intent, data=enriched_data)
        except ValueError as ve:
            logger.error(f"Input parsing failed: {ve}")
            raise ValueError(str(ve))

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
            raise ValueError(f"Input must be at least {settings.MIN_INPUT_LENGTH} characters")

        if len(cleaned_text) > settings.MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds max length of {settings.MAX_INPUT_LENGTH} characters")

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
                r"(?i)\b(?:cancel|remove|delete|terminate|abort|discard|void)\s+(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\s+(?:for\s+.+)$"
            ],
            "List all bookings": [
                # e.g., "List my bookings", "Show all reservations", "Display my appointments"
                r"(?i)\b(?:list|show|display|view|see|get|fetch|present|give\s+me)\s+(?:all\s+)?(?:my\s+)?(?:bookings|reservations|appointments|schedules|sessions)\b",
                # e.g., "Can you list my bookings?", "Please show all reservations"
                r"(?i)\b(?:can\s+you\s+|please\s+)?(?:list|show|display|view|see|get|fetch|present|give\s+me)\s+(?:all\s+)?(?:my\s+)?(?:bookings|reservations|appointments|schedules|sessions)\b"
            ],
            "Retrieve booking details": [
                # e.g., "Get details of my booking", "Find my reservation", "Retrieve appointment information"
                r"(?i)\b(?:get|find|retrieve|access|look\s+up)\s+(?:the\s+)?(?:details\s+of\s+)?(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b",
                # e.g., "Can you get booking details for me?", "Retrieve my reservation information"
                r"(?i)\b(?:can\s+you\s+)?(?:get|find|retrieve|access|look\s+up)\s+(?:the\s+)?(?:details\s+of\s+)?(?:my\s+)?(?:booking|reservation|appointment|schedule|session)\b"
            ],
            "Create a booking": [
                # e.g., "I want to book a plumber", "Schedule an electrician", "Need a technician"
                r"(?i)\b(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\b",
                # e.g., "Can you book a plumber for me?", "Please schedule an electrician"
                r"(?i)\b(?:can\s+you\s+|please\s+)?(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\b",
                # e.g., "I need to book a technician tomorrow", "Schedule a plumber on Monday"
                r"(?i)\b(?:book|schedule|reserve|arrange|set\s+up|make\s+a)\s+(?:an?\s+)?(?:plumber|electrician|technician)\s+(?:for\s+.+)$"
            ]
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
            text,
            candidate_labels=settings.CANDIDATE_INTENTS,
            multi_label=False
        )
        top_label = classification["labels"][0]
        top_conf = classification["scores"][0]

        if top_conf < settings.INTENT_CONFIDENCE_THRESHOLD:
            raise ValueError(f"Intent confidence too low: {top_conf}")

        logger.debug(f"Zero-shot predicted '{top_label}' ({top_conf:.2f})")
        return top_label, top_conf

    def _validate_intent_requirements(self, intent: str, data: Dict[str, Any]) -> None:
        """Ensure required fields exist for each intent."""
        missing_booking_id = intent in ["cancel_booking", "retrieve_booking"] and "booking_id" not in data
        if missing_booking_id:
            raise ValueError(f"Booking ID is required for {intent}")

        missing_profession = (intent == "create_booking") and ("profession" not in data)
        if missing_profession:
            raise ValueError("Profession is required for booking creation")

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract high-level booking-related entities (Booking ID, Technician Name, 
        Customer Name, Time, Profession, etc.) from a raw user command. 

        This function is central to the Technician Booking System's 
        natural language processing workflow. It processes incoming 
        text (e.g., "I want to book a gardener for tomorrow") and 
        produces structured information needed to create or manage 
        a booking.

        Business Rationale:
        -------------------
        1. **Identify the Customer & Technician**:
        - Determines who is booking (customer_name) and which technician 
            is requested (technician_name), supporting personalized 
            interactions and recordkeeping.
        2. **Determine Time Expressions**:
        - Accurately captures the user's intended scheduling time 
            (start_time), facilitating timely bookings and 
            preventing misunderstandings about "tomorrow," 
            "Friday at 3 PM," etc.
        3. **Associate Booking IDs**:
        - Enables direct references to existing appointments (e.g. 
            "cancel booking 123") by extracting numerical booking IDs.
        4. **Detect Profession**:
        - Infers the requested technician's specialization (e.g., 
            "plumber," "gardener") to streamline service matching.

        Data Flow Summary:
        ------------------
        1. **Extract Booking ID**: 
        - Searches for patterns like "booking 123" or "#456" 
            to recognize references to specific reservations.

        2. **Named Entity Recognition** (NER):
        - Leverages a trained pipeline to label tokens as TIME, 
            PROFESSION, PER, ORG, LOC, etc.
        - Collects partial information on names (technician vs. 
            customer via the " for " keyword), times, and 
            profession references.

        3. **Time/Date Token Merging**:
        - Aggregates consecutive TIME tokens (e.g., ["Friday", "3", "PM"]) 
            into one expression ("Friday 3 PM"), then attempts to parse 
            it as a future booking time.

        4. **Fallback Parsing**:
        - If no time was captured, checks common patterns:
            - "tomorrow at 2pm"
            - "tomorrow" (alone) → default hour
            - "friday at 3 pm" (day-of-week + time)
        - Ensures the user-specified date is recognized even if NER 
            didn't handle it.

        5. **Assign Entities**:
        - Synthesizes a final dictionary of extracted fields, 
            enabling the booking logic to create or manage 
            reservations seamlessly.

        Args:
        -----
            text (str):
                Raw user input containing instructions about 
                bookings, dates/times, technician requirements, etc.

        Returns:
        --------
            Dict[str, Any]:
                A dictionary of recognized entities, potentially including:
                
                - "booking_id":     str or None
                - "technician_name": str
                - "customer_name":   str
                - "profession":     str (e.g., "gardener")
                - "start_time":     datetime (the intended booking time)
        
        Raises:
        -------
            None explicitly here, though sub-calls may raise `ValueError` 
            if time parsing fails. The function logs warnings if 
            pipelines are unavailable.

        Usage Example:
        --------------
            >>> text_cmd = "I want to book a gardener for tomorrow"
            >>> entities = self._extract_entities(text_cmd)
            >>> print(entities)
            {
            "booking_id": None,
            "customer_name": "Anonymous", 
            "technician_name": None, 
            "profession": "gardener",
            "start_time": datetime.datetime(2025, 1, 30, 9, 0)
            }

        Overall, this function ensures that the unstructured text from 
        users or clients is converted into the structured information 
        needed by our booking services. It is the backbone of the 
        Technician Booking System's natural language layer.
        """

        entities: Dict[str, Any] = {}

        # 1) Attempt to find a booking ID (e.g., "booking 123" or "#456").
        booking_id = self._extract_booking_id(text)
        if booking_id:
            entities["booking_id"] = booking_id

        # 2) Run the NER pipeline for name/time detection.
        if self.ner_pipeline is None:
            logger.warning("No NER pipeline detected. Limited entity extraction.")
            return entities

        ner_results = self.ner_pipeline(text)
        text_lower = text.lower()
        for_index = text_lower.find(" for ")

        # Prepare arrays to collate partial data
        tech_names, cust_names, time_tokens = [], [], []

        for item in ner_results:
            group = item["entity_group"]
            word = item["word"]
            start_idx = item["start"]

            # Gather tokens labeled as TIME
            if group == "TIME":
                time_tokens.append(word)

            # Occupation references
            elif group == "PROFESSION":
                entities["profession"] = word.lower()

            # People (technician or customer)
            elif group == "PER":
                # Decide if after " for " => customer
                if for_index != -1 and start_idx > for_index:
                    cust_names.append(word)
                else:
                    tech_names.append(word)

            # Fallback for ORG/LOC if capitalized => potential name
            elif group in ("LOC", "ORG") and word.istitle():
                if for_index != -1 and start_idx > for_index:
                    cust_names.append(word)
                else:
                    tech_names.append(word)

        # Aggregate multi-token names
        if tech_names:
            entities["technician_name"] = " ".join(tech_names)
        if cust_names:
            entities["customer_name"] = " ".join(cust_names)

        # 3) Merge time tokens => single expression & parse
        if time_tokens:
            merged_time = " ".join(time_tokens)
            logger.debug(f"Merged time tokens => '{merged_time}'")
            self._assign_time(merged_time, entities)

        # 4) Fallback for "tomorrow at X pm"
        if "start_time" not in entities:
            tomorrow_match = re.search(r"\btomorrow at \d{1,2}(?:am|pm)\b", text_lower)
            if tomorrow_match:
                snippet = tomorrow_match.group(0)
                self._assign_time(snippet, entities)

        # Additional fallback for "tomorrow" alone
        if "start_time" not in entities:
            if re.search(r"\btomorrow\b", text_lower):
                snippet = "tomorrow at 9am"  # default hour if user omitted any time
                logger.debug(f"No explicit hour with 'tomorrow'; defaulting to '{snippet}'")
                self._assign_time(snippet, entities)

        # Day-of-week fallback (e.g. "friday at 3 pm")
        if "start_time" not in entities:
            day_time_match = re.search(
                r"\b(mon|tue|wed|thu|fri|sat|sun)\w*\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)\b",
                text_lower
            )
            if day_time_match:
                snippet = day_time_match.group(0)
                logger.debug(f"Day-of-week fallback snippet => '{snippet}'")
                self._assign_time(snippet, entities)

        logger.debug(f"Final extracted entities: {entities}")
        return entities

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
            r"\bid\s*(\d+)\b"
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1)

        # 3) If user said e.g. "retrieve booking 123", forcibly parse any standalone number
        if any(k in text_lower for k in ["cancel", "booking", "retrieve", "find", "details"]):
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
                microsecond=0
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
            "mon": 0, "monday": 0,
            "tue": 1, "tues": 1, "tuesday": 1,
            "wed": 2, "weds": 2, "wednesday": 2,
            "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
            "fri": 4, "friday": 4,
            "sat": 5, "saturday": 5,
            "sun": 6, "sunday": 6
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
    def _enrich_data(self, intent: str, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
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
            return now.replace(hour=settings.DEFAULT_BOOKING_HOUR, minute=0, second=0) + timedelta(days=1)
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
            f"date_time_model={settings.DATE_TIME_MODEL_NAME})"
        )


# Initialize a global (singleton) instance if desired
try:
    llm_processor = NLPService()
except Exception as e:
    logger.critical(f"Failed to initialize NLPService: {e}")
    raise
