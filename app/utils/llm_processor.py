"""
llm_processor.py

A robust implementation of a Hugging Face Transformers-based NLP processor for:
1) Zero-shot classification (intent detection)
2) Token classification (NER)

Features:
- Task-specific pipeline configurations
- Automatic GPU/CPU handling with graceful fallback
- Comprehensive error handling
- Detailed logging
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import dateutil.parser
from transformers import pipeline, Pipeline

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception class for LLM processing errors."""
    pass


class ModelInitializationError(ProcessingError):
    """Raised when model initialization fails."""
    pass


class InferenceError(ProcessingError):
    """Raised when model inference fails."""
    pass


class ValidationError(ProcessingError):
    """Raised when input validation fails."""
    pass


class ParsedCommand:
    """Container for parsed user input (intent + data)."""
    
    def __init__(self, intent: str, data: Dict[str, Any]):
        """
        Initialize a parsed command.
        
        Args:
            intent: The classified intent of the command
            data: Associated data extracted from the command
        """
        self.intent = intent
        self.data = data

    def __repr__(self) -> str:
        return f"ParsedCommand(intent='{self.intent}', data={self.data})"


class LLMProcessor:
    """
    A professional implementation of an NLP processor using Hugging Face Transformers.
    
    Features:
    - Zero-shot classification for intent detection
    - Token classification for named entity recognition (NER)
    - Automatic GPU utilization with CPU fallback
    - Comprehensive error handling and logging
    """

    def __init__(self) -> None:
        """Initialize the LLM processor with appropriate device selection and pipeline setup."""
        self.candidate_intents = [
            "create_booking",
            "cancel_booking",
            "retrieve_booking",
            "list_bookings",
        ]
        
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing LLM processor with device: {'GPU' if self.device == 0 else 'CPU'}")
        
        self._initialize_pipelines()

    def _get_pipeline_config(self, task: str) -> Dict[str, Any]:
        """
        Get task-specific pipeline configuration.
        
        Args:
            task: The pipeline task type
            
        Returns:
            Dictionary of pipeline configuration parameters
        """
        base_config = {
            "model": settings.MODEL_NAME,
            "device": self.device,
        }

        if task == "zero-shot-classification":
            if settings.HUGGINGFACE_API_KEY:
                base_config["use_auth_token"] = settings.HUGGINGFACE_API_KEY
                
        elif task == "token-classification":
            base_config["aggregation_strategy"] = "simple"
            
        return base_config

    def _initialize_pipelines(self) -> None:
        """Initialize NLP pipelines with error handling."""
        try:
            self.intent_classifier = self._create_pipeline("zero-shot-classification")
            self.ner_pipeline = self._create_pipeline("token-classification")
        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {str(e)}")
            raise ModelInitializationError(f"Pipeline initialization failed: {str(e)}")

    def _create_pipeline(self, task: str) -> Pipeline:
        """
        Create a Hugging Face pipeline with proper error handling.
        
        Args:
            task: The pipeline task type
            
        Returns:
            Configured pipeline instance
            
        Raises:
            ModelInitializationError: If pipeline creation fails
        """
        try:
            pipeline_args = self._get_pipeline_config(task)
            return pipeline(task, **pipeline_args)
            
        except torch.OutOfMemoryError:
            logger.warning(f"GPU OOM during {task} pipeline creation, falling back to CPU")
            self.device = -1
            pipeline_args = self._get_pipeline_config(task)
            pipeline_args["device"] = -1
            return pipeline(task, **pipeline_args)
            
        except Exception as e:
            logger.error(f"Failed to create {task} pipeline: {str(e)}")
            raise ModelInitializationError(f"Failed to create {task} pipeline: {str(e)}")

    def _classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (classified intent, confidence score)
            
        Raises:
            InferenceError: If classification fails
        """
        try:
            result = self.intent_classifier(
                sequences=text,
                candidate_labels=self.candidate_intents,
                multi_label=False
            )
            return result["labels"][0], result["scores"][0]
            
        except torch.OutOfMemoryError:
            logger.warning("GPU OOM during intent classification, falling back to CPU")
            self.device = -1
            self.intent_classifier = self._create_pipeline("zero-shot-classification")
            result = self.intent_classifier(
                sequences=text,
                candidate_labels=self.candidate_intents,
                multi_label=False
            )
            return result["labels"][0], result["scores"][0]
            
        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            raise InferenceError(f"Intent classification failed: {str(e)}")

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from the input text.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            Dictionary of extracted entities and their values
            
        Raises:
            InferenceError: If entity extraction fails
        """
        try:
            ner_results = self.ner_pipeline(text)
        except torch.OutOfMemoryError:
            logger.warning("GPU OOM during NER, falling back to CPU")
            self.device = -1
            self.ner_pipeline = self._create_pipeline("token-classification")
            ner_results = self.ner_pipeline(text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise InferenceError(f"Entity extraction failed: {str(e)}")

        return self._process_entities(ner_results, text)

    def _process_entities(self, ner_results: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Process and structure the extracted entities."""
        data: Dict[str, Any] = {}
        
        # Extract booking ID if present
        booking_id_match = re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            text,
            re.IGNORECASE
        )
        if booking_id_match:
            data["booking_id"] = booking_id_match.group(0)

        # Process recognized entities
        recognized_profession = None
        recognized_time = None

        for entity in ner_results:
            word = entity.get("word", "").lower()
            if word in ["plumber", "electrician", "welder"]:
                recognized_profession = word
            
            try:
                recognized_time = dateutil.parser.parse(word)
            except (ValueError, TypeError):
                continue

        if recognized_profession:
            data["profession"] = recognized_profession.capitalize()

        # Handle time-related information
        if recognized_time is None and "tomorrow" in text.lower():
            tomorrow = datetime.now() + timedelta(days=1)
            recognized_time = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)

        if recognized_time:
            data["start_time"] = recognized_time

        return data

    def parse_user_input(self, user_input: str) -> ParsedCommand:
        """
        Parse user input into a structured command with intent and data.
        
        Args:
            user_input: Raw user input text
            
        Returns:
            ParsedCommand object containing intent and extracted data
            
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        if not user_input or not user_input.strip():
            raise ValidationError("Input text cannot be empty")

        text = user_input.strip()
        
        # Classify intent
        intent, confidence = self._classify_intent(text)
        if confidence < 0.25:
            raise ValidationError(f"Low confidence ({confidence:.2f}) for intent '{intent}'")

        # Extract entities
        data = self._extract_entities(text)
        
        # Create and validate command
        command = ParsedCommand(intent=intent, data=data)
        self._validate_command(command)
        
        return command

    def _validate_command(self, command: ParsedCommand) -> None:
        """
        Validate the parsed command and set defaults where appropriate.
        
        Args:
            command: ParsedCommand to validate
            
        Raises:
            ValidationError: If command validation fails
        """
        intent = command.intent
        data = command.data

        if intent == "create_booking":
            data.setdefault("profession", "Plumber")
            if "start_time" not in data:
                tomorrow = datetime.now() + timedelta(days=1)
                data["start_time"] = tomorrow.replace(hour=10, minute=0, second=0)
            data.setdefault("customer_name", "Anonymous User")
            data.setdefault("technician_name", "Unknown Technician")

        elif intent in ["cancel_booking", "retrieve_booking"]:
            if "booking_id" not in data:
                raise ValidationError(f"{intent} requires a booking_id")


# Global instance with proper error handling
try:
    llm_processor = LLMProcessor()
except Exception as e:
    logger.critical(f"Failed to initialize global LLM processor: {str(e)}")
    raise


def parse_user_input(user_input: str) -> ParsedCommand:
    """
    Convenience function to parse user input with the global LLM processor instance.
    
    Args:
        user_input: Raw user input text
        
    Returns:
        ParsedCommand object containing intent and extracted data
        
    Raises:
        ProcessingError: If processing fails
    """
    return llm_processor.parse_user_input(user_input)