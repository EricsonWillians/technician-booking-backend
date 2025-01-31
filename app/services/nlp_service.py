# app/services/nlp_service.py

"""
==========================================================
               TECHNICIAN BOOKING SYSTEM - NLP SERVICE
==========================================================

Implements the Natural Language Processing (NLP) service for the
Technician Booking System. Utilizes Zero-Shot Classification and
Named Entity Recognition (NER) pipelines from Hugging Face's Transformers
library to interpret user inputs, extract intents and relevant entities,
and facilitate booking operations.

Key Features:
- Intent Classification: Determines user intent (e.g., create, cancel, query bookings).
- Entity Extraction: Extracts professions, technician names, datetime, and booking IDs.
- Booking Operations: Interfaces with the booking service to create, cancel, or query bookings.
- Multi-Interface Support: Designed to be used seamlessly with both CLI and FastAPI routes.
- Detailed Intent Scores: Returns intent classification scores for monitoring model performance.

Author : Ericson Willians  
Email  : ericsonwillians@protonmail.com  
Date   : January 2025  

==========================================================
"""

import logging
import re
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta

from transformers import pipeline
from dateutil import parser

from app.models.professions import ProfessionEnum
from app.services.booking_service import (
    create_booking,
    get_booking_by_id,
    cancel_booking,
    get_all_bookings,  # Ensure this function exists in booking_service.py
)
from app.schemas.booking import BookingCreate
from app.config.settings import settings
from app.utils.datetime_utils import DateTimeExtractor, DateTimeExtractionError
from app.services import booking_service

from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class MessageResponse:
    """Data class to encapsulate the response message and intent scores."""
    response: str
    intent_scores: Dict[str, float]
class NLPService:
    def __init__(self):
        """
        Initialize NLP components with improved classification.
        """
        logger.info("Initializing NLPService...")
        
        # Initialize Zero-Shot Classification pipeline with specific hypothesis
        try:
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model=settings.ZERO_SHOT_MODEL_NAME,
                device=0 if self._is_gpu_available() else -1,
            )
            logger.info("Zero-Shot Classification pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Zero-Shot Classification pipeline: {e}")
            raise

        # Initialize NER pipeline
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=settings.NER_MODEL_NAME,
                aggregation_strategy="simple",
                device=0 if self._is_gpu_available() else -1,
            )
            logger.info("NER pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize NER pipeline: {e}")
            raise

        self.datetime_extractor = DateTimeExtractor()

        # Enhanced intent patterns with better coverage
        self.intent_patterns = {
            "create_booking": [
                r"\b(want|need|looking|book|schedule|get|make|arrange|set)\s+(?:to|a|an)?\s+(?:book|schedule|appointment|service|visit)\b",
                r"\b(?:book|schedule|need|want)\s+(?:a|an)?\s+(?:gardener|plumber|electrician|carpenter|mechanic|painter|chef|teacher|developer|nurse)\b",
                r"\bi(?:\s+would)?\s+(?:like|want|need)\s+to\s+(?:book|schedule|make|get)\b"
            ],
            "cancel_booking": [
                r"\b(?:cancel|delete|remove|stop)\s+(?:the|my|this)?\s*(?:booking|appointment|service|reservation)\b",
                r"\bi\s+(?:want|need|would\s+like)\s+to\s+cancel\b"
            ],
            "query_booking": [
                r"\b(?:what|where|when|how|show|get|check|find|view)\s+(?:is|are)?\s+(?:the|my)?\s*(?:booking|appointment|reservation)\b",
                r"\b(?:booking|appointment|reservation)\s+(?:status|details|info|information)\b",
                r"\bstatus\s+of\s+(?:my|the)\s+booking\b"
            ],
            "list_bookings": [
                r"\b(?:list|show|view|display|get)\s+(?:all|my)?\s*(?:bookings|appointments|reservations|schedule)\b",
                r"\bwhat\s+(?:bookings|appointments|reservations)\s+do\s+i\s+have\b"
            ]
        }

        # Enhanced intent descriptions for zero-shot classification
        self.intent_descriptions = {
            "create_booking": [
                "make a new appointment or booking",
                "schedule a service",
                "book a professional",
                "arrange an appointment",
                "request a service booking",
            ],
            "cancel_booking": [
                "cancel an existing booking",
                "delete a scheduled appointment",
                "remove a service booking",
                "stop a scheduled service",
            ],
            "query_booking": [
                "find information about a booking",
                "check booking details",
                "view appointment information",
                "get booking status",
            ],
            "list_bookings": [
                "view all scheduled appointments",
                "show all my bookings",
                "display booking schedule",
                "list all reservations",
            ]
        }

        self.candidate_intents = list(self.intent_patterns.keys())
        
        self.booking_id_pattern = re.compile(
            r'\b(?:booking\s+id|booking-id|booking)\s*(?:is|=)?\s*([A-Za-z0-9-]+)\b',
            re.IGNORECASE
        )
        
        self.profession_patterns = {
            ProfessionEnum.CARPENTER: r'\b(?:carpenter|woodwork(?:er)?|cabinet\s*maker)\b',
            ProfessionEnum.CHEF: r'\b(?:chef|cook|culinary|kitchen)\b',
            ProfessionEnum.DEVELOPER: r'\b(?:developer|programmer|coder|software|web\s*dev)\b',
            ProfessionEnum.ELECTRICIAN: r'\b(?:electrician|electrical\s*worker)\b',
            ProfessionEnum.GARDENER: r'\b(?:garden(?:er)?|landscap(?:er|ing)|yard\s*work(?:er)?)\b',
            ProfessionEnum.MECHANIC: r'\b(?:mechanic|auto\s*tech|car\s*repair)\b',
            ProfessionEnum.NURSE: r'\b(?:nurse|nursing|health\s*care|medical)\b',
            ProfessionEnum.PAINTER: r'\b(?:paint(?:er)?|decorat(?:or|ing))\b',
            ProfessionEnum.PLUMBER: r'\b(?:plumb(?:er|ing)|pipe\s*fitt(?:er|ing))\b',
            ProfessionEnum.TEACHER: r'\b(?:teach(?:er)?|tutor|instructor|educator)\b',
            ProfessionEnum.WELDER: r'\b(?:weld(?:er|ing)|metal\s*work(?:er)?)\b'
        }

    def _is_gpu_available(self) -> bool:
        """
        Checks if a GPU is available for model inference.

        Returns:
            bool: True if GPU is available, False otherwise.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("Torch not installed. Running on CPU.")
            return False

    def classify_intent(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Enhanced intent classification using pattern matching and zero-shot classification.
        """
        logger.debug(f"Classifying intent for text: '{text}'")
        text_lower = text.lower()
        
        # First try pattern matching
        pattern_scores = {}
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pattern_scores[intent] = pattern_scores.get(intent, 0) + 1

        if pattern_scores:
            # Found pattern matches
            max_matches = max(pattern_scores.values())
            matching_intents = [i for i, s in pattern_scores.items() if s == max_matches]
            
            if len(matching_intents) == 1:
                # Clear pattern match
                logger.info(f"Clear pattern match found for intent: {matching_intents[0]}")
                return matching_intents[0], {intent: 1.0 if intent == matching_intents[0] else 0.0 
                                        for intent in self.candidate_intents}
        
        # Use zero-shot classification with better prompting
        try:
            # Create candidate labels with descriptions
            candidate_labels = []
            label_map = {}
            
            for intent, descriptions in self.intent_descriptions.items():
                for desc in descriptions:
                    candidate_labels.append(desc)
                    label_map[desc] = intent

            # Run classification with multi_label=False to force single intent
            result = self.intent_classifier(
                text,
                candidate_labels,
                hypothesis_template="This request is about {}.",
                multi_label=False
            )
            
            # Aggregate scores by intent
            intent_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                intent = label_map[label]
                intent_scores[intent] = max(intent_scores.get(intent, 0), score)
                
            # Boost pattern-matched intents
            if pattern_scores:
                for intent in pattern_scores:
                    intent_scores[intent] = min(1.0, intent_scores.get(intent, 0) * 1.5)

            # Get highest scoring intent
            max_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            
            # Normalize scores
            total = sum(intent_scores.values())
            intent_scores = {k: v/total for k, v in intent_scores.items()}
            
            logger.info(f"Classified intent: {max_intent} with scores {intent_scores}")
            return max_intent, intent_scores
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "unknown", {intent: 0.0 for intent in self.candidate_intents}

    def extract_entities(
        self, text: str
    ) -> Tuple[Optional[ProfessionEnum], Optional[str], Optional[datetime], Optional[str]]:
        """
        Enhanced entity extraction with better profession and datetime handling.
        """
        logger.debug(f"Extracting entities from text: '{text}'")
        try:
            entities = self.ner_pipeline(text)
            
            # Initialize return values
            profession = self.extract_profession(text)  # Extract profession first
            technician_name = None
            date_time = None
            booking_id = None

            # Extract PERSON entities
            persons = [ent['word'] for ent in entities if ent['entity_group'] == 'PER']
            if persons:
                technician_name = ' '.join(persons)
                logger.info(f"Extracted technician name: {technician_name}")
            else:
                technician_name = "Anonymous Technician"
                logger.info("Using default technician name")

            # Extract datetime entities with better handling
            date_entities = [ent['word'] for ent in entities if ent['entity_group'] == 'DATE']
            time_entities = [ent['word'] for ent in entities if ent['entity_group'] == 'TIME']
            
            datetime_str = f"{' '.join(date_entities)} {' '.join(time_entities)}".strip()

            if datetime_str:
                try:
                    extracted_datetime = self.datetime_extractor.extract_datetime_entities(
                        {"date": ' '.join(date_entities), "time": ' '.join(time_entities)},
                        datetime_str
                    )
                    date_time = extracted_datetime.get("start_time")
                    if date_time:
                        logger.info(f"Extracted datetime: {date_time}")
                except DateTimeExtractionError as e:
                    logger.error(f"Failed to parse explicit datetime: {e}")

            # If no datetime found, try parsing from full text
            if not date_time:
                try:
                    extracted = self.datetime_extractor.extract_datetime_entities({}, text)
                    date_time = extracted.get("start_time")
                    if date_time:
                        logger.info(f"Extracted datetime from full text: {date_time}")
                except DateTimeExtractionError as e:
                    logger.error(f"Failed to parse datetime from full text: {e}")

            # Extract booking ID
            booking_id_match = self.booking_id_pattern.search(text)
            if booking_id_match:
                booking_id = booking_id_match.group(1)
                logger.info(f"Extracted booking ID: {booking_id}")

            # Log extraction results
            logger.info(f"Extraction results - Profession: {profession}, "
                       f"Technician: {technician_name}, "
                       f"DateTime: {date_time}, "
                       f"BookingID: {booking_id}")

            return profession, technician_name, date_time, booking_id

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return None, None, None, None

    def extract_profession(self, text: str) -> Optional[ProfessionEnum]:
        """
        Enhanced profession extraction using regex patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            ProfessionEnum or None: Extracted profession if found
        """
        text_lower = text.lower()
        
        # Try to match profession patterns
        for profession, pattern in self.profession_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.info(f"Matched profession {profession.value} with pattern")
                return profession
                
        logger.debug("No profession pattern matched")
        return None

    def handle_message(
    self, message: str, customer_name: str = "Anonymous Customer"
) -> MessageResponse:
        """
        Processes the user's message with improved error handling.
        Always returns a MessageResponse object.
        """
        try:
            logger.info(f"Handling message: '{message}' from customer: '{customer_name}'")
            
            # Get intent and scores
            intent, intent_scores = self.classify_intent(message)
            
            if not intent_scores:
                logger.warning("No intent scores received")
                return MessageResponse(
                    response="I couldn't understand that request. Could you please rephrase it?",
                    intent_scores={"unknown": 1.0}
                )
                
            # Extract entities with proper error handling
            try:
                profession, technician_name, date_time, booking_id = self.extract_entities(message)
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
                return MessageResponse(
                    response=f"I had trouble understanding the details of your request: {str(e)}",
                    intent_scores=intent_scores
                )

            # Handle each intent
            if intent == "create_booking":
                # Set up default datetime if none extracted
                if not date_time:
                    tomorrow = datetime.now(settings.TIMEZONE_OBJ) + timedelta(days=1)
                    date_time = tomorrow.replace(
                        hour=settings.DEFAULT_BOOKING_HOUR,
                        minute=0,
                        second=0,
                        microsecond=0
                    )
                    logger.info(f"Using default booking time: {date_time}")

                # Validate booking time
                current_time = datetime.now(settings.TIMEZONE_OBJ)
                if date_time <= current_time:
                    return MessageResponse(
                        response="Cannot book a technician in the past.",
                        intent_scores=intent_scores
                    )

                try:
                    booking_create = BookingCreate(
                        customer_name=customer_name,
                        technician_name=technician_name,
                        profession=profession,
                        start_time=date_time
                    )

                    booking_response = create_booking(booking_create)
                    formatted_time = booking_response.start_time.strftime("%A at %I:%M %p")
                    response = f"Booking confirmed for {formatted_time} with {booking_response.technician_name} (ID: {booking_response.id})"
                    logger.info(f"Booking created successfully: {response}")
                    return MessageResponse(response=response, intent_scores=intent_scores)
                except ValueError as ve:
                    error_msg = f"Failed to create booking: {str(ve)}"
                    logger.error(error_msg)
                    return MessageResponse(response=error_msg, intent_scores=intent_scores)

            elif intent == "query_booking":
                if not booking_id:
                    return MessageResponse(
                        response="Please provide your booking ID to retrieve details.",
                        intent_scores=intent_scores
                    )

                booking = get_booking_by_id(booking_id)
                if booking:
                    formatted_time = booking.start_time.strftime("%A at %I:%M %p")
                    response = f"Your booking ID is {booking.id} for a {booking.profession} on {formatted_time}."
                    return MessageResponse(response=response, intent_scores=intent_scores)
                else:
                    return MessageResponse(
                        response=f"No booking found with ID {booking_id}.",
                        intent_scores=intent_scores
                    )

            elif intent == "cancel_booking":
                if not booking_id:
                    return MessageResponse(
                        response="Please provide the booking ID you wish to cancel.",
                        intent_scores=intent_scores
                    )

                success = cancel_booking(booking_id)
                if success:
                    return MessageResponse(
                        response=f"Booking ID {booking_id} cancelled successfully.",
                        intent_scores=intent_scores
                    )
                else:
                    return MessageResponse(
                        response=f"No booking found with ID {booking_id} to cancel.",
                        intent_scores=intent_scores
                    )

            elif intent == "list_bookings":
                bookings = get_all_bookings()
                if bookings:
                    response = "Here are all your bookings:\n"
                    for booking in bookings:
                        response += f"- ID: {booking.id}, Technician: {booking.technician_name}, "
                        response += f"Profession: {booking.profession}, "
                        response += f"Start: {booking.start_time.strftime('%Y-%m-%d %I:%M %p')}\n"
                    return MessageResponse(response=response, intent_scores=intent_scores)
                else:
                    return MessageResponse(
                        response="You have no bookings at the moment.",
                        intent_scores=intent_scores
                    )

            else:
                logger.warning(f"Unrecognized intent: {intent}")
                return MessageResponse(
                    response="I'm sorry, I didn't understand that request. Could you please rephrase it?",
                    intent_scores=intent_scores
                )

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            return MessageResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                intent_scores={"error": 1.0}
            )


# Create a global NLPService instance to be imported by other modules
nlp_service = NLPService()


# Example Usage (For Testing Purposes)
if __name__ == "__main__":
    # Configure basic logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    nlp_service = NLPService()

    # Sample Inputs
    test_messages = [
        "I want to book a gardener for tomorrow",
        "What is my booking ID?",
        "cancel booking 123e4567-e89b-12d3-a456-426614174000",
        "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak.",
        "Schedule welder James Brown for Friday at 10 AM for metal work.",
        "I need electrician Bob Wilson on Monday at 9 AM.",
        "Book carpenter Emily White for Saturday at 3 PM to build a cabinet.",
        "Schedule mechanic Robert Miller on Tuesday at 5 PM for a car check.",
        "Book painter Lisa Davis for Thursday at 11 AM to repaint my living room.",
        "Schedule chef Daniel Martinez for Sunday at 7 PM for dinner.",
        "Book gardener Nancy Clark for Friday at 8 AM for landscaping.",
        "Schedule teacher Samuel Harris for Monday at 6 PM for tutoring.",
        "Book developer Laura Evans for Wednesday at 4 PM for a project.",
        "Schedule nurse Kevin Adams for Saturday at 9 AM for home care.",
        "List all bookings",
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        response = nlp_service.handle_message(msg)
        print(f"> {response.response}")
        print("Intent Classification Scores:")
        for intent, score in response.intent_scores.items():
            print(f"  - {intent}: {score:.4f}")
        print()