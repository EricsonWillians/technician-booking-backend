# app/config/settings.py

from typing import List, Optional, Dict, Union
from pydantic_settings import BaseSettings
from pydantic import Field, AnyHttpUrl, field_validator
from zoneinfo import ZoneInfo
from tzlocal import get_localzone_name
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Global application settings, loaded from environment variables.
    """

    # Timezone Configuration
    TIMEZONE: Optional[str] = Field(None, env="TIMEZONE")

    @property
    def TIMEZONE_OBJ(self) -> ZoneInfo:
        if self.TIMEZONE:
            try:
                return ZoneInfo(self.TIMEZONE)
            except Exception as e:
                logger.error(f"Invalid timezone '{self.TIMEZONE}'. Defaulting to UTC. Error: {e}")
        try:
            # Use tzlocal to get the system's local timezone
            local_timezone = get_localzone_name()
            return ZoneInfo(local_timezone)
        except Exception as e:
            # Default to UTC if system timezone cannot be determined
            logger.warning(f"Failed to determine system timezone. Defaulting to UTC. Error: {e}")
            return ZoneInfo("UTC")

    @field_validator("TIMEZONE")
    def validate_timezone(cls, v):
        """
        Validates and sets the timezone configuration.

        Priority Handling:
        1. Environment Variable (Preferred)
        2. System's Local Timezone (Fallback)
        3. Default to UTC (Final Fallback)
        """
        if v:
            try:
                ZoneInfo(v)  # Validate timezone
                logger.debug(f"Using TIMEZONE from environment: {v}")
                return v
            except Exception as e:
                logger.warning(
                    f"Invalid TIMEZONE '{v}' provided. Attempting to detect system timezone. Error: {e}"
                )
        try:
            # Use tzlocal to get the system's local timezone
            local_timezone = get_localzone_name()
            logger.debug(f"Detected system timezone: {local_timezone}")
            return local_timezone
        except Exception as e:
            logger.warning(
                f"Could not detect system timezone, defaulting to UTC. Error: {e}"
            )
            return "UTC"

    # General Application Settings
    ENV: str = Field("development", env="ENV")
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    DEBUG: bool = Field(True, env="DEBUG")
    CORS_ORIGINS: Union[List[AnyHttpUrl], str] = Field("*", env="CORS_ORIGINS")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")

    # NLP Model Settings
    ZERO_SHOT_MODEL_NAME: str = Field(
        "facebook/bart-large-mnli", env="ZERO_SHOT_MODEL_NAME"
    )
    NER_MODEL_NAME: str = Field("dslim/bert-base-NER", env="NER_MODEL_NAME")

    # Hardware and Performance Settings
    USE_GPU: bool = Field(True, env="USE_GPU")

    # Logging Settings
    LOG_LEVEL: str = Field("DEBUG", env="LOG_LEVEL")

    # Data Source Settings
    DATA_SOURCE: str = Field("CLI", env="DATA_SOURCE")

    # Intent Classification Settings
    CANDIDATE_INTENTS: Dict[str, str] = Field(
        {
            "Create a booking": "create_booking",
            "Cancel a booking": "cancel_booking",
            "Retrieve a booking": "retrieve_booking_details",
            "List all bookings": "list_all_bookings",
        },
        env="CANDIDATE_INTENTS",
    )

    DEFAULT_INTENT: str = Field("unknown", env="DEFAULT_INTENT")
    INTENT_CONFIDENCE_THRESHOLD: float = Field(0.3, env="INTENT_CONFIDENCE_THRESHOLD")

    # Booking Defaults
    DEFAULT_BOOKING_HOUR: int = Field(9, env="DEFAULT_BOOKING_HOUR")
    LAST_BOOKING_HOUR: int = Field(18, env="LAST_BOOKING_HOUR")

    # Default Values for Missing Data
    DEFAULT_CUSTOMER_NAME: str = Field(
        "Anonymous Customer", env="DEFAULT_CUSTOMER_NAME"
    )
    DEFAULT_PROFESSION: str = Field("Plumber", env="DEFAULT_PROFESSION")

    # Profession Detection Keywords
    PROFESSION_KEYWORDS: Dict[str, List[str]] = Field(
        {
            "plumber": [
                "plumber",
                "pipe",
                "leak",
                "drain",
                "clog",
                "sewer",
                "faucet",
                "toilet",
                "sink",
                "bathroom",
                "water heater",
                "pipe fitting",
                "plumbing repair",
                "shower",
                "valve",
                "overflow",
                "gutter",
                "trap",
                "sealant",
            ],
            "welder": [
                "welder",
                "welding",
                "metal work",
                "fabrication",
                "steel",
                "iron",
                "metalworking",
                "soldering",
                "brazing",
                "metal joining",
                "metallurgy",
            ],
            "electrician": [
                "electrician",
                "electric",
                "wiring",
                "circuit",
                "voltage",
                "current",
                "breaker",
                "switch",
                "outlet",
                "fuse",
                "electrical panel",
                "power supply",
                "lighting",
                "generator",
                "transformer",
                "socket",
                "grounding",
                "installation",
                "repair",
                "maintenance",
                "electrical troubleshooting",
            ],
            "carpenter": [
                "carpenter",
                "woodwork",
                "furniture",
                "cabinet",
                "joinery",
                "wood",
                "table",
                "chair",
                "frame",
                "panel",
                "door",
                "window",
                "deck",
                "woodcraft",
                "sawing",
                "hammering",
                "wood carving",
                "sanding",
                "cabinetry",
                "shelf",
                "construction",
                "bench",
                "trim",
                "molding",
            ],
            "mechanic": [
                "mechanic",
                "engine",
                "transmission",
                "oil change",
                "tire",
                "brakes",
                "clutch",
                "vehicle repair",
                "automotive",
                "diagnostics",
                "car maintenance",
                "motorcycle",
                "truck",
                "fuel system",
                "battery",
                "alternator",
                "radiator",
                "suspension",
                "alignment",
                "exhaust",
                "drivetrain",
            ],
            "painter": [
                "painter",
                "painting",
                "roller",
                "brush",
                "primer",
                "coating",
                "staining",
                "wall",
                "ceiling",
                "furniture painting",
                "spray paint",
                "latex paint",
                "acrylic paint",
                "oil-based paint",
                "tape",
                "decorative painting",
                "stripping",
                "sanding",
                "color matching",
                "finish",
                "varnish",
            ],
            "chef": [
                "chef",
                "cooking",
                "kitchen",
                "recipe",
                "meal",
                "cuisine",
                "food",
                "baking",
                "grill",
                "sauté",
                "ingredients",
                "plating",
                "dish",
                "menu",
                "soup",
                "dessert",
                "sauce",
                "knife skills",
                "garnish",
                "pastry",
                "culinary",
                "catering",
                "roasting",
                "grilling",
                "chopping",
            ],
            "gardener": [
                "gardener",
                "gardening",
                "plant",
                "soil",
                "landscaping",
                "flower",
                "tree",
                "shrub",
                "hedge",
                "lawn",
                "mowing",
                "pruning",
                "fertilizer",
                "pest control",
                "mulch",
                "watering",
                "weeding",
                "garden design",
                "harvesting",
                "compost",
                "horticulture",
                "greenhouse",
                "irrigation",
            ],
            "teacher": [
                "teacher",
                "teaching",
                "education",
                "lesson",
                "classroom",
                "students",
                "learning",
                "curriculum",
                "instruction",
                "lecture",
                "assignments",
                "grading",
                "tutoring",
                "pedagogy",
                "lesson plan",
                "syllabus",
                "training",
                "workshop",
                "teaching methods",
                "educator",
                "school",
                "academic",
            ],
            "developer": [
                "developer",
                "programming",
                "coding",
                "software",
                "application",
                "web",
                "frontend",
                "backend",
                "fullstack",
                "framework",
                "API",
                "database",
                "testing",
                "deployment",
                "DevOps",
                "algorithm",
                "debugging",
                "version control",
                "integration",
                "JavaScript",
                "Python",
                "Java",
                "C++",
                "Ruby",
                "HTML",
                "CSS",
            ],
            "nurse": [
                "nurse",
                "nursing",
                "patient",
                "healthcare",
                "medical",
                "hospital",
                "clinic",
                "caregiver",
                "medication",
                "treatment",
                "vitals",
                "charting",
                "emergency",
                "surgery",
                "wound care",
                "infection control",
                "assisted living",
                "rehabilitation",
                "health assessment",
                "pediatric care",
                "geriatric care",
            ],
        },
        env="PROFESSION_KEYWORDS",
    )

    # Hugging Face Cache Directory
    HF_CACHE_DIR: str = Field("~/.cache/huggingface/hub/", env="HF_CACHE_DIR")

    class Config:
        """
        Pydantic configuration class.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a global settings instance to be imported throughout the application
settings = Settings()
