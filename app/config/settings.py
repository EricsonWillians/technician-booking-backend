from typing import List, Optional, Dict
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global application settings, loaded from environment variables.

    Attributes:
        ENV (str): Current environment (e.g., 'development', 'production').
        HUGGINGFACE_API_KEY (Optional[str]): API key for Hugging Face (if needed).
        MODEL_NAME (str): Default model for NLP processing with Transformers.
        ZERO_SHOT_MODEL_NAME (str): Model name for zero-shot classification.
        NER_MODEL_NAME (str): Model name for Named Entity Recognition.
        USE_GPU (bool): Flag to indicate whether to use GPU if available.
        LOG_LEVEL (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
        CANDIDATE_INTENTS (List[str]): List of possible intents for classification.
        INTENT_CONFIDENCE_THRESHOLD (float): Confidence threshold for intent classification.
        DEFAULT_CUSTOMER_NAME (str): Default name for customers if not provided.
        DEFAULT_PROFESSION (str): Default profession if not detected.
        PROFESSION_KEYWORDS Dict[str, List[str]]: Keywords to detect professions in user input.
        DEFAULT_START_HOUR (int): Default hour for booking start time.
        APP_NAME (str): Name of the FastAPI application, shown in documentation.
        APP_VERSION (str): Version of the application, for documentation and logs.
        HF_CACHE_DIR (str): Directory path for Hugging Face cache.
    """

    ENV: str = Field("development", env="ENV")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    MODEL_NAME: str = Field("bert-base-uncased", env="MODEL_NAME")
    MODEL_LOAD_RETRIES: int = Field(3, env="MODEL_LOAD_RETRIES")
    ZERO_SHOT_MODEL_NAME: str = Field(
        "facebook/bart-large-mnli", env="ZERO_SHOT_MODEL_NAME"
    )
    NER_MODEL_NAME: str = Field(
        "dbmdz/bert-large-cased-finetuned-conll03-english", env="NER_MODEL_NAME"
    )
    USE_GPU: bool = Field(True, env="USE_GPU")
    USE_HALF_PRECISION: bool = Field(False, env="USE_HALF_PRECISION")
    MIN_INPUT_LENGTH: int = Field(3, env="MIN_INPUT_LENGTH")
    MAX_INPUT_LENGTH: int = Field(512, env="MAX_INPUT_LENGTH")
    LOG_LEVEL: str = Field("DEBUG", env="LOG_LEVEL")
    DATA_SOURCE: str = Field("CLI", env="DATA_SOURCE")

    CANDIDATE_INTENTS: List[str] = Field(
    ["Create a booking", "Cancel a booking", "Retrieve booking details", "List all bookings"],
        env="CANDIDATE_INTENTS",
    )
    
    DEFAULT_INTENT: str = Field("unknown", env="DEFAULT_INTENT")
    DEFAULT_BOOKING_HOUR: int = Field(9, env="DEFAULT_BOOKING_HOUR")
    LAST_BOOKING_HOUR: int = Field(18, env="LAST_BOOKING_HOUR")
    
    INTENT_CONFIDENCE_THRESHOLD: float = Field(0.4, env="INTENT_CONFIDENCE_THRESHOLD")
    DEFAULT_CUSTOMER_NAME: str = Field("Anonymous Customer", env="DEFAULT_CUSTOMER_NAME")
    DEFAULT_PROFESSION: str = Field("Plumber", env="DEFAULT_PROFESSION")
   
    PROFESSION_KEYWORDS: Dict[str, List[str]] = Field(
        {
            "plumber": [
                "plumber", "pipe", "leak", "drain", "clog", "sewer", "faucet", "toilet",
                "sink", "bathroom", "water heater", "pipe fitting", "plumbing repair",
                "shower", "valve", "overflow", "gutter", "trap", "sealant"
            ],
            "welder": [
                "welder", "welding", "metal work", "fabrication", "steel", "iron",
                "metalworking", "soldering", "brazing", "metal joining", "metallurgy"
            ],
            "electrician": [
                "electrician", "electric", "wiring", "circuit", "voltage", "current",
                "breaker", "switch", "outlet", "fuse", "electrical panel", "power supply",
                "lighting", "generator", "transformer", "socket", "grounding", "installation",
                "repair", "maintenance", "electrical troubleshooting"
            ],
            "carpenter": [
                "carpenter", "woodwork", "furniture", "cabinet", "joinery", "wood",
                "table", "chair", "frame", "panel", "door", "window", "deck",
                "woodcraft", "sawing", "hammering", "wood carving", "sanding",
                "cabinetry", "shelf", "construction", "bench", "trim", "molding"
            ],
            "mechanic": [
                "mechanic", "engine", "transmission", "oil change", "tire", "brakes",
                "clutch", "vehicle repair", "automotive", "diagnostics", "car maintenance",
                "motorcycle", "truck", "fuel system", "battery", "alternator", "radiator",
                "suspension", "alignment", "exhaust", "drivetrain"
            ],
            "painter": [
                "painter", "painting", "roller", "brush", "primer", "coating", "staining",
                "wall", "ceiling", "furniture painting", "spray paint", "latex paint",
                "acrylic paint", "oil-based paint", "tape", "decorative painting", "stripping",
                "sanding", "color matching", "finish", "varnish"
            ],
            "chef": [
                "chef", "cooking", "kitchen", "recipe", "meal", "cuisine", "food",
                "baking", "grill", "sauté", "ingredients", "plating", "dish", "menu",
                "soup", "dessert", "sauce", "knife skills", "garnish", "pastry",
                "culinary", "catering", "roasting", "grilling", "chopping"
            ],
            "gardener": [
                "gardener", "gardening", "plant", "soil", "landscaping", "flower",
                "tree", "shrub", "hedge", "lawn", "mowing", "pruning", "fertilizer",
                "pest control", "mulch", "watering", "weeding", "garden design",
                "harvesting", "compost", "horticulture", "greenhouse", "irrigation"
            ],
            "teacher": [
                "teacher", "teaching", "education", "lesson", "classroom", "students",
                "learning", "curriculum", "instruction", "lecture", "assignments",
                "grading", "tutoring", "pedagogy", "lesson plan", "syllabus", "training",
                "workshop", "teaching methods", "educator", "school", "academic"
            ],
            "developer": [
                "developer", "programming", "coding", "software", "application", "web",
                "frontend", "backend", "fullstack", "framework", "API", "database",
                "testing", "deployment", "DevOps", "algorithm", "debugging", "version control",
                "integration", "JavaScript", "Python", "Java", "C++", "Ruby", "HTML", "CSS"
            ],
            "nurse": [
                "nurse", "nursing", "patient", "healthcare", "medical", "hospital",
                "clinic", "caregiver", "medication", "treatment", "vitals", "charting",
                "emergency", "surgery", "wound care", "infection control", "assisted living",
                "rehabilitation", "health assessment", "pediatric care", "geriatric care"
            ]
        },
        env="PROFESSION_KEYWORDS",
    )
    
    DEFAULT_START_HOUR: int = Field(10, env="DEFAULT_START_HOUR")
    HF_CACHE_DIR: str = Field("~/.cache/huggingface/hub/", env="HF_CACHE_DIR")  

    APP_NAME: str = Field("Technician Booking System", env="APP_NAME")
    APP_VERSION: str = Field("0.1.0", env="APP_VERSION")

    @field_validator('CANDIDATE_INTENTS', 'PROFESSION_KEYWORDS', mode='before')
    def split_comma_separated_lists(cls, v):
        """
        Splits comma-separated strings into lists.

        Args:
            v (str | List[str]): The environment variable value.

        Returns:
            List[str]: Parsed list of strings.
        """
        if isinstance(v, str):
            # Remove any surrounding brackets or quotes if present
            v = v.strip().strip('[]').strip('"').strip("'")
            # Split by comma and strip whitespace
            return [item.strip() for item in v.split(',') if item.strip()]
        return v

    class Config:
        """
        Pydantic configuration class.

        The env_file attribute points to a local .env file.
        This can be overridden for different environments or CI/CD pipelines.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a global settings instance to be imported throughout the application
settings = Settings()
