"""
settings.py

This module manages application configuration by loading environment
variables via Pydantic's BaseSettings. It provides a centralized
Settings object to be imported wherever configuration data is needed.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global application settings, loaded from environment variables.

    Attributes:
        ENV (str): Current environment (e.g., 'development', 'production').
        HUGGINGFACE_API_KEY (Optional[str]): API key for Hugging Face (if needed).
        MODEL_NAME (str): Default model for NLP processing with Transformers.
        APP_NAME (str): Name of the FastAPI application, shown in documentation.
        APP_VERSION (str): Version of the application, for documentation and logs.
    """

    ENV: str = Field("development", env="ENV")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    MODEL_NAME: str = Field("bert-base-uncased", env="MODEL_NAME")
    APP_NAME: str = Field("Technician Booking System", env="APP_NAME")
    APP_VERSION: str = Field("0.1.0", env="APP_VERSION")

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
