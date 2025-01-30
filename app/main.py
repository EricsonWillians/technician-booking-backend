"""
==========================================================
          TECHNICIAN BOOKING SYSTEM - API ENTRY POINT
==========================================================

  Defines the FastAPI application entry point for the  
  Technician Booking System.

  Features:
  - FastAPI initialization with CORS setup  
  - Metadata and versioning configuration  
  - Router inclusion for managing bookings  
  - Startup event to load initial data  
  - General error handling with structured responses  
  - Health check endpoint for monitoring  

  Author : Ericson Willians  
  Email  : ericsonwillians@protonmail.com  
  Date   : January 2025  

==========================================================
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from app.config.settings import settings
from app.core.initial_data import load_initial_data
from app.routers.bookings import router as bookings_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        FastAPI: A fully configured FastAPI application.
    """
    app = FastAPI(
        title="Technician Booking System",
        description=(
            "A system to schedule, retrieve, and cancel technician "
            "bookings, featuring an NLP-based console interface."
        ),
        version=settings.APP_VERSION,  # e.g., "0.1.0"
        docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production if desired
        redoc_url="/redoc" if settings.DEBUG else None,  # Disable redoc in production if desired
        openapi_url="/openapi.json" if settings.DEBUG else None  # Control OpenAPI schema visibility
    )

    # CORS Configuration
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,  # List of allowed origins or ["*"]
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS}")

    # Register Event Handlers
    @app.on_event("startup")
    async def startup_event() -> None:
        """
        Define startup logic for the FastAPI application.

        This event hook can be used to:
          - Initialize or seed in-memory data
          - Establish database connections
          - Perform any other one-time setup
        """
        try:
            await load_initial_data()
            logger.info("Initial data loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load initial data during startup.")
            raise e  # Optionally, you can handle the exception as needed

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """
        Define shutdown logic for the FastAPI application.

        This event hook can be used to:
          - Close database connections
          - Clean up resources
        """
        logger.info("Shutting down the Technician Booking System.")

    # Register Routers
    app.include_router(bookings_router, prefix="/bookings", tags=["Bookings"])
    logger.info("Bookings router included with prefix '/bookings'.")

    # Health Check Endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict:
        """
        Health check endpoint to verify that the application is running.

        Returns:
            dict: A simple status message.
        """
        return {"status": "healthy"}

    # General Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """
        Handle HTTP exceptions and return JSON responses.

        Args:
            request (Request): The incoming request.
            exc (HTTPException): The exception raised.

        Returns:
            JSONResponse: A JSON response with error details.
        """
        logger.warning(f"HTTPException: {exc.detail} (status code: {exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle unexpected exceptions and return a generic error response.

        Args:
            request (Request): The incoming request.
            exc (Exception): The exception raised.

        Returns:
            JSONResponse: A JSON response with a generic error message.
        """
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An unexpected error occurred. Please try again later."},
        )

    return app


# Create the global app instance
app = create_app()

# Run the application if this file is executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,  # e.g., "0.0.0.0"
        port=settings.PORT,  # e.g., 8000
        reload=settings.DEBUG,  # Enable reload in development
        log_level="debug" if settings.DEBUG else "warning",
    )
