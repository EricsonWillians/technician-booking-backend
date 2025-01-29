"""
main.py

This module defines the FastAPI application entry point for
the Technician Booking System. It includes:
  - FastAPI initialization
  - Metadata and versioning
  - Router inclusion
  - Startup event to load initial data
"""

from fastapi import FastAPI
from app.config.settings import settings
from app.core.initial_data import load_initial_data
from app.routers.bookings import router as bookings_router


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
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        """
        Define startup logic for the FastAPI application.

        This event hook can be used to:
          - Initialize or seed in-memory data
          - Establish database connections
          - Perform any other one-time setup
        """
        # If load_initial_data is synchronous, remove 'async' and 'await'.
        await load_initial_data()

    # Register all routers here
    app.include_router(bookings_router, prefix="/bookings", tags=["Bookings"])

    return app


# Create the global app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
