# app/main.py

"""
==========================================================
          TECHNICIAN BOOKING SYSTEM - API ENTRY POINT
==========================================================

Production-grade FastAPI application entry point with:
- Robust error handling
- CORS configuration
- Health monitoring
- Graceful shutdown
- Structured logging
- Environment-based configuration

Author : Ericson Willians  
Email  : ericsonwillians@protonmail.com  
Date   : January 2025  

==========================================================
"""

import logging
import sys
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.config.settings import settings
from app.core.initial_data import load_initial_data
from app.routers.bookings import router as bookings_router, APIResponse, ErrorDetail

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("technician_booking_api")

def create_app() -> FastAPI:
    """Initialize and configure the FastAPI application."""
    app = FastAPI(
        title="Technician Booking System",
        description="Advanced booking system with NLP capabilities for technician scheduling.",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None
    )

    # CORS setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register startup event
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize application state on startup."""
        try:
            logger.info("Starting Technician Booking System...")
            await load_initial_data()
            logger.info("System initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}", exc_info=True)
            sys.exit(1)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Clean up resources on shutdown."""
        logger.info("Shutting down Technician Booking System...")
        # Add any cleanup tasks here if needed

    # Register routers
    app.include_router(
        bookings_router,
        prefix="/api/v1/bookings",
        tags=["Bookings"]
    )

    # Health check endpoint
    @app.get(
        "/health",
        tags=["Health"],
        response_model=Dict[str, Any],
        summary="System health check"
    )
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint with system status information.
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": app.version,
            "environment": settings.ENV
        }

    # Enhanced error handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        errors = []
        for error in exc.errors():
            errors.append({
                "loc": " -> ".join(str(loc) for loc in error["loc"]),
                "msg": error["msg"],
                "type": error["type"]
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=APIResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Request validation failed",
                    details={"errors": errors},
                    timestamp=datetime.now()
                )
            ).dict()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        error_id = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.error(
            f"Unhandled exception {error_id}: {str(exc)}",
            exc_info=True,
            extra={
                "error_id": error_id,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                success=False,
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    details={
                        "error_id": error_id,
                        "support_message": "Please contact support with this error ID"
                    },
                    timestamp=datetime.now()
                )
            ).dict()
        )

    return app

# Create application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )