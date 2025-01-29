"""
Enhanced CLI implementation for the Technician Booking System with professional-grade
styling, consistent visual hierarchy, and robust error handling.
"""

import sys
import typer
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.theme import Theme
from rich.style import Style
from rich.padding import Padding
from typing import Callable, Optional, Dict, Any

from app.utils.llm_processor import llm_processor, LLMProcessor
from app.services import booking_service
from app.schemas.booking import BookingCreate
from app.config.settings import settings

# Define a consistent color theme for the application
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "highlight": "magenta",
    "muted": "dim white",
    "accent": "blue",
    "primary": "cyan",
    "secondary": "blue",
})

# Initialize core components
app = typer.Typer()
console = Console(theme=custom_theme)

# Configure logging with enhanced formatting
logger = logging.getLogger("technician_booking_cli")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=False
    )
    rich_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

# Suppress external logging noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def create_styled_panel(content: str, title: str, style: str = "info") -> Panel:
    """Create a consistently styled panel with proper padding and borders."""
    return Panel(
        Padding(Text(content, justify="left"), (1, 2)),
        title=f"[{style} bold]{title}[/{style} bold]",
        border_style=style,
        padding=(1, 2),
        title_align="center"
    )

def display_welcome():
    """Display a properly formatted welcome message using Rich markup."""
    welcome_text = Text()
    welcome_text.append("Welcome to the Technician Booking System\n\n", style="bold")
    welcome_text.append("Available Commands:\n", style="blue")
    
    # Book a service section
    welcome_text.append("└─ ", style="dim")
    welcome_text.append("Book a service:\n", style="cyan")
    welcome_text.append("   • Book a plumber for tomorrow at 2pm\n")
    welcome_text.append("   • Book an electrician named John for next Monday\n")
    
    # Manage bookings section
    welcome_text.append("└─ ", style="dim")
    welcome_text.append("Manage bookings:\n", style="cyan")
    welcome_text.append("   • Show booking details for {booking_id}\n")
    welcome_text.append("   • Cancel booking {booking_id}\n")
    welcome_text.append("   • List all bookings\n\n")
    
    # Exit instructions
    welcome_text.append("Type 'quit', 'exit', or 'q' to stop.", style="dim")
    
    panel = Panel(
        welcome_text,
        title="Technician Booking System",
        border_style="cyan",
        padding=(1, 2),
        title_align="center"
    )
    console.print(panel)

def format_booking_details(booking, include_separator: bool = True) -> Text:
    """Format booking details using Text object for proper styling."""
    details = Text()
    
    details.append("Booking ID: ", style="bold")
    details.append(f"{booking.id}\n", style="blue")
    
    details.append("Customer: ", style="bold")
    details.append(f"{booking.customer_name}\n")
    
    details.append("Technician: ", style="bold")
    details.append(f"{booking.technician_name}\n")
    
    details.append("Profession: ", style="bold")
    details.append(f"{booking.profession}\n", style="cyan")
    
    details.append("Start Time: ", style="bold")
    details.append(f"{booking.start_time.strftime('%Y-%m-%d %I:%M %p')}\n")
    
    details.append("End Time: ", style="bold")
    details.append(booking.end_time.strftime('%Y-%m-%d %I:%M %p'))
    
    if include_separator:
        details.append("\n" + "─" * 50, style="dim")
    
    return details

def display_success_message(title: str, content: Text):
    """Display success message using proper Rich styling."""
    panel = Panel(
        content,
        title=title,
        border_style="green",
        padding=(1, 2),
        title_align="center"
    )
    console.print(panel)

def display_error_message(message: str, title: str = "Error"):
    """Display error message using proper Rich styling."""
    error_text = Text()
    error_text.append(message, style="red")
    
    panel = Panel(
        error_text,
        title=title,
        border_style="red",
        padding=(1, 2),
        title_align="center"
    )
    console.print(panel)
    
def display_warning_message(message: str, title: str = "Warning"):
    """Display warning message using proper Rich styling."""
    warning_text = Text()
    warning_text.append(message, style="yellow")
    
    panel = Panel(
        warning_text,
        title=title,
        border_style="yellow",
        padding=(1, 2),
        title_align="center"
    )
    console.print(panel)

def handle_booking_creation(processor: LLMProcessor, command: str):
    """Handle booking creation with enhanced visual feedback."""
    try:
        logger.debug(f"Processing booking creation: {command}")
        parsed = processor.parse_user_input(command)

        if parsed.intent != "create_booking":
            display_error_message("Unable to process booking request")
            return

        booking_data = parsed.data
        customer_name = booking_data.get("customer_name", settings.DEFAULT_CUSTOMER_NAME)
        technician_name = booking_data.get("technician_name", "Available Technician")
        profession = booking_data.get("profession", settings.DEFAULT_PROFESSION)
        start_time = booking_data.get("start_time")

        if not start_time:
            display_error_message("Start time is required for booking")
            return

        booking_create = BookingCreate(
            customer_name=customer_name,
            technician_name=technician_name,
            profession=profession,
            start_time=start_time
        )

        booking = booking_service.create_booking(booking_create)
        success_message = format_booking_details(booking, include_separator=False)
        display_success_message("Booking Confirmation", success_message)
        logger.info(f"Booking created: {booking.id}")

    except ValueError as ve:
        display_error_message(str(ve))
    except Exception as e:
        display_error_message(f"Unexpected error: {str(e)}")
        logger.exception("Booking creation failed")

def handle_cancel_booking(processor: LLMProcessor, command: str):
    """Handle booking cancellation with enhanced visual feedback."""
    try:
        logger.debug(f"Processing cancellation request: {command}")
        parsed = processor.parse_user_input(command)

        if parsed.intent != "cancel_booking":
            display_error_message("Unable to process cancellation request")
            return

        booking_id = parsed.data.get("booking_id")
        if not booking_id:
            display_error_message("Please provide a booking ID", "Missing Information")
            return

        if booking_service.cancel_booking(booking_id):
            success_message = f"Booking [accent]{booking_id}[/accent] has been cancelled"
            display_success_message("Cancellation Successful", success_message)
            logger.info(f"Booking cancelled: {booking_id}")
        else:
            display_error_message(f"Booking {booking_id} not found", "Cancellation Failed")

    except ValueError as ve:
        display_error_message(str(ve))
    except Exception as e:
        display_error_message(f"Failed to cancel booking: {str(e)}")
        logger.exception("Booking cancellation failed")

def handle_retrieve_booking(processor: LLMProcessor, command: str):
    """Handle booking retrieval with enhanced visual feedback."""
    try:
        logger.debug(f"Processing retrieval request: {command}")
        parsed = processor.parse_user_input(command)

        if parsed.intent != "retrieve_booking":
            display_error_message("Unable to process retrieval request")
            return

        booking_id = parsed.data.get("booking_id")
        if not booking_id:
            display_error_message("Please provide a booking ID", "Missing Information")
            return

        booking = booking_service.get_booking_by_id(booking_id)
        if booking:
            details = format_booking_details(booking, include_separator=False)
            console.print(create_styled_panel(details, "Booking Details", "info"))
            logger.info(f"Retrieved booking: {booking_id}")
        else:
            display_error_message(f"Booking {booking_id} not found", "Retrieval Failed")

    except ValueError as ve:
        display_error_message(str(ve))
    except Exception as e:
        display_error_message(f"Failed to retrieve booking: {str(e)}")
        logger.exception("Booking retrieval failed")

def handle_list_bookings():
    """Display bookings with enhanced visual formatting."""
    try:
        bookings = booking_service.get_all_bookings()
        if not bookings:
            display_warning_message("No current bookings found", "Bookings")
            return

        # Create table for bookings
        table = Table(
            title="Current Bookings",
            show_header=True,
            header_style="bold accent",
            border_style="accent",
            title_style="bold accent",
            padding=(0, 1),
            show_edge=True
        )

        # Define columns
        columns = [
            ("ID", "accent"),
            ("Customer", "info"),
            ("Technician", "info"),
            ("Profession", "primary"),
            ("Start Time", "secondary"),
            ("End Time", "secondary")
        ]
        
        for col_name, style in columns:
            table.add_column(col_name, style=style)

        # Add booking rows
        for booking in bookings:
            table.add_row(
                booking.id,
                booking.customer_name,
                booking.technician_name,
                booking.profession,
                booking.start_time.strftime('%Y-%m-%d %I:%M %p'),
                booking.end_time.strftime('%Y-%m-%d %I:%M %p')
            )

        console.print("\n")
        console.print(table)
        console.print("\n")
        logger.info(f"Displayed {len(bookings)} bookings")

    except Exception as e:
        display_error_message(f"Failed to list bookings: {str(e)}")
        logger.exception("Failed to list bookings")

def handle_unknown_command(command: str):
    """Handle unknown commands with helpful feedback."""
    display_error_message(
        f"Unrecognized command: '{command}'\nType 'help' for available commands.",
        "Unknown Command"
    )
    logger.warning(f"Unknown command received: {command}")

def handle_command(processor: LLMProcessor, command: str):
    """
    Processes user commands by routing to appropriate handlers based on intent.

    Args:
        processor (LLMProcessor): The LLMProcessor instance.
        command (str): The user command input.
    """
    try:
        logger.debug(f"Processing command: {command}")
        parsed = processor.parse_user_input(command)  # Updated usage

        intent_handlers = {
            "create_booking": lambda cmd: handle_booking_creation(processor, cmd),
            "list_bookings": lambda cmd: handle_list_bookings(),
            "cancel_booking": lambda cmd: handle_cancel_booking(processor, cmd),
            "retrieve_booking": lambda cmd: handle_retrieve_booking(processor, cmd),
        }

        handler = intent_handlers.get(parsed.intent, handle_unknown_command)
        handler(command)

    except Exception as e:
        console.print(f"[bold red]Error processing command:[/bold red] {str(e)}")
        logger.exception(f"Error processing command '{command}': {e}")

def initialize_processor() -> LLMProcessor:
    """Initialize the language processor with error handling."""
    try:
        logger.info("Initializing language processor...")
        processor = llm_processor
        logger.info("Language processor initialized successfully")
        return processor
    except Exception as e:
        display_error_message(f"Failed to initialize language processor: {str(e)}")
        logger.exception("Language processor initialization failed")
        raise typer.Exit(code=1)

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Main application entry point with enhanced error handling."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        processor = initialize_processor()
        display_welcome()

        while True:
            try:
                command = console.input("\n[accent]Enter command:[/accent] ").strip()

                if not command:
                    continue

                if command.lower() in ["quit", "exit", "q"]:
                    console.print(create_styled_panel(
                        "Thank you for using the Technician Booking System!",
                        "Goodbye",
                        "info"
                    ))
                    break

                handle_command(processor, command)

            except KeyboardInterrupt:
                console.print("\n")
                console.print(create_styled_panel(
                    "Session terminated by user",
                    "Goodbye",
                    "info"
                ))
                break
            except Exception as e:
                display_error_message(str(e))
                logger.exception("Unexpected error in command loop")

    except Exception as e:
        display_error_message(f"Fatal error: {str(e)}")
        logger.exception("Fatal error in main application")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()