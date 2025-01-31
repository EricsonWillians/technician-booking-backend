# app/core/cli.py

"""
==========================================================
            TECHNICIAN BOOKING SYSTEM - CLI
==========================================================

Enhanced Command-Line Interface for the Technician Booking System
with professional-grade styling, consistent visual hierarchy,
and robust error handling.

Features:
- Rich, formatted output with enhanced readability
- Interactive command processing with real-time NLP parsing
- Displays intent classification scores for monitoring
- Robust error handling and logging integration
- Intuitive user experience with clear visual feedback

Author : Ericson Willians
Email  : ericsonwillians@protonmail.com
Date   : January 2025

==========================================================
"""

# app/core/cli.py

import typer
import logging
import asyncio
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.theme import Theme
from rich.padding import Padding
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import Traceback
from typing import Callable, Any, Dict, Optional
from app.services.nlp_service import NLPService
import traceback
from datetime import datetime

from app.services.nlp_service import nlp_service, MessageResponse
from app.core import initial_data

# Enhanced color theme with scientific aesthetics
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "highlight": "magenta",
    "muted": "dim white",
    "accent": "blue",
    "primary": "#00A4BD",  # Cyan-blue for primary actions
    "secondary": "#7B2CBF",  # Purple for secondary elements
    "metric_high": "#10B981",  # Green for high confidence
    "metric_medium": "#F59E0B",  # Amber for medium confidence
    "metric_low": "#EF4444",  # Red for low confidence
    "code": "grey70",  # For code-like elements
})

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

def create_metric_color(score: float) -> str:
    """Determine color based on confidence score."""
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"

def create_intent_analysis_table(intent_scores: Dict[str, float]) -> Table:
    """Create a compact intent analysis table."""
    table = Table(
        show_header=True,
        header_style="bold",
        border_style="bright_black",
        title="Intent Analysis",
        padding=(0, 1),  # Reduced padding
        min_width=50
    )
    
    table.add_column("Intent", style="dim", width=20)
    table.add_column("Conf", justify="right", width=8)
    table.add_column("Viz", width=20)
    
    for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(score * 20)  # Shorter bars
        bar = "█" * bar_length + "▒" * (20 - bar_length)
        color = create_metric_color(score)
        
        table.add_row(
            intent.replace("_", " ").title(),
            f"{score:.4f}",
            f"[{color}]{bar}[/{color}]"
        )

    return table

def create_compact_analysis_table(intent_scores: Dict[str, float], max_width: int = 40) -> Table:
    """Create an extremely compact analysis table with better horizontal space usage."""
    table = Table(
        show_header=True,
        header_style="bold blue",
        border_style="bright_black",
        padding=(0, 1),
        min_width=max_width,
        show_edge=False
    )
    
    table.add_column("Intent", style="cyan", width=15)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Analysis", width=max_width - 23)
    
    for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(score * 15)
        bar = "█" * bar_length + "▒" * (15 - bar_length)
        color = create_metric_color(score)
        
        table.add_row(
            intent.split("_")[0][:15],
            f"{score:.2f}",
            f"[{color}]{bar}[/{color}]"
        )

    return table

def display_nlp_analysis(text: str, response: Optional[MessageResponse]):
    """Display NLP analysis with minimal vertical space."""
    try:
        if response is None:
            raise ValueError("No response received from NLP service")

        # Create a single-panel layout with grid
        grid = Table.grid(padding=0)
        grid.add_column("main", justify="left")

        # Input row
        input_text = Text()
        input_text.append("Input: ", style="blue bold")
        input_text.append(text, style="cyan")
        
        # Analysis section
        if response.intent_scores:
            max_score = max(response.intent_scores.values())
            avg_score = sum(response.intent_scores.values()) / len(response.intent_scores)
            
            # Analysis header
            analysis_header = Text()
            analysis_header.append("\nNLP Analysis ", style="blue bold")
            analysis_header.append(f"[dim](max: {max_score:.2f}, avg: {avg_score:.2f})")

            # Analysis table
            analysis_table = create_compact_analysis_table(response.intent_scores)
            
            # Response text
            response_text = Text("\n")
            response_text.append("Response: ", style="blue bold")
            response_text.append(response.response, 
                              style="green" if not response.response.startswith("Error:") else "red")

            # Add all components
            grid.add_row(input_text)
            grid.add_row(analysis_header)
            grid.add_row(analysis_table)
            grid.add_row(response_text)

        # Create main panel
        panel = Panel(
            grid,
            title="NLP Analysis",
            border_style="blue",
            padding=(0, 1),
            subtitle=f"[dim]{datetime.now().strftime('%H:%M:%S')}"
        )
        
        console.print(panel)

    except Exception as e:
        display_error("Analysis Error", e)

def create_styled_panel(content: Any, title: str, style: str = "info") -> Panel:
    """
    Create a consistently styled Rich Panel containing either a string or a Rich Text object.

    If content is already a Rich Text, we use it directly.
    Otherwise, we convert content to a new Text(...).
    """
    if isinstance(content, Text):
        # It's already a Rich Text object; just use it as is
        content_text = content
    else:
        # Convert plain string (or other objects) to Rich Text
        content_text = Text(str(content), justify="left")

    return Panel(
        Padding(content_text, (1, 2)),
        title=f"[{style} bold]{title}[/{style} bold]",
        border_style=style,
        padding=(1, 2),
        title_align="center"
    )


def display_welcome():
    """Display compact welcome message."""
    console.print(Panel(
        Text.assemble(
            ("Technician Booking System ", "cyan bold"),
            ("v1.0\n", "dim"),
            ("Type ", "dim"),
            ("'help'", "cyan"),
            (" for available commands or ", "dim"),
            ("'quit'", "cyan"),
            (" to exit", "dim")
        ),
        border_style="blue",
        padding=(0, 1)
    ))


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
    details.append(f"{booking.profession}\n", style="magenta")

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


def display_error(error_title: str, error: Exception):
    """
    Display a formatted error message with traceback.
    
    Args:
        error_title: Title for the error message
        error: The exception object
    """
    layout = Layout()
    layout.split_column(
        Layout(name="error"),
        Layout(name="traceback", ratio=2)
    )

    # Create error message
    error_text = Text()
    error_text.append(f"{str(error)}\n\n", style="red")
    error_text.append("Error Type: ", style="red dim")
    error_text.append(error.__class__.__name__, style="red bold")

    error_panel = Panel(
        error_text,
        title=error_title,
        border_style="red",
        padding=(0, 1)
    )
    layout["error"].update(error_panel)

    # Get the full traceback if available
    if isinstance(error, Exception) and error.__traceback__:
        tb = Traceback.extract(
            type(error),
            error,
            traceback.extract_tb(error.__traceback__),
            show_locals=True
        )
        traceback_panel = Panel(
            Traceback(tb),
            title="Traceback",
            border_style="red dim",
            padding=(0, 1)
        )
        layout["traceback"].update(traceback_panel)

    console.print(layout)

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


def handle_command(command: str):
    """Handle command with minimal output."""
    try:
        logger.debug(f"Processing command: {command}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(description="Processing...", total=None)
            
            try:
                response = nlp_service.handle_message(command)
                if response is None:
                    raise ValueError("NLP service returned None response")
                display_nlp_analysis(command, response)
            except Exception as e:
                logger.error(f"Failed to handle message: {str(e)}")
                error_response = MessageResponse(
                    response=f"Error: {str(e)}",
                    intent_scores={}
                )
                display_nlp_analysis(command, error_response)

    except Exception as e:
        console.print(Panel(
            Text(f"Error: {str(e)}", style="red"),
            border_style="red",
            padding=(0, 1)
        ))


def async_handler(func: Callable) -> Callable:
    """
    Decorator to handle async functions in synchronous context.
    Ensures proper event loop handling across different platforms.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper

def initialize_processor() -> NLPService:
    """Initialize the language processor with error handling."""
    try:
        logger.info("Initializing language processor...")
        processor = nlp_service
        logger.info("Language processor initialized successfully")
        return processor
    except Exception as e:
        error = RuntimeError(f"Failed to initialize language processor: {str(e)}")
        display_error("Initialization Error", error)
        logger.exception("Language processor initialization failed")
        raise typer.Exit(code=1)

@app.callback(invoke_without_command=True)
@async_handler
async def main(ctx: typer.Context):
    """
    Main application entry point with proper async/sync handling.
    Manages both the async initial data loading and synchronous CLI operations.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        # Initialize the language processor
        processor = initialize_processor()
        display_welcome()

        # Handle async initial data loading
        await initial_data.load_initial_data()

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

                handle_command(command)

            except KeyboardInterrupt:
                console.print("\n")
                console.print(create_styled_panel(
                    "Session terminated by user.",
                    "Goodbye",
                    "info"
                ))
                break
            except Exception as e:
                display_error(str(e), "Unexpected Error")
                logger.exception("Unexpected error in command loop")

    except Exception as e:
        display_error(f"Fatal error: {str(e)}", "Fatal Error")
        logger.exception("Fatal error in main application")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
