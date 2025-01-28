"""
cli.py

Enhanced CLI implementation for the Technician Booking System with improved
error handling, better model management, and more reliable command processing.
"""

import sys
import typer
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler

# Configure logging to suppress transformer warnings
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize core components
console = Console()
app = typer.Typer()

def create_processor():
    """Create and initialize the LLM processor with proper error handling."""
    try:
        from app.utils.llm_processor import parse_user_input
        return parse_user_input
    except Exception as e:
        console.print(f"[bold red]Error initializing language processor:[/bold red] {str(e)}")
        sys.exit(1)

def display_welcome():
    """Display the welcome message and usage instructions."""
    console.print(
        Panel(
            Text(
                "Welcome to the Technician Booking System\n\n"
                "Example Commands:\n"
                "- 'Book a plumber for tomorrow at 2pm'\n"
                "- 'Book an electrician named John for next Monday'\n"
                "- 'Show booking details for {booking_id}'\n"
                "- 'Cancel booking {booking_id}'\n"
                "- 'List all bookings'\n\n"
                "Type 'quit' or 'exit' to stop.",
                justify="center"
            ),
            title="[bold cyan]Technician Booking System[/bold cyan]",
            border_style="cyan",
        )
    )

def handle_booking_creation(processor, command: str):
    """Handle the booking creation process with proper error handling."""
    try:
        # Import services here to avoid circular imports
        from app.services import booking_service
        
        parsed = processor(command)
        if parsed.intent != "create_booking":
            console.print("[bold red]Error:[/bold red] Unable to process booking request.")
            return

        # Ensure required data is present
        booking_data = parsed.data
        booking_data.setdefault('customer_name', 'Anonymous Customer')
        booking_data.setdefault('technician_name', 'Available Technician')

        # Create the booking
        booking = booking_service.create_booking_from_llm(booking_data)
        
        # Display success message
        console.print(
            Panel(
                Text(
                    f"Booking Created Successfully!\n\n"
                    f"Booking ID: {booking.id}\n"
                    f"Customer: {booking.customer_name}\n"
                    f"Technician: {booking.technician_name}\n"
                    f"Profession: {booking.profession}\n"
                    f"Start Time: {booking.start_time}",
                    justify="left"
                ),
                title="[bold green]✓ Booking Confirmation[/bold green]",
                border_style="green",
            )
        )
    except ValueError as e:
        console.print(f"[bold red]Invalid booking request:[/bold red] {str(e)}")
    except Exception as e:
        console.print(f"[bold red]Error creating booking:[/bold red] {str(e)}")

def handle_command(processor, command: str):
    """Process user commands with enhanced error handling."""
    from app.services import booking_service
    
    try:
        parsed = processor(command)
        
        if parsed.intent == "create_booking":
            handle_booking_creation(processor, command)
            
        elif parsed.intent == "list_bookings":
            bookings = booking_service.get_all_bookings()
            if not bookings:
                console.print(
                    Panel(
                        Text(
                            "No bookings found in the system.\n"
                            "Try creating a booking using commands like:\n"
                            "- 'Book a plumber for tomorrow'\n"
                            "- 'Schedule an electrician for Monday at 2PM'",
                            justify="center"
                        ),
                        title="[bold yellow]System Status[/bold yellow]",
                        border_style="yellow"
                    )
                )
                return
                
            # Display bookings table
            table = Table(
                title="Current Bookings",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("ID", style="cyan")
            table.add_column("Customer", style="green")
            table.add_column("Technician", style="blue")
            table.add_column("Profession", style="yellow")
            table.add_column("Time", style="magenta")
            
            for booking in bookings:
                table.add_row(
                    booking.id,
                    booking.customer_name,
                    booking.technician_name,
                    booking.profession,
                    str(booking.start_time)
                )
            console.print(table)
            
        elif parsed.intent in ["cancel_booking", "retrieve_booking"]:
            if "booking_id" not in parsed.data:
                console.print(
                    "[bold red]Error:[/bold red] Please provide a booking ID. "
                    "Example: 'Cancel booking 12345'"
                )
                return
                
            booking_id = parsed.data["booking_id"]
            
            if parsed.intent == "cancel_booking":
                if booking_service.cancel_booking(booking_id):
                    console.print(f"[bold green]✓ Booking {booking_id} cancelled successfully.[/bold green]")
                else:
                    console.print(f"[bold red]Error:[/bold red] Booking {booking_id} not found.")
            else:
                booking = booking_service.get_booking_by_id(booking_id)
                if booking:
                    console.print(
                        Panel(
                            Text(
                                f"Booking ID: {booking.id}\n"
                                f"Customer: {booking.customer_name}\n"
                                f"Technician: {booking.technician_name}\n"
                                f"Profession: {booking.profession}\n"
                                f"Start Time: {booking.start_time}\n"
                                f"End Time: {booking.end_time}",
                                justify="left"
                            ),
                            title="[bold cyan]Booking Details[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                else:
                    console.print(f"[bold red]Error:[/bold red] Booking {booking_id} not found.")
        
        else:
            console.print("[bold yellow]Tip:[/bold yellow] Try using one of the example commands shown in the welcome message.")
            
    except Exception as e:
        console.print(f"[bold red]Error processing command:[/bold red] {str(e)}")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Launch the interactive booking system interface."""
    if ctx.invoked_subcommand is not None:
        return
        
    try:
        # Initialize processor
        processor = create_processor()
        
        # Display welcome message
        display_welcome()
        
        # Main interaction loop
        while True:
            try:
                command = console.input("\n[bold green]Enter command:[/bold green] ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ["quit", "exit", "q"]:
                    console.print("\n[bold cyan]Thank you for using the Technician Booking System. Goodbye![/bold cyan]")
                    break
                    
                handle_command(processor, command)
                
            except KeyboardInterrupt:
                console.print("\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()