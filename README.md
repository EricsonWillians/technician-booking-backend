# technician-booking-backend

A FastAPI-based system to manage technician bookings, ensuring that time slots do not overlap for the same technician. This repository demonstrates an **in-memory** reference implementation suitable for prototyping or internal demos. You can easily replace the in-memory data store with a real database in production.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Installation](#installation)  
5. [Configuration](#configuration)  
6. [Usage](#usage)  
   - [CLI Usage](#cli-usage)  
   - [API Usage](#api-usage)  
7. [Project Structure](#project-structure)  
8. [Examples](#examples)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## Overview

**technician-booking-backend** provides a platform to schedule, retrieve, and cancel technician appointments. It comes with:

- A **RESTful API** (FastAPI) that covers core CRUD operations.  
- A **Natural Language (NLP/LLM) CLI** for end-users who prefer commands like “_I want to book an electrician for tomorrow_.”

This project enforces:

1. **No overlapping bookings** for the same technician.  
2. Each booking is **1 hour** long.  
3. Basic **profession validation** (e.g., plumber, electrician, welder).  
4. Optional **Rich-enhanced** terminal experience for colorful output.

---

## Key Features

- **FastAPI CRUD Endpoints** for bookings:  
  - List all bookings  
  - Retrieve a booking by ID  
  - Create a booking (with time checks)  
  - Delete a booking  
- **NLP-based CLI** (Typer + Rich) to parse natural language commands such as:  
  - _“I want to book a plumber for tomorrow”_  
  - _“cancel booking 123”_  
- **In-Memory Data Store** for quick demos and local testing.  
- **Validation** of booking times and allowed professions.  
- **One-Hour Slot** enforcement—end time is automatically calculated.  

---

## Tech Stack

- **Python 3.10+**  
- **FastAPI** for the web framework.  
- **Typer + Rich** for CLI interactions.  
- **Poetry** for dependency management.  
- **Pydantic** for data validation.  
- **Transformers** (optional) if using an NLP model (Hugging Face) for parsing.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-organization/technician-booking-backend.git
   cd technician-booking-backend
   ```

2. **Install Poetry** (if not already installed)  
   ```bash
   # For Linux/macOS:
   curl -sSL https://install.python-poetry.org | python3 -
   
   # or check the Poetry docs for Windows options
   ```

3. **Install Dependencies**  
   ```bash
   poetry install
   ```

4. **Activate the Poetry Shell**  
   ```bash
   poetry shell
   ```

5. **Set up Environment Variables** (optional but recommended)  
   - Copy `.env.example` to `.env` and fill in your values, for example:
     ```bash
     HUGGINGFACE_API_KEY=<YOUR_TOKEN_HERE>
     MODEL_NAME="bert-base-uncased"
     ```
   - This helps manage secrets and environment-specific configurations.

---

## Configuration

- **`pyproject.toml`** handles all dependencies and project metadata.  
- **`.env`** (not committed) stores sensitive credentials like `HUGGINGFACE_API_KEY`.  
- **`app/config/settings.py`** loads environment variables via Pydantic `BaseSettings`.

---

## Usage

### CLI Usage

Run the **interactive CLI** (with NLP-based commands):

```bash
poetry run python app/core/cli.py run
```

You should see a Rich-enhanced interface. Examples of valid commands:

1. **Create a booking (NLP example)**  
   ```
   I want to book an electrician for tomorrow
   ```
   Output might be:  
   ```
   Booking created successfully!
   Booking ID: 7fefd8a2-8bbf-4183-9c2b-a6e4fd31dfaf
   ```

2. **Retrieve booking details**  
   ```
   what is my booking id?
   ```
   or  
   ```
   retrieve booking 7fefd8a2-8bbf-4183-9c2b-a6e4fd31dfaf
   ```

3. **Cancel a booking**  
   ```
   cancel booking 7fefd8a2-8bbf-4183-9c2b-a6e4fd31dfaf
   ```

4. **List all bookings**  
   ```
   list all bookings
   ```

5. **Quit**  
   ```
   quit
   ```

**Note**: The LLM commands are parsed by the placeholder function `parse_user_input` in `llm_processor.py`. You can integrate any Hugging Face or custom NLP model to handle user text more reliably.

---

### API Usage

Run the **FastAPI** server:

```bash
poetry run uvicorn app.main:app --reload
```

The application starts on `http://127.0.0.1:8000`.

1. **List All Bookings**  
   - **GET** `/bookings/`  
   - Response: `[{ "id": "...", "customer_name": "...", ... }, ... ]`

2. **Retrieve One Booking by ID**  
   - **GET** `/bookings/{booking_id}`  
   - Example: `GET /bookings/bdff5803-7c45-4fac-9848-234dfc36da06`

3. **Create a Booking**  
   - **POST** `/bookings/`  
   - JSON body:
     ```json
     {
       "customer_name": "Alice Johnson",
       "technician_name": "Mark Roberts",
       "profession": "Welder",
       "start_time": "2024-01-15T14:00:00"
     }
     ```
   - Returns a **201** status with the new booking info in JSON.

4. **Delete a Booking**  
   - **DELETE** `/bookings/{booking_id}`  
   - Returns **204** on success or **404** if not found.

You can explore all these endpoints via the **interactive docs** at:

```
http://127.0.0.1:8000/docs
```

or the ReDoc docs at:

```
http://127.0.0.1:8000/redoc
```

---

## Project Structure

```
technician-booking-backend/
├── app
│   ├── config
│   │   └── settings.py          # Loads env variables
│   ├── core
│   │   ├── cli.py               # CLI entry point
│   │   └── initial_data.py      # Seeds initial data
│   ├── main.py                  # FastAPI app factory & startup
│   ├── models
│   │   └── booking.py           # Dataclass domain model
│   ├── routers
│   │   └── bookings.py          # CRUD endpoints for bookings
│   ├── schemas
│   │   └── booking.py           # Pydantic schemas
│   ├── services
│   │   ├── booking_service.py   # Booking CRUD + validations & overlap checks
│   │   └── validation.py        # Custom validation utilities
│   └── utils
│       └── llm_processor.py     # Placeholder NLP parser
├── tests
│   ├── integration
│   │   └── test_api.py
│   └── unit
│       └── test_services.py
├── pyproject.toml               # Poetry config & dependencies
├── README.md
└── .env.example                 # Example env file
```

---

## Examples

**Creating a Booking via API**  
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Bob", "technician_name": "James", "profession": "Plumber", "start_time": "2025-02-10T10:00:00"}' \
  http://127.0.0.1:8000/bookings/
```
Response:
```json
{
  "id": "efa1b6ba-983e-4aaa-8500-5fcdbf45aef9",
  "customer_name": "Bob",
  "technician_name": "James",
  "profession": "Plumber",
  "start_time": "2025-02-10T10:00:00",
  "end_time": "2025-02-10T11:00:00"
}
```

**CLI “Natural Language” Command**  
```
> "I want to book a plumber for tomorrow"

[CLI Response]
Booking created successfully!
Booking ID: 7c1af07e-ecdf-4a0b-90f4-edae3a76d402
```

---

## Contributing

1. **Fork** the repo and create a new branch:  
   ```bash
   git checkout -b feature/my-new-feature
   ```  
2. **Make Your Changes** with appropriate tests.  
3. **Open a Pull Request** describing your changes thoroughly.

We appreciate suggestions, bug reports, and improvements to make this project more robust and developer-friendly.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.