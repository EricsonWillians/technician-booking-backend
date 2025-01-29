# Technician Booking System — README

## Overview

This repository contains the **Technician Booking System**, an **NLP-driven** backend for handling bookings (create, cancel, retrieve) with automated date/time parsing, entity recognition, and robust intent classification. The project demonstrates how a multi-pipeline architecture (Zero-Shot, NER, and optional date/time LLM) can form a cohesive solution for natural language–based scheduling and technician dispatch.

Developed as part of an **internal company assignment**, I've decided to open-source it for educational and demonstration purposes. The design supports **in-memory** storage by default but can be easily modified to use external databases such as **MongoDB** or **PostgreSQL** with **SQLAlchemy** integration.

## Architecture

1. **Core CLI**  
   - **`app/core/cli.py`** defines a Rich/typer-based command-line interface to interact with the booking logic.
   - Presents user-friendly commands (e.g. “Book a plumber tomorrow at 2 PM,” “Show booking details for {id}”).

2. **NLP Service**  
   - **`app/services/nlp_service.py`** orchestrates **three** Hugging Face Transformers pipelines:
     - **Zero-shot** classification → Intent detection
     - **NER** (Named Entity Recognition) → Technician names, customer names, partial time tokens
     - **Text2Text LLM** → More advanced date/time parsing (e.g. “Friday at 3pm,” “tomorrow,” “in 2 weeks”)
   - Combines rule-based fallback with pipeline-based extraction.  
   - Maps recognized “Create a booking,” “Retrieve booking details,” etc., to actual booking logic calls.

3. **Booking Services**  
   - **`app/services/booking_service.py`** manages the booking CRUD operations (create, list, retrieve, cancel).
   - Uses an **in-memory** Python dictionary for data storage by default. Switch out the logic to connect with an RDBMS like Postgres or a NoSQL store like MongoDB.

4. **Models & Schemas**  
   - **`app/models/booking.py`** and **`app/schemas/booking.py`** define Pydantic data models.  
   - **`BookingCreate`** is used on creation, while **`BookingResponse`** or an internal `Booking` object describes the fully stored data.

5. **Settings & Environment**  
   - **`app/config/settings.py`** loads environment variables (using [pydantic-settings](https://github.com/pydantic/pydantic-settings)) for model names, ports, and other system-wide defaults.

## Running Locally

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/your-org/technician-booking-backend
cd technician-booking-backend
poetry install
```
> Ensure you have [Poetry](https://python-poetry.org/docs/) installed for dependency management.

### 2. Configure Environment Variables
Create or modify `.env` in the root directory:
```bash
ENV=development
LOG_LEVEL=DEBUG

# Name or path for Hugging Face models
ZERO_SHOT_MODEL_NAME=facebook/bart-large-mnli
NER_MODEL_NAME=dbmdz/bert-large-cased-finetuned-conll03-english
DATE_TIME_MODEL_NAME=google/flan-t5-large

# If you want to override default huggingface cache, set:
HF_CACHE_DIR=~/.cache/huggingface/hub/

# GPU usage
USE_GPU=True
```
> You can specify custom model names for zero-shot, NER, date/time LLM. These run **locally** unless they are not cached and must be downloaded from Hugging Face.

### 3. Launch the CLI
```bash
poetry run python -m app.core.cli
```
This starts an interactive REPL-like environment where you can type commands such as:
```
Book a plumber for tomorrow at 2pm
List all bookings
Show booking details for {booking_id}
Cancel booking {booking_id}
```

---

## Booking Data Storage

### In-Memory Storage

The default `booking_service.py` uses a Python dictionary for storing Bookings:

```python
in_memory_bookings_db: Dict[str, Booking] = {}
```
This is ideal for **demonstration** or testing but **not** persistent once the application restarts.

### Switching to MongoDB or PostgreSQL

Replace or extend `booking_service.py` with your DB logic. For instance:

- **SQLAlchemy** for Postgres:
  ```python
  from sqlalchemy.orm import Session
  # ...
  def create_booking(booking_data: BookingCreate, db: Session) -> Booking:
      # insert into Postgres via SQLAlchemy
      ...
  ```
- **PyMongo** for MongoDB:
  ```python
  from pymongo import MongoClient
  # ...
  db = MongoClient()["booking_database"]
  def create_booking(booking_data: BookingCreate) -> Booking:
      db.bookings.insert_one(booking_data.dict())
      ...
  ```

The rest of the application logic remains the same, as the CLI calls the same `create_booking_from_llm(...)`, etc.

---

## Model Switching

The following pipelines can be changed by adjusting environment variables in `.env`:

- **ZERO_SHOT_MODEL_NAME**: e.g. `"facebook/bart-large-mnli"`, or any zero-shot classification model on Hugging Face.
- **NER_MODEL_NAME**: e.g. `"dbmdz/bert-large-cased-finetuned-conll03-english"`, or your custom NER model.
- **DATE_TIME_MODEL_NAME**: e.g. `"google/flan-t5-large"`, or a smaller/flan-t5-base if memory is a concern.

The system automatically loads them via the `_init_pipeline_with_retry` method. If you prefer GPU usage, set `USE_GPU=True`.

---

## Project Folder Structure

```
technician-booking-backend/
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── core/
│   │   ├── cli.py          # CLI for user interactions
│   │   └── initial_data.py # Optional initial data load
│   ├── models/
│   │   └── booking.py      # Domain model or DB model
│   ├── schemas/
│   │   └── booking.py      # Pydantic schemas for requests/responses
│   ├── services/
│   │   ├── booking_service.py # In-memory CRUD logic
│   │   └── nlp_service.py     # Orchestrates 3 Hugging Face pipelines
│   └── utils/                # Additional utility modules if needed
├── tests/                     # Unit tests
├── poetry.lock
├── pyproject.toml             # Poetry config (Dependencies listed)
└── README.md                  # This file
```

---

## Running the CLI

Inside the repo:

```bash
poetry run python -m app.core.cli
```
You should see:

```
╭──────────────────────────────────────────────────────────────────────────────── Technician Booking System ─────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                            │
│  Welcome to the Technician Booking System                                                                                                                                                  │
│                                                                                                                                                                                            │
│  Available Commands:                                                                                                                                                                       │
│  └─ Book a service:                                                                                                                                                                        │
│     • Book a plumber for tomorrow at 2pm                                                                                                                                                   │
│     • Book an electrician named John for next Monday                                                                                                                                       │
│  └─ Manage bookings:                                                                                                                                                                       │
│     • Show booking details for {booking_id}                                                                                                                                                │
│     • Cancel booking {booking_id}                                                                                                                                                          │
│     • List all bookings                                                                                                                                                                    │
│                                                                                                                                                                                            │
│  Type 'quit', 'exit', or 'q' to stop.                                                                                                                                                      │
│                                                                                                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Enter command:
```
**Start** typing commands, e.g.:

```
Book a gardener for next friday at 1pm
Show booking details for b56a726a-4ec2-4681-8fd4-b51ff2c19c19
List all bookings
Cancel booking b56a726a-4ec2-4681-8fd4-b51ff2c19c19
```

The system will parse your command, create or retrieve bookings from the in-memory store, and display the results nicely in a Rich-formatted panel or table.

---

## Further Notes

- **Assignment Origin**: This project began as an internal company assignment exploring advanced NLP pipelines. I'm sharing it as an educational resource.
- **Open-Source**: Licensed under the MIT License for wide usage and adaptation.
- **Additional Tools**:
  - `pytest` for unit testing.
  - `mypy`, `black`, `flake8`, `isort` for code quality & formatting.
- **Production Considerations**:
  1. Switch in-memory DB to a real database.
  2. Possibly wrap the CLI in a web server (FastAPI) for a multi-user environment.
  3. Performance tune or reduce large model usage if you scale to high concurrency.

---

## Conclusion

This **Technician Booking Backend** exemplifies a robust approach to **NLP-driven** scheduling, from ephemeral storage to advanced “text2text” date/time logic. It’s **highly customizable** via environment variables, allowing easy swapping of **Hugging Face** models or DB solutions. We hope it serves as both a reference and a stepping stone for real-world, intelligent booking systems.

Feel free to contribute or adapt to your environment—**pull requests** are welcome!