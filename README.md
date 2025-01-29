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
.
├── app
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core
│   │   ├── cli.py
│   │   ├── initial_data.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── booking.py
│   │   └── __init__.py
│   ├── routers
│   │   ├── bookings.py
│   │   └── __init__.py
│   ├── schemas
│   │   ├── booking.py
│   │   └── __init__.py
│   ├── services
│   │   ├── booking_service.py
│   │   ├── __init__.py
│   │   ├── nlp_service.py
│   │   └── validation.py
│   └── utils
│       └── __init__.py
├── init_project.sh
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
└── tests
    ├── __init__.py
    └── unit
        ├── __init__.py
        └── test_intent_classification.py      
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

## FastAPI HTTP API

In addition to the CLI, the Technician Booking System can be run as a **FastAPI** web service, exposing booking operations via REST endpoints.

### **1. Running the FastAPI App**

1. **Ensure dependencies installed** (see general instructions above):
   ```bash
   poetry install
   ```
2. **Launch** the FastAPI server (using Uvicorn) from the project root:
   ```bash
   poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. **Access** the HTTP API in your browser or via tools like `curl`, Postman, or httpie:
   - **Base URL**: `http://localhost:8000`

### **2. Available Endpoints**

The main routes live in **`app.routers.bookings`**, which are included under `"/bookings"`. Summaries:

1. **GET** `/<base>/bookings`  
   - **List** all bookings in memory.
   - Returns an array of booking objects or empty array `[]`.

2. **GET** `/<base>/bookings/{booking_id}`  
   - **Retrieve** a single booking by its unique ID (UUID or numeric).
   - Returns `404 Not Found` if the booking does not exist.

3. **POST** `/<base>/bookings`  
   - **Create** a new booking from JSON data matching `BookingCreate`.
   - Returns a `BookingResponse` with `201 Created` on success, `400 Bad Request` if validation fails.

4. **DELETE** `/<base>/bookings/{booking_id}`  
   - **Cancel** or remove a booking by its ID.
   - Returns `204 No Content` if the booking was deleted, or `404` if not found.

> These endpoints map directly to the logic in `booking_service.py` (in-memory by default, but easily adapted to databases).

### **3. Exploring the API Documentation**

FastAPI automatically generates **OpenAPI** documentation, accessible via your browser:

- **Swagger UI**:  
  ```
  http://localhost:8000/docs
  ```
  Graphical interface to test each endpoint (list bookings, create a booking, etc.).

- **ReDoc**:  
  ```
  http://localhost:8000/redoc
  ```
  Alternative documentation with a clean, single-page design.

Here you’ll see:

- **Schema definitions** for `BookingCreate` & `BookingResponse`.
- **Endpoint** descriptions (HTTP methods, request/response bodies).
- **Example calls** and standard error codes (400, 404, 500).

### **4. Customizing for Production**

1. **Database**:
   - Replace or augment the **in-memory** logic in `booking_service.py` with real DB connectors (MongoDB, PostgreSQL).
   - Keep the same endpoints—only the backend storage changes.

2. **Authentication / Authorization**:
   - Add FastAPI dependencies (e.g. `fastapi-security` or OAuth2) if you need secure endpoints.

3. **Deployment**:
   - Containerize with Docker or deploy to cloud providers (e.g., AWS, Azure) using standard FastAPI best practices.

---

## **cURL Examples**

### 1. **List All Bookings**
```bash
curl -X GET http://127.0.0.1:8000/bookings/ \
     -H "Accept: application/json"
```
**Response (HTTP 200):**
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "customer_name": "John Doe",
    "technician_name": "Alice Tech",
    "profession": "Plumber",
    "start_time": "2025-01-30T10:00:00",
    "end_time": "2025-01-30T11:00:00"
  }
]
```
*(If empty, returns `[]`.)*

---

### 2. **Retrieve a Specific Booking**
```bash
curl -X GET http://127.0.0.1:8000/bookings/123e4567-e89b-12d3-a456-426614174000 \
     -H "Accept: application/json"
```
**Response (HTTP 200):**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "customer_name": "John Doe",
  "technician_name": "Alice Tech",
  "profession": "Plumber",
  "start_time": "2025-01-30T10:00:00",
  "end_time": "2025-01-30T11:00:00"
}
```

**If not found (HTTP 404):**
```json
{
  "detail": "Booking with ID 123e4567-e89b-12d3-a456-426614174000 not found."
}
```

---

### 3. **Create a New Booking**
```bash
curl -X POST http://127.0.0.1:8000/bookings/ \
     -H "Content-Type: application/json" \
     -d '{
           "customer_name": "Maria Garcia",
           "technician_name": "Mike Davis",
           "profession": "Plumber",
           "start_time": "2025-02-10T09:00:00"
         }'
```
**Response (HTTP 201):**
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "customer_name": "Maria Garcia",
  "technician_name": "Mike Davis",
  "profession": "Plumber",
  "start_time": "2025-02-10T09:00:00",
  "end_time": "2025-02-10T10:00:00"
}
```
*(Note the auto-generated `id` and the one-hour offset `end_time`.)*

**If validation fails (HTTP 400):**
```json
{
  "detail": "Profession must be one of: Plumber, Electrician, Gardener, ...etc"
}
```

---

### 4. **Cancel (Delete) a Booking**
```bash
curl -X DELETE http://127.0.0.1:8000/bookings/3fa85f64-5717-4562-b3fc-2c963f66afa6
```
**Response (HTTP 204)** *(No Content)*

**If not found (HTTP 404):**
```json
{
  "detail": "Booking with ID 3fa85f64-5717-4562-b3fc-2c963f66afa6 not found."
}
```

---

## **Notes**
- **Accept Header**:  
  You can include `-H "Accept: application/json"` to explicitly request JSON responses (FastAPI defaults to JSON).
- **Content-Type**:  
  When creating a booking (POST), specify `-H "Content-Type: application/json"` with the JSON body.

These **cURL** examples provide a straightforward way to test your **FastAPI** endpoints quickly and confirm that your **in-memory** booking system or future database integration is functioning as expected.

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