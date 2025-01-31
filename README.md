# Technician Booking System

## Overview

An advanced Natural Language Processing (NLP) driven scheduling system that implements sophisticated booking operations through multiple machine learning pipelines. The system features robust intent classification, named entity recognition, and temporal expression parsing.

### Key Features

- Multi-pipeline NLP architecture with confidence scoring
- Advanced temporal expression parsing with timezone awareness
- Scientific visualization of intent classification metrics
- Production-grade error handling and validation
- Comprehensive automated testing suite

## Technical Architecture

### 1. Core NLP Service (`app/services/nlp_service.py`)
- **Intent Classification Pipeline**
  - Zero-shot classification with confidence metrics
  - Pattern-based intent reinforcement
  - Normalized confidence scoring
  - Intent hierarchy analysis

- **Entity Recognition Pipeline**
  - Professional designation extraction
  - Temporal entity recognition
  - Named entity extraction for technicians
  - Contextual entity validation

- **Temporal Processing**
  - Timezone-aware datetime handling
  - Relative time expression parsing
  - Business hours enforcement
  - Booking conflict detection

### 2. Booking Service (`app/services/booking_service.py`)
- **CRUD Operations**
  - Atomic booking creation
  - Concurrent access handling
  - Validation middleware
  - Transaction integrity

- **Data Validation**
  - Professional qualification verification
  - Temporal constraint checking
  - Booking conflict resolution
  - Entity relationship validation

### 3. CLI Interface (`app/core/cli.py`)
- **Scientific Output**
  - Intent confidence visualization
  - Real-time analysis metrics
  - Color-coded confidence indicators
  - Structured response formatting

### 4. API Layer (`app/routers/bookings.py`)
- **RESTful Endpoints**
  - Standardized response structure
  - Comprehensive error handling
  - Request validation
  - Rate limiting support

## Lessons Learned

The development of the **Technician Booking System** has been a profound journey of exploration, experimentation, and refinement in applying Natural Language Processing (NLP) techniques to real-world scheduling challenges. This section delineates the critical lessons learned, emphasizing the evolution from initial complex approaches to the adoption of more streamlined and effective methodologies.

### 1. **Importance of Aligning Model Selection with Task Requirements**

#### **Initial Approach: Overcomplicated Model Utilization**

At the outset, the objective was to create a highly sophisticated NLP service capable of interpreting and extracting detailed temporal and contextual information from user inputs. To achieve this, multiple models were experimented with, including:

- **Text-to-Text Models:** Utilized models such as `google/flan-t5-large` and `google/flan-t5-xxl` to perform advanced date and time extraction through prompt engineering.
- **Feature Extraction with Embeddings:** Explored models focused on extracting features via embeddings to capture semantic nuances.

**Challenges Encountered:**

- **Prompt Engineering Complexity:** Crafting effective prompts for text-to-text models proved to be an arduous and time-consuming process. The endeavor often resulted in "prompt hell," where significant time investment yielded inconsistent and unreliable extraction of temporal data.
  
  *Example Prompt:*  
  ```
  "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
  ```
  
  Extracting the relative time ("Wednesday at 2 PM") required intricate prompt designs that were not only time-consuming but also failed to generalize across varied user inputs.

- **Model Limitations:** The attempt to build a "Swiss clock agent" — an agent capable of handling intricate scheduling nuances — was hampered by the inherent limitations of the selected text-to-text models. These models struggled with reliably parsing relative time expressions and contextual information necessary for accurate scheduling.

- **Overengineering Risks:** The pursuit of a versatile agent introduced unnecessary complexity, making the system harder to maintain and less efficient without delivering proportional benefits in performance.

### 2. **Realization and Transition to Simpler, Task-Specific Techniques**

Through iterative experimentation and performance analysis, it became evident that the complexity of the initial approach was disproportionate to the system's actual requirements. This realization prompted a strategic pivot towards more streamlined and effective methodologies.

#### **Adoption of Named Entity Recognition (NER) and Zero-Shot Classification**

- **Named Entity Recognition (NER):** Transitioning to NER allowed for precise extraction of essential entities such as professions, technician names, dates, and times. Utilizing Hugging Face's `dslim/bert-base-NER` model facilitated reliable entity extraction without the overhead of complex prompt engineering.

- **Zero-Shot Classification:** Implementing zero-shot classification provided an efficient means to determine user intent by categorizing inputs into predefined intents (e.g., `create_booking`, `cancel_booking`, `query_booking`). This approach obviated the need for extensive labeled datasets and leveraged descriptive intent labels to enhance classification accuracy.

**Benefits Realized:**

- **Efficiency Gains:** Simplifying the model architecture significantly reduced development time and computational overhead, enabling a more focused approach to refining core functionalities.

- **Enhanced Reliability:** NER and zero-shot classification demonstrated consistent performance in extracting relevant entities and accurately classifying intents, thereby improving the system's overall reliability.

- **Maintainability:** A streamlined approach with clear model responsibilities enhanced the system's maintainability and scalability, facilitating easier updates and integrations.

### 3. **Effectiveness of Combining Rule-Based Techniques with Machine Learning Models**

The evolution of the `nlp_service.py` module underscores the efficacy of integrating rule-based methods with machine learning models to achieve superior performance in domain-specific applications.

#### **Key Implementations:**

- **Intent Patterns:** Incorporating regex-based intent patterns enabled preliminary intent detection through pattern matching, serving as a first-pass filter before invoking zero-shot classification. This hybrid approach leveraged the strengths of both rule-based and machine learning techniques.

- **Enhanced Intent Descriptions:** Developing enriched intent descriptions provided nuanced context for zero-shot classification, improving the model's ability to discern between similar intents based on descriptive labels.

- **Sophisticated Entity Extraction:** Implementing comprehensive regex patterns for professions and booking IDs, coupled with advanced datetime extraction logic, ensured accurate and context-aware entity extraction.

**Outcome:**

These implementations demonstrated that combining rule-based techniques with machine learning models can significantly enhance entity extraction and intent classification accuracy, particularly in specialized domains like technician booking systems.

### 4. **Avoiding Overengineering: Emphasizing Simplicity and Task-Specific Solutions**

One of the paramount lessons was recognizing the pitfalls of overengineering — the pursuit of overly complex solutions when simpler alternatives suffice.

#### **Strategic Takeaways:**

- **Simplicity Enhances Performance:** Embracing simpler models and methodologies not only accelerates development but also often results in better performance due to reduced complexity.

- **Task-Specific Model Selection:** Opting for models specifically tailored to the task (e.g., NER for entity extraction) rather than general-purpose text-to-text models ensures higher accuracy and efficiency.

- **Resource Optimization:** Simplified architectures demand fewer computational resources, enabling more sustainable and scalable system deployments.

### 5. **Iterative Development and Continuous Refinement**

The project's evolution highlighted the significance of an iterative development process, where continuous testing and evaluation informed strategic pivots and optimizations.

#### **Lessons Highlighted:**

- **Experimentation Encourages Innovation:** Allowing room for experimentation with different models and techniques fostered innovation and led to the discovery of more effective solutions.

- **Feedback-Driven Refinement:** Regularly assessing model performance against real-world scenarios provided actionable insights, driving refinements that enhanced system capabilities.

- **Comprehensive Documentation and Logging:** Detailed logging within the `nlp_service.py` facilitated debugging and performance monitoring, ensuring that lessons learned were systematically captured and leveraged for ongoing improvements.

### 6. **Evolution of `nlp_service.py`: From Complexity to Elegance**

The transition from the initial to the current version of `nlp_service.py` embodies the lessons learned throughout the project's lifecycle.

#### **Initial Version Highlights:**

- **Multi-Pipeline Complexity:** The early implementation featured a multi-pipeline approach, integrating zero-shot classification, NER, and text-to-text models for advanced date/time interpretation.
  
- **Advanced Feature Extraction:** Attempts to utilize feature extraction models with embeddings aimed at capturing deeper semantic relationships but added layers of complexity without commensurate benefits.

- **Robust Error Handling:** While comprehensive, the initial error handling mechanisms were intertwined with the complex pipeline architecture, making maintenance challenging.

#### **Refined Version Highlights:**

- **Streamlined Pipelines:** The current implementation focuses on zero-shot classification and NER, eliminating the reliance on text-to-text models for temporal data extraction. This simplification enhances reliability and reduces computational demands.

- **Enhanced Intent Classification:** By integrating regex-based intent patterns alongside zero-shot classification, the system achieves more accurate and context-aware intent determination without extensive prompt engineering.

- **Sophisticated Entity Extraction:** The refined entity extraction process leverages both NER and regex patterns for professions and booking IDs, ensuring precise and contextually relevant data extraction.

- **Improved Maintainability:** The modular design of the current `nlp_service.py`, with clear separation of concerns and robust logging, facilitates easier maintenance and scalability.

### 7. **Conclusion**

The development of the **Technician Booking System** serves as a compelling case study in the effective application of NLP techniques within a specialized domain. The journey from an overcomplicated, multi-model approach to a streamlined, task-specific methodology underscores the critical importance of aligning model selection with task requirements. By prioritizing simplicity, efficiency, and reliability, the project not only overcame initial challenges but also established a robust foundation for future enhancements.

These lessons advocate for a thoughtful, pragmatic approach to NLP model selection and system design, emphasizing the balance between complexity and functionality to achieve optimal performance and maintainability.

## System Requirements

```bash
Python >= 3.10
Poetry (Dependency Management)
```

## Installation

```bash
# Clone repository
git clone https://github.com/ericsonwillians/technician-booking-backend

# Install dependencies
cd technician-booking-backend
poetry install

# Configure environment
cp .env.example .env
```

## Environment Configuration

```bash
# .env configuration
ENV=development
LOG_LEVEL=DEBUG
TIMEZONE=UTC

# NLP Models
ZERO_SHOT_MODEL_NAME=facebook/bart-large-mnli
NER_MODEL_NAME=dslim/bert-base-NER

# System Settings
DEFAULT_BOOKING_HOUR=9
LAST_BOOKING_HOUR=17
```

# Usage

## HTTP API

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The base URL for all endpoints is:

```
http://localhost:8000/api/v1/bookings
```

---

## API Usage Examples

This section provides detailed and formatted examples on how to interact with the **Technician Booking System API** using `cURL`. Each example demonstrates how to perform common operations such as listing bookings, retrieving booking details, creating a new booking, cancelling a booking, and processing natural language commands.

---

### 1. List All Bookings

**Endpoint:**

```
GET /api/v1/bookings/
```

**Description:**

Retrieves a list of all existing bookings.

**cURL Command:**

```bash
curl -X GET "http://localhost:8000/api/v1/bookings/" \
     -H "Accept: application/json"
```

**Sample Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "87963f8a-9915-4e1d-9eda-629a3ae948dc",
      "technician_name": "Alice Smith",
      "profession": "Plumber",
      "start_time": "2025-02-01T14:00:00Z",
      "end_time": "2025-02-01T15:00:00Z"
    },
    {
      "id": "4321a2b4-f16a-4cfb-9e47-5206c02a0ca8",
      "technician_name": "Bob Johnson",
      "profession": "Electrician",
      "start_time": "2025-02-02T10:00:00Z",
      "end_time": "2025-02-02T11:00:00Z"
    },
    {
      "id": "838d3646-855e-46e1-a1be-843f4ebb5f77",
      "technician_name": "Griselda Dickson",
      "profession": "Welder",
      "start_time": "2025-02-03T09:00:00Z",
      "end_time": "2025-02-03T10:00:00Z"
    },
    {
      "id": "13806ace-540b-464f-8087-87b40c50b37f",
      "technician_name": "John Do Alice Smith",
      "profession": "Plumber",
      "start_time": "2025-02-01T14:00:00Z",
      "end_time": "2025-02-01T15:00:00Z"
    },
    {
      "id": "43919cc7-01db-4bee-a818-d08d0d01d7a0",
      "technician_name": "Karen",
      "profession": "Electrician",
      "start_time": "2025-02-01T09:00:00Z",
      "end_time": "2025-02-01T10:00:00Z"
    }
  ],
  "error": null,
  "metadata": {
    "total_count": 5,
    "timestamp": "2025-01-31T12:03:47Z"
  }
}
```

**Sample Response (No Bookings Found):**

```json
{
  "success": true,
  "data": [],
  "error": null,
  "metadata": {
    "total_count": 0,
    "timestamp": "2025-01-31T12:03:47Z"
  }
}
```

**Sample Response (Error):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKINGS_RETRIEVAL_FAILED",
    "message": "Failed to retrieve bookings.",
    "details": null,
    "timestamp": "2025-01-31T12:03:47Z"
  },
  "metadata": {}
}
```

---

### 2. Retrieve Booking Details

**Endpoint:**

```
GET /api/v1/bookings/{booking_id}
```

**Description:**

Retrieves detailed information about a specific booking identified by `booking_id`.

**cURL Command:**

```bash
curl -X GET "http://localhost:8000/api/v1/bookings/87963f8a-9915-4e1d-9eda-629a3ae948dc" \
     -H "Accept: application/json"
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "id": "87963f8a-9915-4e1d-9eda-629a3ae948dc",
    "technician_name": "Alice Smith",
    "profession": "Plumber",
    "start_time": "2025-02-01T14:00:00Z",
    "end_time": "2025-02-01T15:00:00Z"
  },
  "error": null,
  "metadata": {
    "retrieved_at": "2025-01-31T12:03:47Z"
  }
}
```

**Sample Response (Booking Not Found):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKING_NOT_FOUND",
    "message": "Booking 87963f8a-9915-4e1d-9eda-629a3ae948dc not found.",
    "details": null,
    "timestamp": "2025-01-31T12:03:47Z"
  },
  "metadata": {}
}
```

---

### 3. Create a New Booking

**Endpoint:**

```
POST /api/v1/bookings/
```

**Description:**

Creates a new booking with the provided details. Note that `customer_name` is handled internally and does not need to be provided in the request.

**Request Body:**

```json
{
  "technician_name": "Alice Smith",
  "profession": "Plumber",
  "start_time": "2025-02-01T14:00:00Z"
}
```

**cURL Command:**

```bash
curl -X POST "http://localhost:8000/api/v1/bookings/" \
     -H "Content-Type: application/json" \
     -d '{
           "technician_name": "Alice Smith",
           "profession": "Plumber",
           "start_time": "2025-02-01T14:00:00Z"
         }'
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "id": "13806ace-540b-464f-8087-87b40c50b37f",
    "technician_name": "Alice Smith",
    "profession": "Plumber",
    "start_time": "2025-02-01T14:00:00Z",
    "end_time": "2025-02-01T15:00:00Z"
  },
  "error": null,
  "metadata": {
    "created_at": "2025-01-31T14:40:00Z",
    "booking_duration": "1 hour"
  }
}
```

**Sample Response (Validation Error - Missing Profession):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKING_CREATION_FAILED",
    "message": "Profession is required for booking creation.",
    "details": {
      "provided_data": {
        "technician_name": "Alice Smith",
        "start_time": "2025-02-01T14:00:00Z"
      }
    },
    "timestamp": "2025-01-31T14:40:00Z"
  },
  "metadata": {}
}
```

**Sample Response (Validation Error - Invalid Start Time):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKING_CREATION_FAILED",
    "message": "Start time must be in the future.",
    "details": {
      "provided_data": {
        "technician_name": "Alice Smith",
        "profession": "Plumber",
        "start_time": "2025-01-30T14:00:00Z"
      }
    },
    "timestamp": "2025-01-31T14:40:00Z"
  },
  "metadata": {}
}
```

---

### 4. Cancel a Booking

**Endpoint:**

```
DELETE /api/v1/bookings/{booking_id}
```

**Description:**

Cancels an existing booking identified by `booking_id`.

**cURL Command:**

```bash
curl -X DELETE "http://localhost:8000/api/v1/bookings/123e4567-e89b-12d3-a456-426614174000" \
     -H "Accept: application/json"
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "booking_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "cancelled"
  },
  "error": null,
  "metadata": {
    "cancelled_at": "2025-01-31T14:45:00Z"
  }
}
```

**Sample Response (Booking Not Found):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKING_NOT_FOUND",
    "message": "Booking 123e4567-e89b-12d3-a456-426614174000 not found.",
    "details": null,
    "timestamp": "2025-01-31T14:45:00Z"
  },
  "metadata": {}
}
```

---

### 5. Process a Natural Language Command

**Endpoint:**

```
POST /api/v1/bookings/commands
```

**Description:**

Processes a natural language command to perform booking operations such as creating, querying, or cancelling bookings.

**Request Body:**

```json
{
  "message": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
}
```

**cURL Command:**

```bash
curl -X POST "http://localhost:8000/api/v1/bookings/commands" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
         }'
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "success": true,
    "intent": "create_booking",
    "message": "Booking confirmed for Wednesday at 02:00 PM with Mike Johnson (ID: 123e4567-e89b-12d3-a456-426614174000)",
    "analysis": [
      {
        "intent": "create_booking",
        "confidence": 0.92,
        "assessment": "high"
      },
      {
        "intent": "query_booking",
        "confidence": 0.05,
        "assessment": "low"
      },
      {
        "intent": "cancel_booking",
        "confidence": 0.03,
        "assessment": "low"
      },
      {
        "intent": "list_bookings",
        "confidence": 0.00,
        "assessment": "low"
      }
    ],
    "booking": {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "technician_name": "Mike Johnson",
      "profession": "Plumber",
      "start_time": "2025-02-05T14:00:00Z",
      "end_time": "2025-02-05T15:00:00Z"
    },
    "metadata": {
      "processed_at": "2025-01-31T14:50:00Z",
      "processing_time_ms": 150
    }
  },
  "error": null,
  "metadata": {}
}
```

**Sample Response (Low Confidence):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "COMMAND_PROCESSING_FAILED",
    "message": "Failed to process command",
    "details": {
      "original_command": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
    },
    "timestamp": "2025-01-31T14:50:00Z"
  },
  "metadata": {}
}
```

**Sample Response (Unknown Intent):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "COMMAND_PROCESSING_FAILED",
    "message": "I couldn't understand that request. Could you please rephrase it?",
    "details": {},
    "timestamp": "2025-01-31T15:05:00Z"
  },
  "metadata": {}
}
```

---

### 4. Cancel a Booking

**Endpoint:**

```
DELETE /api/v1/bookings/{booking_id}
```

**Description:**

Cancels an existing booking identified by `booking_id`.

**cURL Command:**

```bash
curl -X DELETE "http://localhost:8000/api/v1/bookings/123e4567-e89b-12d3-a456-426614174000" \
     -H "Accept: application/json"
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "booking_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "cancelled"
  },
  "error": null,
  "metadata": {
    "cancelled_at": "2025-01-31T14:45:00Z"
  }
}
```

**Sample Response (Booking Not Found):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "BOOKING_NOT_FOUND",
    "message": "Booking 123e4567-e89b-12d3-a456-426614174000 not found",
    "details": null,
    "timestamp": "2025-01-31T14:45:00Z"
  },
  "metadata": {}
}
```

---

### 5. Process a Natural Language Command

**Endpoint:**

```
POST /api/v1/bookings/commands
```

**Description:**

Processes a natural language command to perform booking operations such as creating, querying, or cancelling bookings.

**Request Body:**

```json
{
  "message": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
}
```

**cURL Command:**

```bash
curl -X POST "http://localhost:8000/api/v1/bookings/commands" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
         }'
```

**Sample Response (Success):**

```json
{
  "success": true,
  "data": {
    "success": true,
    "intent": "create_booking",
    "message": "Booking confirmed for Wednesday at 2:00 PM with Mike Johnson (ID: 123e4567-e89b-12d3-a456-426614174000)",
    "analysis": [
      {
        "intent": "create_booking",
        "confidence": 0.92,
        "assessment": "high"
      },
      {
        "intent": "query_booking",
        "confidence": 0.05,
        "assessment": "low"
      },
      {
        "intent": "cancel_booking",
        "confidence": 0.03,
        "assessment": "low"
      },
      {
        "intent": "list_bookings",
        "confidence": 0.00,
        "assessment": "low"
      }
    ],
    "booking": {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "customer_name": "Anonymous Customer",
      "technician_name": "Mike Johnson",
      "profession": "Plumber",
      "start_time": "2025-02-05T14:00:00Z",
      "end_time": "2025-02-05T15:00:00Z"
    },
    "metadata": {
      "processed_at": "2025-01-31T14:50:00Z",
      "processing_time_ms": 150
    }
  },
  "error": null,
  "metadata": {}
}
```

**Sample Response (Low Confidence):**

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "COMMAND_PROCESSING_FAILED",
    "message": "Failed to process command",
    "details": {
      "original_command": "Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak."
    },
    "timestamp": "2025-01-31T14:50:00Z"
  },
  "metadata": {}
}
```

---

## CLI Usage Examples

Welcome to the **Technician Booking System CLI**! This section provides detailed and professionally formatted examples on how to interact with the system using the Command-Line Interface (CLI). Leveraging the power of `typer` and `rich`, the CLI offers an intuitive and visually appealing experience for managing technician bookings directly from your terminal.

---

### Starting the CLI

**Description:**

Launches the Technician Booking System CLI, providing an interactive interface for managing bookings.

**Command:**

```bash
poetry run python -m app.core.cli
```

**Sample Output:**

```
┌─────────────────────────────────────────────┐
│          Technician Booking System          │
│               v1.0 - January 2025           │
└─────────────────────────────────────────────┘

Enter command:
```

---

### Available Commands

The CLI provides the following primary commands:

1. **List All Bookings**
2. **Retrieve Booking Details**
3. **Create a New Booking**
4. **Cancel a Booking**
5. **Process a Natural Language Command**
6. **Exit the CLI**

---

### 1. List All Bookings

**Description:**

Retrieves and displays a list of all existing bookings in a structured and readable format.

**Command:**

```
list all bookings
```

**Sample Output (Success):**

```
┌─────────────────────────────────────────────┐
│                All Bookings                 │
├─────────────────────────────────────────────┤
│ Total Bookings: 5                           │
├─────────────────────────────────────────────┤
│ 1. ID: f0d55292-1d26-48ff-ae2f-e7a9838a5d60 │
│    Technician: Nicolas Woollett             │
│    Profession: Plumber                      │
│    Start Time: 2022-10-15 10:00 AM          │
├─────────────────────────────────────────────┤
│ 2. ID: 07111419-edbf-4f0e-9c19-b93095eadef4 │
│    Technician: Franky Flay                  │
│    Profession: Electrician                  │
│    Start Time: 2022-10-16 06:00 PM          │
├─────────────────────────────────────────────┤
│ 3. ID: 838d3646-855e-46e1-a1be-843f4ebb5f77 │
│    Technician: Griselda Dickson             │
│    Profession: Welder                       │
│    Start Time: 2022-10-18 11:00 AM          │
├─────────────────────────────────────────────┤
│ 4. ID: 13806ace-540b-464f-8087-87b40c50b37f │
│    Technician: John Do Alice Smith          │
│    Profession: Plumber                      │
│    Start Time: 2025-02-01 02:00 PM          │
├─────────────────────────────────────────────┤
│ 5. ID: 43919cc7-01db-4bee-a818-d08d0d01d7a0 │
│    Technician: Karen                        │
│    Profession: Electrician                  │
│    Start Time: 2025-02-01 09:00 AM          │
└─────────────────────────────────────────────┘
```

**Sample Output (No Bookings Found):**

```
┌─────────────────────────────────────────────┐
│                All Bookings                 │
├─────────────────────────────────────────────┤
│ Total Bookings: 0                           │
├─────────────────────────────────────────────┤
│ You have no bookings at the moment.         │
└─────────────────────────────────────────────┘
```

**Sample Output (Error):**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Failed to retrieve bookings. Please try again. 
└─────────────────────────────────────────────┘
```

---

### 2. Retrieve Booking Details

**Description:**

Fetches and displays detailed information about a specific booking using its unique `booking_id`.

**Command:**

```
get booking details 87963f8a-9915-4e1d-9eda-629a3ae948dc
```

**Sample Output (Success):**

```
┌─────────────────────────────────────────────┐
│            Booking Details                  │
├─────────────────────────────────────────────┤
│ Booking ID: 87963f8a-9915-4e1d-9eda-629a3ae948dc 
│                                             │
│ Technician: Alice Smith                     │
│ Profession: Plumber                         │
│ Start Time: 2025-02-01 02:00 PM             │
│ End Time: 2025-02-01 03:00 PM               │
└─────────────────────────────────────────────┘
```

**Sample Output (Booking Not Found):**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Booking with ID 87963f8a-9915-4e1d-9eda-629a3ae948dc not found. 
└─────────────────────────────────────────────┘
```

---

### 3. Create a New Booking

**Description:**

Creates a new booking with the specified details using natural language commands.

**Command Examples:**

- **Plumber:**
  
  ```
  Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak.
  ```
  
- **Welder:**
  
  ```
  Schedule welder James Brown for Friday at 10 AM for metal work.
  ```
  
- **Electrician:**
  
  ```
  I need electrician Bob Wilson on Monday at 9 AM.
  ```
  
- **Carpenter:**
  
  ```
  Book carpenter Emily White for Saturday at 3 PM to build a cabinet.
  ```
  
- **Mechanic:**
  
  ```
  Schedule mechanic Robert Miller on Tuesday at 5 PM for a car check.
  ```
  
- **Painter:**
  
  ```
  Book painter Lisa Davis for Thursday at 11 AM to repaint my living room.
  ```
  
- **Chef:**
  
  ```
  Schedule chef Daniel Martinez for Sunday at 7 PM for dinner.
  ```
  
- **Gardener:**
  
  ```
  Book gardener Nancy Clark for Friday at 8 AM for landscaping.
  ```
  
- **Teacher:**
  
  ```
  Schedule teacher Samuel Harris for Monday at 6 PM for tutoring.
  ```
  
- **Developer:**
  
  ```
  Book developer Laura Evans for Wednesday at 4 PM for a project.
  ```
  
- **Nurse:**
  
  ```
  Schedule nurse Kevin Adams for Saturday at 9 AM for home care.
  ```

**Sample Output (Success):**

```
┌─────────────────────────────────────────────┐
│           Booking Created Successfully      │
├─────────────────────────────────────────────┤
│ ID: 13806ace-540b-464f-8087-87b40c50b37f    │
│ Technician: John Do Alice Smith             │
│ Profession: Plumber                         │
│ Start Time: 2025-02-01 02:00 PM             │
│ End Time: 2025-02-01 03:00 PM               │
├─────────────────────────────────────────────┤
│ Metadata:                                   │
│ - Created At: 2025-01-31T14:40:00Z          │
│ - Booking Duration: 1 hour                  │
└─────────────────────────────────────────────┘
```

**Sample Output (Validation Error - Missing Profession):**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Profession is required for booking creation.│
│                                             │
│ Provided Data:                              │
│ - Technician: Alice Smith                   │
│ - Start Time: 2025-02-01T14:00:00Z          │
└─────────────────────────────────────────────┘
```

**Sample Output (Validation Error - Invalid Start Time):**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Start time must be in the future.           │
│                                             │
│ Provided Data:                              │
│ - Technician: Alice Smith                   │
│ - Profession: Plumber                       │
│ - Start Time: 2025-01-30T14:00:00Z          │
└─────────────────────────────────────────────┘
```

---

### 4. Cancel a Booking

**Description:**

Cancels an existing booking identified by its unique `booking_id`.

**Command:**

```
cancel booking 123e4567-e89b-12d3-a456-426614174000
```

**Sample Output (Success):**

```
┌─────────────────────────────────────────────┐
│           Booking Cancelled Successfully    │
├─────────────────────────────────────────────┤
│ Booking ID: 123e4567-e89b-12d3-a456-426614174000 │
│ Status: Cancelled                          │
├─────────────────────────────────────────────┤
│ Metadata:                                   │
│ - Cancelled At: 2025-01-31T14:45:00Z        │
└─────────────────────────────────────────────┘
```

**Sample Output (Booking Not Found):**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Booking with ID 123e4567-e89b-12d3-a456-426614174000 not found. │
└─────────────────────────────────────────────┘
```

---

### 5. Process a Natural Language Command

**Description:**

Processes a natural language command to perform booking operations such as creating, querying, or cancelling bookings.

**Command Examples:**

- **Creating a Booking:**
  
  ```
  Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak.
  ```
  
- **Cancelling a Booking:**
  
  ```
  Cancel booking 123e4567-e89b-12d3-a456-426614174000.
  ```
  
- **Listing All Bookings:**
  
  ```
  List all bookings.
  ```
  
- **Creating Another Booking:**
  
  ```
  I need electrician Bob Wilson on Monday at 9 AM.
  ```

**Sample Output (Success - Creating a Booking):**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak.                │
│                                                                                      │
│ NLP Analysis [dim](max: 0.92, avg: 0.28)                                             │
│  Intent          ┃   Conf ┃ Analysis                                                 │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                       │
│ create_booking  │  0.92  │ ███████████████▒▒▒▒                                       │
│ query_booking   │  0.05  │ ███▒▒▒▒▒▒▒▒▒▒▒▒                                           │
│ cancel_booking  │  0.03  │ █▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                          │
│ list_bookings   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                           │
│                                                                                      │
│ Response: Booking confirmed for Wednesday at 02:00 PM with Mike Johnson (ID: 123e4567-e89b-12d3-a456-426614174000)                                
╰───────────────────────────────────────────────────────────────────────────────────────── 12:02:18 ─────────────────────────────────────────────────────────────────────────────────────────╯
```

**Sample Output (Success - Listing All Bookings):**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: List all bookings                                                                                                                                         
│                                                                                                                                                           │
│ NLP Analysis [dim](max: 1.00, avg: 0.25)                                                                                                                                     │
│  Intent          ┃   Conf                                                             ┃ Analysis                                                                                                                                │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                                                                                                     │
│ list_bookings   │  1.00  │ ███████████████                                                                                                                         │
│ create_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│ cancel_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│ query_booking   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│                                                                                                                                                                   │
│ Response: Here are all your bookings:                                                                                                                               │
│ - ID: f0d55292-1d26-48ff-ae2f-e7a9838a5d60, Technician: Nicolas Woollett, Profession: Plumber, Start: 2022-10-15 10:00 AM                                            │
│ - ID: 07111419-edbf-4f0e-9c19-b93095eadef4, Technician: Franky Flay, Profession: Electrician, Start: 2022-10-16 06:00 PM                                            │
│ - ID: 838d3646-855e-46e1-a1be-843f4ebb5f77, Technician: Griselda Dickson, Profession: Welder, Start: 2022-10-18 11:00 AM                                            │
│ - ID: 13806ace-540b-464f-8087-87b40c50b37f, Technician: John Do Alice Smith, Profession: Plumber, Start: 2025-02-01 02:00 PM                                            │
│ - ID: 43919cc7-01db-4bee-a818-d08d0d01d7a0, Technician: Karen, Profession: Electrician, Start: 2025-02-01 09:00 AM                                        │
╰───────────────────────────────────────────────────────────────────────────────────────── 12:02:39 ─────────────────────────────────────────────────────────────────────────────────────────╯
```

**Sample Output (Success - Creating a Booking without Customer):**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: I'm Karen and I want to book an Electrician for tomorrow                                                                                             │
│                                                                                                                                              │
│ NLP Analysis [dim](max: 1.00, avg: 0.25)                                                                                                     │
│  Intent          ┃   Conf ┃ Analysis                                                                                                         │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                                                                               │
│ create_booking  │  1.00  │ ███████████████                                                                                                   │
│ cancel_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                   │
│ query_booking   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                   │
│ list_bookings   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                   │
│                                                                                                                                              │
│ Response: Booking confirmed for Saturday at 09:00 AM with Karen (ID: 43919cc7-01db-4bee-a818-d08d0d01d7a0)                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────── 12:03:37 ─────────────────────────────────────────────────────────────────────────────────────────╯
```

**Sample Output (Error - Fuzzy Parsing Failed):**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: List all bookings                                                                                                                                        │
│                                                                                                                                                                   │
│ NLP Analysis [dim](max: 1.00, avg: 0.25)                                                                                                                                             │
│  Intent          ┃   Conf ┃ Analysis                                                                                                                                        │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                                                                                                     │
│ list_bookings   │  1.00  │ ███████████████                                                                                                                         │
│ create_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│ cancel_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│ query_booking   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                                                                                                         │
│                                                                                                                                                                   │
│ Response: Here are all your bookings:                                                                                                                               │
│ - ID: f0d55292-1d26-48ff-ae2f-e7a9838a5d60, Technician: Nicolas Woollett, Profession: Plumber, Start: 2022-10-15 10:00 AM                                            │
│ - ID: 07111419-edbf-4f0e-9c19-b93095eadef4, Technician: Franky Flay, Profession: Electrician, Start: 2022-10-16 06:00 PM                                            │
│ - ID: 838d3646-855e-46e1-a1be-843f4ebb5f77, Technician: Griselda Dickson, Profession: Welder, Start: 2022-10-18 11:00 AM                                            │
│ - ID: 13806ace-540b-464f-8087-87b40c50b37f, Technician: John Do Alice Smith, Profession: Plumber, Start: 2025-02-01 02:00 PM                                            │
│ - ID: 43919cc7-01db-4bee-a818-d08d0d01d7a0, Technician: Karen, Profession: Electrician, Start: 2025-02-01 09:00 AM                                        │
╰───────────────────────────────────────────────────────────────────────────────────────── 12:03:47 ─────────────────────────────────────────────────────────────────────────────────────────╯
```

---

### 6. Exit the CLI

**Description:**

Gracefully terminates the CLI session.

**Command:**

```
exit
```

**Sample Output:**

```
┌─────────────────────────────────────────────┐
│                Goodbye!                     │
├─────────────────────────────────────────────┤
│ Thank you for using the Technician Booking  │
│ System. Have a great day!                   │
└─────────────────────────────────────────────┘
```

---

## Additional Examples

To further illustrate the capabilities of the **Technician Booking System CLI**, here are more examples covering various scenarios:

### Example 1: Creating a Booking with Missing Information

**Command:**

```
create booking --technician "Bob Builder" --start_time "2025-02-05T10:00:00Z"
```

**Sample Output:**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Profession is required for booking creation.│
│                                             │
│ Provided Data:                              │
│ - Technician: Bob Builder                   │
│ - Start Time: 2025-02-05T10:00:00Z          │
└─────────────────────────────────────────────┘
```

---

### Example 2: Processing a Command with Multiple Intents

**Command:**

```
process command "Book electrician Jane Doe for next Monday at 9 AM and then list all bookings."
```

**Sample Output:**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: Book electrician Jane Doe for next Monday at 9 AM and then list all bookings. │
│                                                                                      │
│ NLP Analysis [dim](max: 0.85, avg: 0.35)                                             │
│  Intent          ┃   Conf ┃ Analysis                                                 │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                       │
│ create_booking  │  0.85  │ ██████████████▒▒▒▒                                        │
│ list_bookings   │  0.15  │ █████▒▒▒▒▒▒▒▒▒▒▒▒▒                                        │
│ cancel_booking  │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                           │
│ query_booking   │  0.00  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                           │
│                                                                                      │
│ Response: Booking confirmed for next Monday at 09:00 AM with Jane Doe (ID: 223e4567-e89b-12d3-a456-426614174001)
│                                             │
│ Metadata:                                   │
│ - Processed At: 2025-01-31T15:00:00Z        │
│ - Processing Time: 170ms                    │
└─────────────────────────────────────────────┘
```

---

### Example 3: Cancelling a Non-Existent Booking

**Command:**

```
cancel booking 00000000-0000-0000-0000-000000000000
```

**Sample Output:**

```
┌─────────────────────────────────────────────┐
│                Error:                       │
├─────────────────────────────────────────────┤
│ Booking with ID 00000000-0000-0000-0000-000000000000 not found. 
└─────────────────────────────────────────────┘
```

---

### Example 4: Processing an Unknown Command

**Command:**

```
process command "Tell me a joke."
```

**Sample Output:**

```
╭─────────────────────────────────────────────────────────────────────────────────────── NLP Analysis ───────────────────────────────────────────────────────────────────────────────────────╮
│ Input: Tell me a joke.                                                               │
│                                                                                      │
│ NLP Analysis [dim](max: 0.10, avg: 0.10)                                             │
│  Intent          ┃   Conf ┃ Analysis                                                 │
│ ━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━                                       │
│ create_booking  │  0.10  │ ████▒▒▒▒▒▒▒▒▒▒▒▒▒                                         │
│ query_booking   │  0.10  │ ████▒▒▒▒▒▒▒▒▒▒▒▒▒                                         │
│ cancel_booking  │  0.10  │ ████▒▒▒▒▒▒▒▒▒▒▒▒▒                                         │
│ list_bookings   │  0.10  │ ████▒▒▒▒▒▒▒▒▒▒▒▒▒                                         │
│                                                                                      │
│ Response: I'm sorry, I didn't understand that request. Could you please rephrase it? │
│                                                                                      │
│ Metadata:                                                                            │
│ - Processed At: 2025-01-31T15:05:00Z                                                 │
│ - Processing Time: 160ms                    │                                        │
└─────────────────────────────────────────────┘
```

---

## Notes

- **Starting the CLI:** Always start the CLI using the command `poetry run python -m app.core.cli` to ensure it runs within the correct virtual environment and with all dependencies loaded.

- **Command Syntax:** Commands must be entered in specific formats to be recognized and processed correctly:
  
  - **List All Bookings:**
    ```
    list all bookings
    ```
  
  - **Retrieve Booking Details:**
    ```
    get booking details <booking_id>
    ```
  
  - **Create a New Booking:** Use natural language commands without requiring flags.
    - **Example:**
      ```
      Book plumber Mike Johnson for Wednesday at 2 PM to fix a leak.
      ```
      ```
      Schedule welder James Brown for Friday at 10 AM for metal work.
      ```
      ```
      I need electrician Bob Wilson on Monday at 9 AM.
      ```
  
  - **Cancel a Booking:**
    ```
    cancel booking <booking_id>
    ```
  
  - **Process a Natural Language Command:**
    ```
    process command "<Natural Language Command>"
    ```
    *Example:*
    ```
    process command "Book carpenter Emily White for Saturday at 3 PM to build a cabinet."
    ```
  
  - **Exit the CLI:**
    ```
    exit
    ```
    *Alternatively:*
    ```
    quit
    ```
    or
    ```
    q
    ```

- **Booking IDs:** Ensure that the `booking_id` used in the commands corresponds to an existing booking in your system. Booking IDs are UUIDs in the format `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

- **Date and Time Formats:** Use natural language expressions for date and time (e.g., "tomorrow", "next Monday at 9 AM") to leverage the NLP capabilities of the CLI.

- **Error Handling:** The CLI provides detailed error messages with clear descriptions and metadata to assist in troubleshooting.

- **Metadata Information:** Each successful command includes metadata such as timestamps and processing times to provide context and facilitate monitoring.

- **Interactive Feedback:** The CLI leverages `rich` to present information in well-formatted panels, tables, and colored text for enhanced readability and user experience.

- **Natural Language Processing:** When using the `process command` feature, ensure that the natural language input is clear and contains necessary information to perform the desired action. The NLP service analyzes the intent and entities to execute commands accurately.

---

## Testing

The system includes comprehensive test coverage:

```bash
# Run test suite
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=app --cov-report=term-missing
```

Key test areas:
- Intent classification accuracy
- Entity extraction precision
- Temporal parsing reliability
- Booking validation integrity

## Architecture Diagram

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   CLI Interface  │     │    HTTP API      │     │  Validation      │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                        NLP Service                               │
├──────────────────┬──────────────────┬────────────────────────────┤
│ Intent           │ Entity           │ Temporal                   │
│ Classification   │ Recognition      │ Processing                 │
└────────┬─────────┴────────┬─────────┴───────────────┬────────────┘
         │                  │                         │
         ▼                  ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Booking Service                              │
└──────────────────────────────────────────────────────────────────┘
```

## Performance Metrics

- Intent Classification Accuracy: >95%
- Entity Recognition Precision: >90%
- Temporal Expression Parsing: >98%
- Average Response Time: <100ms

## Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and coding standards.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{technician_booking_system,
  author = {Ericson Willians},
  title = {Technician Booking System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ericsonwillians/technician-booking-backend}
}
```