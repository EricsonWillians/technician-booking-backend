#!/bin/bash

###############################################################################
# init_project.sh
#
# Description:
#   This script initializes the directory structure and boilerplate files
#   for a FastAPI-based Technician Booking System, using:
#       - Python 3.10
#       - Poetry (for dependency management)
#       - Hugging Face Transformers integration
#   It also creates an example .env file with placeholders for your
#   Hugging Face API Key and model name.
#
# Usage:
#   bash init_project.sh
#
# Author:
#   Ericson Willians <ericsonwillians@protonmail.com>
###############################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Uncomment this for debugging
# set -x

#------------------------------------------------------------------------------#
# Helper Functions
#------------------------------------------------------------------------------#

log() {
    echo "[INFO] $1"
}

error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

#------------------------------------------------------------------------------#
# Preliminary Checks
#------------------------------------------------------------------------------#

# Validate script is being run from a suitable location:
# (Not strictly required, but helpful if you want to ensure a fresh directory.)
if [[ -d "./app" || -d "./tests" || -d "./scripts" || -f "./pyproject.toml" ]]; then
    log "Some project files or directories already exist. Proceeding carefully."
else
    log "Initializing a new FastAPI + Poetry + HuggingFace project structure."
fi

#------------------------------------------------------------------------------#
# Directory Creation
#------------------------------------------------------------------------------#

directories=(
    "./app/core"
    "./app/routers"
    "./app/models"
    "./app/services"
    "./app/schemas"
    "./app/config"
    "./app/utils"
    "./tests/unit"
    "./tests/integration"
    "./scripts"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log "Created directory: $dir"
    else
        log "Directory already exists: $dir"
    fi
done

#------------------------------------------------------------------------------#
# File Creation
#------------------------------------------------------------------------------#

files=(
    "./app/__init__.py"
    "./app/main.py"
    "./app/core/__init__.py"
    "./app/core/initial_data.py"
    "./app/core/cli.py"
    "./app/routers/__init__.py"
    "./app/routers/bookings.py"
    "./app/models/__init__.py"
    "./app/models/booking.py"
    "./app/services/__init__.py"
    "./app/services/booking_service.py"
    "./app/services/validation.py"
    "./app/schemas/__init__.py"
    "./app/schemas/booking.py"
    "./app/config/__init__.py"
    "./app/config/settings.py"
    "./app/utils/__init__.py"
    "./app/utils/llm_processor.py"
    "./tests/__init__.py"
    "./tests/unit/test_services.py"
    "./tests/integration/test_api.py"
    "./scripts/setup.sh"
    "./.env.example"
    "./README.md"
)

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        log "Created file: $file"
    else
        log "File already exists: $file"
    fi
done

#------------------------------------------------------------------------------#
# Create or Update .env.example
#------------------------------------------------------------------------------#

ENV_EXAMPLE_CONTENT="# .env.example
# Copy this file to .env and replace the placeholder values as needed.
# Security: Never commit your actual .env to version control.

# Hugging Face API Key (if using a private or hosted model)
HUGGINGFACE_API_KEY=\"<YOUR_HUGGING_FACE_API_KEY>\"

# Default model for your Transformers pipelines
MODEL_NAME=\"bert-base-uncased\"

# Add any other environment variables you need:
# e.g., DB_URL, SECRET_KEY, etc.
"

# Overwrite (or create fresh) .env.example
echo "$ENV_EXAMPLE_CONTENT" > "./.env.example"
log "Wrote placeholder content to .env.example"

#------------------------------------------------------------------------------#
# Create or Update pyproject.toml (if desired)
#------------------------------------------------------------------------------#
# If you want to automate creation of a minimal pyproject.toml, uncomment below:
: '
PYPROJECT_CONTENT="[tool.poetry]
name = \"TechBooker\"
version = \"0.1.0\"
description = \"Technician Booking System with NLP Integration\"
authors = [\"Ericson Willians <ericsonwillians@protonmail.com>\"]
license = \"MIT\"
readme = \"README.md\"

[tool.poetry.dependencies]
python = \"^3.10\"
fastapi = \"^0.103.2\"
uvicorn = \"^0.23.2\"
pydantic = \"^2.4.2\"
python-dotenv = \"^1.0.0\"
transformers = \"^4.36.2\"
torch = \"^2.1.1\"
datasets = \"^2.16.1\"
sentence-transformers = \"^2.2.2\"
accelerate = \"^0.25.0\"
python-dateutil = \"^2.8.2\"
typer = \"^0.9.0\"

[tool.poetry.group.dev.dependencies]
pytest = \"^7.4.2\"
mypy = \"^1.6.1\"
black = \"^23.10.1\"
flake8 = \"^6.1.0\"
isort = \"^5.12.0\"
httpx = \"^0.25.0\"

[build-system]
requires = [\"poetry-core\"]
build-backend = \"poetry.core.masonry.api\"
"

if [ ! -f "./pyproject.toml" ]; then
    echo "$PYPROJECT_CONTENT" > "./pyproject.toml"
    log "Created pyproject.toml for Poetry."
else
    log "pyproject.toml already exists. Skipping creation."
fi
'

#------------------------------------------------------------------------------#
# Final Message
#------------------------------------------------------------------------------#

log "Project structure has been initialized successfully!"
log "Next steps:"
log "1) Check your .env.example and create a .env file with real values."
log "2) Run 'poetry init' or verify your existing pyproject.toml to manage dependencies."
log "3) Install dependencies with 'poetry install'."
log "4) You're ready to start developing your FastAPI + Hugging Face project!"
