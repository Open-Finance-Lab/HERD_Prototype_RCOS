#!/bin/bash

set -e

if [ -f .env ]; then
    echo "Loading .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "No .env file found, continuing without it."
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama server..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5
fi

if ! ollama list | grep -q "${OLLAMA_MODEL:-gemma:2b}"; then
    echo "Pulling Ollama model ${OLLAMA_MODEL:-gemma:2b}..."
    ollama pull "${OLLAMA_MODEL:-gemma:2b}"
fi

echo "Starting FastAPI app..."
uvicorn app:app --host 0.0.0.0 --port 8000
