#!/bin/bash
# EASE Setup Script
# Installs dependencies for both frontend and server

set -e

echo "======================================"
echo "EASE - Effortless Audio-Synesthesia Experience"
echo "Setup Script"
echo "======================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is required but not installed.${NC}"
        echo "$2"
        exit 1
    fi
    echo -e "${GREEN}Found $1${NC}"
}

echo "Checking prerequisites..."
check_command python3 "Install Python 3.10+ from https://python.org"
check_command node "Install Node.js 18+ from https://nodejs.org"
check_command npm "Install npm (comes with Node.js)"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l 2>/dev/null || python3 -c "print(1 if $PYTHON_VERSION < 3.10 else 0)") == "1" ]]; then
    echo -e "${RED}Error: Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Check Node version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [[ "$NODE_VERSION" -lt 18 ]]; then
    echo -e "${RED}Error: Node.js 18+ required (found v$NODE_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Node.js version: v$NODE_VERSION${NC}"

# Check for CUDA (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected. AI features require CUDA.${NC}"
fi

echo
echo "Setting up server..."
cd "$(dirname "$0")/../server"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
fi

# Create virtual environment and install dependencies
echo "Installing Python dependencies..."
uv sync

# Copy .env.example to .env if it doesn't exist
if [[ ! -f .env ]]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo
echo "Setting up frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo
echo "Downloading models (this may take a while on first run)..."
cd ../server
uv run python ../scripts/download-models.py

echo
echo "======================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "======================================"
echo
echo "To start EASE:"
echo
echo "  Terminal 1 (Server):"
echo "    cd server && uv run python -m src.main"
echo
echo "  Terminal 2 (Frontend):"
echo "    cd frontend && npm run dev"
echo
echo "Then open http://localhost:5173 in your browser."
echo
