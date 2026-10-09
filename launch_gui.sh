#!/usr/bin/env bash
# Launch script for Custom GPT Desktop Playground on Linux/macOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========================================================"
echo " Launching Custom GPT Desktop Playground..."
echo "========================================================"

PYTHON_EXEC=""
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="$(command -v python3)"
elif command -v python &> /dev/null; then
    PYTHON_EXEC="$(command -v python)"
fi

if [ -z "$PYTHON_EXEC" ]; then
    echo "[ERROR] Python interpreter not found in .venv or system PATH!"
    echo "Please create a virtual environment: python3 -m venv .venv"
    exit 1
fi

echo "[OK] Using Python interpreter: $PYTHON_EXEC"
echo "Starting desktop interface..."
"$PYTHON_EXEC" "$SCRIPT_DIR/scripts/gui_app.py" "$@"

echo ""
echo "========================================================"
echo " GUI process has closed."
echo "========================================================"
