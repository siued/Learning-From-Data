#!/bin/bash

# Check if a Python file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <python_file> [output_dir]"
    exit 1
fi

PYTHON_FILE="$1"
BASE_DIR="$(dirname "$PYTHON_FILE")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set output directory: use second argument if provided, otherwise create a timestamped directory
OUT_DIR="${BASE_DIR}/${2:-output_$TIMESTAMP}"
mkdir -p "$OUT_DIR"

OUT_FILE="$OUT_DIR/output.txt"
PYTHON_FILE_COPY="$OUT_DIR/$(basename "$PYTHON_FILE" .py)_copy.py"

# Run the Python file, save output to file and print it to stdout
python "$PYTHON_FILE" | tee "$OUT_FILE"

# Copy the Python file to the output directory
cp "$PYTHON_FILE" "$PYTHON_FILE_COPY"
