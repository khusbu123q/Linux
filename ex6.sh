#!/bin/bash
SEARCH_DIR="example_directory"
OUTPUT_FILE="empty_subfolders.txt"
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR does not exist."
    exit 1
fi
find "$SEARCH_DIR" -type d -empty > "$OUTPUT_FILE"
echo "Empty subdirectories have been listed in $OUTPUT_FILE."

