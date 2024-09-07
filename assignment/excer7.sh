#!/bin/bash

input="input.txt"
temp="temp_file.txt"

if [ ! -f "$input" ]; then
    echo "Error: File $input does not exist."
    exit 1
fi
sort "$input" | uniq > "$temp"

mv "$temp" "$temp"

echo "Duplicate lines were removed from $input"

