#!/bin/bash


input="input.txt"

if [ ! -f "$input" ]; then
  echo "Error: File '$input' not found."
  exit 1
fi

temp=$(mktemp)

sort "$input" | uniq > "$temp"

mv "$temp" "$input"

echo "Duplicate lines removed from '$input'."
