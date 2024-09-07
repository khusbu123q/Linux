#!/bin/bash

# Get the current hour in 24-hour format
current_hour=$(date +"%H")

# Determine the greeting based on the current hour
if [ "$current_hour" -ge 5 ] && [ "$current_hour" -lt 12 ]; then
    echo "Good morning!"
elif [ "$current_hour" -ge 12 ] && [ "$current_hour" -lt 18 ]; then
    echo "Good afternoon!"
else
    echo "Good night!"
fi
