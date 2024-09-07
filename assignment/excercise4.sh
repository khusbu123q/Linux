#!/bin/bash

THRESHOLD=70

filesystem=$(df -P "$HOME" | awk 'NR==2 {print $1}')

usage=$(df -P "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')

if [ "$usage" -ge "$THRESHOLD" ]; then
    echo "Warning: Disk usage on the filesystem where \$HOME is located is at ${usage}%."
    echo "Threshold of ${THRESHOLD}% exceeded."
else
    echo "Disk usage is within safe limits: ${usage}%."
fi

