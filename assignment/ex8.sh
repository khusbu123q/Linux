#!/bin/bash
THRESHOLD=90

for ((i=1; i<=100; i++)); do
    echo "$i"
    
    if [ "$i" -gt "$THRESHOLD" ]; then
        echo "$i is greater than $THRESHOLD"
    fi
done
