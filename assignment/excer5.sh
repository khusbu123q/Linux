#!/bin/bash

# Get the total disk space and used space for the root directory
root_total=$(df --output=size -B1 / | tail -n1)
root_used=$(df --output=used -B1 / | tail -n1)

# Get the total disk space and used space for the $HOME directory
home_total=$(df --output=size -B1 "$HOME" | tail -n1)
home_used=$(df --output=used -B1 "$HOME" | tail -n1)

# Calculate the percentage of used space in the $HOME directory relative to the root directory
home_usage_percentage=$(echo "scale=2; ($home_used / $root_total) * 100" | bc)

# Display the results
echo "Total disk space for the root directory: $(df -h / | tail -n1 | awk '{print $2}')"
echo "Used disk space for the root directory: $(df -h / | tail -n1 | awk '{print $3}')"
echo "Total disk space for the \$HOME directory: $(df -h "$HOME" | tail -n1 | awk '{print $2}')"
echo "Used disk space for the \$HOME directory: $(df -h "$HOME" | tail -n1 | awk '{print $3}')"
echo "Percentage of $HOME usage relative to root: $home_usage_percentage%"

