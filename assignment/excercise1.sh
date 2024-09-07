#!/bin/bash

#allocating variables

list_executables() {
    local dir="$1"
    echo "Scanning directory: $dir"
    
    find "$dir" -maxdepth 1 -type f -executable -print
}
#print the path
path_dirs=$(echo "$PATH" | tr ':' '\n')

if [[ -z "$path_dirs" ]]; then
    echo "PATH environment variable is empty or not set."
    exit 1
fi
#checking whether directory is present or not 
for dir in $path_dirs; do
    if [[ -d "$dir" ]]; then
        list_executables "$dir"
    else
        echo "Directory not found: $dir"
    fi
done
