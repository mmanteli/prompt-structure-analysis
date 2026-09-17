#!/bin/bash

# Check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

ROOT_DIR="$1"

# Check if the provided path is a valid directory
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: '$ROOT_DIR' is not a valid directory"
    exit 1
fi

# Find all leaf directories (directories with no subdirectories)
find "$ROOT_DIR" -type d | while read -r dir; do
    # Count direct subdirectories
    subdir_count=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [[ "$dir" == "*/.ipynb_checkpoints*" ]]; then
        continue 1
    fi
    # If no subdirectories exist, it's a leaf directory
    if [ "$subdir_count" -eq 0 ]; then
        # Count JSON files directly in this directory
        json_count=$(find "$dir" -maxdepth 1 -name "*.json" -type f | wc -l)

        # Print directory if it doesn't contain exactly 3 JSON files
        if [ "$json_count" -ne 4 ]; then
            echo "$dir (found $json_count JSON file(s))"
        fi
    fi
done
