#!/bin/bash
CURRENT_DIR=$(pwd)
cd /home/sizhe/fit-cnice/vidpiper
ORIG_ARGS=("$@")
MODIFIED_ARGS=()

for arg in "${ORIG_ARGS[@]}"; do
  # Skip arguments that start with a dash (options)
  if [[ "$arg" =~ ^- ]]; then
    MODIFIED_ARGS+=("$arg")
    continue
  fi
  
  # Skip numeric values including decimals (e.g., 60.0)
  if [[ "$arg" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    MODIFIED_ARGS+=("$arg")
    continue
  fi
  
  # Check if argument is a path (exists as file/dir or has path-like format)
  if [[ -e "$CURRENT_DIR/$arg" || -d "$CURRENT_DIR/$arg" || -f "$CURRENT_DIR/$arg" || 
        "$arg" =~ \/ || "$arg" =~ \.[a-zA-Z0-9]+$ ]]; then
    # If it's a relative path, make it absolute
    if [[ "$arg" != /* ]]; then
      MODIFIED_ARGS+=("$CURRENT_DIR/$arg")
    else
      # Already an absolute path
      MODIFIED_ARGS+=("$arg")
    fi
  else
    # Not a path, keep as is
    MODIFIED_ARGS+=("$arg")
  fi
done

uv run vidpiper_cli "${MODIFIED_ARGS[@]}"

