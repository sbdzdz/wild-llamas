#!/usr/bin/env bash
set -euo pipefail

# Determine repo root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

delete_file() {
  local rel_path="$1"
  local abs_path="$REPO_ROOT/$rel_path"
  if [ -f "$abs_path" ]; then
    rm -f "$abs_path"
    echo "Deleted file: $rel_path"
  else
    echo "File not found, skipped: $rel_path"
  fi
}

delete_dir() {
  local rel_path="$1"
  local abs_path="$REPO_ROOT/$rel_path"
  if [ -d "$abs_path" ]; then
    rm -rf "$abs_path"
    echo "Deleted directory: $rel_path"
  else
    echo "Directory not found, skipped: $rel_path"
  fi
}

delete_file "data/skipped_models.csv"
delete_file "data/all_models.csv"
delete_file "outputs/opencompass/merge_log.csv"

delete_dir "models/merged_model"
delete_dir "outputs/opencompass/merged_model"
delete_dir "icl_inference_output"
delete_dir "tmp"


