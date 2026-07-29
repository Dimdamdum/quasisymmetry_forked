#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [angle_in_title]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [angle_in_title]}"

# Default to "False" (no angle) if the 3rd argument is not provided
angle_in_title="${3:-False}"

PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}" python3 -c "
import sys
from K_sectors_plots import generate_plots_across_bond_lengths
generate_plots_across_bond_lengths(sys.argv[1], sys.argv[2], sys.argv[3])
" "$INPUT_DIR" "$OUTPUT_DIR" "$angle_in_title"