#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Convenience wrapper to run the batch pipeline inside Docker.
# Usage:
#   ./scripts/run_pipeline.sh            # full run using CSV data
#   ./scripts/run_pipeline.sh --db       # use PostgreSQL as data source
#   ./scripts/run_pipeline.sh --find-k   # run elbow analysis first
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SOURCE="csv"
FIND_K=""
WRITE_DB=""

# Parse flags
for arg in "$@"; do
  case $arg in
    --db)     SOURCE="db" ;;
    --find-k) FIND_K="--find-k" ;;
    --write-db) WRITE_DB="--write-db" ;;
  esac
done

echo "╔══════════════════════════════════════════════╗"
echo "║   User Segmentation Pipeline                ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Source  : $SOURCE"
echo "  Find K  : ${FIND_K:-no}"
echo "  Write DB: ${WRITE_DB:-no}"
echo ""

# Step 1: Generate sample data (skip if raw CSVs exist)
if [ ! -f "data/raw/users.csv" ]; then
  echo "▶ Generating sample data …"
  python data/sample/generate_sample_data.py
else
  echo "▶ Raw data already exists, skipping generation."
fi

# Step 2: Run pipeline
echo ""
echo "▶ Running segmentation pipeline …"
python -m src.pipeline.batch_pipeline \
  --source "$SOURCE" \
  $FIND_K \
  $WRITE_DB \
  --plots-dir reports/plots

echo ""
echo "✅ Pipeline complete!"
echo "   Segments → data/processed/user_segments.csv"
echo "   Plots    → reports/plots/"
echo "   Models   → models/saved/"
