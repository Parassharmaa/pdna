#!/bin/bash
# Download results from RunPod and generate analysis report.
# Usage: bash scripts/download_and_report.sh

set -e

RUNPOD_HOST="root@205.196.17.124"
RUNPOD_PORT="8513"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=== Downloading results from RunPod ==="

# Download all_results.json
scp $SSH_OPTS -P $RUNPOD_PORT $RUNPOD_HOST:/root/pdna/runs/all_results.json runs/all_results.json 2>/dev/null || {
    echo "ERROR: Could not download all_results.json"
    exit 1
}

echo "Downloaded runs/all_results.json"

# Download experiment log
scp $SSH_OPTS -P $RUNPOD_PORT $RUNPOD_HOST:/root/exp_fast.log docs/experiment_log.txt 2>/dev/null || true

echo "Downloaded experiment log"

# Count results
RUNS=$(python3 -c "import json; d=json.load(open('runs/all_results.json')); print(len(d))")
echo "Total runs in results: $RUNS"

echo ""
echo "=== Generating report ==="
uv run python scripts/generate_report.py --results-dir runs --output-dir reports

echo ""
echo "=== Report generated ==="
echo "Reports saved to: reports/"
ls -la reports/
echo ""
echo "Figures:"
ls -la reports/figures/ 2>/dev/null || echo "No figures directory"
