#!/usr/bin/env bash
# End-to-end demo (bash/macOS/Linux equivalent of scripts/run_demo.ps1).
set -euo pipefail
cd "$(dirname "$0")/.."

ASOF="${AICWF_ASOF:-2026-11-12}"

echo "1/5 replay run"
python -m aicwf run --as-of "$ASOF" --llm replay --run-id "demo-$ASOF"

echo "2/5 approve and finish"
python -m aicwf approve "demo-$ASOF" --action publish_report --note "Demo approval"

echo "3/5 crash + resume"
python -m aicwf fault-demo --as-of "$ASOF"

echo "4/5 evaluation and rollback"
python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0
python -m aicwf rollback --to v1.0.0

echo "5/5 monthly monitor"
python -m aicwf monitor --as-of "$ASOF" --llm replay

echo "Demo finished. See outputs/ and evidence/."
