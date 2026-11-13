# End-to-end demo (Windows PowerShell).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$ASOF = if ($env:AICWF_ASOF) { $env:AICWF_ASOF } else { "2026-11-12" }

Write-Host "1/5 replay run" -ForegroundColor Cyan
python -m aicwf run --as-of $ASOF --llm replay --run-id "demo-$ASOF"

Write-Host "2/5 approve and finish" -ForegroundColor Cyan
python -m aicwf approve "demo-$ASOF" --action publish_report --note "Demo approval"

Write-Host "3/5 crash + resume" -ForegroundColor Cyan
python -m aicwf fault-demo --as-of $ASOF

Write-Host "4/5 evaluation and rollback" -ForegroundColor Cyan
python -m aicwf eval --candidate v1.1.0 --baseline v1.0.0
python -m aicwf rollback --to v1.0.0

Write-Host "5/5 monthly monitor" -ForegroundColor Cyan
python -m aicwf monitor --as-of $ASOF --llm replay

Write-Host "Demo finished. See outputs/ and evidence/." -ForegroundColor Green
