# Project commands (Windows PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/check.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== compile ==" -ForegroundColor Cyan
python -m compileall -q aicwf

Write-Host "== unit + integration tests ==" -ForegroundColor Cyan
python -m pytest -q

Write-Host "== environment / data / model doctor ==" -ForegroundColor Cyan
python -m aicwf doctor

Write-Host "All local checks passed." -ForegroundColor Green
