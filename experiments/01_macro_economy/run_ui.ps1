# Launches the Country-Year Forecast Studio (Streamlit UI).
#
# Usage (from project root):
#   .\run_ui.ps1
#
# Optional:
#   .\run_ui.ps1 -Port 8502        # bind a different port
#   .\run_ui.ps1 -NoBrowser        # don't auto-open browser
#
# Notes:
#   - Forces the project root on PYTHONPATH so `scripts.web_app` resolves
#     both `from src...` and `from scripts...` style imports.
#   - Streamlit caches the modules between tab switches, so first paint
#     takes ~5-10s (panel + models + FAISS load).

[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -Path $PSScriptRoot).Path
Set-Location -Path $ProjectRoot

# Make 'src' and 'scripts' importable from web_app.py's `from src...` and
# `from scripts...` style imports (rootdir-style).
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\scripts"

$streamlitArgs = @(
    "run", "scripts/web_app.py",
    "--server.port", $Port,
    "--server.headless", $(if ($NoBrowser) { "true" } else { "false" }),
    "--browser.gatherUsageStats", "false"
)

Write-Host "Starting Country Forecast Studio on http://localhost:$Port" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host ""

& streamlit @streamlitArgs