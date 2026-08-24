# PowerShell Launcher for Forecast Studio Web Application
$host.UI.RawUI.WindowTitle = "Quad-Domain Macroeconomic Forecast Studio"
Write-Host "Starting Country-Year Forecast Studio (Streamlit)..." -ForegroundColor Cyan

if (Test-Path "experiments/01_macro_economy/scripts/web_app.py") {
    streamlit run experiments/01_macro_economy/scripts/web_app.py
} elseif (Test-Path "projectresearch/scripts/web_app.py") {
    streamlit run projectresearch/scripts/web_app.py
} else {
    streamlit run app.py
}
