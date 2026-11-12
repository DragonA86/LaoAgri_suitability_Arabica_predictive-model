Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Create the environment first: py -3.11 -m venv .venv" -ForegroundColor Yellow
    exit 1
}
. .\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
