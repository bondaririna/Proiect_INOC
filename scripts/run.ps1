param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
    & $Python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\main.py
