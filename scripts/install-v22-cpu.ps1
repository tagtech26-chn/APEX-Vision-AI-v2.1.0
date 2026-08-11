param(
    [string]$Python = "py -3.11",
    [string]$VenvPath = "D:\v22env"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\")).Path

Write-Host "APEX Vision AI v2.2 CPU environment" -ForegroundColor Green
Write-Host "Repository: $RepoRoot" -ForegroundColor DarkGray
Write-Host "Environment: $VenvPath" -ForegroundColor DarkGray

if (-not (Test-Path -LiteralPath $VenvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & py -3.11 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Unable to create $VenvPath" }
}

$Py = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) { throw "Python executable not found at $Py" }

& $Py -m pip install --upgrade "pip<27" "setuptools<82" wheel
if ($LASTEXITCODE -ne 0) { throw "Base packaging install failed." }

& $Py -m pip install "numpy==2.4.6" "opencv-python==5.0.0.93" "pytest==9.1.1"
if ($LASTEXITCODE -ne 0) { throw "Base validation dependencies failed." }

& $Py -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed." }

Write-Host "CPU v2.2 environment ready: $Py" -ForegroundColor Green
Write-Host "Next: install the validated AI model stack separately." -ForegroundColor Yellow
