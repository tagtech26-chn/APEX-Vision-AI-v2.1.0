param(
    [string]$Python = (Join-Path $PSScriptRoot ".venv-v22\Scripts\python.exe"),
    [switch]$SmokeModels
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Python)) {
    throw "v2.2 Python environment not found: $Python. Create .venv-v22 first."
}

Write-Host "[1/4] Python version" -ForegroundColor Cyan
& $Python --version

Write-Host "[2/4] Compile all Python sources" -ForegroundColor Cyan
& $Python -m compileall -q (Join-Path $PSScriptRoot "app") (Join-Path $PSScriptRoot "tests")
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

Write-Host "[3/4] Metric floor regression" -ForegroundColor Cyan
& $Python -m pytest -q (Join-Path $PSScriptRoot "tests/test_v22_metric_floor.py")
if ($LASTEXITCODE -ne 0) { throw "Metric floor regression failed." }

Write-Host "[4/4] Provider import smoke" -ForegroundColor Cyan
& $Python -c "from app.ai.config import resolve_provider; from app.ai.geometry.metric_floor import MetricFloorEstimator; print('provider=', resolve_provider('v22')); print('MetricFloorEstimator=OK')"
if ($LASTEXITCODE -ne 0) { throw "v2.2 provider import validation failed." }

if ($SmokeModels) {
    Write-Host "[MODEL SMOKE] Loading Metric3D + SAM3. This may download several GB and requires the approved SAM3 checkpoint." -ForegroundColor Yellow
    & $Python -c "from app.ai.scene.analyzer import build_scene_analyzer; a=build_scene_analyzer('v22'); print('detector=',a.detector.name); print('segmenter=',a.segmenter.name); print('depth=',a.depth.name)"
    if ($LASTEXITCODE -ne 0) { throw "v2.2 model smoke failed." }
}

Write-Host "V2.2 validation PASSED." -ForegroundColor Green
