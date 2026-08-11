param(
    [string]$Python = "D:\v22env\Scripts\python.exe",
    [switch]$SmokeModels
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = "D:\v22env\Scripts\python.exe"
}

if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path -Path $RepoRoot -ChildPath $Python
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "v2.2 Python environment not found:" -ForegroundColor Yellow
    Write-Host "  $Python" -ForegroundColor Yellow
    Write-Host "Pass the external environment explicitly, for example:" -ForegroundColor Cyan
    Write-Host "  .\validate-v2.2.ps1 -Python D:\v22env\Scripts\python.exe" -ForegroundColor White
    throw "v2.2 Python environment not found."
}

Write-Host "Using v2.2 Python: $Python" -ForegroundColor Green

Write-Host "[1/4] Python version" -ForegroundColor Cyan
& $Python --version
if ($LASTEXITCODE -ne 0) { throw "Unable to execute v2.2 Python." }

Write-Host "[2/4] Compile all Python sources" -ForegroundColor Cyan
$AppPath = Join-Path -Path $RepoRoot -ChildPath "app"
$TestsPath = Join-Path -Path $RepoRoot -ChildPath "tests"
& $Python -m compileall -q $AppPath $TestsPath
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

Write-Host "[3/4] Metric floor regression" -ForegroundColor Cyan
$TestPath = Join-Path -Path $RepoRoot -ChildPath "tests\test_v22_metric_floor.py"
& $Python -m pytest -q $TestPath
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
