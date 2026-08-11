<#
.SYNOPSIS
    Starts the APEX Vision AI production server (single service: SPA + API).

.DESCRIPTION
    Loads models.env (heavy model paths, created by setup-windows.ps1 /
    download-heavy-models.ps1) if present, then runs uvicorn with a single
    worker. Production defaults to the Heavy AI provider. Set
    APEX_AI_PROVIDER=auto or light explicitly when a fallback is required.

    A local cache signing key is generated once when one is not supplied.
    The frontend is automatically rebuilt when its source is newer than the
    current production bundle, preventing stale UI changes from being served.

.PARAMETER Port
    Port to bind. Default 8000.

.PARAMETER HostAddr
    Address to bind. Default 0.0.0.0.
#>
param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$HostAddr = "0.0.0.0",
    [string]$Python = (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path $Python)) { Write-Error "Virtualenv python not found at $Python - run scripts\setup-windows.ps1 first."; exit 1 }

$envFile = Join-Path $Root "models.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
    Write-Host "Loaded model paths from models.env" -ForegroundColor DarkGray
}

if (-not $env:APEX_AI_PROVIDER) {
    $env:APEX_AI_PROVIDER = "heavy"
    Write-Host "APEX_AI_PROVIDER unset - using production Heavy AI." -ForegroundColor DarkGray
}
if (-not $env:APEX_AI_DEVICE) {
    $env:APEX_AI_DEVICE = "auto"
    Write-Host "APEX_AI_DEVICE unset - selecting CUDA when available, otherwise CPU." -ForegroundColor DarkGray
}

if (-not $env:APEX_CACHE_SIGNING_KEY) {
    $cacheKeyFile = Join-Path $Root ".apex-cache-signing-key"
    if (Test-Path $cacheKeyFile) {
        $env:APEX_CACHE_SIGNING_KEY = (Get-Content $cacheKeyFile -Raw).Trim()
    }
    else {
        $bytes = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
        $env:APEX_CACHE_SIGNING_KEY = [Convert]::ToBase64String($bytes)
        Set-Content -Path $cacheKeyFile -Value $env:APEX_CACHE_SIGNING_KEY -NoNewline
    }
    Write-Host "Scene analysis cache enabled with local signing key." -ForegroundColor DarkGray
}

$frontend = Join-Path $Root "frontend"
$dist = Join-Path $frontend "dist"
$distIndex = Join-Path $dist "index.html"
$packageJson = Join-Path $frontend "package.json"
$packageLock = Join-Path $frontend "package-lock.json"
$nodeModules = Join-Path $frontend "node_modules"

$needsBuild = -not (Test-Path $distIndex)
if (-not $needsBuild) {
    $bundleTime = (Get-Item $distIndex).LastWriteTimeUtc
    $sourceFiles = Get-ChildItem -Path (Join-Path $frontend "src") -Recurse -File -Include *.ts,*.tsx,*.css,*.html -ErrorAction SilentlyContinue
    $configPaths = @($packageJson, (Join-Path $frontend "vite.config.ts"), (Join-Path $frontend "tsconfig.json")) | Where-Object { Test-Path $_ }
    $configFiles = $configPaths | ForEach-Object { Get-Item $_ }
    $newestSource = @($sourceFiles + $configFiles) | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($newestSource -and $newestSource.LastWriteTimeUtc -gt $bundleTime) { $needsBuild = $true }
}

if ($needsBuild) {
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) { $npmCommand = Get-Command npm -ErrorAction SilentlyContinue }
    if (-not $npmCommand) {
        Write-Error "Node.js/npm is required to build the frontend. Install Node.js or run the frontend build manually."
        exit 1
    }

    $npmPath = $npmCommand.Source
    Write-Host "Using npm: $npmPath" -ForegroundColor DarkGray

    Set-Location $frontend
    if (-not (Test-Path $nodeModules)) {
        if (-not (Test-Path $packageLock)) { Write-Error "frontend/package-lock.json is missing; cannot perform a reproducible frontend install." }
        Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
        & $npmPath ci
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Write-Host "Building latest frontend bundle..." -ForegroundColor Yellow
    & $npmPath run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Set-Location $Root
}
else {
    Write-Host "Frontend bundle is current." -ForegroundColor DarkGray
}

Write-Host "Starting APEX Vision AI on http://$HostAddr`:$Port" -ForegroundColor Green
Write-Host "AI Provider: $env:APEX_AI_PROVIDER | Device: $env:APEX_AI_DEVICE" -ForegroundColor Cyan
Set-Location $Root
& $Python -m uvicorn app.main:app --host $HostAddr --port $Port --workers 1
