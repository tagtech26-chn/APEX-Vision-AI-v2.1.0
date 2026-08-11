<#
.SYNOPSIS
    Starts the APEX Vision AI production server (single service: SPA + API).
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
    $npmCommand = $null
    $npmCmdPath = Join-Path $env:ProgramFiles "nodejs\npm.cmd"
    if (Test-Path $npmCmdPath) {
        $npmCommand = $npmCmdPath
    }
    else {
        $npmCommandInfo = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if ($npmCommandInfo) { $npmCommand = $npmCommandInfo.Source }
    }
    if (-not $npmCommand) {
        $npmCommandInfo = Get-Command npm -ErrorAction SilentlyContinue
        if ($npmCommandInfo) { $npmCommand = $npmCommandInfo.Source }
    }
    if (-not $npmCommand) {
        Write-Error "Node.js/npm is required to build the frontend. Install Node.js and ensure npm.cmd is available."
        exit 1
    }

    Write-Host "Using npm: $npmCommand" -ForegroundColor DarkGray
    Set-Location $frontend

    if (-not (Test-Path $nodeModules)) {
        if (-not (Test-Path $packageLock)) { Write-Error "frontend/package-lock.json is missing; cannot perform a reproducible frontend install." }
        Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
        & $npmCommand ci
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Host "Building latest frontend bundle..." -ForegroundColor Yellow
    & $npmCommand run build
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
