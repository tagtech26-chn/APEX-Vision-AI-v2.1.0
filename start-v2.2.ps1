<#
.SYNOPSIS
    Starts the isolated APEX Vision AI v2.2 AI geometry lab.
    This never changes the production v2.1 branch/runtime configuration.
#>
param(
    [ValidateRange(1, 65535)][int]$Port = 8010,
    [string]$HostAddr = "0.0.0.0",
    [string]$Python = "D:\v22env\Scripts\python.exe",
    [ValidateSet("metric3d", "unidepth")][string]$Depth = "metric3d",
    [ValidateSet("off", "gemini")][string]$GeometryAdvisor = "off"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path -Path $Root -ChildPath $Python
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "v2.2 Python not found at $Python. Create the isolated environment first."
}

$env:APEX_AI_PROVIDER = "v22"
$env:APEX_VERSION = "2.2.0-ai-geometry-lab"
$env:APEX_V22_DEPTH = $Depth
$env:APEX_V22_GEOMETRY_ADVISOR = $GeometryAdvisor
$env:APEX_V22_DEVICE = if ($env:APEX_V22_DEVICE) { $env:APEX_V22_DEVICE } else { "cpu" }

Write-Host "APEX Vision AI v2.2 AI Geometry Lab" -ForegroundColor Green
Write-Host "Python: $Python" -ForegroundColor DarkGray
Write-Host "Provider: v22 | Segmentation: SAM3 concept | Depth: $Depth | Device: $env:APEX_V22_DEVICE" -ForegroundColor Cyan
Write-Host "Spatial advisor: $GeometryAdvisor" -ForegroundColor Cyan
Write-Host "SAM3.1 multiplex is reserved for the future video/multi-object path." -ForegroundColor DarkGray
Write-Host "Production v2.1 environment remains untouched." -ForegroundColor DarkGray

& (Join-Path $Root "start.ps1") -Port $Port -HostAddr $HostAddr -Python $Python
exit $LASTEXITCODE
