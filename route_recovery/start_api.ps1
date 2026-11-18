param(
    [int]$Port = 8000,
    [string]$DataRoot = '',
    [string]$UiOrigins = 'http://localhost:8080,http://127.0.0.1:8080'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$interpreter = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) {
    throw 'Create .venv and install route_recovery/requirements-inference.txt first.'
}
if ($DataRoot) { $env:PATHSNAP_DATA_ROOT = $DataRoot }
$env:PATHSNAP_UI_ORIGINS = $UiOrigins
Push-Location $projectRoot
try {
    Write-Host "Laptop 1 API: port $Port; allowed UI origins: $UiOrigins"
    & $interpreter -m uvicorn route_recovery.main:app --host 0.0.0.0 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'API server failed. Read the error above.' }
} finally { Pop-Location }
