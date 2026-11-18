param([int]$Port = 8080)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$interpreter = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) {
    throw 'Create .venv and install route_recovery/requirements-ui.txt first.'
}
Push-Location $projectRoot
try {
    Write-Host "Laptop 2 UI: http://localhost:$Port; enter Laptop 1's API address in Connect."
    & $interpreter -m uvicorn route_recovery.ui_server:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'UI server failed. Read the error above.' }
} finally { Pop-Location }
