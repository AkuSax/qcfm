$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$BasePython = 'C:\Users\Saxen\AppData\Local\Programs\Python\Python311\python.exe'
$VenvPython = Join-Path $Here 'venv\Scripts\python.exe'
$CachePath = Join-Path $Here 'pip-cache'
$TempPath = Join-Path $Here 'tmp'
New-Item -ItemType Directory -Force -Path $CachePath, $TempPath | Out-Null

$OldPipCache = $env:PIP_CACHE_DIR
$OldTemp = $env:TEMP
$OldTmp = $env:TMP
$env:PIP_CACHE_DIR = $CachePath
$env:TEMP = $TempPath
$env:TMP = $TempPath

try {
    if (-not (Test-Path -LiteralPath (Join-Path $Here 'source\.git'))) {
        & (Join-Path $Here 'checkout_source.ps1')
        if (-not (Test-Path -LiteralPath (Join-Path $Here 'source\.git'))) {
            throw 'Pinned source checkout did not complete.'
        }
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $BasePython -m venv (Join-Path $Here 'venv')
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed with exit code $LASTEXITCODE" }
    }

    & $VenvPython -m pip install --upgrade 'pip==25.3'
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }

    Push-Location $Here
    try {
        & $VenvPython -m pip install -r requirements.lock
        if ($LASTEXITCODE -ne 0) { throw "Locked dependency install failed with exit code $LASTEXITCODE" }
        & $VenvPython receipts.py
        if ($LASTEXITCODE -ne 0) { throw "Runtime receipt update failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $OldPipCache) { Remove-Item Env:PIP_CACHE_DIR -ErrorAction SilentlyContinue } else { $env:PIP_CACHE_DIR = $OldPipCache }
    if ($null -eq $OldTemp) { Remove-Item Env:TEMP -ErrorAction SilentlyContinue } else { $env:TEMP = $OldTemp }
    if ($null -eq $OldTmp) { Remove-Item Env:TMP -ErrorAction SilentlyContinue } else { $env:TMP = $OldTmp }
}
