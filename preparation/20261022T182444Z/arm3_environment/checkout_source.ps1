$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Here 'source'
$Commit = 'd14692839f4e5544f4eb38c15cf01298c27960f8'
if (Test-Path -LiteralPath (Join-Path $Source '.git')) {
    $Current = (& git -C $Source rev-parse HEAD).Trim()
    if ($Current -ne $Commit) { throw "Existing source checkout has commit $Current, expected $Commit" }
    Write-Output "Source checkout already matches $Commit"
    exit 0
}
if (Test-Path -LiteralPath $Source) { throw "Source path exists but is not a Git checkout: $Source" }
& git clone --filter=blob:none --no-checkout https://github.com/isayevlab/aimnetcentral.git $Source
if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
& git -C $Source checkout $Commit
if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed' }
