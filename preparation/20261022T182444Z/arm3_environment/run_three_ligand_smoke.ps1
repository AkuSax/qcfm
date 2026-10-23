param(
    [Parameter(Mandatory = $true)][switch]$Execute,
    [Parameter(Mandatory = $true)][string]$Ligand1,
    [Parameter(Mandatory = $true)][string]$Ligand2,
    [Parameter(Mandatory = $true)][string]$Ligand3,
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu'
)
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Here 'venv\Scripts\python.exe'
foreach ($Path in @($Ligand1, $Ligand2, $Ligand3)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ligand manifest: $Path"
    }
}
& $Python (Join-Path $Here 'smoke_three.py') --execute `
    --input $Ligand1 $Ligand2 $Ligand3 --output $Output --device $Device
if ($LASTEXITCODE -ne 0) { throw "AIMNet2 smoke failed with exit code $LASTEXITCODE" }
