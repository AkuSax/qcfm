param(
    [Parameter(Mandatory=$true)][string]$GateReceipt,
    [Parameter(Mandatory=$true)][string]$Features,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$ReceiptDirectory
)
$ErrorActionPreference = 'Stop'
$Package = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Package 'arm3_environment/venv/Scripts/python.exe'
$Bounded = Join-Path $Package 'orchestrator/run_bounded.py'
$Script = Join-Path $PSScriptRoot 'cpu_thread_bootstrap.py'
foreach ($Path in @($GateReceipt,$Features,$RunDirectory,$ReceiptDirectory,$Python,$Bounded,$Script)) {
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) { throw "Use absolute paths: $Path" }
}
$PackageFull = [System.IO.Path]::GetFullPath($Package).TrimEnd('\') + '\'
foreach ($Path in @($GateReceipt,$Features,$RunDirectory,$ReceiptDirectory)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith($PackageFull,[System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Keep inputs, fit outputs, and process receipts under the preparation package: $Path"
    }
}
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:TORCH_NUM_THREADS = '1'
$env:PYTHONPATH = Join-Path $Package 'arm4_protocol'
& $Python $Bounded --max-seconds 14400 --receipt-dir $ReceiptDirectory --script $Script -- --authorize-fitting --gate-receipt $GateReceipt --features $Features --output $RunDirectory --device cpu
exit $LASTEXITCODE
