$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Files = @('setup.ps1', 'checkout_source.ps1', 'run_three_ligand_smoke.ps1')
$Rows = foreach ($Name in $Files) {
    $FullPath = Join-Path $Here $Name
    $Tokens = $null
    $Errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($FullPath, [ref]$Tokens, [ref]$Errors) | Out-Null
    [pscustomobject]@{ file = $Name; error_count = $Errors.Count; errors = @($Errors | ForEach-Object { $_.Message }) }
    if ($Errors.Count -gt 0) { throw "PowerShell parse failed: $Name" }
}
$Rows | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $Here 'powershell_parse_receipt.json')
