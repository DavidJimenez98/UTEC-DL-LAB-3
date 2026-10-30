param(
    [ValidateRange(1,100)] [int] $Rounds = 8,
    [ValidateRange(1,100)] [int] $EpochsPerRound = 8,
    [ValidateRange(4,512)] [int] $Batch = 64,
    [ValidateRange(1,1024)] [int] $ExtractBatch = 256,
    [ValidateRange(0,32)] [int] $Workers = 8,
    [ValidateRange(0.01,5)] [double] $MaxHours = 5.0,
    [string] $Output = 'runs_workstation_v3',
    [switch] $ValidationOnly
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project '.conda-reid\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Environment not found: $python" }
$arguments = @(
    (Join-Path $project 'workstation_v3.py'),
    '--rounds', "$Rounds", '--epochs-per-round', "$EpochsPerRound",
    '--batch', "$Batch", '--extract-batch', "$ExtractBatch", '--workers', "$Workers",
    '--max-hours', $MaxHours.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--output', $Output
)
if ($ValidationOnly) { $arguments += '--validation-only' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'v3 failed. See runs_workstation_v3\execution.log and execution_status.json.' }
