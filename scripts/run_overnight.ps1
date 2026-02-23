# PATCHWORK ALPHAZERO — OVERNIGHT LAUNCHER (PowerShell)
# =====================================================
#
# USAGE:
#   .\run_overnight.ps1                              # defaults: config_best.yaml
#   .\run_overnight.ps1 -Config configs\config_best.yaml
#   .\run_overnight.ps1 -SkipPreflight               # skip preflight checks
#   .\run_overnight.ps1 -BenchmarkOnly               # just run benchmark
#
# This script:
#   1. Runs preflight checks (hardware, config, sanity, smoke test)
#   2. Starts the training pipeline
#   3. Logs everything to a timestamped log file

param(
    [string]$Config = "configs\config_best.yaml",
    [switch]$SkipPreflight,
    [switch]$BenchmarkOnly,
    [switch]$QuickPreflight
)

$ErrorActionPreference = "Stop"

# Timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logDir = "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "run_$timestamp.log"

function Log {
    param([string]$msg)
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# Header
Log "=========================================="
Log "PATCHWORK ALPHAZERO — OVERNIGHT RUN"
Log "=========================================="
Log "Config:    $Config"
Log "Timestamp: $timestamp"
Log "Log file:  $logFile"
Log "Python:    $(python --version 2>&1)"
Log "=========================================="

# Check config exists
if (-not (Test-Path $Config)) {
    Log "ERROR: Config file not found: $Config"
    exit 1
}

# Benchmark only mode
if ($BenchmarkOnly) {
    Log ""
    Log "Running benchmark..."
    python tools\benchmark.py --config $Config --output "logs\benchmark_$timestamp.json" 2>&1 | Tee-Object -Append -FilePath $logFile
    Log "Benchmark complete. Results in logs\benchmark_$timestamp.json"
    exit $LASTEXITCODE
}

# Step 1: Preflight
if (-not $SkipPreflight) {
    Log ""
    Log "[STEP 1/2] Running preflight checks..."
    $preflightArgs = @("tools\preflight.py", "--config", $Config)
    if ($QuickPreflight) { $preflightArgs += "--skip-smoke" }

    python @preflightArgs 2>&1 | Tee-Object -Append -FilePath $logFile
    if ($LASTEXITCODE -ne 0) {
        Log ""
        Log "PREFLIGHT FAILED — aborting overnight run."
        Log "Fix the issues above and re-run."
        exit 1
    }
    Log "Preflight passed."
} else {
    Log ""
    Log "[STEP 1/2] Preflight SKIPPED (--SkipPreflight)"
}

# Step 2: Training
Log ""
Log "[STEP 2/2] Starting training pipeline..."
Log "Press Ctrl+C to stop gracefully."
Log ""

$startTime = Get-Date

try {
    python -m src.training.main --config $Config 2>&1 | Tee-Object -Append -FilePath $logFile
    $exitCode = $LASTEXITCODE
} catch {
    Log "ERROR: Training crashed with exception: $_"
    $exitCode = 1
}

$duration = (Get-Date) - $startTime
Log ""
Log "=========================================="
Log "Training finished."
Log "Exit code:   $exitCode"
Log "Duration:    $($duration.ToString('hh\:mm\:ss'))"
Log "Log file:    $logFile"
Log "=========================================="

exit $exitCode
