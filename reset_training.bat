@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: SAFETY: Only run from project root (src + configs exist)
if not exist "src\" (
  echo ERROR: src\ not found. Not in project root. Aborting.
  pause
  exit /b 1
)
if not exist "configs\" (
  echo ERROR: configs\ not found. Not in project root. Aborting.
  pause
  exit /b 1
)

:: Skip confirmation if -y or --yes passed (for scripting)
set "SKIP_CONFIRM="
if /i "%1"=="-y" set SKIP_CONFIRM=1
if /i "%1"=="--yes" set SKIP_CONFIRM=1

echo ========================================
echo   Patchwork Az - Reset Training Data
echo ========================================
echo.
echo REMOVES (training/tool output only):
echo   - runs\           per-run state, staging, committed iters
echo   - checkpoints\    best_model.pt, latest_model.pt
echo   - logs\           training.log, tensorboard, metadata, config snapshots
echo   - data\           replay_buffer, selfplay (if used)
echo   - tuning_2d\      Optuna trial dirs
echo   - wandb\          Weights ^& Biases run data
echo   - elo_ratings.json
echo   - bench_results.json
echo.
echo PRESERVED (never touched):
echo   - src\, configs\, tools\, tests\, scripts\
echo.
if not defined SKIP_CONFIRM (
  set /p "CONFIRM=Type Y to confirm and delete (anything else cancels): "
  if /i not "!CONFIRM!"=="Y" (
    echo Cancelled.
    exit /b 0
  )
)
echo.
echo Removing...
echo.

if exist runs (rmdir /s /q runs 2>nul && echo   [OK] runs || echo   [FAIL] runs) else echo   [skip] runs
if exist checkpoints (rmdir /s /q checkpoints 2>nul && echo   [OK] checkpoints || echo   [FAIL] checkpoints) else echo   [skip] checkpoints
if exist logs (rmdir /s /q logs 2>nul && echo   [OK] logs || echo   [FAIL] logs) else echo   [skip] logs
if exist data (rmdir /s /q data 2>nul && echo   [OK] data || echo   [FAIL] data) else echo   [skip] data
if exist tuning_2d (rmdir /s /q tuning_2d 2>nul && echo   [OK] tuning_2d || echo   [FAIL] tuning_2d) else echo   [skip] tuning_2d
if exist wandb (rmdir /s /q wandb 2>nul && echo   [OK] wandb || echo   [FAIL] wandb) else echo   [skip] wandb

if exist elo_ratings.json (del /q elo_ratings.json 2>nul && echo   [OK] elo_ratings.json || echo   [FAIL] elo_ratings.json) else echo   [skip] elo_ratings.json
if exist bench_results.json (del /q bench_results.json 2>nul && echo   [OK] bench_results.json || echo   [FAIL] bench_results.json) else echo   [skip] bench_results.json

echo.
echo ========================================
echo   Reset complete. Ready for fresh training.
echo ========================================
echo.
pause
