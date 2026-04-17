@echo off
REM One-click launcher: API + GUI to play Patchwork vs AlphaZero
REM Max strength: E_max_pack (config_gui_max_strength.yaml) + latest committed model + 2048 sims

set REPO=%~dp0
cd /d "%REPO%"

REM Activate venv if present
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

REM Resolve latest checkpoint (highest committed iter_XXX, else checkpoints\latest_model.pt)
for /f "delims=" %%i in ('python tools\get_latest_checkpoint.py 2^>nul') do set MODEL=%%i
if not defined MODEL set MODEL=checkpoints\latest_model.pt
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Run training first or place a checkpoint in checkpoints\latest_model.pt
  pause
  exit /b 1
)
echo Using latest model: %MODEL%

REM E_max_pack + 2048 sims (config_gui_max_strength.yaml has packing preset E)
echo Starting API with E_max_pack, 2048 sims...
start "Patchwork API" cmd /k "cd /d "%REPO%" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && python GUI/patchwork_api.py --model "%MODEL%" --config configs/config_gui_max_strength.yaml --simulations 2048 --host 127.0.0.1 --port 8000"

echo Waiting for API to start...
timeout /t 4 /nobreak >nul

echo Starting GUI dev server...
start "Patchwork GUI" cmd /k "cd /d "%REPO%\GUI" && (if exist ..\venv\Scripts\activate.bat call ..\venv\Scripts\activate.bat) && npm run dev"

echo Waiting for GUI to start...
timeout /t 6 /nobreak >nul
start http://localhost:5173

echo.
echo API: http://localhost:8000  (latest model, E_max_pack, 2048 sims)
echo GUI: http://localhost:5173
echo Close the API and GUI windows when done.
pause
