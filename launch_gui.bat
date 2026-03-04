@echo off
REM One-click launcher: API + GUI to play Patchwork vs AlphaZero
REM API starts with checkpoints\latest_model.pt (always the latest)

set REPO=%~dp0
cd /d "%REPO%"

set MODEL=checkpoints\latest_model.pt
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Make sure checkpoints\latest_model.pt exists in the repo.
  pause
  exit /b 1
)

REM Activate venv if present
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

echo Starting API with latest_model.pt...
start "Patchwork API" cmd /k "cd /d "%REPO%" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && python GUI/patchwork_api.py --model "%MODEL%" --config configs/config_best.yaml --simulations 800 --host 127.0.0.1 --port 8000"

echo Waiting for API to start...
timeout /t 4 /nobreak >nul

echo Starting GUI dev server...
start "Patchwork GUI" cmd /k "cd /d "%REPO%\GUI" && (if exist ..\venv\Scripts\activate.bat call ..\venv\Scripts\activate.bat) && npm run dev"

echo Waiting for GUI to start...
timeout /t 6 /nobreak >nul
start http://localhost:5173

echo.
echo API: http://localhost:8000  (latest_model.pt)
echo GUI: http://localhost:5173
echo Close the API and GUI windows when done.
pause
