@echo off
REM One-click launcher: API + GUI to play Patchwork vs AlphaZero
REM Best model: iter69 in committed. Fallback: checkpoints\latest_model.pt

set REPO=%~dp0
cd /d "%REPO%"

REM Prefer explicit best-iter (iter69); fallback to latest_model.pt
set ITER69=runs\patchwork_production\committed\iter_069\iteration_069.pt
set LATEST=checkpoints\latest_model.pt
if exist "%ITER69%" (
  set MODEL=%ITER69%
  echo Using best model: iter_069
) else (
  set MODEL=%LATEST%
  echo Using fallback: latest_model.pt (ensure iter69 exists for best strength)
)
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Provide either %ITER69% or %LATEST%
  pause
  exit /b 1
)

REM Activate venv if present
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

REM 3000 sims = strong play (GUI/API can override via Load / request body)
echo Starting API with %MODEL% (3000 sims)...
start "Patchwork API" cmd /k "cd /d "%REPO%" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && python GUI/patchwork_api.py --model "%MODEL%" --config configs/config_best.yaml --simulations 3000 --host 127.0.0.1 --port 8000"

echo Waiting for API to start...
timeout /t 4 /nobreak >nul

echo Starting GUI dev server...
start "Patchwork GUI" cmd /k "cd /d "%REPO%\GUI" && (if exist ..\venv\Scripts\activate.bat call ..\venv\Scripts\activate.bat) && npm run dev"

echo Waiting for GUI to start...
timeout /t 6 /nobreak >nul
start http://localhost:5173

echo.
echo API: http://localhost:8000  (%MODEL%, 3000 sims)
echo GUI: http://localhost:5173
echo Close the API and GUI windows when done.
pause
