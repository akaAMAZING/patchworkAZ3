@echo off
REM One-click launcher: API + GUI to play Patchwork vs AlphaZero
REM Edit ITER below to play vs a different iteration (e.g. 59 for iter59)

set ITER=59
set REPO=%~dp0
cd /d "%REPO%"

REM Zero-pad ITER to 3 digits (59 -> 059)
set ITER3=%ITER%
if %ITER% LSS 10 set ITER3=00%ITER%
if %ITER% GEQ 10 if %ITER% LSS 100 set ITER3=0%ITER%

set MODEL=runs\patchwork_production\committed\iter_%ITER3%\iteration_%ITER3%.pt
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Check that iter %ITER% exists in runs\patchwork_production\committed\
  pause
  exit /b 1
)

REM Activate venv if present
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

echo Starting API with iter%ITER3%...
start "Patchwork API (iter%ITER3%)" cmd /k "cd /d "%REPO%" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && python GUI/patchwork_api.py --model "%MODEL%" --config configs/config_best.yaml --simulations 800 --host 127.0.0.1 --port 8000"

echo Waiting for API to start...
timeout /t 4 /nobreak >nul

echo Starting GUI dev server...
start "Patchwork GUI" cmd /k "cd /d "%REPO%\GUI" && (if exist ..\venv\Scripts\activate.bat call ..\venv\Scripts\activate.bat) && npm run dev"

echo Waiting for GUI to start...
timeout /t 6 /nobreak >nul
start http://localhost:5173

echo.
echo API (iter%ITER3%): http://localhost:8000
echo GUI: http://localhost:5173
echo Close the API and GUI windows when done.
pause
