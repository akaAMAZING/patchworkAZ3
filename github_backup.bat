@echo off
color 0A
echo ==========================================
echo    PatchworkAZ - Auto GitHub Backup
echo ==========================================
echo.

:: Stage all files
echo [1/3] Staging files...
git add .

:: Create a timestamp and commit
echo [2/3] Saving local snapshot...
set TIMESTAMP=%DATE% %TIME%
git commit -m "Auto-backup: %TIMESTAMP%"

:: Push to your online repo
echo [3/3] Uploading to GitHub...
git push origin main

echo.
echo ==========================================
echo    BACKUP COMPLETE!
echo ==========================================
pause