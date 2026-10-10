@echo off
REM Blood Cancer Detection - start the app (Windows). Run setup.bat first.
setlocal
cd /d "%~dp0"
where python >nul 2>&1 || ( echo [X] Python not found. Run setup.bat first. & pause & exit /b 1 )
echo Starting the app at http://localhost:8000  (log in: admin / admin)
echo Close this window to stop it.
python app.py
endlocal
