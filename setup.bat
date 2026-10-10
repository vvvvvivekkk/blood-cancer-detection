@echo off
REM ============================================================
REM  Blood Cancer Detection - first-time setup (Windows)
REM  Installs dependencies, unpacks the trained models, starts
REM  the app. Run this ONCE. After that, use run.bat.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ===============================================
echo   Blood Cancer Detection - Setup
echo ===============================================
echo.

REM --- check Python is on PATH ---
where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python was not found.
  echo.
  echo     Install Python 3.12 from https://www.python.org/downloads/
  echo     IMPORTANT: on the first screen, TICK "Add Python to PATH".
  echo     Then run this setup again.
  echo.
  pause
  exit /b 1
)

echo [1/4] Installing dependencies (this can take 5-10 minutes)...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [X] Dependency install failed. & pause & exit /b 1 )

echo.
echo [2/4] Unpacking the trained models...
python run.py --fetch-models
if errorlevel 1 ( echo [X] Could not unpack the models. & pause & exit /b 1 )

echo.
echo [3/4] Verifying the models...
python run.py --check
if errorlevel 1 ( echo [X] Model verification failed. & pause & exit /b 1 )

echo.
echo [4/4] Starting the app at http://localhost:8000
echo       Log in with  admin / admin
echo       (Close this window to stop the app.)
echo.
python app.py

endlocal
