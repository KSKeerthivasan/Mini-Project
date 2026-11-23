@echo off
setlocal enabledelayedexpansion
title LocalAIContentStudio

echo =====================================================================
echo                LocalAIContentStudio Launcher
echo       100%% Local Multi-Model AI Generation ^& Refinement
echo =====================================================================
echo.

cd /d "%~dp0"
set "ROOT_DIR=%cd%"

echo [*] Checking Python Virtual Environment...
set "SYSTEM_PYTHON="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYSTEM_PYTHON=py -3.12"
) else (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=py -3"
    ) else (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "SYSTEM_PYTHON=python"
        )
    )
)

set "VENV_OK=0"
if exist "backend\.venv\Scripts\python.exe" (
    backend\.venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if not errorlevel 1 set "VENV_OK=1"
)

if "!VENV_OK!"=="0" (
    echo [*] Virtual environment is missing or invalid for this machine.
    if "!SYSTEM_PYTHON!"=="" (
        echo [ERROR] No compatible Python interpreter found on PATH or via py launcher.
        echo Please install Python 3.12 [64-bit] and ensure it is on your PATH.
        pause
        exit /b 1
    )
    if exist "backend\.venv" (
        echo [*] Repairing virtual environment using !SYSTEM_PYTHON!...
        !SYSTEM_PYTHON! -m venv --upgrade backend\.venv
    ) else (
        echo [*] Creating virtual environment using !SYSTEM_PYTHON!...
        !SYSTEM_PYTHON! -m venv backend\.venv
    )
    if not exist "backend\.venv\Scripts\python.exe" (
        echo [ERROR] Failed to configure virtual environment in backend\.venv
        pause
        exit /b 1
    )
)

backend\.venv\Scripts\python.exe -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [*] Required backend packages not found. Installing from backend\requirements.txt...
    backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies.
        pause
        exit /b 1
    )
)

echo [*] Checking llama.cpp runtime...
if not exist "runtime\llama.cpp\llama-server.exe" (
    echo [ERROR] runtime\llama.cpp\llama-server.exe not found!
    pause
    exit /b 1
)

echo [*] Checking Frontend dependencies...
if not exist "frontend\node_modules" (
    echo [*] Installing frontend npm packages...
    cd frontend
    call npm install
    cd ..
)

echo [*] Starting FastAPI Backend on http://127.0.0.1:8000...
start "LocalAI Backend (FastAPI)" cmd /k "title LocalAI Backend && cd /d "%ROOT_DIR%" && backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [*] Starting React Frontend on http://localhost:5173...
start "LocalAI Frontend (Vite)" cmd /k "title LocalAI Frontend && cd /d "%ROOT_DIR%\frontend" && npm run dev"

echo.
echo [*] Waiting 3 seconds for services to initialize...
timeout /t 3 >nul 2>&1

echo [*] Opening LocalAIContentStudio in browser...
start http://localhost:5173

echo.
echo =====================================================================
echo   LocalAIContentStudio is now running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://127.0.0.1:8000
echo   Docs:     http://127.0.0.1:8000/docs
echo =====================================================================
echo   Close this window or the background windows to stop the servers.
echo =====================================================================
pause

