@echo off
title EmotiLens Launcher
color 0A
echo.
echo  ============================================
echo   EmotiLens - Starting all services...
echo  ============================================
echo.

:: 1. Start FastAPI (Python backend) in its own window
echo [1/3] Starting FastAPI backend on port 8000...
start "EmotiLens - FastAPI (port 8000)" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"

:: Wait a moment for FastAPI to start loading
timeout /t 3 /nobreak >nul

:: 2. Start Node server in its own window
echo [2/3] Starting Node frontend on port 3000...
start "EmotiLens - Node (port 3000)" cmd /k "cd /d %~dp0\app && node server.js"

:: Wait for Node to be ready
timeout /t 2 /nobreak >nul

:: 3. Start Cloudflare tunnel
echo [3/3] Starting Cloudflare tunnel...
echo.
echo  ============================================
echo   Copy the trycloudflare.com URL below
echo   and open it on your phone!
echo  ============================================
echo.
cloudflared tunnel --url http://localhost:3000

pause
