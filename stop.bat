@echo off
title EmotiLens - Stop All
echo Stopping all EmotiLens processes...
taskkill /FI "WINDOWTITLE eq EmotiLens*" /F >nul 2>&1
taskkill /IM cloudflared.exe /F >nul 2>&1
echo Done. All EmotiLens services stopped.
timeout /t 2 /nobreak >nul
