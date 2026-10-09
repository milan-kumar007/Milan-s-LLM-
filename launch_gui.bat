@echo off
setlocal
cd /d "%~dp0"
title Milan's LLM Desktop Playground

echo ========================================================
echo  Launching Milan's LLM Desktop GUI on NVIDIA GPU...
echo ========================================================

if not exist "%~dp0.venv\Scripts\python.exe" goto :no_venv

echo [OK] Python virtual environment detected.
echo Starting desktop interface...
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\gui_app.py"
goto :done

:no_venv
echo [ERROR] Virtual environment not found at .venv!
echo Please ensure the .venv folder exists in this directory.

:done
echo.
echo ========================================================
echo  GUI process has closed.
echo ========================================================
pause
