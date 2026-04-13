@echo off
:: SoloCropper
:: Copyright (c) 2026 Solo
:: Original work by Solo | https://sololo.xyz
:: Set the code page to UTF-8 to avoid garbled text.
chcp 65001 >nul
cd /d "%~dp0"

echo [1/2] Activating virtual environment...
:: Check whether the venv folder exists.
if not exist ".\venv\Scripts\activate.bat" (
    echo [Error] Could not find the venv virtual environment in the current directory.
    echo.
    echo Create a local virtual environment first:
    echo   py -3.11 -m venv venv
    echo   .\venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

:: Activate the virtual environment.
call .\venv\Scripts\activate.bat

echo [2/2] Running SoloCropper.py...
:: Run the Python script.
python SoloCropper.py

echo.
echo Script finished.
pause
