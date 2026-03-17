@echo off
echo Starting AI Proctoring System...
echo.

:: Start dashboard in background
start "Dashboard" cmd /k "cd /d %~dp0 && proctor_env\Scripts\activate && streamlit run dashboard.py"

:: Wait 3 seconds for dashboard to load
timeout /t 3 /nobreak > nul

:: Start proctoring engine
echo Starting proctoring engine...
call proctor_env\Scripts\activate
python proctor_v2.py