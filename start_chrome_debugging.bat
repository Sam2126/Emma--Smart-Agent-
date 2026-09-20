@echo off
echo ===================================================
echo   Opening the agent's Chrome window (port 9222)
echo ===================================================
echo.
echo This window uses the agent's own profile. Sign in there, once, to the
echo sites the agent should use. Your own Chrome profiles are not touched
echo and can stay open next to it.
echo.

"%~dp0backend\.venv\Scripts\python.exe" "%~dp0backend\scripts\launch_linked_chrome.py"
