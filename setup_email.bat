@echo off
echo ===================================================
echo   Email setup for the Self-Improving Agent
echo ===================================================
echo.
echo Lets the agent read, draft and send email from your own mailbox.
echo Gmail needs an App Password: https://myaccount.google.com/apppasswords
echo.

"%~dp0backend\.venv\Scripts\python.exe" "%~dp0backend\scripts\setup_email.py"
echo.
pause
