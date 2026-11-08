@echo off
setlocal
cd /d "%~dp0"
if exist "%USERPROFILE%\.conda\envs\env_ame\python.exe" (
  "%USERPROFILE%\.conda\envs\env_ame\python.exe" bootstrap.py %*
) else (
  python bootstrap.py %*
)
exit /b %errorlevel%
