@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.cmd first.
  exit /b 1
)
".venv\Scripts\python.exe" -m workbench.cli start --open %*
