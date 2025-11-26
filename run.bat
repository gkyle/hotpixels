@echo off
Pushd "%~dp0"

powershell.exe -ExecutionPolicy Bypass -File "src/setup/setup.ps1"

echo Starting Hot Pixels...
uv sync --extra cu124
uv run --no-sync src/setup/lrc_path.py
uv run --no-sync src/main.py %*
popd
