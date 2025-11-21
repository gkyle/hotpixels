@echo off
Pushd "%~dp0"

echo Starting Hot Pixels...
uv sync --extra cu124
uv run --no-sync src/main.py %*
popd
