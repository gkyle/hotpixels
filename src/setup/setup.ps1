try {
    Write-Host "Checking for uv..."
    Invoke-Expression -Command "uv -V"
}
catch {
    Invoke-Expression -Command  "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH += ";$env:USERPROFILE\.local\bin"
}

$torch_variant = Invoke-Expression -Command "uv run --no-sync src/setup/probeGPU.py"
Write-Host "Using torch variant: $torch_variant"
if ($torch_variant.Length -eq 0 -or $torch_variant -Match "not found") {
    $torch_variant = "cpu"
}
if ($torch_variant -eq "cpu") {
    Write-Host "Setup did not find a compatible GPU. Setup will continue with CPU-only dependencies. You can re-run after installing drivers to enable GPU support."
}

Invoke-Expression -Command "uv sync --extra $torch_variant"

try {
    Write-Host "Checking for exiftool..."
    # Put .\vendor on PATH for this session
    $env:PATH += ";$PWD\vendor"
    Invoke-Expression -Command "exiftool -ver"
}
catch {
    Write-Host "Downloading and installing exiftool..."
    # Create vendor directory if it doesn't exist
    if (-Not (Test-Path -Path ".\vendor")) {
        New-Item -ItemType Directory -Path ".\vendor" | Out-Null
    }
    Invoke-WebRequest -Uri "https://cytranet-dal.dl.sourceforge.net/project/exiftool/exiftool-13.42_64.zip?viasf=1" -OutFile "exiftool.zip"
    Expand-Archive -Path "exiftool.zip" -DestinationPath ".\" -Force
    # Move the .exe and exiftool_files to .\vendor
    Move-Item -Path ".\exiftool-*\exiftool*.exe" -Destination ".\vendor\exiftool.exe" -Force
    Move-Item -Path ".\exiftool-*\exiftool_files" -Destination ".\vendor\exiftool_files" -Force
    # Clean up
    Remove-Item -Path "exiftool.zip"
    Remove-Item -Path ".\exiftool-*" -Recurse -Force
}

