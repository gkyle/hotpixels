#!/bin/sh

# Change directory to the script's location
cd "$(dirname "$0")"

# Install UV if not already installed
uv_version=`uv -V`
if [ "$uv_version" = "" ]; then 
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi;

# Install exiftool if not already installed
export PATH="$PATH:$(pwd)/vendor"
exiftool_version=`exiftool -ver`
if [ "$exiftool_version" = "" ]; then 
    echo "Installing exiftool..."
    curl -LsSf https://gigenet.dl.sourceforge.net/project/exiftool/Image-ExifTool-13.42.tar.gz?viasf=1 | tar xz
    mv Image-ExifTool-13.42 vendor
fi;

# Check for compatible GPU.
echo "Determining correct configuration for your GPU..."
torch_variant=`uv run --no-sync src/setup/probeGPU.py`
if [ "$torch_variant" = "" ]; then 
    torch_variant="cpu"
fi;
if [ "$torch_variant" = "cpu" ]; then 
    echo "Setup did not find a compatible GPU. Setup will continue with CPU-only dependencies. You can re-run after installing drivers to enable GPU support."
fi;

# Sync with final GPU configuration
uv sync --extra $torch_variant

# Run
echo "Starting HotPixels..."
uv run --no-sync src/setup/lrcPath.py
uv run --no-sync src/main.py $*