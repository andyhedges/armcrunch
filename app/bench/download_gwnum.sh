#!/usr/bin/env bash
#
# Download the pre-built gwnum_bench binary for the current platform
# from GitHub Releases.
#
# Usage: ./download_gwnum.sh [--force]
#
# The binary is downloaded to the same directory as this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="andyhedges/armcrunch"
FORCE=false

if [ "${1:-}" = "--force" ]; then
    FORCE=true
fi

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux-amd64"
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # Apple Silicon: default to native ARM64 build
    PLATFORM="macos-arm64"
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
    PLATFORM="macos-amd64"
else
    echo "Error: No pre-built gwnum_bench available for $OS/$ARCH" >&2
    exit 1
fi

BINARY_NAME="gwnum_bench-${PLATFORM}"
TARGET_PATH="${SCRIPT_DIR}/${BINARY_NAME}"

# Check if already downloaded
if [ -x "$TARGET_PATH" ] && [ "$FORCE" = false ]; then
    echo "gwnum_bench already exists at $TARGET_PATH"
    exit 0
fi

echo "Downloading gwnum_bench for ${PLATFORM}..."

# Try to get the latest release download URL
DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${BINARY_NAME}"

if command -v curl &>/dev/null; then
    HTTP_CODE=$(curl -sL -w "%{http_code}" -o "$TARGET_PATH" "$DOWNLOAD_URL")
    if [ "$HTTP_CODE" != "200" ]; then
        rm -f "$TARGET_PATH"
        echo "Error: Failed to download ${BINARY_NAME} (HTTP ${HTTP_CODE})" >&2
        echo "" >&2
        echo "No GitHub Release found. To create one:" >&2
        echo "  1. Push a tag:  git tag gwnum-v1 && git push origin gwnum-v1" >&2
        echo "  2. Wait for the GitHub Actions workflow to complete" >&2
        echo "  3. Re-run this script" >&2
        echo "" >&2
        echo "Or build locally: see bench/README.md" >&2
        exit 1
    fi
elif command -v wget &>/dev/null; then
    wget -q -O "$TARGET_PATH" "$DOWNLOAD_URL" || {
        rm -f "$TARGET_PATH"
        echo "Error: Failed to download ${BINARY_NAME}" >&2
        exit 1
    }
else
    echo "Error: Neither curl nor wget found" >&2
    exit 1
fi

chmod +x "$TARGET_PATH"
echo "Downloaded $TARGET_PATH"

# On macOS Apple Silicon, also download the Rosetta (amd64) version
# for side-by-side comparison
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    ROSETTA_NAME="gwnum_bench-macos-amd64"
    ROSETTA_PATH="${SCRIPT_DIR}/${ROSETTA_NAME}"
    if [ ! -x "$ROSETTA_PATH" ] || [ "$FORCE" = true ]; then
        echo "Also downloading Rosetta (x86_64) version for comparison..."
        ROSETTA_URL="https://github.com/${REPO}/releases/latest/download/${ROSETTA_NAME}"
        curl -sL -o "$ROSETTA_PATH" "$ROSETTA_URL" && chmod +x "$ROSETTA_PATH" \
            && echo "Downloaded $ROSETTA_PATH" \
            || echo "Warning: Could not download Rosetta version (optional)"
    fi
fi