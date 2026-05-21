#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT/build"
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"
