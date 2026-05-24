#!/usr/bin/env bash
# Build and install the AscendC custom operators under
# $REPO_ROOT/custom_opp_install/ (or $CUSTOM_OPP_INSTALL_DIR if set).
#
# The installed location matches the default `CUSTOM_OPP_VENDOR` in the root
# CMakeLists and the `ASCEND_CUSTOM_OPP_PATH` exported by scripts/set_env.sh,
# so engine builds will pick it up automatically.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUSTOM_OPS_DIR="$REPO_ROOT/src/csrc/custom_ops"
INSTALL_DIR="${CUSTOM_OPP_INSTALL_DIR:-$REPO_ROOT/custom_opp_install}"

cd "$CUSTOM_OPS_DIR"
./build.sh

RUN_PKG="$CUSTOM_OPS_DIR/build_out/custom_opp_ubuntu_aarch64.run"
if [[ ! -f "$RUN_PKG" ]]; then
  echo "ERROR: $RUN_PKG was not produced by build.sh" >&2
  exit 1
fi
mkdir -p "$INSTALL_DIR"
"$RUN_PKG" --install-path="$INSTALL_DIR" --quiet
echo "Custom ops installed at $INSTALL_DIR"
