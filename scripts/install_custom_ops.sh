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

# The AscendC build script expects ASCEND_HOME_PATH (or ASCEND_AICPU_PATH /
# BASE_LIBS_PATH). Source the CANN env script if available and the user
# hasn't already exported it.
if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_AICPU_PATH:-}" && -z "${BASE_LIBS_PATH:-}" ]]; then
    CANN_ENV="${MINICPMV_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}/bin/setenv.bash"
    if [[ -f "$CANN_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$CANN_ENV"
    else
        export ASCEND_HOME_PATH="${MINICPMV_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
    fi
fi

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
