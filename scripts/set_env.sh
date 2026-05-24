#!/usr/bin/env bash
# Source this script before running any engine binary. Sets the Ascend toolkit
# paths and the custom_opp vendor path (where the custom_opp .run installer
# was extracted, defaulting to $REPO_ROOT/custom_opp_install/).
#
# Override either path with the env vars MINICPMV_ASCEND_TOOLKIT_ROOT or
# MINICPMV_CUSTOM_OPP_VENDOR if your install lives elsewhere.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export ASCEND_TOOLKIT_ROOT="${MINICPMV_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux}"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_ROOT/lib64:${LD_LIBRARY_PATH:-}"

CUSTOM_OPP_VENDOR="${MINICPMV_CUSTOM_OPP_VENDOR:-$REPO_ROOT/custom_opp_install/vendors/customize}"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
