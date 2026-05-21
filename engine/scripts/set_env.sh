#!/usr/bin/env bash
set -euo pipefail
export ASCEND_TOOLKIT_ROOT="/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="/mnt/data/minicpm/custom_opp_install/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}"
