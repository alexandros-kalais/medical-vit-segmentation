#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

# Path to ops and test
OPS_DIR="/home/akalais/medseg/repo/medical-vit-segmentation/external/utils/ops"
TEST_SCRIPT="/home/akalais/medseg/repo/medical-vit-segmentation/external/test.py"

echo "[INFO] DATA_ROOT=${MEDSEG_DATA_ROOT:-<unset>}"
echo "[INFO] EXPS_ROOT=${MEDSEG_EXPERIMENTS_ROOT:-<unset>}"
echo "[INFO] Ops dir: $OPS_DIR"
echo "[INFO] Test script: $TEST_SCRIPT"

# --- build the CUDA extension if not already built ---
pushd "$OPS_DIR" >/dev/null
python3 setup.py build_ext --inplace
popd >/dev/null

# --- run import test ---
echo "[INFO] Running import test ..."
python3 "$TEST_SCRIPT"
