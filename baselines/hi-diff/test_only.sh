#!/usr/bin/env bash
# HI-Diff: run ONLY the test pass and the collect step.
#   GPU=0 bash test_only.sh <qxs|s2o>
#
# WHY THIS EXISTS.  run.sh is NOT idempotent: both train ymls set
# `resume_state: ~`, no --auto_resume is passed, and basicsr's make_exp_dirs ->
# mkdir_and_rename archives the finished experiments/train_HI_Diff_<TAG>_S{1,2}
# and retrains from scratch.  If training completed and only the publish step
# is missing, re-running run.sh throws away both stages.  This script costs a
# few minutes of GPU and touches no experiment directory.
#
# CHECKPOINT CHOICE: options/test/<TAG>.yml pins net_g_latest.pth,
# net_le_dm_latest.pth and net_d_latest.pth -- iteration 25,000, the LAST
# checkpoint, NOT the best-validation one.  That is deliberate and consistent
# across this table: every other baseline publishes its last checkpoint.  Do
# not repoint it at a best-val file for the paper row.
#
# NOTE the same archive-on-rerun behaviour applies to
# $WORK_DIR/hidiff/results/test_HI_Diff_<TAG>.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/HI-Diff)
#   WORK_DIR        $WORK_DIR/hidiff/results + $WORK_DIR/results + logs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -euo pipefail
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) TAG=QXS; WANT=3999 ;;
  s2o) TAG=S2O; WANT=627 ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
cd "$BASELINES_ROOT/HI-Diff"
OUT=$WORK_DIR/results/${DS}_hidiff_eo
mkdir -p "$OUT" "$WORK_DIR/logs"

$PY test.py -opt options/test/${TAG}.yml 2>&1 | tee "$WORK_DIR/logs/hidiff_${DS}_test.log"
cp -f "$WORK_DIR/hidiff/results/test_HI_Diff_${TAG}/visualization/${TAG}/"*.png "$OUT/"
n=$(ls "$OUT" | wc -l)
echo "[hidiff-$DS] $n/$WANT PNGs in $OUT"
[ "$n" -eq "$WANT" ] || { echo "[hidiff-$DS] WRONG IMAGE COUNT"; exit 1; }
