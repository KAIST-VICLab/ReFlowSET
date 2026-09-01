#!/usr/bin/env bash
# HI-Diff (zhengchen1999/HI-Diff, NeurIPS 2023) SAR->EO.
#   GPU=0 bash run.sh <qxs|s2o>
# LQ = SAR (grayscale, replicated to 3 channels by cv2.IMREAD_COLOR),
# GT = EO.  Stage 1 trains the Transformer + a latent encoder that sees LQ||GT;
# stage 2 replaces that encoder with an LQ-only one and learns an 8-step latent
# denoiser that produces the prior at test time.
#
# BUDGET: S1 25,000 + S2 25,000 = 50,000 iterations, batch 8 @ 256, both
#   datasets.  The official recipe is 300k + 300k with a progressive 128->384
#   schedule (16@128 down to 2@384); progressive training is DISABLED here and
#   the budget is matched to the other diffusion rows.  Batch 16 @ 256 does not
#   fit even on a very large card.
#   SAR2Opt trains on random 256 crops of its 600px tiles (1450 tiles) and is
#   TESTED on the 627 center-512 crops -- the model is fully convolutional.
#
# !!! THIS SCRIPT IS NOT IDEMPOTENT !!!  Both train configs set
#   `resume_state: ~` and no --auto_resume is passed, so re-running RETRAINS
#   FROM SCRATCH -- and basicsr's make_exp_dirs -> mkdir_and_rename first
#   renames the finished experiments/train_HI_Diff_<TAG>_S{1,2} to
#   *_archived_<timestamp>.  Re-running a finished cell destroys ~15 GPU-hours
#   and spends as many again.  If only the test pass is missing, use
#   test_only.sh instead.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/HI-Diff)
#   DATA_ROOT       prepared data (QXS_AB, sar2opt, <ds>_hidiff val split, s2o_512)
#   WORK_DIR        $WORK_DIR/hidiff/{experiments,results} + logs + results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
# The option ymls in configs/ carry ${DATA_ROOT} / ${WORK_DIR} placeholders;
# expand them with envsubst into $BASELINES_ROOT/HI-Diff/options/ before the
# first run (see ../README.md), and make the repo's experiments/ and results/
# point at $WORK_DIR/hidiff/ (symlink or edit the ymls).
set -euo pipefail
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) TAG=QXS ;;
  s2o) TAG=S2O ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/HI-Diff
LOGDIR=$WORK_DIR/logs
OUT=$WORK_DIR/results/${DS}_hidiff_eo
mkdir -p "$OUT" "$LOGDIR"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

echo "[hidiff-$DS] Stage 1: Transformer + latent encoder, 25k iters (GPU $GPU)"
$PY train.py -opt options/train/${TAG}_S1.yml 2>&1 | tee "$LOGDIR/hidiff_${DS}_s1.log"

echo "[hidiff-$DS] Stage 2: + diffusion prior, 25k iters (loads S1 net_g/net_le)"
$PY train.py -opt options/train/${TAG}_S2.yml 2>&1 | tee "$LOGDIR/hidiff_${DS}_s2.log"

echo "[hidiff-$DS] Test: full test split -> EO predictions"
$PY test.py -opt options/test/${TAG}.yml 2>&1 | tee "$LOGDIR/hidiff_${DS}_test.log"

cp -f "$WORK_DIR/hidiff/results/test_HI_Diff_${TAG}/visualization/${TAG}/"*.png "$OUT/"
echo "[hidiff-$DS] done: $(ls "$OUT" | wc -l) PNGs in $OUT"
