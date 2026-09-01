#!/usr/bin/env bash
# pix2pix (junyanz/pytorch-CycleGAN-and-pix2pix) SAR->EO on SAR2Opt.
# 600px native tiles, random 512 crops at train time; inference on the frozen
# center-512 test set, which is the evaluation protocol for this dataset.
#
# BUDGET: batch 8, 100 + 100 epochs, ceil(1450/8) = 182 it/epoch
#         -> 200 x 182 = 36,400 GENERATOR UPDATES.
#         (Loader has no drop_last -> ceil. This is the small-set convention
#         inherited from the upstream default, and it is well under the
#         120k QXS-SAROPT reference -- quote updates, not epochs.)
#
# Data: $DATA_ROOT/S2O_AB_combined       train, A|B 1200x600 (combine_A_and_B.py)
#       $DATA_ROOT/S2O_AB_combined_test512  test, A|B 1024x512 (center crops)
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (S2O_AB_combined, S2O_AB_combined_test512)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/s2o_pix2pix/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
CK=$CKPT_ROOT/gan_ckpts
LOG=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$CK" "$LOG" "$WORK_DIR/results"
cd "$REPO"

echo "[s2o-pix2pix] $(date '+%F %T') train"
$PY train.py --dataroot "$DATA_ROOT/S2O_AB_combined" --name s2o_pix2pix --model pix2pix \
  --direction AtoB --batch_size 8 --load_size 600 --crop_size 512 \
  --n_epochs 100 --n_epochs_decay 100 --num_threads 8 --no_html \
  --checkpoints_dir "$CK" > "$LOG/s2o_pix2pix_train.log" 2>&1 \
&& $PY test.py --dataroot "$DATA_ROOT/S2O_AB_combined_test512" --name s2o_pix2pix --model pix2pix \
  --direction AtoB --num_test 99999 --load_size 512 --crop_size 512 \
  --checkpoints_dir "$CK" --results_dir "$WORK_DIR/results/s2o_pix2pix/" \
  > "$LOG/s2o_pix2pix_test.log" 2>&1
echo "[s2o-pix2pix] $(date '+%F %T') exit $?"
