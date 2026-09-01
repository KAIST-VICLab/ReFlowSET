#!/usr/bin/env bash
# CycleGAN (junyanz/pytorch-CycleGAN-and-pix2pix) SAR->EO on SAR2Opt.
# 600px native tiles, random 512 crops at train time; inference on the frozen
# center-512 test set (the evaluation protocol crop for this dataset).
#
# BUDGET: batch 4, 100 + 100 epochs, ceil(1450/4) = 363 it/epoch
#         -> 200 x 363 = 72,600 GENERATOR UPDATES.
#         (No drop_last in this loader -> ceil.)
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (sar2opt native tiles; S2O_test512 crops)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/s2o_cyclegan/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
S2O=$DATA_ROOT/sar2opt
CK=$CKPT_ROOT/gan_ckpts
LOG=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$CK" "$LOG" "$WORK_DIR/results"
cd "$REPO"

echo "[s2o-cyclegan] $(date '+%F %T') train"
$PY train.py --dataroot "$S2O" --name s2o_cyclegan --model cycle_gan \
  --batch_size 4 --load_size 600 --crop_size 512 \
  --n_epochs 100 --n_epochs_decay 100 --num_threads 8 --no_html \
  --checkpoints_dir "$CK" > "$LOG/s2o_cyclegan_train.log" 2>&1 \
&& $PY test.py --dataroot "$DATA_ROOT/S2O_test512" --name s2o_cyclegan --model cycle_gan \
  --num_test 99999 --load_size 512 --crop_size 512 \
  --checkpoints_dir "$CK" --results_dir "$WORK_DIR/results/s2o_cyclegan/" \
  > "$LOG/s2o_cyclegan_test.log" 2>&1
echo "[s2o-cyclegan] $(date '+%F %T') exit $?"
