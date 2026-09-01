#!/usr/bin/env bash
# CycleGAN (junyanz/pytorch-CycleGAN-and-pix2pix) SAR->EO on QXS-SAROPT.
# Unpaired: the loader reads trainA (SAR) and trainB (EO) independently.
# Train + full test pass; the scored image is fake_B.
#
# BUDGET NOTE: the repo default is 100+100 epochs, tuned for ~1-2k-image
# datasets. QXS-SAROPT has 16,001 train images, so an epoch is ~10x more
# expensive; 25+25 epochs at batch 8 is ceil(16001/8) = 2001 it/epoch x 50 =
# 100,050 GENERATOR UPDATES -- a comparable optimisation budget at a feasible
# single-GPU wall clock, and in the same band as the 120,120 of pix2pix.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (QXS_AB with trainA/trainB/testA/testB)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/qxs_cyclegan/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
QXS_AB=$DATA_ROOT/QXS_AB          # trainA/trainB 16001, testA/testB 3999
CKPT=$CKPT_ROOT/gan_ckpts
RESULTS=$WORK_DIR/results
LOGS=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$CKPT" "$RESULTS" "$LOGS"
cd "$REPO"

echo "[$(date)] cyclegan train start"
$PY train.py \
    --dataroot "$QXS_AB" \
    --name qxs_cyclegan --model cycle_gan --direction AtoB \
    --batch_size 8 --load_size 256 --crop_size 256 \
    --n_epochs 25 --n_epochs_decay 25 \
    --num_threads 16 --no_html \
    --checkpoints_dir "$CKPT" \
    > "$LOGS/qxs_cyclegan_train.log" 2>&1 \
&& echo "[$(date)] cyclegan test start" \
&& $PY test.py \
    --dataroot "$QXS_AB" \
    --name qxs_cyclegan --model cycle_gan \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir "$CKPT" --results_dir "$RESULTS/" \
    > "$LOGS/qxs_cyclegan_test.log" 2>&1 \
&& echo "[$(date)] done -> $RESULTS/qxs_cyclegan/test_latest/images/<stem>_fake_B.png" \
|| echo "[$(date)] FAILED (see $LOGS/qxs_cyclegan_*)"
