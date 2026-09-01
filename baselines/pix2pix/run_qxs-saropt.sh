#!/usr/bin/env bash
# pix2pix (junyanz/pytorch-CycleGAN-and-pix2pix) SAR->EO on QXS-SAROPT.
# Train + full test pass. The test pass writes one fake_B PNG per test image,
# 256x256 RGB, scored against the EO test set by the unified evaluator.
#
# BUDGET: batch 16, 60 + 60 epochs, ceil(16001/16) = 1001 it/epoch
#         -> 120 x 1001 = 120,120 GENERATOR UPDATES.
#         (The junyanz loader has no drop_last, so iterations per epoch round
#         UP. This 120k is this table's paired-GAN reference budget; the
#         pix2pixHD and SPADE rows were matched to it.)
#
# Data: the aligned A|B dataset, 512x256 tiles built by the upstream helper
#   $BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py
#   --fold_A <SAR dir> --fold_B <EO dir> --fold_AB $DATA_ROOT/QXS_AB_combined
# SAR is the LEFT half (A) and EO the right (B), hence --direction AtoB.
# NOTE that helper is broken at upstream HEAD -- it uses Path() without
# importing it. See README, patch 1.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (QXS_AB_combined)
#   CKPT_ROOT       checkpoints  -> $CKPT_ROOT/gan_ckpts/qxs_pix2pix/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
COMBINED=$DATA_ROOT/QXS_AB_combined       # 16001 train / 3999 test, A|B 512x256
CKPT=$CKPT_ROOT/gan_ckpts
RESULTS=$WORK_DIR/results
LOGS=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU   # set BEFORE python starts; the repo then sees it as cuda:0
mkdir -p "$CKPT" "$RESULTS" "$LOGS"
cd "$REPO"

echo "[$(date)] pix2pix train start"
$PY train.py \
    --dataroot "$COMBINED" \
    --name qxs_pix2pix --model pix2pix --direction AtoB \
    --batch_size 16 --load_size 256 --crop_size 256 \
    --n_epochs 60 --n_epochs_decay 60 \
    --num_threads 16 --no_html \
    --checkpoints_dir "$CKPT" \
    > "$LOGS/qxs_pix2pix_train.log" 2>&1 \
&& echo "[$(date)] pix2pix test start" \
&& $PY test.py \
    --dataroot "$COMBINED" \
    --name qxs_pix2pix --model pix2pix --direction AtoB \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir "$CKPT" --results_dir "$RESULTS/" \
    > "$LOGS/qxs_pix2pix_test.log" 2>&1 \
&& echo "[$(date)] done -> $RESULTS/qxs_pix2pix/test_latest/images/<stem>_fake_B.png" \
|| echo "[$(date)] FAILED (see $LOGS/qxs_pix2pix_*)"
