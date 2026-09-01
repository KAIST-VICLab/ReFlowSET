#!/usr/bin/env bash
# StegoGAN (sian-wusidi/StegoGAN, CVPR 2024) SAR->EO on SAR2Opt.
# 600px native tiles, random 512 crops at train time; inference on the frozen
# center-512 test set (the evaluation protocol crop for this dataset).
#
# BUDGET: batch 2, 60 + 60 epochs, ceil(1450/2) = 725 it/epoch
#         -> 120 x 725 = 87,000 DATA ITERATIONS (= 174,000 optimizer_G.step()
#         calls; StegoGAN steps the generator optimiser twice per iteration).
#
# NOTE --display_id 0 is a TRAIN-ONLY flag; passing it to test.py aborts the
# test pass while the && chain still reports success on an empty output dir.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (sar2opt native tiles; S2O_test512 crops)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/s2o_stegogan/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/StegoGAN
S2O=$DATA_ROOT/sar2opt
CK=$CKPT_ROOT/gan_ckpts
LOG=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$CK" "$LOG" "$WORK_DIR/results"
cd "$REPO"

echo "[s2o-stegogan] $(date '+%F %T') train"
$PY train.py --dataroot "$S2O" --name s2o_stegogan --model stego_gan --gpu_ids 0 \
  --lambda_reg 0.3 --lambda_consistency 1 --resnet_layer 8 --fusionblock \
  --batch_size 2 --load_size 600 --crop_size 512 --lr_policy linear \
  --n_epochs 60 --n_epochs_decay 60 --display_id 0 \
  --checkpoints_dir "$CK" > "$LOG/s2o_stegogan_train.log" 2>&1 \
&& $PY test.py --dataroot "$DATA_ROOT/S2O_test512" --name s2o_stegogan --model stego_gan --gpu_ids 0 \
  --phase test --no_dropout --resnet_layer 8 --fusionblock --num_test 99999 \
  --load_size 512 --crop_size 512 \
  --checkpoints_dir "$CK" --results_dir "$WORK_DIR/results/s2o_stegogan/" \
  > "$LOG/s2o_stegogan_test.log" 2>&1
echo "[s2o-stegogan] $(date '+%F %T') exit $?"
