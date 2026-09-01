#!/usr/bin/env bash
# StegoGAN (sian-wusidi/StegoGAN, CVPR 2024) SAR->EO on QXS-SAROPT.
# Unpaired, CycleGAN-derived, with a mismatch mask that suppresses content
# present in one domain and absent in the other.
#
# Settings are the README-recommended ones for non-bijective translation (the
# Google_mismatch recipe, the closest analogue to SAR->EO): lambda_reg 0.3,
# lambda_consistency 1, resnet_layer 8, --fusionblock, LSGAN, 256px,
# no_dropout (the model default).
#
# BUDGET: batch 4, 15 + 15 epochs, ceil(16001/4) = 4001 it/epoch
#         -> 30 x 4001 = 120,030 DATA ITERATIONS.
#         StegoGAN calls optimizer_G.step() TWICE per iteration, so that is
#         240,060 optimizer steps. The whole family is quoted in data
#         iterations; do not re-derive one cell the other way.
#         The upstream README uses batch_size 1 on ~1-2k-image datasets.
#
# NOTE --display_id 0 is passed to train.py and MUST NOT be passed to test.py:
# the flag is declared only in options/train_options.py, and test.py aborts on
# an unrecognised argument -- while the && chain still reports the cell done,
# with an empty output directory.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (QXS_AB with trainA/trainB/testA/testB)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/qxs_stegogan/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/StegoGAN
QXS_AB=$DATA_ROOT/QXS_AB
CKPT=$CKPT_ROOT/gan_ckpts
RESULTS=$WORK_DIR/results
LOGS=$WORK_DIR/logs

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$CKPT" "$RESULTS" "$LOGS"
cd "$REPO"

echo "[$(date)] stegogan train start"
$PY train.py \
    --dataroot "$QXS_AB" \
    --name qxs_stegogan --model stego_gan --gpu_ids 0 \
    --lambda_reg 0.3 --lambda_consistency 1 --resnet_layer 8 --fusionblock \
    --batch_size 4 --load_size 256 --crop_size 256 \
    --lr_policy linear --n_epochs 15 --n_epochs_decay 15 \
    --num_threads 16 --no_html --display_id 0 \
    --checkpoints_dir "$CKPT" \
    > "$LOGS/qxs_stegogan_train.log" 2>&1 \
&& echo "[$(date)] stegogan test start" \
&& $PY test.py \
    --dataroot "$QXS_AB" \
    --name qxs_stegogan --model stego_gan --gpu_ids 0 --phase test \
    --no_dropout --resnet_layer 8 --fusionblock \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir "$CKPT" --results_dir "$RESULTS/" \
    > "$LOGS/qxs_stegogan_test.log" 2>&1 \
&& echo "[$(date)] done -> $RESULTS/qxs_stegogan/test_latest/images/fake_B_clean/<stem>.png" \
|| echo "[$(date)] FAILED (see $LOGS/qxs_stegogan_*)"
