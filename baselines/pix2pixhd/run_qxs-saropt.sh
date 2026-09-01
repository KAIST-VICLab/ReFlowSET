#!/usr/bin/env bash
# pix2pixHD (NVIDIA/pix2pixHD) SAR->EO on QXS-SAROPT, via the official
# image-to-image route: --label_nc 0 --no_instance, the SAR PNG fed as the
# "label" A and the EO PNG as B.
#
# BUDGET: batch 16, 60 + 60 epochs, floor(16001/16) = 1000 it/epoch
#         -> 120 x 1000 = 120,000 GENERATOR UPDATES.
#         (pix2pixHD's data/aligned_dataset.py __len__ floor-rounds to a
#         multiple of batchSize, so iterations per epoch round DOWN -- unlike
#         the junyanz loaders, which round up. Matched to the 120,120 of the
#         pix2pix row; verified against the training log: total_steps
#         1,920,000 / 16 = 120,000.)
#
# Data: $DATA_ROOT/QXS_p2phd, whose train_A/train_B/test_A/test_B are symlinks
# to the QXS_AB trainA/trainB/testA/testB directories (pix2pixHD wants the
# underscore spelling).
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (QXS_AB, QXS_p2phd)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/qxs_p2phd/
#   WORK_DIR        logs, raw webpage dump, collected EO PNGs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -euo pipefail
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pix2pixHD
DATAROOT=$DATA_ROOT/QXS_p2phd
CKPT=$CKPT_ROOT/gan_ckpts                 # checkpoints land in $CKPT/qxs_p2phd/
RAW=$WORK_DIR/results/_raw_p2phd          # pix2pixHD webpage output (kept for inspection)
OUT=$WORK_DIR/results/qxs_p2phd_eo        # final per-image fake EO PNGs (<basename>.png)
LOGS=$WORK_DIR/logs
NAME=qxs_p2phd

mkdir -p "$DATAROOT" "$CKPT" "$RAW" "$OUT" "$LOGS"
ln -sfn "$DATA_ROOT/QXS_AB/trainA" "$DATAROOT/train_A"
ln -sfn "$DATA_ROOT/QXS_AB/trainB" "$DATAROOT/train_B"
ln -sfn "$DATA_ROOT/QXS_AB/testA"  "$DATAROOT/test_A"
ln -sfn "$DATA_ROOT/QXS_AB/testB"  "$DATAROOT/test_B"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[p2phd] training on GPU $GPU (log: $LOGS/${NAME}_train.log)"
$PY train.py \
  --name $NAME \
  --dataroot "$DATAROOT" \
  --label_nc 0 --no_instance \
  --resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 \
  --batchSize 16 --nThreads 8 \
  --niter 60 --niter_decay 60 \
  --save_epoch_freq 20 --save_latest_freq 16000 \
  --checkpoints_dir "$CKPT" --no_html \
  > "$LOGS/${NAME}_train.log" 2>&1

echo "[p2phd] testing 3999 images (log: $LOGS/${NAME}_test.log)"
rm -rf "$RAW/$NAME/test_latest/images"    # a previous pass's PNGs survive here
$PY test.py \
  --name $NAME \
  --dataroot "$DATAROOT" \
  --label_nc 0 --no_instance \
  --resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 \
  --checkpoints_dir "$CKPT" --results_dir "$RAW" \
  --phase test --which_epoch latest --how_many 99999 \
  > "$LOGS/${NAME}_test.log" 2>&1

echo "[p2phd] collecting fake EO PNGs into $OUT"
shopt -s nullglob
n=0
for f in "$RAW/$NAME/test_latest/images/"*_synthesized_image.png; do
  b=$(basename "$f" _synthesized_image.png)
  cp "$f" "$OUT/$b.png"; n=$((n+1))
done
WANT=$(ls "$DATAROOT/test_A" | wc -l)
echo "[p2phd] done: $n/$WANT images in $OUT"
# A test pass can exit 0 and write nothing. Check the EXACT count, not >0 --
# a truncated test pass scores just as wrongly as an empty one.
[ "$n" -eq "$WANT" ] || { echo "[p2phd] WRONG IMAGE COUNT ($n != $WANT)"; exit 1; }
