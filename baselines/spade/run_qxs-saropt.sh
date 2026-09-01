#!/usr/bin/env bash
# SPADE (NVlabs/SPADE) SAR->EO on QXS-SAROPT, semantic-image-synthesis -> I2I
# adaptation: the SAR image is fed as the "label" input with --label_nc 0
# --no_instance, dataset_mode custom with --label_dir trainA --image_dir trainB.
# THIS REQUIRES THE 5 PATCHES DESCRIBED IN README.md -- stock SPADE cannot do
# image-to-image at all and will build a 1-channel-input generator.
#
# BUDGET: batch 16, 60 + 60 epochs, floor(16001/16) = 1000 it/epoch
#         -> 120 x 1000 = 120,000 GENERATOR UPDATES.
#         (SPADE's data/__init__.py passes drop_last=opt.isTrain, so iterations
#         per epoch round DOWN, same as pix2pixHD. Matched to the pix2pix row.)
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (QXS_AB with trainA/trainB/testA/testB)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/qxs_spade/
#   WORK_DIR        logs, raw webpage dump, collected EO PNGs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -euo pipefail
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/SPADE
DATA=$DATA_ROOT/QXS_AB
CKPT=$CKPT_ROOT/gan_ckpts               # checkpoints land in $CKPT/qxs_spade/
RAW=$WORK_DIR/results/_raw_spade        # SPADE webpage output (kept for inspection)
OUT=$WORK_DIR/results/qxs_spade_eo      # final per-image fake EO PNGs (<basename>.png)
LOGS=$WORK_DIR/logs
NAME=qxs_spade

mkdir -p "$CKPT" "$RAW" "$OUT" "$LOGS"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[spade] training on GPU $GPU (log: $LOGS/${NAME}_train.log)"
$PY train.py \
  --name $NAME \
  --dataset_mode custom \
  --label_dir "$DATA/trainA" --image_dir "$DATA/trainB" \
  --label_nc 0 --no_instance --no_pairing_check \
  --preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --aspect_ratio 1 \
  --batchSize 16 --nThreads 8 \
  --niter 60 --niter_decay 60 \
  --save_epoch_freq 20 --save_latest_freq 16000 \
  --checkpoints_dir "$CKPT" --no_html \
  > "$LOGS/${NAME}_train.log" 2>&1

echo "[spade] testing 3999 images (log: $LOGS/${NAME}_test.log)"
$PY test.py \
  --name $NAME \
  --dataset_mode custom \
  --label_dir "$DATA/testA" --image_dir "$DATA/testB" \
  --label_nc 0 --no_instance --no_pairing_check \
  --preprocess_mode resize_and_crop --load_size 256 --crop_size 256 --aspect_ratio 1 \
  --batchSize 8 --nThreads 8 \
  --checkpoints_dir "$CKPT" --results_dir "$RAW" \
  --which_epoch latest \
  > "$LOGS/${NAME}_test.log" 2>&1

echo "[spade] collecting fake EO PNGs into $OUT"
IMGD="$RAW/$NAME/test_latest/images"
# Count what THIS test pass produced, before moving: counting $OUT afterwards
# would let a stale/partial dump from an earlier attempt satisfy the guard.
m=$(find "$IMGD/synthesized_image" -maxdepth 1 -name '*.png' | wc -l)
find "$IMGD/synthesized_image" -maxdepth 1 -name '*.png' -exec mv -t "$OUT" {} +
rm -rf "$IMGD/input_label"              # SPADE also dumps a copy of the SAR input
WANT=$(ls "$DATA/testA" | wc -l)
echo "[spade] done: this run produced $m/$WANT; $(ls "$OUT" | wc -l) images now in $OUT"
# A test pass can exit 0 and write nothing. Exact count, not >0.
[ "$m" -eq "$WANT" ] || { echo "[spade] WRONG IMAGE COUNT ($m != $WANT)"; exit 1; }
