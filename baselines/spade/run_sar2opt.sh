#!/usr/bin/env bash
# SPADE (NVlabs/SPADE) SAR->EO on SAR2Opt, semantic-image-synthesis -> I2I
# adaptation (--label_nc 0 --no_instance, dataset_mode custom, SAR as "label").
# THIS REQUIRES THE 5 PATCHES DESCRIBED IN README.md.
#
# BUDGET: batch 8, 100 + 100 epochs, floor(1450/8) = 181 it/epoch
#         -> 200 x 181 = 36,200 GENERATOR UPDATES.
#         (drop_last=isTrain -> floor. 100+100 came from pix2pix convention,
#         not from an update count; it sits well under the 120,000 of the
#         QXS-SAROPT cell. Quote updates, not epochs.)
#
# PROTOCOL: random 512 crops of the native 600px tiles at train time
# (preprocess_mode=crop, no aspect-distorting resize); inference on the
# pre-made center-512 test set, because SPADE's get_params() draws crop_pos
# with random.randint even when isTrain is false.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (sar2opt native tiles; S2O_test512 crops)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/s2o_spade/
#   WORK_DIR        logs, raw webpage dump, collected EO PNGs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -euo pipefail
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/SPADE
TRA=$DATA_ROOT/sar2opt/trainA;      TRB=$DATA_ROOT/sar2opt/trainB
TEA=$DATA_ROOT/S2O_test512/testA;   TEB=$DATA_ROOT/S2O_test512/testB
NAME=s2o_spade
CKPT=$CKPT_ROOT/gan_ckpts
RAW=$WORK_DIR/results/_raw_spade
OUT=$WORK_DIR/results/s2o_spade_eo
LOGS=$WORK_DIR/logs

for d in "$TRA" "$TRB" "$TEA" "$TEB"; do
  [ -d "$d" ] || { echo "missing data dir: $d"; exit 1; }
done
mkdir -p "$CKPT" "$RAW" "$OUT" "$LOGS"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[spade-s2o] train on GPU $GPU: bs8 512px 100+100 ep (log: $LOGS/${NAME}_train.log)"
$PY train.py \
  --name $NAME \
  --dataset_mode custom \
  --label_dir "$TRA" --image_dir "$TRB" \
  --label_nc 0 --no_instance --no_pairing_check \
  --preprocess_mode crop --load_size 600 --crop_size 512 --aspect_ratio 1 \
  --batchSize 8 --nThreads 8 \
  --niter 100 --niter_decay 100 \
  --save_epoch_freq 50 --save_latest_freq 160000 \
  --checkpoints_dir "$CKPT" --no_html \
  > "$LOGS/${NAME}_train.log" 2>&1 || { echo "[spade-s2o] TRAIN FAILED"; exit 1; }

echo "[spade-s2o] test (log: $LOGS/${NAME}_test.log)"
$PY test.py \
  --name $NAME \
  --dataset_mode custom \
  --label_dir "$TEA" --image_dir "$TEB" \
  --label_nc 0 --no_instance --no_pairing_check \
  --preprocess_mode crop --load_size 512 --crop_size 512 --aspect_ratio 1 \
  --batchSize 4 --nThreads 8 \
  --checkpoints_dir "$CKPT" --results_dir "$RAW" \
  --which_epoch latest \
  > "$LOGS/${NAME}_test.log" 2>&1 || { echo "[spade-s2o] TEST FAILED"; exit 1; }

echo "[spade-s2o] collecting fake EO PNGs into $OUT"
IMGD="$RAW/$NAME/test_latest/images"
m=$(find "$IMGD/synthesized_image" -maxdepth 1 -name '*.png' | wc -l)
find "$IMGD/synthesized_image" -maxdepth 1 -name '*.png' -exec mv -t "$OUT" {} +
rm -rf "$IMGD/input_label"
WANT=$(ls "$TEA" | wc -l)
echo "[spade-s2o] done: this run produced $m/$WANT; $(ls "$OUT" | wc -l) images now in $OUT"
[ "$m" -eq "$WANT" ] || { echo "[spade-s2o] WRONG IMAGE COUNT ($m != $WANT)"; exit 1; }
