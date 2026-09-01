#!/usr/bin/env bash
# pix2pixHD (NVIDIA/pix2pixHD) SAR->EO on SAR2Opt, image-to-image route
# (--label_nc 0 --no_instance, SAR as "label" A, EO as B).
#
# BUDGET: batch 8, 100 + 100 epochs, floor(1450/8) = 181 it/epoch
#         -> 200 x 181 = 36,200 GENERATOR UPDATES.
#         (This loader floor-rounds. 100+100 is the small-set convention
#         inherited from pix2pix, NOT an update count -- it lands well under
#         the 120,000 of the QXS-SAROPT cell. Quote updates, not epochs.)
#
# PROTOCOL: train on the native 600px tiles with RANDOM 512 crops
# (--resize_or_crop crop, no aspect-distorting resize); test on the pre-made
# center-512 set, because get_params() picks crop_pos with random.randint even
# when isTrain is false -- a 'crop' test pass over native 600px tiles would
# score a RANDOM 512 window against the evaluator's center crop of the GT.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories
#   DATA_ROOT       prepared data (sar2opt native tiles; S2O_test512 crops)
#   CKPT_ROOT       checkpoints -> $CKPT_ROOT/gan_ckpts/s2o_p2phd/
#   WORK_DIR        logs, raw webpage dump, collected EO PNGs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -euo pipefail
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/pix2pixHD
NAME=s2o_p2phd
DATAROOT=$DATA_ROOT/s2o_p2phd
CKPT=$CKPT_ROOT/gan_ckpts
RAW=$WORK_DIR/results/_raw_p2phd
OUT=$WORK_DIR/results/s2o_p2phd_eo
LOGS=$WORK_DIR/logs

# pix2pixHD expects train_A/train_B/test_A/test_B; the sets are trainA/... .
mkdir -p "$DATAROOT" "$CKPT" "$RAW" "$OUT" "$LOGS"
ln -sfn "$DATA_ROOT/sar2opt/trainA"    "$DATAROOT/train_A"
ln -sfn "$DATA_ROOT/sar2opt/trainB"    "$DATAROOT/train_B"
ln -sfn "$DATA_ROOT/S2O_test512/testA" "$DATAROOT/test_A"
ln -sfn "$DATA_ROOT/S2O_test512/testB" "$DATAROOT/test_B"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU
echo "[p2phd-s2o] train on GPU $GPU: bs8 512px 100+100 ep"
$PY train.py --name $NAME --dataroot "$DATAROOT" \
  --label_nc 0 --no_instance \
  --resize_or_crop crop --loadSize 600 --fineSize 512 \
  --batchSize 8 --nThreads 8 --niter 100 --niter_decay 100 \
  --save_epoch_freq 50 --save_latest_freq 16000 \
  --checkpoints_dir "$CKPT" --no_html \
  > "$LOGS/${NAME}_train.log" 2>&1

rm -rf "$RAW/$NAME/test_latest/images"
echo "[p2phd-s2o] test"
$PY test.py --name $NAME --dataroot "$DATAROOT" \
  --label_nc 0 --no_instance \
  --resize_or_crop crop --loadSize 512 --fineSize 512 \
  --checkpoints_dir "$CKPT" --results_dir "$RAW" \
  --phase test --which_epoch latest --how_many 99999 \
  > "$LOGS/${NAME}_test.log" 2>&1

echo "[p2phd-s2o] collecting fake EO PNGs into $OUT"
shopt -s nullglob
n=0
for f in "$RAW/$NAME/test_latest/images/"*_synthesized_image.png; do
  b=$(basename "$f" _synthesized_image.png)
  cp "$f" "$OUT/$b.png"; n=$((n+1))
done
WANT=$(ls "$DATAROOT/test_A" | wc -l)
echo "[p2phd-s2o] done: $n/$WANT images in $OUT"
[ "$n" -eq "$WANT" ] || { echo "[p2phd-s2o] WRONG IMAGE COUNT ($n != $WANT)"; exit 1; }
