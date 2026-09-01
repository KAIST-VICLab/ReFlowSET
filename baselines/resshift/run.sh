#!/usr/bin/env bash
# ResShift (NeurIPS 2023) as a SAR->EO baseline.   GPU=0 bash run.sh <qxs|s2o>
# SAR (grayscale, replicated to 3 channels by the loader) is the LQ input and
# EO is the GT; sf=1 gives a same-size mapping through ResShift's official
# paired route.
#
# BUDGET: 50,000 ITERATIONS on both datasets, batch 16 (microbatch 8, so two
#   accumulation micro-steps per optimizer update).  qxs trains on its native
#   256px images; s2o trains on random 256 crops of the 600px tiles and is
#   INFERRED on the 627 center-512 crops -- the sampler chops each 512 input
#   into four clean 256 tiles (chop_size = chop_stride = lq_size = 256).
#
# NO AUTO-RESUME.  A death at hour 7 needs a manual relaunch:
#   RESUME=<save_dir>/<run>/ckpts/model_XXXX.pth GPU=0 bash run.sh <ds>
# There is no --auto_resume in this repository.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/ResShift)
#   DATA_ROOT       prepared data (QXS_AB, sar2opt, s2o_512)
#   CKPT_ROOT       must contain autoencoder_vq_f4.pth (see README)
#   WORK_DIR        run directories, logs, results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
#   RESUME          optional trainer checkpoint to resume from
# The configs carry ${DATA_ROOT} / ${CKPT_ROOT} placeholders; expand them with
# envsubst first -- OmegaConf would otherwise read them as interpolations.
set -euo pipefail
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) TESTA=$DATA_ROOT/QXS_AB/testA;   INF_BS=16 ;;
  s2o) TESTA=$DATA_ROOT/s2o_512/testA;  INF_BS=8  ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
REPO=$BASELINES_ROOT/ResShift
CFG=$REPO/configs/${DS}_sar2eo_256.yaml
SAVE_DIR=$WORK_DIR/resshift_$DS            # the trainer adds a timestamped subdir
OUT_DIR=$WORK_DIR/results/${DS}_resshift_eo
LOG_DIR=$WORK_DIR/logs

mkdir -p "$SAVE_DIR" "$OUT_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$GPU   # a single visible GPU keeps the trainer on its non-distributed path

# ---------------- Train ----------------
cd "$REPO"
echo "[resshift-$DS] training on GPU $GPU, log: $LOG_DIR/resshift_${DS}_train.log"
$PY main.py \
    --cfg_path "$CFG" \
    --save_dir "$SAVE_DIR" \
    ${RESUME:+--resume "$RESUME"} \
    >> "$LOG_DIR/resshift_${DS}_train.log" 2>&1

# ---------------- Inference ----------------
RUN_DIR=$(ls -1d "$SAVE_DIR"/*/ | sort | tail -1)
EMA_CKPT=$(ls -1v "$RUN_DIR"/ema_ckpts/ema_model_*.pth | tail -1)
echo "[resshift-$DS] inference with $EMA_CKPT -> $OUT_DIR"
$PY inference_qxs.py \
    --cfg_path "$CFG" \
    --ckpt "$EMA_CKPT" \
    --in_dir "$TESTA" \
    --out_dir "$OUT_DIR" \
    --bs $INF_BS \
    >> "$LOG_DIR/resshift_${DS}_infer.log" 2>&1

echo "[resshift-$DS] done: $(ls "$OUT_DIR" | wc -l) images in $OUT_DIR"
