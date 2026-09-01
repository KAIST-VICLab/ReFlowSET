#!/usr/bin/env bash
# "SD2.1 fine-tune only" = C-DiffSET STAGE 1.  Train + inference, stage 1 only.
#
#   GPU=0 bash run_stage1.sh <qxs|s2o>
#
# This is the same repository and the same trainer as ../c-diffset/; that
# script runs both stages.  Use this one to reproduce the SD2.1-FT row alone,
# or as the initialisation step for stage 2.
#
# BUDGET: the published row is the fixed-step `checkpoint-40000`, in GENERATOR
#   UPDATES (num_iter counts optimizer steps).  Epochs differ wildly for the
#   same budget: qxs bs64 -> 160 epochs, s2o bs16 -> 445 epochs.  Batch is
#   chosen for equal pixels per update: 64 x 256^2 == 16 x 512^2.
#   Measured reason for 40k rather than the authors' 50k: stage-1 validation
#   LPIPS bottoms out at ~41k (0.5297) and DEGRADES to 0.5348 by 50k.
#   As trained here, both cells ran to 50,000 and 40,000 is a snapshot of that
#   run -- see ../c-diffset/run.sh for the full note.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/C-DiffSET)
#   DATA_ROOT       QXSLAB_SAROPT / sar2opt roots, plus s2o_512 for inference
#   SPLITS_ROOT     this repository's splits/ directory
#   CKPT_ROOT       must contain sd21_unet8ch_init.safetensors (see README)
#   WORK_DIR        stage work dir, logs, results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
#   STEPS           override the training length (default: the config's 50000)
set -u
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) TESTA=$DATA_ROOT/QXS_AB/testA ;;
  s2o) TESTA=$DATA_ROOT/s2o_512/testA ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
STEPS=${STEPS:-50000}
PUBLISH=checkpoint-40000
REPO=$BASELINES_ROOT/C-DiffSET
LOG=$WORK_DIR/logs
S1_DIR=$WORK_DIR/${DS}_sd21ft
[ -d "$TESTA" ] || { echo "missing test SAR dir $TESTA"; exit 1; }
mkdir -p "$LOG"
cd "$REPO"

if [ -d "$S1_DIR/$PUBLISH" ]; then
  echo "[sd21ft-$DS] $PUBLISH present, skipping training"
else
  echo "[sd21ft-$DS] stage-1 -> $STEPS iters"
  $PY main_stage1.py --config configs/${DS}_sd21_stage1.yaml \
      --work-dir "$S1_DIR/" --gpu "$GPU" --num-iter "$STEPS" --save-iter 10000 \
      > "$LOG/${DS}_sd21ft_train.log" 2>&1 \
    || { echo "[sd21ft-$DS] TRAINING FAILED"; exit 1; }
fi

echo "[sd21ft-$DS] inference from $PUBLISH"
$PY test_stage1.py --sar-dir "$TESTA" --output-dir "$WORK_DIR/results/${DS}_sd21ft_eo/" \
    --checkpoint "$S1_DIR/$PUBLISH/model.safetensors" \
    --num-inference-steps 50 \
    > "$LOG/${DS}_sd21ft_test.log" 2>&1
echo "[sd21ft-$DS] DONE $(date '+%F %T')"
