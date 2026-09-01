#!/usr/bin/env bash
# C-DiffSET family: stage 1 ("SD2.1 fine-tune only", ../sd21-ft/) and stage 2
# (C-DiffSET), then inference for both from the fixed-step checkpoint.
#
#   GPU=0 bash run.sh <qxs|s2o>
#
# ============================ THE BUDGET RULE ============================
# The row published in the table is the FIXED-STEP `checkpoint-40000` of each
# stage -- NOT the authors' 50k, and NOT the repo's val-PSNR-selected `best/`.
# Two reasons, both measured on our own runs:
#   * stage-2 val LPIPS reaches 0.5080 by ~32.5k steps and only 0.5064 at 50k
#     -- 0.3% over the last third of training.  Converged.
#   * stage-1 val LPIPS BOTTOMS OUT at ~41k (0.5297) and then DEGRADES to
#     0.5348 by 50k.  The 50k stage 1 overfits; 40k is faster AND better.
# And `best/` is selected on validation PSNR while every other baseline in this
# table publishes its LAST checkpoint -- an asymmetry in C-DiffSET's favour.
# 40k also sits on the save_iter=10000 grid, so checkpoint-40000 exists in any
# 50k run with no retraining.
#
# BUDGET IS IN GENERATOR UPDATES, NOT EPOCHS.  num_iter counts optimizer steps
# (train.py increments global_step once per optimizer.step()), so 40,000 is
# 40,000 updates whatever the train-set size.  The epoch counts it works out to
# are wildly different and are NOT the budget:
#     qxs  bs64  250 it/ep -> 160 epochs
#     s2o  bs16   90 it/ep -> 445 epochs
# Batch size is picked for equal pixels per update: 64 x 256^2 == 16 x 512^2.
#
# WHAT ACTUALLY HAPPENED ON THESE TWO DATASETS.  Both cells were trained by a
# 50,000-step run (the config's own num_iter), and checkpoint-40000 is an
# intermediate snapshot of it; stage 2 was initialised from stage 1's FINAL
# (50k) checkpoint, `lastest/model.safetensors`, which is what the stage-2
# config's accelerator_path names.  The scripted flow below reproduces exactly
# that.  If you prefer a strict 40k+40k schedule, pass STEPS=40000 and point
# --accelerator-path at the stage-1 checkpoint-40000 -- but say which you did.
#
# Training is skipped when the target checkpoint already exists, so re-running
# this script on a trained cell is a no-op-and-infer.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/C-DiffSET)
#   DATA_ROOT       QXSLAB_SAROPT / sar2opt roots, plus s2o_512 for inference
#   SPLITS_ROOT     this repository's splits/ directory
#   CKPT_ROOT       must contain sd21_unet8ch_init.safetensors (see README)
#   WORK_DIR        stage work dirs, logs, results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
#   STEPS           override the training length (default: the config's 50000)
# The configs carry ${DATA_ROOT} / ${SPLITS_ROOT} / ${CKPT_ROOT} / ${WORK_DIR}
# placeholders; expand them with envsubst into $BASELINES_ROOT/C-DiffSET/configs/
# before the first run (see ../README.md).
set -u
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) TESTA=$DATA_ROOT/QXS_AB/testA ;;
  s2o) TESTA=$DATA_ROOT/s2o_512/testA ;;   # the 512 centre crops = the eval protocol
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
# C-DiffSET's main.py / main_stage1.py re-export CUDA_VISIBLE_DEVICES from the
# --gpu argument, so pass --gpu $GPU explicitly to keep it identical to the
# shell export (physical index under PCI_BUS_ID ordering).

STEPS=${STEPS:-50000}
PUBLISH=checkpoint-40000        # the checkpoint the table's numbers come from
REPO=$BASELINES_ROOT/C-DiffSET
LOG=$WORK_DIR/logs
S1_DIR=$WORK_DIR/${DS}_sd21ft
S2_DIR=$WORK_DIR/${DS}_cdiffset
[ -d "$TESTA" ] || { echo "missing test SAR dir $TESTA"; exit 1; }
mkdir -p "$LOG"
cd "$REPO"

# ---- stage 1: SD2.1 fine-tune only -----------------------------------------
if [ -d "$S1_DIR/$PUBLISH" ]; then
  echo "[cdiffset-$DS] stage-1 $PUBLISH present, skipping training"
else
  echo "[cdiffset-$DS] stage-1 -> $STEPS iters"
  $PY main_stage1.py --config configs/${DS}_sd21_stage1.yaml \
      --work-dir "$S1_DIR/" --gpu "$GPU" --num-iter "$STEPS" --save-iter 10000 \
      > "$LOG/${DS}_sd21ft_train.log" 2>&1 \
    || { echo "[cdiffset-$DS] stage-1 FAILED"; exit 1; }
fi

# ---- stage 2: C-DiffSET, initialised from stage 1 --------------------------
if [ -d "$S2_DIR/$PUBLISH" ]; then
  echo "[cdiffset-$DS] stage-2 $PUBLISH present, skipping training"
else
  echo "[cdiffset-$DS] stage-2 -> $STEPS iters"
  $PY main.py --config configs/${DS}_eps_conf_run.yaml \
      --work-dir "$S2_DIR/" --gpu "$GPU" --num-iter "$STEPS" --save-iter 10000 \
      --accelerator-path "$S1_DIR/lastest/model.safetensors" \
      > "$LOG/${DS}_cdiffset_train.log" 2>&1 \
    || { echo "[cdiffset-$DS] stage-2 FAILED"; exit 1; }
fi

# ---- inference from the fixed-step checkpoints -----------------------------
echo "[cdiffset-$DS] inference: C-DiffSET (stage 2)"
$PY test.py --sar-dir "$TESTA" --output-dir "$WORK_DIR/results/${DS}_cdiffset_eo/" \
    --checkpoint "$S2_DIR/$PUBLISH/model.safetensors" \
    --num-inference-steps 50 \
    > "$LOG/${DS}_cdiffset_test.log" 2>&1
echo "[cdiffset-$DS] inference: SD2.1-FT (stage 1)"
$PY test_stage1.py --sar-dir "$TESTA" --output-dir "$WORK_DIR/results/${DS}_sd21ft_eo/" \
    --checkpoint "$S1_DIR/$PUBLISH/model.safetensors" \
    --num-inference-steps 50 \
    > "$LOG/${DS}_sd21ft_test.log" 2>&1
echo "[cdiffset-$DS] DONE $(date '+%F %T')"
