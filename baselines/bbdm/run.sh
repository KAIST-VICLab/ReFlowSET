#!/usr/bin/env bash
# BBDM (LBBDM-f4) SAR->EO.   GPU=0 bash run.sh <qxs|s2o>
#   job1: training
#   job2: --sample_to_eval over the full test set (200 sampling steps)
#
# BUDGET, IN GENERATOR UPDATES
#   qxs  batch 32 @ 256, 500 it/epoch, n_epochs 100 capped by n_steps 50000
#        -> 50,000 updates
#   s2o  batch  8 @ 512, 181 it/epoch, n_epochs 280 capped by n_steps 50000
#        -> 50,137 updates (the runner breaks at the first epoch boundary past
#           n_steps, so the s2o cell overshoots by 137)
#   The two are pixel-matched: 32 x 256^2 == 8 x 512^2.  512 is also the
#   evaluation crop for SAR2Opt and the resolution the conditional-BBDM paper's
#   own SAR2Opt template trains at.
#
# NOTE SAR2Opt has only 1450 train pairs -> ~277 epochs.  Watch overfitting.
#
# Data: $DATA_ROOT/{qxs,s2o}_bbdm, a custom_aligned tree
#   {train,val,test}/{A,B} with A = SAR condition, B = EO target.
#   For s2o these are the 512 CENTER CROPS of the 600px tiles, not the natives.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/BBDM)
#   DATA_ROOT       prepared data (qxs_bbdm / s2o_bbdm)
#   CKPT_ROOT       must contain vqgan/vq-f4/model.ckpt (see README)
#   WORK_DIR        results and logs
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
# The config file also carries ${DATA_ROOT} / ${CKPT_ROOT} placeholders; expand
# them once with envsubst before the first run (see ../README.md).
set -u
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) CFG=configs/QXS-LBBDM-f4.yaml; TAG=QXS ;;
  s2o) CFG=configs/S2O-LBBDM-f4.yaml; TAG=S2O ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
# BBDM's main.py only re-exports CUDA_VISIBLE_DEVICES for multi-GPU --gpu_ids;
# with a single id it uses cuda:<id> INSIDE the visible set, so --gpu_ids 0
# plus this export lands on physical GPU $GPU.
export CUDA_VISIBLE_DEVICES=$GPU

REPO=$BASELINES_ROOT/BBDM
RES=$WORK_DIR/results/${DS}_bbdm
LOGDIR=$WORK_DIR/logs
mkdir -p "$RES" "$LOGDIR"
cd "$REPO" || exit 1

overall=0
note() { echo "[$DS-bbdm] $(date '+%F %T') $*"; }

# ---------------- Job 1: training ----------------
note "job1: BBDM LBBDM-f4 train (GPU $GPU) -> $RES"
"$PY" main.py -c "$CFG" -t --gpu_ids 0 -r "$RES" \
    > "$LOGDIR/${DS}_bbdm_train.log" 2>&1
rc=$?
note "job1 exit code $rc"

# ---------------- Job 2: sample_to_eval on the test set ----------------
CKPT="$RES/$TAG/LBBDM-f4/checkpoint/last_model.pth"
if [ $rc -eq 0 ] && [ -f "$CKPT" ]; then
    note "job2: sample_to_eval -> $RES/$TAG/LBBDM-f4/sample_to_eval/"
    "$PY" main.py -c "$CFG" --gpu_ids 0 -r "$RES" \
        --sample_to_eval --resume_model "$CKPT" \
        > "$LOGDIR/${DS}_bbdm_sample.log" 2>&1
    rc=$?
    # outputs: sample_to_eval/{200,condition,ground_truth}
    # '200' == sample_step and holds the GENERATED EO; condition/ is the SAR
    # input and ground_truth/ the target.  Score the numeric directory.
    note "job2 exit code $rc"
    [ $rc -ne 0 ] && overall=1
else
    note "job2 SKIPPED (training failed or no last_model.pth)"
    overall=1
fi
note "done, overall=$overall"
exit $overall
