#!/usr/bin/env bash
# ControlNet (SD 2.1-base) SAR->EO.    GPU=0 bash run.sh <qxs|s2o>
#   job1: ControlNet training with the diffusers example trainer
#   job2: inference over the test SAR directory, 50-step UniPC
#
# BUDGET: 50,000 UPDATES on both datasets -- the same band as the BBDM/cBBDM
#   rows.  Batch 32 @ 256 (qxs) and 8 @ 512 (s2o); lr 1e-5, bf16, seed 42.
#   ControlNet is NOT an SD2.1 fine-tune: the base UNet stays frozen and only
#   the 364M-parameter adapter trains.
#   NOTE SAR2Opt has only 1450 train pairs -> ~276 epochs at batch 8;
#   checkpoints are kept every 10k steps (limit 2). Watch overfitting.
#
# Data: $DATA_ROOT/<ds>_controlnet, a HuggingFace imagefolder with
#   metadata.jsonl carrying the columns image / conditioning_image / text
#   (conditioning_image = the SAR PNG, image = the EO target, text = the fixed
#   prompt "electro-optical image").  For s2o these are the 512 CENTER CROPS.
#
# Environment
#   BASELINES_ROOT  vendored code ($BASELINES_ROOT/controlnet/train_controlnet.py)
#   DATA_ROOT       prepared data (<ds>_controlnet, s2o_512)
#   CKPT_ROOT       trained adapter -> $CKPT_ROOT/<ds>_controlnet/
#   WORK_DIR        logs and results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS=${1:?dataset: qxs|s2o}
case "$DS" in
  qxs) RES=256; BS=32; SAR_DIR=$DATA_ROOT/QXS_AB/testA ;;
  s2o) RES=512; BS=8;  SAR_DIR=$DATA_ROOT/s2o_512/testA ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

CN=$BASELINES_ROOT/controlnet
CN_OUT=$CKPT_ROOT/${DS}_controlnet
OUT_DIR=$WORK_DIR/results/${DS}_controlnet_eo
LOGDIR=$WORK_DIR/logs
mkdir -p "$LOGDIR" "$CN_OUT"

overall=0
note() { echo "[$DS-controlnet] $(date '+%F %T') $*"; }

# ---------------- Job 1: ControlNet training ----------------
note "job1: train 50000 steps (bs $BS @ $RES, bf16, GPU $GPU) -> $CN_OUT"
"$PY" "$CN/train_controlnet.py" \
    --pretrained_model_name_or_path Manojb/stable-diffusion-2-1-base \
    --train_data_dir "$DATA_ROOT/${DS}_controlnet" \
    --image_column image \
    --conditioning_image_column conditioning_image \
    --caption_column text \
    --resolution $RES \
    --train_batch_size $BS \
    --dataloader_num_workers 8 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --max_train_steps 50000 \
    --checkpointing_steps 10000 \
    --checkpoints_total_limit 2 \
    --resume_from_checkpoint latest \
    --seed 42 \
    --output_dir "$CN_OUT" \
    > "$LOGDIR/${DS}_controlnet_train.log" 2>&1
rc=$?
note "job1 exit code $rc"

# ---------------- Job 2: inference over the test SAR directory ----------------
if [ $rc -eq 0 ] && [ -f "$CN_OUT/diffusion_pytorch_model.safetensors" ]; then
    note "job2: inference (UniPC-50, guidance 7.5, seed 42, ${RES}px) -> $OUT_DIR"
    "$PY" "$HERE/infer_controlnet.py" \
        --controlnet_dir "$CN_OUT" \
        --sar_dir "$SAR_DIR" \
        --out_dir "$OUT_DIR" \
        --steps 50 --batch_size $BS --resolution $RES --seed 42 \
        > "$LOGDIR/${DS}_controlnet_infer.log" 2>&1
    rc=$?
    note "job2 exit code $rc"
    [ $rc -ne 0 ] && overall=1
else
    note "job2 SKIPPED (training failed or no final controlnet saved)"
    overall=1
fi
note "done, overall=$overall"
exit $overall
