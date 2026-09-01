#!/usr/bin/env bash
# Conditional Diffusion for SAR-to-Optical Image Translation (IEEE GRSL).
# Official guided-diffusion fork, vendored and patched -- see README.md; the
# patches are not optional, one of them fixes a correctness bug in the sampler.
#
#   GPU=0 bash run.sh <qxs|s2o>
#
# ======================= BUDGET: GENERATOR UPDATES =======================
# 50,000 UPDATES on both datasets == the authors' protocol (50k iterations at
# global batch 24) and the same band as the BBDM / cBBDM / ControlNet rows.
# bs 24 @ 256 (qxs) and bs 6 @ 512 (s2o) -- equal pixels per update:
# 24*256^2 == 6*512^2.  --lr_anneal_steps 50000 is the released code's ONLY
# stop mechanism (constant lr 1e-4 with linear decay to 0; the paper's
# warmup+cosine schedule is NOT in the released code -- we run the code).
# Model: the 164.3M-parameter guided-diffusion UNet (ch 128, num_res_blocks 3,
# attn 16/8, learn_sigma False), T=2000 linear, eps-prediction, with the SAR
# condition concatenated NOISE-FREE at every step.
#
# Train data: $DATA_ROOT/conddiff_<ds>/train/{sar,opt} with INTEGER filenames
# (the training loader sorts with int(stem) and raises on anything else), plus
# $DATA_ROOT/conddiff_<ds>/test/{sar,opt} and manifest_test.json mapping the
# integer ids back to the ground-truth stems.  For s2o the tiles must be
# DETERMINISTIC CENTRE-512 CROPS of the 600px images: the released
# center_crop_arr RESIZES 600->512, which would break the crop-not-resize
# protocol this table holds fixed.
#
# Inference: respaced DDPM, 250 steps (the authors' own sample.sh protocol).
# The DDIM path is BROKEN upstream -- ddim_sample_loop() has no `condition`
# argument at all, so --use_ddim True raises TypeError.  EMA 0.9999 weights,
# clip_denoised.  Outputs are renamed through manifest_test.json.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/CondDiff)
#   DATA_ROOT       prepared data (conddiff_<ds>)
#   WORK_DIR        run directories, logs, results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS=${1:?usage: GPU=<n> bash run.sh <qxs|s2o>}
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

REPO=$BASELINES_ROOT/CondDiff
DATA=$DATA_ROOT/conddiff_$DS
# _stubs FIRST: blobfile and mpi4py are imported unconditionally by dist_util
# and are replaced here by a local-filesystem / single-process stand-in.
export PYTHONPATH=$HERE/_stubs:$REPO${PYTHONPATH:+:$PYTHONPATH}

case "$DS" in
  qxs) RES=256; BS=24 ;;
  s2o) RES=512; BS=6  ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
STEPS=50000
RESPACE=250
SAVE_INT=10000
OUT=$WORK_DIR/results/${DS}_conddiff_eo
LOGDIR=$WORK_DIR/conddiff_runs/$DS
NSAMP=$(ls "$DATA/test/sar" | wc -l)

# Fail in the first second, not in the tenth hour.
[ -d "$DATA/train/sar" ] || { echo "missing $DATA/train/sar -- build the integer-named adapter tree first (README)"; exit 1; }
[ -d "$DATA/test/sar" ]  || { echo "missing $DATA/test/sar"; exit 1; }
[ -f "$DATA/manifest_test.json" ] || { echo "missing manifest_test.json"; exit 1; }
mkdir -p "$LOGDIR" "$WORK_DIR/logs"

MODEL_FLAGS="--image_size $RES --num_channels 128 --num_res_blocks 3 --learn_sigma False"
DIFF_FLAGS="--diffusion_steps 2000 --noise_schedule linear"

# ---- train (skip-guard: any EMA checkpoint at >= $STEPS) -------------------
LAST_EMA=$(ls "$LOGDIR"/ema_0.9999_*.pt 2>/dev/null | sort | tail -1 || true)
need_train=1
if [ -n "$LAST_EMA" ]; then
  st=$(basename "$LAST_EMA" .pt); st=${st##*_}; st=$((10#$st))
  [ "$st" -ge "$STEPS" ] && need_train=0
fi
if [ "$need_train" = "1" ]; then
  RESUME=""
  LAST_MODEL=$(ls "$LOGDIR"/model*.pt 2>/dev/null | sort | tail -1 || true)
  [ -n "$LAST_MODEL" ] && RESUME="--resume_checkpoint $LAST_MODEL"
  OPENAI_LOGDIR=$LOGDIR $PY "$REPO/scripts/image_train.py" \
    --data_dir_sar "$DATA/train/sar" --data_dir_opt "$DATA/train/opt" \
    $MODEL_FLAGS $DIFF_FLAGS \
    --lr 1e-4 --batch_size $BS --lr_anneal_steps $STEPS \
    --save_interval $SAVE_INT $RESUME \
    > "$WORK_DIR/logs/conddiff_${DS}_train.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && { echo "train failed rc=$rc"; exit $rc; }
fi
EMA=$(ls "$LOGDIR"/ema_0.9999_*.pt 2>/dev/null | sort | tail -1)
[ -n "$EMA" ] || { echo "no EMA checkpoint after training"; exit 1; }

# ---- sample ----------------------------------------------------------------
SDIR=$LOGDIR/sample_out
n_done=$(ls "$SDIR/gen_opt" 2>/dev/null | wc -l || echo 0)
if [ "$n_done" -lt "$NSAMP" ]; then
  OPENAI_LOGDIR=$LOGDIR/sample_log $PY "$REPO/scripts/image_sample_realtime.py" \
    --model_path "$EMA" --test_dir "$DATA/test" --out_dir "$SDIR" \
    $MODEL_FLAGS $DIFF_FLAGS \
    --timestep_respacing $RESPACE --batch_size $BS --num_samples $NSAMP \
    > "$WORK_DIR/logs/conddiff_${DS}_sample.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && { echo "sample failed rc=$rc"; exit $rc; }
fi

# ---- rename the integer-indexed outputs back to GT stems -------------------
$PY - "$DATA/manifest_test.json" "$SDIR/gen_opt" "$OUT" "$NSAMP" << 'PYEOF'
import json, os, sys
man_p, gen, out, nsamp = sys.argv[1:5]
man = json.load(open(man_p))
os.makedirs(out, exist_ok=True)
n = 0
for pid, stem in man.items():
    if int(pid) >= int(nsamp):
        continue
    src = os.path.join(gen, f"{int(pid)}.png")
    if not os.path.exists(src):
        sys.exit(f"missing generated {src}")
    dst = os.path.join(out, stem + ".png")
    if not os.path.exists(dst):
        os.link(src, dst)
    n += 1
print(f"linked {n} generated images -> {out}")
PYEOF
rc=$?; [ $rc -ne 0 ] && { echo "rename failed rc=$rc"; exit $rc; }
echo "conddiff $DS done: $(ls "$OUT" | wc -l) images in $OUT"
