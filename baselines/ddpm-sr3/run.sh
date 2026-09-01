#!/usr/bin/env bash
# SR3-class conditional DDPM == E3Diff STAGE 1.  This is the "DDPM (SR3-class)"
# row of the table: the SR3 method class, run in E3Diff's stage-1 configuration.
# It is NOT the SR3 authors' implementation -- say so wherever it is quoted.
#
#   GPU=0 bash run.sh <qxs|s2o>
#
# ======================= BUDGET: GENERATOR UPDATES =======================
#   250,000 UPDATES on BOTH datasets.  batch 16 @ 256 (qxs) and batch 4 @ 512
#   (s2o) -- equal pixels per update: 16*256^2 == 4*512^2.
#   Epochs are a derived readout and are NOT the budget:
#     qxs  16,001 train, bs16 -> 1001 it/ep ->  250 epochs
#     s2o   1,450 train, bs4  ->  363 it/ep ->  690 epochs
#   Cross-dataset comparability inside the row is what is held fixed.
#
# PER-EPOCH DDIM BLOCK -- a real cost, easy to miss.  E3Diff's main.py runs a
#   full DDIM-50 sample of the training batch at data_it == 0 of EVERY epoch
#   (pure logging; an image is only written every 100k steps).  It is part of
#   the frozen recipe and consumes RNG, so it is deliberately NOT removed --
#   dropping it on one dataset would make that the only non-reproducing cell.
#   Small training sets therefore pay MORE: the block costs ~104 s per epoch at
#   bs16/256 and ~12 s at bs4/512, so the s2o cell (690 epochs of a cheap
#   block) and the qxs cell (250 epochs of an expensive one) land at roughly
#   2 h and 7 h of pure logging on top of training.
#
# TONE OFFSET, INHERITED DELIBERATELY: this config is known to produce
#   generations noticeably brighter than the ground truth on some data, costing
#   several dB of PSNR.  It is NOT corrected per dataset, because
#   cross-dataset comparability inside the row was judged worth more than
#   per-cell tuning.  Read the DDPM PSNR values with that footnote.
#
# Deviation from the reference config: save_checkpoint_freq 10000 -> 25000
#   (a gen+opt pair is 2.3 GB; 25 of them is 54 GB per run for checkpoints
#   nobody reads -- 25k keeps 10 and matches val_freq).
#
# PREREQUISITE: the condition tree
#   $DATA_ROOT/e3diff_<ds>/{train,val}/{SAR,EO,SAR-PPB,SAR-canny}
#   with matching basenames.  See README.md "Data preparation" -- this row
#   CANNOT be run from a SAR PNG alone.
#
# Re-running is safe and cheap: training is skipped if the final checkpoint
#   exists, and inference is skipped if its DDIM dump exists (upstream main.py
#   os.rename()s its dump directory and CRASHES if the renamed directory is
#   already there -- after paying for the whole inference).
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/E3Diff)
#   DATA_ROOT       prepared data, incl. the condition tree e3diff_<ds>/
#   WORK_DIR        run directories, logs, results
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS=${1:-}
case "$DS" in
    qxs|s2o) ;;
    *) echo "usage: GPU=<n> bash run.sh <qxs|s2o>"; exit 2 ;;
esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export E3DIFF_REPO=$BASELINES_ROOT/E3Diff
# SoftPool + vision_aided_loss shims.  E3Diff imports both unconditionally; the
# CLIP discriminator is only USED at stage 2, which this row never runs, so the
# constant-output stub is correct HERE and wrong for the E3Diff row.
export PYTHONPATH=$HERE/shims${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAIN=$HERE/e3diff_rgb_main.py              # RGB wrapper around E3Diff main.py
BASECFG=$HERE/configs/ddpm_rgb_base.json
ITERS=${ITERS:-250000}
DATAROOT=$DATA_ROOT/e3diff_$DS
RUNDIR=$WORK_DIR/e3diff_ddpm_$DS
LOGDIR=$WORK_DIR/logs
OUT=$WORK_DIR/results/${DS}_ddpm_eo
TAG=ddpm_$DS

# resolution / batch per dataset (s2o = the frozen 512 centre-crop protocol;
# both sides are 600px on disk and the E3Diff loader neither crops nor resizes,
# while its 5-level UNet needs a size divisible by 16 and 600/16 = 37.5, so the
# staged condition tree must already hold 512px images)
case "$DS" in
    s2o) RES=512; BS=4  ;;
    *)   RES=256; BS=16 ;;
esac

[ -d "$DATAROOT/train/SAR-PPB" ] || { echo "[$TAG] missing condition tree $DATAROOT"; exit 1; }
mkdir -p "$RUNDIR" "$LOGDIR" "$(dirname "$OUT")"

# job 3 publishes $OUT as a symlink; if a real directory is sitting there the
# ln lands INSIDE it and the cell silently evaluates as empty.  Fail now, not
# after ~12 h of GPU.
if [ ! -L "$OUT" ] && [ -e "$OUT" ]; then
    echo "[$TAG] refusing: $OUT exists and is not a symlink"; exit 2
fi

# ---------------- derive the run config ----------------
CFG=$RUNDIR/${TAG}_train.json
"$PY" - "$BASECFG" "$CFG" "$TAG" "$DATAROOT" "$RES" "$BS" "$ITERS" <<'PYEOF'
import json, sys
base, out, name, root, res, bs, iters = sys.argv[1:8]
cfg = json.load(open(base))
res, bs, iters = int(res), int(bs), int(iters)
cfg['name'] = name
for split in ('train', 'val'):
    d = cfg['datasets'][split]
    d['dataroot'] = f'{root}/{split}'
    d['l_resolution'] = d['r_resolution'] = res
cfg['datasets']['train']['batch_size'] = bs
cfg['model']['diffusion']['image_size'] = res
cfg['train']['n_iter'] = iters
json.dump(cfg, open(out, 'w'), indent=2)
print('wrote', out)
PYEOF

cd "$RUNDIR"

# ---------------- Job 1: training ----------------
# -enable_wandb "" is MANDATORY: the flag defaults to the truthy STRING 'false'.
# CUDA_VISIBLE_DEVICES is exported above and core/logger.py is guard-patched so
# it is not overridden, hence no -gpu flag (config gpu_ids [0] == this card).
CKPT=$(ls -t "$RUNDIR"/experiments/${TAG}_*/checkpoint/I${ITERS}_E*_gen.pth 2>/dev/null | head -1)
if [ -n "${CKPT:-}" ]; then
    echo "[$TAG] $(date) job1 SKIPPED: final checkpoint present ($CKPT)"
else
    echo "[$TAG] $(date) job1: training $ITERS iters, batch $BS @ ${RES}px (GPU $GPU)"
    "$PY" "$MAIN" -c "$CFG" -p train -enable_wandb "" --seed 1 \
        > "$LOGDIR/${TAG}_train.log" 2>&1
    rc=$?
    echo "[$TAG] $(date) job1 exit $rc"
    [ $rc -ne 0 ] && { tail -30 "$LOGDIR/${TAG}_train.log"; exit $rc; }
fi

# ---------------- Job 2: DDIM-50 inference over the full test set ----------------
CKPT=$(ls -t "$RUNDIR"/experiments/${TAG}_*/checkpoint/I${ITERS}_E*_gen.pth 2>/dev/null | head -1)
[ -z "${CKPT:-}" ] && CKPT=$(ls -t "$RUNDIR"/experiments/${TAG}_*/checkpoint/I*_gen.pth 2>/dev/null | head -1)
[ -z "${CKPT:-}" ] && { echo "[$TAG] no checkpoint under $RUNDIR/experiments/${TAG}_*"; exit 1; }
PREFIX="${CKPT%_gen.pth}"

# -p val writes NEXT TO THE CHECKPOINT and renames the directory with its
# metrics (<prefix>_S<ssim>_P<psnr>_l2<..>_Lp<lpips>/sample), never into
# results/.  That os.rename() raises if the renamed directory already exists,
# so a re-run must not repeat a finished inference.
SAMPLE=$(ls -dt "${PREFIX}"_S*/sample 2>/dev/null | head -1)
if [ -n "${SAMPLE:-}" ] && [ "$(ls "$SAMPLE"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$TAG] $(date) job2 SKIPPED: dump already present ($SAMPLE)"
else
    VALCFG=$RUNDIR/${TAG}_val.json
    "$PY" - "$CFG" "$PREFIX" "$VALCFG" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg['phase'] = 'val'
cfg['path']['resume_state'] = sys.argv[2]   # a PREFIX, with no _gen.pth suffix
cfg['datasets']['val']['data_len'] = -1     # the whole test split
json.dump(cfg, open(sys.argv[3], 'w'), indent=2)
PYEOF
    echo "[$TAG] $(date) job2: DDIM-50 inference, ckpt=$PREFIX"
    "$PY" "$MAIN" -c "$VALCFG" -p val -enable_wandb "" --seed 1 \
        > "$LOGDIR/${TAG}_val.log" 2>&1
    rc=$?
    echo "[$TAG] $(date) job2 exit $rc"
    [ $rc -ne 0 ] && { tail -30 "$LOGDIR/${TAG}_val.log"; exit $rc; }
    SAMPLE=$(ls -dt "${PREFIX}"_S*/sample 2>/dev/null | head -1)
fi

# ---------------- Job 3: publish where the evaluator looks ----------------
[ -n "${SAMPLE:-}" ] || { echo "[$TAG] no <prefix>_S*/sample directory found"; exit 1; }
ln -sfn "$SAMPLE" "$OUT"
N=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
echo "[$TAG] $(date) done: $N PNGs -> $OUT -> $SAMPLE"
[ "$N" -gt 0 ] || { echo "[$TAG] FAILED: inference produced no PNGs"; exit 1; }
