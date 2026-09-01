#!/usr/bin/env bash
# E3Diff STAGE 2 (the authors' own two-stage method), SAR->EO.
#
#   GPU=0 bash run.sh <qxs|s2o>
#
# WHAT STAGE 2 IS.  E3Diff's two stages share ONE UNet (define_G is
# stage-agnostic), so this is the same architecture as the DDPM row:
#   stage 1 ("stage": 1) = the SR3-class conditional DDPM   -> ../ddpm-sr3/
#   stage 2 ("stage": 2) = the SAME net fine-tuned into a 1-step generator:
#     diffusion.py p_losses() runs ddim_sample() WITH GRAD from pure noise for
#     `ddim_steps` steps and takes L1(GT, sample) directly on pixels, plus
#     LPIPS (lpips_w 5), focal-frequency (fft_w 10) and a vision-aided CLIP GAN
#     (lambda_gan 0.5) -- the authors' SAR2EO_256_s2_1step.json recipe.
# All stage 2 needs from stage 1 is the generator weights: model.py
# load_network() does torch.load(f'{resume_state}_gen.pth') with strict=False.
# Hence S1_CKPT below is a checkpoint PREFIX and can point anywhere.
#
# BUDGET.  The authors resume stage 2 from stage-1 I640000 and run to n_iter
# 800000 -> 160k stage-2 iterations = 25% of the stage-1 budget.  We keep that
# ratio: make_e3diff_s2_cfg.py defaults to 25% of whatever step count the
# stage-1 checkpoint carries.  With our 250,000-update stage 1 that is
#   250,000 (stage 1, inherited) + 60,000 (stage 2) = 310,000 ABSOLUTE.
# n_iter is ABSOLUTE (main.py resumes current_step from the stage-1 _opt.pth
# and loops `while current_step < n_iter`), which the config maker handles.
#
# INFERENCE.  Stage 2 samples in ddim_steps = 1 DDIM step (~0.19 s/img), so the
# full test set is affordable and no subset is used.  `-p val` writes images
# NEXT TO THE CHECKPOINT and then renames that directory with its metrics; we
# locate the renamed directory and symlink it where the evaluator looks.
#
# Environment
#   BASELINES_ROOT  vendored upstream repositories ($BASELINES_ROOT/E3Diff)
#   DATA_ROOT       prepared data, incl. the condition tree e3diff_<ds>/
#   WORK_DIR        run directories, logs, results (and the stage-1 run dirs)
#   VAENV           directory holding the REAL vision_aided_loss package
#   PY              python interpreter (default: python)
#   GPU             GPU index (default: 0)
#   S1_CKPT         override the stage-1 checkpoint prefix (no _gen.pth)
#   S2_ITERS        override the stage-2 iteration count (0 = 25% of stage 1)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS=${1:?dataset: qxs|s2o}
case "$DS" in qxs|s2o) ;; *) echo "[e3diff-s2] unknown dataset $DS"; exit 1 ;; esac
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export E3DIFF_REPO=$BASELINES_ROOT/E3Diff
# VAENV FIRST: it holds the REAL vision-aided CLIP discriminator, which must
# shadow ../ddpm-sr3/shims/vision_aided_loss.py -- a constant-output stub that
# exists only so stage 1 (lambda_gan = 0) can import it.  Getting this wrong
# trains stage 2 against a constant GAN loss, i.e. not the authors' method.
export PYTHONPATH=${VAENV:?set VAENV to the directory holding the real vision_aided_loss}:$HERE/../ddpm-sr3/shims
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGS=$WORK_DIR/logs
WORK=$WORK_DIR/e3diff_s2/$DS
OUT=$WORK_DIR/results/${DS}_e3diff_eo
TAG=e3diff_s2_$DS
mkdir -p "$LOGS" "$WORK"

# batch / resolution.  SAR2Opt is the frozen 512 centre-crop protocol; E3Diff's
# transform_augment does NO crop or resize, so the staged tree already holds
# 512px images.  batch 4 @ 512 == the pixel budget of 16 @ 256.
case "$DS" in
  s2o) DEFBATCH=4; DEFRES=512 ;;
  *)   DEFBATCH=16; DEFRES=256 ;;
esac
BATCH=${BATCH:-$DEFBATCH}
RES=${RES:-$DEFRES}

# dataroot must carry SAR/ EO/ SAR-PPB/ SAR-canny/ under both train/ and val/
# (data/LRHR_dataset.py matches PPB and canny by the EO file's basename).
ROOT=${DATAROOT:-$DATA_ROOT/e3diff_$DS}
for s in train val; do for c in SAR EO SAR-PPB SAR-canny; do
  [ -d "$ROOT/$s/$c" ] || { echo "[e3diff-s2] missing $ROOT/$s/$c -- build the condition tree first (see ../ddpm-sr3/README.md)"; exit 1; }
done; done

# stage-1 checkpoint prefix = the highest-ITERATION I<N>_E<M>_gen.pth we can
# find (highest iteration, NOT newest mtime: a resumed run can rewrite older
# files).  This is the DDPM row's output.
if [ -n "${S1_CKPT:-}" ]; then S1=$S1_CKPT; else
  S1=$(for f in "$WORK_DIR"/e3diff_ddpm_${DS}/experiments/*/checkpoint/I*_gen.pth; do
         [ -f "$f" ] && echo "$f"; done \
       | sed 's/_gen\.pth$//' \
       | awk -F/ '{n=$NF; sub(/^I/,"",n); sub(/_E.*/,"",n); print n"\t"$0}' \
       | sort -k1,1n | tail -1 | cut -f2-)
fi
[ -n "${S1:-}" ] && [ -f "${S1}_gen.pth" ] || {
  echo "[e3diff-s2] no stage-1 checkpoint for $DS; run ../ddpm-sr3/run.sh $DS first,"
  echo "  or set S1_CKPT=<prefix without _gen.pth>"; exit 1; }

# Stage 2 must match its own stage 1's channel count, so read the answer out of
# the checkpoint rather than guessing.  Our stage 1 is 3-channel on both
# datasets and therefore uses the RGB wrapper as its entry point.
CH=$("$PY" -c "
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(sd['denoise_fn.downs.0.weight'].shape[1])" "${S1}_gen.pth") || exit 1
if [ "$CH" = "3" ]; then
  MAIN=$HERE/../ddpm-sr3/e3diff_rgb_main.py; S1_CFG_DEFAULT=$HERE/configs/ddpm_rgb_base.json
else
  MAIN=$BASELINES_ROOT/E3Diff/main.py; S1_CFG_DEFAULT=$BASELINES_ROOT/E3Diff/config/SAR2EO_256_s1_ddpm.json
fi
# config template: architecture + optimiser come from it; data roots, batch and
# resolution are always set explicitly below, so a base config is safe.
S1_CFG=${S1_CFG:-$WORK_DIR/e3diff_ddpm_$DS/ddpm_${DS}_train.json}
[ -f "$S1_CFG" ] || S1_CFG=$S1_CFG_DEFAULT
[ -f "$S1_CFG" ] || { echo "[e3diff-s2] no stage-1 config template"; exit 1; }

echo "[e3diff-s2/$DS] GPU $GPU | dataroot $ROOT | ${CH}ch ${RES}px batch $BATCH"
echo "[e3diff-s2/$DS] stage-1 $S1"
echo "[e3diff-s2/$DS] entry point $MAIN"

# the stub-vs-real discriminator check: a stub would silently train stage 2
# with a constant GAN loss, i.e. not the authors' method at all.
"$PY" - <<'PYEOF' || exit 1
import os, sys, vision_aided_loss
p = os.path.abspath(vision_aided_loss.__file__)
if os.path.abspath(os.environ['VAENV']) not in p:
    sys.exit('[e3diff-s2] vision_aided_loss resolved to %s, not the real package '
             'under $VAENV; PYTHONPATH must put VAENV first' % p)
print('[e3diff-s2] vision_aided_loss OK:', p)
PYEOF

# ---------------- stage-2 config ----------------
S2_ITERS=${S2_ITERS:-0}          # 0 = 25% of the stage-1 step count -> 60,000
VAL_LEN=${VAL_LEN:--1}           # -1 = the whole test split
NAME=${DS}_e3diff_s2
TRAINCFG=$WORK/${DS}_s2_train.json
"$PY" "$HERE/make_e3diff_s2_cfg.py" \
  --s1-cfg "$S1_CFG" --resume "$S1" --out "$TRAINCFG" --name "$NAME" \
  --phase train --iters "$S2_ITERS" --ddim-steps 1 \
  --train-root "$ROOT/train" --val-root "$ROOT/val" --batch "$BATCH" --res "$RES" \
  --val-freq 20000 --save-freq 10000 --print-freq 50 --val-len 20 || exit 1

# ---------------- stage-2 training ----------------
cd "$WORK" || exit 1
echo "[e3diff-s2/$DS] training -> $LOGS/${TAG}_train.log"
"$PY" "$MAIN" -c "$TRAINCFG" -p train -enable_wandb "" --seed 1 \
  > "$LOGS/${TAG}_train.log" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "[e3diff-s2/$DS] TRAINING FAILED rc=$rc"; tail -25 "$LOGS/${TAG}_train.log"; exit $rc; }

# newest experiment dir for THIS name (each run stamps its own timestamp), then
# its highest-iteration checkpoint -- never a leftover from an earlier attempt.
EXP=$(ls -dt "$WORK"/experiments/"${NAME}"_*/ 2>/dev/null | head -1)
S2=$(ls "$EXP"checkpoint/I*_gen.pth 2>/dev/null \
     | sed 's/_gen\.pth$//' \
     | awk -F/ '{n=$NF; sub(/^I/,"",n); sub(/_E.*/,"",n); print n"\t"$0}' \
     | sort -k1,1n | tail -1 | cut -f2-)
if [ -z "$S2" ]; then
  echo "[e3diff-s2/$DS] training exited 0 but produced NO checkpoint under $EXP"
  echo "  (main.py saves only on current_step % save_checkpoint_freq == 0 and"
  echo "   never at end-of-training -- check n_iter is on that grid)"
  exit 1
fi
echo "[e3diff-s2/$DS] stage-2 checkpoint $S2"

# ---------------- inference on the test split ----------------
VALCFG=$WORK/${DS}_s2_val.json
"$PY" "$HERE/make_e3diff_s2_cfg.py" \
  --s1-cfg "$S1_CFG" --resume "$S2" --out "$VALCFG" --name "${NAME}_val" \
  --phase val --ddim-steps 1 \
  --train-root "$ROOT/train" --val-root "$ROOT/val" --batch "$BATCH" --res "$RES" \
  --val-len "$VAL_LEN" || exit 1

echo "[e3diff-s2/$DS] inference (1-step DDIM, data_len $VAL_LEN) -> $LOGS/${TAG}_val.log"
"$PY" "$MAIN" -c "$VALCFG" -p val -enable_wandb "" --seed 1 \
  > "$LOGS/${TAG}_val.log" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "[e3diff-s2/$DS] INFERENCE FAILED rc=$rc"; tail -25 "$LOGS/${TAG}_val.log"; exit $rc; }

# main.py renames "$S2" (a directory it makes beside the checkpoint) to
# "$S2"_S<ssim>_P<psnr>_l2<l2>_Lp<lpips> -- the suffix is not known in advance.
SAMPLE=$(ls -d "$S2"_S*/sample 2>/dev/null | tail -1)
[ -n "$SAMPLE" ] && [ -d "$SAMPLE" ] || SAMPLE=$([ -d "$S2/sample" ] && echo "$S2/sample")
[ -n "$SAMPLE" ] || { echo "[e3diff-s2/$DS] NO sample dir beside $S2 -- inference wrote nothing"; exit 1; }

# never report success on an empty output directory
n=$(ls "$SAMPLE" | wc -l)
[ "$n" -eq 0 ] && { echo "[e3diff-s2/$DS] $SAMPLE is EMPTY"; exit 1; }
mkdir -p "$(dirname "$OUT")"
[ -L "$OUT" ] && rm -f "$OUT"
[ -d "$OUT" ] && rmdir "$OUT" 2>/dev/null
[ -e "$OUT" ] && { echo "[e3diff-s2/$DS] $OUT exists and is not an empty dir/symlink; refusing"; exit 1; }
ln -s "$SAMPLE" "$OUT"
echo "[e3diff-s2/$DS] done: $n images, $OUT -> $SAMPLE"
