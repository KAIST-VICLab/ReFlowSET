# Comparison-method reproduction kit

Everything needed to retrain and re-infer the **fifteen** comparison rows of the
paper's main table, on both datasets, and land on the numbers we print.

Each row was retrained by us. No number in the table is copied from a source
paper. That is deliberate: published SAR-to-EO numbers are not comparable across
papers — different splits, different resolutions, different LPIPS conventions,
different test-set sizes — so the only way to get a table that means something is
to run all of it under one protocol. This directory is that protocol, written
down.

```
baselines/
├── README.md                  <- you are here
├── pix2pix/                   CVPR'17
├── cyclegan/                  ICCV'17
├── pix2pixhd/                 CVPR'18
├── spade/                     CVPR'19
├── ddpm-sr3/                  TPAMI'22 row  (E3Diff stage 1 — see below)
├── sd21-ft/                   CVPR'22 row   (C-DiffSET stage 1 — see below)
├── bbdm/                      CVPR'23
├── controlnet/                ICCV'23
├── hi-diff/                   NeurIPS'23
├── resshift/                  NeurIPS'23
├── stegogan/                  CVPR'24
├── conditional-diffusion/     GRSL'24
├── cbbdm/                     GRSL'25
├── e3diff/                    GRSL'25
└── c-diffset/                 TCSVT'26
```

Every method directory contains a `README.md` (what the method is, the upstream
commit, **every patch we applied and why**, the budget, the exact train and
inference commands, which released weight file is which module, and a **Traps**
section), the launch scripts we actually ran, and any config file we had to add
to the upstream repository.

---

## The fifteen methods

| Directory | Paper name in the table | Venue | Upstream repository | Commit we vendored |
|---|---|---|---|---|
| `pix2pix` | pix2pix | CVPR'17 | <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix> | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| `cyclegan` | CycleGAN | ICCV'17 | <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix> | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| `pix2pixhd` | pix2pixHD | CVPR'18 | <https://github.com/NVIDIA/pix2pixHD> | `14b3b3c7fff413086e3b58df52096f16b6891172` |
| `spade` | SPADE | CVPR'19 | <https://github.com/NVlabs/SPADE> | `fecacc920c1367a038995c45a39c15f6521ca64f` |
| `ddpm-sr3` | DDPM (SR3-class) | TPAMI'22 | <https://github.com/DeepSARRS/E3Diff> (stage 1) | `38601093ab8f8e4b478144621f20890b100a3b74` |
| `sd21-ft` | SD2.1 fine-tune only | CVPR'22 | <https://github.com/KAIST-VICLab/C-DiffSET> (stage 1) | `6fc3802` |
| `bbdm` | BBDM | CVPR'23 | <https://github.com/xuekt98/BBDM> | `02c3b13c9f9dfab0853e32123100680a0640c4ed` |
| `controlnet` | ControlNet | ICCV'23 | `examples/controlnet/train_controlnet.py` from <https://github.com/huggingface/diffusers> | — (loose scripts, see that README) |
| `hi-diff` | HI-Diff | NeurIPS'23 | <https://github.com/zhengchen1999/HI-Diff> | `b3bfd167997e27f8edd57681cf70e5031a0e35f2` |
| `resshift` | ResShift | NeurIPS'23 | <https://github.com/zsyOAOA/ResShift> | `bb03b7d21614cace01787e097c8a6ab6b945227d` |
| `stegogan` | StegoGAN | CVPR'24 | <https://github.com/sian-wusidi/StegoGAN> | `cad61997c0f82793444f60f81298142b80cdf3c1` |
| `conditional-diffusion` | Conditional Diffusion | GRSL'24 | <https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation> | **not recorded — see below** |
| `cbbdm` | cBBDM | GRSL'25 | <https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical> | `8ce15934f4d4e3f01efe70d11e2d9b9e0859210c` |
| `e3diff` | E3Diff | GRSL'25 | <https://github.com/DeepSARRS/E3Diff> (stage 2) | `38601093ab8f8e4b478144621f20890b100a3b74` |
| `c-diffset` | C-DiffSET | TCSVT'26 | <https://github.com/KAIST-VICLab/C-DiffSET> | `6fc3802` |

**Conditional Diffusion is the one reproducibility hole in the table.** Our
vendored copy of that repository carries no `.git`, no `.gitmodules` and no
version marker, so the exact upstream commit we built against **is not
recorded** and cannot be recovered. Everything else pins a commit. See
`conditional-diffusion/README.md`; treat the patch list there as the
specification and re-apply it to whatever upstream state you clone.

### Two rows are named for a paper whose code we did not run

State this plainly wherever these numbers are quoted:

* **DDPM (SR3-class)** is **E3Diff's stage 1**, not the SR3 authors' release. The
  row reproduces the SR3 *method class* — a conditional pixel-space DDPM — using
  the stage-1 configuration of the E3Diff code base. Both cells additionally ran
  through a small in-memory wrapper that makes E3Diff's loader emit RGB targets
  (`ddpm-sr3/e3diff_rgb_main.py`); upstream `main.py` hard-codes a 1-channel
  target and would crash on an RGB dataset.
* **SD2.1 fine-tune only** is **C-DiffSET's stage 1**, not a generic Stable
  Diffusion fine-tune written from the LDM paper. It is an SD 2.1-base UNet whose
  input convolution is widened from 4 to 8 channels so the SAR latent can be
  concatenated; the output convolution keeps its 4 channels (no confidence head).
  A reader who tries to load it into a stock SD 2.1 pipeline will fail on
  `conv_in`.

---

## What makes these numbers comparable — four rules

Everything below was held fixed across all sixteen rows, ours included.

**1. Identical splits.** Both datasets use the official C-DiffSET split lists,
frozen and shipped in this repository under [`../splits/`](../splits/):
QXS-SAROPT **16,001 train / 3,999 test**, SAR2Opt **1,450 train / 627 test**. No
method re-partitions, re-samples or sub-samples. Train ∩ test = ∅ on both.

**2. Identical test items, at identical resolution.** Every method is scored on
the same **3,999** QXS-SAROPT images at 256×256 and the same **627** SAR2Opt
images at 512×512. SAR2Opt's 600 px tiles are reduced to 512 by a **center crop
at offset 44 — never a resize**; where a method's own loader would have resized,
we pre-materialised the center crops and pointed it at those (this bites
pix2pixHD, SPADE and Conditional Diffusion; each README says so). No row is a
subset row, and no row's `n` differs from any other's.

**3. One evaluator, run by us, on the images each method actually wrote.**
PSNR and SSIM per image via `torchmetrics` at `data_range=1`; LPIPS with the VGG
backbone fed `[-1, 1]`; FID via `pytorch-fid` against the ground-truth test set;
DISTS via `pyiqa`. Where a generated image and the ground truth differ in size
the **ground truth** is center-cropped to the generated size. See
[`../README.md`](../README.md#evaluation) — and note the LPIPS warning there: the
other convention in this literature (feeding `[0,1]` with `normalize=False`)
reads about 0.05 lower and is not comparable with anything here.

**4. Budgets are quoted in GENERATOR UPDATES, never epochs.** This is not
pedantry. The pix2pix row's "100 + 100 epochs" is 36,400 updates on the
1,450-image set at batch 8, and the same epoch count on the 16,001-image set at
batch 16 would be 200,200 — a 5.5× difference that reads as budget-matched and
produced two wrong conclusions in this project before the rule was adopted.
Every launch script header states the update count and how it was derived, and
every method README repeats it per dataset. Two counting conventions are in play
and they disagree by up to a few hundred updates: the junyanz and StegoGAN
loaders have no `drop_last`, so their iterations per epoch **round up**
(`ceil(N/batch)`); SPADE (`drop_last=isTrain`) and pix2pixHD (its
`__len__` floor-rounds to a multiple of the batch) **round down**. Both are
given where they matter; do not re-derive one member of a family the other way.

### Two disclosures that belong with the table

* **Four cells fail an input-copy audit.** CycleGAN and StegoGAN, on *both*
  datasets, produce output that is closer to the SAR **input** than to the EO
  **target** (mean |gen − SAR| < mean |gen − GT|). We report those rows in place,
  unchanged and marked, rather than removing or substituting them: what a reader
  is owed is what the released implementation does at its own published
  protocol. Ratios are in `cyclegan/README.md` and `stegogan/README.md`.
* **StegoGAN has an oracle output and a deployable one**, and the difference cost
  this project two days. We score the deployable one. See `stegogan/README.md`.

---

## Environment contract

Every script in this kit is parameterised by these variables and contains no
absolute paths. Set them once:

| Variable | Meaning |
|---|---|
| `BASELINES_ROOT` | directory holding the vendored upstream repositories, one subdirectory per method (`$BASELINES_ROOT/BBDM`, `$BASELINES_ROOT/SPADE`, …) |
| `DATA_ROOT` | prepared data: the raw `QXS_AB/` and `sar2opt/` trees plus the per-method adapter trees each README describes |
| `WORK_DIR` | scratch — `logs/`, `results/`, per-run experiment directories |
| `CKPT_ROOT` | where training checkpoints are written and read from |
| `SPLITS_ROOT` | this repository's [`splits/`](../splits/) directory |
| `PY` | python interpreter (defaults to `python`) |
| `GPU` | index of the GPU to use (defaults to `0`) |

```bash
export BASELINES_ROOT=/path/to/vendored/repos
export DATA_ROOT=/path/to/data
export WORK_DIR=/path/to/work
export CKPT_ROOT=/path/to/checkpoints
export SPLITS_ROOT="$PWD/splits"
export PY=python
```

**Config files carry the same `${VAR}` spelling, but YAML and JSON loaders do not
expand shell variables.** Either edit those few lines by hand, or expand them
once:

```bash
envsubst < baselines/bbdm/configs/QXS-LBBDM-f4.yaml > $BASELINES_ROOT/BBDM/configs/QXS-LBBDM-f4.yaml
```

Each method README lists exactly which lines carry a placeholder.

## The dataset directory layout the scripts assume

```
$DATA_ROOT/
├── QXS_AB/                 trainA,trainB,testA,testB   (A = SAR, B = EO; 256 px PNG)
├── QXSLAB_SAROPT/          the raw QXS-SAROPT release   (opt_256_oc_0.2/, sar_256_oc_0.2/)
├── sar2opt/                trainA,trainB,testA,testB   (A = SAR, B = EO; 600 px JPEG)
├── s2o_512/                testA,testB — the center-512 crops of the 627 test tiles
└── <method adapter trees>  built by the prep step each method README documents
```

`QXS_AB` and `QXSLAB_SAROPT` are two views of the same imagery: the `_AB` tree is
the split-list selection materialised as flat `trainA/trainB/testA/testB`
directories, which is what every GAN-family loader wants; the raw tree plus the
split lists is what the list-driven methods (C-DiffSET family) read. Build the
`_AB` view from the split lists in [`../splits/`](../splits/) so the two can
never disagree.

**Nothing in this kit redistributes imagery.** Obtain QXS-SAROPT and SAR2Opt from
their authors; the links are in the top-level README.

## Weights

Every cell's trained weights are published — see
[`../MODEL_ZOO.md`](../MODEL_ZOO.md) for the per-row file list, size, metric
values and licence. Four methods' upstream repositories ship **no licence file at
all** (StegoGAN, E3Diff, the E3Diff-derived DDPM row, Conditional Diffusion);
those cells carry an explicit disclosure in their own README and in the model
card, because a reader deciding whether to redistribute them needs the fact, not
an implication. Two carry non-commercial terms (SPADE, ResShift). Read the
method's README before reusing its checkpoint.
