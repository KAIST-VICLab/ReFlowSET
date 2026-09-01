# DDPM, SR3-class (the TPAMI'22 row)

A conditional pixel-space denoising diffusion model in the SR3 style: the source
image conditions the reverse process directly, with no latent space and no
adversarial term.

**This row is not the SR3 authors' code.** It is **E3Diff run in its stage-1
configuration** (`"stage": 1`), which is exactly an SR3-class conditional DDPM,
and it is the same code base and the same architecture as the
[`../e3diff/`](../e3diff/) row — that row is this network fine-tuned into a
one-step generator. Anywhere this number is quoted, say that it reproduces the
SR3 *method class* using E3Diff's stage-1 code.

* **Upstream:** <https://github.com/DeepSARRS/E3Diff>
* **Commit vendored:** `38601093ab8f8e4b478144621f20890b100a3b74` (2026-07-11)
* **Network:** the repo's `sr3` UNet — `inner_channel 64`,
  `channel_multiplier [1,2,4,8,16]`, `attn_res []`, `res_blocks 1`, 3 in / 3 out,
  3-channel condition. T = 1000 linear at train time, **DDIM-50 at inference**.

## Patches we applied

One inside the repository, and **three outside it that are just as
load-bearing**.

| File:line | Change | Why |
|---|---|---|
| `core/logger.py:54-62` | `os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list` made conditional, so an outer `CUDA_VISIBLE_DEVICES` wins | upstream unconditionally rewrites the variable from the JSON `gpu_ids`, which silently remaps every job onto physical GPU 0 |

Outside the repository, shipped here:

1. **`e3diff_rgb_main.py`** — a 60-line wrapper that patches
   `SAR2EODataset.__getitem__` **in memory** and then `runpy`s the repo's
   `main.py` verbatim. Upstream's loader hard-codes `ch = 1` (a grayscale EO
   target) and a 2-channel `[PPB, canny]` condition; both of our datasets have
   RGB optical targets. With `channels = 3`, upstream's `ddim_sample()` line
   `img_onestep = [condition_x[:, :self.channels, ...]]` slices a 2-channel
   condition and `torch.concat`s it with 3-channel predictions →
   `RuntimeError` on the first validation image. The wrapper returns
   `HR = EO[0:3]`, `LR = SAR[0:3]` and `SR = cat(PPB[0:1], canny[0:1], SAR[0:1])`.
   **Both table cells of this row ran through this wrapper**, not through
   `E3Diff/main.py` directly.
2. **`shims/SoftPool.py`** — a pure-PyTorch drop-in for the SoftPool CUDA
   extension, which E3Diff imports unconditionally in
   `model/sr3_modules/unet.py` (CPEN, 2×2 stride 2).
   `softpool(x) = avgpool(x·eˣ) / avgpool(eˣ)`: exact maths, exact autograd, no
   build step.
3. **`shims/vision_aided_loss.py`** — a **constant-output stub** for the CLIP
   discriminator, needed only so this row can *import* it. Stage 1 has
   `lambda_gan = 0`, so it is never used here. **The E3Diff (stage 2) row must
   NOT use this stub** — see [`../e3diff/README.md`](../e3diff/README.md).

`run.sh` puts `shims/` on `PYTHONPATH` and exports `E3DIFF_REPO` for you.

## Budget, in generator updates

**250,000 updates on both datasets.** Batch 16 @ 256 (QXS-SAROPT) and batch 4 @
512 (SAR2Opt) — equal pixels per update, `16·256² = 4·512²`. Epochs are a
derived readout, not the budget: 250 and 690 respectively.

## Data preparation — this row cannot be run from a SAR PNG alone

The loader looks conditions up by the **EO basename**, and needs, under both
`train/` and `val/`:

```
$DATA_ROOT/e3diff_<ds>/{train,val}/
    SAR/<stem>.png         the SAR image
    EO/<stem>.png          the EO target
    SAR-PPB/<stem>.png     PPB-despeckled SAR
    SAR-canny/<stem>.png   Canny edges of the despeckled SAR
```

with identical basenames across the four directories. We generate the two
condition images as E3Diff itself defines them:

* **SAR-PPB** = FAST_PPB (Deledalle et al., 2009) at `P=3, W=10, h=0.5` — the
  same filter as the upstream repo's `FAST_PPB.m` / `SAR2EO_filter.m`, ported to
  run on GPU;
* **SAR-canny** = `cv2.Canny(ppb_uint8, 50, 150, L2gradient=True)`, as in the
  repo's `canny_dataset.py`.

Our generator script is not shipped (it is entangled with this project's data
layout), but the contract above and the upstream sources fully determine the
result. For SAR2Opt, stage the **center-512 crops** into this tree: the E3Diff
loader neither crops nor resizes, and its five-level UNet needs a size divisible
by 16 (600/16 = 37.5).

## Train + inference

```bash
GPU=0 bash run.sh qxs     # 250,000 updates, batch 16 @ 256
GPU=0 bash run.sh s2o     # 250,000 updates, batch  4 @ 512
```

Inference alone is the same entry point with `-p val`:

```bash
export E3DIFF_REPO=$BASELINES_ROOT/E3Diff
export PYTHONPATH=$PWD/shims
$PY e3diff_rgb_main.py -c <ddpm_<ds>_val.json> -p val -enable_wandb "" --seed 1
```

The val config is the train config with `phase: "val"`,
`path.resume_state = "<...>/checkpoint/I250000_E<E>"` (**a prefix — no
`_gen.pth` suffix**) and `datasets.val.data_len = -1`.

## The released weights

`gen.pth` (768.4 MB) — the `sr3` UNet at `I250000`, stage 1. One file, the whole
row. The E3Diff row ships the same architecture at `I310000`.

## Traps

* **`-enable_wandb ""` is mandatory.** The flag defaults to the truthy *string*
  `'false'`; omitting it turns wandb on.
* **`-p val` writes next to the checkpoint and then `os.rename`s the directory**
  to `<prefix>_S<ssim>_P<psnr>_l2<l2>_Lp<lpips>` (`main.py:319`). That rename
  **raises if the target already exists — after the entire inference has been
  paid for.** `run.sh` guards by detecting an existing `<prefix>_S*/sample`.
* **The published output directory is a SYMLINK** to that renamed dump. If a real
  directory is sitting at the destination, `ln` lands *inside* it and the cell
  silently evaluates as empty. `run.sh` refuses to start in that state.
* **The model loads with `strict=False`** (`model/model.py:317`). A
  config/checkpoint mismatch loads **nothing** and raises no error. Verify the
  `Loading pretrained model for G [...]` line in the log and eyeball the first
  output.
* **The per-epoch DDIM block is part of the recipe.** `main.py` runs a full
  DDIM-50 sample of the training batch at the first iteration of every epoch,
  purely for logging. It is not removed — it consumes RNG, and dropping it on one
  dataset would make that cell a different run than the others — but it means a
  *small* training set pays *more* logging time, because it has more epochs for
  the same number of updates.
* **Tone offset, inherited deliberately.** This frozen config is known to produce
  output noticeably brighter than the ground truth on some data, costing several
  dB of PSNR. It is not tuned per dataset; read the DDPM PSNR column with that
  footnote.
* **Licence: the upstream repository publishes none.** No LICENSE, COPYING or
  NOTICE at the repository root, no licence section in the README, and the GitHub
  API reports no declared licence (checked 2026-08-28). The one licence file in
  the tree, `SoftPool/LICENSE.txt`, is the MIT licence of a vendored third-party
  dependency (Copyright (c) 2020 Alexandros Stergiou) and is **not** a grant
  covering E3Diff. Under default copyright all rights are reserved by the authors
  and **no express permission to redistribute derived work has been granted**. We
  publish this checkpoint anyway so the comparison is reproducible, and state the
  position plainly. E3Diff builds on SR3
  (`Janspiry/Image-Super-Resolution-via-Iterative-Refinement`, Apache-2.0),
  `GaParmar/img2img-turbo` (MIT) and `alexandrosstergiou/SoftPool` (MIT); those
  licences cover the borrowed parts only.
