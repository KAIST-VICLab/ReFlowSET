# E3Diff (GRSL 2025)

An efficient end-to-end diffusion model for **one-step** SAR-to-optical
translation: a conditional DDPM is first trained normally, then fine-tuned into a
single-step generator with pixel L1, LPIPS, a focal-frequency term and a
vision-aided CLIP GAN loss.

* **Upstream:** <https://github.com/DeepSARRS/E3Diff>
* **Commit vendored:** `38601093ab8f8e4b478144621f20890b100a3b74` (2026-07-11)
* **This row is our own retraining**, not the authors' released weights.

Stage 1 and stage 2 share **one** UNet (`define_G` is stage-agnostic), so this
row and the [`../ddpm-sr3/`](../ddpm-sr3/) row are the same architecture at two
points in the same schedule. Read that README first — the patches and the
condition-tree contract live there and apply here too.

## Patches we applied

The in-repo patch and the two shims are described in
[`../ddpm-sr3/README.md`](../ddpm-sr3/README.md#patches-we-applied):
`core/logger.py` no longer overrides `CUDA_VISIBLE_DEVICES`; `SoftPool.py` is a
pure-PyTorch drop-in for the CUDA extension; `e3diff_rgb_main.py` is the
in-memory loader wrapper that makes the model 3-channel. **Both cells of this row
ran through that wrapper.**

**One difference, and it decides whether you are running the authors' method at
all:** stage 2 must import the **real** `vision_aided_loss` package, not the
constant-output stub that lets stage 1 import. `run.sh` requires you to set
`VAENV` to the directory holding the real package, puts it first on
`PYTHONPATH`, and then runs an explicit assertion that
`vision_aided_loss.__file__` resolves inside it. A reproducer who gets this
wrong trains stage 2 against a constant GAN loss and silently does not run
E3Diff.

Also shipped here: **`make_e3diff_s2_cfg.py`**, our stage-2 config builder. It is
not cosmetic — it handles two upstream traps:

* In `phase=train`, `load_network()` also reads `{resume_state}_opt.pth` and
  sets `begin_step` from it, and `main.py` loops `while current_step < n_iter`.
  So **`n_iter` is an ABSOLUTE step count continuing stage 1**, not a stage-2
  budget. (The authors' own configs show this: resume at I640000, `n_iter`
  800000 = 160 k stage-2 iterations.) The script writes
  `n_iter = begin_step + iters`.
* `main.py` saves only when `current_step % save_checkpoint_freq == 0` and does
  **not** save at end-of-training, so an `n_iter` off that grid throws away every
  iteration since the last multiple. The script snaps `n_iter` down onto the
  grid and says so.

It also arch-checks the stage-1 weights against the config's UNet before
starting, and refuses to emit a config containing `//` — `core/logger.py` strips
everything after a double slash on each line, which would truncate a path.

## Budget, in generator updates

**250,000 (stage 1, inherited) + 60,000 (stage 2) = 310,000 absolute**, on both
datasets. The 60 k is the authors' own ratio: they run stage 2 for 25 % of the
stage-1 budget, and `make_e3diff_s2_cfg.py` defaults to 25 % of whatever step
count the stage-1 checkpoint carries. Checkpoints are therefore named `I310000_*`.
Batch 16 @ 256 (QXS-SAROPT), 4 @ 512 (SAR2Opt) — equal pixels per update.

## Train + inference

```bash
export VAENV=/path/to/real/vision_aided_loss/site-packages
GPU=0 bash ../ddpm-sr3/run.sh qxs     # stage 1 first: 250,000 updates
GPU=0 bash run.sh qxs                 # stage 2: +60,000 -> I310000
GPU=0 bash run.sh s2o
```

`run.sh` finds the stage-1 checkpoint by **highest iteration** (not newest
mtime — a resumed run can rewrite older files), builds the stage-2 config,
trains, then runs `-p val` at `ddim_steps = 1` over the full test split and
symlinks the renamed dump where the evaluator looks. Override the stage-1
checkpoint with `S1_CKPT=<prefix without _gen.pth>`.

## The released weights

`gen.pth` (768.4 MB) — the same `sr3` UNet architecture as the DDPM row, at
`I310000` (250 k stage 1 + 60 k stage 2). At inference it is a **one-step DDIM
generator**, roughly 0.19 s per image.

## Traps

* **The CLIP discriminator stub.** Repeated because it is the expensive one: with
  `../ddpm-sr3/shims` ahead of the real package on `PYTHONPATH`, stage 2 trains
  against a constant GAN loss and produces a plausible-looking, wrong row. Keep
  the assertion in `run.sh`.
* **`n_iter` is absolute, not a stage-2 budget.** See above.
* **`main.py` never saves at end-of-training.** If `n_iter` is off the
  `save_checkpoint_freq` grid, everything since the last multiple is lost.
* **`-p val` writes next to the checkpoint and `os.rename`s the directory** with
  its metrics, and that rename raises if the target exists — after the whole
  inference has been paid for. The published output path is a **symlink** to
  that dump; a real directory sitting there makes `ln` land *inside* it and the
  cell evaluates as empty.
* **The model loads with `strict=False`.** A config/checkpoint mismatch loads
  nothing and raises nothing. Verify the `Loading pretrained model for G [...]`
  log line.
* **This is not an oracle.** `model/sr3_modules/diffusion.py:511` passes
  `condition_x = x = data['SR']`, i.e. the `[PPB, canny, SAR]` stack; the ground
  truth never enters the sampler. (Worth stating, because one other method in
  this table does condition on the ground truth — see
  [`../stegogan/README.md`](../stegogan/README.md).)
* **`-enable_wandb ""` is mandatory** — the flag defaults to the truthy string
  `'false'`.
* **Licence: the upstream repository publishes none.** The full disclosure is in
  [`../ddpm-sr3/README.md`](../ddpm-sr3/README.md#traps) and applies verbatim
  here: no LICENSE/COPYING/NOTICE, no README licence section, `license: null`
  from the GitHub API (checked 2026-08-28); `SoftPool/LICENSE.txt` is a vendored
  dependency's MIT licence, not a grant for E3Diff. All rights reserved by
  default; we publish the checkpoint anyway so the comparison is reproducible and
  say so plainly.
