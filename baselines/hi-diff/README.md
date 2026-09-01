# HI-Diff (NeurIPS 2023)

Hierarchical integration diffusion for image restoration: rather than diffusing
the image, HI-Diff runs a small diffusion model in a **compact latent prior
space** and injects the resulting prior into a Transformer restoration network.
Here the degraded input is the SAR image and the target is the EO image.

* **Upstream:** <https://github.com/zhengchen1999/HI-Diff>
* **Commit vendored:** `b3bfd167997e27f8edd57681cf70e5031a0e35f2` (2025-01-15)
* Built on BasicSR, Restormer and DiffIR (per the upstream README).

## Patches we applied

Two, both torch ≥ 2.6 compatibility.

| File:line | Change | Why |
|---|---|---|
| `hi_diff/utils/base_model.py:294` | `torch.load(..., weights_only=False)` | torch 2.6 flipped the default |
| `train.py:92` | same, for the resume state | ditto |

We also **deleted** three placeholder READMEs (`experiments/README.md`,
`experiments/pretrained_models/README.md`, `results/README.md`) because
`experiments/` and `results/` were replaced by **symlinks** into a work
directory. A reproducer must either recreate those two symlinks or edit the
absolute paths in the option ymls.

We **added** the six option files in [`configs/`](configs/) — three per dataset.

## Budget, in generator updates

**25,000 (stage 1) + 25,000 (stage 2) = 50,000 iterations**, batch 8 @ 256, on
both datasets. The official recipe is 300 k + 300 k with a progressive
128 → 384 patch/batch schedule (16 @ 128 down to 2 @ 384); progressive training
is **disabled** here — both datasets are fixed-size — and the budget is matched
to the other diffusion rows in the table. Batch 16 @ 256 does not fit even on a
very large card.

SAR2Opt trains on random **256** crops of its 600 px tiles and is **tested on the
627 center-512 crops**; the network is fully convolutional, so this is
consistent, and the random crops give a small training set more pixel variety.

## Data preparation

The train ymls read `<DATA_ROOT>/QXS_AB/train{A,B}` and
`<DATA_ROOT>/sar2opt/train{A,B}` directly, plus a small monitoring split at
`<DATA_ROOT>/<ds>_hidiff/val{A,B}` (used only to watch PSNR during training — it
selects nothing, because the test ymls pin the *last* checkpoint). The test ymls
read `QXS_AB/test{A,B}` and `s2o_512/test{A,B}`.

The LQ side is read with `cv2.IMREAD_COLOR`, so a grayscale SAR PNG is replicated
to three channels automatically.

## Train + test

```bash
GPU=0 bash run.sh qxs      # S1 25k + S2 25k, then the test pass
GPU=0 bash run.sh s2o
```

If training already finished and only the test/collect step is missing:

```bash
GPU=0 bash test_only.sh qxs
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/HI-Diff
$PY test.py -opt options/test/QXS.yml
```

Output lands at
`<results>/test_HI_Diff_<TAG>/visualization/<TAG>/<stem>.png`, which the scripts
copy to `results/<ds>_hidiff_eo/`.

## The released weights — five files, of which only three are used at test time

| File | Module | Needed at inference? |
|---|---|---|
| `S2_net_g_latest.pth` (101.4 MB) | the final Transformer | **yes** |
| `S2_net_le_dm_latest.pth` (2.2 MB) | stage-2 latent encoder, `in_chans: 3` (LQ only) | **yes** |
| `S2_net_d_latest.pth` (10.5 MB) | the 8-step latent denoiser | **yes** |
| `S1_net_g_latest.pth` (101.4 MB) | stage-1 Transformer | no — reproduces stage-2 training |
| `S1_net_le_latest.pth` (2.4 MB) | stage-1 latent encoder, `in_chans: 6` (LQ‖GT) | no — **not usable at test time** |

The stage-1 latent encoder takes the ground truth as part of its input; that is
what stage 2 exists to replace. `options/test/*.yml` loads
`pretrain_network_g`, `pretrain_network_le_dm` and `pretrain_network_d` and
nothing else. The latent prior runs 8 timesteps on a 4×4 latent
(`diffusion_schedule.timesteps: 8`, `linear_start 0.1`, `linear_end 0.99`).

To drive it manually: instantiate `hi_diff.archs`' `Transformer`,
`latent_encoder_gelu` (stage 2, `in_chans: 3`) and `denoising` with the yml's
parameters, load the `params` key from each `.pth`, then follow
`hi_diff/models/hi_diff_s2_model.py`'s `test()`: `le_dm` produces the prior from
the LQ alone, `net_d` denoises it, `net_g` conditions on it.

## Traps

* **`run.sh` is not idempotent, and re-running it is expensive.** Both train ymls
  set `resume_state: ~`, `--auto_resume` is never passed (it defaults to False),
  and lines 84-90 of our original launcher ran S1, S2 and test unconditionally
  with no "checkpoint exists" guard. basicsr's `make_exp_dirs → mkdir_and_rename`
  then **renames the finished `experiments/train_HI_Diff_<TAG>_S{1,2}` to
  `*_archived_<timestamp>` and retrains from scratch** — about 15 GPU-hours
  thrown away and as many spent again. Use `test_only.sh` when only the publish
  step is missing. The same archive-on-rerun behaviour applies to the results
  directory.
* **The test config pins `net_*_latest.pth` deliberately** — iteration 25,000,
  the last checkpoint, not the best-validation one. Every other baseline in this
  table publishes its last checkpoint; repointing this at a best-val file would
  give HI-Diff an asymmetric advantage.
* **Read this row's numbers together.** HI-Diff wins PSNR and SSIM on both
  datasets while sitting near-worst on every perceptual metric — the
  regression-to-the-mean signature of a model optimised for pixel fidelity. If
  you print PSNR/SSIM without the perceptual columns, this row will look like the
  best method in the table.
* **torch ≥ 2.6 will not load its checkpoints** without the two
  `weights_only=False` patches.
* **Licence:** Apache-2.0 (`LICENSE`, "Copyright 2023 HI-Diff Authors"). Ship the
  licence, keep the notices, and **state that files were modified** — which the
  patch table above does. HI-Diff itself builds on BasicSR, Restormer and DiffIR;
  attribute those too.
