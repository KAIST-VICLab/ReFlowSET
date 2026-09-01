# cBBDM (GRSL 2025)

Conditional Brownian Bridge Diffusion Model for very-high-resolution SAR-to-
optical translation: a fork of BBDM in which the SAR image additionally enters
the UNet as an explicit condition (`in_channels: 6`, `condition_key:
SpatialRescaler`) rather than only as the bridge endpoint.

* **Upstream:** <https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical>
* **Commit vendored:** `8ce15934f4d4e3f01efe70d11e2d9b9e0859210c` (2026-04-20)
* Fork of <https://github.com/xuekt98/BBDM> — see [`../bbdm/`](../bbdm/), whose
  structure this row shares.

## Patches we applied

The same four torch ≥ 2.6 compatibility fixes as BBDM
(`model/VQGAN/vqgan.py:64`, `runners/BaseRunner.py:105` and `:131`,
`runners/DiffusionBasedModelRunners/BBDMRunner.py:35`, and the removal of
`verbose=True` from `ReduceLROnPlateau` at `:69`) — **plus one performance patch
that changes no maths but must be documented**:

| File:line | Change | Why |
|---|---|---|
| `runners/BaseRunner.py:350,355,360,374,379,384` and `DiffusionBasedModelRunners/BBDMRunner.py:97` | `num_workers=0` → `num_workers=8` in all six train/val/test DataLoaders | the fork regressed this from upstream BBDM's `8`. Measured: one 32-pair batch takes 6.53 s serial versus 1.10 s across 8 threads — the whole of the 3.9–6 s/iter we observed |

**Both table cells were trained *before* that patch, i.e. with 0 workers** — same
data, same order, same step count, only slower. Their numbers stand.

A caution about our own comment: the in-file justification for the worker patch
asserts "flip is False everywhere", and that is **not true for SAR2Opt**, whose
config sets `flip: True`. The worker-count argument still holds (the DataLoader
worker count does not change which samples the deterministic sampler visits),
but do not repeat the claim as written.

We also **added** the two dataset configs in [`configs/`](configs/).

## Budget, in generator updates

| Dataset | Batch | it/epoch | Config | **Updates** | flip |
|---|---|---|---|---|---|
| QXS-SAROPT | 32 @ 256 | 500 | `n_epochs 100`, `n_steps 50000` | **50,000** | False |
| SAR2Opt | 8 @ 512 | 181 | `n_epochs 280`, capped by `n_steps 50000` | **50,137** | True |

Pixel-matched batches (`32·256² = 8·512²`); the SAR2Opt cell overshoots by 137
updates because the runner breaks at the first epoch boundary past `n_steps`.
512 is also the conditional-BBDM paper's own SAR2Opt training resolution.

## External dependency, data, commands

Identical to [`../bbdm/README.md`](../bbdm/README.md): the CompVis **vq-f4**
VQGAN at `${CKPT_ROOT}/vqgan/vq-f4/model.ckpt` (not redistributed), the
`$DATA_ROOT/<ds>_bbdm/{train,val,test}/{A,B}` tree shared with the BBDM row, and
the same two-job flow.

```bash
GPU=0 bash run.sh qxs      # 50,000 updates
GPU=0 bash run.sh s2o      # 50,137 updates
```

Sampling alone:

```bash
cd $BASELINES_ROOT/cBBDM
$PY main.py -c configs/QXS-cBBDM-f4.yaml --gpu_ids 0 -r $WORK_DIR/results/qxs_cbbdm \
    --sample_to_eval --resume_model <.../cBBDM-f4/checkpoint/last_model.pth>
```

Output: `.../<ds>/cBBDM-f4/sample_to_eval/{200,condition,ground_truth}/`, and
**`200/` is the generated EO**.

## The released weights

`last_model.pth` (2118.3 MB) — the full model state dict as in BBDM, but the
UNet is `in_channels: 6` with `condition_key: SpatialRescaler`, and **the
SpatialRescaler is itself trained** (it appears in
`LatentBrownianBridgeModel.get_parameters()`). Sampling uses the EMA shadow
stored in the same file.

## Traps

* Everything in [`../bbdm/README.md#traps`](../bbdm/README.md) applies: score
  `sample_to_eval/200/`, mind `drop_last=True` on the test loader, the
  `weights_only=False` patches, and the externally-supplied vq-f4 checkpoint.
* **This fork ships `num_workers=0`.** If your run is inexplicably 5× slower than
  BBDM's, that is why.
* **`flip` differs between our two configs** (False on QXS-SAROPT, True on
  SAR2Opt, which is small and where the upstream template enables flips). It is
  a deliberate per-dataset choice, recorded here so it is not mistaken for an
  accident.
* **Licence:** MIT (`LICENSE`, Copyright (c) 2025 egshkim), a fork of
  `xuekt98/BBDM` (MIT, Copyright (c) 2023 xuekt98). Requires the CompVis vq-f4
  VQGAN (MIT), not redistributed here. Note that the upstream repository's real
  name is `ConditionalBBDM-for-VHR-SAR-to-Optical`; some of our own older notes
  shorten it to `cBBDM`, which is not a repository that exists.
