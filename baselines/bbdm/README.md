# BBDM (CVPR 2023)

Brownian Bridge Diffusion Model: instead of denoising from Gaussian noise
conditioned on the source, BBDM learns a *bridge* whose endpoints are the source
and target images. Run here in its latent form (LBBDM-f4) on a frozen vq-f4
VQGAN.

* **Upstream:** <https://github.com/xuekt98/BBDM>
* **Commit vendored:** `02c3b13c9f9dfab0853e32123100680a0640c4ed` (2024-08-01)

## Patches we applied

Four, all torch ≥ 2.6 compatibility. No behavioural change.

| File:line | Change | Why |
|---|---|---|
| `model/VQGAN/vqgan.py:64` | `torch.load(..., weights_only=False)` | torch 2.6 flipped the default to `True`; the vq-f4 `model.ckpt` is a pickled Lightning checkpoint |
| `runners/BaseRunner.py:115`, `:131` | same, for the model and optimiser/scheduler states | ditto |
| `runners/DiffusionBasedModelRunners/BBDMRunner.py:35` | same | ditto |
| `runners/DiffusionBasedModelRunners/BBDMRunner.py:63` | deleted `verbose=True,` from `ReduceLROnPlateau(...)` | the keyword was removed in torch 2.x |

We also **added** the two dataset configs in [`configs/`](configs/) and an empty
`datasets/__init__.py`.

## Budget, in generator updates

| Dataset | Batch | it/epoch | Config | **Updates** |
|---|---|---|---|---|
| QXS-SAROPT | 32 @ 256 | 500 | `n_epochs 100`, `n_steps 50000` | **50,000** |
| SAR2Opt | 8 @ 512 | 181 | `n_epochs 280`, capped by `n_steps 50000` | **50,137** |

The SAR2Opt cell overshoots by 137 updates because the runner breaks at the
first epoch boundary past `n_steps` (`runners/BaseRunner.py:387-388`). The two
batches are pixel-matched: `32·256² = 8·512²`.

## External dependency you must obtain

Both configs load the CompVis latent-diffusion **vq-f4 VQGAN** at
`${CKPT_ROOT}/vqgan/vq-f4/model.ckpt` (756,175,527 bytes). The runner constructs
it at build time **even though the same frozen weights are also inside our
`last_model.pth`**, so the file has to exist on disk or the config must be
edited. It is not redistributed here; get it from
<https://github.com/CompVis/latent-diffusion> (MIT).

## Data preparation

`$DATA_ROOT/<ds>_bbdm/{train,val,test}/{A,B}` with `A` = SAR condition and
`B` = EO target, aligned by filename (`dataset_type: custom_aligned`). For
SAR2Opt these are the **512 center crops**, not the native 600 px tiles.

Note the test batch sizes in the configs: **31** for QXS-SAROPT (3999 = 129×31)
and **11** for SAR2Opt (627 = 57×11). The loaders use `drop_last=True`, so a
batch size that does not divide the test-set size silently drops the remainder.

## Train + sample

```bash
GPU=0 bash run.sh qxs      # 50,000 updates
GPU=0 bash run.sh s2o      # 50,137 updates
```

Sampling alone, as we ran it:

```bash
cd $BASELINES_ROOT/BBDM
$PY main.py -c configs/QXS-LBBDM-f4.yaml --gpu_ids 0 -r $WORK_DIR/results/qxs_bbdm \
    --sample_to_eval --resume_model <.../checkpoint/last_model.pth>
```

`--resume_model` sets `config.model.model_load_path`, which
`runners/BaseRunner.py:115` loads with `weights_only=False`; the EMA shadow is
restored at `:124-126` and applied at `:577-578` before sampling.

Output: `<-r>/<dataset_name>/<model_name>/sample_to_eval/{200,condition,ground_truth}/`.
**`200/` (= `sample_step`) holds the generated EO.** `condition/` is the SAR
input and `ground_truth/` the target — scoring the wrong directory produces a
perfect or a nonsensical row.

## The released weights

`last_model.pth` (2118.3 MB) — the full `LatentBrownianBridgeModel` state dict:
the UNet **plus** the frozen vq-f4 VQGAN submodule, **plus** an `['ema']` shadow
and `['step']`/`['epoch']`. Sampling uses the EMA weights. There is no smaller
entry point, and the external vq-f4 checkpoint above is still needed at
construction time.

## Traps

* **Score `sample_to_eval/200/`, not `condition/`.** The sibling directories are
  the SAR input and the ground truth.
* **`drop_last=True` on the test loader.** Pick a test batch size that divides
  the test-set size, or lose the remainder silently.
* **torch ≥ 2.6 will not load the checkpoints** without the four
  `weights_only=False` patches — and the failure is a pickle error far from the
  cause.
* **The vq-f4 VQGAN is constructed from its own path even when the same weights
  are inside the checkpoint.** A missing `model.ckpt` fails at build time.
* **Compare budgets in generator updates, never epochs**: 100 epochs here and
  280 there are the same 50 k updates.
* **Licence:** MIT (`LICENSE`, Copyright (c) 2023 xuekt98). The checkpoint
  additionally requires the CompVis vq-f4 VQGAN (MIT), not redistributed here.
