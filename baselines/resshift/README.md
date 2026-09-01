# ResShift (NeurIPS 2023)

Efficient diffusion for image restoration by *residual shifting*: instead of
diffusing towards Gaussian noise, the Markov chain shifts between the degraded
and the target image in a VQ latent space, so few steps suffice. Used here as a
same-size (`sf=1`) SAR → EO mapper through the repository's paired route.

* **Upstream:** <https://github.com/zsyOAOA/ResShift>
* **Commit vendored:** `bb03b7d21614cace01787e097c8a6ab6b945227d` (2026-07-08)
* **Network:** `models.unet.UNetModelSwin`, latent-space, on a frozen vq-f4
  autoencoder.

## Patches we applied

Four in the repository, plus one file we added.

| File:line | Change | Why |
|---|---|---|
| `ldm/modules/attention.py:16` | `XFORMERS_IS_AVAILBLE = True` → `False` inside the successful-import branch | on recent large-memory GPUs the installed xformers has no usable kernel |
| `ldm/modules/diffusionmodules/model.py:16` | same | same, plus "fp32 unsupported, attn head dim 512 > 256" |
| `models/unet.py:25` | same | same |
| `trainer.py:389-392` | `torch.nan_to_num(im_tensor.float(), nan=0.0, posinf=1.0, neginf=-1.0)` before `make_grid`, then `.clamp(0, 1)` after | fp16 diffused latents decoded through the fp32 VQGAN produce NaN/Inf at early iterations and **crashed training inside the image logger**. Logging only — the loss path is untouched |

(The `nan_to_num` on the LPIPS loss at `trainer.py:1018` is **upstream**, not
ours; our diff adds exactly four lines, at 389-392.)

**Added:** the two configs in [`configs/`](configs/) **and
[`inference_qxs.py`](inference_qxs.py)** — see below.

## Budget, in generator updates

**50,000 iterations on both datasets**, batch 16 with microbatch 8 (two
accumulation micro-steps per optimizer update). QXS-SAROPT trains on native 256
px images; SAR2Opt trains on random 256 crops of its 600 px tiles.

## External dependency you must obtain

The config loads a frozen CompVis **vq-f4** autoencoder at
`${CKPT_ROOT}/autoencoder_vq_f4.pth` (221,364,711 bytes). It is **not** shipped
with the released weights and the model is unusable without it. Source:
<https://github.com/CompVis/latent-diffusion> (MIT).

## Train + inference

```bash
GPU=0 bash run.sh qxs      # 50,000 iterations
GPU=0 bash run.sh s2o
# resume after a crash (there is no auto-resume):
RESUME=$WORK_DIR/resshift_qxs/<run>/ckpts/model_XXXX.pth GPU=0 bash run.sh qxs
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/ResShift
$PY inference_qxs.py --cfg_path configs/qxs_sar2eo_256.yaml \
    --ckpt <.../ema_ckpts/ema_model_50000.pth> \
    --in_dir $DATA_ROOT/QXS_AB/testA \
    --out_dir $WORK_DIR/results/qxs_resshift_eo --bs 16
```

`inference_qxs.py` sets `configs.model.ckpt_path` from `--ckpt` and builds
`sampler.ResShiftSampler` with `sf=1`, `chop_size = chop_stride = lq_size = 256`,
`use_amp=True`, `seed=12345`. At 256 px nothing is chopped; at 512 (SAR2Opt,
`--bs 8`) each input is processed as four clean 256 tiles. Output is
`<out_dir>/<stem>.png` with the input stems.

## The released weights

`ema_model.pth` (478.4 MB) — the **EMA** weights of `UNetModelSwin` at iteration
50,000. Latent-space: the external vq-f4 autoencoder above is required in
addition.

## Traps

* **There is no auto-resume.** A crash at hour 7 needs a manual
  `RESUME=<save_dir>/<run>/ckpts/model_XXXX.pth` relaunch. Nothing in the
  repository will notice a half-finished run for you.
* **`inference_qxs.py` is ours, and it is the only working inference path for
  these weights.** Upstream's `inference_resshift.py` is a super-resolution CLI
  keyed to the authors' released tasks and does not accept this config. Copy our
  file into the ResShift checkout — it imports the repo's `sampler` module.
* **The vq-f4 autoencoder is not in the release.** The checkpoint alone will not
  run.
* **The config placeholders are not OmegaConf interpolations.** These YAMLs use
  real OmegaConf interpolation (`${data.train.params.transform_kwargs.mean}`),
  so our `${DATA_ROOT}` / `${CKPT_ROOT}` markers must be expanded — for example
  with `envsubst`, which leaves the dotted interpolations alone because they are
  not valid shell variable names — before OmegaConf sees the file.
* **xformers must be disabled** on hardware where its kernels do not apply; the
  three one-line patches above are how, and without them the failure is a kernel
  error deep in attention.
* **Licence: S-Lab License 1.0 — non-commercial only.** `LICENSE`, "Copyright
  2022 S-Lab": redistribution and use **for non-commercial purposes** are
  permitted with the notice reproduced; commercial use requires contacting the
  contributors. There is also a no-endorsement clause — neither S-Lab's name nor
  its contributors' names may be used to promote this release. Our ResShift
  checkpoint is published on those terms.
