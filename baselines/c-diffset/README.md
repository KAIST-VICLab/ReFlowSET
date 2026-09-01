# C-DiffSET (IEEE TCSVT 2026)

Latent diffusion for SAR-to-EO built directly on a pretrained Stable Diffusion
2.1 backbone: the SAR latent is **concatenated** to the noisy EO latent (the
UNet's `conv_in` is widened from 4 to 8 channels), and a confidence head lets
the loss down-weight regions where the two modalities genuinely disagree.

* **Upstream:** <https://github.com/KAIST-VICLab/C-DiffSET>
* **Commit vendored:** `6fc3802`
* **Base model:** `Manojb/stable-diffusion-2-1-base` — a community mirror of
  `stabilityai/stable-diffusion-2-1-base`, which is no longer available on the
  Hub. The upstream README states this. Every SD2.1-derived row in this table
  loads that mirror.
* **Stage 1 of the same trainer is a separate table row** — see
  [`../sd21-ft/`](../sd21-ft/).

## Patches we applied

**None to the model or the trainer.** We added the four dataset configs — the two
stage-2 configs in [`configs/`](configs/) and the two stage-1 configs in
[`../sd21-ft/configs/`](../sd21-ft/configs/). **All four must be installed into
`$BASELINES_ROOT/C-DiffSET/configs/`** (with their `${...}` placeholders
expanded) before either script runs, because both stages are driven from the
same repository.

## Budget, in generator updates

**The published row is the fixed-step `checkpoint-40000` of each stage — not the
authors' 50 k, and not the repository's validation-PSNR-selected `best/`.**

| Dataset | Batch | it/epoch | **Published checkpoint** | epochs at 40 k |
|---|---|---|---|---|
| QXS-SAROPT | 64 @ 256 | 250 | `checkpoint-40000` | 160 |
| SAR2Opt | 16 @ 512 | 90 | `checkpoint-40000` | 445 |

Two measured reasons, both from our own runs:

* stage-2 validation LPIPS reaches 0.5080 by ~32.5 k steps and only 0.5064 at
  50 k — 0.3 % over the last third of training; it has converged;
* stage-1 validation LPIPS **bottoms out at ~41 k (0.5297) and then degrades to
  0.5348 by 50 k** — the 50 k stage 1 overfits.

And `best/` is selected on validation PSNR while every other baseline in this
table publishes its **last** checkpoint; using it would be an asymmetry in
C-DiffSET's favour. 40 k also sits on the `save_iter = 10000` grid, so the
snapshot exists inside any 50 k run at no extra cost.

**What actually happened on these two datasets, stated exactly.** Both cells
were trained by a **50,000-step** run (the configs' own `num_iter`), and
`checkpoint-40000` is an intermediate snapshot of that run; stage 2 was
initialised from stage 1's **final (50 k)** checkpoint,
`lastest/model.safetensors`, which is what the stage-2 config's
`accelerator_path` names. So the honest description of the released weights is:

> **C-DiffSET** — stage-2 confidence-guided UNet, snapshot at **40,000
> optimizer updates** of a 50,000-update run, initialised from the
> **50,000-update** stage-1 checkpoint.

Do **not** write "40 k + 40 k" for these two cells. `run.sh` reproduces exactly
the flow above; pass `STEPS=40000` and repoint `--accelerator-path` if you want a
strict 40 k + 40 k schedule instead, and say which you did.

A 50 k twin of both stages also exists (`cdiffset50k`, `unet_50k.safetensors`);
it is the authors' protocol and it is **not** the table row. This release does
not publish it — only the 40 k checkpoint behind the table ships. Its numbers differ (for example QXS-SAROPT FID 20.4 versus the
table's 19.9).

Common hyper-parameters, both stages, both datasets: AdamW, `lr 3e-5`, weight
decay 0.01, cosine schedule, 100-step warm-up, `mixed_precision: "no"` (fp32),
seed 2024, `prediction_type: epsilon`, prompt `"electro-optical image"`, DDIM-50
at test time. QXS-SAROPT augments with hflip/vflip/rot90; SAR2Opt takes a random
512 crop of the 600 px images at train time and the centre 512 at test time.

## External dependency

Stage 1 is initialised from `${CKPT_ROOT}/sd21_unet8ch_init.safetensors` — the
SD 2.1-base UNet with `conv_in` duplicated and halved to 8 channels
(`_weight.repeat((1, 2, 1, 1)) * 0.5`). Rebuild it from the base repository with
the repository's own `replace_unet_conv_in`; it is not redistributed here.

## Train + inference

```bash
GPU=0 bash run.sh qxs      # stage 1, stage 2, then both inferences
GPU=0 bash run.sh s2o
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/C-DiffSET
$PY test.py --sar-dir $DATA_ROOT/QXS_AB/testA \
    --output-dir $WORK_DIR/results/qxs_cdiffset_eo/ \
    --checkpoint $WORK_DIR/qxs_cdiffset/checkpoint-40000/model.safetensors \
    --num-inference-steps 50
```

## The released weights

`unet_final.safetensors` (3,463,784,116 B) — a **bare `UNet2DConditionModel`
state dict**: 686 tensors, all fp32, 865,925,125 parameters, no prefix, no EMA,
no accelerate wrapper. `conv_in` is `[320, 8, 3, 3]` and `conv_out` is
`[5, 320, 3, 3]`.

Channel semantics, which a user will otherwise get wrong:

* **latent channels 0–3 are the SAR latent, 4–7 the noisy EO latent** —
  `torch.cat([sar_latents, eo_latents], dim=1)`, SAR first;
* **output channel 4 is a raw variance**, not part of the prediction. Feed only
  `out[:, :4]` to the scheduler. To interpret the variance channel it needs
  `softplus(v + init_var_offset) + min_var`, clamped to `[1e-6, 10]`;
* the VAE encoding uses `latent_dist.mean`, not `.sample()`.

```python
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

BASE = "Manojb/stable-diffusion-2-1-base"
unet = UNet2DConditionModel.from_pretrained(<repo>, subfolder=<cell>,
                                            torch_dtype=torch.float32).cuda().eval()
vae  = AutoencoderKL.from_pretrained(BASE, subfolder="vae").cuda().eval()
tok  = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
txt  = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").cuda().eval()
sch  = DDIMScheduler.from_pretrained(BASE, subfolder="scheduler"); sch.set_timesteps(50, device="cuda")

ids   = tok("electro-optical image", padding="do_not_pad",
            max_length=tok.model_max_length, truncation=True,
            return_tensors="pt").input_ids.cuda()
embed = txt(ids)[0]

with torch.no_grad():                       # sar: [1,3,H,W] in [-1,1]
    sar_lat = vae.encode(sar).latent_dist.mean * vae.config.scaling_factor
    eo_lat  = torch.randn_like(sar_lat)
    for t in sch.timesteps:
        out = unet(torch.cat([sar_lat, eo_lat], 1), t, encoder_hidden_states=embed).sample
        eo_lat = sch.step(out[:, :4], t, eo_lat).prev_sample   # ch 4 = variance, dropped
    eo = (vae.decode(eo_lat / vae.config.scaling_factor).sample * 0.5 + 0.5).clamp(0, 1)
```

The matching `config.json` is the SD 2.1-base UNet config with `in_channels: 8`
and `out_channels: 5`; `sample_size: 64` is inherited and inert here, because
the UNet is driven directly rather than through a pipeline.

## Traps

* **The checkpoint is 8-in / 5-out.** It will not load into a stock SD 2.1
  pipeline, and the fifth output channel is not part of the prediction.
* **Concat order is SAR first.** Reversing it produces noise that looks like a
  training bug somewhere else.
* **`checkpoint-40000`, not `best/`, not 50 k.** See the budget section; quoting
  a `best/` number silently changes the protocol relative to every other row.
* **Compare budgets in generator updates, never epochs.** 160 epochs and 445
  epochs are the same 40,000 updates here.
* **Do not reuse an already-published C-DiffSET blob for this table.** The
  project's earlier public release of this method ships the **50,000**-update
  checkpoint, which is a different cell with different numbers.
* **Licence:** the **code** is MIT (`LICENSE`, Copyright (c) 2026 KAIST VICLab);
  the **weights** are a Stable Diffusion 2.1 derivative and are distributed under
  **CreativeML Open RAIL++-M**, whose Attachment A use restrictions travel to
  every derivative. Cite: Do, Lee, Lee and Kim, *C-DiffSET*, IEEE TCSVT 2026,
  doi:10.1109/TCSVT.2026.3701447.
