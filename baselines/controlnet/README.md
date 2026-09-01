# ControlNet (ICCV 2023)

A trainable copy of the diffusion UNet's encoder, connected to the frozen base
model through zero-initialised convolutions, so a spatial condition can steer
generation without touching the base weights. Here the condition is the SAR
image and the base model is Stable Diffusion 2.1-base.

* **Method:** Zhang et al., ICCV 2023.
* **Code we ran:** `examples/controlnet/train_controlnet.py` from
  <https://github.com/huggingface/diffusers> (Apache-2.0, Copyright 2025 The
  HuggingFace Inc. team), vendored as four loose scripts rather than as a repo
  checkout. **No code from `lllyasviel/ControlNet` was used.**
* **Initialisation:** `ControlNetModel.from_unet(<SD 2.1-base UNet>)`.
* **Base model:** `Manojb/stable-diffusion-2-1-base` — a community mirror of
  `stabilityai/stable-diffusion-2-1-base`, which is no longer available on the
  Hub. Every one of our SD2.1-derived rows loads that mirror. Substituting a
  different mirror should work but we have not verified it.

## Patches we applied

**None.** `train_controlnet.py` is the upstream example script, unmodified; the
only other file we wrote is the inference driver
[`infer_controlnet.py`](infer_controlnet.py) in this directory, which is the
exact script that produced the table's images.

## Budget, in generator updates

**50,000 updates on both datasets** — the same band as the BBDM, cBBDM and
Conditional Diffusion rows. Batch 32 @ 256 (QXS-SAROPT), 8 @ 512 (SAR2Opt);
`lr 1e-5`, bf16, `seed 42`.

Note that **only the adapter trains**: the base UNet, VAE and text encoder stay
frozen, so this row's 50,000 updates buy 364 M parameters of new capacity, not
866 M as in the SD2.1-FT and C-DiffSET rows.

## Data preparation

`$DATA_ROOT/<ds>_controlnet` is a HuggingFace **imagefolder** with a
`metadata.jsonl` carrying three columns:

| Column | Content |
|---|---|
| `image` | the EO target |
| `conditioning_image` | the SAR input |
| `text` | the fixed prompt `"electro-optical image"` |

For SAR2Opt, stage the **512 center crops** into that folder.

## Train + inference

```bash
GPU=0 bash run.sh qxs      # 50,000 updates, batch 32 @ 256
GPU=0 bash run.sh s2o      # 50,000 updates, batch  8 @ 512
```

Inference alone, exactly as the table's images were produced:

```bash
$PY infer_controlnet.py \
    --controlnet_dir $CKPT_ROOT/qxs_controlnet \
    --sar_dir $DATA_ROOT/QXS_AB/testA \
    --out_dir $WORK_DIR/results/qxs_controlnet_eo \
    --steps 50 --batch_size 32 --resolution 256 --seed 42
```

The pipeline is `StableDiffusionControlNetPipeline` on
`Manojb/stable-diffusion-2-1-base` with the scheduler swapped to
`UniPCMultistepScheduler`, 50 steps, `guidance_scale 7.5`, seed 42, and the SAR
PNG opened as RGB and **bilinearly resized** to the target resolution as the
conditioning image.

## The released weights

A complete diffusers `ControlNetModel` folder — `config.json` (1,293 B) plus
`diffusion_pytorch_model.safetensors` (1,456,953,560 B, 340 tensors, all fp32,
364,228,240 parameters). It loads with `ControlNetModel.from_pretrained(dir)`
as-is; we verified that `from_config(cfg).state_dict()` has exactly the same 340
keys, with none missing and none unexpected.

```python
import torch
from PIL import Image
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)

controlnet = ControlNetModel.from_pretrained(<dir>, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Manojb/stable-diffusion-2-1-base", controlnet=controlnet,
    torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

sar = Image.open("sar.png").convert("RGB").resize((256, 256), Image.BILINEAR)
img = pipe("electro-optical image", image=sar, height=256, width=256,
           num_inference_steps=50, guidance_scale=7.5,
           generator=torch.Generator("cuda").manual_seed(42)).images[0]
```

**The adapter cannot generate alone** — it is the encoder half plus the
zero-convolutions, with no `up_blocks` and no `conv_out`. It only means anything
paired with the SD 2.1-base pipeline above.

## Traps

* **The base repository matters.** `stabilityai/stable-diffusion-2-1-base` is
  gone from the Hub; we trained against the `Manojb` mirror and the snippets
  above load it. Do not silently substitute a different SD 2.1 build and expect
  our numbers.
* **The conditioning image is bilinearly resized to the working resolution**, at
  train time and at inference. For SAR2Opt that means the 512 center crop is what
  is resized, not the 600 px tile.
* **Only the adapter trains.** Comparing this row's 50,000 updates against a full
  fine-tune's 40,000 is comparing different amounts of trainable capacity; say so
  if you rank on training cost.
* **Licence:** Apache-2.0 for the training code (the diffusers example's own
  header). The **weights** are a Stable Diffusion 2.1 derivative and are
  distributed under **CreativeML Open RAIL++-M** — the Attachment A use
  restrictions travel to you and to anyone you pass them to.
