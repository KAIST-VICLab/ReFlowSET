# SD2.1 fine-tune only (the CVPR'22 row)

The "just fine-tune a latent diffusion model" reference point: Stable Diffusion
2.1-base, fine-tuned on SAR → EO with a plain ε-prediction objective and no
confidence head.

**This row is not a generic Stable Diffusion fine-tune written from the LDM
paper. It is C-DiffSET's stage 1**, trained by the same repository and the same
trainer as [`../c-diffset/`](../c-diffset/) with the second-stage machinery
switched off. Say so wherever the number is quoted, and cite C-DiffSET alongside
the latent-diffusion paper.

* **Upstream:** <https://github.com/KAIST-VICLab/C-DiffSET> (stage 1 only)
* **Commit vendored:** `6fc3802`
* **Base model:** `Manojb/stable-diffusion-2-1-base`, a mirror of
  `stabilityai/stable-diffusion-2-1-base`, which is no longer on the Hub.

## What "stage 1" changes relative to stock SD 2.1

* **`conv_in` is widened from 4 to 8 channels** so the SAR latent can be
  concatenated to the noisy EO latent: `_weight.repeat((1, 2, 1, 1)) * 0.5`,
  the pretrained kernel duplicated and halved. Channels 0–3 are the SAR latent,
  4–7 the noisy EO latent.
* **`conv_out` keeps its 4 channels.** `train_stage1.py:34-36` overrides
  `replace_unet_conv_out` to a no-op, which is exactly what makes this row
  "C-DiffSET without the confidence head".
* Everything else — the text encoder, the VAE, the scheduler — is the frozen
  SD 2.1-base stack, driven with the fixed prompt `"electro-optical image"`.

Measured on the released checkpoints, the two halves of `conv_in` have diverged
(`|w[:, :4]|`.mean 0.0142 versus `|w[:, 4:]|`.mean 0.0281), which confirms the
row really trained rather than merely inheriting the duplicated initialisation.

## Patches we applied

**None to the model or the trainer.** We added the two stage-1 configs in
[`configs/`](configs/).

## Budget, in generator updates

Identical in structure to [`../c-diffset/README.md`](../c-diffset/README.md#budget-in-generator-updates)
and governed by the same rule: the published row is the **fixed-step
`checkpoint-40000`**, not the authors' 50 k and not the val-PSNR-selected
`best/`. QXS-SAROPT batch 64 @ 256 (250 it/epoch → 160 epochs at 40 k);
SAR2Opt batch 16 @ 512 (90 it/epoch → 445 epochs). Equal pixels per update.

The measured reason applies with particular force to this row: **stage-1
validation LPIPS bottoms out at ~41 k (0.5297) and then *degrades* to 0.5348 by
50 k.** The authors' 50 k stage 1 overfits; 40 k is both cheaper and better.

As trained here, both cells ran to 50,000 updates and `checkpoint-40000` is a
snapshot of that run — so the accurate card wording is:

> **SD2.1-FT** — stage-1 ε-only UNet, snapshot at **40,000 optimizer updates** of
> a 50,000-update run, initialised from the SD 2.1 UNet with 8-channel `conv_in`.

## External dependency

`${CKPT_ROOT}/sd21_unet8ch_init.safetensors` — the SD 2.1-base UNet with the
duplicated-and-halved 8-channel `conv_in` and the original 4-channel `conv_out`.
Rebuild it from the base repository with the C-DiffSET repository's own
`replace_unet_conv_in`; it is not redistributed here.

## Train + inference

```bash
GPU=0 bash run_stage1.sh qxs      # stage 1 only
GPU=0 bash run_stage1.sh s2o
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/C-DiffSET
$PY test_stage1.py --sar-dir $DATA_ROOT/QXS_AB/testA \
    --output-dir $WORK_DIR/results/qxs_sd21ft_eo/ \
    --checkpoint $WORK_DIR/qxs_sd21ft/checkpoint-40000/model.safetensors \
    --num-inference-steps 50
```

## The released weights

`unet_final.safetensors` (3,463,772,592 B) — a bare `UNet2DConditionModel` state
dict: 686 tensors, all fp32, 865,922,244 parameters, `conv_in` `[320, 8, 3, 3]`,
`conv_out` `[4, 320, 3, 3]`. (It is exactly 2,881 parameters smaller than the
C-DiffSET checkpoint — one `conv_out` output channel: 320·3·3 weights plus one
bias.)

Usage is the snippet in [`../c-diffset/README.md`](../c-diffset/README.md#the-released-weights)
with two changes: load this checkpoint, and pass **all four** output channels to
the scheduler, since there is no variance head:

```python
out = unet(torch.cat([sar_lat, eo_lat], 1), t, encoder_hidden_states=embed).sample
eo_lat = sch.step(out, t, eo_lat).prev_sample     # all 4 channels
```

The matching `config.json` is the SD 2.1-base UNet config with `in_channels: 8`
and `out_channels: 4`.

## Traps

* **The row's name is misleading and the checkpoint proves it.** "SD2.1
  fine-tune only" is 8-in / 4-out C-DiffSET stage 1, whose upstream is the
  C-DiffSET repository — not stabilityai. Anyone who reads the label and tries
  `StableDiffusionPipeline.from_pretrained` will fail on `conv_in`. Card text
  that names Stability as the upstream of this cell is wrong.
* **`checkpoint-40000`, not 50 k, not `best/`** — and here the difference is not
  neutral: 50 k is measurably *worse* on validation LPIPS.
* **Compare budgets in generator updates, never epochs.**
* **The base repository matters**: `stabilityai/stable-diffusion-2-1-base` is
  gone from the Hub; we trained against the `Manojb` mirror.
* **Licence:** the **code** is MIT (C-DiffSET, Copyright (c) 2026 KAIST VICLab);
  the **weights** are an SD 2.1 derivative under **CreativeML Open RAIL++-M**,
  and its Attachment A use restrictions propagate to you and to anyone you pass
  them to.
