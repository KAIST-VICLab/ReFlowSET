# ReFlowSET: Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation

Official PyTorch implementation of the paper **"ReFlowSET: Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation"**, arXiv preprint, 2026.

<a href="https://jeonghyeokdo.github.io/">Jeonghyeok Do</a><sup>1</sup>, <a href="https://stellarvision.co.kr/english/">Seungchul Lee</a><sup>2</sup>, <a href="https://www.viclab.kaist.ac.kr/">Munchurl Kim</a><sup>1*</sup>

<sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Stellarvision Inc.

<sup>&dagger;</sup> Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-{{ARXIV_ID}}-red)](https://arxiv.org/abs/{{ARXIV_ID}})
[![Project Page](https://img.shields.io/badge/Project%20Page-ReFlowSET-green)](https://kaist-viclab.github.io/ReFlowSET_site/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ReFlowSET-yellow)](https://huggingface.co/JeonghyeokDo/ReFlowSET)

---

## 📰 News

- **2026-09-01:** Preprint on arXiv; code and pretrained models released. 🎉

---

## Overview

SAR images in all weather and at all hours, but speckle and non-optical geometry
make it hard to read, which motivates SAR-to-EO translation. Recent SAR-to-EO
methods fine-tune a pretrained latent diffusion model, and are therefore bound to
that model's autoencoder — **whose reconstruction quality is a ceiling no
generator built on top of it can exceed.**

**ReFlowSET keeps a frozen high-fidelity autoencoder and replaces the generator.**
A conditional flow-matching transformer is trained *from scratch* directly in the
FLUX.2 latent space:

1. **Shared latent endpoint.** One frozen autoencoder encodes both the SAR input
   and the EO target. A 3×256×256 image becomes a 128×16×16 packed latent through
   an 8× convolutional reduction and a 2×2 space-to-depth pack; the encoder's
   posterior *mean* is used and its log-variance discarded.
2. **A linear flow bridge.** Training defines `z_t = (1−t)·ε + t·z_e` and asks the
   network for the velocity `u* = z_e − ε`, conditioned on the SAR latent `z_s`.
   Sampling starts from `N(0, I)` at `t = 0` and integrates to `t = 1`.
3. **Representation alignment.** A projector on the block-8 activations is matched
   patch-wise, by cosine, to the tokens of a frozen DINOv3 ViT-L/16 applied to the
   clean EO image. Both the projector and the teacher are **discarded at
   inference**.

---

## Architecture in a nutshell

| Component | Setting | Parameters |
|---|---|---|
| DiT (deployed at inference) | hidden 1024, depth 24 = **8 double-stream + 16 single-stream**, 16 heads (head dim 64), SwiGLU MLP ratio 4.0, 2-D RoPE over axes (32, 32), θ=10000 | **509,324,417** |
| REPA projector | 3 × `Linear(1024, 1024)`, tapped at block 8 | 3,148,800 (training only) |
| FLUX.2 autoencoder | frozen, `z_channels` 32 → 128 packed channels, spatial factor 16 | 84,046,115 (frozen) |
| DINOv3 ViT-L/16 teacher | frozen, LVD-1689M, `λ_repa = 0.5`, gate `w(t) = t` | 303,154,176 (training only) |

The SAR condition uses **no separate encoder**: the 1-channel SAR image is
replicated to 3 channels and pushed through the same frozen autoencoder as the EO
image. The resulting `z_s` enters its own projection and forms an EO/SAR tower
pair with joint attention through the eight double-stream blocks, after which the
two token streams are concatenated and run through the sixteen single-stream
blocks.

Classifier-free guidance is trained by zeroing `z_s` on 10 % of rows; all reported
results use scale **1.5**.

---

## Installation

```bash
git clone https://github.com/KAIST-VICLab/ReFlowSET.git
cd ReFlowSET

conda create -n reflowset python=3.12 -y
conda activate reflowset

# Install the torch build matching your CUDA setup first (see https://pytorch.org),
# then the remaining dependencies:
pip install -r requirements.txt
```

### The autoencoder

ReFlowSET's latent space is the **Apache-2.0** autoencoder shipped inside
[`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B).
It is bundled with the released checkpoints, so inference needs no extra download.
To rebuild it yourself:

```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
    vae/diffusion_pytorch_model.safetensors --local-dir klein4b
python scripts/convert_flux2_ae.py \
    --src klein4b/vae/diffusion_pytorch_model.safetensors \
    --key-map assets/flux2_ae_key_map.json \
    --out weights/ae.safetensors
```

### The REPA teacher (training only)

Training additionally needs the DINOv3 ViT-L/16 LVD-1689M backbone, which we do
**not** redistribute. Download it from Meta's official release and accept its
licence: <https://github.com/facebookresearch/dinov3>. Inference never loads it.

---

## Inference

```python
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

# One repository holds both arms, one per subfolder. `DiffusionPipeline` has no
# `subfolder` argument, so fetch the arm and load it as a local pipeline.
ARM = "qxs-saropt"                                   # or "sar2opt"
root = snapshot_download("JeonghyeokDo/ReFlowSET", allow_patterns=[f"{ARM}/*"])
pipe = DiffusionPipeline.from_pretrained(
    f"{root}/{ARM}", custom_pipeline=f"{root}/{ARM}", torch_dtype=torch.float32,
).to("cuda")

sar = Image.open("path/to/sar.png")
eo = pipe(sar, num_inference_steps=50, guidance_scale=1.5,
          generator=torch.Generator("cuda").manual_seed(2024)).images[0]
eo.save("eo.png")
```

`custom_pipeline` points at the same directory because the pipeline, transformer,
autoencoder and scheduler classes ship with the checkpoint rather than living in
`diffusers`.

Or translate a folder:

```bash
python scripts/translate.py \
    --checkpoint JeonghyeokDo/ReFlowSET --subfolder qxs-saropt \
    --sar-dir  /path/to/test/SAR \
    --out-dir  ./results/eo \
    --num-inference-steps 50 --guidance-scale 1.5 --seed 2024
```

**NFE is a reported operating point, not a free knob.** The paper's main table is
`--num-inference-steps 50`. NFE 4 samples **11× faster at 256² and 13× faster at
512²** (163 ms vs 1824 ms per 256² image; 371 ms vs 4807 ms per 512² image, batch 1
on one B200) and trades distribution metrics against pixel metrics. Never mix the
two in one comparison.

---

## Data preparation

| Dataset | Resolution used | Link |
|---|---|---|
| QXS-SAROPT | 256×256 (native) | https://github.com/yaoxu008/QXS-SAROPT |
| SAR2Opt | 512×512, **center-cropped** from 600×600 | https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset |

The frozen split lists are in `splits/`. Only the EO list is stored; the SAR path
is derived from it (`opt_256_oc_0.2` → `sar_256_oc_0.2` for QXS-SAROPT;
`testB` → `testA` for SAR2Opt). Test sets are **3,999** and **627** items.

Images are **never resized** — SAR2Opt's 600 → 512 reduction is a center crop at
offset 44, and training uses a random crop of the same size. SAR is read without a
colour conversion, collapsed to one channel, and scaled by `x / 127.5 − 1`.

---

## Training

```bash
# QXS-SAROPT — 40,000 steps, global batch 64
python scripts/train.py --config configs/qxs_saropt.yaml

# SAR2Opt — 20,000 steps, global batch 32
python scripts/train.py --config configs/sar2opt.yaml
```

| | QXS-SAROPT | SAR2Opt |
|---|---|---|
| Steps × global batch | 40,000 × 64 = **2.56 M samples** | 20,000 × 32 = **640 k samples** |
| Resolution | 256 | 512 |
| Wall clock (1× B200) | 6.14 h | 5.99 h |
| Peak allocated | 55.55 GiB | 101.57 GiB |

Common to both: AdamW, `lr 5e-4` cosine to `1e-6`, 1,000-step warmup, EMA decay
0.9999, bf16, `cond_drop_p = 0.1`, `λ_repa = 0.5`, seed 2024. The two arms differ
only in resolution, batch size and step count — SAR2Opt stops at 20,000 to hold a
comparable sample budget on a 1,450-image training set.

Augmentation is dihedral only — horizontal flip, vertical flip and 90° rotation
(`aug.hflip/vflip/rot90`), applied jointly to the SAR/EO pair. No speckle
augmentation, no speckle-consistency loss and no metadata conditioning are used. **The monitoring split is not held out**; it is drawn from training data
and is used only to watch the loss.

---

## Evaluation

```bash
python scripts/evaluate.py --pred-dir ./results/eo --gt-dir /path/to/test/EO
```

Reports per-image PSNR/SSIM (`torchmetrics`, `data_range=1`), LPIPS (VGG), DISTS
and FID (`pytorch-fid`) against the ground-truth test set, with the ground truth
center-cropped to the generated resolution when they differ.

> **LPIPS has two conventions in this literature and they differ by ~0.05.** We
> report the standard one, feeding `x*2−1` to the LPIPS network. Feeding `[0,1]`
> with `normalize=False` — which several released evaluators do — produces a
> systematically lower number that is not comparable with ours. If you reproduce
> our table, check which one your evaluator uses before concluding anything.

---

## Autoencoder reconstruction ceiling

The Overview's claim — that a method built on a pretrained latent diffusion model
inherits its autoencoder's reconstruction quality as a ceiling — is a
measurement, not an assertion, and `vae_audit/` is that measurement. Each
autoencoder is frozen; an image is encoded to the posterior **mean**, never a
sample, decoded again, and scored against the input. Whatever the round trip
loses, no generator trained in that latent space can recover.

It covers six autoencoders — **SD2.1, SDXL, SD3.0, SD3.5, FLUX.1 and FLUX.2** —
on four SAR/EO benchmarks — **QXS-SAROPT, SAR2Opt, SpaceNet6 and SAR-1M** — with
EO and SAR scored separately, since none of the six has a SAR-native input. No
imagery and no third-party autoencoder weights are redistributed here:
[`vae_audit/README.md`](vae_audit/README.md) gives the download link, licence and
terms for every dataset and every autoencoder, and states the protocol in full.

Its `LPIPS` column uses the **other** convention to the one Evaluation warns
about above — `[0, 1]` with `normalize=False`, which is what the published
ceiling tables were measured with. The audit emits both and labels them; never
quote a ceiling against a translation number without checking which is which.

```bash
python vae_audit/vae_recon_audit.py --dataset qxs-saropt --data-root /path/to/QXS-SAROPT \
    --resolution 256 --vae-root weights/vae_zoo --flux2-weights weights/ae.safetensors \
    --out reports/vae_ceiling
```

---

## Comparison methods

All fifteen prior methods in the main table were **retrained by us** on the same
splits and scored by this evaluator. See [`MODEL_ZOO.md`](MODEL_ZOO.md) for every
cell's metrics, weights, licence and training budget in generator updates, and
[`baselines/`](baselines/) for the launch script, config and patch notes per
method.

Two rows are named for a paper whose code we did not run, and we say so in the
table: *DDPM (SR3-class)* is E3Diff's stage 1, and *SD2.1 fine-tune only* is
C-DiffSET's stage 1 without the confidence channel.

Four cells are marked **‡ identity collapse** — the released implementation, at
its own published protocol, emits something measurably close to a copy of its SAR
input. We report them in place rather than removing or substituting them.

---

## Repository structure

```
ReFlowSET/
├── src/reflowset/
│   ├── pipeline_reflowset.py     # ReFlowSETPipeline (diffusers)
│   ├── transformer_reflowset.py  # ReFlowSETTransformer2DModel
│   ├── autoencoder_flux2.py      # frozen FLUX.2 autoencoder
│   └── scheduler_flow_bridge.py  # the linear flow bridge
├── scripts/
│   ├── train.py                  # training entry point
│   ├── translate.py              # folder-level inference
│   ├── evaluate.py               # the unified evaluator
│   └── convert_flux2_ae.py       # rebuild the Apache-2.0 autoencoder
├── configs/                      # one config per arm
├── splits/                       # frozen train/test lists
├── assets/
│   └── flux2_ae_key_map.json     # value-recovered rename table for the autoencoder
├── baselines/                    # reproduction kit for the 15 comparison methods
├── vae_audit/                    # autoencoder reconstruction-ceiling audit: 6 AEs, 4 datasets
├── MODEL_ZOO.md                  # every cell: metrics, weights, budget, licence
├── LICENSE                       # Apache-2.0, our code
├── LICENSE-WEIGHTS.md            # CC BY-NC 4.0, the checkpoints
├── NOTICE                        # third-party attributions
└── requirements.txt
```

---

## Citation

```bibtex
@article{do2026reflowset,
  title   = {ReFlowSET: Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation},
  author  = {Do, Jeonghyeok and Lee, Seungchul and Kim, Munchurl},
  journal = {arXiv preprint arXiv:{{ARXIV_ID}}},
  year    = {2026}
}
```

Our earlier SAR-to-EO work, which ReFlowSET builds on and compares against:

```bibtex
@article{do2026cdiffset,
  title   = {C-DiffSET: Leveraging Latent Diffusion for SAR-to-EO Image Translation with Confidence-Guided Reliable Object Generation},
  author  = {Do, Jeonghyeok and Lee, Jaehyup and Lee, Seungchul and Kim, Munchurl},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026}
}
```

---

## Acknowledgements

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) under the Sejong Science Fellowship Program (RS-2026-25484549, “Generative AI-based High-Resolution SAR Image Visualization and Analysis Technology for All-Weather Earth Observation”, 50%) and in part by the NRF grant funded by the MSIT (RS-2025-02222525, “Development of AI-based SAR-to-EO image conversion technology”, 50%).

This work builds on the Apache-2.0 autoencoder of
[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
by Black Forest Labs, used frozen and unmodified, and on the Hugging Face
[Diffusers](https://github.com/huggingface/diffusers) library. The
representation-alignment objective follows REPA and uses
[DINOv3](https://github.com/facebookresearch/dinov3) (Meta Platforms) as its
frozen teacher during training only; DINOv3 weights are not redistributed here and
remain subject to the DINOv3 License. We thank the authors of the QXS-SAROPT and
SAR2Opt datasets, and the authors of every method we retrained for comparison.

## License

Our code is released under the [Apache License 2.0](LICENSE), with third-party
attributions in [`NOTICE`](NOTICE). Apache-2.0 rather than MIT because the
transformer primitives and the autoencoder port are adapted from Black Forest
Labs' Apache-2.0 FLUX.2 code, whose notice and modified-file marking obligations
travel with any redistribution.

**The code licence does not reach the weights.** The released checkpoints and each
comparison method carry their own terms — see [`MODEL_ZOO.md`](MODEL_ZOO.md) and
[`LICENSE-WEIGHTS.md`](LICENSE-WEIGHTS.md). The datasets are **not** redistributed
here.
