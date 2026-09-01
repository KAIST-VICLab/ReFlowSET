# ReFlowSET Model Zoo

Every row of the paper's main table, retrained by us under one protocol and
scored by one evaluator on identical test items. Weights are on the Hugging
Face Hub; the code that produced each one is in this repository.

| | |
|---|---|
| Ours | [`JeonghyeokDo/ReFlowSET`](https://huggingface.co/JeonghyeokDo/ReFlowSET) |
| Comparison methods | [`JeonghyeokDo/ReFlowSET`](https://huggingface.co/JeonghyeokDo/ReFlowSET/tree/main/baselines), under `baselines/` |

> **These numbers are not comparable with the ones printed in the source
> papers.** Different splits, resolutions and evaluator conventions make
> cross-paper comparison meaningless. Within this table everything is
> commensurable, which is the entire point of retraining all of it.

## QXS-SAROPT  ·  n=3,999 @ 256x256

| Method | Venue | FID↓ | DISTS↓ | LPIPS↓ | SSIM↑ | PSNR↑ | Updates | Weights | Size | Licence |
|---|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 174.6 | 0.373 | 0.665 | 0.203 | 12.33 | 120,120 | `net_G.pth` | 208 MB | BSD-3-Clause |
| CycleGAN‡ | ICCV'17 | 104.4 | 0.376 | 0.653 | 0.262 | 12.92 | 100,050 | `net_G_A.pth`, `net_G_B.pth` | 87 MB | BSD-3-Clause |
| pix2pixHD | CVPR'18 | 85.7 | 0.298 | 0.573 | 0.358 | 16.13 | 120,000 | `net_G.pth` | 696 MB | BSD-style (NVIDIA) |
| SPADE | CVPR'19 | 90.7 | 0.292 | 0.599 | 0.320 | 14.53 | 120,000 | `net_G.pth` | 352 MB | CC BY-NC-SA 4.0 |
| DDPM (SR3-class) | TPAMI'22 | 43.8 | 0.311 | 0.620 | 0.359 | 14.04 | 250,000 | `gen.pth` | 733 MB | none upstream ⚠ |
| SD2.1 fine-tune only | CVPR'22 | 19.1 | 0.257 | 0.561 | 0.348 | 15.40 | 40,000 of 50,000 † | `diffusion_pytorch_model.safetensors` | 3,303 MB | CreativeML-OpenRAIL-M |
| BBDM | CVPR'23 | 76.6 | 0.270 | 0.568 | 0.352 | 15.34 | 50,000 | `last_model.pth` | 2,020 MB | MIT |
| ControlNet | ICCV'23 | 50.4 | 0.307 | 0.604 | 0.297 | 13.42 | 50,000 | `diffusion_pytorch_model.safetensors`, `config.json` | 1,389 MB | Apache-2.0 code / CreativeML Open RAIL++-M weights |
| HI-Diff | NeurIPS'23 | 324.3 | 0.539 | 0.692 | 0.457 | 17.10 | 25,000+25,000 | `S1_net_g_latest.pth`, `S1_net_le_latest.pth`, `S2_net_d_latest.pth`, `S2_net_g_latest.pth`, `S2_net_le_dm_latest.pth` | 208 MB | Apache-2.0 |
| ResShift | NeurIPS'23 | 140.2 | 0.334 | 0.607 | 0.217 | 14.20 | 50,000 | `ema_model.pth` | 456 MB | S-Lab License 1.0 (non-commercial) |
| StegoGAN‡ | CVPR'24 | 106.8 | 0.384 | 0.658 | 0.254 | 12.96 | 120,030 it | `net_G_A.pth`, `net_G_B.pth` | 98 MB | none upstream ⚠ |
| Conditional Diffusion | GRSL'23 | 88.6 | 0.355 | 0.730 | 0.213 | 11.55 | 50,000 | `ema_final.pt` | 627 MB | none upstream ⚠ |
| cBBDM | GRSL'25 | 50.6 | 0.246 | 0.539 | 0.372 | 16.02 | 50,000 | `last_model.pth` | 2,020 MB | MIT |
| E3Diff | GRSL'24 | 47.8 | 0.278 | 0.530 | 0.302 | 16.44 | 250,000+60,000 | `gen.pth` | 733 MB | none upstream ⚠ |
| C-DiffSET | TCSVT'26 | 19.9 | 0.233 | 0.526 | 0.380 | 16.92 | 40,000 of 50,000 † | `diffusion_pytorch_model.safetensors` | 3,303 MB | MIT |
| ReFlowSET (Ours) | this work | 19.1 | 0.231 | 0.534 | 0.355 | 16.09 | 40,000 | *(see below)* | — | — |

## SAR2Opt  ·  n=627 @ 512x512

| Method | Venue | FID↓ | DISTS↓ | LPIPS↓ | SSIM↑ | PSNR↑ | Updates | Weights | Size | Licence |
|---|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 261.9 | 0.347 | 0.657 | 0.199 | 13.39 | 36,400 | `net_G.pth` | 208 MB | BSD-3-Clause |
| CycleGAN‡ | ICCV'17 | 143.5 | 0.330 | 0.650 | 0.178 | 12.90 | 72,600 | `net_G_A.pth`, `net_G_B.pth` | 87 MB | BSD-3-Clause |
| pix2pixHD | CVPR'18 | 146.3 | 0.283 | 0.567 | 0.268 | 15.95 | 36,200 | `net_G.pth` | 696 MB | BSD-style (NVIDIA) |
| SPADE | CVPR'19 | 142.5 | 0.265 | 0.597 | 0.234 | 14.47 | 36,200 | `net_G.pth` | 352 MB | CC BY-NC-SA 4.0 |
| DDPM (SR3-class) | TPAMI'22 | 122.5 | 0.295 | 0.610 | 0.313 | 13.65 | 250,000 | `gen.pth` | 733 MB | none upstream ⚠ |
| SD2.1 fine-tune only | CVPR'22 | 71.8 | 0.211 | 0.541 | 0.293 | 16.24 | 40,000 of 50,000 † | `diffusion_pytorch_model.safetensors` | 3,303 MB | CreativeML-OpenRAIL-M |
| BBDM | CVPR'23 | 143.1 | 0.290 | 0.590 | 0.276 | 15.29 | 50,137 | `last_model.pth` | 2,020 MB | MIT |
| ControlNet | ICCV'23 | 140.5 | 0.350 | 0.643 | 0.217 | 11.73 | 50,000 | `diffusion_pytorch_model.safetensors`, `config.json` | 1,389 MB | Apache-2.0 code / CreativeML Open RAIL++-M weights |
| HI-Diff | NeurIPS'23 | 319.8 | 0.473 | 0.692 | 0.384 | 17.36 | 25,000+25,000 | `S1_net_g_latest.pth`, `S1_net_le_latest.pth`, `S2_net_d_latest.pth`, `S2_net_g_latest.pth`, `S2_net_le_dm_latest.pth` | 208 MB | Apache-2.0 |
| ResShift | NeurIPS'23 | 141.7 | 0.304 | 0.597 | 0.177 | 14.31 | 50,000 | `ema_model.pth` | 456 MB | S-Lab License 1.0 (non-commercial) |
| StegoGAN‡ | CVPR'24 | 150.1 | 0.347 | 0.655 | 0.158 | 12.47 | 87,000 it | `net_G_A.pth`, `net_G_B.pth` | 98 MB | none upstream ⚠ |
| Conditional Diffusion | GRSL'23 | 211.8 | 0.415 | 0.686 | 0.248 | 12.48 | 50,000 | `ema_final.pt` | 632 MB | none upstream ⚠ |
| cBBDM | GRSL'25 | 222.3 | 0.377 | 0.571 | 0.361 | 17.05 | 50,137 | `last_model.pth` | 2,020 MB | MIT |
| E3Diff | GRSL'24 | 104.7 | 0.232 | 0.529 | 0.249 | 16.09 | 250,000+60,000 | `gen.pth` | 733 MB | none upstream ⚠ |
| C-DiffSET | TCSVT'26 | 78.1 | 0.214 | 0.529 | 0.314 | 16.81 | 40,000 of 50,000 † | `diffusion_pytorch_model.safetensors` | 3,303 MB | MIT |
| ReFlowSET (Ours) | this work | 66.3 | 0.185 | 0.522 | 0.287 | 16.06 | 20,000 | *(see below)* | — | — |

**‡ Identity collapse.** The output is measurably close to a copy of the SAR
input (mean|generated−SAR| / mean|generated−GT| < 1). We report these rows in
place rather than removing or substituting them — what a reader is owed is what
the released implementation does at its published protocol. Ratios:

- QXS-SAROPT: CycleGAN 0.85, StegoGAN 0.92 (ReFlowSET 2.47)
- SAR2Opt: CycleGAN 0.80, StegoGAN 0.76 (ReFlowSET 1.91)

## Provenance

| Method | Upstream | Pinned commit / note |
|---|---|---|
| pix2pix | <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix> | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| CycleGAN | <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix> | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` |
| pix2pixHD | <https://github.com/NVIDIA/pix2pixHD> | `14b3b3c7fff413086e3b58df52096f16b6891172` |
| SPADE | <https://github.com/NVlabs/SPADE> | `fecacc920c1367a038995c45a39c15f6521ca64f` |
| DDPM (SR3-class) | <https://github.com/DeepSARRS/E3Diff> | `38601093ab8f8e4b478144621f20890b100a3b74` |
| SD2.1 fine-tune only | <https://github.com/KAIST-VICLab/C-DiffSET> | `C-DiffSET stage 1 only` |
| BBDM | <https://github.com/xuekt98/BBDM> | `02c3b13c9f9dfab0853e32123100680a0640c4ed` |
| ControlNet | <https://github.com/lllyasviel/ControlNet> | `diffusers train_controlnet.py on SD2.1-base` |
| HI-Diff | <https://github.com/zhengchen1999/HI-Diff> | `b3bfd167997e27f8edd57681cf70e5031a0e35f2` |
| ResShift | <https://github.com/zsyOAOA/ResShift> | `bb03b7d21614cace01787e097c8a6ab6b945227d` |
| StegoGAN | <https://github.com/sian-wusidi/StegoGAN> | `cad61997c0f82793444f60f81298142b80cdf3c1` |
| Conditional Diffusion | <https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation> | `not recorded (vendored tree carries no .git)` |
| cBBDM | <https://github.com/egshkim/ConditionalBBDM-for-VHR-SAR-to-Optical> | `8ce15934f4d4e3f01efe70d11e2d9b9e0859210c` |
| E3Diff | <https://github.com/DeepSARRS/E3Diff> | `38601093ab8f8e4b478144621f20890b100a3b74` |
| C-DiffSET | <https://github.com/KAIST-VICLab/C-DiffSET> | `fixed-step checkpoint-40000` |
| ReFlowSET (Ours) | <https://github.com/KAIST-VICLab/ReFlowSET> | `this repository` |

Venue labels follow the paper's bibliography: Conditional Diffusion is
IEEE GRSL 21:1-5, 2023 (DOI 10.1109/LGRS.2023.3337143), E3Diff is IEEE GRSL
22:1-5, 2024, and cBBDM is IEEE GRSL 22:1-5, 2025.

Conditional Diffusion is the one **reproducibility hole** in this table: the
vendored tree carries no `.git`, so the exact upstream commit we built against
is not recorded. Everything else pins a commit.

**†** C-DiffSET and SD2.1-FT on these two datasets were each a single
**50,000-update** run, and the released checkpoint is the **40,000-update**
snapshot — the one that produced the table row. Stage 2 was initialised from
stage 1's 50,000-update checkpoint. The choice of 40,000 is measured, not
arbitrary: stage-1 validation LPIPS bottoms out near 41,000 updates and then
degrades, and every other baseline here publishes its last checkpoint, so
taking C-DiffSET's `best/` would have been an asymmetry in its favour.
Do not write "40k + 40k" for these two cells.

`Updates` counts **generator/optimizer updates**, never epochs — on a
1,450-image set and a 16,001-image set the same epoch count differs by 2×, and
comparing epochs produced two wrong conclusions in this project before the rule
was adopted. StegoGAN is quoted in data iterations; it calls `optimizer_G.step()`
twice per iteration, so its optimizer-step count is double the figure shown.

## Licences

Each checkpoint inherits the terms of the implementation it was trained with.
**Upstream ships no LICENSE file.** Absent an explicit grant, the upstream code is all-rights-reserved by default; we publish these weights so the comparison is reproducible, and downstream users should form their own view before redistributing or using them commercially.

Rows in that position: **StegoGAN**, **E3Diff**, **DDPM (SR3-class)** (E3Diff's
stage 1), and **Conditional Diffusion**. Rows with restrictive but explicit
terms: **SPADE** (CC BY-NC-SA 4.0, non-commercial + ShareAlike), **ResShift**
(S-Lab 1.0, non-commercial research), **SD2.1-FT** and **ControlNet**
(CreativeML OpenRAIL-M use restrictions propagate to derivatives).

Two rows are named for a paper whose code we did not run, and this is worth
stating plainly: **DDPM (SR3-class)** is E3Diff's stage 1, and **SD2.1
fine-tune only** is C-DiffSET's stage 1 without the confidence channel.

## Note on two re-measured cells

SAR2Opt's C-DiffSET and SD2.1-FT prediction dumps were replaced in place
after the only extended-metric pass that ever scored them. That pass resumes
on item count alone, so every later run skipped them and their DISTS went
stale. Both were re-measured at full n=627; the values above are the
re-measurement. The harness was validated on a cell that is *not* stale
(ReFlowSET's DISTS reproduces to 6e-6).

- `s2o/cdiffset: 0.214130 -> 0.213559`
- `s2o/sd21ft: 0.211540 -> 0.210986`
