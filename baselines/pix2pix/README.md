# pix2pix (CVPR 2017)

Conditional GAN for paired image-to-image translation: a U-Net generator with an
L1 + adversarial objective on aligned image pairs. Here the pair is
(SAR, EO) and the direction is SAR → EO.

* **Upstream:** <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
* **Commit vendored:** `2a7afba2895d52556dd5dfe07e8555ef657ced6f` (2025-08-06)
* **Generator:** `unet_256`, `norm=batch`, `ngf 64`, 3 → 3 channels.

## Patches we applied

**One**, and it is not in the model code.

| File:line | Change | Why |
|---|---|---|
| `datasets/combine_A_and_B.py:5` | added `from pathlib import Path` | The script already calls `Path(args.fold_A)` at lines 28-30 but never imports it, so **upstream HEAD is broken for this script**. It is what builds the aligned `A|B` tiles both pix2pix cells train on. |

The model code, the training loop and the generator are **stock**. Nothing else
in this repository was touched, so a clean clone at the pinned commit plus that
one-line fix reproduces our setup exactly.

## Budget, in generator updates

| Dataset | Batch | Epochs (`n_epochs` + `n_epochs_decay`) | it/epoch | **Updates** |
|---|---|---|---|---|
| QXS-SAROPT | 16 | 60 + 60 | `ceil(16001/16)` = 1001 | **120,120** |
| SAR2Opt | 8 | 100 + 100 | `ceil(1450/8)` = 182 | **36,400** |

The junyanz loader passes no `drop_last`, so iterations per epoch **round up**.
The QXS-SAROPT figure is this project's paired-GAN reference budget: pix2pixHD
and SPADE were matched to it.

## Data preparation

pix2pix wants a single image per sample with A and B side by side. Build it with
the upstream helper (after applying the patch above):

```bash
$PY $BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py \
    --fold_A $DATA_ROOT/QXS_AB/<split>A \
    --fold_B $DATA_ROOT/QXS_AB/<split>B \
    --fold_AB $DATA_ROOT/QXS_AB_combined
```

For SAR2Opt build `S2O_AB_combined` from the native 600 px tiles and
`S2O_AB_combined_test512` from the center-512 crops — training samples random
512 windows out of 600, inference must see the frozen center crop.

## Train + test

```bash
GPU=0 bash run_qxs-saropt.sh      # 120,120 updates
GPU=0 bash run_sar2opt.sh         #  36,400 updates
```

Both scripts run training and then the full test pass. Inference alone, as we
ran it:

```bash
cd $BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
$PY test.py --dataroot $DATA_ROOT/QXS_AB_combined \
    --name qxs_pix2pix --model pix2pix --direction AtoB \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir $CKPT_ROOT/gan_ckpts --results_dir $WORK_DIR/results/
```

`load_networks(opt.epoch)` reads `<checkpoints_dir>/<name>/<epoch>_net_G.pth`,
and `epoch` defaults to `latest`. Output lands at
`<results_dir>/<name>/test_latest/images/<stem>_fake_B.png` — that is the
scored image.

## The released weights

`net_G.pth` (217.7 MB) — the whole row, one file. It is the `unet_256`
generator, `norm=batch`, `ngf 64`, 3 → 3, mapping **SAR → EO** (SAR is the left
half of the `A|B` tile, i.e. `A`, and we train `--direction AtoB`).

Standalone use, without the repository's option machinery:

```python
import torch, numpy as np
from PIL import Image
from models.networks import define_G           # from the junyanz repo

G = define_G(3, 3, 64, 'unet_256', 'batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[])
sd = torch.load('net_G.pth', map_location='cpu')
if hasattr(sd, '_metadata'):
    del sd._metadata                           # upstream strips this too
G.load_state_dict(sd); G.eval()

x = Image.open('sar.png').convert('RGB').resize((256, 256), Image.BICUBIC)
x = torch.from_numpy(np.asarray(x, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
with torch.no_grad():
    y = G(x)
Image.fromarray((((y[0].permute(1, 2, 0).numpy() + 1) * 127.5)
                 .clip(0, 255).astype('uint8'))).save('eo.png')
```

`define_G` is called with `gpu_ids=[]` so no `DataParallel` prefix is expected;
our checkpoints were saved from a single-GPU run and have bare keys.

## Traps

* **Compare budgets in generator updates, never epochs.** "100 + 100 epochs"
  means 36,400 updates on SAR2Opt and would mean 200,200 on QXS-SAROPT. Two
  wrong conclusions in this project came from treating the epoch count as the
  budget.
* **The upstream `combine_A_and_B.py` does not run** at the pinned commit
  (missing `Path` import). If your combined tree is empty, that is why.
* **`--direction` decides which half is the input.** With SAR on the left,
  `AtoB` is SAR → EO. Getting it backwards produces a beautifully converged model
  that predicts SAR from EO and scores like noise.
* **Both `--load_size` and `--crop_size` matter at test time.** For SAR2Opt the
  test pass must run at 512 against the pre-made center crops; pointing it at the
  native 600 px tiles makes the loader take a *random* 512 window and score it
  against the evaluator's center crop of the ground truth.
* **Licence:** the upstream `LICENSE` is a multi-part BSD notice (CycleGAN and
  pix2pix two-clause sections, plus a three-clause DCGAN section from Facebook,
  Inc.). Ship the whole file rather than naming one SPDX identifier, and do not
  use the DCGAN copyright holder's name to promote anything.
