# pix2pixHD (CVPR 2018)

High-resolution paired translation: a coarse-to-fine generator, multi-scale
discriminators and a feature-matching loss. Run here through its
image-to-image route (`--label_nc 0 --no_instance`), with the SAR image fed as
the "label" input A and the EO image as B.

* **Upstream:** <https://github.com/NVIDIA/pix2pixHD>
* **Commit vendored:** `14b3b3c7fff413086e3b58df52096f16b6891172` (2024-09-30)
* **Generator:** `netG=global` (`n_downsample_global 4`, `n_blocks_global 9`,
  `ngf 64`, `norm=instance`), 3 → 3 channels. `n_local_enhancers` defaults to 1
  but the local enhancer is not used at `netG=global`.

## Patches we applied

Four. Three are Python-3.12 / modern-torchvision compatibility; **the fourth
changes the output file format and is load-bearing for the metrics.**

| File:line | Change | Why |
|---|---|---|
| `data/base_dataset.py:37` | `transforms.Scale` → `transforms.Resize` | `Scale` was removed from torchvision. |
| `train.py:8-9` | `import fractions; fractions.gcd` → `import math; math.gcd`, and `/` → `//` | `fractions.gcd` was removed in Python 3.9; the `lcm()` result must stay integral. |
| `util/visualizer.py:7-10` | `import scipy.misc` wrapped in `try/except ImportError` | removed in scipy ≥ 1.12; only needed for `--tf_log`, which we never pass. |
| `util/visualizer.py:127` | `'%s_%s.jpg'` → `'%s_%s.png'` | **Upstream writes test outputs as JPEG.** Every other row in this table writes PNG, and JPEG re-compression would have biased FID and LPIPS for this row alone. The launcher then harvests `*_synthesized_image.png`. |

A clone without the fourth patch will produce a pix2pixHD row that does not
match ours, and the difference is compression artefacts, not the model.

## Budget, in generator updates

| Dataset | Batch | Epochs (`niter` + `niter_decay`) | it/epoch | **Updates** |
|---|---|---|---|---|
| QXS-SAROPT | 16 | 60 + 60 | `floor(16001/16)` = 1000 | **120,000** |
| SAR2Opt | 8 | 100 + 100 | `floor(1450/8)` = 181 | **36,200** |

pix2pixHD's `data/aligned_dataset.py.__len__` floor-rounds to a multiple of
`batchSize`, so iterations per epoch round **down** — unlike the junyanz loaders,
which round up. Verified against the training log: `total_steps 1,920,000 / 16 =
120,000`.

## Data preparation

pix2pixHD wants `train_A/ train_B/ test_A/ test_B` (underscore spelling); the
scripts create that directory of symlinks for you. On SAR2Opt, `test_A`/`test_B`
must point at the **pre-made center-512 crops**, not the native 600 px tiles —
see the trap below.

## Train + test

```bash
GPU=0 bash run_qxs-saropt.sh      # 120,000 updates
GPU=0 bash run_sar2opt.sh         #  36,200 updates
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/pix2pixHD
$PY test.py --name qxs_p2phd --dataroot $DATA_ROOT/QXS_p2phd \
    --label_nc 0 --no_instance \
    --resize_or_crop resize_and_crop --loadSize 256 --fineSize 256 \
    --checkpoints_dir $CKPT_ROOT/gan_ckpts --results_dir $WORK_DIR/results/_raw_p2phd \
    --phase test --which_epoch latest --how_many 99999
```

It loads `<checkpoints_dir>/<name>/<which_epoch>_net_G.pth` and writes
`<results_dir>/<name>/test_latest/images/<stem>_synthesized_image.png`; the
scripts copy those to `results/<ds>_p2phd_eo/<stem>.png`, which is what the
evaluator reads. SAR2Opt uses `--resize_or_crop crop --loadSize 600 --fineSize
512` at train time and `--loadSize 512 --fineSize 512` against the pre-cropped
test set.

## The released weights

`net_G.pth` (729.8 MB) — the global generator described above; the whole row in
one file.

```python
import torch, torch.nn as nn
from models.networks import GlobalGenerator     # from the pix2pixHD repo

G = GlobalGenerator(3, 3, 64, 4, 9, nn.InstanceNorm2d)
G.load_state_dict(torch.load('net_G.pth', map_location='cpu')); G.eval()
```

Same `[-1, 1]` normalisation as the other GAN rows.

## Traps

* **Upstream writes JPEG.** Without the `util/visualizer.py:127` change, this row
  is scored on JPEG-recompressed images and will not reproduce.
* **The SAR2Opt test pass must run on the pre-made center-512 crops.**
  `get_params()` picks `crop_pos` with `random.randint` **even when `isTrain` is
  false**, so a `crop` test pass over the native 600 px tiles scores a *random*
  512 window against the evaluator's center crop of the ground truth.
* **A test pass can exit 0 and write nothing.** Both scripts count the collected
  PNGs against the test-set size and fail loudly on a mismatch. Do not weaken
  that to `> 0`: a truncated test pass scores just as wrongly as an empty one.
  Also note the scripts delete the previous run's webpage dump first — pix2pixHD
  leaves stale PNGs there, and 5 stale + 3 fresh files would otherwise satisfy a
  count of 8.
* **`--save_epoch_freq` must divide the epoch count** if you want a numbered
  end-of-training checkpoint: pix2pixHD only writes one inside
  `if epoch % save_epoch_freq == 0`. Otherwise you are left with `latest`, which
  is up to one save-interval stale.
* **Compare budgets in generator updates, never epochs** — and remember this
  family floor-rounds while the junyanz family rounds up.
* **Licence:** BSD-style NVIDIA notice (`LICENSE.txt`, Copyright (C) 2019 NVIDIA
  Corporation — Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu) plus a bundled
  pytorch-CycleGAN-and-pix2pix notice. No non-commercial clause. Ship both
  notices with the weights.
