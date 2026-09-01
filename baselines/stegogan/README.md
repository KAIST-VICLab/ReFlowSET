# StegoGAN (CVPR 2024)

Unpaired translation for *non-bijective* domain pairs: a CycleGAN-derived model
with a mismatch mask that identifies content present in one domain and absent
from the other, so the generator is not forced to hallucinate it.

* **Upstream:** <https://github.com/sian-wusidi/StegoGAN> (the official CVPR 2024 code)
* **Commit vendored:** `cad61997c0f82793444f60f81298142b80cdf3c1` (2024-11-07)
* **Generators:** `net_G_A` = `resnet_9blocks_maskv1` (SAR → EO),
  `net_G_B` = `resnet_9blocks_maskv3` (EO → SAR, returns a 3-tuple including the
  mismatch mask).

## Patches we applied

**None.** `git diff -- '*.py'` over the vendored tree is empty and there are no
untracked `.py` or config additions; the row is stock upstream at `cad6199`.
Every StegoGAN-specific correction this project made was made **outside** the
repository — in the launcher flags and in our evaluator's directory selection
(see Traps). That matters for reproduction: clone the pinned commit and change
nothing, then get the two things below right.

## Budget, in generator updates

| Dataset | Batch | Epochs | it/epoch | **Data iterations** | `optimizer_G.step()` calls |
|---|---|---|---|---|---|
| QXS-SAROPT | 4 | 15 + 15 | `ceil(16001/4)` = 4001 | **120,030** | 240,060 |
| SAR2Opt | 2 | 60 + 60 | `ceil(1450/2)` = 725 | **87,000** | 174,000 |

`models/stego_gan_model.py:221-222` calls `self.optimizer_G.step()` **twice**
off a single `backward_G()`. We quote the family in data iterations and state
the 2× explicitly; do not re-derive one cell the other way or the family becomes
internally incomparable. The upstream README uses `batch_size 1` on ~1–2 k-image
sets. A longer SAR2Opt budget-control twin also exists (`stegoganlong`,
160,950 iterations); it is not a table row, and it collapses as well (ratio
0.697 against the short twin's 0.758), so the behaviour below is not simply an
under-training artefact.

## Train + test

```bash
GPU=0 bash run_qxs-saropt.sh      # 120,030 iterations
GPU=0 bash run_sar2opt.sh         #  87,000 iterations
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/StegoGAN
$PY test.py --dataroot $DATA_ROOT/QXS_AB \
    --name qxs_stegogan --model stego_gan --gpu_ids 0 --phase test \
    --no_dropout --resnet_layer 8 --fusionblock \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir $CKPT_ROOT/gan_ckpts --results_dir $WORK_DIR/results/
```

`--resnet_layer 8` and `--fusionblock` are **architecture** flags, not logging
flags: they must match training or the state dict will not load.

Output is one subdirectory per visual — `real_A/`, `fake_B/`, `fake_B_clean/`,
`rec_A/`, `latent_fake_B_mask_upsampled/`, … — under
`<results_dir>/<name>/test_latest/images/`. **`fake_B_clean/` is the scored
row.** Read the next section before assuming otherwise.

## The released weights

| File | Module | Needed at inference? |
|---|---|---|
| `net_G_A.pth` (50.3 MB) | `resnet_9blocks_maskv1`, SAR → EO; takes an **optional** second latent argument | **yes — this is the row** |
| `net_G_B.pth` (52.6 MB) | `resnet_9blocks_maskv3`, EO → SAR | no (training only) |

Deployable path — SAR only, no ground truth:

```python
import torch
from models.networks import define_G           # from the StegoGAN repo

G_A = define_G(3, 3, 64, 'resnet_9blocks_maskv1', 'instance',
               use_dropout=False, init_type='normal', init_gain=0.02,
               gpu_ids=[], resnet_layer=8, fusionblock=True)
G_A.load_state_dict(torch.load('net_G_A.pth', map_location='cpu')); G_A.eval()
with torch.no_grad():
    fake_B_clean = G_A(x)      # ONE argument. Adding a second makes it an oracle.
```

Check `define_G`'s signature in the vendored `models/networks.py` before copying
this — StegoGAN adds `resnet_layer` / `fusionblock` keyword arguments that
upstream junyanz does not have.

## Traps

* **`fake_B` is an ORACLE. `fake_B_clean` is the deployable output.** This is the
  single most expensive trap in this table. `models/stego_gan_model.py:122-135`:

  ```
  129:  self.fake_B_clean = self.netG_A(self.real_A)                       # SAR only
  131:  self.fake_B       = self.netG_A(self.real_A, self.latent_real_B.detach())
  ```

  `latent_real_B` comes from `self.netG_B(self.real_B)` at line 124 — a feature
  map computed **from the ground-truth EO image** and injected into the
  generator trunk. `fake_B` therefore cannot be produced at deployment time. We
  scored the oracle as "StegoGAN" for two days before catching it; on another
  dataset it put a 23.19 dB StegoGAN row in the table where the deployable value
  was 12.74 dB. **Score `fake_B_clean/`.** Publish which one you used. Run an
  input-copy / leakage audit on any new method's first row before believing it.
* **‡ This row audits as IDENTITY COLLAPSE on both datasets.** The output is
  closer to the SAR **input** than to the EO **target**:

  | Dataset | mean\|gen − GT\| | mean\|gen − SAR\| | ratio | verdict |
  |---|---|---|---|---|
  | QXS-SAROPT | 46.908 | 43.087 | **0.919** | collapsed |
  | SAR2Opt | 48.694 | 36.915 | **0.758** | collapsed |

  We report the row unchanged and marked, as with CycleGAN. Do not read its
  perceptual metrics as translation quality.
* **`test.py` rejects `--display_id` and fails silently in a shell chain.** The
  flag is declared only in `options/train_options.py`, not `test_options.py`.
  Passing it to `test.py` aborts with an unrecognised argument — and an `&&`
  chain will happily report the cell "done" with an **empty output directory**.
  This killed several test passes in this project. Pass `--display_id 0` to
  `train.py` only, and always check the output count.
* **`optimizer_G.step()` runs twice per iteration** — see the budget table.
* **Licence: the upstream repository publishes none.** No LICENSE, LICENCE,
  COPYING or NOTICE file at any depth, no licence section in the README, and the
  GitHub API reports no declared licence (checked 2026-08-28). Under default
  copyright that means all rights are reserved by the authors and **no express
  permission to redistribute derived work has been granted** to us or to you. We
  publish this checkpoint anyway so that the comparison is reproducible, and we
  state the position plainly rather than implying a permission that does not
  exist. Assess redistribution for yourself; consider asking the authors.
  StegoGAN is a fork of `junyanz/pytorch-CycleGAN-and-pix2pix` (its README,
  Acknowledgement section); the parts that are unmodified CycleGAN carry that
  project's BSD notice, which does **not** extend to StegoGAN's own
  contributions.
