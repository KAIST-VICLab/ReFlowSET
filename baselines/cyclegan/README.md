# CycleGAN (ICCV 2017)

Unpaired image-to-image translation with two generators and a cycle-consistency
loss. Here `G_A` maps SAR → EO and `G_B` maps EO → SAR; only `G_A` produces the
scored output.

* **Upstream:** <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
* **Commit vendored:** `2a7afba2895d52556dd5dfe07e8555ef657ced6f` (2025-08-06)
* **Generator:** `resnet_9blocks`, `norm=instance`, `ngf 64`, 3 → 3 channels.

## Patches we applied

**None to anything CycleGAN uses.** The single modification in this repository is
a one-line import fix in `datasets/combine_A_and_B.py`, a helper that only the
pix2pix cells need (see [`../pix2pix/README.md`](../pix2pix/README.md)).
CycleGAN reads the unaligned `trainA`/`trainB` directories directly, so a clean
clone at the pinned commit is exactly what we ran.

## Budget, in generator updates

| Dataset | Batch | Epochs | it/epoch | **Updates** |
|---|---|---|---|---|
| QXS-SAROPT | 8 | 25 + 25 | `ceil(16001/8)` = 2001 | **100,050** |
| SAR2Opt | 4 | 100 + 100 | `ceil(1450/4)` = 363 | **72,600** |

The upstream default of 100 + 100 epochs is tuned for ~1–2 k-image sets. On
16,001 images that would be 400,200 updates and days of wall clock, so
QXS-SAROPT runs 25 + 25 for a budget comparable to the other GAN rows. We also
trained a longer budget-control twin on each dataset (`cycleganlong`, 400,000
and 160,809 updates) to check whether the short budget is what produces the
behaviour below. It is not the whole story: the SAR2Opt long twin still collapses
(ratio 0.767 against the short twin's 0.795), while the QXS-SAROPT long twin does
not (1.127) and is still nowhere near the target. Those twins are not table
rows, and this release does not publish their weights — it ships exactly the 15
methods of the main table on both datasets. The launch script below trains them
if you want them.

## Train + test

```bash
GPU=0 bash run_qxs-saropt.sh      # 100,050 updates
GPU=0 bash run_sar2opt.sh         #  72,600 updates
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/pytorch-CycleGAN-and-pix2pix
$PY test.py --dataroot $DATA_ROOT/QXS_AB \
    --name qxs_cyclegan --model cycle_gan \
    --load_size 256 --crop_size 256 --num_test 99999 \
    --checkpoints_dir $CKPT_ROOT/gan_ckpts --results_dir $WORK_DIR/results/
```

`--model cycle_gan` loads **both** generators and writes both `*_fake_B.png` and
`*_fake_A.png`. **`*_fake_B.png` is the scored image.** For a SAR-only run,
upstream also supports `--model test --dataset_mode single --no_dropout
--model_suffix _A`, which loads `latest_net_G_A.pth` alone.

## The released weights

| File | Module | Needed at inference? |
|---|---|---|
| `net_G_A.pth` (45.5 MB) | `resnet_9blocks` generator, **SAR → EO** | **yes — this is the row** |
| `net_G_B.pth` (45.5 MB) | the EO → SAR cycle generator | no |

`net_G_A` is the SAR → EO direction because
`models/cycle_gan_model.py:116` is `self.fake_B = self.netG_A(self.real_A)` and
`data/unaligned_dataset.py:29-30` binds `A_paths → trainA`, which is the SAR
side. Load it exactly as in [`../pix2pix/README.md`](../pix2pix/README.md) but
with `define_G(3, 3, 64, 'resnet_9blocks', 'instance', ...)`.

## Traps

* **‡ This row audits as IDENTITY COLLAPSE on both datasets.** The output is
  measurably closer to the SAR **input** than to the EO **target**:

  | Dataset | mean\|gen − GT\| | mean\|gen − SAR\| | ratio | verdict |
  |---|---|---|---|---|
  | QXS-SAROPT | 47.548 | 40.274 | **0.847** | collapsed |
  | SAR2Opt | 50.812 | 40.415 | **0.795** | collapsed |

  (Ratio = mean\|gen − SAR\| / mean\|gen − GT\|; below 1.0 means the model is
  reproducing its input. For reference, our own method scores 2.47 and 1.91 on
  the same test items.) We report the row unchanged and marked, because what a
  reader is owed is what the released implementation does at its own published
  protocol — not a substitution. But do not read its FID or LPIPS as translation
  quality. Training 2.2x longer does not fix it on SAR2Opt (the long twin's ratio
  is 0.767); on QXS-SAROPT the long twin clears the threshold at 1.127 but is
  still far from the target, so this is not a pure budget artefact either.
* **Compare budgets in generator updates, never epochs.** CycleGAN is where this
  rule was learned: at our batch sizes, "100 + 100 epochs" is 72,600 updates on
  the 1,450-image set and would be 400,200 on the 16,001-image set — and
  comparing epochs produced two successive wrong findings about why CycleGAN
  collapses.
* **`fake_A` is not your output.** The test pass writes both directions.
* **At test time the SAR2Opt pass must run on the pre-made center-512 crops.**
  Pointed at the native 600 px tiles, the loader takes a random 512 window and
  scores it against the evaluator's center crop.
* **Licence:** the multi-part BSD notice described in
  [`../pix2pix/README.md`](../pix2pix/README.md).
