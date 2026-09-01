# SPADE (CVPR 2019)

Spatially-adaptive normalisation for semantic image synthesis. SPADE is *not* an
image-to-image model out of the box: upstream one-hot-encodes its "label" input.
We run it with `--label_nc 0`, feeding the SAR image where the segmentation map
would go — which upstream treats as an error condition until it is patched.

* **Upstream:** <https://github.com/NVlabs/SPADE>
* **Commit vendored:** `fecacc920c1367a038995c45a39c15f6521ca64f` (2021-09-02)
* **Generator:** `netG=spade`, `ngf 64`, `norm_G=spectralspadesyncbatch3x3`,
  `num_upsampling_layers=normal`, and — because of the patches below —
  **`semantic_nc = 3`**.

## Patches we applied

Five. **Four of them are a functional adaptation, not a compatibility fix.**

| File:line | Change | Why |
|---|---|---|
| `data/pix2pix_dataset.py:62-70` | when `opt.label_nc == 0`, load the "label" through the *image* transform (`.convert('RGB')`, normalised) instead of `Image.NEAREST` + `×255` + the `==255 → label_nc` remap | the SAR PNG is a real image, not a segmentation map |
| `models/pix2pix_model.py:109-122` | new early-return branch in `preprocess_input()`: when `label_nc == 0`, skip `.long()` and the `scatter_` one-hot encoding and pass the float image straight through as `input_semantics` (still concatenating the instance-edge map when `--no_instance` is off) | one-hot encoding a normalised float image is undefined |
| `options/base_options.py:162-164` | when `label_nc == 0`, force `opt.semantic_nc = 3 + (0 if no_instance else 1)` | upstream computes `semantic_nc = label_nc + …` = 0 or 1 and builds a generator with a **1-channel** input |
| `models/networks/discriminator.py:104` | `input_nc = (opt.label_nc if opt.label_nc > 0 else 3) + opt.output_nc` | same problem, discriminator side |
| `util/visualizer.py:12` and `:137` | guard `import scipy.misc`; give `tensor2label` `n_label=0` when `label_nc == 0` so it falls through to `tensor2im` | scipy ≥ 1.12 removed `scipy.misc`; and colour-mapping a real image as a label map is meaningless |

**Consequence for anyone using the released weights:** `net_G.pth` was built with
`semantic_nc = 3` (we pass `--no_instance`), which **stock NVlabs/SPADE will
never construct**. The four functional patches must travel with the checkpoint. A
clean clone plus this checkpoint does not load.

## Budget, in generator updates

| Dataset | Batch | Epochs (`niter` + `niter_decay`) | it/epoch | **Updates** |
|---|---|---|---|---|
| QXS-SAROPT | 16 | 60 + 60 | `floor(16001/16)` = 1000 | **120,000** |
| SAR2Opt | 8 | 100 + 100 | `floor(1450/8)` = 181 | **36,200** |

`data/__init__.py:52` passes `drop_last=opt.isTrain`, so iterations per epoch
round **down** — the same convention as pix2pixHD and the opposite of the
junyanz family.

## Train + test

```bash
GPU=0 bash run_qxs-saropt.sh      # 120,000 updates
GPU=0 bash run_sar2opt.sh         #  36,200 updates
```

Inference alone, as we ran it:

```bash
cd $BASELINES_ROOT/SPADE
$PY test.py --name qxs_spade --dataset_mode custom \
    --label_dir $DATA_ROOT/QXS_AB/testA --image_dir $DATA_ROOT/QXS_AB/testB \
    --label_nc 0 --no_instance --no_pairing_check \
    --preprocess_mode resize_and_crop --load_size 256 --crop_size 256 \
    --aspect_ratio 1 --batchSize 8 --nThreads 8 \
    --checkpoints_dir $CKPT_ROOT/gan_ckpts --results_dir $WORK_DIR/results/_raw_spade \
    --which_epoch latest
```

It loads `<checkpoints_dir>/<name>/<which_epoch>_net_G.pth` and writes
`<results_dir>/<name>/test_latest/images/synthesized_image/<stem>.png`.
SAR2Opt uses `--preprocess_mode crop --load_size 600 --crop_size 512` at train
time and 512/512 against the pre-cropped test set.

`--image_dir` is required even at test time — the loader is paired — so a
SAR-only run needs a dummy EO directory or a direct generator call.

## The released weights

`net_G.pth` (368.8 MB) — the SPADE generator, `semantic_nc = 3`.

```python
# with the PATCHED repo on sys.path
from models.networks.generator import SPADEGenerator
# opt needs: semantic_nc=3, label_nc=0, no_instance=True, ngf=64,
#            num_upsampling_layers='normal', norm_G='spectralspadesyncbatch3x3',
#            crop_size=256, aspect_ratio=1, use_vae=False
G = SPADEGenerator(opt)
G.load_state_dict(torch.load('net_G.pth', map_location='cpu')); G.eval()
y = G(seg=x, z=None)          # x = the normalised SAR image
```

## Traps

* **Stock SPADE cannot load this checkpoint.** See the patch table: without them
  the generator is built with a 1-channel input.
* **The SAR2Opt test pass must run on the pre-made center-512 crops.** SPADE's
  `get_params()` picks `crop_pos` with `random.randint` even when `isTrain` is
  false, so a `crop` test pass over native 600 px tiles scores a *random* 512
  window against the evaluator's center crop.
* **A test pass can exit 0 and write nothing.** The scripts count what *this*
  run produced (before moving the files out), not what is sitting in the output
  directory, so a stale partial dump cannot satisfy the guard.
* **SPADE also dumps a copy of the SAR input** as `input_label/`; the scripts
  delete it. Keeping it triples the file count for no benefit.
* **Compare budgets in generator updates, never epochs** — and note this family
  floor-rounds.
* **Licence: CC BY-NC-SA 4.0 — non-commercial AND share-alike.** `LICENSE.md` is
  the verbatim Creative Commons legal code; the copyright line lives in the
  README ("Copyright (C) 2019 NVIDIA Corporation… released for academic research
  use only. For commercial use or business inquiries, contact
  researchinquiries@nvidia.com"). ShareAlike means a trained SPADE checkpoint, as
  Adapted Material, **must itself be distributed under CC BY-NC-SA 4.0** — it
  cannot sit under a permissive repository-wide weights licence. Our SPADE
  checkpoint is published on those terms and no others.
