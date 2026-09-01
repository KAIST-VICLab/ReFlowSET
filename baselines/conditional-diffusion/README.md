# Conditional Diffusion (IEEE GRSL)

A conditional DDPM for SAR-to-optical translation, built on OpenAI's
guided-diffusion: the SAR image is concatenated to the noisy sample **noise-free**
at every reverse step, and the model predicts ε. That is the paper's claim and
the code does do it (`gaussian_diffusion.py:260-262`, `x_t = th.cat([x,
condition], dim=1)`, and the same at training time, `:772`).

* **Upstream:** <https://github.com/Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation>
* **Commit vendored:** **NOT RECORDED — see the first trap.**
* **Cite:** Bai, Pu and Xu, *Conditional Diffusion for SAR to Optical Image
  Translation*, IEEE GRSL, doi:10.1109/LGRS.2023.3337143. (Our own launcher
  header records the venue as GRSL 2023 while the table prints GRSL'24 — 2023
  early access, 2024 issue. Pick one and use it consistently.)

## Patches we applied

Located by in-file markers and confirmed by diffing the repository's
`guided_diffusion/` against the authors' own `deployment/guided_diffusion/`
copy, so that author-vs-author differences are not mistaken for ours.

| File:line | Change | Why |
|---|---|---|
| `guided_diffusion/dist_util.py:27-30` | `os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank % GPUS_PER_NODE}"` made conditional on the variable being unset | upstream pins rank%8, i.e. physical GPU 0, whatever you asked for |
| `scripts/image_sample_realtime.py:46-48` | `sorted(os.listdir(self.path_sar))` and `sorted(os.listdir(self.path_opt))` | **CORRECTNESS BUG upstream.** The released sampler pairs SAR with EO by *unsorted* `os.listdir` position — i.e. by filesystem order — so every SAR/EO test pair can be mismatched. Any number produced with the unpatched sampler is measured against effectively random ground truth |
| `scripts/image_sample_realtime.py:89-95` | test path taken from a new `--test_dir`; `transforms.Resize((args.image_size, args.image_size))` built at run time | the module-level `transform_opt` hard-codes 256, which is wrong for a 512 dataset |
| `scripts/image_sample_realtime.py:120-130` | noise shape uses `condition.shape[0]` instead of `args.batch_size` | the loader runs with `drop_last=False`, so the last batch is partial and a fixed-size noise tensor mismatches |
| `scripts/image_sample_realtime.py:99,218` | new `--out_dir`, output tree `<out>/{gen_opt,cond_sar,gt_opt}/` | upstream hard-codes `sample_results/<respacing>/` |
| `_stubs/blobfile.py`, `_stubs/mpi4py/{__init__,MPI}.py` | local-filesystem and single-process stand-ins, prepended to `PYTHONPATH` | `blobfile` and `mpi4py` are imported unconditionally by `dist_util.py`; `MPI.COMM_WORLD` here is a world-size-1 fake |

Copies of the stubs are in [`_stubs/`](_stubs/); `run.sh` puts them first on
`PYTHONPATH`.

**Confirmed NOT ours** (differences between the repository's
`guided_diffusion/` and its own `deployment/` copy): the removal of hard-coded
CSV path lists, the gradio progress hooks, and the `script_util.py` default
changes. The `deployment/` tree is the authors' demo variant.

## Budget, in generator updates

**50,000 updates on both datasets** — the authors' own protocol (50 k iterations
at global batch 24) and the same band as the BBDM / cBBDM / ControlNet rows.
Batch 24 @ 256 (QXS-SAROPT) and 6 @ 512 (SAR2Opt): equal pixels per update.
`--lr_anneal_steps 50000` at a constant `lr 1e-4` with linear decay to zero.

**The paper and the code disagree about the schedule.** The paper describes
warm-up plus cosine; the released code has neither. We ran the code, and
`--lr_anneal_steps` is also its only stop mechanism.

## Data preparation

```
$DATA_ROOT/conddiff_<ds>/
    train/{sar,opt}/<integer>.png
    test/{sar,opt}/<integer>.png
    manifest_test.json          {"<integer>": "<ground-truth stem>", ...}
```

**The filenames must be integers.** `guided_diffusion/image_datasets.py:52-53`
sorts with `key=lambda x: int(x[len(dir)+1:].split('.')[0])` and a non-integer
stem raises `ValueError`. That is why the tree is an adapter with integer names
plus a manifest that maps them back; `run.sh` uses the manifest to rename the
outputs to the ground-truth stems at the end. Do not point the launcher at the
raw dataset.

For SAR2Opt the tiles must be **deterministic centre-512 crops** of the 600 px
images: the fork's `center_crop_arr` **resizes** 600 → 512, which would break the
crop-not-resize protocol this table holds fixed for every other method.

## Train + sample

```bash
GPU=0 bash run.sh qxs      # 50,000 updates, batch 24 @ 256
GPU=0 bash run.sh s2o      # 50,000 updates, batch  6 @ 512
```

Sampling alone, as we ran it:

```bash
export PYTHONPATH=$PWD/_stubs:$BASELINES_ROOT/CondDiff
OPENAI_LOGDIR=<logdir>/sample_log $PY $BASELINES_ROOT/CondDiff/scripts/image_sample_realtime.py \
  --model_path <.../ema_0.9999_050000.pt> \
  --test_dir $DATA_ROOT/conddiff_qxs/test \
  --out_dir  <logdir>/sample_out \
  --image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False \
  --diffusion_steps 2000 --noise_schedule linear \
  --timestep_respacing 250 --batch_size 24 --num_samples 3999
```

(SAR2Opt: `--image_size 512 --batch_size 6 --num_samples 627`.) Output is
`<out_dir>/{gen_opt,cond_sar,gt_opt}/<i>.png`, indexed by position.

Minimal standalone:

```python
import torch, sys
sys.path[:0] = ['<kit>/_stubs', '<CondDiff>']
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)
d = model_and_diffusion_defaults()
d.update(image_size=256, num_channels=128, num_res_blocks=3, learn_sigma=False,
         diffusion_steps=2000, noise_schedule='linear', timestep_respacing='250')
model, diffusion = create_model_and_diffusion(**d)
model.load_state_dict(torch.load('ema_final.pt', map_location='cpu'))  # a BARE state dict
model.cuda().eval()
# sar: (1,3,256,256) in [-1,1]
sample = diffusion.p_sample_loop(model, (1, 3, 256, 256), clip_denoised=True,
                                 model_kwargs={}, noise=None, condition=sar)
```

## The released weights

`ema_final.pt` — 657.5 MB (QXS-SAROPT) / 662.5 MB (SAR2Opt). The EMA (decay
0.9999) of the 164.3 M-parameter guided-diffusion UNet at 50,000 updates,
`ch 128`, `num_res_blocks 3`, `learn_sigma False`. It is a **bare** state dict
with no wrapper key. The size difference between the two datasets is the 256 vs
512 positional/attention buffers.

## Traps

* **The upstream commit is not recorded — this is the one reproducibility hole in
  the table.** Our vendored copy has no `.git`, no `.gitmodules`, no version
  file and no upstream URL inside the tree; its README names the paper and links
  the authors' *later* rewrite, not this fork's commit. **The exact upstream
  state we built against cannot be recovered.** Treat the patch table above as
  the specification: clone the upstream repository, re-apply each change, and
  say in your own write-up that the base commit is unknown.
* **DDIM is broken upstream. Do not pass `--use_ddim True`.**
  `p_sample_loop(..., condition=None, ...)` accepts and threads the SAR
  condition; `ddim_sample_loop(...)` has **no `condition` parameter at all**, so
  the call raises `TypeError`. Sampling is respaced DDPM-250, which is also what
  the authors' own `sample.sh` uses.
* **The unpatched sampler pairs SAR and EO by unsorted `os.listdir` order.**
  Without the `sorted()` fix your metrics are computed against effectively random
  ground truth — and they will look plausible, not broken.
* **Integer filenames are mandatory** (see Data preparation).
* **`center_crop_arr` resizes rather than crops** on the 512 dataset.
* **Licence: the upstream repository publishes none.** No LICENSE/COPYING/NOTICE
  file, no licence section in the README, and the GitHub API reports no declared
  licence (checked 2026-08-28). Under default copyright all rights are reserved
  by the authors and **no express permission to redistribute derived work has
  been granted**. We publish this checkpoint anyway so the comparison is
  reproducible, and state the position plainly. The repository is a modified copy
  of <https://github.com/openai/guided-diffusion> (MIT); the unmodified
  guided-diffusion parts carry that MIT licence, which does not extend to the
  authors' modifications — nor, on its own, to ours, which are listed above so
  that anyone can tell what we changed.
