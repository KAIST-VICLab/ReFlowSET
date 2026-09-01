#!/usr/bin/env python
"""RGB entry point for the E3Diff conditional DDPM (the DDPM / SR3-class row).

Why a wrapper instead of the repo's main.py directly: E3Diff's SAR_EO loader
hard-codes `ch = 1` (a grayscale EO target) and a 2-channel condition
[PPB, canny].  QXS-SAROPT and SAR2Opt have RGB optical targets, so the diffusion
must run at 3 channels; and with `channels = 3` the repo's ddim_sample() line

    img_onestep = [condition_x[:, :self.channels, ...]]

would slice a 2-channel condition and then torch.concat it with 3-channel
predictions -> RuntimeError on the first validation image.  Both problems are
fixed by returning a 3-channel EO target together with a 3-channel condition
[PPB, canny, SAR]; keeping the raw SAR as the third condition channel is also
closer to plain SR3, which conditions on the source image itself.

No upstream file is touched: this module patches SAR2EODataset.__getitem__ in
memory and then runs the repo's main.py verbatim, so every flag, log path and
the `-p val` dump behaviour are unchanged.

Both the DDPM (stage 1) and E3Diff (stage 2) cells of the paper's table ran
through this wrapper.

Usage: identical to E3Diff main.py, e.g.
  E3DIFF_REPO=/path/to/E3Diff \
  python e3diff_rgb_main.py -c cfg.json -p train -enable_wandb "" --seed 1
"""
import os
import runpy
import sys

import torch
from PIL import Image

REPO = os.environ.get('E3DIFF_REPO')
if not REPO:
    raise SystemExit('set E3DIFF_REPO to the vendored E3Diff checkout')
sys.path.insert(0, REPO)

import data.util as Util                       # noqa: E402
from data.LRHR_dataset import SAR2EODataset    # noqa: E402


def _getitem_rgb(self, index):
    hr = self.hr_path[index]
    filename = os.path.basename(hr)
    img_EO = Image.open(hr).convert('RGB')
    img_SAR = Image.open(os.path.join(self.dataroot, 'SAR', filename)).convert('RGB')
    img_ppb = Image.open(os.path.join(self.dataroot, 'SAR-PPB', filename)).convert('RGB')
    img_canny = Image.open(os.path.join(self.dataroot, 'SAR-canny', filename)).convert('RGB')
    # same augmentation call as upstream (4 images -> flips, rot90, and random
    # brightness on the SAR/PPB pair only)
    img_SAR, img_ppb, img_EO, img_canny = Util.transform_augment(
        [img_SAR, img_ppb, img_EO, img_canny], split=self.split, min_max=(-1, 1))
    cond = torch.cat((img_ppb[0:1], img_canny[0:1], img_SAR[0:1]), dim=0)
    return {'HR': img_EO[0:3], 'LR': img_SAR[0:3], 'SR': cond,
            'Index': index, 'filename': filename}


SAR2EODataset.__getitem__ = _getitem_rgb

_main = os.path.join(REPO, 'main.py')
sys.argv[0] = _main
runpy.run_path(_main, run_name='__main__')
