#!/usr/bin/env python3
"""Score a folder of generated EO images against the ground-truth test set.

This is the evaluator that produced every number in the paper's main table, for
ReFlowSET and for all fifteen retrained comparison methods alike. It reports

    PSNR, SSIM   torchmetrics, `data_range=1`, computed **one image at a time**
    LPIPS        `lpips.LPIPS(net='vgg')`, fed `x * 2 - 1`
    DISTS        `pyiqa.create_metric('dists')`, fed [0, 1]
    FID          `pytorch-fid`, InceptionV3 pool3, 2048 dims

against the whole ground-truth folder, with the ground truth centre-cropped to
the generated resolution when the two differ (which is how SAR2Opt's 512-crop
protocol falls out).

LPIPS convention — read this before comparing with anything
-----------------------------------------------------------
LPIPS is called two different ways in this literature and they differ by about
0.05. We use the standard one: `[0, 1]` images are mapped to `[-1, 1]` and
handed to the network with `normalize=False`. Several released evaluators
instead hand `[0, 1]` straight to a network that expects `[-1, 1]`, halving the
effective contrast; that reads systematically **lower** — measured 0.534 vs
0.480 on QXS-SAROPT and 0.522 vs 0.480 on SAR2Opt for the same images — and is
not comparable with the numbers in our table. Check which one your evaluator
uses before concluding anything.

Four invariants, and why they are not flags
-------------------------------------------
1. **`--denorm standard`.** Predictions must have been written with
   `x * 0.5 + 0.5`, which is what `scripts/translate.py` does. The alternative
   `x + 0.5` convention is a 2x contrast stretch applied to prediction *and*
   ground truth; on real QXS pairs it reads 21.42 dB where the standard
   convention reads 15.81. It cannot be detected from the PNGs, so it is on you
   not to mix the two.
2. **Per-image PSNR/SSIM.** They are computed at batch size 1 and then averaged.
   Pooling the mean-squared error over a batch first silently produces a
   different, higher number, and every comparison row was scored per image.
3. **KID is not reported here.** It is not a column of the paper's table. If you
   add it, size the subsets explicitly — the usual default of 1,000 is clamped
   to `n`, which makes every "subset" the full set and collapses the reported
   standard deviation to ~1e-16. Where we do report KID, it is 100 subsets of
   500.
4. **No DINO-feature metric.** Scoring in DINOv3's feature space would be partly
   in-sample: a DINOv3 ViT-L/16 is ReFlowSET's own REPA teacher during training.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

#: sar2opt ships .jpg, QXS-SAROPT .png.
SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="  python scripts/evaluate.py --pred-dir results/eo --gt-dir DATA/testB",
    )
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Folder of generated EO images, one per test item, named by "
                         "the test item's stem.")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="Folder of ground-truth EO images for the same test split.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Compute device (default: cuda when available, else cpu).")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size for LPIPS, DISTS and FID (default: 16). PSNR and "
                         "SSIM ignore it and are always computed one image at a time; "
                         "no reported number depends on this.")
    return ap.parse_args()


def load01(path: Path) -> torch.Tensor:
    """Read an image as `[3, H, W]` float32 in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def center_crop(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    _, height, width = t.shape
    top, left = max(0, (height - h) // 2), max(0, (width - w) // 2)
    return t[:, top:top + h, left:left + w]


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    gt = {p.stem: p for p in images_in(args.gt_dir)}
    if not gt:
        raise SystemExit(f"no images in {args.gt_dir}")
    pairs = [(p, gt[p.stem]) for p in images_in(args.pred_dir) if p.stem in gt]
    if not pairs:
        raise SystemExit(
            f"none of the files in {args.pred_dir} share a stem with {args.gt_dir}")
    pairs.sort()
    if len(pairs) < len(gt):
        print(f"[warn] {len(pairs)}/{len(gt)} ground-truth stems matched — this row was "
              f"scored on a different set of images than a complete one")

    height, width = load01(pairs[0][0]).shape[1:]

    from torchmetrics.functional.image import (
        peak_signal_noise_ratio, structural_similarity_index_measure)
    import lpips as lpips_mod
    import pyiqa

    lpips_vgg = lpips_mod.LPIPS(net="vgg").to(device).eval()
    dists = pyiqa.create_metric("dists", device=device)

    psnr_all: list[float] = []
    ssim_all: list[float] = []
    lpips_all: list[float] = []
    dists_all: list[float] = []

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch):
            chunk = pairs[start:start + args.batch]
            x = torch.stack([center_crop(load01(p), height, width)
                             for p, _ in chunk]).to(device)
            y = torch.stack([center_crop(load01(g), height, width)
                             for _, g in chunk]).to(device)
            for i in range(len(chunk)):
                # One image at a time: invariant 2.
                psnr_all.append(float(peak_signal_noise_ratio(
                    x[i:i + 1], y[i:i + 1], data_range=1.0)))
                ssim_all.append(float(structural_similarity_index_measure(
                    x[i:i + 1], y[i:i + 1], data_range=1.0)))
            # Standard LPIPS scaling: [0, 1] -> [-1, 1] before the network.
            lpips_all.extend(lpips_vgg(x * 2 - 1, y * 2 - 1).flatten().cpu().tolist())
            dists_all.extend(dists(x, y).flatten().cpu().tolist())

    from pytorch_fid.fid_score import calculate_fid_given_paths

    with tempfile.TemporaryDirectory() as pred_tmp, tempfile.TemporaryDirectory() as gt_tmp:
        # FID over exactly the matched predictions, against the whole ground-truth
        # split. Symlinks, so nothing is copied.
        for path, _ in pairs:
            (Path(pred_tmp) / path.name).symlink_to(path.resolve())
        gt_files = images_in(args.gt_dir)
        if Image.open(gt_files[0]).size == (width, height):
            gt_path = str(args.gt_dir)
        else:
            # Same centre crop the per-image metrics used, materialised because
            # pytorch-fid reads from disk.
            for path in gt_files:
                image = Image.open(path).convert("RGB")
                w0, h0 = image.size
                left, top = (w0 - width) // 2, (h0 - height) // 2
                image.crop((left, top, left + width, top + height)).save(
                    Path(gt_tmp) / f"{path.stem}.png")
            gt_path = gt_tmp
        fid = calculate_fid_given_paths(
            [gt_path, pred_tmp], batch_size=args.batch, device=device, dims=2048)

    print(f"\n{args.pred_dir}  vs  {args.gt_dir}")
    print(f"  n         {len(pairs)}  @ {height}x{width}")
    print(f"  PSNR      {np.mean(psnr_all):.4f}")
    print(f"  SSIM      {np.mean(ssim_all):.5f}")
    print(f"  LPIPS     {np.mean(lpips_all):.5f}   (VGG, x*2-1)")
    print(f"  DISTS     {np.mean(dists_all):.5f}")
    print(f"  FID       {fid:.4f}")
    print("\n  LPIPS convention: standard — [0,1] mapped to [-1,1] before the VGG "
          "network.\n  The alternative [0,1]/normalize=False convention several "
          "released\n  evaluators use reads about 0.05 LOWER and is not comparable "
          "with this\n  number or with the paper's table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
