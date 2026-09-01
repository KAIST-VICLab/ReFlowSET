#!/usr/bin/env python3
"""Autoencoder reconstruction ceilings — the measurement behind ReFlowSET's premise.

A latent-space generator can never beat the reconstruction of the autoencoder it
lives in.  Encode an image, decode it again with a *perfect* generator in
between, and the result is still only as good as the codec round trip.  That
round trip is therefore a **ceiling** on every row of a latent-model comparison,
and it is a property of the autoencoder, not of the generator.

This script measures that ceiling for six image autoencoders on four SAR/EO
benchmarks, under one protocol, on the same images:

    autoencoders   SD2.1, SDXL, SD3.0, SD3.5, FLUX.1, FLUX.2
    datasets       QXS-SAROPT, SAR2Opt, SpaceNet6, SAR-1M
    modalities     EO (the target of the translation) and SAR (the condition)

Ceilings are quoted from a measurement, never from each autoencoder's own paper:
published numbers are on different images, at different resolutions, under
different de-normalisations, and are not comparable with each other or with a
SAR/EO benchmark.

What is measured
----------------
Every autoencoder is frozen and is fed the **posterior mean**, never a random
posterior sample, so the number is the deterministic latent bottleneck a
diffusion model actually lives in.

    EO and SAR   PSNR, SSIM, LPIPS(vgg), MS-SSIM (only at >= 161 px),
                 Pearson (global, corrected), HF corr., FID, KID
    SAR only     Grad corr. (Sobel), and a speckle-severity breakdown by
                 per-image ENL tertile

``HF corr.`` is a **global** correlation of 3x3-Laplacian high-pass images.  It
is deliberately never called SCC: the classic SCC averages a *local* 8x8
correlation, collapses to ~0.00-0.02 on this data, and is not interchangeable
with this column.

SAR through an RGB autoencoder
------------------------------
None of these six autoencoders has a SAR-native input.  Every scalar SAR channel
is therefore replicated to 3 channels and encoded on its own — which is exactly
the configuration a latent SAR-to-EO model is forced into.  For SpaceNet6 this
yields one number per polarisation band (HH, HV, VH, VV) for free, plus their
mean; the ceiling for a quad-pol tile is not the ceiling for one band.

Latent-path integrity (fatal)
-----------------------------
The FLUX.2 path is not ``decoder(encoder(x))``.  A 2x2 space-to-depth pack and an
``affine=False`` BatchNorm with shipped running statistics (``eps=1e-4``) sit
between the codec and the model.  If either leg were not an exact inverse of the
other, every FLUX.2 number below would silently measure that bug instead of the
autoencoder.  ``verify_latent_path`` therefore checks, on real data and before
any metric is computed:

  1. ``unpack(pack(z)) == z`` bitwise, over several shapes and dtypes;
  2. ``inv_normalize(normalize(z)) == z`` to float32 round-off, with the same
     round trip redone in float64 to show the algebra itself is exact;
  3. ``decode(encode(x)) == decoder(mean(encoder(x)))`` bitwise — i.e. the whole
     pack/normalise sandwich is the identity on the tensor the decoder sees.

Check 3 is the one that matters: 1 and 2 can both pass while the two legs are
wired in the wrong order.  **The run aborts if any of them fails.**  There is no
flag to skip it.

LPIPS convention — read this before comparing with anything
-----------------------------------------------------------
``LPIPS`` in the output is the convention the published ceiling tables were
measured with: ``[0, 1]`` images handed to ``lpips.LPIPS(net='vgg')`` with
``normalize=False``.  ``LPIPS (standard scaling)`` is the other convention,
``[0, 1]`` mapped to ``[-1, 1]`` first, which is what ``scripts/evaluate.py``
reports for the paper's translation table.  The two differ by roughly 0.05 and
are **not** comparable.  Both are emitted so neither has to be guessed at; which
one you quote must match whatever you are comparing against.

Downloads
---------
This repository redistributes no imagery and no third-party autoencoder weights.
The script will fetch two things on your behalf, each from its own publisher and
under its own licence: ``pytorch-fid``'s InceptionV3 weights on first use, and —
when neither ``--vae-root`` nor ``--vae-path`` points at a local copy — each
autoencoder from its Hugging Face repository.  Three of those five repositories
are gated, so you must accept their terms first.  ``vae_audit/fetch_vaes.py``
downloads them ahead of time instead.  Imagery is always yours to obtain: see
``vae_audit/README.md`` for every link and its terms.

Examples
--------
    # get the autoencoders first (vae_audit/fetch_vaes.py), then:

    # QXS-SAROPT, all six autoencoders, EO and SAR
    python vae_audit/vae_recon_audit.py \\
        --dataset qxs-saropt --data-root /data/QXS-SAROPT \\
        --vae-root weights/vae_zoo --flux2-weights weights/ae.safetensors \\
        --out results/vae_ceiling.json

    # SpaceNet6, per-polarisation SAR ceilings, FLUX.2 only
    python vae_audit/vae_recon_audit.py \\
        --dataset spacenet6 --data-root /data/SpaceNet6 \\
        --flux2-weights weights/ae.safetensors --vae flux2 --modality sar

    # freeze a reproducible SAR-1M draw, then measure it
    python vae_audit/vae_recon_audit.py --make-sar1m-manifest \\
        --data-root /data/SAR-1M --n 1000 --out sar1m_audit.jsonl
    python vae_audit/vae_recon_audit.py --dataset sar1m \\
        --data-root /data/SAR-1M --sar1m-manifest sar1m_audit.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reflowset.autoencoder_flux2 import AutoencoderFlux2  # noqa: E402


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #
#: Metric column names.  Kept verbatim from the published ceiling tables so a
#: run of this script and the numbers in the paper can be diffed key by key.
PEARSON_KEY = "Pearson (global, corrected)"
HF_CORR_KEY = "HF corr."
GRAD_CORR_KEY = "Grad corr. (Sobel)"
LPIPS_STD_KEY = "LPIPS (standard scaling)"

#: MS-SSIM's 5 default scales halve the image four times, so it needs >= 161 px.
#: Below that the column is omitted rather than silently computed at fewer
#: scales, which would not be comparable with the rest of the table.
MS_SSIM_MIN_PX = 161

#: ``--vae`` name -> (published label, Hugging Face repo id, subfolder).
#:
#: The label is what appears as the JSON key, so a run of this script lines up
#: with the published ceiling files column for column.  The repo ids are the ones
#: the measured weights were fetched from; each was re-checked against the HF
#: model API on 2026-09-01 (id resolves, licence and gating as noted in
#: ``vae_audit/README.md``).  Gated repos need a licence accepted by the account
#: whose token is in your environment.
#:
#: SD2.1 is a community mirror, flagged here as it is in the README:
#: ``stabilityai/stable-diffusion-2-1`` and ``-2-1-base`` return 404 from the HF
#: API, so the canonical repo cannot be pulled and the mirror cannot be diffed
#: against an original that is no longer served.
VAE_REGISTRY: dict[str, tuple[str, str, str]] = {
    "sd21": ("SD2.1", "Manojb/stable-diffusion-2-1-base", "vae"),
    "sdxl": ("SDXL", "stabilityai/stable-diffusion-xl-base-1.0", "vae"),
    "sd30": ("SD3.0", "stabilityai/stable-diffusion-3-medium-diffusers", "vae"),
    "sd35": ("SD3.5", "stabilityai/stable-diffusion-3.5-large", "vae"),
    # The published FLUX.1 numbers were produced from
    # black-forest-labs/FLUX.1-dev, whose licence is non-commercial.
    # FLUX.1-schnell serves the same autoencoder blob under Apache-2.0
    # (vae/diffusion_pytorch_model.safetensors, sha256 f5b59a26...604d40a3 in
    # both, verified from Hub blob metadata — see vae_audit/README.md), so the
    # Apache-2.0 copy is the default here and the row is unchanged.
    "flux1": ("FLUX.1", "black-forest-labs/FLUX.1-schnell", "vae"),
    # Not a diffusers AutoencoderKL: ReFlowSET's own packed/BN-normalised codec,
    # loaded from the Apache-2.0 weights this repository ships instructions for.
    "flux2": ("FLUX.2", "", ""),
}

VAE_ORDER = ["flux2", "flux1", "sd35", "sd30", "sdxl", "sd21"]

#: ``--dataset`` name -> (label, default crop, default relative split list).
#: Alternative spellings accepted on the command line, normalised to the keys of
#: DATASET_REGISTRY.
DATASET_ALIASES = {"sar-1m": "sar1m", "sar_1m": "sar1m", "qxs": "qxs-saropt",
                   "qxs_saropt": "qxs-saropt", "saropt": "sar2opt",
                   "sn6": "spacenet6", "spacenet": "spacenet6"}

DATASET_REGISTRY: dict[str, tuple[str, int, str]] = {
    "qxs-saropt": ("QXS-SAROPT", 256, "splits/qxs-saropt/test_eo_list.txt"),
    "sar2opt": ("SAR2Opt", 512, "splits/sar2opt/test_eo_list.txt"),
    "spacenet6": ("SpaceNet6", 512, ""),
    "sar1m": ("SAR-1M", 256, ""),
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
_GRAY_WEIGHTS = (0.299, 0.587, 0.114)  # ITU-R BT.601 luma
_HP_LAPLACIAN = ((-1.0, -1.0, -1.0), (-1.0, 8.0, -1.0), (-1.0, -1.0, -1.0))
_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
_EPS = 1e-12


def _check_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)} vs b={tuple(b.shape)}")
    if a.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(a.shape)}")
    if a.shape[1] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got {a.shape[1]}")
    if b.device != a.device:
        b = b.to(a.device)
    return a.float(), b.float()


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] -> [B, 1, H, W]`` BT.601 luma; 1-channel input passes through."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor(_GRAY_WEIGHTS, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def _corrected_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image mean-centred Pearson r over all pixels.  ``[B,1,H,W] -> [B]``.

    "Corrected" is the mean-centring.  Without it this degenerates into a cosine
    similarity dominated by the shared DC offset, which reads ~0.95 for unrelated
    remote-sensing crops.  A constant image has no defined correlation and
    contributes 0.0 rather than NaN, so one flat no-data crop cannot poison the
    run-level mean.
    """
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()
    return torch.where(den > _EPS, num / den.clamp_min(_EPS), torch.zeros_like(num))


def _highpass(x: torch.Tensor) -> torch.Tensor:
    """3x3 Laplacian high-pass, reflect padding.  ``[B,1,H,W] -> [B,1,H,W]``."""
    k = torch.tensor(_HP_LAPLACIAN, dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k)


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of ``[B,1,H,W]``, reflect padding."""
    kx = torch.tensor(_SOBEL_X, dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(xp, kx)
    gy = F.conv2d(xp, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR.  Identical images give ``inf``, deliberately: a silently
    capped PSNR would hide a degenerate (e.g. all-zero) evaluation."""
    a, b = _check_pair(a, b)
    mse = (a - b).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10((data_range**2) / mse)


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """SSIM, torchmetrics defaults (11x11 Gaussian, sigma 1.5), on 3-channel RGB
    and averaged over channels — not on grayscale."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    a, b = _check_pair(a, b)
    return structural_similarity_index_measure(
        a, b, gaussian_kernel=True, sigma=1.5, kernel_size=11,
        reduction="none", data_range=data_range, k1=0.01, k2=0.03,
    )


def ms_ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Multi-scale SSIM, torchmetrics defaults, 5 scales.  Needs H, W >= 161."""
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure,
    )

    a, b = _check_pair(a, b)
    if min(a.shape[-2:]) < MS_SSIM_MIN_PX:
        raise ValueError(
            f"ms_ssim needs H, W >= {MS_SSIM_MIN_PX} for its 5 default scales, got "
            f"{tuple(a.shape[-2:])}."
        )
    return multiscale_structural_similarity_index_measure(
        a, b, gaussian_kernel=True, sigma=1.5, kernel_size=11,
        reduction="none", data_range=data_range, k1=0.01, k2=0.03,
    )


_LPIPS_CACHE: dict = {}


def _get_lpips(device: torch.device):
    key = str(device)
    if key not in _LPIPS_CACHE:
        import lpips as _lpips

        model = _lpips.LPIPS(net="vgg").to(device).eval()
        model.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def lpips_dist(a: torch.Tensor, b: torch.Tensor, standard_scaling: bool) -> torch.Tensor:
    """LPIPS(vgg), per image.

    ``standard_scaling=False`` hands the ``[0, 1]`` tensors to the network with
    ``normalize=False``, i.e. the network treats them as if they were already
    ``[-1, 1]``.  That is the convention the published ceiling tables use, and it
    systematically compresses the value.  ``standard_scaling=True`` maps
    ``[0, 1] -> [-1, 1]`` first.  The two are NOT comparable.
    """
    a, b = _check_pair(a, b)
    if a.shape[1] == 1:
        a, b = a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)
    model = _get_lpips(a.device)
    with torch.no_grad():
        return model(a, b, normalize=standard_scaling).flatten()


def pearson_global(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Corrected global Pearson correlation on BT.601 grayscale, per image."""
    a, b = _check_pair(a, b)
    return _corrected_pearson(_to_gray(a), _to_gray(b))


def hf_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Global high-pass correlation.  ALWAYS reported as ``HF corr.``, never "SCC".

    Grayscale -> 3x3 Laplacian high-pass (reflect padded) -> one corrected
    Pearson correlation over the whole high-passed image, per image.  The
    difference from the classic SCC is that last step: SCC averages a *local* 8x8
    correlation, and local windows over flat regions are near-random, which is
    why SCC collapses to 0.000-0.02 on this data while this column has usable
    dynamic range.  They must never share a column.
    """
    a, b = _check_pair(a, b)
    return _corrected_pearson(_highpass(_to_gray(a)), _highpass(_to_gray(b)))


def grad_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Edge/gradient correlation: same grayscale and same corrected-Pearson core
    as :func:`hf_corr`, with a Sobel magnitude instead of a Laplacian as the
    band-selecting operator, so the two columns are directly comparable."""
    a, b = _check_pair(a, b)
    return _corrected_pearson(_sobel_mag(_to_gray(a)), _sobel_mag(_to_gray(b)))


def enl_proxy(x01: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Per-image equivalent-number-of-looks proxy on ``[B,1,H,W]`` in ``[0,1]``.

    ENL = mean^2 / var estimated in ``win x win`` windows; the per-image value is
    the median over windows.  A **low** ENL means strong speckle.  This is a
    severity *ordering* on display-domain rasters, not a calibrated ENL: the SAR
    here is 8-bit display imagery (or a percentile-stretched dB tile), so no
    absolute reading is claimed and only the within-dataset ranking is used.
    """
    k = win * win
    ones = torch.ones(1, 1, win, win, device=x01.device, dtype=x01.dtype) / k
    xp = F.pad(x01, (win // 2,) * 4, mode="reflect")
    mu = F.conv2d(xp, ones)
    mu2 = F.conv2d(xp * xp, ones)
    var = (mu2 - mu * mu).clamp_min(0.0)
    return ((mu * mu) / (var + 1e-6)).flatten(1).median(dim=1).values


# --------------------------------------------------------------------------- #
# FID / KID
# --------------------------------------------------------------------------- #
class FIDAccumulator:
    """Streaming InceptionV3 pool3 feature collector for ``pytorch_fid`` FID.

    ``dims=2048`` (pool3), TF-ported inception weights, ``resize_input=True``
    (bilinear to 299x299), ``normalize_input=True``.  ``update`` quantises to the
    uint8 grid by default so the in-memory path is numerically identical to the
    usual save-8-bit-PNGs-then-score path.

    FID is a set statistic and is only comparable when the sample count and the
    crop protocol match the reference exactly.  Below n ~ 2048 the 2048-dim
    covariance is rank-deficient and the estimate is biased upward — which is why
    :func:`kid` sits next to it.

    The first call downloads ``pytorch-fid``'s InceptionV3 weights into the torch
    hub cache if they are not already there; that is the only download this
    script performs.
    """

    def __init__(self, device, dims: int = 2048, quantize_uint8: bool = True,
                 batch_size: int = 50) -> None:
        self.device = torch.device(device)
        self.dims = int(dims)
        self.quantize_uint8 = bool(quantize_uint8)
        self.batch_size = int(batch_size)
        self._feats: list[torch.Tensor] = []
        self._model = None

    def _get_model(self):
        if self._model is None:
            from pytorch_fid.inception import InceptionV3

            block = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block], resize_input=True, normalize_input=True)
            self._model = model.to(self.device).eval()
            self._model.requires_grad_(False)
        return self._model

    @torch.no_grad()
    def update(self, x01: torch.Tensor) -> None:
        """Accumulate features for a batch of ``[B, 3, H, W]`` images in ``[0, 1]``."""
        if x01.ndim != 4 or x01.shape[1] != 3:
            raise ValueError(f"expected [B, 3, H, W], got {tuple(x01.shape)}")
        x = x01.float().clamp(0.0, 1.0)
        if self.quantize_uint8:
            x = torch.round(x * 255.0) / 255.0
        model = self._get_model()
        for i in range(0, x.shape[0], self.batch_size):
            feat = model(x[i:i + self.batch_size].to(self.device))[0]
            # Only the dims=2048 block ends in an adaptive average pool; the
            # smaller blocks return spatial maps.  Same guard as
            # pytorch_fid.fid_score.get_activations.
            if feat.shape[2] != 1 or feat.shape[3] != 1:
                feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))
            self._feats.append(feat.squeeze(3).squeeze(2).double().cpu())

    @property
    def n(self) -> int:
        return int(sum(f.shape[0] for f in self._feats))

    @property
    def features(self) -> np.ndarray:
        if not self._feats:
            raise RuntimeError("FIDAccumulator has no samples; call update() first.")
        return torch.cat(self._feats, dim=0).numpy().astype(np.float64)


def frechet_from_features(fa, fb) -> float:
    """Frechet distance between two ``[N, D]`` feature arrays."""
    from pytorch_fid.fid_score import calculate_frechet_distance

    fa = np.asarray(fa, dtype=np.float64)
    fb = np.asarray(fb, dtype=np.float64)
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        raise RuntimeError(
            f"FID needs >= 2 samples per set, got {fa.shape[0]} and {fb.shape[0]}.")
    return float(calculate_frechet_distance(
        fa.mean(axis=0), np.cov(fa, rowvar=False),
        fb.mean(axis=0), np.cov(fb, rowvar=False),
    ))


def kid(fa, fb, subset_size: int = 500, n_subsets: int = 100, degree: int = 3,
        seed: int = 0) -> dict:
    """Kernel Inception Distance — unbiased MMD^2 with the polynomial kernel.

    Binkowski et al., "Demystifying MMD GANs" (ICLR 2018), eq. 2 and sec. 4::

        k(x, y) = (x . y / d + 1) ** degree,   d = feature dimension

    with the *unbiased* estimator (within-set diagonals excluded).  It exists
    next to FID because FID's plug-in estimator is biased upward whenever
    N < D = 2048, which is the regime every test set here except QXS-SAROPT is
    in.  KID is unbiased, so 0 means 0 at any n, and the spread over subsets is
    an honest error bar.

    ``subset_size`` is clipped to ``min(len(fa), len(fb))``.  Quote the mean with
    the std as the error bar; a gap smaller than the std is not a gap.
    """
    fa = np.asarray(fa, dtype=np.float64)
    fb = np.asarray(fb, dtype=np.float64)
    if fa.ndim != 2 or fb.ndim != 2:
        raise ValueError(f"kid expects [N, D] arrays, got {fa.shape} and {fb.shape}")
    if fa.shape[1] != fb.shape[1]:
        raise ValueError(f"feature dim mismatch: {fa.shape[1]} vs {fb.shape[1]}")
    m = min(fa.shape[0], fb.shape[0])
    if m < 2:
        raise RuntimeError(
            f"kid needs >= 2 samples per set, got {fa.shape[0]} and {fb.shape[0]}.")
    subset_size = int(min(subset_size, m))
    d = fa.shape[1]
    rng = np.random.default_rng(seed)

    def _k(x, y):
        return (x @ y.T / d + 1.0) ** degree

    vals = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        x = fa[rng.choice(fa.shape[0], subset_size, replace=False)]
        y = fb[rng.choice(fb.shape[0], subset_size, replace=False)]
        kxx, kyy, kxy = _k(x, x), _k(y, y), _k(x, y)
        n = subset_size
        vals[i] = (
            (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
            + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
            - 2.0 * kxy.mean()
        )
    return {
        "kid_mean": float(vals.mean()),
        "kid_std": float(vals.std(ddof=1)) if n_subsets > 1 else 0.0,
        "subset_size": subset_size,
        "n_subsets": int(n_subsets),
    }


# --------------------------------------------------------------------------- #
# autoencoder wrappers
# --------------------------------------------------------------------------- #
class DiffusersAE:
    """A `diffusers` ``AutoencoderKL``, wrapped to one ``roundtrip`` contract.

    ``encode(x).latent_dist.mean`` — the posterior MEAN, never ``.sample()``.
    ``scaling_factor`` is a pure affine on the latent and cancels exactly in a
    round trip, so it is not applied here; it would change nothing.
    """

    def __init__(self, source: str, subfolder: str, device, dtype=torch.float32):
        from diffusers import AutoencoderKL

        local = Path(source)
        if local.is_dir():
            # A local zoo directory: either the AutoencoderKL folder itself, or
            # the model folder that contains it as `vae/`.
            inner = local / subfolder if subfolder else local
            path, kwargs = str(inner if inner.is_dir() else local), {}
        else:
            path, kwargs = source, ({"subfolder": subfolder} if subfolder else {})
        self.m = AutoencoderKL.from_pretrained(path, torch_dtype=dtype, **kwargs)
        self.m = self.m.to(device).eval()
        self.m.requires_grad_(False)
        self.source = path
        self.latent_channels = int(self.m.config.latent_channels)
        # f = 2 ** (number of downsampling stages)
        self.spatial_factor = 2 ** (len(self.m.config.block_out_channels) - 1)

    @torch.no_grad()
    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.decode(self.m.encode(x).latent_dist.mean).sample


class Flux2AE:
    """ReFlowSET's frozen FLUX.2 autoencoder, loaded exactly as inference loads it.

    Its public latent is ``[B, 128, H/16, W/16]``: an 8x convolutional stride
    followed by a 2x2 space-to-depth pack, then a per-channel BatchNorm with the
    checkpoint's running statistics.  ``encode`` returns the posterior mean.
    """

    def __init__(self, source: str, device, dtype=torch.float32):
        path = Path(source)
        if path.is_dir():
            self.m = AutoencoderFlux2.from_pretrained(str(path), torch_dtype=dtype)
            self.m = self.m.to(device).eval()
            self.m.requires_grad_(False)
        elif path.is_file():
            self.m = AutoencoderFlux2.from_single_file(str(path), torch_dtype=dtype)
            self.m = self.m.to(device)
        else:
            raise FileNotFoundError(
                f"FLUX.2 autoencoder not found at {source}. Rebuild it with "
                f"scripts/convert_flux2_ae.py (see the README, 'The autoencoder'), "
                f"or point --flux2-weights at a released checkpoint's vae/ folder."
            )
        self.source = str(path)
        self.latent_channels = self.m.latent_channels      # 128 = 32ch x 2x2 pack
        self.spatial_factor = self.m.spatial_factor        # 16

    @torch.no_grad()
    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.decode(self.m.encode(x))


def build_ae(name: str, args, device):
    """Instantiate one autoencoder by its ``--vae`` name."""
    label, repo, subfolder = VAE_REGISTRY[name]
    if name == "flux2":
        return Flux2AE(args.flux2_weights, device), label, args.flux2_weights
    source = args.vae_path.get(name)
    if source is None and args.vae_root:
        source = str(Path(args.vae_root) / label)
    if source is None:
        source = repo
    return DiffusersAE(source, subfolder, device), label, source


# --------------------------------------------------------------------------- #
# latent-path integrity — fatal
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_latent_path(codec: Flux2AE, x: torch.Tensor) -> dict:
    """Pack/unpack + normalize/inv_normalize + full-sandwich identity checks.

    ``x`` is a real image batch in ``[-1, 1]``.  Returns the evidence dict; the
    caller must abort when ``["passed"]`` is False.
    """
    ae = codec.m
    dev = next(ae.parameters()).device
    bn_eps = float(ae.config.bn_eps)
    out: dict = {}

    # 1. pack/unpack bitwise, several shapes and dtypes.  It is a pure
    #    permutation: there is no excuse for any error at all.
    shape_checks = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for shp in [(2, 32, 32, 32), (1, 32, 16, 16), (3, 32, 64, 48), (5, 32, 8, 8)]:
        for dt in (torch.float32, torch.float64):
            z = torch.randn(*shp, generator=g, dtype=torch.float64).to(dev, dt)
            zp = ae.pack(z)
            zu = ae.unpack(zp)
            shape_checks.append({
                "shape": list(shp),
                "dtype": str(dt).split(".")[-1],
                "packed_shape": list(zp.shape),
                "packed_shape_expected": [shp[0], shp[1] * 4, shp[2] // 2, shp[3] // 2],
                "bitwise_exact": bool(torch.equal(zu, z)),
            })
    out["pack_unpack"] = {
        "cases": shape_checks,
        "all_bitwise_exact": all(c["bitwise_exact"] for c in shape_checks),
        "all_shapes_correct": all(
            c["packed_shape"] == c["packed_shape_expected"] for c in shape_checks),
    }

    # 2. normalize / inv_normalize on real encoder statistics.
    moments = ae.encoder(x)
    mean = torch.chunk(moments, 2, dim=1)[0]
    zpk = ae.pack(mean)
    zn = ae.normalize(zpk)
    zb = ae.inv_normalize(zn)
    denom = zpk.abs().max().clamp_min(1e-12)
    # The same round trip in float64.  The BatchNorm affine and its inverse are
    # algebraically exact, so whatever is left in float32 is pure round-off.
    m64 = ae.bn.running_mean.double().view(1, -1, 1, 1)
    s64 = torch.sqrt(ae.bn.running_var.double().view(1, -1, 1, 1) + bn_eps)
    z64 = zpk.double()
    zb64 = ((z64 - m64) / s64) * s64 + m64
    z_from64 = zb64.float()  # float64 round trip, cast back to the stored dtype
    out["normalize_inverse"] = {
        "input": "packed posterior mean of a real image batch",
        "max_abs_err": float((zb - zpk).abs().max().item()),
        "max_rel_err": float(((zb - zpk).abs().max() / denom).item()),
        "max_rel_err_float64": float(((zb64 - z64).abs().max() / denom.double()).item()),
        "float64_roundtrip_recovers_float32_latent_bitwise": bool(
            torch.equal(z_from64, zpk)),
        "float32_eps": float(torch.finfo(torch.float32).eps),
        "packed_absmax": float(zpk.abs().max().item()),
        "bn_affine": bool(ae.bn.affine),
        "bn_eps": bn_eps,
        "bn_training_mode": bool(ae.bn.training),
        "bn_running_mean_absmax": float(ae.bn.running_mean.abs().max().item()),
        "bn_running_var_minmax": [
            float(ae.bn.running_var.min().item()),
            float(ae.bn.running_var.max().item()),
        ],
    }

    # 3. THE check: the whole pack/normalize sandwich must be the identity on the
    #    tensor the decoder sees.  Checks 1 and 2 can both pass while the two
    #    legs are wired in the wrong order; this one cannot.
    ref = ae.decoder(mean)
    got = ae.decode(ae.encode(x))
    z_pub = ae.encode(x)
    z_rt = ae.unpack(ae.inv_normalize(z_pub))
    dec_mse = float(((got - ref) ** 2).mean().item())
    # Decoder determinism baseline: the same input twice.  Any residue beyond
    # this is attributable to the 1-ulp latent delta, not to nondeterministic
    # kernels.
    ref2 = ae.decoder(mean)
    # And the decisive test: redo the normalize/inv_normalize leg in float64 and
    # cast back.  The recovered latent is then bitwise identical to the posterior
    # mean, so the decoded image must be bitwise identical too — which is only
    # possible if the pack/normalize sandwich is the identity.
    got64 = ae.decoder(ae.unpack(z_from64))
    out["sandwich_identity"] = {
        "decoder_determinism_max_abs": float((ref2 - ref).abs().max().item()),
        "float64_sandwich_decode_bitwise_equal": bool(torch.equal(got64, ref)),
        "float64_sandwich_decode_max_abs": float((got64 - ref).abs().max().item()),
        "decode_encode_vs_decoder_encoder_mean_bitwise": bool(torch.equal(got, ref)),
        "decode_encode_vs_decoder_encoder_mean_max_abs": float(
            (got - ref).abs().max().item()),
        # The same delta as an image-domain PSNR against data_range=2 ([-1,1]).
        # This is the only figure that matters for "does the plumbing perturb the
        # reconstruction I am about to measure".
        "decode_encode_vs_decoder_encoder_mean_psnr_db": (
            float("inf") if dec_mse == 0.0 else 10.0 * math.log10(4.0 / dec_mse)),
        "latent_recovered_vs_posterior_mean_max_abs": float(
            (z_rt - mean).abs().max().item()),
        "public_latent_shape": list(z_pub.shape),
        "public_latent_shape_expected": [
            x.shape[0], codec.latent_channels,
            x.shape[2] // codec.spatial_factor, x.shape[3] // codec.spatial_factor,
        ],
        "public_latent_mean": float(z_pub.mean().item()),
        "public_latent_std": float(z_pub.std().item()),
    }
    # Pass criteria.  pack/unpack must be bitwise exact.  The BatchNorm leg is a
    # float32 affine and its inverse, so it is allowed a few ulp (rel ~1.2e-7);
    # the float64 leg proves the algebra itself is exact.  The decoded delta that
    # round-off produces must sit >= 80 dB below the data range, i.e. far below
    # every reconstruction PSNR this script reports.
    ni, si = out["normalize_inverse"], out["sandwich_identity"]
    out["passed"] = bool(
        out["pack_unpack"]["all_bitwise_exact"]
        and out["pack_unpack"]["all_shapes_correct"]
        and ni["max_rel_err"] <= 4 * ni["float32_eps"]
        and ni["max_rel_err_float64"] < 1e-14
        and ni["float64_roundtrip_recovers_float32_latent_bitwise"]
        and si["float64_sandwich_decode_bitwise_equal"]
        and si["public_latent_shape"] == si["public_latent_shape_expected"]
        and si["decode_encode_vs_decoder_encoder_mean_psnr_db"] > 80.0
    )
    return out


# --------------------------------------------------------------------------- #
# datasets — small loaders, a directory plus a file list
# --------------------------------------------------------------------------- #
def _read_list(path) -> list[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"split list not found: {p}")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _pil_hwc(path, mode: str | None = None) -> np.ndarray:
    """Open an 8-bit display image as an HWC float32 array in ``[0, 255]``."""
    with Image.open(path) as im:
        arr = np.array(im.convert(mode) if mode else im)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 4:  # drop a container alpha channel
        arr = arr[..., :3]
    return arr.astype(np.float32)


def _is_repeated_rgb(a: np.ndarray) -> bool:
    """True when a multi-channel array is a replicated single-channel raster."""
    if a.ndim != 3 or not (2 <= a.shape[-1] <= 4):
        return False
    ref = a[..., :1].astype(np.float64)
    return bool(np.abs(a.astype(np.float64) - ref).max() == 0.0)


def _collapse_sar(a: np.ndarray, path) -> np.ndarray:
    """Display SAR raster -> exactly one real amplitude channel.

    When the channels are bitwise identical, channel 0 *is* the grayscale, and
    taking it is exact — unlike ``convert("L")``, which applies the BT.601 luma
    weights and rounds.  A raster whose channels are not identical here would be
    something this loader has no documented semantics for, so it is refused
    rather than silently averaged.
    """
    if a.shape[-1] == 1:
        return a
    if _is_repeated_rgb(a):
        return a[..., :1]
    raise ValueError(
        f"{path}: SAR raster has {a.shape[-1]} non-identical channels. This loader "
        f"expects single-channel amplitude (channel count is not polarisation "
        f"count); check that the SAR directory is the one you think it is."
    )


def _center_crop(arrays: list[np.ndarray], crop: int) -> list[np.ndarray]:
    """One identical centre crop over every HWC array.  **Never resizes.**

    Offset is ``(n - crop) // 2``, the albumentations centre-crop arithmetic the
    benchmark protocols use — so SAR2Opt's 600 -> 512 lands at offset 44.
    """
    h, w = arrays[0].shape[:2]
    for a in arrays[1:]:
        if a.shape[:2] != (h, w):
            raise ValueError(
                f"paired rasters disagree in size: {arrays[0].shape[:2]} vs {a.shape[:2]}")
    if h < crop or w < crop:
        raise ValueError(
            f"image {h}x{w} is smaller than the requested crop {crop}. Images are "
            f"never resized by this script: an upsampled image has different "
            f"high-frequency content and its ceiling would not be a ceiling.")
    if h == crop and w == crop:
        return arrays
    top, left = (h - crop) // 2, (w - crop) // 2
    return [a[top:top + crop, left:left + crop] for a in arrays]


def _to_signed(x255: np.ndarray) -> torch.Tensor:
    """HWC ``[0, 255]`` -> CHW float32 ``[-1, 1]`` (``x / 127.5 - 1``)."""
    t = torch.from_numpy(np.ascontiguousarray(x255.transpose(2, 0, 1)))
    return t.float().div_(127.5).sub_(1.0)


class _Paired:
    """Common interface: ``len(ds)`` and ``ds[i] -> (eo, sar, band_names, item_id)``.

    ``eo`` is ``[3, H, W]`` and ``sar`` is ``[C, H, W]``, both float32 in
    ``[-1, 1]``.  ``C`` is 1 everywhere except SpaceNet6, which keeps its four
    native polarisation bands.
    """

    bands: list[str] = ["AMP"]

    def __len__(self) -> int:
        return len(self.eo_paths)


class QXSSarOpt(_Paired):
    """QXS-SAROPT — 256x256 8-bit PNG pairs, so the crop is a no-op.

    The list holds EO paths ``opt_256_oc_0.2/<n>.png``; the SAR path is the same
    string with ``opt_256_oc_0.2`` -> ``sar_256_oc_0.2``.
    """

    def __init__(self, root, list_file, crop: int = 256):
        self.root, self.crop = Path(root), int(crop)
        rel = _read_list(list_file)
        self.eo_paths = [self.root / r for r in rel]
        self.sar_paths = [
            self.root / r.replace("opt_256_oc_0.2", "sar_256_oc_0.2") for r in rel]
        self.ids = [Path(r).stem for r in rel]

    def __getitem__(self, i):
        eo = _pil_hwc(self.eo_paths[i], mode="RGB")
        sar = _collapse_sar(_pil_hwc(self.sar_paths[i]), self.sar_paths[i])
        sar, eo = _center_crop([sar, eo], self.crop)
        return _to_signed(eo), _to_signed(sar), self.bands, self.ids[i]


class SAR2Opt(_Paired):
    """SAR2Opt — native 600x600 JPEG, centre-cropped to 512 (offset 44).

    The list holds EO paths ``testB/<name>.jpg``; the SAR path is the same string
    with the ``B`` directory replaced by its ``A`` sibling.
    """

    def __init__(self, root, list_file, crop: int = 512):
        self.root, self.crop = Path(root), int(crop)
        rel = _read_list(list_file)
        self.eo_paths = [self.root / r for r in rel]
        self.sar_paths = [
            self.root / r.replace("trainB/", "trainA/").replace("testB/", "testA/")
            for r in rel]
        self.ids = [Path(r).stem for r in rel]

    def __getitem__(self, i):
        eo = _pil_hwc(self.eo_paths[i], mode="RGB")
        sar = _collapse_sar(_pil_hwc(self.sar_paths[i]), self.sar_paths[i])
        sar, eo = _center_crop([sar, eo], self.crop)
        return _to_signed(eo), _to_signed(sar), self.bands, self.ids[i]


def _sn6_percent_norm(x: np.ndarray, p=(1, 99), eps: float = 1 / (2**10)) -> np.ndarray:
    """Per-band 1/99 percentile clip and rescale to ``[0, 255]``.

    Computed on the **full uncropped tile**, including its zero no-data border.
    Recomputing it after cropping would change the contrast per sample and the
    numbers would no longer be comparable across tiles.
    """
    pv = np.percentile(x, p, axis=(0, 1))
    y = x.astype(np.float32)
    pmin, pmax = pv[0, ...], pv[1, ...]
    y = np.clip(y, pmin, pmax)
    return (y - pmin) / np.maximum(pmax - pmin, eps) * 255.0


class SpaceNet6(_Paired):
    """SpaceNet6 AOI-11 Rotterdam — 900x900 tiles, centre-cropped to 512.

    EO is the ``PS-RGB`` TIFF read with **PIL** (those tiles are
    ``PlanarConfiguration=2``, so ``tifffile`` would return CHW) and gets no
    stretch.  SAR is the same path with ``PS-RGB`` -> ``SAR-Intensity``, read with
    **tifffile** as a 4-band float32 HWC raster and given the per-band percentile
    stretch above.

    The four bands are kept native — ``(HH, HV, VH, VV)`` — and each is encoded on
    its own, so this dataset reports one ceiling per polarisation.  That band
    order comes from the SpaceNet6 specification; it is not recorded in the
    raster itself.  Bands carry different offsets and noise floors, so do not
    compare band magnitudes against each other in absolute terms.
    """

    bands = ["HH", "HV", "VH", "VV"]

    def __init__(self, root, list_file, crop: int = 512):
        self.root, self.crop = Path(root), int(crop)
        if list_file:
            rel = _read_list(list_file)
            self.eo_paths = [self.root / r for r in rel]
        else:
            found = sorted(self.root.glob("*/AOI_*/PS-RGB/*.tif"))
            if not found:
                found = sorted(self.root.rglob("PS-RGB/*.tif"))
            if not found:
                raise FileNotFoundError(
                    f"no PS-RGB tiles under {self.root}. Expected the official "
                    f"layout <root>/train/AOI_11_Rotterdam/PS-RGB/*.tif, or pass "
                    f"--file-list spacenet6=<list of EO paths relative to root>.")
            self.eo_paths = found
        self.sar_paths = [
            Path(str(p).replace("PS-RGB", "SAR-Intensity")) for p in self.eo_paths]
        self.ids = [p.stem for p in self.eo_paths]

    def __getitem__(self, i):
        import tifffile  # SpaceNet6-only, and heavy

        eo = _pil_hwc(self.eo_paths[i], mode="RGB")
        sar = tifffile.imread(self.sar_paths[i]).astype(np.float32)
        if sar.ndim != 3 or sar.shape[-1] != 4:
            raise ValueError(
                f"expected a 4-band (H, W, 4) SAR-Intensity tile, got {sar.shape} "
                f"from {self.sar_paths[i]}")
        sar = _sn6_percent_norm(sar)  # full-tile stretch, before cropping
        sar, eo = _center_crop([sar, eo], self.crop)
        return _to_signed(eo), _to_signed(sar), self.bands, self.ids[i]


class SAR1M(_Paired):
    """SAR-1M paired subset — driven by a manifest, never by globbing a directory.

    The archive ships a ``paired.json`` of 731,080 pairs and no split, no
    grouping and no per-item attribution.  Enumerating a directory would
    therefore give a draw nobody can reproduce and no way to say which items were
    scored — so this loader reads a JSONL manifest whose sha256 identifies the
    exact draw, and ``--make-sar1m-manifest`` writes one.

    There is a second, harder reason the manifest is mandatory: **AFRL
    distribution-restricted MSTAR and FARAD imagery sits in the same directory
    tree** as the redistributable material, and only the pairing manifest
    separates them.  A ``glob("**/*.png")`` over the corpus root would pull in
    files that must not be used or redistributed.  If you add a loader here,
    keep that property.

    No SAR-1M file list ships with this release: the imagery and its pairing are
    the dataset's to distribute, under its own terms, and a list of our local
    items is not ours to publish.  A manifest you build yourself will therefore
    contain different items from the ones behind the published SAR-1M ceilings,
    and your numbers will be close to but not identical to them.

    Manifest rows are JSON objects with ``sar_path`` and ``eo_path`` (absolute,
    or relative to ``--data-root``) and an optional ``split``.  Pairs must
    already be ``resolution x resolution``; an off-size pair is a hard error, not
    a silent resize, because resizing changes speckle statistics.
    """

    def __init__(self, root, manifest, split: str | None = None, resolution: int = 256):
        self.root, self.resolution = Path(root), int(resolution)
        if not manifest:
            raise ValueError(
                "sar1m needs --sar1m-manifest. Build one with --make-sar1m-manifest; "
                "see the class docstring for why a directory scan is not offered.")
        self.manifest = str(manifest)
        self.eo_paths, self.sar_paths, self.ids = [], [], []
        n_rows = 0
        with open(manifest) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                row = json.loads(line)
                if split and str(row.get("split", "")) != split:
                    continue
                self.sar_paths.append(_resolve(row["sar_path"], self.root, "SAR"))
                self.eo_paths.append(_resolve(row["eo_path"], self.root, "OPT"))
                self.ids.append(str(row.get("sample_id") or Path(row["sar_path"]).stem))
        if not self.eo_paths:
            raise RuntimeError(
                f"{manifest}: no rows selected ({n_rows} read"
                + (f", split={split!r}" if split else "") + ").")

    def __getitem__(self, i):
        eo = _pil_hwc(self.eo_paths[i], mode="RGB")
        sar = _collapse_sar(_pil_hwc(self.sar_paths[i]), self.sar_paths[i])
        r = self.resolution
        for name, a in (("EO", eo), ("SAR", sar)):
            if a.shape[:2] != (r, r):
                raise ValueError(
                    f"{self.ids[i]}: {name} is {a.shape[1]}x{a.shape[0]}, expected "
                    f"{r}x{r}. Nothing is resized here — drop the pair or pass a "
                    f"matching --crop sar1m=<n>.")
        sar, eo = _center_crop([sar, eo], r)
        return _to_signed(eo), _to_signed(sar), self.bands, self.ids[i]


def _resolve(name: str, root: Path, subdir: str) -> Path:
    """Manifest path value -> real path: absolute, root-relative, or bare."""
    p = Path(name)
    if p.is_absolute():
        return p
    if len(p.parts) > 1:
        return root / p
    return root / subdir / p


def build_dataset(name: str, args):
    """Instantiate one dataset by its ``--dataset`` name."""
    root = args.data_root.get(name)
    if root is None:
        raise SystemExit(
            f"--dataset {name} needs a root: --data-root {name}=/path/to/dataset")
    label, default_crop, default_list = DATASET_REGISTRY[name]
    crop = args.crop.get(name, default_crop)
    list_file = args.file_list.get(name)
    if list_file is None and default_list:
        candidate = _REPO / default_list
        list_file = str(candidate) if candidate.is_file() else None
        if list_file is None:
            raise SystemExit(
                f"--dataset {name} needs a split list: {default_list} was not found "
                f"in this checkout. Pass --file-list {name}=<path>.")
    if name == "qxs-saropt":
        return QXSSarOpt(root, list_file, crop), label, crop
    if name == "sar2opt":
        return SAR2Opt(root, list_file, crop), label, crop
    if name == "spacenet6":
        return SpaceNet6(root, list_file, crop), label, crop
    if name == "sar1m":
        return SAR1M(root, args.sar1m_manifest, args.sar1m_split, crop), label, crop
    raise SystemExit(f"unknown dataset {name!r}")


# --------------------------------------------------------------------------- #
# SAR-1M manifest builder
# --------------------------------------------------------------------------- #
def make_sar1m_manifest(root: str, n: int, seed: int, out_path: Path) -> dict:
    """Freeze a reproducible, seeded draw from SAR-1M's ``paired.json``.

    Emits one JSON object per line with ``sample_id``, ``sar_path``, ``eo_path``
    (all relative to ``root``, so the manifest is portable), ``group_id`` and
    ``split``, plus the sha256 of the file so the draw can be cited.

    Candidates are size-checked from the image header before they enter the
    sample: the audit refuses anything that is not 256x256, and counting the
    rejects here is better than crashing halfway through a GPU run.
    """
    rootp = Path(root)
    paired_json = rootp / "paired.json"
    if not paired_json.is_file():
        raise SystemExit(
            f"{paired_json} not found. --make-sar1m-manifest reads the archive's own "
            f"paired.json; point --data-root at the extracted SAR-1M root.")
    with open(paired_json) as fh:
        paired = json.load(fh)

    rng = random.Random(seed)
    order = rng.sample(range(len(paired)), len(paired))
    kept, rejected, scanned = [], 0, 0
    for i in order:
        if n and len(kept) >= n:
            break
        e = paired[i]
        scanned += 1
        sp, ep = rootp / e["sar"], rootp / e["optical"]
        try:
            with Image.open(sp) as ims, Image.open(ep) as ime:
                ok = ims.size == ime.size == (256, 256)
        except OSError:
            ok = False
        if not ok:
            rejected += 1
            continue
        kept.append(e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for e in kept:
            stem = Path(e["sar"]).stem
            fh.write(json.dumps({
                "sample_id": f"sar1m/{stem}",
                "sar_path": e["sar"],
                "eo_path": e["optical"],
                "group_id": f"sar1m/{stem}",
                "split": "audit",
            }) + "\n")
    return {
        "manifest": str(out_path),
        "manifest_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "paired_json_total": len(paired),
        "sampled": len(kept),
        "scanned": scanned,
        "rejected_not_256x256_or_unreadable": rejected,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _summary_mean(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


@torch.no_grad()
def measure(ae, ds, modality: str, device, args) -> dict:
    """Round-trip one dataset through one autoencoder and score it.

    EO is encoded as RGB.  SAR is encoded one scalar band at a time, replicated to
    3 channels — no autoencoder here has a SAR-native input.  Every metric is
    computed per image on ``[0, 1]`` tensors, where the reconstruction is
    ``clamp(decode(encode(x)) * 0.5 + 0.5)`` and the reference is the source image
    divided by 255.
    """
    n_items = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    if n_items < 1:
        raise SystemExit("nothing to measure: the dataset resolved to 0 items")
    per: dict[str, list[float]] = {}
    per_band: dict[str, dict[str, list[float]]] = {}
    enl_first: list[float] = []
    hf_first: list[float] = []
    grad_first: list[float] = []
    fa = FIDAccumulator(device) if not args.no_fid else None
    fb = FIDAccumulator(device) if not args.no_fid else None
    band_names: list[str] = list(ds.bands)
    n_planes = 1 if modality == "eo" else len(band_names)
    n_scored = 0
    t0 = time.time()

    def _add(store: dict, key: str, v) -> None:
        store.setdefault(key, []).extend(
            v.detach().float().cpu().tolist() if isinstance(v, torch.Tensor) else [v])

    for start in range(0, n_items, args.batch_size):
        chunk = range(start, min(start + args.batch_size, n_items))
        eo, sar = [], []
        for i in chunk:
            e, s, band_names, _ = ds[i]
            eo.append(e)
            sar.append(s)
        eo = torch.stack(eo).to(device)
        sar = torch.stack(sar).to(device)

        if modality == "eo":
            planes = [("RGB", eo)]
        else:
            planes = [(band_names[b] if b < len(band_names) else f"ch{b}",
                       sar[:, b:b + 1].repeat(1, 3, 1, 1))
                      for b in range(sar.shape[1])]

        for bi, (band, x) in enumerate(planes):
            h, w = x.shape[-2:]
            f = ae.spatial_factor
            if h % f or w % f:
                raise SystemExit(
                    f"{h}x{w} is not divisible by the autoencoder stride {f}. Pick a "
                    f"crop that is a multiple of {f}; nothing is resized here.")
            gt = (x * 0.5 + 0.5).clamp(0.0, 1.0)
            rec = (ae.roundtrip(x).float() * 0.5 + 0.5).clamp(0.0, 1.0)

            vals = {
                "PSNR": psnr(rec, gt),
                "SSIM": ssim(rec, gt),
                "LPIPS": lpips_dist(rec, gt, standard_scaling=False),
                LPIPS_STD_KEY: lpips_dist(rec, gt, standard_scaling=True),
                PEARSON_KEY: pearson_global(rec, gt),
                HF_CORR_KEY: hf_corr(rec, gt),
            }
            if modality == "sar":
                vals[GRAD_CORR_KEY] = grad_corr(rec, gt)
            if min(h, w) >= MS_SSIM_MIN_PX:
                vals["MS-SSIM"] = ms_ssim(rec, gt)

            for k, v in vals.items():
                _add(per, k, v)
                if n_planes > 1:
                    _add(per_band.setdefault(band, {}), k, v)
            if fa is not None:
                fa.update(rec.cpu())
                fb.update(gt.cpu())
            n_scored += int(x.shape[0])

            if modality == "sar" and bi == 0:
                enl_first += [float(v) for v in enl_proxy(gt[:, :1]).cpu().tolist()]
                hf_first += [float(v) for v in vals[HF_CORR_KEY].cpu().tolist()]
                grad_first += [float(v) for v in vals[GRAD_CORR_KEY].cpu().tolist()]

        if args.verbose:
            done = min(start + args.batch_size, n_items)
            print(f"      {done}/{n_items} items "
                  f"({(time.time() - t0) / max(done, 1) * 1000:.0f} ms/item)", flush=True)

    out: dict = {k: _summary_mean(v) for k, v in per.items()}
    if fa is not None:
        ga, gb = fa.features, fb.features
        # scipy's sqrtm on the 2048x2048 covariance dominates wall clock here.
        print("      FID (2048-d sqrtm) ...", flush=True)
        out["FID"] = frechet_from_features(ga, gb)
        out["KID"] = kid(ga, gb, subset_size=args.kid_subset_size,
                         n_subsets=args.kid_subsets, seed=args.seed)
    out["_n"] = n_items
    out["_modality"] = modality
    out["_latent_channels"] = ae.latent_channels
    out["_spatial_factor"] = ae.spatial_factor
    # Latent channels per pixel of image — the compression axis the table compares.
    out["_compression"] = round(ae.latent_channels / (ae.spatial_factor**2), 4)
    out["_sec"] = round(time.time() - t0, 1)
    if n_planes > 1:
        out["_n_scored"] = n_scored
        out["_bands"] = {b: {k: _summary_mean(v) for k, v in m.items()}
                         for b, m in per_band.items()}

    # Speckle severity: per-image ENL tertiles on the first SAR band.  Low ENL =
    # strong speckle.  Ranking only; no calibrated ENL is claimed.
    if modality == "sar" and len(enl_first) >= 9:
        enl = np.asarray(enl_first, dtype=np.float64)
        q1, q2 = np.quantile(enl, [1 / 3, 2 / 3])
        tiers = np.digitize(enl, [q1, q2])  # 0 = lowest ENL = worst speckle
        by_tier = {}
        for t, tname in enumerate(["severe(low ENL)", "moderate", "mild(high ENL)"]):
            sel = tiers == t
            by_tier[tname] = {
                "n": int(sel.sum()),
                HF_CORR_KEY: _summary_mean(np.asarray(hf_first)[sel].tolist()),
                GRAD_CORR_KEY: _summary_mean(np.asarray(grad_first)[sel].tolist()),
            }
        out["_speckle_severity"] = {
            "band": band_names[0],
            "enl_proxy_median": float(np.median(enl)),
            "enl_proxy_tertile_edges": [float(q1), float(q2)],
            "by_tier": by_tier,
        }
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
_TABLE_METRICS = ["PSNR", "SSIM", "LPIPS", HF_CORR_KEY]
_HIGHER_IS_BETTER = {"PSNR", "SSIM", HF_CORR_KEY, PEARSON_KEY, "MS-SSIM", GRAD_CORR_KEY}


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def write_summary(results: dict, out_md: Path) -> None:
    """Human-readable companion to the JSON: one table per headline metric."""
    meta = results.get("_meta", {})
    rows = [k for k in results if not k.startswith("_")]
    present = [VAE_REGISTRY[n][0] for n in VAE_ORDER
               if any(VAE_REGISTRY[n][0] in results[k] for k in rows)]

    L: list[str] = []
    A = L.append
    A("# Autoencoder reconstruction ceilings\n")
    A("Every number is a frozen autoencoder's own round trip on the posterior "
      "MEAN (never a sample), so it is the deterministic latent bottleneck a "
      "generator lives inside. **These are ceilings: no generator built on an "
      "autoencoder can beat its own row.**\n")
    A("SAR has no native input on any of these autoencoders, so each scalar SAR "
      "band is replicated to RGB and encoded on its own. `HF corr.` is a global "
      "3x3-Laplacian high-pass correlation and is **not** SCC. `LPIPS` uses the "
      "`normalize=False` convention of the published ceiling tables; "
      f"`{LPIPS_STD_KEY}` is in the JSON alongside it and is about 0.05 apart. "
      "The two are not comparable.\n")
    if meta.get("run", {}).get("timestamp"):
        A(f"Generated {meta['run']['timestamp']} on `{meta['run'].get('device')}`.\n")

    for metric in _TABLE_METRICS:
        nd = 2 if metric == "PSNR" else 4
        A(f"\n## {metric}\n")
        A("| dataset | mod | res | " + " | ".join(present) + " |")
        A("|" + "---|" * (3 + len(present)))
        for key in rows:
            cells = []
            vals = []
            for lab in present:
                v = results[key].get(lab, {}).get(metric)
                vals.append(v if isinstance(v, (int, float)) else None)
            finite = [v for v in vals if v is not None and math.isfinite(v)]
            best = (max(finite) if metric in _HIGHER_IS_BETTER else min(finite)) \
                if finite else None
            for v in vals:
                s = _fmt(v, nd)
                cells.append(f"**{s}**" if best is not None and v == best else s)
            first = next((r for r in (results[key].get(lab) for lab in present)
                          if r), {})
            ds_name, _, mod = key.rpartition("/")
            A(f"| {first.get('_dataset', ds_name)} | {mod.upper()} | "
              f"{first.get('_resolution', '?')} | "
              + " | ".join(cells) + " |")
        A("")
    A("\n**Bold = best in row.** The autoencoders are **not rate-matched**: the "
      "`_compression` field in the JSON records latent channels per input pixel "
      "for each, and a codec that keeps more of them is expected to reconstruct "
      "better. What a ceiling table establishes is what each codec loses *at the "
      "operating point it is used at*, not which encoder is better per latent "
      "bit.\n")
    lp = meta.get("latent_path")
    if lp:
        A("\n## FLUX.2 latent-path integrity\n")
        A(f"- pack/unpack bitwise exact on all "
          f"{len(lp['pack_unpack']['cases'])} shape/dtype cases: "
          f"**{lp['pack_unpack']['all_bitwise_exact']}**")
        ni, si = lp["normalize_inverse"], lp["sandwich_identity"]
        A(f"- `inv_normalize(normalize(z))` max rel err "
          f"{ni['max_rel_err']:.3e} "
          f"({ni['max_rel_err'] / ni['float32_eps']:.2f} float32 ulp); the same "
          f"round trip in float64 gives {ni['max_rel_err_float64']:.3e} and "
          f"recovers the packed latent bitwise "
          f"(`{ni['float64_roundtrip_recovers_float32_latent_bitwise']}`)")
        A(f"- `decode(encode(x))` vs `decoder(mean(encoder(x)))`: max abs "
          f"{si['decode_encode_vs_decoder_encoder_mean_max_abs']:.3e}, i.e. "
          f"{si['decode_encode_vs_decoder_encoder_mean_psnr_db']:.1f} dB against "
          f"the [-1,1] data range — the entire footprint of the pack/BatchNorm "
          f"sandwich on the decoded image, far below every PSNR above")
        A(f"\n**VERDICT: the pack/BatchNorm round trip is "
          f"{'EXACT' if lp['passed'] else 'NOT EXACT'}.**\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _kv(values: list[str] | None, what: str, single: str | None) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` arguments; a bare value is allowed when
    exactly one ``--dataset`` is selected."""
    out: dict[str, str] = {}
    for v in values or []:
        if "=" in v:
            k, _, val = v.partition("=")
            out[k] = val
        elif single is not None:
            out[single] = v
        else:
            raise SystemExit(
                f"--{what} needs NAME=VALUE when more than one dataset is selected "
                f"(got {v!r})")
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="vae_recon_audit.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset", nargs="+", default=[],
        choices=sorted(DATASET_REGISTRY) + sorted(DATASET_ALIASES), metavar="NAME",
        help="Datasets to measure: " + ", ".join(sorted(DATASET_REGISTRY))
             + ". Each needs a --data-root.")
    ap.add_argument(
        "--vae", nargs="+", default=VAE_ORDER, choices=list(VAE_REGISTRY),
        metavar="NAME",
        help="Autoencoders, default all six in table order: "
             + ", ".join(VAE_ORDER)
             + ". Five are diffusers AutoencoderKL checkpoints (see "
               "vae_audit/README.md for the repo ids and their licences); flux2 is "
               "this repository's own packed/BN-normalised FLUX.2 codec.")
    ap.add_argument(
        "--modality", nargs="+", default=["eo", "sar"], choices=["eo", "sar"],
        metavar="MOD",
        help="Which side of each pair to round-trip (default: both). EO is the "
             "translation target; SAR is the condition and is replicated to 3 "
             "channels because no autoencoder here has a SAR input.")
    ap.add_argument(
        "--data-root", nargs="+", metavar="NAME=PATH",
        help="Dataset root, e.g. --data-root qxs-saropt=/data/QXS-SAROPT. A bare "
             "path is accepted when exactly one --dataset is selected.")
    ap.add_argument(
        "--file-list", nargs="+", metavar="NAME=PATH",
        help="Split list of EO paths relative to that dataset's root. Defaults to "
             "this repository's frozen splits/ lists for qxs-saropt and sar2opt. "
             "SpaceNet6 falls back to every PS-RGB tile under the root, which is "
             "the whole AOI and not any published split — say so if you quote it.")
    ap.add_argument(
        "--resolution", type=int, default=None, metavar="INT",
        help="Shorthand for --crop applied to every selected dataset. Loses to an "
             "explicit --crop for the same dataset.")
    ap.add_argument(
        "--crop", nargs="+", metavar="NAME=INT", default=None,
        help="Centre-crop size per dataset (defaults: qxs-saropt 256, sar2opt 512, "
             "spacenet6 512, sar1m 256). Images are NEVER resized: the crop offset "
             "is (n - crop) // 2, so SAR2Opt's 600 -> 512 lands at offset 44. Must "
             "be a multiple of the autoencoder stride (16 for FLUX.2, 8 otherwise).")
    ap.add_argument(
        "--flux2-weights", default="weights/ae.safetensors", metavar="PATH",
        help="FLUX.2 autoencoder: the single-file ae.safetensors, or a diffusers "
             "folder. Default weights/ae.safetensors — rebuild it with "
             "scripts/convert_flux2_ae.py (README, 'The autoencoder').")
    ap.add_argument(
        "--vae-root", "--zoo", default=None, metavar="DIR",
        help="Directory of locally downloaded autoencoders, one subfolder per "
             "published label (SD2.1/, SDXL/, SD3.0/, SD3.5/, FLUX.1/), each "
             "holding a vae/ folder. Without it, the HF repo ids are used.")
    ap.add_argument(
        "--vae-path", nargs="+", metavar="NAME=PATH_OR_REPO",
        help="Override one autoencoder's source, e.g. --vae-path sd21=/local/sd21.")
    ap.add_argument("--batch-size", type=int, default=8, metavar="N",
                    help="Items per forward pass (default: 8).")
    ap.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Score only the first N items, in list order (default 0 = all). Smoke "
             "tests only: FID and KID are set statistics and a truncated run is not "
             "comparable with a full-set one.")
    ap.add_argument("--no-fid", action="store_true",
                    help="Per-image metrics only; skips the FID/KID pass, whose "
                         "2048x2048 sqrtm dominates the runtime.")
    ap.add_argument("--kid-subset-size", type=int, default=500, metavar="N",
                    help="KID subset size (default 500, as published).")
    ap.add_argument("--kid-subsets", type=int, default=100, metavar="N",
                    help="Number of KID subsets (default 100, as published).")
    ap.add_argument("--sar1m-manifest", "--manifest", default=None, metavar="PATH",
                    help="JSONL manifest for --dataset sar1m. Build one with "
                         "--make-sar1m-manifest.")
    ap.add_argument("--sar1m-split", default=None, metavar="SPLIT",
                    help="Only use manifest rows whose 'split' equals this.")
    ap.add_argument(
        "--make-sar1m-manifest", action="store_true",
        help="Manifest-builder mode: read <data-root>/paired.json, take a seeded "
             "random sample of --n size-checked 256x256 pairs, write it to --out "
             "as JSONL, print its sha256, and exit. No GPU, no autoencoder.")
    ap.add_argument("--n", type=int, default=1000, metavar="N",
                    help="Pairs to sample in --make-sar1m-manifest mode "
                         "(default 1000).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Compute device (default: cuda when available, else cpu).")
    ap.add_argument("--seed", type=int, default=0, metavar="N",
                    help="Seed for the KID subset draw and the manifest sampler "
                         "(default 0, which is the seed the published KID rows "
                         "used). Nothing else in this script is random: the "
                         "posterior mean is deterministic and every crop is a "
                         "centre crop.")
    ap.add_argument("--out", default="results/vae_ceiling.json", metavar="PATH",
                    help="Output JSON (default results/vae_ceiling.json); a path "
                         "without a .json suffix is treated as a directory and gets "
                         "vae_ceiling.json inside it. An existing file is MERGED "
                         "into, so a run that adds one autoencoder does not drop "
                         "rows an earlier run measured. A readable summary is "
                         "written beside it as .md. (In --make-sar1m-manifest mode "
                         "--out is the JSONL manifest to write.)")
    ap.add_argument("--verbose", action="store_true", help="Per-batch progress.")

    args = ap.parse_args(argv)
    args.dataset = [DATASET_ALIASES.get(d, d) for d in args.dataset]
    # A bare path is unambiguous when only one dataset is in play — including
    # manifest-builder mode, which only ever concerns SAR-1M.
    single = args.dataset[0] if len(args.dataset) == 1 else None
    if single is None and args.make_sar1m_manifest and not args.dataset:
        single = "sar1m"
    args.data_root = _kv(args.data_root, "data-root", single)
    args.file_list = _kv(args.file_list, "file-list", single)
    args.vae_path = _kv(args.vae_path, "vae-path", None)
    crop = {d: args.resolution for d in args.dataset} if args.resolution else {}
    crop.update({k: int(v) for k, v in _kv(args.crop, "crop", single).items()})
    args.crop = crop
    if not args.make_sar1m_manifest and not args.dataset:
        ap.error("nothing to do: pass --dataset, or --make-sar1m-manifest")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.make_sar1m_manifest:
        root = args.data_root.get("sar1m") or next(iter(args.data_root.values()), None)
        if root is None:
            raise SystemExit("--make-sar1m-manifest needs --data-root <SAR-1M root>")
        info = make_sar1m_manifest(root, args.n, args.seed, Path(args.out))
        print(json.dumps(info, indent=2))
        print(f"\n  wrote {info['manifest']} — cite it by its sha256; it is the only "
              f"record of which items were scored.")
        return 0

    device = torch.device(args.device)
    out_path = Path(args.out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "vae_ceiling.json"
    # Merge, so a run that adds autoencoders does not drop earlier rows.
    results: dict = json.loads(out_path.read_text()) if out_path.is_file() else {}

    datasets = {}
    for name in args.dataset:
        ds, label, crop = build_dataset(name, args)
        datasets[name] = (ds, label, crop)
        print(f"  {label:<12} {len(ds)} items @ {crop}px, bands {ds.bands}", flush=True)

    latent_path = None
    t0 = time.time()
    for vae_name in args.vae:
        try:
            ae, label, source = build_ae(vae_name, args, device)
        except Exception as exc:  # noqa: BLE001 — a missing codec must not kill the run
            print(f"  SKIP {vae_name}: {type(exc).__name__}: {str(exc)[:160]}",
                  flush=True)
            continue
        print(f"\n  {label}  ch={ae.latent_channels} f={ae.spatial_factor}  "
              f"<- {source}", flush=True)

        if vae_name == "flux2":
            # Fatal, and before any metric: if the pack/BatchNorm sandwich is not
            # the identity, every FLUX.2 number below would measure that instead
            # of the autoencoder.
            first = datasets[args.dataset[0]][0]
            probe = torch.stack([first[i][0] for i in range(min(4, len(first)))])
            latent_path = verify_latent_path(ae, probe.to(device))
            print(f"  latent-path integrity: passed={latent_path['passed']}",
                  flush=True)
            if not latent_path["passed"]:
                print(json.dumps(latent_path, indent=2))
                raise SystemExit(
                    "FLUX.2 pack/normalize round trip is NOT exact; aborting before "
                    "any number is produced.")

        for name in args.dataset:
            ds, ds_label, crop = datasets[name]
            for modality in args.modality:
                key = f"{name}/{modality}"
                rec = measure(ae, ds, modality, device, args)
                rec["_resolution"] = crop
                rec["_dataset"] = ds_label
                results.setdefault(key, {})[label] = rec
                fid = f"  FID {rec['FID']:8.4f}" if "FID" in rec else ""
                print(f"  {key:<20} {label:<8} PSNR {rec['PSNR']:7.4f}  "
                      f"SSIM {rec['SSIM']:.4f}  LPIPS {rec['LPIPS']:.4f}"
                      f"{fid}  ({rec['_sec']}s)", flush=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(results, indent=2))
        del ae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import platform
    from datetime import datetime, timezone

    results["_meta"] = {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": (torch.cuda.get_device_name(device) if device.type == "cuda"
                       else platform.processor() or "cpu"),
            "wall_seconds": round(time.time() - t0, 1),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "seed": args.seed,
            "argv": [os.path.basename(sys.argv[0])] + list(sys.argv[1:]),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "autoencoders": {VAE_REGISTRY[n][0]: {"name": n, "hf_repo": VAE_REGISTRY[n][1],
                                              "subfolder": VAE_REGISTRY[n][2]}
                         for n in args.vae},
        "datasets": {n: {"label": DATASET_REGISTRY[n][0],
                         "n_items": len(datasets[n][0]),
                         "crop": datasets[n][2],
                         "bands": datasets[n][0].bands}
                     for n in args.dataset},
        "lpips_convention": {
            "LPIPS": "lpips.LPIPS(net='vgg') on [0,1] with normalize=False — the "
                     "convention of the published ceiling tables",
            LPIPS_STD_KEY: "[0,1] mapped to [-1,1] first — the convention of "
                           "scripts/evaluate.py; about 0.05 apart, not comparable",
        },
        "latent_path": latent_path,
    }
    out_path.write_text(json.dumps(results, indent=2))
    write_summary(results, out_path.with_suffix(".md"))
    print(f"\n  wrote {out_path} and {out_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
