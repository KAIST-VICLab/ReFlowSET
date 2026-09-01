#!/usr/bin/env python3
"""Train a ReFlowSET arm from scratch.

    python scripts/train.py --config configs/qxs_saropt.yaml
    python scripts/train.py --config configs/sar2opt.yaml

Implements the recipe both released checkpoints were trained with:

  * linear flow bridge, Design B — ``z_t = (1 - t) * eps + t * z_e`` with
    ``eps ~ N(0, I)``, ``t ~ U(0, 1)`` clamped to ``[0, 1 - 1e-3]``, velocity
    target ``u* = z_e - eps``, and the SAR latent as the conditioning stream;
  * both latents from the **same frozen** FLUX.2 autoencoder (posterior mean),
    with the 1-channel SAR replicated to 3 at the model boundary;
  * classifier-free guidance trained by zeroing the SAR latent on 10 % of rows,
    per row, which is the same all-zero null the sampler uses;
  * plain velocity MSE;
  * REPA — the block-8 EO activations, through a 3-layer projector, matched
    patch-wise by cosine to a frozen DINOv3 ViT-L/16 LVD-1689M teacher applied
    to the **clean** EO image, gated by ``w(t) = t`` per sample, weight 0.5;
  * AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0, gradient clip 1.0,
    lr 5e-4 cosine to 1e-6 after a 1,000-step linear warmup;
  * EMA 0.9999 (with the usual ``(1 + step) / (10 + step)`` warmup, without
    which 60 % of the random initialisation is still in the EMA at step 5,000);
  * bf16 mixed precision, frozen modules in bf16, latents carried in float32;
  * seed 2024.

The released weights are the **EMA**, and that is what every checkpoint written
here contains: each one is a complete, loadable pipeline folder that
``scripts/translate.py --checkpoint`` accepts directly.

DINOv3 is not redistributed with this repository. Clone
https://github.com/facebookresearch/dinov3, accept its licence, download the
ViT-L/16 LVD-1689M weights, and set ``repa_teacher.repo`` and
``repa_teacher.weights`` in the config. Training without the teacher is
supported — ``--lambda-repa 0`` — but it is an ablation, not the released
recipe.

NOT IMPLEMENTED
---------------
This trainer is deliberately narrower than the research code that produced the
checkpoints. What is missing, precisely:

1. **Distributed training.** Single process, single device. Both released arms
   ran on one GPU, so this costs no fidelity, but there is no multi-GPU path.
2. **Micro-batch autotuning.** Set ``train.micro_batch_size`` yourself; it must
   divide ``train.global_batch_size``. The as-run values are in the configs.
3. **Resume.** No optimiser, EMA or dataloader state is written, so an
   interrupted run restarts from step 0.
4. **The training-subset monitoring loop.** The reference logged a loss on a
   fixed subset of the *training* set every 5,000 steps. It was never a
   validation split and was never used to select a checkpoint — the released
   weights are the last step of a fixed-length schedule — so nothing about the
   recipe depends on it, and it is not reproduced here.
5. **Bit-exact reproduction of the released tensors.** The recipe is the same;
   the data-loading RNG stream and micro-batch order are not, so a rerun lands
   near the released checkpoints, not on them.
6. **Every disabled research branch.** The beta-NLL flow loss and its ``logvar``
   head, speckle augmentation, the speckle-consistency loss, the pixel/LPIPS
   loss, metadata conditioning and the trainable native-SAR encoder are all off
   in both released arms and none of them is reimplemented here.
   ``final_layer.logvar_proj`` therefore keeps its zero initialisation, exactly
   as it did in the released checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from reflowset import (  # noqa: E402
    AutoencoderFlux2,
    FlowBridgeScheduler,
    ReFlowSETPipeline,
    ReFlowSETTransformer2DModel,
)

#: The bridge time is clamped away from 1 so the velocity target stays finite.
BRIDGE_T_EPS = 1e-3
#: Input normalisation is a property of the DINOv3 checkpoint, not a global
#: default: the LVD-1689M weights are ImageNet-normalised. The satellite
#: SAT-493M checkpoint wants different constants and was not the teacher either
#: released arm used.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: The released ViT-L/16 LVD-1689M file. The hash suffix is load-bearing: the
#: DINOv3 hub builder parses it to choose an architecture flag.
DINOV3_LVD_HASH = "8aa4cbdd"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sar_path_for(eo_rel: str, rule: str) -> str:
    """Derive the SAR path from an EO list entry. Only the EO side is stored."""
    if rule == "qxs-saropt":
        return eo_rel.replace("opt_256_oc_0.2", "sar_256_oc_0.2")
    if rule == "sar2opt":
        head, _, tail = eo_rel.partition("/")
        if head not in ("trainB", "testB"):
            raise ValueError(
                f"SAR2Opt list entries must start with 'trainB/' or 'testB/', got {eo_rel!r}")
        return f"{head[:-1]}A/{tail}"
    raise ValueError(f"unknown data.path_rule {rule!r}; expected 'qxs-saropt' or 'sar2opt'")


class PairedSARDataset(Dataset):
    """SAR/EO pairs read exactly the way the released arms read them.

    SAR is opened with no mode conversion and collapsed to its single amplitude
    channel; EO is opened forced to RGB. One random crop window is drawn per
    item and applied to both, and both are mapped to ``[-1, 1]`` by
    ``x / 127.5 - 1`` with no per-image or per-dataset statistics. Nothing is
    ever resized. (At test time the window is instead the exact centre crop,
    which the inference pipeline applies.)
    """

    def __init__(self, root: Path, list_file: Path, rule: str, crop: int, augment) -> None:
        entries = [line for line in Path(list_file).read_text().splitlines() if line]
        if not entries:
            raise SystemExit(f"empty split list: {list_file}")
        self.eo_paths = [root / e for e in entries]
        self.sar_paths = [root / sar_path_for(e, rule) for e in entries]
        missing = next((p for p in (self.eo_paths[0], self.sar_paths[0]) if not p.exists()), None)
        if missing is not None:
            raise SystemExit(
                f"{missing} does not exist — is data.root pointing at the dataset root?")
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.eo_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with Image.open(self.sar_paths[index]) as im:
            # Reused from the pipeline so training and inference cannot drift:
            # mode guard, alpha drop, replicated-RGB collapse.
            sar = ReFlowSETPipeline._sar_hwc(im)
        with Image.open(self.eo_paths[index]) as im:
            eo = np.asarray(im.convert("RGB"), np.float32)

        if sar.shape[:2] != eo.shape[:2]:
            raise ValueError(
                f"SAR {sar.shape[:2]} and EO {eo.shape[:2]} disagree on size for "
                f"{self.eo_paths[index]}; there is no resampling path")
        height, width = sar.shape[:2]
        if height < self.crop or width < self.crop:
            raise ValueError(
                f"{self.eo_paths[index]} is {height}x{width}, smaller than the "
                f"{self.crop} crop; ReFlowSET never upscales")
        top = int(torch.randint(height - self.crop + 1, ()))
        left = int(torch.randint(width - self.crop + 1, ()))
        sar = sar[top:top + self.crop, left:left + self.crop]
        eo = eo[top:top + self.crop, left:left + self.crop]

        sar = torch.from_numpy(np.ascontiguousarray(sar.transpose(2, 0, 1))) / 127.5 - 1.0
        eo = torch.from_numpy(np.ascontiguousarray(eo.transpose(2, 0, 1))) / 127.5 - 1.0
        sar, eo = d4_augment(sar, eo, self.augment)
        return {"sar": sar, "eo": eo}


def d4_augment(sar: Tensor, eo: Tensor, cfg) -> tuple[Tensor, Tensor]:
    """One element of the 8-element dihedral group, applied to both modalities.

    The draw order — horizontal flip, vertical flip, then a quarter-turn count —
    is part of the recipe, and a disabled flip consumes no randomness.
    """
    flip_h = bool(cfg.hflip) and float(torch.rand(())) < 0.5
    flip_v = bool(cfg.vflip) and float(torch.rand(())) < 0.5
    turns = int(torch.randint(4, ())) if bool(cfg.rot90) else 0
    if flip_h:
        sar, eo = torch.flip(sar, dims=(-1,)), torch.flip(eo, dims=(-1,))
    if flip_v:
        sar, eo = torch.flip(sar, dims=(-2,)), torch.flip(eo, dims=(-2,))
    if turns:
        sar = torch.rot90(sar, turns, dims=(-2, -1))
        eo = torch.rot90(eo, turns, dims=(-2, -1))
    return sar, eo


# --------------------------------------------------------------------------- #
# REPA
# --------------------------------------------------------------------------- #
class RepaProjector(nn.Module):
    """DiT hidden state -> DINOv3 patch space. Training only; never released.

    The output layer is **not** zero-initialised. Zero-initialising it and then
    taking a cosine distance gives a Jacobian of ``1 / eps`` at the origin,
    which in the predecessor of this work produced a measured 2530x gradient
    spike concentrated entirely in this projector.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class DinoV3Teacher(nn.Module):
    """Frozen DINOv3 ViT-L/16 patch-token extractor (LVD-1689M).

    The weights are **not** redistributed with ReFlowSET and are never loaded at
    inference. CLS and the register tokens are excluded by the backbone's own
    ``x_norm_patchtokens`` slice, so the returned token grid is exactly
    ``(H/16) * (W/16)`` — which is also the packed latent grid the DiT works on.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def load(cls, repo: str | None, weights: str | None, variant: str,
             device: torch.device, dtype: torch.dtype) -> "DinoV3Teacher":
        hint = (
            "ReFlowSET does not redistribute DINOv3. Clone "
            "https://github.com/facebookresearch/dinov3, accept its licence, download "
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth, and set repa_teacher.repo and "
            "repa_teacher.weights in the config. To train without the teacher instead, "
            "pass --lambda-repa 0 (an ablation, not the released recipe)."
        )
        if variant != "lvd":
            raise SystemExit(
                f"repa_teacher.variant={variant!r} is not supported. Both released arms "
                "used the LVD-1689M ViT-L/16 checkpoint; the SAT-493M one needs different "
                "input normalisation and was never trained against.")
        if not repo or not weights:
            raise SystemExit(f"repa_teacher.repo and repa_teacher.weights are unset.\n{hint}")
        repo_path, weights_path = Path(repo).expanduser(), Path(weights).expanduser()
        if not (repo_path / "dinov3").is_dir():
            raise SystemExit(f"no importable 'dinov3' package under {repo_path}.\n{hint}")
        if not weights_path.is_file():
            raise SystemExit(f"DINOv3 weights not found at {weights_path}.\n{hint}")
        if DINOV3_LVD_HASH not in weights_path.name:
            raise SystemExit(
                f"{weights_path.name} is not the released LVD-1689M ViT-L/16 file. Its "
                f"'-{DINOV3_LVD_HASH}.pth' suffix must be kept: the DINOv3 hub builder "
                "parses that hash to pick an architecture flag, so a renamed file loads "
                "as a subtly different model.")

        sys.path.insert(0, str(repo_path.resolve()))
        import dinov3.hub.backbones as hub  # noqa: PLC0415

        # Pass the path even though nothing is downloaded: the builder reads the
        # filename hash to set untie_global_and_local_cls_norm.
        backbone = hub.dinov3_vitl16(pretrained=False, weights=str(weights_path.resolve()))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        teacher = cls(backbone)
        teacher.backbone.to(device=device, dtype=dtype)
        teacher.to(device=device)
        teacher.eval().requires_grad_(False)
        return teacher

    @torch.no_grad()
    def forward(self, eo01: Tensor) -> Tensor:
        """``[B, 3, H, W]`` in [0, 1] -> ``[B, P, 1024]`` float32 patch tokens."""
        x = (eo01.float() - self.mean) / self.std
        dtype = next(self.backbone.parameters()).dtype
        out = self.backbone.forward_features(x.to(dtype))
        return out["x_norm_patchtokens"].float()


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def initialize_weights(model: ReFlowSETTransformer2DModel) -> None:
    """From-scratch initialisation: Xavier trunk, AdaLN-Zero heads.

    The timestep embedder is normal-initialised rather than zeroed: zeroing it
    makes ``vec == 0``, hence ``silu(vec) == 0``, which starves every AdaLN
    modulation weight of gradient and the conditioning path never trains.
    """
    def basic(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    model.apply(basic)
    nn.init.normal_(model.time_in.in_layer.weight, std=0.02)
    nn.init.normal_(model.time_in.out_layer.weight, std=0.02)
    for block in model.blocks:
        nn.init.constant_(block.modulation.lin.weight, 0)
        nn.init.constant_(block.modulation.lin.bias, 0)
    for double in model.double_stream:
        for tower in (double.eo, double.sar):
            nn.init.constant_(tower.modulation.lin.weight, 0)
            nn.init.constant_(tower.modulation.lin.bias, 0)
    for head in (model.final_layer.adaLN, model.final_layer.proj,
                 model.final_layer.logvar_proj):
        nn.init.constant_(head.weight, 0)
        nn.init.constant_(head.bias, 0)


def load_autoencoder(spec: str, dtype: torch.dtype) -> AutoencoderFlux2:
    """Load the frozen FLUX.2 autoencoder from a single file or a diffusers folder."""
    path = Path(spec).expanduser()
    if path.is_dir():
        ae = AutoencoderFlux2.from_pretrained(path, torch_dtype=dtype)
    elif path.is_file():
        ae = AutoencoderFlux2.from_single_file(path, torch_dtype=dtype)
    else:
        raise SystemExit(
            f"model.ae={spec!r} is neither a file nor a directory. Build the autoencoder "
            "with scripts/convert_flux2_ae.py, or point this at the vae/ subfolder of a "
            "released ReFlowSET checkpoint.")
    return ae.eval().requires_grad_(False)


def cosine_warmup(warmup: int, total: int, min_ratio: float):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA over parameters and buffers alike."""
    source = model.state_dict()
    for key, value in ema.state_dict().items():
        other = source[key]
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(other.detach().to(value.device, value.dtype), alpha=1 - decay)
        else:
            value.copy_(other.detach().to(value.device, value.dtype))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="  python scripts/train.py --config configs/qxs_saropt.yaml",
    )
    ap.add_argument("--config", required=True, type=Path,
                    help="Arm config: configs/qxs_saropt.yaml or configs/sar2opt.yaml.")
    ap.add_argument("--lambda-repa", type=float, default=None,
                    help="Override loss.lambda_repa. 0 trains without the DINOv3 teacher, "
                         "which needs no DINOv3 download but is an ablation, not the "
                         "released recipe (which uses 0.5).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Compute device (default: cuda when available, else cpu).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.lambda_repa is not None:
        cfg.loss.lambda_repa = args.lambda_repa

    device = torch.device(args.device)
    seed = int(cfg.train.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    amp_dtype = {"bf16": torch.bfloat16, "fp32": None}.get(str(cfg.train.mixed_precision))
    if amp_dtype is None and str(cfg.train.mixed_precision) != "fp32":
        raise SystemExit("train.mixed_precision must be 'bf16' (as released) or 'fp32'")
    frozen_dtype = amp_dtype or torch.float32

    global_batch = int(cfg.train.global_batch_size)
    micro_batch = int(cfg.train.micro_batch_size)
    if global_batch % micro_batch:
        raise SystemExit(
            f"train.micro_batch_size ({micro_batch}) must divide "
            f"train.global_batch_size ({global_batch})")
    accum = global_batch // micro_batch

    # ---- data ----
    list_file = Path(cfg.data.train_list)
    if not list_file.is_absolute():
        list_file = REPO / list_file
    dataset = PairedSARDataset(
        root=Path(cfg.data.root).expanduser(),
        list_file=list_file,
        rule=str(cfg.data.path_rule),
        crop=int(cfg.data.resolution),
        augment=cfg.aug,
    )
    loader = DataLoader(
        dataset, batch_size=micro_batch, shuffle=True, drop_last=True,
        num_workers=int(cfg.data.num_workers), pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

    # ---- models ----
    model_cfg = {k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
                 if k != "ae"}
    model_cfg["axes_dim"] = tuple(model_cfg["axes_dim"])
    model = ReFlowSETTransformer2DModel(**model_cfg)
    initialize_weights(model)
    model.to(device).train()

    ae = load_autoencoder(str(cfg.model.ae), frozen_dtype).to(device)
    # Deepcopied before the REPA hook is attached, so the EMA copy carries only
    # the weights that get released.
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    lambda_repa = float(cfg.loss.lambda_repa)
    repa_block = int(cfg.loss.repa_block)
    projector, teacher, tapped = None, None, {}
    if lambda_repa != 0.0:
        if not 1 <= repa_block <= int(cfg.model.double_blocks):
            raise SystemExit(
                f"loss.repa_block={repa_block} must lie in [1, model.double_blocks="
                f"{cfg.model.double_blocks}]. The tap reads the EO stream leaving a "
                "double-stream block; a tap inside the single-stream stack is not "
                "implemented here (both released arms use 8).")
        projector = RepaProjector(int(cfg.model.hidden_size), int(cfg.loss.repa_dim)).to(device)
        model.double_stream[repa_block - 1].register_forward_hook(
            lambda _module, _inputs, output: tapped.__setitem__("h", output[0]))
        teacher = DinoV3Teacher.load(
            cfg.repa_teacher.repo, cfg.repa_teacher.weights, str(cfg.repa_teacher.variant),
            device, frozen_dtype)

    # ---- optimiser ----
    steps = int(cfg.train.steps)
    lr = float(cfg.train.lr)
    parameters = list(model.parameters()) + (list(projector.parameters()) if projector else [])
    optimizer = torch.optim.AdamW(
        parameters, lr=lr, betas=tuple(cfg.train.betas),
        eps=float(cfg.train.eps), weight_decay=float(cfg.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_warmup(int(cfg.train.warmup_steps), steps,
                      min(1.0, float(cfg.train.min_lr) / lr)))

    output_dir = Path(cfg.train.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.resolved.yaml")

    print(f"[train] {cfg.name}: {steps} steps, global batch {global_batch} "
          f"({micro_batch} x {accum}), lr {lr:g} {cfg.train.lr_schedule} to "
          f"{float(cfg.train.min_lr):g} after {int(cfg.train.warmup_steps)} warmup steps")
    print(f"[train] resolution {int(cfg.data.resolution)}, {len(dataset)} training images, "
          f"seed {seed}, {cfg.train.mixed_precision}, device {device}")
    print(f"[train] lambda_repa {lambda_repa}"
          + ("" if teacher is not None else "  (no DINOv3 teacher — this is an ablation)"))
    print(f"[model] transformer {sum(p.numel() for p in model.parameters()):,} params"
          + (f", projector {sum(p.numel() for p in projector.parameters()):,} (training only)"
             if projector else ""))

    cond_drop_p = float(cfg.bridge.cond_drop_p)
    grad_clip = float(cfg.train.max_grad_norm)
    ema_decay = float(cfg.train.ema_decay)
    log_every, ckpt_every = int(cfg.train.log_every), int(cfg.train.ckpt_every)

    def batches():
        while True:
            yield from loader

    stream = batches()
    window_loss, window_start = 0.0, time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logs = {"flow": 0.0, "repa": 0.0, "total": 0.0}

        for _ in range(accum):
            batch = next(stream)
            eo = batch["eo"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)

            with torch.no_grad():
                # Both endpoints come from the same frozen encoder; the latent
                # is carried in float32 whatever precision the encoder ran at.
                z_e = ae.encode(eo.to(frozen_dtype)).float()
                z_s = ae.encode(sar.repeat(1, 3, 1, 1).clamp(-1, 1).to(frozen_dtype)).float()
                tokens = teacher((eo * 0.5 + 0.5).clamp(0, 1)) if teacher is not None else None

            t = torch.rand(z_e.shape[0], device=device).clamp(0.0, 1.0 - BRIDGE_T_EPS)
            eps = torch.randn_like(z_e)
            t_b = t.view(-1, 1, 1, 1)
            z_t = (1.0 - t_b) * eps + t_b * z_e
            u_star = z_e - eps  # constant along the path at sigma_b = 0

            if cond_drop_p > 0.0:
                keep = (torch.rand(z_s.shape[0], device=device) >= cond_drop_p)
                z_s = z_s * keep.to(z_s.dtype).view(-1, 1, 1, 1)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                velocity = model(z_t, t, z_s, return_dict=False)[0]
                loss = F.mse_loss(velocity.float(), u_star)
                logs["flow"] += float(loss.detach()) / accum

                if projector is not None:
                    projected = projector(tapped["h"])
                    if projected.shape[1] != tokens.shape[1]:
                        raise SystemExit(
                            f"REPA token mismatch: the DiT gives {projected.shape[1]} tokens "
                            f"and DINOv3 gives {tokens.shape[1]} for a "
                            f"{eo.shape[-1]}px image. The packed latent grid (H/16) must "
                            "equal the ViT-L/16 patch grid.")
                    # Cosine distance per patch, averaged per sample, then gated
                    # by that sample's own w(t) = t. Reducing to a scalar first
                    # and multiplying by mean(w) is a different quantity.
                    per_patch = 1.0 - F.cosine_similarity(
                        projected.float(), tokens.detach(), dim=-1, eps=1e-6)
                    per_sample = per_patch.mean(dim=1)
                    loss = loss + lambda_repa * (t * per_sample).mean()
                    logs["repa"] += float(per_sample.detach().mean()) / accum

            logs["total"] += float(loss.detach()) / accum
            (loss / accum).backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, grad_clip))
        optimizer.step()
        scheduler.step()
        # Warm the decay in: the EMA starts as a copy of the random
        # initialisation, and a flat 0.9999 leaves 60 % of it in the weights at
        # step 5,000 — every checkpoint would be a near-random model blended
        # with the trained one.
        ema_update(ema_model, model, min(ema_decay, (1.0 + step) / (10.0 + step)))

        window_loss += logs["total"]
        if step % log_every == 0:
            elapsed = time.time() - window_start
            print(f"[train] step={step} lr={scheduler.get_last_lr()[0]:.3g} "
                  f"loss={window_loss / log_every:.4f} flow={logs['flow']:.4f} "
                  f"repa={logs['repa']:.4f} gnorm={grad_norm:.3f} "
                  f"{log_every * global_batch / elapsed:.1f} img/s", flush=True)
            window_loss, window_start = 0.0, time.time()

        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(output_dir, step, ema_model, ae, cfg)

    print(f"[done] {steps} steps -> {output_dir}")
    return 0


def save_checkpoint(output_dir: Path, step: int, ema_model: ReFlowSETTransformer2DModel,
                    ae: AutoencoderFlux2, cfg) -> None:
    """Write the EMA weights as a complete, loadable pipeline folder.

    The released checkpoints are the EMA, not the live optimiser copy, and the
    two genuinely differ. ``scripts/translate.py --checkpoint <this folder>``
    runs it directly.
    """
    destination = output_dir / f"step_{step:07d}"
    ReFlowSETPipeline(
        transformer=ema_model,
        vae=ae,
        scheduler=FlowBridgeScheduler(t_end=float(cfg.bridge.t_end)),
    ).save_pretrained(destination)
    print(f"[ckpt] {destination}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
