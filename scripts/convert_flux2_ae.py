#!/usr/bin/env python3
"""Build the ReFlowSET autoencoder file from the Apache-2.0 FLUX.2-klein-base-4B VAE.

ReFlowSET's latent endpoint is the FLUX.2 autoencoder, frozen and never fine-tuned.
Black Forest Labs ship the same autoencoder in two repositories:

  black-forest-labs/FLUX.2-dev            ae.safetensors   flux-non-commercial-license
  black-forest-labs/FLUX.2-klein-base-4B  vae/...           apache-2.0, ungated

They are the same weights under two serialisations (BFL key names vs diffusers key
names, fp32 vs bf16, and eight tensors stored as 1x1 convolutions rather than
linears). ReFlowSET is released against the **Apache-2.0** copy.

This script needs only the Apache file and the key table shipped beside it, so a
downstream user never has to touch the non-commercial repository.

Usage
-----
  huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
      vae/diffusion_pytorch_model.safetensors --local-dir klein4b
  python convert_flux2_ae.py \
      --src klein4b/vae/diffusion_pytorch_model.safetensors \
      --key-map flux2_ae_key_map.json \
      --out ae.safetensors
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True,
                    help="FLUX.2-klein-base-4B vae/diffusion_pytorch_model.safetensors")
    ap.add_argument("--key-map", required=True, help="flux2_ae_key_map.json")
    ap.add_argument("--out", required=True, help="destination ae.safetensors")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    detail = json.loads(pathlib.Path(args.key_map).read_text())["detail"]
    dtype = getattr(torch, args.dtype)

    out: dict[str, torch.Tensor] = {}
    with safe_open(args.src, framework="pt") as f:
        have = set(f.keys())
        missing = sorted({d["src"] for d in detail.values()} - have)
        if missing:
            raise SystemExit(f"source is missing {len(missing)} mapped tensors, "
                             f"first: {missing[:3]}")
        for dst, d in detail.items():
            t = f.get_tensor(d["src"])
            want = tuple(d["shape"])
            if tuple(t.shape) != want:
                # BFL stores eight 512x512 projections as 1x1 convolutions.
                if t.numel() != int(torch.tensor(want).prod()):
                    raise SystemExit(f"{dst}: cannot reshape {tuple(t.shape)} -> {want}")
                t = t.reshape(want)
            out[dst] = t.to(dtype).contiguous()

    # The latent normaliser is a BatchNorm2d(affine=False) held in eval() with
    # shipped running statistics; the step counter is never read. Publish it as 0
    # rather than carrying the 400000 that the non-commercial file recorded.
    out["bn.num_batches_tracked"] = torch.zeros((), dtype=torch.int64)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out, metadata={
        "source_repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "source_file": "vae/diffusion_pytorch_model.safetensors",
        "source_license": "apache-2.0",
        "layout": "BFL Flux2AE key names",
    })
    print(f"wrote {args.out}: {len(out)} tensors, dtype={args.dtype}")


if __name__ == "__main__":
    main()
