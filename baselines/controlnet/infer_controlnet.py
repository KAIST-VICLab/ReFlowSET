#!/usr/bin/env python
"""SAR->EO inference with a trained ControlNet + SD 2.1-base over a directory of SAR images.

Writes generated EO images (same basenames) to --out_dir. Skips already-generated files,
so the job is resumable.
"""
import argparse
import os

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

SD_BASE = "Manojb/stable-diffusion-2-1-base"
PROMPT = "electro-optical image"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlnet_dir", required=True, help="dir with trained ControlNet (config.json + safetensors)")
    ap.add_argument("--sar_dir", required=True, help="directory of test SAR images")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE, controlnet=controlnet, torch_dtype=torch.bfloat16, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    names = sorted(f for f in os.listdir(args.sar_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    todo = [n for n in names if not os.path.exists(os.path.join(args.out_dir, n))]
    print(f"{len(names)} test images, {len(todo)} to generate")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        conds = [
            Image.open(os.path.join(args.sar_dir, n)).convert("RGB").resize(
                (args.resolution, args.resolution), Image.BILINEAR
            )
            for n in batch
        ]
        with torch.no_grad():
            images = pipe(
                [PROMPT] * len(batch),
                image=conds,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images
        for n, im in zip(batch, images):
            im.save(os.path.join(args.out_dir, n))
        print(f"[{i + len(batch)}/{len(todo)}] done", flush=True)


if __name__ == "__main__":
    main()
