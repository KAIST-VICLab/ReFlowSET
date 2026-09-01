#!/usr/bin/env python3
"""Translate a folder of SAR images to EO with ReFlowSET.

Reproduces the prediction dumps behind the paper's main table: NFE 50,
classifier-free guidance 1.5, seed 2024, float32, one image per call. The
output PNG stem is the input stem, so `scripts/evaluate.py` can pair the two
folders against the ground truth by name.

Four properties of the reported numbers are baked in rather than exposed as
options, because changing any of them changes the numbers:

  * `--denorm standard` — the decoded image is mapped to [0, 1] by
    `x * 0.5 + 0.5`, never by C-DiffSET's `x + 0.5` (a 2x contrast stretch that
    reads ~5.6 dB higher and invalidates any table it is mixed into). This
    happens inside the pipeline.
  * one image per call, so the per-image PSNR the evaluator computes is
    comparable with every retrained baseline.
  * a generator re-seeded before every image, so all test items start from the
    same noise draw. That is a property of the published numbers, not an
    accident; per-image noise will not reproduce them.
  * the input is centre-cropped to the arm's training resolution and **never**
    resized. SAR2Opt's 600x600 tiles become the central 512 at offset 44; QXS
    is natively 256 and the crop is a no-op.

Note on reproducibility across devices: the initial noise is drawn on the
compute device, so a CPU run and a CUDA run produce different images from the
same seed. The paper's dumps were drawn on CUDA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflowset import ReFlowSETPipeline  # noqa: E402

#: sar2opt ships .jpg, QXS-SAROPT .png.
SUFFIXES = {".png", ".jpg", ".jpeg"}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Reproduce the paper's dumps:\n"
               "  python scripts/translate.py --checkpoint KAIST-VICLab/ReFlowSET \\\n"
               "      --subfolder qxs-saropt --sar-dir DATA/sar_256_oc_0.2 \\\n"
               "      --out-dir results/eo --num-inference-steps 50 \\\n"
               "      --guidance-scale 1.5 --seed 2024",
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Hub id or local directory of a ReFlowSET checkpoint.")
    ap.add_argument("--subfolder", default=None,
                    help="Arm inside the checkpoint: 'qxs-saropt' or 'sar2opt'. "
                         "Omit when --checkpoint already points at one arm.")
    ap.add_argument("--sar-dir", required=True, type=Path,
                    help="Folder of SAR images (.png/.jpg/.jpeg), read non-recursively.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination folder; one PNG per input, same stem.")
    ap.add_argument("--num-inference-steps", type=int, default=50,
                    help="NFE, the number of velocity evaluations (default: 50, the "
                         "paper's main table). NFE 4 is the efficiency operating point "
                         "and trades FID for PSNR/SSIM; never mix the two in one "
                         "comparison.")
    ap.add_argument("--guidance-scale", type=float, default=1.5,
                    help="Classifier-free guidance scale (default: 1.5, the published "
                         "setting). 1.0 disables guidance and halves the cost.")
    ap.add_argument("--seed", type=int, default=2024,
                    help="Noise seed, re-applied before every image (default: 2024).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Compute device (default: cuda when available, else cpu).")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES),
                    help="Transformer precision (default: float32, as evaluated). The "
                         "autoencoder always runs in float32.")
    return ap.parse_args()


def resolve_checkpoint(checkpoint: str, subfolder: str | None) -> str:
    """Locate one arm inside a multi-arm checkpoint repository.

    `diffusers`' pipeline loader has no `subfolder` argument — it reads
    `model_index.json` from whatever it is handed — so the arm folder is
    resolved here: joined directly for a local checkpoint, and fetched on its
    own for a Hub id so the other arm is not downloaded too.
    """
    path = Path(checkpoint)
    if path.is_dir():
        resolved = path / subfolder if subfolder else path
        if not (resolved / "model_index.json").is_file():
            raise SystemExit(
                f"{resolved} has no model_index.json"
                + ("" if subfolder else " — pass --subfolder qxs-saropt or --subfolder sar2opt"))
        return str(resolved)
    if not subfolder:
        return checkpoint
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=checkpoint, allow_patterns=f"{subfolder}/*")
    return str(Path(root) / subfolder)


def main() -> int:
    args = parse_args()

    sar_paths = sorted(p for p in args.sar_dir.iterdir() if p.suffix.lower() in SUFFIXES)
    if not sar_paths:
        raise SystemExit(f"no {'/'.join(sorted(SUFFIXES))} files in {args.sar_dir}")

    device = torch.device(args.device)
    pipe = ReFlowSETPipeline.from_pretrained(
        resolve_checkpoint(args.checkpoint, args.subfolder), torch_dtype=DTYPES[args.dtype],
    )
    # The autoencoder ran in float32 in every evaluation, whatever the
    # transformer's precision.
    pipe.vae.to(torch.float32)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # The arm's training resolution, recorded in the transformer config. Inputs
    # larger than it are centre-cropped, which is the released protocol; the
    # pipeline refuses anything smaller rather than upscaling.
    crop = pipe.transformer.config.sample_size
    if device.type == "cpu":
        print("[warn] noise is drawn on the CPU; the published dumps were drawn on CUDA "
              "and the two differ for the same seed", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[translate] {len(sar_paths)} images -> {args.out_dir}  "
          f"nfe={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed} "
          f"crop={crop} device={device} dtype={args.dtype}")

    for path in tqdm(sar_paths, unit="img"):
        with Image.open(path) as im:
            sar = im.copy()
        # A fresh generator per image: every test item starts from the same
        # seeded z0, which is what the reported numbers were produced with.
        generator = torch.Generator(device=device).manual_seed(args.seed)
        image = pipe(
            sar,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pil",
            crop=crop,
        ).images[0]
        image.save(args.out_dir / f"{path.stem}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
