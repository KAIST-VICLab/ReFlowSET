#!/usr/bin/env python
"""Inference for the ResShift SAR-to-EO baseline (both datasets).

Runs the trained model on a folder of SAR images (gray PNGs are replicated to
3 channels by the loader) and writes one EO-prediction PNG per input, keyed by
the input stem. At 256px with sf=1 no chopping or padding occurs; at 512 each input is
processed as four clean 256 tiles (chop_size = chop_stride = lq_size = 256).

Upstream's inference_resshift.py is a super-resolution CLI keyed to the
authors' released tasks and does not accept this config, so this is the only
working inference entry point for these weights. Drop it into the ResShift
checkout (it imports the repo's sampler module).

Usage:
  python inference_qxs.py --cfg_path configs/qxs_sar2eo_256.yaml \
      --ckpt <ema_model_50000.pth> --in_dir <testA> --out_dir <out> --bs 16
"""
import argparse

from omegaconf import OmegaConf
from sampler import ResShiftSampler

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_path', type=str,
                    default='./configs/qxs_sar2eo_256.yaml')
parser.add_argument('--ckpt', type=str, required=True,
                    help='Trained diffusion model checkpoint (use the EMA one)')
parser.add_argument('--in_dir', type=str, required=True, help='Folder of SAR images')
parser.add_argument('--out_dir', type=str, required=True, help='Output folder for EO predictions')
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--seed', type=int, default=12345)
args = parser.parse_args()

configs = OmegaConf.load(args.cfg_path)
configs.model.ckpt_path = args.ckpt

sampler = ResShiftSampler(
        configs,
        sf=configs.diffusion.params.sf,          # 1
        use_amp=True,
        chop_size=configs.model.params.lq_size,  # 256 -> no chopping for 256px inputs
        chop_stride=configs.model.params.lq_size,
        chop_bs=1,
        padding_offset=configs.model.params.get('lq_size', 64),
        seed=args.seed,
        )
sampler.inference(args.in_dir, args.out_dir, mask_path=None, bs=args.bs, noise_repeat=False)
