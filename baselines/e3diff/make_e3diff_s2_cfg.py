"""Build an E3Diff stage-2 config from a stage-1 (SR3-class DDPM) config + checkpoint.

E3Diff is two stages sharing ONE UNet architecture:
  stage 1  (`stage: 1`)  eps-prediction conditional DDPM, DDIM-50 at test time.
  stage 2  (`stage: 2`)  the SAME net fine-tuned as a `ddim_steps`-step generator:
           diffusion.p_losses() runs ddim_sample() WITH GRAD from pure noise and
           takes L1(x_start, x_recon) directly on pixels, plus LPIPS (lpips_w),
           focal-frequency (fft_w) and a vision-aided CLIP GAN (lambda_gan).
All stage 2 needs from stage 1 is the generator weights: model.py load_network()
does `torch.load(f'{resume_state}_gen.pth')` + load_state_dict(strict=False).

TRAP this script exists to handle: in phase=train, load_network() ALSO reads
`{resume_state}_opt.pth` and sets begin_step from it, and main.py loops
`while current_step < n_iter`. So n_iter is an ABSOLUTE step count continuing
stage 1, not a stage-2 budget. The authors' own configs show this: stage-1
resume at I640000 with n_iter 800000 == 160k stage-2 iterations. We therefore
write n_iter = begin_step + iters.

Second trap: core/logger.py strips everything after '//' on each config line to
support comments, so a path containing a double slash would be truncated. Every
path written here is normpath'd.
"""
import argparse
import json
import os
import sys

import torch

# authors' ratio: stage-1 640k iters, stage-2 continues to 800k -> 160k = 25%
S2_RATIO = 0.25


def load_cfg(path):
    """Read a config the way core/logger.parse does (it allows // comments)."""
    txt = ''.join(line.split('//')[0] + '\n' for line in open(path, encoding='utf-8'))
    return json.loads(txt)


def check_arch(gen_path, unet):
    """Fail loudly if the stage-1 weights do not match the config's UNet.

    downs.0        takes the noisy target only  -> in_channel
    condition.E1.0 takes the PPB+canny stack    -> condition_ch
    """
    sd = torch.load(gen_path, map_location='cpu', weights_only=True)
    for key, opt_key in (('denoise_fn.downs.0.weight', 'in_channel'),
                         ('denoise_fn.condition.E1.0.weight', 'condition_ch')):
        if key not in sd:
            sys.exit(f'ARCH CHECK FAILED: {gen_path} has no tensor {key}')
        got, want = sd[key].shape[1], unet[opt_key]
        if got != want:
            sys.exit(f'ARCH CHECK FAILED: {key} has {got} input channels, '
                     f'config says {opt_key}={want} ({gen_path})')
    return len(sd)


def begin_step_of(prefix):
    """What main.py will set current_step to when it resumes from `prefix`."""
    opt_path = f'{prefix}_opt.pth'
    if not os.path.isfile(opt_path):
        return 0  # load_network() cannot recover iter/epoch -> starts at 0
    return int(torch.load(opt_path, map_location='cpu', weights_only=False)['iter'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1-cfg', required=True, help='stage-1 config used as template')
    ap.add_argument('--resume', required=True, help='checkpoint prefix, no _gen.pth')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--phase', choices=['train', 'val'], default='train')
    ap.add_argument('--iters', type=int, default=0,
                    help='stage-2 iterations (0 = %d%% of the stage-1 step count)'
                         % int(S2_RATIO * 100))
    ap.add_argument('--ddim-steps', type=int, default=1)
    ap.add_argument('--train-root', default=None)
    ap.add_argument('--val-root', default=None)
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--res', type=int, default=0,
                    help='image size; s2o runs the frozen 512 centre-crop protocol')
    ap.add_argument('--val-len', type=int, default=None)
    ap.add_argument('--val-freq', type=int, default=None)
    ap.add_argument('--save-freq', type=int, default=None)
    ap.add_argument('--print-freq', type=int, default=None)
    ap.add_argument('--train-len', type=int, default=None)
    a = ap.parse_args()

    cfg = load_cfg(a.s1_cfg)
    gen_path = f'{a.resume}_gen.pth'
    if not os.path.isfile(gen_path):
        sys.exit(f'stage-1 weights not found: {gen_path}')
    nkeys = check_arch(gen_path, cfg['model']['unet'])

    cfg['name'] = a.name
    cfg['phase'] = a.phase
    cfg['stage'] = 2
    cfg['ddim_steps'] = a.ddim_steps
    # SAR2EO_256_s2_1step.json (the authors' stage-2 recipe)
    cfg['loss_w'] = {'fft_w': 10, 'lpips_w': 5, 'lambda_gan': 0.5, 'lcondition_w': 0}
    cfg['path']['resume_state'] = os.path.normpath(a.resume)
    # stage 2 is a few-step generator: sample with exactly ddim_steps DDIM steps
    cfg['model']['beta_schedule']['val']['n_timestep'] = a.ddim_steps
    cfg['model']['beta_schedule']['val']['ddim'] = 1

    d = cfg['datasets']
    if a.train_root:
        d['train']['dataroot'] = os.path.normpath(a.train_root)
    if a.val_root:
        d['val']['dataroot'] = os.path.normpath(a.val_root)
    if a.batch:
        d['train']['batch_size'] = a.batch
    if a.res:
        cfg['model']['diffusion']['image_size'] = a.res
        for k in ('train', 'val'):
            d[k]['l_resolution'] = d[k]['r_resolution'] = a.res
    if a.train_len is not None:
        d['train']['data_len'] = a.train_len
    if a.val_len is not None:
        d['val']['data_len'] = a.val_len
    for k in ('train', 'val'):
        d[k]['dataroot'] = os.path.normpath(d[k]['dataroot'])

    begin = begin_step_of(a.resume) if a.phase == 'train' else 0
    if a.phase == 'train':
        cfg['train']['scheduler']['milestones'] = []
        for key, val in (('val_freq', a.val_freq), ('save_checkpoint_freq', a.save_freq),
                         ('print_freq', a.print_freq)):
            if val is not None:
                cfg['train'][key] = val
        iters = a.iters if a.iters > 0 else max(1, round(S2_RATIO * begin))
        # main.py saves only on `current_step % save_checkpoint_freq == 0` and
        # does NOT save at "End of training", so an n_iter off that grid throws
        # away every iteration since the last multiple. Snap n_iter down onto it.
        freq = cfg['train']['save_checkpoint_freq']
        n_iter = max(begin + freq, (begin + iters) // freq * freq)
        if n_iter != begin + iters:
            print(f'[s2cfg] n_iter {begin + iters} -> {n_iter} '
                  f'(snapped to the save_checkpoint_freq={freq} grid so the '
                  f'final iteration writes a checkpoint)')
        cfg['train']['n_iter'] = n_iter
        print(f'[s2cfg] stage-1 resume={a.resume} ({nkeys} tensors), begin_step={begin}')
        print(f'[s2cfg] stage-2 iterations={n_iter - begin} -> n_iter={n_iter}')
        if begin == 0:
            print('[s2cfg] WARNING: no _opt.pth beside the checkpoint, so main.py '
                  'starts at step 0 and n_iter IS the stage-2 budget')
    else:
        print(f'[s2cfg] val from {a.resume} ({nkeys} tensors), '
              f'ddim_steps={a.ddim_steps}, data_len={d["val"]["data_len"]}')
        print(f'[s2cfg] -p val writes images to {os.path.normpath(a.resume)}/sample '
              'and then renames that dir with the metrics')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    for line in open(a.out, encoding='utf-8'):
        if '//' in line:
            sys.exit(f'REFUSING: written config has a // that logger.py would '
                     f'truncate: {line.strip()}')
    print(f'[s2cfg] wrote {a.out}')
    print(f'[s2cfg] {cfg["model"]["diffusion"]["channels"]}-channel target, '
          f'{cfg["model"]["unet"]["condition_ch"]}-channel condition, '
          f'{cfg["model"]["diffusion"]["image_size"]}px')
    print(f'[s2cfg] train root {d["train"]["dataroot"]} batch {d["train"]["batch_size"]}')
    print(f'[s2cfg] val   root {d["val"]["dataroot"]} data_len {d["val"]["data_len"]}')


if __name__ == '__main__':
    main()
