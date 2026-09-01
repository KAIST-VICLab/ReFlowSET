#!/usr/bin/env python3
"""Fetch the six frozen autoencoders measured by the ReFlowSET VAE reconstruction audit.

A latent generator can never beat the reconstruction ceiling of the autoencoder it
lives in, so that ceiling has to be measured on the *same* images under the *same*
protocol for every candidate -- quoting each autoencoder's own paper would compare
six different protocols. This helper obtains the six autoencoders; `README.md` in
this directory explains what is then measured.

Only the **autoencoder subfolder** of each repository is downloaded, never the
whole checkpoint: ~1.5 GB in total instead of ~200 GB. Nothing here redistributes
third-party weights -- each one is pulled from its own publisher, under its own
licence, by the person running this script.

Three of the six repositories -- SD3.0, SD3.5 and FLUX.1 -- are gated: the Hub
serves them only to an account that has accepted the publisher's licence. Log in
first with

    hf auth login

A repository whose licence has not been accepted is reported and skipped, not
raised -- a partial zoo still measures the rows it has.

Usage
-----
    python vae_audit/fetch_vaes.py --dest weights/vae_zoo
    python vae_audit/fetch_vaes.py --only FLUX.2 SDXL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

# name -> (repo id, subfolder, licence recorded when this file was written, gated).
#
# Latent channels and downsample factor are deliberately NOT recorded here: the
# audit re-reads them from each downloaded config, so a repository that changes
# cannot silently disagree with a hardcoded table.
#
# Licence and gating are recorded so the script can WARN when the Hub now says
# something else. They were verified against the Hub API on 2026-09-01; treat the
# live value the script prints as authoritative, not this table.
REPOS: dict[str, tuple[str, str, str, bool]] = {
    # Stability's own stable-diffusion-2-1 / -2-1-base repositories are not
    # publicly reachable (HTTP 401, 2026-09-01). This community mirror serves a
    # vae/diffusion_pytorch_model.safetensors that is byte-identical
    # (sha256 a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815)
    # to the MIT-licensed official stabilityai/sd-vae-ft-mse, which is how the
    # file is authenticated. See README.md.
    "SD2.1":  ("Manojb/stable-diffusion-2-1-base",             "vae", "openrail++", False),
    "SDXL":   ("stabilityai/stable-diffusion-xl-base-1.0",     "vae", "openrail++", False),
    "SD3.0":  ("stabilityai/stable-diffusion-3-medium-diffusers",
               "vae", "stabilityai-nc-research-community", True),
    "SD3.5":  ("stabilityai/stable-diffusion-3.5-large",       "vae", "stabilityai-ai-community", True),
    # The published numbers were produced from black-forest-labs/FLUX.1-dev,
    # whose licence is non-commercial. FLUX.1-schnell serves the identical file
    # (same blob, sha256 f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3)
    # under Apache-2.0, so this helper fetches the Apache-2.0 copy.
    "FLUX.1": ("black-forest-labs/FLUX.1-schnell",             "vae", "apache-2.0", True),
    # ReFlowSET's own latent endpoint. scripts/convert_flux2_ae.py turns this
    # subfolder into the BFL-layout ae.safetensors the pipeline loads.
    "FLUX.2": ("black-forest-labs/FLUX.2-klein-base-4B",       "vae", "apache-2.0", False),
}


def _live_info(api: HfApi, repo: str) -> tuple[str, str]:
    """(licence, gating) as the Hub reports them right now; '?' when unreadable."""
    try:
        info = api.model_info(repo)
    except Exception:
        return "?", "?"
    card = info.card_data or {}
    lic = card.get("license_name") or card.get("license") or "?"
    gate = info.gated
    return str(lic), ("no" if gate in (False, None) else str(gate))


def fetch(name: str, dest: Path, api: HfApi) -> bool:
    repo, sub, want_lic, want_gated = REPOS[name]
    lic, gate = _live_info(api, repo)
    print(f"\n  {name}")
    print(f"    repo     https://huggingface.co/{repo}")
    print(f"    licence  {lic}" + ("" if lic in (want_lic, "?") else f"   [!] recorded as {want_lic}"))
    print(f"    gated    {gate}" + ("" if (gate != 'no') == want_gated or gate == "?" else "   [!] gating changed"))

    try:
        # fp16 duplicates are excluded: the audit runs the fp32 file.
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=[f"{sub}/*.json", f"{sub}/*.safetensors"],
            ignore_patterns=["*.fp16.*"],
            local_dir=str(dest / name),
        )
    except GatedRepoError:
        print(f"    SKIP     licence not accepted for this account. Open "
              f"https://huggingface.co/{repo}, accept the licence, then "
              f"`hf auth login` and re-run.")
        return False
    except RepositoryNotFoundError:
        print(f"    SKIP     repository not found (it may have been moved or made private).")
        return False
    except Exception as exc:  # network, disk, revision -- report, keep going
        print(f"    SKIP     {type(exc).__name__}: {str(exc)[:120]}")
        return False

    files = sorted(f.name for f in (Path(path) / sub).glob("*.safetensors"))
    print(f"    OK       {dest / name / sub}  {files}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default="weights/vae_zoo",
                    help="directory to download into (default: weights/vae_zoo)")
    ap.add_argument("--only", nargs="+", choices=sorted(REPOS),
                    help="fetch only these autoencoders (default: all six)")
    args = ap.parse_args()

    names = args.only or list(REPOS)
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    print(f"  destination: {dest.resolve()}")
    print("  downloading the autoencoder subfolder only, never the full checkpoint.")

    ok = [n for n in names if fetch(n, dest, api)]
    bad = [n for n in names if n not in ok]

    print(f"\n  {len(ok)}/{len(names)} fetched: {', '.join(ok) or '-'}")
    if bad:
        print(f"  not fetched: {', '.join(bad)}")
        print("  gated repositories need the licence accepted on the Hub and `hf auth login`.")
    if "FLUX.2" in ok:
        print("\n  FLUX.2: ReFlowSET loads the BFL-layout file, not the diffusers folder.")
        print(f"  Build it with scripts/convert_flux2_ae.py --src "
              f"{dest / 'FLUX.2' / 'vae' / 'diffusion_pytorch_model.safetensors'} ...")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
