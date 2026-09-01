# VAE reconstruction audit

The measurement behind ReFlowSET's motivating claim: **a latent generator can
never beat the reconstruction ceiling of the autoencoder it lives in.** Encode,
decode, and a perfect generator in between still costs whatever the codec lost.
So before comparing generators, measure the codecs — on the same images, under
one protocol.

This directory holds the code. **No data and no third-party weights are
redistributed here.** Everything is downloaded by you, from its publisher, under
its own licence.

---

## What it measures, and why that is a ceiling

Each autoencoder's own **frozen round trip**: encode an image, take the
**posterior mean** — never a random posterior sample — decode, and score the
result against the input. Frozen weights and a deterministic latent make the
number a hard bound, not an average-case estimate: no model trained in that
latent space can beat its own row.

Metrics: **PSNR**, **SSIM**, **LPIPS**, and **HF corr.**, plus reconstruction
**FID**/**KID** over the set.

> **LPIPS here is not the convention `scripts/evaluate.py` uses.** The `LPIPS`
> column is `[0, 1]` handed to `lpips.LPIPS(net='vgg')` with `normalize=False`,
> which is what the published ceiling tables were measured with; the repository's
> translation table uses the standard `[0, 1] → [-1, 1]` scaling instead. The two
> differ by roughly 0.05 and are not comparable. The script emits both, the
> second as `LPIPS (standard scaling)`; check which one you are quoting.

> `HF corr.` is a global high-pass correlation — a 3×3 Laplacian applied to both
> images, then a mean-centred Pearson correlation over the whole image. It is
> **not** SCC, and should not be reported as SCC.

**SAR has no native input on any of these autoencoders.** They are all RGB
codecs. Each scalar SAR channel is therefore replicated to three channels and
encoded on its own — which is exactly the C-DiffSET configuration, so the SAR
rows measure the bottleneck a C-DiffSET-style pipeline actually operates in. For
SpaceNet6 this falls out per polarisation band (HH, HV, VH, VV) for free.

Two breakdowns come with it: SpaceNet6 **per polarisation**, and a secondary
**speckle-severity** split into per-image ENL-proxy tertiles (median over 7×7
windows of mean²/variance on the first SAR band, in the display domain — a
within-source *ranking* only; no calibrated ENL is claimed, because the packaged
SAR is 8-bit display raster).

### The latent-path integrity check is not optional

FLUX.2's latent path is **not** plain `decode(encode(x))`. A 2×2 space-to-depth
pack and an `affine=False` BatchNorm with shipped running statistics
(`eps=1e-4`) sit between the codec and the model. If the two legs were not exact
inverses, every number in the table would be silently measuring that bug instead
of the autoencoder.

`verify_latent_path` therefore runs on real data, before any metric, and
**aborts the run** on failure:

1. `unpack(pack(z)) == z` bitwise, over several shapes and dtypes;
2. `inv_normalize(normalize(z)) == z` to float32 round-off;
3. `decode(encode(x)) == decoder(mean(encoder(x)))` bitwise — the whole
   pack/normalise sandwich is the identity on the tensor that reaches the
   decoder.

Check 3 is the one that matters: 1 and 2 can both pass while the legs are wired
in the wrong order. Do not disable it.

---

## The six autoencoders

Only the **autoencoder subfolder** of each repository is needed — about 1.5 GB
in total, against ~200 GB for the six full checkpoints. `fetch_vaes.py` pulls
exactly that.

| # | Repository (Hugging Face) | Subfolder | Licence | Gated | Downsample | Latent ch. |
|---|---|---|---|---|---|---|
| SD2.1 | `Manojb/stable-diffusion-2-1-base` | `vae` | CreativeML Open RAIL++-M (`openrail++`) | no | ×8 | 4 |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | `vae` | CreativeML Open RAIL++-M (`openrail++`) | no | ×8 | 4 |
| SD3.0 | `stabilityai/stable-diffusion-3-medium-diffusers` | `vae` | Stability AI Non-Commercial Research Community License (`stabilityai-nc-research-community`) | **yes** | ×8 | 16 |
| SD3.5 | `stabilityai/stable-diffusion-3.5-large` | `vae` | Stability AI Community License (`stabilityai-ai-community`) | **yes** | ×8 | 16 |
| FLUX.1 | `black-forest-labs/FLUX.1-schnell` | `vae` | Apache-2.0 | **yes** | ×8 | 16 |
| FLUX.2 | `black-forest-labs/FLUX.2-klein-base-4B` | `vae` | Apache-2.0 | no | ×8 codec, ×16 after the 2×2 pack | 32 codec, 128 packed |

Repository ids, licence identifiers and gating were **verified against the
Hugging Face API on 2026-09-01**; `fetch_vaes.py` re-reads all three at run time
and warns if the Hub now says something else. Downsample factor and latent
channel count are read back from each downloaded config by the audit itself, and
are reproduced above from the published run's own output — nothing here is
hardcoded into the measurement.

The six are **not rate-matched**, and the tables must not be read as if they
were: FLUX.2 keeps 0.5 latent channels per pixel, SD3.x and FLUX.1 keep 0.25,
SD2.1 and SDXL keep 0.0625. A larger latent should reconstruct better. What the
audit establishes is what each codec loses **at the operating point it is
actually used at**, which is the number a generator built on it inherits.

### Notes that matter

**Gated repositories.** SD3.0, SD3.5 and FLUX.1 are served only to an account
that has accepted the publisher's licence on the Hub. Accept it on the model
page, then `hf auth login`. `fetch_vaes.py` reports and skips a repository whose
licence has not been accepted rather than failing the run.

**SD2.1 is a community mirror, and here is why.** Stability's own
`stable-diffusion-2-1` and `stable-diffusion-2-1-base` repositories are not
publicly reachable (HTTP 401 as of 2026-09-01), so the row cannot be pulled from
its original. The mirror is authenticated by content instead: its
`vae/diffusion_pytorch_model.safetensors` has sha256
`a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`, which is
**byte-identical** to `diffusion_pytorch_model.safetensors` in the official,
ungated, MIT-licensed `stabilityai/sd-vae-ft-mse` (verified from Hub blob
metadata, 2026-09-01). *What we verified is that equality.* We did **not**
independently verify against a Stability-served SD 2.1 repository that this is
the file Stability shipped with SD 2.1, because no such repository is currently
reachable.

**FLUX.1: which repository, and are they the same codec?** The published numbers
were produced from `black-forest-labs/FLUX.1-dev`, whose licence
(`flux-1-dev-non-commercial-license`) is non-commercial.
`black-forest-labs/FLUX.1-schnell` is Apache-2.0 and serves the **same blob**:
`vae/diffusion_pytorch_model.safetensors` is sha256
`f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3` in both
repositories, same blob id, same 167,666,902 bytes (verified from Hub blob
metadata, 2026-09-01). They are the same autoencoder file, so `fetch_vaes.py`
fetches the Apache-2.0 copy and the FLUX.1 row is unchanged.

**FLUX.2 is the Apache-2.0 serialisation.** ReFlowSET's latent endpoint is the
FLUX.2 autoencoder taken from `black-forest-labs/FLUX.2-klein-base-4B`
(Apache-2.0, ungated), **not** the `FLUX.2-dev` file, whose licence forbids
"research and development related to surveillance". The pipeline loads a
BFL-layout `ae.safetensors` rather than the diffusers folder;
[`../scripts/convert_flux2_ae.py`](../scripts/convert_flux2_ae.py) rebuilds it
from the download, and the repository README documents that step. Do not
duplicate those instructions here — run that script.

Stated for the same reason the FLUX.1 note above exists: the published FLUX.2
ceiling row was measured with the **FLUX.2-dev serialisation**, and unlike the
FLUX.1 pair the two FLUX.2 files are not byte-identical — they are the same
network re-serialised, 250 of 251 tensors pairing by value with a worst absolute
deviation of 7.8e-03 (bfloat16 rounding), as
[`../LICENSE-WEIGHTS.md`](../LICENSE-WEIGHTS.md) and [`../NOTICE`](../NOTICE)
record. Reproducing the FLUX.2 row from the Apache-2.0 file should land on the
same numbers, but not on the same bits.

---

## The four datasets

| Dataset | Imagery | Resolution used | Download | Licence / terms |
|---|---|---|---|---|
| **QXS-SAROPT** | 20,000 SAR/optical patch pairs — GaoFen-3 SAR against Google Earth optical, over San Diego, Shanghai and Qingdao | 256 (native) | <https://github.com/yaoxu008/QXS-SAROPT> — the README links a request form; distribution is **request-gated** | **No LICENSE file** in the repository. The README states the dataset paper "must be cited when the dataset is used for research purposes" — cite [arXiv:2103.08259](https://arxiv.org/abs/2103.08259). No further terms are published. |
| **SAR2Opt** | Manually co-registered SAR/optical pairs, cropped to 600×600 (the repository does not name the sensors) | 512, **center crop** from 600 (offset 44); never resized | <https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset> — the README links Google Drive and Baidu Disk | **MIT** (GitHub licence field, verified 2026-09-01) |
| **SpaceNet6** | AOI 11 Rotterdam: 0.5 m quad-pol SAR from Capella Space (HH, HV, VH, VV) against 0.5 m Maxar WorldView-2 EO | 512 and 768 | <https://spacenet.ai/sn6-challenge/> — hosted on AWS Open Data, `s3://spacenet-dataset/spacenet/SN6_buildings/`. Free, but an AWS account is required. | **CC BY-SA 4.0** — "The SpaceNet Dataset by SpaceNet Partners is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License." |
| **SAR-1M** | Large SAR/optical corpus aggregating several source datasets | 256 | <https://huggingface.co/datasets/Wenquandan777/SAR-1M> (gated; accept on the Hub, then `hf auth login`) | **Licence is in conflict — we do not resolve it.** The Hub card says **CC BY-NC 4.0**; the dataset paper's datasheet ([arXiv:2512.16635](https://arxiv.org/abs/2512.16635), §11.6 Q4) says **CC BY-NC-SA 4.0**. Assume the stricter of the two until the authors clarify. |

### SAR-1M needs three warnings, not one

1. **The licence conflict above is real and unresolved.** State both, assume
   CC BY-NC-SA 4.0, do not pick one silently.
2. **AFRL distribution-restricted files sit in the same directory tree.** MSTAR
   and FARAD imagery is present alongside the redistributable material. This is
   precisely why the audit is **manifest-driven and never globs a directory**: a
   `glob("**/*.png")` over the corpus root will pick up files that must not be
   used or redistributed. Build the manifest, read the manifest, and if you add a
   loader, keep that property.
3. **The name may oversell the resolution — check your own draw.** SAR-1M
   aggregates several source datasets, and `--make-sar1m-manifest` samples
   `paired.json` **uniformly at random**, so which sources a manifest contains
   depends on the draw and on the archive you downloaded. The largest
   contributor in the paired portion is M4-SAR, whose SAR is Sentinel-1
   exclusively — roughly **10 m** GSD, medium resolution, not the 1–10 m high
   resolution "SAR-1M" suggests ([arXiv:2505.10931](https://arxiv.org/abs/2505.10931)).
   Establish the composition of your own manifest before calling any SAR-1M row
   high-resolution.

---

## Nothing here is redistributed

No imagery and no third-party model weights ship in this repository. Each
autoencoder comes from its publisher's Hugging Face repository under that
publisher's licence, and each dataset comes from its own distributor under the
terms in the table above. Accepting those terms is the user's responsibility.
ReFlowSET's own code is Apache-2.0 ([`../LICENSE`](../LICENSE),
[`../NOTICE`](../NOTICE)).

---

## How to run

Fetch the autoencoders first (gated rows need `hf auth login` beforehand):

```bash
python vae_audit/fetch_vaes.py --dest weights/vae_zoo
```

The helper prints, per model, the repository, the licence the Hub reports, and
whether the repository is gated, and skips — politely — anything whose licence
has not been accepted. Then build ReFlowSET's FLUX.2 autoencoder file from the
Apache-2.0 download, as the repository README describes:

```bash
python scripts/convert_flux2_ae.py \
    --src weights/vae_zoo/FLUX.2/vae/diffusion_pytorch_model.safetensors \
    --key-map assets/flux2_ae_key_map.json \
    --out weights/ae.safetensors
```

Then run the audit, one dataset at a time. Point `--data-root` at your own copy;
paths are yours, nothing is assumed.

```bash
# QXS-SAROPT — 256, EO and SAR
python vae_audit/vae_recon_audit.py --dataset qxs-saropt --data-root /path/to/QXS-SAROPT \
    --resolution 256 --vae-root weights/vae_zoo --flux2-weights weights/ae.safetensors \
    --out reports/vae_ceiling

# SAR2Opt — 512 center crop from 600
python vae_audit/vae_recon_audit.py --dataset sar2opt --data-root /path/to/sar2opt \
    --resolution 512 --vae-root weights/vae_zoo --flux2-weights weights/ae.safetensors \
    --out reports/vae_ceiling

# SpaceNet6 — 512 (and 768); the per-polarisation breakdown is automatic
python vae_audit/vae_recon_audit.py --dataset spacenet6 --data-root /path/to/SN6 \
    --resolution 512 --file-list spacenet6=/path/to/your_sn6_eo_list.txt \
    --vae-root weights/vae_zoo \
    --flux2-weights weights/ae.safetensors --out reports/vae_ceiling

# SAR-1M — 256, manifest-driven; build the manifest once, then read it
python vae_audit/vae_recon_audit.py --dataset sar-1m --data-root /path/to/SAR-1M \
    --resolution 256 --manifest reports/manifests/sar1m.jsonl \
    --vae-root weights/vae_zoo --flux2-weights weights/ae.safetensors \
    --out reports/vae_ceiling
```

> Flag names above follow the audit script in this directory; run it with
> `--help` for the exact spelling, which is authoritative.

> **SpaceNet6 has no split list in this repository.** `splits/` ships the
> QXS-SAROPT and SAR2Opt lists only, and those two datasets pick theirs up
> automatically. Without `--file-list spacenet6=<list of EO paths relative to the
> root>`, the audit falls back to **every** `PS-RGB` tile under the root — the
> whole AOI, not any published split — and the resulting row is not comparable
> with one measured on a held-out split. Say which it is whenever you quote it.

**A GPU is required.** The audit encodes and decodes full test splits at
512–768 px and computes FID/KID, so CPU is not a practical fallback.

**Cost, measured.** The published run's own per-row timings put a single
(autoencoder × dataset × modality) row at **3–10 minutes** — 186 s at the
fastest (256 px, 1,000 items), 622 s at the slowest (768 px). Summed over every
row that run recorded — about 61, because SpaceNet6 was measured at 512 *and*
768 and one early pass was repeated — the total is roughly **6.3 GPU-hours**; a
clean 6 × 4 × 2 = 48-row grid at one resolution per dataset costs less. The GPU
model is not recorded in those artefacts, so no throughput claim is made for a
particular card. The `sqrtm` inside FID dominates the wall clock; a
per-image-metrics-only pass is much faster.

---

## The published numbers

The measurements behind the paper's motivating claim already exist. They cover
**PSNR, SSIM, LPIPS and HF corr.** (plus MS-SSIM, Pearson, FID and KID), for
**EO and SAR separately**, for all six autoencoders on all four datasets.

The **per-polarisation** and **speckle-severity** breakdowns described above are
produced by this script; they are **not** part of those six-autoencoder tables,
which report one SAR number per dataset. Do not expect the two to line up
without re-running.

They are **not included in this pass**, which ships the code only: the audit
above regenerates them from your own copies of the data.
