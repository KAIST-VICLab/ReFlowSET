"""ReFlowSET — SAR to EO, on free CPU hardware.

The pipeline classes ship with the checkpoint rather than living in `diffusers`,
so the arm is fetched and loaded as a local pipeline with `custom_pipeline`.
Only one arm is held in memory at a time: each is ~2.1 GB in float32 and the
free tier has 16 GB to cover the process, gradio and both.
"""
from __future__ import annotations

import gc
import pathlib
import time
import urllib.request

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image

REPO = "JeonghyeokDo/ReFlowSET"
ARMS = {
    "QXS-SAROPT (256x256)": ("qxs-saropt", 256),
    "SAR2Opt (512x512)": ("sar2opt", 512),
}
DEVICE = "cpu"                    # set to "cuda" and drop the thread cap on GPU

torch.set_num_threads(max(1, torch.get_num_threads()))
torch.set_grad_enabled(False)

_loaded: dict[str, DiffusionPipeline] = {}

# The examples are the very crops already published on the project page, fetched
# at startup rather than vendored: this repository redistributes no imagery.
PAGE = "https://kaist-viclab.github.io/ReFlowSET_site/static/image"
EXAMPLE_SRC = [
    (f"{PAGE}/qxs-saropt/strip/13306_sar.jpg", "qxs_13306.jpg", 0),
    (f"{PAGE}/qxs-saropt/strip/15477_sar.jpg", "qxs_15477.jpg", 0),
    (f"{PAGE}/qxs-saropt/strip/5078_sar.jpg", "qxs_5078.jpg", 0),
]


def fetch_examples() -> list[list]:
    """Cache the published crops locally. Missing network just means no examples."""
    out, d = [], pathlib.Path("examples")
    d.mkdir(exist_ok=True)
    for url, name, arm_i in EXAMPLE_SRC:
        f = d / name
        if not f.exists():
            try:
                urllib.request.urlretrieve(url, f)
            except Exception as exc:      # noqa: BLE001 - a demo without examples still works
                print(f"example {name}: {exc}")
                continue
        out.append([str(f), list(ARMS)[arm_i]])
    return out


def get_pipe(arm: str) -> DiffusionPipeline:
    """Load one arm, evicting the other. Keeps peak RSS near a single model."""
    if arm in _loaded:
        return _loaded[arm]
    for other in list(_loaded):
        del _loaded[other]
    gc.collect()
    root = snapshot_download(REPO, allow_patterns=[f"{arm}/*"])
    pipe = DiffusionPipeline.from_pretrained(
        f"{root}/{arm}", custom_pipeline=f"{root}/{arm}", torch_dtype=torch.float32
    ).to(DEVICE)
    _loaded[arm] = pipe
    return pipe


def translate(image, arm_label, steps, guidance, seed, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("Upload a SAR image first, or pick one of the examples below.")
    arm, size = ARMS[arm_label]
    if min(image.size) < size:
        raise gr.Error(
            f"The {arm} arm needs at least {size}x{size} pixels and never resizes "
            f"its input; this image is {image.size[0]}x{image.size[1]}. Use the "
            f"other arm, or crop from a larger tile."
        )
    pipe = get_pipe(arm)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    t0 = time.perf_counter()
    eo = pipe(
        image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).images[0]
    dt = time.perf_counter() - t0
    passes = int(steps) * (2 if float(guidance) != 1.0 else 1)
    note = (
        f"{arm} · NFE {int(steps)} · guidance {float(guidance):g} · seed {int(seed)} · "
        f"{dt:.1f} s for {passes} network evaluations on CPU"
    )
    if int(steps) != 50 or float(guidance) != 1.5:
        note += "  —  not the paper's setting (NFE 50, guidance 1.5)"
    return eo, note


CSS = """
.big-note { font-size: 0.92rem; line-height: 1.5; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="ReFlowSET", theme=gr.themes.Soft(primary_hue="cyan"), css=CSS) as demo:
    gr.Markdown(
        "# ReFlowSET\n"
        "### Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation\n"
        "[Paper](https://arxiv.org/abs/2609.00968) · "
        "[Weights](https://huggingface.co/JeonghyeokDo/ReFlowSET) · "
        "[Code](https://github.com/KAIST-VICLab/ReFlowSET) · "
        "[Project page](https://kaist-viclab.github.io/ReFlowSET_site/)"
    )
    gr.Markdown(
        "This Space runs on **free CPU hardware**, so it defaults to **NFE 4** — the "
        "efficiency operating point, not the paper's NFE 50. At the defaults expect "
        "roughly **15 s** on the 256 arm and **a minute** on the 512 arm. Time scales "
        "with NFE, so NFE 50 on the 512 arm takes on the order of a quarter hour; "
        "for that, run the checkpoint locally on a GPU.",
        elem_classes="big-note",
    )

    with gr.Row():
        with gr.Column():
            sar = gr.Image(label="SAR input", type="pil", height=340)
            arm = gr.Radio(
                list(ARMS), value=list(ARMS)[0], label="Arm",
                info="Each arm was trained from scratch on its own dataset. "
                     "The input is center-cropped, never resized.",
            )
            with gr.Accordion("Sampling", open=False):
                steps = gr.Slider(1, 50, value=4, step=1, label="NFE (inference steps)",
                                  info="The paper reports NFE 50. On this CPU each "
                                       "step costs about 1.5 s at 256 and 9 s at 512, "
                                       "doubled whenever guidance is above 1.0.")
                guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                     label="Guidance scale",
                                     info="1.5 is published. 1.0 halves the cost.")
                seed = gr.Number(value=2024, precision=0, label="Seed")
            run = gr.Button("Translate", variant="primary")
        with gr.Column():
            out = gr.Image(label="Generated EO", height=340)
            note = gr.Markdown()

    examples = fetch_examples()
    if examples:
        gr.Examples(
            examples=examples, inputs=[sar, arm],
            label="Examples — QXS-SAROPT crops, fetched from the project page",
        )

    gr.Markdown(
        "Neither dataset is redistributed here. QXS-SAROPT asks that "
        "[arXiv:2103.08259](https://arxiv.org/abs/2103.08259) be cited for research use. "
        "Weights are CC BY-NC 4.0.",
        elem_classes="big-note",
    )

    run.click(translate, [sar, arm, steps, guidance, seed], [out, note])

demo.queue(max_size=12).launch()
