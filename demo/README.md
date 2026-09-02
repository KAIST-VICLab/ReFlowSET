# Interactive demo

A Gradio app that runs the released checkpoints on your own SAR imagery, with
sliders for NFE, guidance and seed. It is the same app that would run as a
Hugging Face Space; hosting one requires a PRO subscription, so it ships here to
be run locally or duplicated onto an account that has one.

```bash
pip install -r demo/requirements.txt gradio
python demo/app.py
```

The app downloads whichever arm you pick from
[`JeonghyeokDo/ReFlowSET`](https://huggingface.co/JeonghyeokDo/ReFlowSET) and
holds one at a time, so peak memory stays near a single model (~4.3 GB observed
in float32 on CPU).

## Speed

Measured on four CPU threads at **NFE 4, guidance 1.5**:

| Arm | One image |
|---|---|
| `qxs-saropt` @256 | 8 s (5 s at guidance 1.0) |
| `sar2opt` @512 | 36 s |

Cost is linear in NFE and doubles whenever guidance is above 1.0, so the
paper's **NFE 50** setting is minutes per image on a CPU and seconds on a GPU.
Set `DEVICE = "cuda"` at the top of `app.py` for the latter.

**NFE 4 is not the paper's setting.** The main table is NFE 50 at guidance 1.5;
the two trade distribution metrics against pixel metrics and must not be mixed
in one comparison. The app labels every result with the setting that produced it.

## Example imagery

The examples are fetched at startup from the crops already published on the
[project page](https://kaist-viclab.github.io/ReFlowSET_site/). No imagery is
redistributed by this repository. QXS-SAROPT asks that arXiv:2103.08259 be cited
for research use.
