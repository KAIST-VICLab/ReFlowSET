# Licence of the ReFlowSET model weights

**Code:** Apache License 2.0 — see [LICENSE](LICENSE).
**Weights:** Creative Commons Attribution-NonCommercial 4.0 International
(**CC BY-NC 4.0**) — <https://creativecommons.org/licenses/by-nc/4.0/>.

Two artefacts, two licences, because they have two different origins. The code is
ours plus Apache-2.0 model code from Black Forest Labs. The weights additionally
carry whatever travels from the frozen autoencoder, the training-time teacher and
the two training datasets.

This is the conservative reading. Where a question is genuinely unsettled it is
stated as unsettled below rather than resolved silently in our favour.

---

## 1. What does *not* constrain these weights

**No pretraining corpus.** ReFlowSET is trained from scratch (`init_from: null`)
on one dataset per arm and nothing else. It inherits none of the obligations of
any large SAR pretraining corpus.

**The autoencoder was substituted after training, and that is worth stating
precisely rather than glossing.** Both arms were **trained and evaluated with the
FLUX.2-dev serialisation** of the FLUX.2 autoencoder: that file encoded and
decoded every image during training and during the evaluation that produced the
paper's table. It is published under the FLUX Non-Commercial License, whose
§4(a)(iii) forbids *"surveillance purposes, including all research and development
related to surveillance"* and whose §1(a) makes that inherit permanently — a
clause squarely aimed at work like this.

**The file published here is the Apache-2.0 serialisation of the same network**,
from `black-forest-labs/FLUX.2-klein-base-4B`. The two are the same autoencoder:
250 of 251 tensors pair one-to-one by value with a worst absolute deviation of
7.8e-03 (bfloat16 rounding; 3.9e-03 relative), and the unpaired entry is a
BatchNorm step counter, not a weight.

Measured end to end on the released checkpoints, the substitution changes
QXS-SAROPT PSNR by **less than 0.004 dB in absolute value**. Four
independent measurements — different guidance scales, sample sets and sample
sizes — land between -0.004 and +0.002 dB, so the sign is not resolved at
these sample sizes and only the magnitude is meaningful. For scale: changing
only the evaluation seed moves the same measurement by +0.395 dB, two orders
of magnitude more, and produces a visibly different image (16 dB agreement)
where the autoencoder swap produces the same image to within a few of 255
grey levels (59 dB agreement). Nothing the paper prints changes.

`scripts/convert_flux2_ae.py` rebuilds the shipped file from the official
Apache-2.0 download and never touches the non-commercial one, so a downstream
user reproducing the release never has to obtain it either.

**No ShareAlike.** Nothing in this training set imposes it.

---

## 2. What does constrain them

### 2.1 DINOv3 was the training-time teacher

Both released arms were trained with a representation-alignment loss
(`lambda_repa = 0.5`) against a frozen DINOv3 ViT-L/16 (LVD-1689M) teacher. The
teacher is not loaded at inference, the projector that consumed it is not in the
released checkpoints, and **no DINOv3 weights are redistributed here**.

Stated plainly so a downstream user can form their own view: DINOv3 License
§1(b)(i) says distribution of *"DINO Materials, and any derivative works thereof"*
must be under that Agreement. Whether a model trained *against* DINOv3 features —
containing none of its parameters — is a derivative work of DINO Materials is not
settled, and we do not assert an answer. What we do:

- redistribute no DINO Materials;
- acknowledge the use, as §1(b)(ii) requires, here and in the paper;
- record that §1(b)(v) restricts military, warfare and espionage end uses, and
  that anyone *retraining* ReFlowSET with this teacher accepts that Agreement
  directly.

The non-commercial tag on these weights is set partly so that this question does
not have to be answered before release.

### 2.2 Training data

| Dataset | Stated terms | Effect |
|---|---|---|
| QXS-SAROPT | **No LICENSE file.** The README states one term: the dataset paper (arXiv:2103.08259) must be cited for research use. Distribution is request-gated behind a survey form. | Cite the paper. Do not mirror the imagery. |
| SAR2Opt | **MIT**, Copyright (c) 2021 MarsZhaoYT. | Attribution. The MIT text grants rights in "the Software"; whether the authors intended it to reach the imagery is not stated in the repository. |

Neither dataset is redistributed by this project.

### 2.3 An open question we are flagging, not deciding

**The optical side of both datasets is Google Earth imagery.** Google Maps
Platform terms prohibit using Maps Content *"to improve machine learning and
artificial intelligence models, including to train, test, validate or fine-tune
the models"*. That is a contract term binding whoever accepted it, not a copyright
term, and it attaches to the dataset authors' collection rather than to our use of
their published dataset — but it is unresolved, it applies to **100 % of the
target side of both datasets**, and it is inherited by every method in the
comparison table, not only by ours.

We raise it because a reader deserves to know it exists. We do not claim it is
settled in either direction. Anyone building commercially on these weights should
form their own view first; the non-commercial tag means we are not inviting them
to skip that step.

---

## 3. Summary

| Artefact | Licence |
|---|---|
| Code in this repository | Apache-2.0 |
| ReFlowSET checkpoints | CC BY-NC 4.0 |
| Bundled FLUX.2 autoencoder weights | Apache-2.0 (Black Forest Labs) |
| DINOv3 teacher | not redistributed; obtain from Meta under the DINOv3 License |
| QXS-SAROPT, SAR2Opt imagery | not redistributed |
| Comparison-method checkpoints | one licence each — see `MODEL_ZOO.md` |

Apache-2.0 §6 withholds trademark rights: this model is named ReFlowSET and is
not a FLUX product, nor endorsed by Black Forest Labs.
