"""ReFlowSET SAR -> EO translation pipeline."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from .autoencoder_flux2 import AutoencoderFlux2
from .scheduler_flow_bridge import FlowBridgeScheduler
from .transformer_reflowset import ReFlowSETTransformer2DModel

#: PIL modes the SAR loader accepts.  The reference loader calls ``np.array(im)``
#: with no ``convert()`` (datasets.py:454, 574), so a 16-bit (``I;16``) or
#: palette (``P``) raster would flow straight into ``x / 127.5 - 1`` and be
#: badly out of range.  That is an unguarded trap upstream; it is guarded here.
_ACCEPTED_SAR_MODES = ("L", "RGB", "RGBA")


class ReFlowSETPipeline(DiffusionPipeline):
    """Generate an EO image from a SAR image with ReFlowSET's flow bridge.

    Args:
        transformer: The velocity transformer.
        vae: The frozen FLUX.2 autoencoder that defines the latent space.
        scheduler: The Design-B flow-bridge Euler solver.

    To reproduce the paper's numbers, sample at ``num_inference_steps=50``,
    ``guidance_scale=1.5``, float32, one image per call, with a generator freshly
    seeded to 2024 on the compute device before each call — every test image in
    the reported evaluation starts from the same seeded noise draw, and CPU-drawn
    noise does not reproduce a CUDA draw.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: ReFlowSETTransformer2DModel,
        vae: AutoencoderFlux2,
        scheduler: FlowBridgeScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

    # ---- preprocessing (datasets.py:124-205, 452-474, 572-589) --------------

    @staticmethod
    def _sar_hwc(raster: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """SAR raster -> ``[H, W, C]`` float32 in ``[0, 255]``, collapsed to 1 channel."""
        if isinstance(raster, Image.Image):
            if raster.mode not in _ACCEPTED_SAR_MODES:
                raise ValueError(
                    f"SAR image mode {raster.mode!r} is not an 8-bit display raster; expected "
                    f"one of {_ACCEPTED_SAR_MODES}. ReFlowSET was trained on 8-bit display "
                    "quicklooks (sar_value_domain='display_png'); convert with .convert('L') "
                    "and be aware that the contrast stretch you choose is part of the input."
                )
            # No .convert() on the SAR side, matching datasets.py:454, 574.
            arr = np.array(raster)
        else:
            arr = np.asarray(raster)
        if arr.ndim == 2:  # PIL mode "L" (datasets.py:171-172)
            arr = arr[:, :, None]
        if arr.shape[-1] == 4:  # drop a container alpha channel (datasets.py:173-174)
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if arr.shape[-1] > 1:
            # Exact-equality test, tol=0.0 (datasets.py:100-111): a display RGB
            # quicklook collapses to its single amplitude channel.
            if np.abs(arr - arr[..., :1]).max() == 0.0:
                arr = arr[..., :1]
            else:
                warnings.warn(
                    "SAR raster has non-identical colour channels; feeding all 3 to the "
                    "frozen encoder. The released arms were trained on single-channel "
                    "amplitude quicklooks, so this is an undeclared input.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return arr

    @staticmethod
    def _center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
        """Center-crop ``[H, W, C]`` to ``crop`` — **never** resize (datasets.py:145-165).

        The offsets are albumentations' ``CenterCrop`` arithmetic ``(n - c) // 2``:
        the SAR2Opt protocol takes the central 512 of 600 at offset 44.
        """
        h, w = arr.shape[:2]
        if h < crop or w < crop:
            raise ValueError(
                f"image {h}x{w} is smaller than the requested crop {crop}; ReFlowSET never "
                "upscales an input"
            )
        top, left = (h - crop) // 2, (w - crop) // 2
        return arr[top : top + crop, left : left + crop]

    def preprocess(
        self,
        sar: Union[Image.Image, np.ndarray, torch.Tensor, list],
        crop: Optional[int] = None,
    ) -> torch.Tensor:
        """Build the model-boundary SAR tensor ``[B, 3, H, W]`` in ``[-1, 1]``.

        Args:
            sar: A PIL image, a list of PIL images, an ``[H, W]`` / ``[H, W, C]``
                uint8 array, or a float tensor already in ``[-1, 1]`` shaped
                ``[H, W]``, ``[C, H, W]`` or ``[B, C, H, W]``.
            crop: Center-crop size applied before normalisation.  ``None``
                center-crops to the arm's own training resolution when the
                raster is larger and not already a multiple of the latent
                stride -- which is exactly the SAR2Opt 600 -> 512 protocol the
                reported numbers use.  Pass an explicit size to override, or
                ``0`` to keep the native raster and fail loudly if it does not
                fit.

        Images are read as 8-bit display rasters and mapped to ``[-1, 1]`` by
        ``x / 127.5 - 1`` (datasets.py:124-126) with no per-image statistics, no
        percentile stretch and no resize.  The single SAR channel is then
        replicated to 3 at the model boundary (evaluate.py:566-570), because the
        frozen FLUX.2 encoder is the same one that encodes EO — ReFlowSET has no
        separate SAR encoder.
        """
        if crop == 0:
            crop = None
        elif crop is None:
            # Fall back to the resolution this arm was trained at. Cropping is
            # the protocol (train.py random-crops, evaluate.py center-crops);
            # ReFlowSET never resizes, so an un-croppable raster is an error
            # rather than something to silently rescale.
            crop = self.transformer.config.sample_size

        if isinstance(sar, torch.Tensor):
            x = sar.float()
            if x.ndim == 2:
                x = x[None, None]
            elif x.ndim == 3:
                x = x[None]
            elif x.ndim != 4:
                raise ValueError(f"SAR tensor must have 2, 3 or 4 dims, got {tuple(sar.shape)}")
            h, w = x.shape[-2:]
            if crop is not None and (h, w) != (crop, crop):
                if h < crop or w < crop:
                    raise ValueError(f"tensor {h}x{w} is smaller than the requested crop {crop}")
                top, left = (h - crop) // 2, (w - crop) // 2
                x = x[..., top : top + crop, left : left + crop]
        else:
            images = sar if isinstance(sar, list) else [sar]
            arrays = []
            for item in images:
                if not isinstance(item, (Image.Image, np.ndarray)):
                    raise TypeError(f"unsupported SAR input type {type(item)!r}")
                arr = self._sar_hwc(item)
                if crop is not None and arr.shape[:2] != (crop, crop):
                    arr = self._center_crop(arr, crop)
                arrays.append(np.ascontiguousarray(arr.transpose(2, 0, 1)))
            x = torch.from_numpy(np.stack(arrays)) / 127.5 - 1.0

        # Train-side clamp (train.py:770); a no-op on 8-bit input, which maps
        # exactly onto [-1, 1].
        x = x.clamp(-1.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"the frozen FLUX.2 encoder takes 1 or 3 SAR channels, got {x.shape[1]}"
            )
        factor = self.vae.spatial_factor
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(
                f"SAR size {x.shape[-2]}x{x.shape[-1]} must be divisible by {factor}; pass "
                "crop= to center-crop (ReFlowSET never resizes)"
            )
        return x

    # ---- postprocessing -----------------------------------------------------

    @staticmethod
    def _to_pil(images: torch.Tensor) -> list[Image.Image]:
        """``[B, 3, H, W]`` in ``[0, 1]`` -> PIL, quantised round-half-up.

        ``255 * x + 0.5`` truncated is what ``torchvision.utils.save_image``
        does and is therefore what the released PNGs contain; numpy's
        ``round()`` is banker's rounding and would differ on exact halves.
        """
        arr = (images * 255 + 0.5).clamp(0, 255).to(torch.uint8)
        arr = arr.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in arr]

    @torch.no_grad()
    def __call__(
        self,
        sar: Union[Image.Image, np.ndarray, torch.Tensor, list],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        crop: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, tuple[list]]:
        """Translate a SAR image into an EO image.

        Args:
            sar: SAR input; see :meth:`preprocess`.
            num_inference_steps: NFE, the number of velocity evaluations.  The
                paper's main results are NFE 50; NFE 4 is the efficiency
                operating point and trades FID for PSNR/SSIM, so the two must
                not be mixed in one comparison.
            guidance_scale: Classifier-free guidance scale.  1.5 is the published
                setting; 1.0 disables guidance and halves the cost.
            generator: Generator for the initial noise.  Create it on the compute
                device — CPU-drawn noise does not reproduce a CUDA draw.
            output_type: ``"pil"``, ``"np"`` or ``"pt"``.
            crop: Center-crop size applied to the SAR input before encoding.
            return_dict: Return an ``ImagePipelineOutput`` instead of a tuple.

        Returns:
            The generated EO image(s) in ``[0, 1]`` (or as PIL).
        """
        if output_type not in ("pil", "np", "pt"):
            raise ValueError(f"output_type must be 'pil', 'np' or 'pt', got {output_type!r}")

        device = self._execution_device
        dtype = self.transformer.dtype

        sar_pm1 = self.preprocess(sar, crop=crop).to(device=device, dtype=self.vae.dtype)
        # The SAR condition is encoded by the SAME frozen autoencoder that
        # defines the EO latent space (evaluate.py:553-577).
        z_s = self.vae.encode(sar_pm1).to(dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # Design B: the bridge starts at t = 0 from pure Gaussian noise
        # (bridge.py:409-433), NOT from the SAR latent.
        latents = randn_tensor(z_s.shape, generator=generator, device=device, dtype=z_s.dtype)

        for t in self.progress_bar(self.scheduler.timesteps):
            timestep = t.expand(latents.shape[0])
            velocity = self.transformer(latents, timestep, z_s, return_dict=False)[0]
            if guidance_scale != 1.0:
                # Two passes; the null branch is cond=None, which the transformer
                # turns into an all-zero conditioning latent (bridge.py:530-535).
                uncond = self.transformer(latents, timestep, None, return_dict=False)[0]
                velocity = uncond + guidance_scale * (velocity - uncond)
            latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]

        image = self.vae.decode(latents.to(self.vae.dtype))
        # `--denorm standard` (evaluate.py:292-295). The `legacy` C-DiffSET
        # convention `(x + 0.5).clamp(0, 1)` is a 2x contrast stretch and must
        # not be used with these numbers.
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)

        self.maybe_free_model_hooks()

        if output_type == "pil":
            image = self._to_pil(image)
        elif output_type == "np":
            image = image.permute(0, 2, 3, 1).float().cpu().numpy()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
