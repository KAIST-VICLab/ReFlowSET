"""Frozen FLUX.2 autoencoder — the latent endpoint of ReFlowSET.

ReFlowSET never fine-tunes this module: it is loaded once, frozen, and used to
encode the SAR condition and to decode the sampled EO latent.  The released
weights are the **Apache-2.0** FLUX.2-klein-base-4B copy of the autoencoder,
re-keyed to the layout below (see ``scripts/convert_flux2_ae.py``).

Three details of the checkpoint are non-standard for `diffusers` and are
preserved exactly, because the file must load with ``strict=True``:

* ``quant_conv`` lives **inside** ``encoder.*`` and is the last op of the
  encoder forward; ``post_quant_conv`` lives **inside** ``decoder.*`` and is the
  first op of the decoder forward.  `diffusers`' ``AutoencoderKL`` makes both
  siblings of the encoder/decoder.
* The latent normaliser is a real ``BatchNorm2d(128, affine=False)`` whose
  running statistics ship in the checkpoint under ``bn.*`` — a per-channel mean
  **and** variance, not a scalar ``scaling_factor``/``shift_factor``.  Its
  epsilon is ``1e-4``, not torch's ``1e-5``.
* ``encode`` returns the posterior **mean**; the log-variance chunk of the
  encoder's moments is discarded, so encoding is deterministic and there is no
  ``DiagonalGaussianDistribution`` and no ``.sample()``.

The public latent is ``[B, 128, H/16, W/16]``: an 8x convolutional stride
followed by a 2x2 space-to-depth pack that is part of the *autoencoder*, not of
the transformer.
"""

from __future__ import annotations

import os

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import functional as F


def swish(x: Tensor) -> Tensor:
    """``x * sigmoid(x)`` — the activation used throughout the FLUX.2 AE."""
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """Single-head self-attention over the spatial grid (head dim == channels)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        # "b c h w -> b 1 (h w) c": ONE head whose head-dim is the full channel
        # count (flux2_ae.py:70-73).
        q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1).contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        return h_.squeeze(1).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Stride-2 conv with FLUX's asymmetric ``(0, 1, 0, 1)`` pad (flux2_ae.py:111-121)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    """Nearest-neighbour 2x followed by a 3x3 conv (flux2_ae.py:124-132)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    """FLUX.2 encoder.  Emits ``2 * z_channels`` moments; ``quant_conv`` is internal."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        # Declared first so the checkpoint key is `encoder.quant_conv.*`
        # (flux2_ae.py:146) — diffusers keeps quant_conv outside the encoder.
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            # Empty at every level in this checkpoint: attention exists only in
            # `mid` (flux2_ae.py:162).  Kept so the forward guard is meaningful.
            down.attn = nn.ModuleList()
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1])))
        h = self.conv_out(swish(self.norm_out(h)))
        return self.quant_conv(h)  # last op of the encoder (flux2_ae.py:207)


class Decoder(nn.Module):
    """FLUX.2 decoder.  ``post_quant_conv`` is internal and runs first."""

    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        # Checkpoint key `decoder.post_quant_conv.*` (flux2_ae.py:223).
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleList()  # empty in this checkpoint (flux2_ae.py:249)
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend so `up.<i>` indexes by resolution level

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)  # first op of the decoder (flux2_ae.py:267)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = h.to(upscale_dtype)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        return self.conv_out(swish(self.norm_out(h)))


class AutoencoderFlux2(ModelMixin, ConfigMixin):
    """Frozen FLUX.2 autoencoder with ReFlowSET's packed, BN-normalised latent.

    ``encode`` maps ``[B, 3, H, W]`` in ``[-1, 1]`` to ``[B, 128, H/16, W/16]``
    and ``decode`` inverts it.  The module is frozen: ``train()`` is a no-op that
    always selects eval mode, and the latent BatchNorm is additionally forced to
    eval on every call so no batch statistic can ever leak into the latent.

    Args:
        resolution: Nominal training resolution of the original autoencoder.
            Only used to size bookkeeping attributes; any ``H``, ``W`` divisible
            by 16 may be encoded.
        in_channels: Input image channels (3).
        ch: Base width.
        out_ch: Output image channels (3).
        ch_mult: Per-level width multipliers; ``len(ch_mult) - 1`` downsamples.
        num_res_blocks: Residual blocks per level.
        z_channels: Pre-pack latent channels (32).
        patch_size: Space-to-depth factor applied after the encoder (2), which
            takes the latent from 32 channels at ``H/8`` to 128 at ``H/16``.
        bn_eps: Epsilon of the latent BatchNorm.  **1e-4**, not torch's 1e-5
            (flux2_ae.py:331); using 1e-5 shifts the latent by up to 2.6e-5.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 32,
        patch_size: int = 2,
        bn_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        ch_mult = list(ch_mult)
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )
        # Per-channel latent normaliser with the checkpoint's running statistics.
        # affine=False, so there is no weight/bias to load (flux2_ae.py:334-340).
        self.bn = nn.BatchNorm2d(
            patch_size * patch_size * z_channels,
            eps=bn_eps,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    @property
    def latent_channels(self) -> int:
        """Channels of the public latent: ``patch_size**2 * z_channels`` = 128."""
        return self.config.patch_size**2 * self.config.z_channels

    @property
    def spatial_factor(self) -> int:
        """Total stride: 8x convolutional times ``patch_size`` packing = 16."""
        return 2 ** (len(self.config.ch_mult) - 1) * self.config.patch_size

    # ---- 2x2 space-to-depth pack / unpack -----------------------------------

    def pack(self, z: Tensor) -> Tensor:
        """``[B, C, H, W] -> [B, C*p*p, H/p, W/p]``, channel-major.

        Bit-identical to the reference ``rearrange("... c (i pi) (j pj) -> ...
        (c pi pj) i j")`` (flux2_ae.py:349-357).  Note this is **not** diffusers'
        ``_pack_latents``, whose channel grouping is transposed.
        """
        return F.pixel_unshuffle(z, self.config.patch_size)

    def unpack(self, z: Tensor) -> Tensor:
        """Exact inverse of :meth:`pack` (flux2_ae.py:359-367)."""
        return F.pixel_shuffle(z, self.config.patch_size)

    # ---- latent normalisation ----------------------------------------------

    def normalize(self, z: Tensor) -> Tensor:
        """``(z - running_mean) / sqrt(running_var + bn_eps)``, per channel."""
        self.bn.eval()  # forced every call (flux2_ae.py:372); train mode shifts z by ~1.67
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        """Exact inverse of :meth:`normalize` — same ``bn_eps`` (flux2_ae.py:375-379)."""
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.config.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    # ---- public API ---------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode an image to the packed, normalised latent.

        Args:
            x: ``[B, 3, H, W]`` in ``[-1, 1]``; ``H`` and ``W`` divisible by 16.

        Returns:
            ``[B, 128, H/16, W/16]`` — the posterior **mean**, packed and
            BN-normalised.  The encoder's log-variance chunk is discarded
            (flux2_ae.py:396), so this is deterministic: there is no posterior
            distribution object and nothing to sample.
        """
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"encode expects [B, {self.config.in_channels}, H, W], got {tuple(x.shape)}"
            )
        h, w = x.shape[-2:]
        if h % self.spatial_factor or w % self.spatial_factor:
            raise ValueError(
                f"encode requires H and W divisible by {self.spatial_factor}, got {h}x{w}"
            )
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        return self.normalize(self.pack(mean))

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode a packed, normalised latent ``[B, 128, h, w]`` to ``[B, 3, 16h, 16w]``.

        The output is approximately ``[-1, 1]`` and is **not** clamped here; the
        pipeline applies ``(x * 0.5 + 0.5).clamp(0, 1)``.
        """
        if z.ndim != 4 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"decode expects [B, {self.latent_channels}, h, w], got {tuple(z.shape)}"
            )
        return self.decoder(self.unpack(self.inv_normalize(z)))

    # ---- construction / freezing -------------------------------------------

    @classmethod
    def from_single_file(
        cls,
        path: str | os.PathLike,
        torch_dtype: torch.dtype = torch.float32,
    ) -> "AutoencoderFlux2":
        """Load the single-file ``ae.safetensors`` (BFL key names) with ``strict=True``.

        The released file is the Apache-2.0 FLUX.2-klein-base-4B autoencoder
        re-keyed to this layout; it is stored in bfloat16 and is upcast to
        ``torch_dtype``.  ReFlowSET runs the autoencoder in float32.
        """
        from safetensors.torch import load_file

        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"FLUX.2 autoencoder weights not found at: {path}. Expected the "
                "single-file 'ae.safetensors' shipped with ReFlowSET."
            )
        model = cls()
        model.load_state_dict(load_file(path, device="cpu"), strict=True)
        model.to(dtype=torch_dtype)
        model.eval()
        model.requires_grad_(False)
        return model

    def train(self, mode: bool = True) -> "AutoencoderFlux2":
        """The autoencoder is frozen: never leave eval mode (flux2_ae.py:437-439)."""
        return super().train(False)
