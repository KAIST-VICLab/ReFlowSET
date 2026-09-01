"""ReFlowSET velocity transformer — a latent DiT with an EO/SAR double stream.

The network predicts the flow-bridge velocity ``dz/dt`` in the frozen FLUX.2
latent space.  It takes the noisy EO latent ``[B, 128, h, w]``, a scalar bridge
time ``t`` in ``[0, 1]``, and the SAR conditioning latent of the same shape; the
first 8 of its 24 blocks are double-stream (one EO tower and one SAR tower over
a single joint attention), the remaining 16 are single-stream over the
concatenated ``[EO | SAR]`` sequence, and only the EO half is decoded.

This is an inference-only port.  The training-only REPA projection head
(``repa_proj``) is a separate module in the reference implementation and is
deliberately absent here.

Deviations from `diffusers`' FLUX blocks that this file has to keep — each one
is silent if you get it wrong:

* ``FinalLayer`` unpacks ``shift, scale`` (dit.py:360), the **opposite** order of
  ``AdaLayerNormContinuous``.
* The single-stream MLP is **SwiGLU** of width 2752, not a 4x GELU of width 4096.
* ``linear1``/``linear2`` are **bias-free**, and the QK-norm parameter is called
  ``scale``, not ``weight``.
* The timestep is multiplied by 1000 *inside* the model and the sinusoid is
  **cos first, then sin**.
* RoPE runs on **two** axes of **centred half-integer** coordinates, not on
  FLUX's three axes of integers starting at 0.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import functional as F

#: Width of the sinusoidal timestep embedding fed to ``time_in`` (dit.py:44).
#: A module constant, deliberately independent of ``hidden_size``.
TIME_EMBED_DIM = 256


def swiglu_hidden_dim(hidden_size: int, mlp_ratio: float) -> int:
    """SwiGLU intermediate width (dit.py:120-127).

    The canonical 2/3 rule rounded to a multiple of 64, so a gated MLP at
    ``mlp_ratio=4.0`` costs the same parameters as a plain 4x GELU MLP.
    ``hidden_size=1024, mlp_ratio=4.0 -> 2752``.
    """
    return int(round(hidden_size * mlp_ratio * 2 / 3 / 64)) * 64


class SwiGLU(nn.Module):
    """``silu(first half) * second half`` — gate first, value second (dit.py:130-133)."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class RMSNorm(nn.Module):
    """RMS norm computed in float32.  The parameter is named ``scale`` (dit.py:136-145)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """Per-head query/key RMS norm, applied **before** RoPE (dit.py:148-155)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class MLPEmbedder(nn.Module):
    """``Linear -> SiLU -> Linear`` time-embedding MLP (dit.py:158-166).

    Checkpoint keys are ``time_in.in_layer.*`` / ``time_in.out_layer.*``, not
    diffusers' ``time_text_embed.timestep_embedder.linear_{1,2}``.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Modulation(nn.Module):
    """AdaLN-Zero triple.  Order is ``shift, scale, gate`` (dit.py:169-181)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        shift, scale, gate = out.chunk(3, dim=-1)
        return shift, scale, gate


def timestep_embedding(
    t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0
) -> Tensor:
    """Sinusoidal embedding of a fractional bridge time (dit.py:184-201).

    Two things differ from `diffusers`' ``get_timestep_embedding`` defaults:
    ``t`` is a fraction in ``[0, 1]`` that is scaled by ``time_factor = 1000``
    **here**, and the concatenation order is ``[cos, sin]`` (FLUX's ordering,
    i.e. ``flip_sin_to_cos=True``).
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Per-axis rotation matrices ``[..., L, dim/2, 2, 2]`` (dit.py:204-211)."""
    if dim % 2:
        raise ValueError(f"RoPE axis dim must be even, got {dim}")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate consecutive dimension pairs — the interleaved (FLUX) convention.

    ``(x0, x1) -> (cos*x0 - sin*x1, sin*x0 + cos*x1)`` on ``(x[2k], x[2k+1])``
    (dit.py:214-219).  Equivalent to diffusers' ``apply_rotary_emb(...,
    use_real_unbind_dim=-1)``; ``-2`` is the split-halves convention and is wrong
    for these weights.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """Concatenates the per-axis RoPE ladders and inserts the head axis (dit.py:222-234).

    Holds no parameters and no buffers: the grid is rebuilt on every forward,
    which is what lets one checkpoint serve 256 and 512 inputs.
    """

    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)


def latent_image_ids(h: int, w: int, device, dtype=torch.float32) -> Tensor:
    """Centred ``(y, x)`` coordinates for an ``h x w`` latent grid, ``[h*w, 2]``.

    ``arange(n) - (n - 1) / 2`` with unit spacing (dit.py:237-252), so for even
    ``n`` the coordinates are half-integers and the central 16x16 region of a
    32x32 grid carries exactly the coordinates a 256-trained model saw — RoPE
    only extrapolates outwards, it never rescales.  Row-major, so token
    ``p = y * w + x``.  This is **not** FLUX's 3-axis integer id grid.
    """
    y = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
    x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2
    ids = torch.zeros(h, w, 2, device=device, dtype=dtype)
    ids[..., 0] = y[:, None]
    ids[..., 1] = x[None, :]
    return ids.reshape(h * w, 2)


class SingleStreamBlock(nn.Module):
    """Fused attention + SwiGLU MLP under one modulation and one residual.

    ``linear1`` emits ``[q | k | v | mlp_gate | mlp_value]`` in that order; the
    qkv slab is K-major (``(K H D)``).  Both linears are bias-free
    (dit.py:255-300).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = swiglu_hidden_dim(hidden_size, mlp_ratio)

        self.linear1 = nn.Linear(hidden_size, 3 * hidden_size + 2 * self.mlp_hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SwiGLU()
        self.modulation = Modulation(hidden_size)

    def pre_attention(self, x: Tensor, vec: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Everything up to (not including) RoPE and attention (dit.py:272-288)."""
        shift, scale, gate = self.modulation(vec)
        x_mod = (1 + scale) * self.pre_norm(x) + shift

        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, 2 * self.mlp_hidden_dim], dim=-1
        )
        b, length, _ = qkv.shape
        # "B L (K H D) -> K B H L D" with K=3, H=num_heads.
        q, k, v = qkv.reshape(b, length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        return q, k, v, mlp, gate

    def post_attention(self, x: Tensor, attn: Tensor, mlp: Tensor, gate: Tensor) -> Tensor:
        """Output projection and the single gated residual (dit.py:290-294)."""
        b, heads, length, head_dim = attn.shape
        attn = attn.transpose(1, 2).reshape(b, length, heads * head_dim)
        out = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + gate * out

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        q, k, v, mlp, gate = self.pre_attention(x, vec)
        q, k = apply_rope(q, k, pe)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.post_attention(x, attn, mlp, gate)


class DoubleStreamBlock(nn.Module):
    """Two independent towers over **one** joint attention across ``[EO | SAR]``.

    The towers have completely separate weights but share the modulation vector
    ``vec`` and the RoPE grid, so an EO token and the SAR token at the same
    ground position carry an identical phase (dit.py:303-343).  SAR plays the
    structural role text plays in FLUX.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.eo = SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
        self.sar = SingleStreamBlock(hidden_size, num_heads, mlp_ratio)

    def forward(self, eo: Tensor, sar: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """``pe`` must already cover the joint 2P-token sequence."""
        q_e, k_e, v_e, mlp_e, gate_e = self.eo.pre_attention(eo, vec)
        q_s, k_s, v_s, mlp_s, gate_s = self.sar.pre_attention(sar, vec)

        q = torch.cat((q_e, q_s), dim=2)
        k = torch.cat((k_e, k_s), dim=2)
        v = torch.cat((v_e, v_s), dim=2)
        q, k = apply_rope(q, k, pe)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_e, attn_s = attn.split([q_e.shape[2], q_s.shape[2]], dim=2)
        return (
            self.eo.post_attention(eo, attn_e, mlp_e, gate_e),
            self.sar.post_attention(sar, attn_s, mlp_s, gate_s),
        )


class FinalLayer(nn.Module):
    """AdaLN output layer.

    ``adaLN`` unpacks ``shift, scale`` — the **opposite** order of diffusers'
    ``AdaLayerNormContinuous`` (dit.py:346-362).  ``logvar_proj`` belongs to a
    beta-NLL loss that was never enabled (``loss.flow = mse``); its weights are
    kept so the published checkpoint loads with ``strict=True``, but inference
    never evaluates it — the sampler reads only the velocity (bridge.py:531).
    """

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, out_channels, bias=True)
        self.logvar_proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        mod = self.adaLN(F.silu(vec))
        if mod.ndim == 2:
            mod = mod[:, None, :]
        shift, scale = mod.chunk(2, dim=-1)
        return self.proj((1 + scale) * self.norm(x) + shift)


class ReFlowSETTransformer2DModel(ModelMixin, ConfigMixin):
    """ReFlowSET's flow-velocity transformer (509.32 M parameters as configured).

    Args:
        in_channels: Channels of the packed FLUX.2 latent (128).
        out_channels: Channels of the predicted velocity (128).
        hidden_size: Residual width (1024).
        depth: **Total** blocks, double plus single (24).
        num_heads: Attention heads (16), so ``head_dim = 64``.
        mlp_ratio: Nominal MLP ratio; the SwiGLU width is derived from it.
        axes_dim: RoPE dims for the ``(y, x)`` axes; must sum to ``head_dim``.
        theta: RoPE base period (10000).
        sample_size: Input image resolution the released arm was trained at
            (256 for QXS-SAROPT, 512 for SAR2Opt).  Recorded for provenance
            only: the forward pass derives every shape from its input and the
            RoPE grid is rebuilt per call, so one checkpoint serves any size
            divisible by 16.
        double_blocks: Leading double-stream blocks (8); the remaining
            ``depth - double_blocks`` are single-stream.
        double_merge: How the two streams become one.  ``"token"`` (the released
            setting) concatenates on the sequence axis, so the single stack runs
            over 2P tokens and the SAR half is dropped only at the very end;
            ``"channel"`` fuses per position and keeps P tokens.

    Forward contract: ``forward(hidden_states, timestep, condition)`` where
    ``hidden_states`` is the bridge state ``[B, 128, h, w]``, ``timestep`` is the
    bridge time in ``[0, 1]`` (**not** an integer diffusion step), and
    ``condition`` is the SAR latent of the same shape or ``None``.  ``None`` is
    the classifier-free-guidance null branch and is turned into an all-zero
    latent inside the model — there is no learned null token.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        axes_dim: tuple[int, ...] = (32, 32),
        theta: int = 10000,
        sample_size: Optional[int] = None,
        double_blocks: int = 8,
        double_merge: str = "token",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"axes_dim {list(axes_dim)} must sum to the per-head dim {pe_dim}")
        if not 0 <= double_blocks < depth:
            raise ValueError(f"double_blocks {double_blocks} must be in [0, depth={depth})")
        if double_merge not in ("token", "channel"):
            raise ValueError(f"double_merge must be 'token' or 'channel', got {double_merge!r}")

        self.pe_embedder = EmbedND(theta=theta, axes_dim=list(axes_dim))
        if double_blocks:
            # Each stream gets its own 1x1 "patchify": they are two token
            # sequences now, not two halves of one channel stack.
            self.in_proj_eo = nn.Linear(in_channels, hidden_size, bias=True)
            self.in_proj_sar = nn.Linear(in_channels, hidden_size, bias=True)
            if double_merge == "channel":
                self.merge = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        else:
            self.in_proj = nn.Linear(2 * in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(TIME_EMBED_DIM, hidden_size)
        self.double_stream = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(double_blocks)]
        )
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
                for _ in range(depth - double_blocks)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, out_channels)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        condition: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, tuple[Tensor]]:
        """Predict the flow velocity ``dz/dt``.

        Args:
            hidden_states: ``[B, in_channels, h, w]`` bridge state.
            timestep: Bridge time in ``[0, 1]``; a scalar or ``[B]``.
            condition: ``[B, in_channels, h, w]`` SAR latent, or ``None`` for the
                null branch (an all-zero conditioning latent, dit.py:531-532).
            return_dict: Return a ``Transformer2DModelOutput`` instead of a tuple.

        Returns:
            The velocity ``[B, out_channels, h, w]``.  This is a flow velocity,
            not ``epsilon`` and not diffusers' ``v_prediction``.
        """
        if hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [B, C, h, w], got {tuple(hidden_states.shape)}")
        batch, _, h, w = hidden_states.shape
        if condition is None:
            condition = torch.zeros_like(hidden_states)
        elif condition.shape != hidden_states.shape:
            raise ValueError(
                f"condition shape {tuple(condition.shape)} must match "
                f"hidden_states shape {tuple(hidden_states.shape)}"
            )
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)

        n_double = self.config.double_blocks
        if n_double:
            eo = self.in_proj_eo(hidden_states.flatten(2).transpose(1, 2))  # [B, P, D]
            sar = self.in_proj_sar(condition.flatten(2).transpose(1, 2))  # [B, P, D]
            ref = eo
        else:
            x = torch.cat([hidden_states, condition], dim=1).flatten(2).transpose(1, 2)
            x = self.in_proj(x)
            ref = x

        vec = self.time_in(timestep_embedding(timestep, TIME_EMBED_DIM).to(ref.dtype))

        ids = latent_image_ids(h, w, device=hidden_states.device, dtype=torch.float32)
        pe = self.pe_embedder(ids[None].expand(batch, -1, -1))

        num_tokens = ref.shape[1]
        pe_single = pe
        if n_double:
            # Token axis of pe is dim 2 ([B, 1, L, head_dim/2, 2, 2]); repeating
            # the same P coordinates gives EO and SAR one shared grid.
            pe_joint = torch.cat((pe, pe), dim=2)
            for block in self.double_stream:
                eo, sar = block(eo, sar, vec, pe_joint)
            if self.config.double_merge == "token":
                x = torch.cat((eo, sar), dim=1)  # [B, 2P, D]
                pe_single = pe_joint
            else:
                x = self.merge(torch.cat((eo, sar), dim=-1))  # [B, P, D]

        for block in self.blocks:
            x = block(x, vec, pe_single)

        if n_double and self.config.double_merge == "token":
            x = x[:, :num_tokens]  # drop the SAR half: only EO is decoded

        v = self.final_layer(x, vec)
        v = v.transpose(1, 2).reshape(batch, self.config.out_channels, h, w)
        if not return_dict:
            return (v,)
        return Transformer2DModelOutput(sample=v)
