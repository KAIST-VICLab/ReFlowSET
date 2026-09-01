"""ReFlowSET's Design-B flow bridge and its explicit-Euler solver.

Forward (training) process, with ``eps ~ N(0, I)`` and ``z_e`` the EO latent::

    z_t = (1 - t) * eps + t * z_e          (bridge.py:311, sigma_b = 0)
    u*  = z_e - eps                        (bridge.py:328 at sigma_b = 0)

Sampling starts from ``z_0 ~ N(0, I)`` and integrates the predicted velocity
with explicit Euler on a uniform grid ``linspace(0, t_end, nfe + 1)``
(bridge.py:519, 536).  The bridge is deterministic: ``sigma_b = 0``, so no
stochastic term ever executes, and the only randomness in a sample is the
initial noise draw.

**Time direction.**  ``t = 0`` is NOISE and ``t = 1`` is DATA, and the solver
integrates ``t`` **ascending** (bridge.py:86-88).  That is the opposite of
`diffusers`' ``sigma`` convention: setting ``sigma := 1 - t`` recovers
``FlowMatchEulerDiscreteScheduler``'s interpolation, but then this bridge's
velocity is the **negative** of the diffusers flow-matching target and the
network must still be fed ``1 - sigma``.  This scheduler keeps ReFlowSET's own
sign and direction so neither flip is needed; ``timesteps`` therefore *increase*
from 0 towards 1, unlike every noise-schedule scheduler in `diffusers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class FlowBridgeSchedulerOutput(BaseOutput):
    """Output of :meth:`FlowBridgeScheduler.step`.

    Args:
        prev_sample: The bridge state at the next time on the grid.
    """

    prev_sample: torch.Tensor


class FlowBridgeScheduler(SchedulerMixin, ConfigMixin):
    """Explicit-Euler solver for ReFlowSET's Design-B flow bridge.

    Args:
        t_end: End time of the integration grid (1.0 — the EO endpoint).  The
            model is evaluated at ``linspace(0, t_end, nfe + 1)[:-1]`` and the
            final Euler step lands on ``t_end``; the network is never queried at
            ``t = t_end``.
    """

    order = 1

    @register_to_config
    def __init__(self, t_end: float = 1.0) -> None:
        if not 0.0 < t_end <= 1.0:
            raise ValueError(f"t_end must lie in (0, 1], got {t_end}")
        self._grid: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None
        self.num_inference_steps: Optional[int] = None

    @property
    def timesteps(self) -> torch.Tensor:
        """The ``nfe`` bridge times at which the model is evaluated, ascending."""
        if self._grid is None:
            raise ValueError("call set_timesteps() before reading timesteps")
        return self._grid[:-1]

    @property
    def step_index(self) -> Optional[int]:
        """Index of the next grid interval; ``None`` until the first :meth:`step`."""
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Build the uniform grid ``linspace(0, t_end, num_inference_steps + 1)``.

        Args:
            num_inference_steps: NFE — the number of velocity evaluations.
                50 reproduces the paper's main results; 4 is the efficiency
                operating point.
            device: Device the grid is built on.

        There is no shift, no dynamic shifting, no Karras or exponential
        spacing, and no timestep-spacing option: the reference solver uses a
        plain uniform grid (bridge.py:519).
        """
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        self.num_inference_steps = num_inference_steps
        self._grid = torch.linspace(
            0.0, self.config.t_end, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowBridgeSchedulerOutput, tuple[torch.Tensor]]:
        """One explicit-Euler step: ``z + (t_next - t_cur) * v`` (bridge.py:536).

        Args:
            model_output: The predicted velocity ``dz/dt`` at ``timestep``,
                already classifier-free-guided by the caller.
            timestep: The current bridge time.  Present for API compatibility and
                checked against the grid; the step size comes from the grid.
            sample: The current bridge state.
            return_dict: Return a :class:`FlowBridgeSchedulerOutput` instead of a
                tuple.

        Steps must be taken in order, starting from the first entry of
        :attr:`timesteps`.
        """
        if self._grid is None or self._step_index is None:
            raise ValueError("call set_timesteps() before step()")
        if self._step_index >= self.num_inference_steps:
            raise ValueError(
                f"already took {self.num_inference_steps} steps; call set_timesteps() again"
            )
        t_cur, t_next = self._grid[self._step_index], self._grid[self._step_index + 1]
        if not torch.isclose(torch.as_tensor(timestep, dtype=torch.float32).to(t_cur.device), t_cur):
            raise ValueError(
                f"step {self._step_index} expects timestep {t_cur.item()}, got {float(timestep)}; "
                "the flow bridge must be integrated in ascending grid order"
            )

        # The state is carried in float32 even if the model ran lower (bridge.py:515-517).
        dtype = sample.dtype if sample.dtype in (torch.float32, torch.float64) else torch.float32
        prev_sample = sample.to(dtype) + (t_next - t_cur) * model_output.to(dtype)
        prev_sample = prev_sample.to(sample.dtype)

        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowBridgeSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """The training-side bridge state ``z_t = (1 - t) * eps + t * z_e`` (bridge.py:311).

        Args:
            original_samples: The EO latent ``z_e`` (the ``t = 1`` endpoint).
            noise: ``eps ~ N(0, I)`` (the ``t = 0`` endpoint).
            timesteps: Bridge times in ``[0, 1]``, broadcastable over the batch.
        """
        t = timesteps.to(original_samples.device, original_samples.dtype)
        t = t.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1.0 - t) * noise + t * original_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """The training target ``u* = z_e - eps`` (bridge.py:328 at ``sigma_b = 0``).

        Constant along the path, hence independent of ``timesteps``; the argument
        is kept for `diffusers` API compatibility.
        """
        del timesteps
        return sample - noise
