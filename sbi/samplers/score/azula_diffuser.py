# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Optional, Type

import azula
import torch
from azula.sample import Sampler
from torch import Tensor, nn

from sbi.samplers.score.diffuser import Diffuser


class SBIDenoiser(nn.Module):
    """Wraps an SBI potential function to act as an Azula Denoiser."""

    def __init__(self, potential_fn: Any, event_shape: torch.Size):
        super().__init__()
        self.potential_fn = potential_fn
        self.event_ndim = len(event_shape)

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Computes the score (or denoiser output) for Azula."""
        batch_ndim = x.ndim - self.event_ndim
        if t.ndim < batch_ndim:
             t = t.view(t.shape + (1,) * (batch_ndim - t.ndim))

        return self.potential_fn.gradient(x, t)


class AzulaDiffuser(Diffuser):
    """Wrapper for Azula diffusion samplers to be compatible with SBI."""

    def __init__(
        self,
        vector_field_potential: Any,
        azula_sampler_cls: Type[Sampler],
        device: str = "cpu",
        **kwargs,
    ):
        """Init method for the AzulaDiffuser class.

        Args:
            vector_field_potential: The SBI potential function (score estimator).
            azula_sampler_cls: The Azula sampler class to use (e.g. DDPMSampler).
            device: Device to run on.
            **kwargs: Arguments passed to the azula_sampler_cls constructor.
        """
        self.potential_fn = vector_field_potential
        self.device = device
        self.azula_sampler_cls = azula_sampler_cls
        self.sampler_kwargs = kwargs

        # Extract limits and shapes from the estimator for compatibility
        estimator = vector_field_potential.vector_field_estimator
        self.t_min = estimator.t_min
        self.t_max = estimator.t_max
        self.init_mean = estimator.mean_base
        self.init_std = estimator.std_base
        self.input_shape = estimator.input_shape
        self.condition_shape = estimator.condition_shape

        condition_dim = len(self.condition_shape)
        if hasattr(vector_field_potential, "x_o"):
            self.batch_shape = vector_field_potential.x_o.shape[:-condition_dim]
        else:
            self.batch_shape = torch.Size([])

        # Initialize the Azula components
        self.denoiser = SBIDenoiser(self.potential_fn, self.input_shape).to(self.device)
        self.sampler = self.azula_sampler_cls(self.denoiser, **self.sampler_kwargs)
        self.predictor = None
        self.corrector = None

    def initialize(self, num_samples: int) -> Tensor:
        """Initialize the sampler by drawing samples from the initial distribution."""
        num_batches = (
            1 if self.potential_fn.x_is_iid else self.batch_shape.numel()
        )
        init_shape = (num_samples, num_batches) + self.input_shape
        init_std = self.init_std
        if (
            hasattr(self.potential_fn, "iid_method")
            and self.potential_fn.iid_method == "fnpe"
        ):
            x_o = self.potential_fn.x_o
            N_iid = x_o.shape[0]
            init_std = (1 / (N_iid**0.5)) * init_std
        eps = torch.randn(init_shape, device=self.device)
        mean, std, eps = torch.broadcast_tensors(self.init_mean, init_std, eps)
        return mean + std * eps

    def run(
        self,
        num_samples: int,
        ts: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        save_intermediate: bool = False,
    ) -> Tensor:
        """Samples from the distribution using the Azula sampler.

        Args:
            num_samples: Number of samples to draw.
            ts: Time discretization (currently unused by Azula wrapper).
            show_progress_bars: Whether to show progress (handled by Azula if supported).
            save_intermediate: Whether to save intermediate steps (not supported).
        """
        x_init = self.initialize(num_samples)
        samples = self.sampler(x_init)

        return samples
