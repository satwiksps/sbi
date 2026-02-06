# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import azula.denoise
import azula.noise
import azula.sample
import torch
from azula.denoise import DiracPosterior
from torch import Tensor

from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential


class SBISchedule(azula.noise.Schedule):
    """Adapts SBI's vector field estimator noise functions to Azula's Schedule."""

    def __init__(self, vector_field_estimator):
        self.vector_field_estimator = vector_field_estimator

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return self.vector_field_estimator.mean_t_fn(t)

    def sigma(self, t: Tensor) -> Tensor:
        return self.vector_field_estimator.std_fn(t)


class SBIDenoiser(azula.denoise.Denoiser):
    """Adapts SBI's vector field estimator to Azula's Denoiser interface."""

    def __init__(self, potential_fn: VectorFieldBasedPotential):
        schedule = SBISchedule(potential_fn.vector_field_estimator)
        super().__init__()
        self.schedule = schedule
        self.potential_fn = potential_fn

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        """Computes the denoised mean given noisy data and time.

        Args:
            x_t: Noisy data `(B, D)`.
            t: Time `(B,)` or `()`.

        Returns:
            Posterior distribution (Dirac delta at projected mean).
        """
        track_gradients = kwargs.get("track_gradients", False)

        with torch.set_grad_enabled(track_gradients):
            score = self.potential_fn.gradient(
                x_t, time=t, track_gradients=track_gradients
            )

        alpha, sigma = self.schedule(t)

        while alpha.ndim < x_t.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        # Recover x_0 (denoised mean) from score.
        x_0 = (x_t + sigma**2 * score) / alpha

        return DiracPosterior(mean=x_0)


class AzulaDiffuser:
    """Wrapper for Azula samplers to be used within SBI's VectorFieldPosterior."""

    def __init__(
        self,
        potential_fn: VectorFieldBasedPotential,
        sampler_type: str = "heun",
        sampler_params: Optional[Dict] = None,
    ):
        """Init method for AzulaDiffuser.

        Args:
            potential_fn: The potential function defining the vector field.
            sampler_type: Type of Azula sampler ('heun', 'euler', 'ddim', 'ddpm').
            sampler_params: Additional parameters for the sampler.
        """
        self.potential_fn = potential_fn
        self.sampler_type = sampler_type
        self.sampler_params = sampler_params or {}

        estimator = potential_fn.vector_field_estimator
        self.t_min = estimator.t_min
        self.t_max = estimator.t_max
        self.input_shape = estimator.input_shape

        self.condition_shape = estimator.condition_shape
        condition_dim = len(self.condition_shape)
        if potential_fn.x_o is None:
            self.batch_shape = torch.Size([])
        else:
            self.batch_shape = potential_fn.x_o.shape[:-condition_dim]

        self.denoiser = SBIDenoiser(potential_fn)

    def run(
        self,
        num_samples: int,
        ts: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        save_intermediate: bool = False,
    ) -> Tensor:
        """Run the Azula sampler.

        Args:
            num_samples: Number of samples per observation.
            ts: Time schedule. If None, default will be used (via steps).
            show_progress_bars: Whether to show progress bars.
            save_intermediate: Whether to save intermediate samples.

        Returns:
            Samples tensor.
        """
        steps = self.sampler_params.get("steps", 64)
        if ts is not None:
            steps = len(ts) - 1

        start = self.t_max
        stop = self.t_min

        sampler_map = {
            "euler": azula.sample.EulerSampler,
            "heun": azula.sample.HeunSampler,
            "ddim": azula.sample.DDIMSampler,
            "ddpm": azula.sample.DDPMSampler,
        }

        sampler_cls = sampler_map.get(self.sampler_type.lower())
        if sampler_cls is None:
            raise ValueError(
                f"Unknown sampler type: {self.sampler_type}. "
                f"Supported: {list(sampler_map.keys())}"
            )

        params = self.sampler_params.copy()
        if "steps" in params:
            del params["steps"]

        sampler = sampler_cls(
            denoiser=self.denoiser,
            start=start,
            stop=stop,
            steps=steps,
            silent=not show_progress_bars,
            device=self.potential_fn.device,
            **params,
        )

        num_batches = 1
        if self.potential_fn.x_is_iid:
            num_batches = self.potential_fn.x_o.shape[0]
        else:
            num_batches = self.batch_shape.numel()

        if num_batches == 0:
            num_batches = 1

        total_samples = num_samples * num_batches

        mean_base = self.potential_fn.vector_field_estimator.mean_base
        std_base = self.potential_fn.vector_field_estimator.std_base
        device = self.potential_fn.device

        shape = (total_samples,) + tuple(self.input_shape)

        if isinstance(mean_base, Tensor):
            mean_base = mean_base.to(device)
        if isinstance(std_base, Tensor):
            std_base = std_base.to(device)

        x_init = sampler.init(shape, mean=mean_base, var=std_base**2).to(device)

        if save_intermediate:
            x = x_init
            timesteps = sampler.timesteps()

            from tqdm.auto import tqdm

            pbar = tqdm(
                range(len(timesteps) - 1),
                disable=not show_progress_bars,
                desc="Sampling",
            )

            results = [x]

            for i in pbar:
                t = timesteps[i]
                s = timesteps[i + 1]
                x = sampler.step(x, t, s)
                results.append(x)

            combined = torch.stack(results, dim=1)
            return combined

        else:
            x_final = sampler(x_init)
            return x_final
