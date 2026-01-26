# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor

from sbi.inference.potentials.vector_field_potential import (
    vector_field_estimator_based_potential,
)
from sbi.neural_nets import posterior_score_nn
from sbi.samplers.score.azula_diffuser import AzulaDiffuser


class DummyNet(torch.nn.Module):
    """A dummy network that returns zeros."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Linear(1, 1)

    def forward(self, input, condition, time):
        return torch.zeros_like(input)


class GradCheckPotential:
    """A simple potential class specifically for testing gradient flow."""
    def __init__(self):
        self.vector_field_estimator = type("E", (), {})()
        self.vector_field_estimator.t_min = 0.0
        self.vector_field_estimator.t_max = 1.0
        self.vector_field_estimator.mean_base = torch.zeros(1)
        self.vector_field_estimator.std_base = torch.ones(1)
        self.vector_field_estimator.input_shape = torch.Size((1,))
        self.vector_field_estimator.condition_shape = torch.Size([])
        self.x_is_iid = True
        self.iid_method = None

    def gradient(self, x, t):
        return x * t


def _build_gaussian_score_estimator(
    sde_type: str,
    input_event_shape: Tuple[int],
    mean0: Tensor,
    std0: Tensor,
):
    """Helper to build a functional SBI Potential without training."""
    building_thetas = (
        torch.randn((100, *input_event_shape), dtype=torch.float32) * std0 + mean0
    )
    building_xs = torch.ones((100, 1))
    score_estimator = posterior_score_nn(
        sde_type=sde_type,
        net=DummyNet(),
        embedding_net=torch.nn.Identity(),
    )(building_thetas, building_xs)
    score_fn, _ = vector_field_estimator_based_potential(
        score_estimator, prior=None, x_o=torch.ones((1,))
    )
    return score_fn


class MockAzulaSampler:
    """
    A compliant mock of an Azula sampler.
    It uses the passed 'denoiser' (the wrapped SBI potential) to perform
    a simplified Euler-Maruyama step.
    """
    def __init__(self, denoiser, step_size=0.1, **kwargs):
        self.denoiser = denoiser
        self.step_size = step_size
        self.kwargs = kwargs

    def __call__(self, x_init: Tensor, **kwargs) -> Tensor:
        """
        Simulates a diffusion process.
        Input x_init shape: (num_samples, num_batches, *event_shape)
        """
        x = x_init.clone()
        batch_size = x.shape[0]
        t = torch.ones(batch_size, device=x.device) * 0.5
        score = self.denoiser(x, t)
        x = x + self.step_size * score
        return x


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_event_shape", ((1,), (2,)))
def test_azula_integration_shapes(sde_type, input_event_shape):
    """
    Verifies that AzulaDiffuser correctly wraps the SBI potential
    and produces outputs of the expected shape.
    """
    mean0 = torch.zeros(input_event_shape)
    std0 = torch.ones(input_event_shape)
    score_fn = _build_gaussian_score_estimator(sde_type, input_event_shape, mean0, std0)
    diffuser = AzulaDiffuser(
        score_fn, MockAzulaSampler, step_size=0.01, custom_param="test"
    )
    assert diffuser.sampler.kwargs["custom_param"] == "test"
    num_samples = 10
    samples = diffuser.run(num_samples)
    expected_shape = (num_samples, 1, *input_event_shape)
    assert samples.shape == expected_shape


def test_azula_gradients_flow():
    """
    Verifies that gradients can be backpropagated through the wrapper.
    This is critical for optimization-based inference methods.
    """
    score_fn = GradCheckPotential()
    diffuser = AzulaDiffuser(score_fn, MockAzulaSampler)
    denoiser = diffuser.sampler.denoiser
    x = torch.tensor([[1.0]], requires_grad=True)
    t = torch.tensor([0.5])
    score = denoiser(x, t)
    loss = score.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([[0.5]]))


@pytest.mark.parametrize("batch_shape", [(1,), (5,)])
def test_azula_batch_broadcasting(batch_shape):
    """
    Tests if the wrapper handles Azula's way of passing time 't'.
    Azula often passes t as a 1D tensor [Batch], while x is [Batch, Dim].
    The wrapper must broadcast t correctly.
    """
    input_shape = (2,)
    mean0 = torch.zeros(input_shape)
    std0 = torch.ones(input_shape)
    score_fn = _build_gaussian_score_estimator("vp", input_shape, mean0, std0)
    diffuser = AzulaDiffuser(score_fn, MockAzulaSampler)
    denoiser = diffuser.sampler.denoiser
    batch_size = batch_shape[0]
    x = torch.randn(batch_size, 1, *input_shape)
    t = torch.rand(batch_size)
    out = denoiser(x, t)
    assert out.shape == x.shape
