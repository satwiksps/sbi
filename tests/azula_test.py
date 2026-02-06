# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
)

try:
    import azula  # noqa: F401

    AZULA_AVAILABLE = True
except ImportError:
    AZULA_AVAILABLE = False


@pytest.mark.skipif(not AZULA_AVAILABLE, reason="Azula not installed")
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("sampler_type", ["heun", "euler"])
def test_azula_linear_gaussian(sampler_type):
    """Checks that Azula samplers run and return the correct shape on a simple task."""
    num_dim = 2
    num_samples = 200
    num_simulations = 1500

    x_o = zeros(1, num_dim)
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    inference = FMPE(prior, show_progress_bars=False)
    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
    inference.append_simulations(theta, x)
    density_estimator = inference.train(max_num_epochs=5, show_train_summary=False)

    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(x_o)

    try:
        samples = posterior.sample(
            (num_samples,),
            sample_with="azula",
            predictor=sampler_type,
            predictor_params={"steps": 20},
            show_progress_bars=False,
        )
        assert samples.shape == (num_samples, num_dim)

    except Exception as e:
        pytest.fail(f"Azula sampling failed: {e}")
