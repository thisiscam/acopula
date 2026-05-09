"""Sampler/likelihood consistency under ``survival=True``.

Until this fix, ``CompiledModel.sample()`` ignored the ``survival``
flag set at compile time: the sampler returned ``T = F^{-1}(U)`` while
the likelihood expected ``U = S(T) = 1 - F(T)`` to be jointly Clayton.
The data drawn from sample() therefore did not come from the
distribution the likelihood scored, with right and left tails swapped
on each margin.

These tests verify the round-trip:
  * the survival flag is persisted on CompiledModel;
  * survival vs CDF samples drawn from the same key are not bit-equal;
  * likelihood scores at the bulk of survival-sampled points are
    higher than at points far from that bulk.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from acopula import compile_model, copula, marginal


jax.config.update("jax_enable_x64", True)


@copula
class Clayton:
    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(1.0 + t, -1.0 / self.theta)

    def generator_inv(self, u: jax.Array) -> jax.Array:
        return jnp.power(u, -self.theta) - 1.0


# Custom Exponential margin with the API expected by acopula's leaves.
class Exponential:
    def __init__(self, rate=1.0):
        self.rate = rate

    @property
    def parameters(self):
        return {}

    def cdf(self, t):
        return 1.0 - jnp.exp(-self.rate * t)

    def quantile(self, u):
        # F^-1(u) = -log(1-u) / rate. We use log1p to keep precision near u=0.
        return -jnp.log1p(-u) / self.rate

    def log_prob(self, t):
        return jnp.log(self.rate) - self.rate * t


def _build_model():
    def model(params, obs):
        c = Clayton(params[0])
        return c([
            marginal(Exponential(rate=0.5), obs=obs[i]) for i in range(2)
        ])
    init = jnp.array([2.5])
    return model, init


def test_survival_flag_persisted_on_compiled_model():
    model, params = _build_model()
    cm = compile_model(model, template=params, survival=True)
    assert cm.survival is True
    cm2 = compile_model(model, template=params, survival=False)
    assert cm2.survival is False


def test_survival_sampler_differs_from_non_survival():
    """survival vs CDF should produce different samples under the same key.
    For Exponential, F^-1(1-u) = -log(u)/rate ≠ F^-1(u) = -log(1-u)/rate."""
    model, params = _build_model()
    cm_surv = compile_model(model, template=params, survival=True)
    cm_cdf = compile_model(model, template=params, survival=False)
    key = jrandom.PRNGKey(2)
    s_surv = np.asarray(cm_surv.sample(key, 32, params, method="rosenblatt"))
    s_cdf = np.asarray(cm_cdf.sample(key, 32, params, method="rosenblatt"))
    assert not np.allclose(s_surv, s_cdf), \
        "survival and CDF samples coincide — flag not threaded through"


def test_survival_sampler_returns_positive_observations():
    model, params = _build_model()
    cm = compile_model(model, template=params, survival=True)
    key = jrandom.PRNGKey(0)
    samples = np.asarray(cm.sample(key, 100, params, method="rosenblatt"))
    assert samples.shape == (100, 2)
    assert np.all(np.isfinite(samples)), "non-finite samples"
    assert np.all(samples > 0), \
        f"non-positive samples (min={samples.min()}) — survival inversion broken"


def test_survival_likelihood_higher_at_sampled_bulk():
    """Likelihood at sampled points should average higher than likelihood at
    points far outside the typical mass — a behavioural consistency check."""
    model, params = _build_model()
    cm = compile_model(model, template=params, survival=True)
    key = jrandom.PRNGKey(1)
    samples = np.asarray(cm.sample(key, 200, params, method="rosenblatt"))

    sample_lls = np.asarray(jax.vmap(
        lambda x: cm.eval(x, params))(jnp.asarray(samples[:50])))

    # Far points: triple the largest sample on each margin.
    far = jnp.full((50, 2), float(samples.max()) * 3.0)
    far_lls = np.asarray(jax.vmap(
        lambda x: cm.eval(x, params))(far))

    assert sample_lls.mean() > far_lls.mean(), (
        f"sample mean ll {sample_lls.mean():.3f} not > far mean ll "
        f"{far_lls.mean():.3f}"
    )


def test_survival_marshall_olkin_also_threads_flag():
    """The fix touches both samplers; verify Marshall-Olkin path too."""
    model, params = _build_model()
    cm_surv = compile_model(model, template=params, survival=True)
    cm_cdf = compile_model(model, template=params, survival=False)
    key = jrandom.PRNGKey(7)
    s_surv = np.asarray(cm_surv.sample(key, 32, params, method="marshall_olkin"))
    s_cdf = np.asarray(cm_cdf.sample(key, 32, params, method="marshall_olkin"))
    assert not np.allclose(s_surv, s_cdf), \
        "marshall_olkin: survival and CDF samples coincide — flag not threaded"
    assert np.all(s_surv > 0), "marshall_olkin survival samples not positive"
