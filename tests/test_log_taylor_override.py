"""The log-space generator_taylor_coefficients override must agree with the
linear override (and with jet where jet is still accurate).

Regression test for ``Copula.log_generator_taylor_coefficients`` and the Bell
dispatch that prefers it. Uses AMH, whose frailty series is computed in the log
domain anyway, so the linear override is just the log override exponentiated.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from acopula import compile_model, copula, marginal


class _Uniform:
    def quantile(self, u): return u
    def cdf(self, x): return x
    def log_prob(self, x): return 0.0


def _amh_sign_logabs(theta, t, k_max, n_terms=300):
    ks = jnp.arange(k_max + 1)
    xs = jnp.arange(1, n_terms + 1, dtype=jnp.float64)
    log_w = jnp.log(1.0 - theta) + (xs - 1.0) * jnp.log(theta) - t * xs
    log_fact_k = jax.scipy.special.gammaln(ks + 1)
    log_term = (ks[:, None] * jnp.log(xs)[None, :] + log_w[None, :]
                - log_fact_k[:, None])
    log_abs_c = jax.scipy.special.logsumexp(log_term, axis=1)
    signs = jnp.where(ks % 2 == 0, 1.0, -1.0)
    return signs, log_abs_c


@copula
class _AMHLinear:
    theta: float
    def generator(self, t):
        return (1.0 - self.theta) / (jnp.exp(t) - self.theta)
    def generator_taylor_coefficients(self, t, k_max, n_terms=300):
        signs, log_abs = _amh_sign_logabs(self.theta, t, k_max, n_terms)
        return signs * jnp.exp(log_abs)


@copula
class _AMHLog:
    theta: float
    def generator(self, t):
        return (1.0 - self.theta) / (jnp.exp(t) - self.theta)
    def log_generator_taylor_coefficients(self, t, k_max, n_terms=300):
        return _amh_sign_logabs(self.theta, t, k_max, n_terms)


def _ll(cls, d, theta_val):
    def model(p, u):
        return cls(p[0])([marginal(_Uniform(), obs=u[i]) for i in range(d)])
    cm = compile_model(model, template=jnp.array([theta_val]), method="bell")
    u = jnp.arange(1, d + 1, dtype=jnp.float64) / (d + 1)
    return float(cm.eval(u, jnp.array([theta_val])))


@pytest.mark.parametrize("d", [8, 12])
def test_log_override_matches_linear(d):
    lin = _ll(_AMHLinear, d, 0.5)
    log = _ll(_AMHLog, d, 0.5)
    assert abs(lin - log) < 1e-9, f"d={d}: linear={lin}, log={log}"
