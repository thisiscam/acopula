"""End-to-end tests for `compile_model(..., log_jet=True)`.

Validates the log-space Taylor-mode path through the bell pipeline:

1. **Equivalence on benign inputs** — at moderate `d` and non-extreme
   `u`, `log_jet=True` and `log_jet=False` must produce identical
   forward log-likelihoods and gradients (rel err ≤ 1e-10).
2. **Stability on pathological inputs** — at `d=75` flat-Frank with
   all leaves at `u=1e-3`, raw produces NaN gradient (the bug we
   built log-jet to fix); `log_jet=True` must produce a finite one.
3. **Nested 11x9 sp500-style failure case** — same finiteness
   guarantee on the structure that triggered the original report.

Run with: `pytest acopula/tests/test_log_jet.py -v`
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from acopula import compile_model, defmodel, marginal, copula
from tensorflow_probability.substrates import jax as tfp

Uniform = tfp.distributions.Uniform


@copula
class Frank:
    theta: float

    def generator(self, t):
        return -jnp.log1p(jnp.exp(-t) * jnp.expm1(-self.theta)) / self.theta


# ---------------------------------------------------------------------------
# Equivalence on benign inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,u_value", [
    (10, 0.5),
    (20, 0.3),
    (50, 0.1),
])
def test_log_jet_matches_raw_benign(d, u_value):
    """At moderate d and non-extreme u, log_jet must agree with raw to
    rel err 1e-10 on both forward and gradient."""
    @defmodel
    def model(theta, obs):
        return Frank(theta[0])([
            marginal(Uniform(0.0, 1.0), obs=obs[j]) for j in range(d)
        ])

    init = jnp.array([1.5])
    obs = jnp.full((d,), u_value)

    cm_raw = compile_model(model, template=init, method="bell")
    cm_log = compile_model(model, template=init, method="bell", log_jet=True)

    ll_raw = cm_raw.ll_fn(obs, init)
    ll_log = cm_log.ll_fn(obs, init)
    np.testing.assert_allclose(float(ll_log), float(ll_raw), rtol=1e-10)

    g_raw = jax.grad(cm_raw.ll_fn, argnums=1)(obs, init)
    g_log = jax.grad(cm_log.ll_fn, argnums=1)(obs, init)
    np.testing.assert_allclose(g_log, g_raw, rtol=1e-10)


# ---------------------------------------------------------------------------
# Stability on pathological inputs (the bug log-jet exists to fix)
# ---------------------------------------------------------------------------

def test_log_jet_finite_grad_at_d75_extreme():
    """Flat Frank at d=75 with all leaves at u=1e-3 makes raw produce
    NaN gradient via the (denormal Taylor coefficient → 1/x → inf →
    0×inf in dot_general) cascade. log_jet must produce a finite
    gradient."""
    d = 75

    @defmodel
    def model(theta, obs):
        return Frank(theta[0])([
            marginal(Uniform(0.0, 1.0), obs=obs[j]) for j in range(d)
        ])

    init = jnp.array([1.2])
    obs = jnp.full((d,), 1e-3)

    cm_raw = compile_model(model, template=init, method="bell")
    cm_log = compile_model(model, template=init, method="bell", log_jet=True)

    g_raw = jax.grad(cm_raw.ll_fn, argnums=1)(obs, init)
    g_log = jax.grad(cm_log.ll_fn, argnums=1)(obs, init)

    # Document the bug: raw produces NaN here.
    assert bool(jnp.any(jnp.isnan(g_raw))), \
        "raw should NaN at d=75 flat-Frank with all u=1e-3 — if this " \
        "starts passing, the bug got fixed elsewhere; either way, log_jet " \
        "should still be finite."
    assert bool(jnp.all(jnp.isfinite(g_log))), \
        f"log_jet grad must be finite, got {g_log}"


def test_log_jet_finite_grad_nested_sp500_pattern():
    """11x9 nested Frank with 5/9 extreme leaves per sector. This is
    the structure that originally triggered the sp500 d=99 failure."""
    n_sec, sec_size = 11, 9

    @defmodel
    def model(theta, obs):
        outer = Frank(theta[0])
        return outer([
            Frank(theta[1 + s])([
                marginal(Uniform(0.0, 1.0), obs=obs[s * sec_size + j])
                for j in range(sec_size)
            ])
            for s in range(n_sec)
        ])

    init = jnp.concatenate([jnp.array([1.2])] + [jnp.array([1.8])] * n_sec)
    obs = jnp.array(([0.000797] * 5 + [0.5] * 4) * n_sec)

    cm_raw = compile_model(model, template=init, method="bell")
    cm_log = compile_model(model, template=init, method="bell", log_jet=True)

    # Forward agreement (raw forward stays finite even where backward NaNs).
    # Looser tolerance than the benign case: log-space accumulation
    # through ~100 signed-logsumexp ops on extreme inputs picks up a
    # few ULPs of drift per op. 1e-7 rel tolerance is the realistic
    # bound for this kind of high-d extreme-input pipeline.
    ll_raw = cm_raw.ll_fn(obs, init)
    ll_log = cm_log.ll_fn(obs, init)
    np.testing.assert_allclose(float(ll_log), float(ll_raw), rtol=1e-7)

    g_raw = jax.grad(cm_raw.ll_fn, argnums=1)(obs, init)
    g_log = jax.grad(cm_log.ll_fn, argnums=1)(obs, init)

    assert bool(jnp.any(jnp.isnan(g_raw))), \
        "raw should NaN on the nested 11x9 pattern with 5 extremes per sector"
    assert bool(jnp.all(jnp.isfinite(g_log))), \
        f"log_jet grad must be finite, got {g_log}"


# ---------------------------------------------------------------------------
# Regression: log_jet=False (default) is byte-identical to before
# ---------------------------------------------------------------------------

def test_default_path_unchanged():
    """compile_model(...) without log_jet must behave exactly as before
    — the log_jet plumbing is invisible when not opted into."""
    d = 30

    @defmodel
    def model(theta, obs):
        return Frank(theta[0])([
            marginal(Uniform(0.0, 1.0), obs=obs[j]) for j in range(d)
        ])

    init = jnp.array([1.5])
    obs = jnp.full((d,), 0.4)

    # Two compile paths: explicit log_jet=False vs no kwarg at all.
    cm_explicit = compile_model(model, template=init, method="bell",
                                log_jet=False)
    cm_default = compile_model(model, template=init, method="bell")

    g_explicit = jax.grad(cm_explicit.ll_fn, argnums=1)(obs, init)
    g_default = jax.grad(cm_default.ll_fn, argnums=1)(obs, init)
    np.testing.assert_array_equal(g_explicit, g_default)
