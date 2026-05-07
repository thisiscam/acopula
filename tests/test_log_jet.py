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

from acopula import compile_model, marginal, copula
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

def test_log_jet_matches_mpmath_at_d75_extreme():
    """Flat Frank at d=75 with all leaves at u=1e-3 sits in the high-d
    extreme regime where the raw bell scan suffers catastrophic
    cancellation (NaN gradient on GPU; finite-but-wrong forward on CPU,
    off from truth by ~10^3).  log_jet is the correct path here, and
    must match a 200-digit mpmath reference computed via Frank's
    closed-form psi^{(d)} expansion in Stirling numbers."""
    from mpmath import mp, mpf, exp, log, fabs, fsum, factorial, stirling2

    d = 75

    def model(theta, obs):
        return Frank(theta[0])([
            marginal(Uniform(0.0, 1.0), obs=obs[j]) for j in range(d)
        ])

    init = jnp.array([1.2])
    obs = jnp.full((d,), 1e-3)

    cm_log = compile_model(model, template=init, method="bell", log_jet=True)

    # 200-digit mpmath reference.  Frank's psi^{(d)}(t) has the closed form
    #   psi^{(d)}(t) = ((-1)^d / theta) * sum_{k=1}^{d} (k-1)! S(d,k) (x/(1-x))^k
    # with x = (1 - e^{-theta}) e^{-t}.  Single-layer log-density:
    #   log c = log|psi^{(d)}(t)| + sum_j log|d psi^{-1}/du(u_j)|
    # where t = sum_j psi^{-1}(u_j).
    mp.dps = 200
    th = mpf("1.2")
    u_val = mpf("0.001")

    def _alpha(t):
        return 1 - exp(-t)

    def _psi_inv(u, t):
        return -log((1 - exp(-t * u)) / _alpha(t))

    def _log_psi_d_at_t(t_val, t, n):
        x = _alpha(t) * exp(-t_val)
        one_minus_x = 1 - x
        terms = [factorial(k - 1) * stirling2(n, k) * (x / one_minus_x) ** k
                 for k in range(1, n + 1)]
        psi_d = ((-1) ** n / t) * fsum(terms)
        return log(fabs(psi_d))

    def _log_psi_inv_prime(u, t):
        return log(fabs(t * exp(-t * u) / (1 - exp(-t * u))))

    def _log_density(t):
        t_inner = d * _psi_inv(u_val, t)
        return _log_psi_d_at_t(t_inner, t, d) + d * _log_psi_inv_prime(u_val, t)

    ll_ref = float(_log_density(th))
    eps = mpf("1e-30")
    g_ref = float((_log_density(th + eps) - _log_density(th - eps)) / (2 * eps))

    ll_log = float(cm_log.ll_fn(obs, init))
    g_log = float(jax.grad(cm_log.ll_fn, argnums=1)(obs, init)[0])

    np.testing.assert_allclose(
        ll_log, ll_ref, rtol=1e-8,
        err_msg=f"log_jet forward {ll_log} vs mpmath {ll_ref}",
    )
    np.testing.assert_allclose(
        g_log, g_ref, rtol=1e-8,
        err_msg=f"log_jet grad {g_log} vs mpmath {g_ref}",
    )


def test_log_jet_robust_at_nested_sp500_pattern():
    """11x9 nested Frank with 5/9 extreme leaves per sector — the d=99
    structure that originally triggered the sp500 NaN.  Raw bell scan is
    unreliable here in two platform-dependent ways, both manifestations of
    the same catastrophic-cancellation fragility in the polynomial-powering
    accumulator:

      - On GPU: raw forward stays approximately correct, but reverse-mode
        autodiff hits 0 * inf inside dot_general and produces NaN gradients.
      - On CPU: forward already loses precision in the polynomial scan
        (BLAS denormal handling differs from CUDA), so the wrong-but-finite
        forward propagates to a wrong-but-finite gradient.

    log_jet is robust on both platforms.  Its correctness is anchored by
    `test_log_jet_matches_mpmath_at_d75_extreme`, which validates the same
    Bell-polynomial machinery against a 200-digit mpmath reference.  An
    mpmath reference for the nested 11x9 case would require reimplementing
    acopula's nested scan at arbitrary precision, so we use log_jet as the
    de-facto reference here and lock in the *finite-ness* of its gradient
    plus the *unreliability* of raw."""
    n_sec, sec_size = 11, 9

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

    ll_raw = cm_raw.ll_fn(obs, init)
    ll_log = cm_log.ll_fn(obs, init)
    g_raw = jax.grad(cm_raw.ll_fn, argnums=1)(obs, init)
    g_log = jax.grad(cm_log.ll_fn, argnums=1)(obs, init)

    # Primary correctness: log_jet stays finite at the extreme.
    assert bool(jnp.all(jnp.isfinite(g_log))), \
        f"log_jet grad must be finite, got {g_log}"

    # Secondary regression lock: raw is unreliable here.  Either failure
    # mode counts as "unreliable" — the test passes as long as raw fails
    # in one of the two documented ways, and raises if raw silently starts
    # producing the right answer (which would mean either the underlying
    # bug got fixed elsewhere or the test point drifted out of the
    # extreme regime).
    raw_nan_grad = bool(jnp.any(jnp.isnan(g_raw)))
    raw_wrong_forward = not bool(jnp.allclose(ll_log, ll_raw, rtol=1e-3))
    assert raw_nan_grad or raw_wrong_forward, (
        f"raw should be unreliable at this point: expected NaN gradient "
        f"(GPU) or forward disagreement with log_jet by > 1e-3 (CPU).  "
        f"Got ll_raw={float(ll_raw):.6f}, ll_log={float(ll_log):.6f}, "
        f"g_raw[0]={float(g_raw[0]):.6f}, g_log[0]={float(g_log[0]):.6f}."
    )


# ---------------------------------------------------------------------------
# Regression: log_jet=False (default) is byte-identical to before
# ---------------------------------------------------------------------------

def test_default_path_unchanged():
    """compile_model(...) without log_jet must behave exactly as before
    — the log_jet plumbing is invisible when not opted into."""
    d = 30

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
