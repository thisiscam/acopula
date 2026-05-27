"""Regression test for the dynamic-censoring mask-indexing bug.

`optimize_order()` reorders leaves, so the leaf position used to index the
dynamic-censoring arrays is not the caller's `obs_index`. The Bell path must
reindex the caller-supplied `censored_mask` into leaf order (see
`acopula.bell._mask_to_leaf_order`); without it, nested models with
heterogeneous panel sizes + censoring computed wrong log-densities (the WRONG
leaves got flagged censored). This test pins acopula's nested censored
log-density against an independent mpmath ground truth on exactly those
structures.

Bug history: a fully-uncensored size-4 panel beside a size-3 panel with one
censored leaf produced z^3 instead of z^4 (one shift dropped). Fixed by
permuting the mask once at each Bell entry point.
"""
import jax
import jax.numpy as jnp
import mpmath as mp
import pytest

from acopula import compile_model, marginal, copula

jax.config.update("jax_enable_x64", True)
mp.mp.dps = 50

# High-precision mpmath regression guard (nested finite-difference up to 7 dims
# at 50-digit precision). Correctness is load-bearing but the run is heavy, so
# keep it out of the fast PR gate and run it in the nightly full suite.
pytestmark = pytest.mark.slow

TOL = 1e-7


@copula
class Clayton:
    """acopula-convention Clayton: psi(t) = (1+t)^(-1/theta)."""
    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(1.0 + t, -1.0 / self.theta)


class _U:
    """Identity (copula-scale) marginal."""
    def cdf(self, x):
        return x

    def log_prob(self, x):
        return jnp.array(0.0)


def _psi(t, th):
    return (1 + t) ** (mp.mpf(-1) / th)


def _psii(u, th):
    return mp.mpf(u) ** (-th) - 1


def _mpmath_nested_logpdf(groups, theta_outer, theta_inner, u, mask):
    """Exact log of the mixed partial of the nested-Clayton CDF wrt the
    uncensored dims only (right-censored: censored dims enter the CDF
    argument but are not differentiated)."""
    tho = mp.mpf(repr(theta_outer))
    thin = [mp.mpf(repr(t)) for t in theta_inner]

    def C(vec):
        if len(groups) == 1:
            g = groups[0]
            return _psi(sum((_psii(vec[j], thin[0]) for j in g), mp.mpf(0)), thin[0])
        to = mp.mpf(0)
        for k, g in enumerate(groups):
            tk = sum((_psii(vec[j], thin[k]) for j in g), mp.mpf(0))
            to += _psii(_psi(tk, thin[k]), tho)
        return _psi(to, tho)

    unc = [j for j in range(len(u)) if not mask[j]]
    base = [mp.mpf(repr(x)) for x in u]

    def dn(vec, dims):
        if not dims:
            return C(vec)
        d0 = dims[0]
        return mp.diff(lambda x: dn([*vec[:d0], x, *vec[d0 + 1:]], dims[1:]),
                       vec[d0])

    return float(mp.log(dn(base, unc)))


def _acopula_nested_logpdf(groups, theta_outer, theta_inner, u, mask):
    def model(params, obs):
        outer = Clayton(params["theta_outer"])
        panels = []
        for k, dims in enumerate(groups):
            inner = Clayton(params[f"theta_inner_{k}"])
            panels.append(inner([
                marginal(_U(), obs=obs[j], censored=False) for j in dims]))
        if len(groups) == 1:
            return panels[0]
        return outer(panels)

    template = {"theta_outer": 0.3}
    for k in range(len(groups)):
        template[f"theta_inner_{k}"] = 1.0
    cm = compile_model(model, template=template, method="bell",
                       with_censored_mask=True)
    flat = jnp.asarray([theta_outer] + list(theta_inner[:len(groups)]))
    return float(cm.ll_fn(jnp.asarray(u), flat, jnp.asarray(mask)))


# Heterogeneous panels + censoring — the structures the bug corrupted, plus an
# uncensored control (correct before and after the fix).
_U7 = [0.7065, 0.5393, 0.8916, 0.7843, 0.0525, 0.8217, 0.0803]
CASES = [
    pytest.param([[0, 1, 2], [3, 4, 5, 6]], 0.8, [2.3, 4.1], _U7,
                 [True, False, False, False, False, False, False],
                 id="panels_3_4-cens_small_panel"),
    pytest.param([[0, 1], [2, 3, 4, 5]], 0.8, [2.3, 4.1], _U7[:6],
                 [True, False, False, False, False, False],
                 id="panels_2_4-cens_small_panel"),
    pytest.param([[0, 1], [2, 3], [4, 5, 6]], 0.8, [2.3, 4.1, 3.0], _U7,
                 [True, False, False, False, False, False, False],
                 id="panels_2_2_3-cens_first"),
    pytest.param([[0, 1, 2], [3, 4, 5, 6]], 0.8, [2.3, 4.1], _U7,
                 [False, False, True, False, False, False, True],
                 id="panels_3_4-cens_2"),
    pytest.param([[0, 1, 2], [3, 4, 5, 6]], 0.8, [2.3, 4.1], _U7,
                 [False] * 7, id="panels_3_4-uncensored_control"),
]


@pytest.mark.parametrize("groups,theta_outer,theta_inner,u,mask", CASES)
def test_nested_censoring_matches_mpmath(groups, theta_outer, theta_inner,
                                         u, mask):
    truth = _mpmath_nested_logpdf(groups, theta_outer, theta_inner, u, mask)
    got = _acopula_nested_logpdf(groups, theta_outer, theta_inner, u, mask)
    assert abs(got - truth) < TOL, (
        f"acopula nested censored log-density {got} != mpmath {truth} "
        f"(diff {abs(got - truth):.2e}); heterogeneous-panel mask-indexing "
        f"regression?")
