import jax
import jax.numpy as jnp
import pytest
from acopula import defmodel, marginal, copula
import jax.scipy.special
from oryx import core as oryx_core
# =============================
# 1. Model & Copula Definitions
# =============================


@copula
class Independence:
    def generator(self, t: jax.Array) -> jax.Array:
        return -jnp.log(t)


@copula
class No12:
    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return 1 / (1 + t ** (1 / self.theta))


# =============================
# Table 2.2 families (by number)
# =============================
# Implement ψ_i(t) as in the paper's table (t >= 0).


@copula
class No1:
    """Family 1: ψ(t) = (1 + t)^(-1/θ)  (Clayton)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(1.0 + t, -1.0 / self.theta)


Clayton = No1


@copula
class No3:
    """Family 3: ψ(t) = (1-θ) / (exp(t) - θ)  (Ali-Mikhail-Haq)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return (1.0 - self.theta) / (jnp.exp(t) - self.theta)


AliMikhailHaq = No3


@copula
class No4:
    """Family 4: ψ(t) = exp(-t^(1/θ))  (Gumbel)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.exp(-jnp.power(t, 1.0 / self.theta))


Gumbel = No4


@copula
class No5:
    """Family 5: ψ(t) = -(1/θ) log( 1 + exp(-t) (exp(-θ) - 1) )  (Frank)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return -jnp.log1p(jnp.exp(-t) * jnp.expm1(-self.theta)) / self.theta


Frank = No5


@copula
class No6:
    """Family 6: ψ(t) = 1 - (1 - exp(-t))^(1/θ)  (Joe)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        one_minus_exp_neg_t = -jnp.expm1(-t)  # 1 - exp(-t), stable for small t
        return 1.0 - jnp.power(one_minus_exp_neg_t, 1.0 / self.theta)


Joe = No6


@copula
class No13:
    """Family 13: ψ(t) = exp( 1 - (1+t)^(1/θ) )"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.exp(1.0 - jnp.power(1.0 + t, 1.0 / self.theta))


@copula
class No14:
    """Family 14: ψ(t) = (1 + t^(1/θ))^(-θ)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(1.0 + jnp.power(t, 1.0 / self.theta), -self.theta)


@copula
class No19:
    """Family 19: ψ(t) = θ / log(t + exp(θ))"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return self.theta / jnp.log(t + jnp.exp(self.theta))


@copula
class No20:
    """Family 20: ψ(t) = (log(t + e))^(-1/θ)"""

    theta: float

    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(jnp.log(t + jnp.e), -1.0 / self.theta)


# === leaf marginals ===========================================================


class Uniform:
    def quantile(self, u):
        return u

    def cdf(self, x):
        return x

    def log_prob(self, x):
        return 0.0


class Gaussian:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale
        self.parameters = {"loc": loc, "scale": scale}

    def quantile(self, u):
        return jax.scipy.stats.norm.ppf(u, loc=self.loc, scale=self.scale)

    def cdf(self, x):
        return jax.scipy.stats.norm.cdf(x, loc=self.loc, scale=self.scale)

    def log_prob(self, x):
        return jax.scipy.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)


# =============================
# 2. Reference Implementation Helpers
# =============================

_COPULA_CLASSES_BY_NAME = {
    # Named families used elsewhere in this test file
    "Clayton": Clayton,
    "Gumbel": Gumbel,
    "Frank": Frank,
    "Joe": Joe,
    "AliMikhailHaq": AliMikhailHaq,
    "Independence": Independence,
    # Table-numbered families
    "No1": No1,
    "No3": No3,
    "No4": No4,
    "No5": No5,
    "No6": No6,
    "No12": No12,
    "No13": No13,
    "No14": No14,
    "No19": No19,
    "No20": No20,
}


def _instantiate_copula(copula_cls, theta):
    """Instantiate a @copula class, passing theta if the class expects it."""
    if "theta" in getattr(copula_cls, "__annotations__", {}):
        return copula_cls(theta)
    return copula_cls()


def get_ref_fns(copula_type, theta):
    copula_cls = _COPULA_CLASSES_BY_NAME.get(copula_type)
    if copula_cls is None:
        raise ValueError(f"Unknown copula type {copula_type}")

    cop = _instantiate_copula(copula_cls, theta)

    def phi(t):
        return cop.generator(t)

    phi_inv = oryx_core.inverse(phi)
    return phi, phi_inv


def compute_exact_mixed_partial(copula_fn, x_vals):
    """Compute log |d^d C / dx1...dxd| using recursive jax.grad."""

    def grad_step(f, idx):
        return lambda x: jax.grad(f)(x)[idx]

    curr_fn = copula_fn
    for i in range(len(x_vals)):
        curr_fn = grad_step(curr_fn, i)

    val = curr_fn(x_vals)
    return jnp.log(jnp.abs(val))


# =============================
# 3. Scenarios
# =============================


@defmodel
def model_single_clayton(params, u):
    c = Clayton(params["theta"])
    return c([marginal(Uniform(), obs=u[i]) for i in range(4)])


def ref_single_clayton(u, params):
    phi, phi_inv = get_ref_fns("Clayton", params["theta"])
    # Avoid vmap in reference if possible for simplicity
    s = phi_inv(u[0]) + phi_inv(u[1]) + phi_inv(u[2]) + phi_inv(u[3])
    return phi(s)


@defmodel
def model_single_clayton_2(params, u):
    c = Clayton(params["theta"])
    return c([marginal(Uniform(), obs=u[i]) for i in range(2)])


def ref_single_clayton_2(u, params):
    phi, phi_inv = get_ref_fns("Clayton", params["theta"])
    s = phi_inv(u[0]) + phi_inv(u[1])
    return phi(s)


@defmodel
def model_single_gumbel(params, u):
    c = Gumbel(params["theta"])
    return c([marginal(Uniform(), obs=u[i]) for i in range(4)])


def ref_single_gumbel(u, params):
    phi, phi_inv = get_ref_fns("Gumbel", params["theta"])
    s = phi_inv(u[0]) + phi_inv(u[1]) + phi_inv(u[2]) + phi_inv(u[3])
    return phi(s)


@defmodel
def model_nested_clayton_2x2(params, u):
    f0 = Clayton(params["theta0"])
    f1 = Clayton(params["theta1"])
    c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
    c2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
    return f0([c1, c2])


def ref_nested_clayton_2x2(u, params):
    phi0, phi0_inv = get_ref_fns("Clayton", params["theta0"])
    phi1, phi1_inv = get_ref_fns("Clayton", params["theta1"])
    c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
    c2 = phi1(phi1_inv(u[2]) + phi1_inv(u[3]))
    return phi0(phi0_inv(c1) + phi0_inv(c2))


@defmodel
def model_alt_clayton_no12(params, u):
    f0 = Clayton(params["theta0"])
    f1 = No12(params["theta1"])
    child1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
    child2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
    return f0([child1, child2])


def ref_alt_clayton_no12(u, params):
    phi0, phi0_inv = get_ref_fns("Clayton", params["theta0"])
    phi1, phi1_inv = get_ref_fns("No12", params["theta1"])
    child1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
    child2 = phi1(phi1_inv(u[2]) + phi1_inv(u[3]))
    return phi0(phi0_inv(child1) + phi0_inv(child2))


def make_nested_2x2_model(outer_name: str, inner_name: str):
    """Factory for a 2x2 nested model: outer( inner(u0,u1), inner(u2,u3) )."""
    outer_cls = _COPULA_CLASSES_BY_NAME[outer_name]
    inner_cls = _COPULA_CLASSES_BY_NAME[inner_name]

    @defmodel
    def _model(params, u, *, _outer_cls=outer_cls, _inner_cls=inner_cls):
        f0 = _instantiate_copula(_outer_cls, params["theta0"])
        f1 = _instantiate_copula(_inner_cls, params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        return f0([c1, c2])

    return _model


def make_nested_2x2_ref(outer_name: str, inner_name: str):
    """Reference CDF for a 2x2 nested construction using (phi, phi_inv)."""

    def _ref(u, params, *, _outer_name=outer_name, _inner_name=inner_name):
        phi0, phi0_inv = get_ref_fns(_outer_name, params["theta0"])
        phi1, phi1_inv = get_ref_fns(_inner_name, params["theta1"])
        c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
        c2 = phi1(phi1_inv(u[2]) + phi1_inv(u[3]))
        return phi0(phi0_inv(c1) + phi0_inv(c2))

    return _ref


def _table3_theta_pairs(outer_no: int, inner_no: int):
    """
    Return a small set of (theta0, theta1) values satisfying Table 3
    admissibility conditions for (outer, inner) = (outer_no, inner_no).
    """
    pair = (outer_no, inner_no)

    # Helper for compact return
    def mk(pairs):
        return [{"theta0": float(a), "theta1": float(b)} for (a, b) in pairs]

    # Table 3 (from attached screenshot)
    if pair == (1, 12):
        # theta0 in (0,1), theta1 in [1,inf)
        return mk([(0.2, 1.0), (0.5, 2.0), (0.9, 5.0)])
    if pair == (1, 14):
        # theta0*theta1 in (0,1], theta1 in [1,inf)
        return mk([(0.5, 1.0), (0.4, 2.0), (0.2, 5.0)])
    if pair == (1, 19):
        # theta0 in (0,1), theta1 in (0,inf)
        return mk([(0.2, 0.7), (0.5, 2.0), (0.9, 5.0)])
    if pair == (1, 20):
        # theta0 <= theta1, both in (0,inf)
        return mk([(0.5, 0.5), (0.5, 2.0), (1.1, 2.0)])
    if pair == (3, 1):
        # theta0 in [0,1], theta1 in [1,inf)
        return mk([(0.2, 1.0), (0.8, 2.0), (0.5, 5.0)])
    if pair == (3, 19):
        # any theta0 in [0,1], any theta1 in (0,inf)
        return mk([(0.2, 0.7), (0.8, 2.0), (0.5, 5.0)])
    if pair == (3, 20):
        # any theta0 in [0,1], any theta1 in (0,inf)
        return mk([(0.2, 0.7), (0.8, 2.0), (0.5, 5.0)])
    if pair == (4, 1):
        # theta0 = 1, theta1 in [1,inf)
        return mk([(1.0, 0.5), (1.0, 2.0)])
    raise ValueError(f"Unhandled Table 3 pair {pair}")


def build_table3_scenarios():
    """
    Build scenarios for all family pairs in Table 3 (attached screenshot),
    using the 2x2 nested structure and a few admissible (theta0, theta1) choices.
    """
    # (outer_no, inner_no)
    pairs = [(1, 12), (1, 14), (1, 19), (1, 20), (4, 1), (3, 1), (3, 19), (3, 20)][:-3]
    u_obs = jnp.array([0.1, 0.1, 0.9, 0.9])

    scenarios = []
    for outer_no, inner_no in pairs:
        outer_name = f"No{outer_no}"
        inner_name = f"No{inner_no}"
        model = make_nested_2x2_model(outer_name, inner_name)
        ref_fn = make_nested_2x2_ref(outer_name, inner_name)
        for params in _table3_theta_pairs(outer_no, inner_no):
            name = f"Table3 ({outer_no},{inner_no}) θ0={params['theta0']}, θ1={params['theta1']}"
            scenarios.append((name, model, ref_fn, params, u_obs, False))
    return scenarios


@defmodel
def model_gaussian_marginals(params, u):
    c = Clayton(params["theta"])
    return c([marginal(Gaussian(0.0, 1.0), obs=u[i]) for i in range(2)])


def ref_gaussian_marginals(x, params):
    phi, phi_inv = get_ref_fns("Clayton", params["theta"])

    # For Gaussian, we need to explicitly apply CDF in the joint function
    def joint_cdf(xx):
        u0 = jax.scipy.stats.norm.cdf(xx[0], loc=0.0, scale=1.0)
        u1 = jax.scipy.stats.norm.cdf(xx[1], loc=0.0, scale=1.0)
        return phi(phi_inv(u0) + phi_inv(u1))

    return joint_cdf(x)


@defmodel
def model_diff_params_same_level(params, u):
    f0 = Clayton(params["theta0"])
    f1 = Clayton(params["theta1"])
    f2 = Clayton(params["theta2"])
    c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
    c2 = f2([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
    return f0([c1, c2])


def ref_diff_params_same_level(u, params):
    phi0, phi0_inv = get_ref_fns("Clayton", params["theta0"])
    phi1, phi1_inv = get_ref_fns("Clayton", params["theta1"])
    phi2, phi2_inv = get_ref_fns("Clayton", params["theta2"])
    c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
    c2 = phi2(phi2_inv(u[2]) + phi2_inv(u[3]))
    return phi0(phi0_inv(c1) + phi0_inv(c2))


@defmodel
def model_nested_1x2(params, u):
    f0 = Clayton(params["theta0"])
    f1 = Gumbel(params["theta1"])
    child = f1(
        [
            marginal(Uniform(), obs=u[1]),
            marginal(Uniform(), obs=u[2]),
        ]
    )
    return f0([marginal(Uniform(), obs=u[0]), child])


def ref_nested_1x2(u, params):
    phi0, phi0_inv = get_ref_fns("Clayton", params["theta0"])
    phi1, phi1_inv = get_ref_fns("Gumbel", params["theta1"])
    return phi0(phi0_inv(u[0]) + phi0_inv(phi1(phi1_inv(u[1]) + phi1_inv(u[2]))))


@defmodel
def model_none_leaf(params, u):
    # 'None' leaf variant: Root with a child subnode and one leaf being a direct variable,
    # but another index in 'u' is unused.
    c = Clayton(params["theta"])
    return c([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[2])])


def ref_none_leaf(u, params):
    phi, phi_inv = get_ref_fns("Clayton", params["theta"])

    # The mixed partial will be wrt u[0], u[1], u[2].
    # Since u[1] is unused, dC/du1 = 0.
    # To test independence or skipped variables, we should probably only differentiate wrt used vars.
    # But acopula.log_likelihood(u) expects u to have size matching max index.
    # If we want a valid log-likelihood, all variables in 'u' must be accounted for.
    # If u[1] is not in the copula, the joint density is f(u0, u1, u2) = c(u0, u2) * f(u0) * f(u1) * f(u2).
    # Since marginals are Uniform, f(u1)=1.
    # So we just take d^2 C / du0 du2.
    def copula_fn(x):
        return phi(phi_inv(x[0]) + phi_inv(x[2]))

    # We will compute d^3 / du0 du1 du2. This SHOULD be 0.
    # To make it non-zero and test "independent" leaves, we can add a marginal for u[1] that is NOT in the copula.
    # But Archimedean copulas in acopula currently require all leaves to be children of a Copula node.
    return copula_fn(u)


@defmodel
def three_level_model(params, u):
    f0 = Clayton(params["theta0"])
    f1 = Clayton(params["theta1"])
    f2 = Clayton(params["theta2"])

    # Structure: Root(u[0], f1(u[1], f2(u[2], u[3])))
    inner = f2(
        [
            marginal(Uniform(), obs=u[2]),
            marginal(Uniform(), obs=u[3]),
        ]
    )
    mid = f1(
        [
            marginal(Uniform(), obs=u[1]),
            inner,
        ]
    )
    return f0([marginal(Uniform(), obs=u[0]), mid])


def ref_three_level(u, params):
    phi0, phi0_inv = get_ref_fns("Clayton", params["theta0"])
    phi1, phi1_inv = get_ref_fns("Clayton", params["theta1"])
    phi2, phi2_inv = get_ref_fns("Clayton", params["theta2"])

    c2 = phi2(phi2_inv(u[2]) + phi2_inv(u[3]))
    c1 = phi1(phi1_inv(u[1]) + phi1_inv(c2))
    return phi0(phi0_inv(u[0]) + phi0_inv(c1))


# =============================
# 4. Pytest Scenario Wrapper
# =============================

SCENARIOS = [
    (
        "Single Clayton",
        model_single_clayton,
        ref_single_clayton,
        {"theta": 2.5},
        jnp.array([0.2, 0.4, 0.6, 0.8]),
        False,
    ),
    (
        "Single Clayton (Integral Method)",
        model_single_clayton_2,
        ref_single_clayton_2,
        {"theta": 2.5},
        jnp.array([0.2, 0.2]),
        True,
    ),
    (
        "Single Gumbel",
        model_single_gumbel,
        ref_single_gumbel,
        {"theta": 2.5},
        jnp.array([0.2, 0.4, 0.6, 0.8]),
        False,
    ),
    (
        "Nested Clayton 2x2",
        model_nested_clayton_2x2,
        ref_nested_clayton_2x2,
        {"theta0": 2.0, "theta1": 3.0},
        jnp.array([0.1, 0.4, 0.6, 0.9]),
        False,
    ),
    (
        "Gaussian Marginals",
        model_gaussian_marginals,
        ref_gaussian_marginals,
        {"theta": 2.0},
        jnp.array([0.5, -0.2]),
        False,
    ),
    (
        "Diff Params Same Level",
        model_diff_params_same_level,
        ref_diff_params_same_level,
        {"theta0": 2.0, "theta1": 3.0, "theta2": 4.0},
        jnp.array([0.1, 0.3, 0.5, 0.7]),
        False,
    ),
    (
        "Nested 1x3",
        model_nested_1x2,
        ref_nested_1x2,
        {"theta0": 2.0, "theta1": 3.0},
        jnp.array([0.9, 0.4, 0.6]),
        False,
    ),
    (
        "Three Level Clayton",
        three_level_model,
        ref_three_level,
        {"theta0": 2.0, "theta1": 3.0, "theta2": 4.0},
        jnp.array([0.1, 0.4, 0.6, 0.9]),
        False,
    ),
] + build_table3_scenarios()


@pytest.mark.parametrize(
    "name, model, ref_fn, params, u_obs, force_integral_method", SCENARIOS
)
def test_consistency(name, model, ref_fn, params, u_obs, force_integral_method):
    print(f"\nRunning scenario: {name}")

    # Gumbel(theta=1) as outer has a degenerate (point-mass) mixing density
    # that no ILS method can represent via continuous quadrature.
    if "(4,1)" in name:
        pytest.xfail("Gumbel(θ=1) outer has degenerate mixing density")

    # Use fixed_talbot (M=24) for the ILS inversion — much more accurate
    # than the default cohen method for exotic generator families.
    ils_kwargs = {"ils_method": "fixed_talbot", "ils_params": {"M": 24}}

    # 1. acopula implementation
    acopula_ll = model.log_likelihood(
        u_obs,
        params=params,
        force_integral_method=force_integral_method,
        **ils_kwargs,
    )

    # 2. reference implementation
    ref_ll = compute_exact_mixed_partial(lambda x: ref_fn(x, params), u_obs)

    print(f"Acopula LL: {acopula_ll}")
    print(f"Reference LL: {ref_ll}")

    atol = 1e-5
    assert jnp.allclose(acopula_ll, ref_ll, atol=atol), f"Mismatch in {name}!"


if __name__ == "__main__":
    pytest.main([__file__])
