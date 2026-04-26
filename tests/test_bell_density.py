"""Tests for Bell polynomial density computation (Approach B)."""

import jax
import jax.numpy as jnp
import pytest
from acopula import compile_model, defmodel, marginal, copula

jax.config.update("jax_enable_x64", True)


def _ll(model, obs, params, *, method="auto", ils_method="cohen", ils_params=None):
    """Test helper bridging the legacy ``model.log_likelihood(obs, params=...)``
    pattern to the pure-functional ``compile_model`` API.  The wrapper
    rebuilds the compiled model on each call (fine for unit tests; in
    production code reuse a single :class:`CompiledModel` across calls)."""
    cm = compile_model(
        model, template=params, method=method,
        ils_method=ils_method, ils_params=ils_params,
    )
    return cm.eval(obs, params)


# ============================
# Trivial marginal (data already on copula scale)
# ============================

class Uniform:
    def quantile(self, u):
        return u
    def cdf(self, x):
        return x
    def log_prob(self, x):
        return 0.0


# ============================
# Copula family definitions
# ============================

@copula
class Clayton:
    theta: float
    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.power(1.0 + t, -1.0 / self.theta)


@copula
class Gumbel:
    theta: float
    def generator(self, t: jax.Array) -> jax.Array:
        return jnp.exp(-jnp.power(t, 1.0 / self.theta))


@copula
class Frank:
    theta: float
    def generator(self, t: jax.Array) -> jax.Array:
        return -jnp.log1p(jnp.exp(-t) * jnp.expm1(-self.theta)) / self.theta


@copula
class AMH:
    theta: float
    def generator(self, t: jax.Array) -> jax.Array:
        return (1.0 - self.theta) / (jnp.exp(t) - self.theta)


@copula
class Joe:
    theta: float
    def generator(self, t: jax.Array) -> jax.Array:
        return 1.0 - jnp.power(1.0 - jnp.exp(-t), 1.0 / self.theta)


# ============================
# Helper: brute-force mixed partial via nested jax.grad
# ============================

def _brute_force_log_density(copula_fn, u_vals):
    """Compute log|d^d C / du_1...du_d| via recursive jax.grad."""
    d = len(u_vals)

    def make_grad_chain(f, indices):
        """Recursively differentiate f w.r.t. u[indices[0]], u[indices[1]], ..."""
        if len(indices) == 0:
            return f
        idx = indices[0]
        rest = indices[1:]
        def inner(u):
            return jax.grad(lambda u: make_grad_chain(f, rest)(u), argnums=0)(u)
        # Actually, for mixed partial we need to differentiate w.r.t. individual components
        return inner

    # Build the copula as a function of a single vector u
    # Then take nested grads one variable at a time
    fn = copula_fn
    for i in reversed(range(d)):
        prev_fn = fn
        fn = lambda u, _prev=prev_fn, _i=i: jax.grad(lambda ui: _prev(u.at[_i].set(ui)))(u[_i])

    val = fn(u_vals)
    return jnp.log(jnp.abs(val))


# ============================
# Model factories
# ============================

def make_nested_2x2(outer_cls, inner_cls):
    """Create a 2x2 nested model: outer([inner([u0, u1]), inner([u2, u3])])."""
    @defmodel
    def model(params, u):
        f0 = outer_cls(params["theta0"])
        f1 = inner_cls(params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        return f0([c1, c2])
    return model


def make_ref_2x2(outer_cls, inner_cls):
    """Create the copula function for brute-force grad reference."""
    from oryx import core as oryx_core

    def copula_fn(u, theta0, theta1):
        phi0 = outer_cls(theta0).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = inner_cls(theta1).generator
        phi1_inv = oryx_core.inverse(phi1)
        c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
        c2 = phi1(phi1_inv(u[2]) + phi1_inv(u[3]))
        return phi0(phi0_inv(c1) + phi0_inv(c2))

    return copula_fn


# ============================
# Category A: Bell vs brute-force jax.grad
# ============================

U_OBS = jnp.array([0.3, 0.5, 0.4, 0.7])


@pytest.mark.parametrize("outer_cls,inner_cls,theta0,theta1", [
    (Clayton, Clayton, 2.0, 3.0),
    (Gumbel, Gumbel, 2.0, 3.0),
    (Clayton, Gumbel, 0.5, 2.0),
])
def test_bell_vs_grad_continuous(outer_cls, inner_cls, theta0, theta1):
    """Bell matches brute-force grad for continuous-mixing families."""
    model = make_nested_2x2(outer_cls, inner_cls)
    params = {"theta0": theta0, "theta1": theta1}

    ll_bell = _ll(model, U_OBS, params, method="bell")

    ref_fn = make_ref_2x2(outer_cls, inner_cls)
    ll_grad = _brute_force_log_density(
        lambda u: ref_fn(u, theta0, theta1), U_OBS
    )

    assert jnp.isfinite(ll_bell), f"Bell log-lik is not finite: {ll_bell}"
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}, "
        f"diff={float(jnp.abs(ll_bell - ll_grad))}"
    )


@pytest.mark.parametrize("outer_cls,inner_cls,theta0,theta1", [
    (Frank, Clayton, 5.0, 2.0),
    (AMH, Clayton, 0.5, 2.0),
    (Joe, Clayton, 2.0, 3.0),
])
def test_bell_vs_grad_discrete(outer_cls, inner_cls, theta0, theta1):
    """Bell matches brute-force grad for discrete-mixing families."""
    model = make_nested_2x2(outer_cls, inner_cls)
    params = {"theta0": theta0, "theta1": theta1}

    ll_bell = _ll(model, U_OBS, params, method="bell")

    ref_fn = make_ref_2x2(outer_cls, inner_cls)
    ll_grad = _brute_force_log_density(
        lambda u: ref_fn(u, theta0, theta1), U_OBS
    )

    assert jnp.isfinite(ll_bell), f"Bell log-lik is not finite: {ll_bell}"
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}, "
        f"diff={float(jnp.abs(ll_bell - ll_grad))}"
    )


# ============================
# Category B: Bell vs integral (where both work)
# ============================

@pytest.mark.parametrize("outer_cls,inner_cls,theta0,theta1", [
    (Clayton, Clayton, 2.0, 3.0),
    (Gumbel, Gumbel, 2.0, 3.0),
])
def test_bell_vs_integral(outer_cls, inner_cls, theta0, theta1):
    """Bell matches integral approach for continuous families."""
    model = make_nested_2x2(outer_cls, inner_cls)
    params = {"theta0": theta0, "theta1": theta1}

    ll_bell = _ll(model, U_OBS, params, method="bell")
    ll_int = _ll(model, U_OBS, params, method="integral",
                 ils_method="fixed_talbot", ils_params={"M": 24})

    assert jnp.allclose(ll_bell, ll_int, atol=1e-4), (
        f"Bell={float(ll_bell)}, Integral={float(ll_int)}, "
        f"diff={float(jnp.abs(ll_bell - ll_int))}"
    )


# ============================
# Category C: Single-layer regression
# ============================

def test_bell_single_layer():
    """Bell on a single-layer copula matches the existing single-layer code."""
    @defmodel
    def model(params, u):
        f = Clayton(params["theta"])
        return f([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1]),
                  marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])

    params = {"theta": 2.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")
    ll_single = _ll(model, U_OBS, params)  # uses single-layer path

    assert jnp.allclose(ll_bell, ll_single, atol=1e-8), (
        f"Bell={float(ll_bell)}, Single={float(ll_single)}"
    )


# ============================
# Category D: 3-level nesting
# ============================

def test_bell_three_level():
    """Bell works for 3-level tree: root -> {node_a, u3}, node_a -> {u1, u2}."""
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_a = Clayton(params["theta_a"])
        inner = f_a([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        return f_r([inner, marginal(Uniform(), obs=u[2])])

    params = {"theta_r": 2.0, "theta_a": 3.0}
    u_obs = jnp.array([0.3, 0.5, 0.7])

    ll_bell = _ll(model, u_obs, params, method="bell")

    # Brute-force reference
    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(2.0).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_a = Clayton(3.0).generator
        phi_a_inv = oryx_core.inverse(phi_a)
        C_a = phi_a(phi_a_inv(u[0]) + phi_a_inv(u[1]))
        return phi_r(phi_r_inv(C_a) + phi_r_inv(u[2]))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)

    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category E: Same structure, different parameters (vmap batching test)
# ============================

def test_bell_diff_params_same_structure():
    """Two sectors with same structure but different theta — tests vmap batching."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f1 = Clayton(params["theta1"])
        f2 = Clayton(params["theta2"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f2([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        return f0([c1, c2])

    params = {"theta0": 2.0, "theta1": 3.0, "theta2": 4.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(2.0).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = Clayton(3.0).generator
        phi1_inv = oryx_core.inverse(phi1)
        phi2 = Clayton(4.0).generator
        phi2_inv = oryx_core.inverse(phi2)
        c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
        c2 = phi2(phi2_inv(u[2]) + phi2_inv(u[3]))
        return phi0(phi0_inv(c1) + phi0_inv(c2))

    ll_grad = _brute_force_log_density(ref_fn, U_OBS)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category F: Asymmetric tree (different d_c across children)
# ============================

def test_bell_asymmetric_3_plus_2():
    """Root -> {sector(3 leaves), sector(2 leaves)}."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f1 = Clayton(params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1]),
                  marginal(Uniform(), obs=u[2])])
        c2 = f1([marginal(Uniform(), obs=u[3]), marginal(Uniform(), obs=u[4])])
        return f0([c1, c2])

    u_obs = jnp.array([0.2, 0.4, 0.6, 0.3, 0.7])
    params = {"theta0": 1.5, "theta1": 3.0}
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(1.5).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = Clayton(3.0).generator
        phi1_inv = oryx_core.inverse(phi1)
        c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]) + phi1_inv(u[2]))
        c2 = phi1(phi1_inv(u[3]) + phi1_inv(u[4]))
        return phi0(phi0_inv(c1) + phi0_inv(c2))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


def test_bell_asymmetric_sector_plus_leaf():
    """Root -> {sector(2 leaves), single leaf}."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f1 = Clayton(params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        return f0([c1, marginal(Uniform(), obs=u[2])])

    u_obs = jnp.array([0.3, 0.5, 0.7])
    params = {"theta0": 2.0, "theta1": 3.0}
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(2.0).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = Clayton(3.0).generator
        phi1_inv = oryx_core.inverse(phi1)
        c1 = phi1(phi1_inv(u[0]) + phi1_inv(u[1]))
        return phi0(phi0_inv(c1) + phi0_inv(u[2]))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category G: Many children (m > 2)
# ============================

def test_bell_four_sectors():
    """Root with 4 identical sectors — tests vmap over 4 children + Cauchy product."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f1 = Clayton(params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        c3 = f1([marginal(Uniform(), obs=u[4]), marginal(Uniform(), obs=u[5])])
        c4 = f1([marginal(Uniform(), obs=u[6]), marginal(Uniform(), obs=u[7])])
        return f0([c1, c2, c3, c4])

    u_obs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    params = {"theta0": 1.0, "theta1": 3.0}
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(1.0).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = Clayton(3.0).generator
        phi1_inv = oryx_core.inverse(phi1)
        cs = [phi1(phi1_inv(u[2*i]) + phi1_inv(u[2*i+1])) for i in range(4)]
        return phi0(sum(phi0_inv(c) for c in cs))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category H: Mixed families at same level
# ============================

def test_bell_mixed_families():
    """Root(Clayton) -> {Gumbel sector, Clayton sector} — different structure groups."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f_g = Gumbel(params["theta_g"])
        f_c = Clayton(params["theta_c"])
        c1 = f_g([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f_c([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        return f0([c1, c2])

    params = {"theta0": 1.5, "theta_g": 2.0, "theta_c": 3.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(1.5).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi_g = Gumbel(2.0).generator
        phi_g_inv = oryx_core.inverse(phi_g)
        phi_c = Clayton(3.0).generator
        phi_c_inv = oryx_core.inverse(phi_c)
        c1 = phi_g(phi_g_inv(u[0]) + phi_g_inv(u[1]))
        c2 = phi_c(phi_c_inv(u[2]) + phi_c_inv(u[3]))
        return phi0(phi0_inv(c1) + phi0_inv(c2))

    ll_grad = _brute_force_log_density(ref_fn, U_OBS)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category I: Deeper nesting with batching
# ============================

def test_bell_deep_two_branches():
    """3-level: root -> {node_a, node_b}, each -> {u1, u2}."""
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_a = Clayton(params["theta_a"])
        a = f_a([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        b = f_a([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        return f_r([a, b])

    params = {"theta_r": 1.5, "theta_a": 3.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(1.5).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_a = Clayton(3.0).generator
        phi_a_inv = oryx_core.inverse(phi_a)
        a = phi_a(phi_a_inv(u[0]) + phi_a_inv(u[1]))
        b = phi_a(phi_a_inv(u[2]) + phi_a_inv(u[3]))
        return phi_r(phi_r_inv(a) + phi_r_inv(b))

    ll_grad = _brute_force_log_density(ref_fn, U_OBS)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category J: True 3-layer nesting (3 distinct generators)
# ============================

def test_bell_true_3_layer_chain():
    """3 generators deep: root(psi_r) -> mid(psi_m) -> inner(psi_i) -> {u1, u2}.

    Chain with 3 nesting levels. Bell recursion propagates beta-coefficients
    through two composition steps.
    """
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_m = Clayton(params["theta_m"])
        f_i = Clayton(params["theta_i"])
        inner = f_i([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        mid = f_m([inner])
        return f_r([mid])

    params = {"theta_r": 1.5, "theta_m": 2.5, "theta_i": 4.0}
    u_obs = jnp.array([0.3, 0.7])
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(1.5).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_m = Clayton(2.5).generator
        phi_m_inv = oryx_core.inverse(phi_m)
        phi_i = Clayton(4.0).generator
        phi_i_inv = oryx_core.inverse(phi_i)
        C_i = phi_i(phi_i_inv(u[0]) + phi_i_inv(u[1]))
        C_m = phi_m(phi_m_inv(C_i))
        return phi_r(phi_r_inv(C_m))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


def test_bell_3_layer_branching():
    """3 generators with branching at mid: root -> mid -> {inner_a, inner_b}, each -> 2 leaves.

    Tests Bell recursion at depth 3 with Cauchy product at the mid level.
    """
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_m = Clayton(params["theta_m"])
        f_i = Clayton(params["theta_i"])
        a = f_i([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        b = f_i([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        mid = f_m([a, b])
        return f_r([mid])

    params = {"theta_r": 1.5, "theta_m": 2.5, "theta_i": 4.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(1.5).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_m = Clayton(2.5).generator
        phi_m_inv = oryx_core.inverse(phi_m)
        phi_i = Clayton(4.0).generator
        phi_i_inv = oryx_core.inverse(phi_i)
        a = phi_i(phi_i_inv(u[0]) + phi_i_inv(u[1]))
        b = phi_i(phi_i_inv(u[2]) + phi_i_inv(u[3]))
        C_m = phi_m(phi_m_inv(a) + phi_m_inv(b))
        return phi_r(phi_r_inv(C_m))

    ll_grad = _brute_force_log_density(ref_fn, U_OBS)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


def test_bell_3_layer_mixed():
    """3 generators with mixed children at every level:
    root -> {mid, leaf_u4}, mid -> {inner, leaf_u3}, inner -> {u1, u2}.

    Full generality: mixed leaf/node children, 3 distinct generators,
    Bell recursion through 2 composition steps.
    """
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_m = Clayton(params["theta_m"])
        f_i = Clayton(params["theta_i"])
        inner = f_i([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        mid = f_m([inner, marginal(Uniform(), obs=u[2])])
        return f_r([mid, marginal(Uniform(), obs=u[3])])

    params = {"theta_r": 1.5, "theta_m": 2.5, "theta_i": 4.0}
    ll_bell = _ll(model, U_OBS, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(1.5).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_m = Clayton(2.5).generator
        phi_m_inv = oryx_core.inverse(phi_m)
        phi_i = Clayton(4.0).generator
        phi_i_inv = oryx_core.inverse(phi_i)
        C_i = phi_i(phi_i_inv(u[0]) + phi_i_inv(u[1]))
        C_m = phi_m(phi_m_inv(C_i) + phi_m_inv(u[2]))
        return phi_r(phi_r_inv(C_m) + phi_r_inv(u[3]))

    ll_grad = _brute_force_log_density(ref_fn, U_OBS)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


def test_bell_3_layer_full_branching():
    """3 generators, multiple children at EVERY level:

    root(psi_r) -> {mid_A(psi_m), mid_B(psi_m)}
    mid_A -> {inner_1(psi_i), inner_2(psi_i)}
    mid_B -> {inner_3(psi_i), inner_4(psi_i)}
    each inner -> {leaf, leaf}

    Total: 8 leaves, 3 generator levels, Cauchy product at every level.
    """
    @defmodel
    def model(params, u):
        f_r = Clayton(params["theta_r"])
        f_m = Clayton(params["theta_m"])
        f_i = Clayton(params["theta_i"])
        i1 = f_i([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        i2 = f_i([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        i3 = f_i([marginal(Uniform(), obs=u[4]), marginal(Uniform(), obs=u[5])])
        i4 = f_i([marginal(Uniform(), obs=u[6]), marginal(Uniform(), obs=u[7])])
        mid_a = f_m([i1, i2])
        mid_b = f_m([i3, i4])
        return f_r([mid_a, mid_b])

    u_obs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    params = {"theta_r": 1.5, "theta_m": 2.5, "theta_i": 4.0}
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi_r = Clayton(1.5).generator
        phi_r_inv = oryx_core.inverse(phi_r)
        phi_m = Clayton(2.5).generator
        phi_m_inv = oryx_core.inverse(phi_m)
        phi_i = Clayton(4.0).generator
        phi_i_inv = oryx_core.inverse(phi_i)
        i1 = phi_i(phi_i_inv(u[0]) + phi_i_inv(u[1]))
        i2 = phi_i(phi_i_inv(u[2]) + phi_i_inv(u[3]))
        i3 = phi_i(phi_i_inv(u[4]) + phi_i_inv(u[5]))
        i4 = phi_i(phi_i_inv(u[6]) + phi_i_inv(u[7]))
        mid_a = phi_m(phi_m_inv(i1) + phi_m_inv(i2))
        mid_b = phi_m(phi_m_inv(i3) + phi_m_inv(i4))
        return phi_r(phi_r_inv(mid_a) + phi_r_inv(mid_b))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )


# ============================
# Category K: Wider dimensions (d=6)
# ============================

def test_bell_3x2():
    """Root with 3 sectors of 2 leaves each (d=6)."""
    @defmodel
    def model(params, u):
        f0 = Clayton(params["theta0"])
        f1 = Clayton(params["theta1"])
        c1 = f1([marginal(Uniform(), obs=u[0]), marginal(Uniform(), obs=u[1])])
        c2 = f1([marginal(Uniform(), obs=u[2]), marginal(Uniform(), obs=u[3])])
        c3 = f1([marginal(Uniform(), obs=u[4]), marginal(Uniform(), obs=u[5])])
        return f0([c1, c2, c3])

    u_obs = jnp.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])
    params = {"theta0": 1.5, "theta1": 3.0}
    ll_bell = _ll(model, u_obs, params, method="bell")

    from oryx import core as oryx_core
    def ref_fn(u):
        phi0 = Clayton(1.5).generator
        phi0_inv = oryx_core.inverse(phi0)
        phi1 = Clayton(3.0).generator
        phi1_inv = oryx_core.inverse(phi1)
        cs = [phi1(phi1_inv(u[2*i]) + phi1_inv(u[2*i+1])) for i in range(3)]
        return phi0(sum(phi0_inv(c) for c in cs))

    ll_grad = _brute_force_log_density(ref_fn, u_obs)
    assert jnp.allclose(ll_bell, ll_grad, atol=1e-6), (
        f"Bell={float(ll_bell)}, Grad={float(ll_grad)}"
    )
