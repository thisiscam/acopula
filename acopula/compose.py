"""Composition dispatch for nested Archimedean copulas.

Computes Taylor coefficients of h(t) = psi_parent_inv(psi_child(t)) using
the best available method:

1. Registered closed-form h(t) → jet through h directly (O(d_c²), stable)
2. Direct jet through psi_inv(psi(t)) → O(d_c²), may overflow at extreme t
3. Implicit differentiation solve → O(d_c³), always stable

Users can register custom compositions for their generator pairs and
configure the fallback strategy.
"""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

import jet_array

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_COMPOSE_REGISTRY: Dict[Tuple[type, type], Callable] = {}
_COMPOSE_REGISTRY_BY_NAME: Dict[Tuple[str, str], Callable] = {}
_FALLBACK = "implicit"  # "implicit" (default, stable, O(d_c^3)) or "direct_jet" (fast, may overflow)


def register_composition(outer_cls, inner_cls, fn):
    """Register a stable closed-form composition h(t) = psi_outer_inv(psi_inner(t)).

    Args:
        outer_cls: The outer (parent) copula class.
        inner_cls: The inner (child) copula class.
        fn: A JAX-traceable function fn(t, parent_cop, child_cop) -> scalar.
    """
    _COMPOSE_REGISTRY[(outer_cls, inner_cls)] = fn
    _COMPOSE_REGISTRY_BY_NAME[(outer_cls.__name__, inner_cls.__name__)] = fn


def get_composition(parent_cop, child_cop) -> Optional[Callable]:
    """Look up a registered composition for this copula pair.

    Tries (type, type) first, then (name, name) for cross-file compatibility.
    Returns the composition function or None.
    """
    key = (type(parent_cop), type(child_cop))
    fn = _COMPOSE_REGISTRY.get(key)
    if fn is not None:
        return fn
    name_key = (type(parent_cop).__name__, type(child_cop).__name__)
    return _COMPOSE_REGISTRY_BY_NAME.get(name_key)


def set_composition_fallback(mode: str):
    """Set the fallback method when no registered composition exists.

    Args:
        mode: "implicit" (O(d_c³), stable, default) or
              "direct_jet" (O(d_c²), may overflow at extreme values)
    """
    global _FALLBACK
    if mode not in ("implicit", "direct_jet"):
        raise ValueError(f"Unknown fallback mode: {mode!r}. Use 'implicit' or 'direct_jet'.")
    _FALLBACK = mode


# ---------------------------------------------------------------------------
# Built-in compositions for classical families
# ---------------------------------------------------------------------------

def _compose_same_family_power(t, parent_cop, child_cop, base_fn, inv_base_fn):
    """Generic same-family composition via h(t) = inv_base(base(t)^r)."""
    r = child_cop.theta / parent_cop.theta
    return inv_base_fn(jnp.power(base_fn(t), r))


def _clayton_compose(t, parent_cop, child_cop):
    # psi_inner(t) = (1+t)^{-1/th_i}, psi_outer_inv(u) = u^{-th_o} - 1
    # h(t) = (1+t)^{th_o/th_i} - 1
    r = parent_cop.theta / child_cop.theta
    return jnp.power(1.0 + t, r) - 1.0


def _gumbel_compose(t, parent_cop, child_cop):
    # psi_inner(t) = exp(-t^{1/th_i}), psi_outer_inv(u) = (-log u)^{th_o}
    # h(t) = t^{th_o/th_i}
    r = parent_cop.theta / child_cop.theta
    return jnp.power(t, r)


def _joe_compose(t, parent_cop, child_cop):
    # psi_inner(t) = 1-(1-e^{-t})^{1/th_i}, psi_outer_inv(u) = -log(1-(1-u)^{th_o})
    # h(t) = -log(1-(1-e^{-t})^{th_o/th_i})
    r = parent_cop.theta / child_cop.theta
    return -jnp.log1p(-jnp.power(1.0 - jnp.exp(-t), r))


# Register by name so they match user-defined copulas with the same class name
def _frank_compose(t, parent_cop, child_cop):
    # h(t) = psi_out_inv(psi_in(t)) for Frank/Frank.
    # Naive: -log(expm1(r*v)/a_o), but log(small) overflows at large t.
    # Stable form: h(t) = t + log(a_o * exp(-t) / expm1(r*v)).
    # Both a_o*exp(-t) and expm1(r*v) are O(exp(-t)), so their ratio is O(1).
    r = parent_cop.theta / child_cop.theta
    a_i = jnp.expm1(-child_cop.theta)
    a_o = jnp.expm1(-parent_cop.theta)
    e = jnp.exp(-t)
    v = jnp.log1p(e * a_i)
    return t + jnp.log(a_o * e / jnp.expm1(r * v))


_COMPOSE_REGISTRY_BY_NAME[("Clayton", "Clayton")] = _clayton_compose
_COMPOSE_REGISTRY_BY_NAME[("Gumbel", "Gumbel")] = _gumbel_compose
_COMPOSE_REGISTRY_BY_NAME[("Joe", "Joe")] = _joe_compose
_COMPOSE_REGISTRY_BY_NAME[("Frank", "Frank")] = _frank_compose


# ---------------------------------------------------------------------------
# Implicit differentiation solve (O(d_c³) fallback)
# ---------------------------------------------------------------------------

def _truncated_convolve(a, b):
    """Truncated polynomial multiplication: (a * b)[:len(a)].

    Uses lax.conv_general_dilated to do the full convolution as a single
    HLO op, then truncates to the original length.  Replaces a Python
    `for k in range(n)` that was emitting O(n) update_slice ops at trace
    time and inflating the compiled XLA program for non-Tier-1 families.
    """
    n = a.shape[0]
    a_pad = jnp.pad(a, (n - 1, 0))
    out = lax.conv_general_dilated(
        a_pad[None, None, :],
        b[::-1][None, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCW", "IOW", "NCW"),
    )[0, 0]
    return out[:n]


def _solve_composition_taylor(
    p_inner: jax.Array,
    p_outer: jax.Array,
    d_c: int,
) -> jax.Array:
    """Solve for Taylor coefficients of h where psi_outer(h(t)) = psi_inner(t).

    From psi_outer(h_0 + Q(e)) = psi_inner(t + e), where Q(e) = sum q[k]e^{k+1}:
        a[0]*Q + a[1]*Q^2 + a[2]*Q^3 + ... = b[0]*e + b[1]*e^2 + ...
    where a = p_outer, b = p_inner.

    Matching e^n: a[0]*q[n-1] + [e^n from a[1]*Q^2 + ...] = b[n-1]
    The correction terms only involve q[0]..q[n-2], giving a triangular system.

    Uses lax.scan with unroll hint: fully unrolled for small d_c (typical
    sector sizes 5-10), compact loop for larger d_c.
    """
    n = d_c + 1  # padded polynomial length
    n_powers = max(d_c - 1, 1)

    q0 = p_inner[0] / p_outer[0]
    Q_init = jnp.zeros(n).at[1].set(q0)

    # Initialize Q power table: C[m] = Q^{m+2} for m=0..n_powers-1
    # Place q0_powers[m-2] at position (m-2, m) for m in 2..min(d_c, n-1).
    # Vectorised advanced-indexing scatter — single HLO op instead of an
    # O(d_c) Python loop emitting individual update_slice ops.
    C_init = jnp.zeros((n_powers, n))
    q0_powers = jnp.power(q0, jnp.arange(2, d_c + 1, dtype=jnp.float64))
    m_range = jnp.arange(2, min(d_c + 1, n))
    C_init = C_init.at[m_range - 2, m_range].set(q0_powers[: m_range.shape[0]])

    q_init = jnp.zeros(d_c).at[0].set(q0)

    def step(carry, k):
        q_arr, Q_poly, C = carry

        # Correction: sum_{m=2}^{d_c} a[m-1] * C[m-2][k+1]
        power_coeffs = C[:n_powers, k + 1]
        a_coeffs = p_outer[1:d_c]
        correction = jnp.dot(a_coeffs[:n_powers], power_coeffs[:n_powers])

        # Solve for q[k]
        q_k = (p_inner[k] - correction) / p_outer[0]
        q_arr = q_arr.at[k].set(q_k)
        Q_poly = Q_poly.at[k + 1].set(q_k)

        # Recompute Q powers: Q^2 = Q*Q, Q^3 = Q^2*Q, ...
        def power_step(prev, _):
            nxt = _truncated_convolve(prev, Q_poly)
            return nxt, nxt

        _, C_new = lax.scan(power_step, Q_poly, None, length=n_powers,
                            unroll=min(n_powers, 16))

        return (q_arr, Q_poly, C_new), None

    if d_c <= 1:
        return q_init

    (q_final, _, _), _ = lax.scan(
        step, (q_init, Q_init, C_init), jnp.arange(1, d_c),
        unroll=min(d_c - 1, 16),
    )

    return q_final


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def compute_composition_taylor(
    parent_cop,
    child_cop,
    t_child: jax.Array,
    d_c: int,
    effective_order=None,
) -> jax.Array:
    """Compute Taylor coefficients of h(t) = psi_parent_inv(psi_child(t)).

    Dispatch order:
    1. Registered closed-form h(t) -> jet(h, ...) — O(d_c^2), stable
    2. Fallback (configurable via set_composition_fallback):
       - "direct_jet": jet(psi_inv(psi(t)), ...) — O(d_c^2), may overflow
       - "implicit": implicit diff solve — O(d_c^3), stable

    Args:
        parent_cop: Outer copula instance.
        child_cop: Inner copula instance.
        t_child: Evaluation point.
        d_c: Number of Taylor coefficients (= uncensored leaves in subtree).
        effective_order: Optional dynamic order limit for jet.

    Returns:
        p: Array of length d_c where p[k] = h^{(k+1)}(t)/(k+1)!
    """
    series_in = jnp.zeros(d_c).at[0].set(1.0)

    # Tier 1: registered closed-form composition
    compose_fn = get_composition(parent_cop, child_cop)
    if compose_fn is not None:
        h_fn = lambda t: compose_fn(t, parent_cop, child_cop)
        _, p = jet_array.jet(h_fn, (t_child,), (series_in,),
                             effective_order=effective_order)
        return jnp.asarray(p)

    # Tier 2/3: fallback
    if _FALLBACK == "direct_jet":
        h_fn = lambda t: parent_cop.generator_inv(child_cop.generator(t))
        _, p = jet_array.jet(h_fn, (t_child,), (series_in,),
                             effective_order=effective_order)
        return jnp.asarray(p)

    # Tier 3: implicit differentiation (default)
    _, p_inner = jet_array.jet(
        child_cop.generator, (t_child,), (series_in,),
        effective_order=effective_order,
    )
    h_0 = parent_cop.generator_inv(child_cop.generator(t_child))
    _, p_outer = jet_array.jet(
        parent_cop.generator, (h_0,), (series_in,),
        effective_order=effective_order,
    )
    return _solve_composition_taylor(jnp.asarray(p_inner), jnp.asarray(p_outer), d_c)
