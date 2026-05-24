"""Test: Root Rosenblatt + recursive simplex fill for 3-level nested copulas."""

import jax
import jax.numpy as jnp
from jax import random as jrandom
jax.config.update("jax_enable_x64", True)
import time

import acopula.core as ac
from acopula.core import (
    copula, Node, Leaf, UPlaceholder, RegistrationContext,
    _collect_leaves_in_order, _instantiate_copula_from_flat,
    _rosenblatt_bisect_uj, _reconstruct_obs_from_marginals,
)
import pytest

# Heavy end-to-end / deep-nesting test; excluded from the fast PR gate.
pytestmark = pytest.mark.slow


@copula
class Clayton:
    theta: float
    def generator(self, t):
        return (1.0 + t) ** (-1.0 / self.theta)


def _fill_node_simplex(node, target_w, key, params, leaf_id_to_idx, U_leaves):
    """Given copula value target_w for this node, fill its leaves via simplex sampling."""
    cop = _instantiate_copula_from_flat(node, params)
    psi = cop.generator
    psi_inv = cop.generator_inv
    children = node.children
    m = len(children)

    T = psi_inv(target_w)

    key, subkey = jrandom.split(key)
    exp_samples = jrandom.exponential(subkey, shape=(m,), dtype=jnp.float64)
    x_vals = T * exp_samples / jnp.sum(exp_samples)

    for i, ch in enumerate(children):
        w_child = psi(x_vals[i])
        if isinstance(ch, Leaf):
            idx = leaf_id_to_idx[id(ch)]
            U_leaves = U_leaves.at[idx].set(w_child)
        elif isinstance(ch, Node):
            child_key = jrandom.fold_in(key, i + 1000)
            U_leaves = _fill_node_simplex(
                ch, w_child, child_key, params, leaf_id_to_idx, U_leaves
            )

    return U_leaves


def sample_rosenblatt_simplex(graph, key, params):
    """Root Rosenblatt + recursive simplex fill. Works for any nesting depth."""
    leaves = _collect_leaves_in_order(graph)
    num_leaves = len(leaves)
    leaf_id_to_idx = {id(lf): i for i, lf in enumerate(leaves)}
    U_leaves = jnp.zeros(num_leaves, dtype=jnp.float64)

    cop = _instantiate_copula_from_flat(graph, params)
    psi = cop.generator
    psi_inv = cop.generator_inv
    m = len(graph.children)

    key, subkey = jrandom.split(key)
    v_arr = jrandom.uniform(subkey, shape=(m,), dtype=jnp.float64)

    w = jnp.zeros(m, dtype=jnp.float64)
    w = w.at[0].set(v_arr[0])
    t_prev = psi_inv(v_arr[0])

    for j_idx in range(1, m):
        w_j = _rosenblatt_bisect_uj(psi, psi_inv, t_prev, v_arr[j_idx], j_idx)
        w = w.at[j_idx].set(w_j)
        t_prev = t_prev + psi_inv(w_j)

    child_keys = jrandom.split(key, m)
    for i, ch in enumerate(graph.children):
        if isinstance(ch, Leaf):
            idx = leaf_id_to_idx[id(ch)]
            U_leaves = U_leaves.at[idx].set(w[i])
        elif isinstance(ch, Node):
            U_leaves = _fill_node_simplex(
                ch, w[i], child_keys[i], params, leaf_id_to_idx, U_leaves
            )

    return U_leaves


def build_3level_graph():
    """Build a 3-level nested Clayton tree with parameter registration."""
    u = UPlaceholder()

    with RegistrationContext():
        inner1 = Clayton(8.0)([u[0], u[1]])
        inner2 = Clayton(8.0)([u[2], u[3]])
        inner3 = Clayton(8.0)([u[4], u[5]])
        inner4 = Clayton(8.0)([u[6], u[7]])
        mid1 = Clayton(3.0)([inner1, inner2])
        mid2 = Clayton(3.0)([inner3, inner4])
        graph = Clayton(1.0)([mid1, mid2])

    # Build flat params from the registration context
    # Collect all params_symbols and build params vector
    params_list = []

    def _collect_params(node):
        if hasattr(node.copula, '_params_symbol'):
            spec = node.copula._params_symbol
            params_list.append((spec.start, spec.size, node.copula))
        for ch in node.children:
            if isinstance(ch, Node):
                _collect_params(ch)

    _collect_params(graph)
    params_list.sort(key=lambda x: x[0])

    total_size = max(s + sz for s, sz, _ in params_list)
    flat_params = jnp.zeros(total_size, dtype=jnp.float64)
    for start, size, cop in params_list:
        flat_params = flat_params.at[start].set(cop.theta)

    return graph, flat_params


def test_3level():
    """Test 3-level nested Clayton sampling."""
    graph, flat_params = build_3level_graph()

    print("=== 3-level nested Clayton via simplex fill ===")
    print("Clayton(1)(Clayton(3)(Clayton(8)(u0,u1), Clayton(8)(u2,u3)),")
    print("           Clayton(3)(Clayton(8)(u4,u5), Clayton(8)(u6,u7)))")
    print(f"Flat params: {flat_params}")
    print()

    key = jrandom.PRNGKey(42)
    n_samples = 200
    samples = []

    t0 = time.time()
    for i in range(n_samples):
        key, subkey = jrandom.split(key)
        u_leaves = sample_rosenblatt_simplex(graph, subkey, flat_params)
        samples.append(u_leaves)
        if i < 3:
            print(f"  Sample {i}: {u_leaves}")

    elapsed = time.time() - t0
    samples = jnp.stack(samples)

    print(f"\n{n_samples} samples in {elapsed:.1f}s ({elapsed/n_samples*1000:.0f} ms/sample)")
    print(f"Shape: {samples.shape}")
    print(f"Range: [{float(jnp.min(samples)):.6f}, {float(jnp.max(samples)):.6f}]")
    print(f"Mean per dim: {jnp.mean(samples, axis=0)}")
    print()

    assert jnp.all(samples > 0) and jnp.all(samples < 1), "Samples out of (0,1)!"

    # Correlation structure
    corr_same_inner = float(jnp.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    corr_same_mid = float(jnp.corrcoef(samples[:, 0], samples[:, 2])[0, 1])
    corr_diff_mid = float(jnp.corrcoef(samples[:, 0], samples[:, 4])[0, 1])

    print(f"Pearson corr (same inner group, expect ~high): {corr_same_inner:.4f}")
    print(f"Pearson corr (same mid, diff inner):           {corr_same_mid:.4f}")
    print(f"Pearson corr (diff mid groups, expect ~low):   {corr_diff_mid:.4f}")
    print(f"\nOrdering correct (inner > mid > between): {corr_same_inner > corr_same_mid > corr_diff_mid}")

    # KS test for uniformity
    for dim in range(8):
        sorted_u = jnp.sort(samples[:, dim])
        n = len(sorted_u)
        ecdf = jnp.arange(1, n + 1) / n
        ks = float(jnp.max(jnp.abs(sorted_u - ecdf)))
        status = "OK" if ks < 0.12 else "FAIL"
        if dim < 4 or status == "FAIL":
            print(f"  Dim {dim}: KS={ks:.4f} [{status}]")

    # Full CDF check
    psi0 = Clayton(1.0).generator
    psi0_inv = Clayton(1.0).generator_inv
    psi1 = Clayton(3.0).generator
    psi1_inv = Clayton(3.0).generator_inv
    psi2 = Clayton(8.0).generator
    psi2_inv = Clayton(8.0).generator_inv

    def full_cdf(u):
        t_inner1 = psi2_inv(u[0]) + psi2_inv(u[1])
        t_inner2 = psi2_inv(u[2]) + psi2_inv(u[3])
        t_inner3 = psi2_inv(u[4]) + psi2_inv(u[5])
        t_inner4 = psi2_inv(u[6]) + psi2_inv(u[7])
        t_mid1 = psi1_inv(psi2(t_inner1)) + psi1_inv(psi2(t_inner2))
        t_mid2 = psi1_inv(psi2(t_inner3)) + psi1_inv(psi2(t_inner4))
        t_root = psi0_inv(psi1(t_mid1)) + psi0_inv(psi1(t_mid2))
        return psi0(t_root)

    cdf_vals = jnp.array([full_cdf(samples[i]) for i in range(min(50, n_samples))])
    print(f"\nCDF values: mean={float(jnp.mean(cdf_vals)):.4f}, "
          f"range=[{float(jnp.min(cdf_vals)):.4f}, {float(jnp.max(cdf_vals)):.4f}]")


if __name__ == "__main__":
    test_3level()
