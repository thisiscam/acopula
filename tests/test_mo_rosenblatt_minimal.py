"""Minimal validation: 5 samples, verify basic correctness."""

import jax
import jax.numpy as jnp
from jax import random as jrandom
jax.config.update("jax_enable_x64", True)
import time

from acopula.core import copula
import jet_array


@copula
class Clayton:
    theta: float
    def generator(self, t):
        return (1.0 + t) ** (-1.0 / self.theta)


def mo_rosenblatt_sample(key, psi0, psi0_inv, psi_g, psi_g_inv, m_g, V0):
    """MO+Rosenblatt for one group given root frailty V0."""
    v_arr = jrandom.uniform(key, shape=(m_g,), dtype=jnp.float64)

    def psi_tilde(t):
        return jnp.exp(-V0 * psi0_inv(psi_g(t)))

    u_group = [v_arr[0]]
    S = psi_g_inv(v_arr[0])

    for j in range(1, m_g):
        v_j = v_arr[j]
        S_old = S
        series_in = jnp.zeros(j).at[0].set(1.0)

        lo, hi = 1e-14, 1.0 - 1e-14
        for _ in range(40):  # 40 iterations ~ 12 digits
            mid = (lo + hi) / 2.0
            S_new = S_old + psi_g_inv(mid)
            _, s_new = jet_array.jet(psi_tilde, (S_new,), (series_in,))
            _, s_old_jet = jet_array.jet(psi_tilde, (S_old,), (series_in,))
            cdf_val = s_new[j - 1] / s_old_jet[j - 1]
            if cdf_val < v_j:
                lo = mid
            else:
                hi = mid

        u_j = (lo + hi) / 2.0
        u_group.append(u_j)
        S = S_old + psi_g_inv(u_j)

    return jnp.array(u_group)


if __name__ == "__main__":
    theta0, theta1 = 2.0, 5.0
    root = Clayton(theta0)
    child = Clayton(theta1)

    psi0 = root.generator
    psi0_inv = root.generator_inv
    psi1 = child.generator
    psi1_inv = child.generator_inv

    print("Testing MO+Rosenblatt hybrid (5 samples, 1 group of 3)...")

    key = jrandom.PRNGKey(42)
    n = 5

    t0 = time.time()
    for i in range(n):
        key, k1, k2 = jrandom.split(key, 3)
        V0 = jrandom.gamma(k1, 1.0 / theta0, dtype=jnp.float64)
        u = mo_rosenblatt_sample(k2, psi0, psi0_inv, psi1, psi1_inv, 3, V0)
        print(f"  V0={float(V0):.4f}, u={u}")

        # Verify: the conditional copula C(u|V0) = psi_tilde(sum psi_g_inv(u_j))
        S = sum(psi1_inv(u[j]) for j in range(3))
        cdf = float(jnp.exp(-V0 * psi0_inv(psi1(S))))
        print(f"    Conditional CDF value: {cdf:.6f}")
        assert 0 < cdf < 1, f"CDF out of range: {cdf}"

    elapsed = time.time() - t0
    print(f"\n{n} samples in {elapsed:.1f}s ({elapsed/n*1000:.0f} ms/sample)")
    print("All assertions passed!")

    # Also verify the full copula CDF using both groups
    print("\nVerifying full 4-leaf nested copula samples...")
    key = jrandom.PRNGKey(99)
    for i in range(3):
        key, k1, k2, k3 = jrandom.split(key, 4)
        V0 = jrandom.gamma(k1, 1.0 / theta0, dtype=jnp.float64)
        u_g1 = mo_rosenblatt_sample(k2, psi0, psi0_inv, psi1, psi1_inv, 2, V0)
        u_g2 = mo_rosenblatt_sample(k3, psi0, psi0_inv, psi1, psi1_inv, 2, V0)

        # Full copula CDF
        t1 = psi1_inv(u_g1[0]) + psi1_inv(u_g1[1])
        t2 = psi1_inv(u_g2[0]) + psi1_inv(u_g2[1])
        s = psi0_inv(psi1(t1)) + psi0_inv(psi1(t2))
        cdf = float(psi0(s))
        print(f"  V0={float(V0):.4f}, u_g1={u_g1}, u_g2={u_g2}")
        print(f"    Full CDF: {cdf:.6f}")
        assert 0 < cdf < 1, f"Full CDF out of range: {cdf}"

    print("\nAll verifications passed!")
