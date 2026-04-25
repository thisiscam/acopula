"""Quick validation of MO+Rosenblatt: 20 samples, check basic properties."""

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


def mo_rosenblatt_sample(key, psi0, psi0_inv, groups, V0):
    """MO+Rosenblatt for 2-level nesting given root frailty V0."""
    u_all = []
    for psi_g, psi_g_inv, m_g in groups:
        key, subkey = jrandom.split(key)
        v_arr = jrandom.uniform(subkey, shape=(m_g,), dtype=jnp.float64)

        def psi_tilde(t, _V0=V0, _psi0_inv=psi0_inv, _psi_g=psi_g):
            return jnp.exp(-_V0 * _psi0_inv(_psi_g(t)))

        u_group = [v_arr[0]]
        S = psi_g_inv(v_arr[0])

        for j in range(1, m_g):
            v_j = v_arr[j]
            S_old = S
            series_in = jnp.zeros(j).at[0].set(1.0)
            _, s_old = jet_array.jet(psi_tilde, (S_old,), (series_in,))

            lo, hi = 1e-14, 1.0 - 1e-14
            for _ in range(60):
                mid = (lo + hi) / 2.0
                S_new = S_old + psi_g_inv(mid)
                _, s_new = jet_array.jet(psi_tilde, (S_new,), (series_in,))
                cdf_val = s_new[j - 1] / s_old[j - 1]
                if cdf_val < v_j:
                    lo = mid
                else:
                    hi = mid

            u_j = (lo + hi) / 2.0
            u_group.append(u_j)
            S = S_old + psi_g_inv(u_j)

        u_all.extend(u_group)

    return jnp.array(u_all)


if __name__ == "__main__":
    theta0, theta1 = 2.0, 5.0
    root = Clayton(theta0)
    child = Clayton(theta1)

    psi0 = root.generator
    psi0_inv = root.generator_inv
    psi1 = child.generator
    psi1_inv = child.generator_inv

    groups = [(psi1, psi1_inv, 2), (psi1, psi1_inv, 2)]

    key = jrandom.PRNGKey(42)
    n = 30
    samples = []

    t0 = time.time()
    for i in range(n):
        key, k1, k2 = jrandom.split(key, 3)
        V0 = jrandom.gamma(k1, 1.0 / theta0, dtype=jnp.float64)
        u = mo_rosenblatt_sample(k2, psi0, psi0_inv, groups, V0)
        samples.append(u)
        if i < 5:
            print(f"  Sample {i}: {u}")
    elapsed = time.time() - t0

    samples = jnp.stack(samples)
    print(f"\n{n} samples in {elapsed:.1f}s ({elapsed/n*1000:.0f} ms/sample)")
    print(f"Shape: {samples.shape}")
    print(f"Range: [{jnp.min(samples):.6f}, {jnp.max(samples):.6f}]")
    print(f"Mean per dim: {jnp.mean(samples, axis=0)}")

    # Basic sanity: all in (0,1), mean near 0.5
    assert jnp.all(samples > 0) and jnp.all(samples < 1), "Samples out of (0,1)"
    for d in range(4):
        m = float(jnp.mean(samples[:, d]))
        assert 0.2 < m < 0.8, f"Mean of dim {d} = {m}, expected near 0.5"
    print("Basic sanity checks passed!")

    # Check correlation structure qualitatively
    # Within-group correlation should be higher
    corr_within = float(jnp.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    corr_between = float(jnp.corrcoef(samples[:, 0], samples[:, 2])[0, 1])
    print(f"\nPearson corr (within group):  {corr_within:.4f}")
    print(f"Pearson corr (between group): {corr_between:.4f}")
    print(f"Within > Between: {corr_within > corr_between}")

    # Verify the CDF formula: plug samples into the copula CDF
    # C(u1,u2,u3,u4) should be in (0,1) and have mean ~ 2^{-d} for high dep
    def C(u):
        t1 = psi1_inv(u[0]) + psi1_inv(u[1])
        t2 = psi1_inv(u[2]) + psi1_inv(u[3])
        s = psi0_inv(psi1(t1)) + psi0_inv(psi1(t2))
        return psi0(s)

    cdf_vals = jnp.array([C(samples[i]) for i in range(n)])
    print(f"\nCDF values: mean={jnp.mean(cdf_vals):.4f}, std={jnp.std(cdf_vals):.4f}")
    print(f"CDF range: [{jnp.min(cdf_vals):.6f}, {jnp.max(cdf_vals):.6f}]")
