"""Quick test for 3-level nested Bell polynomial Rosenblatt."""
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)

from acopula import compile_model, copula, marginal

@copula
class Clayton:
    theta: float
    def generator(self, t):
        return jnp.power(1.0 + t, -1.0 / self.theta)

@copula
class Gumbel:
    theta: float
    def generator(self, t):
        return jnp.exp(-jnp.power(t, 1.0 / self.theta))

class UniformDist:
    def __init__(self, **kwargs):
        pass
    def quantile(self, u):
        return u
    def log_prob(self, x):
        return jnp.array(0.0)
    def cdf(self, x):
        return x
    @property
    def parameters(self):
        return {}


def main():
    # 3-level: Clayton(1.0) > Gumbel(3.0) > Clayton(5.0)
    # 2 groups x 2 sectors x 2 leaves = 8 dims
    n_groups, n_sectors, n_leaves = 2, 2, 2
    d = n_groups * n_sectors * n_leaves

    def model(params, obs):
        root = Clayton(params[0])
        groups = []
        leaf_idx = 0
        for g in range(n_groups):
            mid = Gumbel(params[1])
            sectors = []
            for s in range(n_sectors):
                leaf_cop = Clayton(params[2])
                lvs = [
                    marginal(UniformDist(), obs=obs[leaf_idx + j])
                    for j in range(n_leaves)
                ]
                sectors.append(leaf_cop(lvs))
                leaf_idx += n_leaves
            groups.append(mid(sectors))
        return root(groups)

    cm = compile_model(model, template=jnp.array([1.0, 3.0, 5.0]))

    key = jrandom.PRNGKey(42)
    n = 500

    print(f"3-level nested: Clayton(1.0) > Gumbel(3.0) > Clayton(5.0)")
    print(f"  {n_groups} groups x {n_sectors} sectors x {n_leaves} leaves = {d} dims")
    print(f"  Sampling {n} points...")

    samples = np.asarray(cm.sample(key, n, jnp.array([1.0, 3.0, 5.0]), method="rosenblatt"))
    print(f"  Shape: {samples.shape}")
    print(f"  Range: [{samples.min():.6f}, {samples.max():.6f}]")
    print(f"  NaN: {np.any(np.isnan(samples))}")

    for j in range(d):
        ks_stat, p_val = stats.kstest(samples[:, j], 'uniform')
        mean = samples[:, j].mean()
        status = "PASS" if p_val > 0.01 else "FAIL"
        print(f"  dim {j}: mean={mean:.4f}, KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

    # Hierarchical tau structure
    tau_same_sector, _ = stats.kendalltau(samples[:, 0], samples[:, 1])
    tau_same_group, _ = stats.kendalltau(samples[:, 0], samples[:, 2])
    tau_diff_group, _ = stats.kendalltau(samples[:, 0], samples[:, n_sectors * n_leaves])
    print(f"\n  Tau structure:")
    print(f"    same sector:            {tau_same_sector:.4f}")
    print(f"    same group diff sector: {tau_same_group:.4f}")
    print(f"    different group:        {tau_diff_group:.4f}")
    ok = tau_same_sector > tau_same_group > tau_diff_group
    print(f"    hierarchy ok: {ok}")

    all_uniform = all(
        stats.kstest(samples[:, j], 'uniform')[1] > 0.01
        for j in range(d)
    )
    print(f"\n  Overall: {'PASS' if all_uniform and ok else 'FAIL'}")


if __name__ == "__main__":
    main()
