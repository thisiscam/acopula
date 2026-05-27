"""Correctness test: tree-recursive Rosenblatt vs Marshall-Olkin sampling.

Tests:
1. Marginal uniformity (KS test)
2. Kendall tau match between Rosenblatt and MO on same-family models
3. Cross-family model (Frank/Clayton) where MO cannot work
"""

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


@copula
class Frank:
    theta: float
    def generator(self, t):
        return -jnp.log1p(jnp.exp(-t) * jnp.expm1(-self.theta)) / self.theta


class UniformDist:
    """Trivial uniform marginals for pure copula testing."""
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


def make_nested_model(outer_cls, outer_theta, inner_cls, inner_theta,
                      n_sectors, leaves_per_sector):
    """Create a 2-level nested copula model."""
    d = n_sectors * leaves_per_sector

    def model(params, obs):
        outer = outer_cls(params[0])
        inner_cops = [inner_cls(params[1 + i]) for i in range(n_sectors)]
        return outer([
            inner_cops[s]([
                marginal(UniformDist(), obs=obs[s * leaves_per_sector + j])
                for j in range(leaves_per_sector)
            ])
            for s in range(n_sectors)
        ])

    params = jnp.array([outer_theta] + [inner_theta] * n_sectors)
    cm = compile_model(model, template=params)
    return cm, params, d


def kendall_tau_matrix(samples):
    """Compute pairwise Kendall tau matrix."""
    d = samples.shape[1]
    tau = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = stats.kendalltau(samples[:, i], samples[:, j]).statistic
            tau[j, i] = tau[i, j]
    return tau


def test_marginal_uniformity():
    """Test 1: Each margin from Rosenblatt should be Uniform(0,1)."""
    print("=" * 60)
    print("TEST 1: Marginal uniformity (Rosenblatt)")
    print("=" * 60)

    cm, params, d = make_nested_model(
        Clayton, 1.0, Clayton, 3.0, n_sectors=2, leaves_per_sector=3
    )

    key = jrandom.PRNGKey(42)
    n = 1000
    samples = cm.sample(key, n, params, method="rosenblatt")
    samples_np = np.array(samples)

    print(f"Model: Clayton(1.0) / Clayton(3.0), 2 sectors x 3 leaves = {d}d")
    print(f"Samples: {n}")

    all_pass = True
    for j in range(d):
        ks_stat, p_val = stats.kstest(samples_np[:, j], 'uniform')
        status = "PASS" if p_val > 0.01 else "FAIL"
        if p_val <= 0.01:
            all_pass = False
        print(f"  Margin {j}: KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_mo_vs_rosenblatt():
    """Test 2a: Compare Kendall tau between MO and Rosenblatt on same-family."""
    print("\n" + "=" * 60)
    print("TEST 2a: MO vs Rosenblatt (Kendall tau comparison)")
    print("=" * 60)

    cm, params, d = make_nested_model(
        Clayton, 1.0, Clayton, 3.0, n_sectors=2, leaves_per_sector=3
    )

    n = 500
    key_mo = jrandom.PRNGKey(100)
    key_ros = jrandom.PRNGKey(200)

    print(f"Model: Clayton(1.0) / Clayton(3.0), 2x3 = {d}d")
    print(f"Samples: {n} per method")

    import time
    print("\nSampling with MO...")
    t0 = time.time()
    samples_mo = cm.sample(key_mo, n, params, method="marshall_olkin")
    t_mo = time.time() - t0
    print(f"  MO time: {t_mo:.1f}s")

    print("Sampling with Rosenblatt...")
    t0 = time.time()
    samples_ros = cm.sample(key_ros, n, params, method="rosenblatt")
    t_ros = time.time() - t0
    print(f"  Rosenblatt time: {t_ros:.1f}s")

    tau_mo = kendall_tau_matrix(np.array(samples_mo))
    tau_ros = kendall_tau_matrix(np.array(samples_ros))

    print("\nKendall tau comparison:")
    max_diff = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            diff = abs(tau_mo[i, j] - tau_ros[i, j])
            max_diff = max(max_diff, diff)
            sector_i = i // 3
            sector_j = j // 3
            pair_type = "within" if sector_i == sector_j else "between"
            print(f"  ({i},{j}) [{pair_type}]: MO={tau_mo[i,j]:.3f}, "
                  f"Ros={tau_ros[i,j]:.3f}, diff={diff:.3f}")

    threshold = 0.06
    status = "PASS" if max_diff < threshold else "FAIL"
    print(f"\nMax tau difference: {max_diff:.4f} (threshold: {threshold})")
    print(f"Result: {status}")
    return max_diff < threshold


def test_kendall_tau_structure():
    """Test 2: Check Kendall tau matches theoretical values for Clayton."""
    print("\n" + "=" * 60)
    print("TEST 2: Kendall tau vs theoretical (Rosenblatt)")
    print("=" * 60)

    # Clayton(theta) has tau = theta / (theta + 2)
    inner_theta = 3.0
    outer_theta = 1.0
    tau_inner_theory = inner_theta / (inner_theta + 2)  # 0.6
    tau_outer_theory = outer_theta / (outer_theta + 2)  # 0.333

    cm, params, d = make_nested_model(
        Clayton, outer_theta, Clayton, inner_theta,
        n_sectors=2, leaves_per_sector=3
    )

    n = 3000
    key = jrandom.PRNGKey(200)

    print(f"Model: Clayton({outer_theta}) / Clayton({inner_theta}), 2x3 = {d}d")
    print(f"Theoretical within-sector tau: {tau_inner_theory:.3f}")
    print(f"Theoretical between-sector tau: ~{tau_outer_theory:.3f} (approx)")
    print(f"Samples: {n}")

    import time
    t0 = time.time()
    samples = cm.sample(key, n, params, method="rosenblatt")
    elapsed = time.time() - t0
    print(f"Sampling time: {elapsed:.1f}s")

    tau = kendall_tau_matrix(np.array(samples))

    within_taus = []
    between_taus = []
    for i in range(d):
        for j in range(i + 1, d):
            sector_i = i // 3
            sector_j = j // 3
            if sector_i == sector_j:
                within_taus.append(tau[i, j])
            else:
                between_taus.append(tau[i, j])

    mean_within = np.mean(within_taus)
    mean_between = np.mean(between_taus)

    print(f"\nWithin-sector tau:  {mean_within:.3f} (theory: {tau_inner_theory:.3f})")
    print(f"Between-sector tau: {mean_between:.3f} (theory: ~{tau_outer_theory:.3f})")

    # Within-sector should be close to theoretical (tolerance for n=3000)
    within_ok = abs(mean_within - tau_inner_theory) < 0.05
    # Between should be less than within and positive
    structure_ok = 0 < mean_between < mean_within
    # Between should be in a reasonable range (not exact for nested)
    between_reasonable = mean_between > 0.1 and mean_between < 0.5

    print(f"\nWithin-sector match: {'PASS' if within_ok else 'FAIL'} "
          f"(diff={abs(mean_within - tau_inner_theory):.3f})")
    print(f"Structure (within > between > 0): {'PASS' if structure_ok else 'FAIL'}")
    print(f"Between range [0.1, 0.5]: {'PASS' if between_reasonable else 'FAIL'}")

    passed = within_ok and structure_ok and between_reasonable
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_cross_family():
    """Test 3: Cross-family Frank/Clayton (Rosenblatt only, MO cannot work)."""
    print("\n" + "=" * 60)
    print("TEST 3: Cross-family Frank(0.5)/Clayton(2.0) — Rosenblatt only")
    print("=" * 60)

    # Frank(0.5) outer / Clayton(2.0) inner
    # This pair passes 8-monotonicity check per our nesting analysis
    cm, params, d = make_nested_model(
        Frank, 0.5, Clayton, 2.0, n_sectors=2, leaves_per_sector=3
    )

    n = 1000
    key = jrandom.PRNGKey(300)

    print(f"Model: Frank(0.5) / Clayton(2.0), 2x3 = {d}d")
    print(f"Samples: {n}")

    print("\nSampling with Rosenblatt...")
    import time
    t0 = time.time()
    samples = cm.sample(key, n, params, method="rosenblatt")
    t_ros = time.time() - t0
    samples_np = np.array(samples)
    print(f"  Time: {t_ros:.1f}s")

    # Check for NaN/Inf
    n_nan = np.isnan(samples_np).sum()
    n_inf = np.isinf(samples_np).sum()
    print(f"  NaN: {n_nan}, Inf: {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("  FAIL: NaN/Inf in samples")
        return False

    # Check range [0, 1]
    in_range = (samples_np >= 0).all() and (samples_np <= 1).all()
    print(f"  All in [0,1]: {in_range}")

    # Marginal uniformity
    print("\nMarginal uniformity:")
    all_pass = True
    for j in range(d):
        ks_stat, p_val = stats.kstest(samples_np[:, j], 'uniform')
        status = "PASS" if p_val > 0.01 else "FAIL"
        if p_val <= 0.01:
            all_pass = False
        print(f"  Margin {j}: KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

    # Kendall tau structure: within-sector should be stronger than between
    tau = kendall_tau_matrix(samples_np)
    within_taus = []
    between_taus = []
    for i in range(d):
        for j in range(i + 1, d):
            if i // 3 == j // 3:
                within_taus.append(tau[i, j])
            else:
                between_taus.append(tau[i, j])

    mean_within = np.mean(within_taus)
    mean_between = np.mean(between_taus)
    print(f"\nDependence structure:")
    print(f"  Mean within-sector tau:  {mean_within:.3f}")
    print(f"  Mean between-sector tau: {mean_between:.3f}")
    print(f"  Within > between: {mean_within > mean_between}")

    structure_ok = mean_within > mean_between
    print(f"\nResult: {'PASS' if all_pass and structure_ok else 'FAIL'}")
    return all_pass and structure_ok


if __name__ == "__main__":
    results = {}
    results["marginal_uniformity"] = test_marginal_uniformity()
    results["mo_vs_rosenblatt"] = test_mo_vs_rosenblatt()
    results["kendall_tau"] = test_kendall_tau_structure()
    results["cross_family"] = test_cross_family()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
