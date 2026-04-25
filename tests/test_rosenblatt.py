"""Test Rosenblatt transform sampler against Marshall-Olkin and basic statistics.

Tests both non-nested (flat) and nested (hierarchical) copulas.
"""

import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)

from acopula.core import copula, defmodel, marginal

# Copula definitions
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

@copula
class AMH:
    theta: float
    def generator(self, t):
        return (1.0 - self.theta) / (jnp.exp(t) - self.theta)

@copula
class Joe:
    theta: float
    def generator(self, t):
        return 1.0 - jnp.power(1.0 - jnp.exp(-t), 1.0 / self.theta)


# Trivial uniform marginals for pure copula testing
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


def make_flat_model(copula_cls, theta, d):
    """Create a d-dimensional non-nested model."""
    @defmodel
    def model(params, obs):
        cop = copula_cls(params[0])
        return cop([marginal(UniformDist(), obs=obs[i]) for i in range(d)])
    model.set_params(jnp.array([theta]))
    return model


def make_nested_2level(outer_cls, outer_theta, inner_cls, inner_theta,
                       groups, leaves_per_group):
    """Create a 2-level nested model: outer(inner(leaves), inner(leaves), ...).

    Args:
        outer_cls, outer_theta: outer (root) copula
        inner_cls, inner_theta: inner (group) copula
        groups: number of groups
        leaves_per_group: leaves per group
    """
    total_leaves = groups * leaves_per_group

    @defmodel
    def model(params, obs):
        outer = outer_cls(params[0])
        children = []
        leaf_idx = 0
        for g in range(groups):
            inner = inner_cls(params[1])
            group_leaves = [
                marginal(UniformDist(), obs=obs[leaf_idx + j])
                for j in range(leaves_per_group)
            ]
            children.append(inner(group_leaves))
            leaf_idx += leaves_per_group
        return outer(children)

    model.set_params(jnp.array([outer_theta, inner_theta]))
    return model


def make_nested_3level(root_cls, root_theta, mid_cls, mid_theta,
                       leaf_cls, leaf_theta,
                       n_groups, n_sectors_per_group, n_leaves_per_sector):
    """Create a 3-level nested model: root(mid(leaf_cop(leaves)))."""
    @defmodel
    def model(params, obs):
        root = root_cls(params[0])
        groups = []
        leaf_idx = 0
        for g in range(n_groups):
            mid = mid_cls(params[1])
            sectors = []
            for s in range(n_sectors_per_group):
                leaf_cop = leaf_cls(params[2])
                lvs = [
                    marginal(UniformDist(), obs=obs[leaf_idx + j])
                    for j in range(n_leaves_per_sector)
                ]
                sectors.append(leaf_cop(lvs))
                leaf_idx += n_leaves_per_sector
            groups.append(mid(sectors))
        return root(groups)

    model.set_params(jnp.array([root_theta, mid_theta, leaf_theta]))
    return model


def check_samples(samples, name, d, expected_tau_range=None):
    """Run standard checks on samples."""
    print(f"  Shape: {samples.shape}")
    print(f"  Range: [{samples.min():.6f}, {samples.max():.6f}]")

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(samples))
    has_inf = np.any(np.isinf(samples))
    in_range = np.all((samples > 0) & (samples < 1))
    print(f"  NaN: {has_nan}, Inf: {has_inf}, In (0,1): {in_range}")

    if has_nan or has_inf or not in_range:
        return False

    # Marginal uniformity
    all_uniform = True
    print(f"  Marginal KS tests:")
    for j in range(min(d, 6)):
        ks_stat, p_val = stats.kstest(samples[:, j], 'uniform')
        status = "PASS" if p_val > 0.01 else "FAIL"
        if p_val <= 0.01:
            all_uniform = False
        print(f"    dim {j}: KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

    # Kendall's tau for adjacent pairs
    print(f"  Kendall tau (adjacent):")
    for i in range(min(3, d - 1)):
        tau, p_val = stats.kendalltau(samples[:, i], samples[:, i + 1])
        print(f"    ({i},{i+1}): tau={tau:.4f}")

    return not has_nan and not has_inf and in_range and all_uniform


def test_flat_copula(name, copula_cls, theta, d=5, n=2000):
    """Test Rosenblatt on a flat (non-nested) copula."""
    print(f"\n{'='*60}")
    print(f"FLAT: {name}(theta={theta}), d={d}, n={n}")
    print(f"{'='*60}")

    model = make_flat_model(copula_cls, theta, d)
    key = jrandom.PRNGKey(42)

    print("  Sampling with Rosenblatt...")
    samples = np.asarray(model.sample(key, n, method="rosenblatt"))
    return check_samples(samples, name, d)


def test_nested_2level(name, outer_cls, outer_theta, inner_cls, inner_theta,
                       groups=3, leaves_per_group=3, n=2000):
    """Test Rosenblatt on a 2-level nested copula."""
    d = groups * leaves_per_group
    print(f"\n{'='*60}")
    print(f"NESTED 2-LEVEL: {name}")
    print(f"  outer={outer_cls.__name__}({outer_theta}), "
          f"inner={inner_cls.__name__}({inner_theta})")
    print(f"  {groups} groups x {leaves_per_group} leaves = {d} dims, n={n}")
    print(f"{'='*60}")

    model = make_nested_2level(outer_cls, outer_theta, inner_cls, inner_theta,
                               groups, leaves_per_group)
    key = jrandom.PRNGKey(42)

    print("  Sampling with Rosenblatt...")
    samples_r = np.asarray(model.sample(key, n, method="rosenblatt"))
    ok_r = check_samples(samples_r, name + " (Rosenblatt)", d)

    print("\n  Sampling with Marshall-Olkin...")
    samples_mo = np.asarray(model.sample(key, n, method="marshall_olkin"))
    ok_mo = check_samples(samples_mo, name + " (Marshall-Olkin)", d)

    # Cross-method comparison: within-group tau should be higher than between-group
    print("\n  Tau structure check (within-group vs between-group):")
    # Within first group
    tau_within, _ = stats.kendalltau(samples_r[:, 0], samples_r[:, 1])
    # Between groups (first leaf of group 0 vs first leaf of group 1)
    tau_between, _ = stats.kendalltau(samples_r[:, 0], samples_r[:, leaves_per_group])
    print(f"    Rosenblatt:    within={tau_within:.4f}, between={tau_between:.4f}, "
          f"within > between: {tau_within > tau_between}")

    tau_within_mo, _ = stats.kendalltau(samples_mo[:, 0], samples_mo[:, 1])
    tau_between_mo, _ = stats.kendalltau(samples_mo[:, 0], samples_mo[:, leaves_per_group])
    print(f"    Marshall-Olkin: within={tau_within_mo:.4f}, between={tau_between_mo:.4f}, "
          f"within > between: {tau_within_mo > tau_between_mo}")

    # Within-group tau should be higher than between-group (nesting structure)
    structure_ok_r = tau_within > tau_between
    structure_ok_mo = tau_within_mo > tau_between_mo

    return ok_r and structure_ok_r


def test_nested_3level(n=2000):
    """Test Rosenblatt on a 3-level nested copula."""
    n_groups = 2
    n_sectors = 2
    n_leaves = 2
    d = n_groups * n_sectors * n_leaves

    print(f"\n{'='*60}")
    print(f"NESTED 3-LEVEL: Clayton(1.0) > Gumbel(3.0) > Clayton(5.0)")
    print(f"  {n_groups} groups x {n_sectors} sectors x {n_leaves} leaves = {d} dims, n={n}")
    print(f"{'='*60}")

    model = make_nested_3level(
        Clayton, 1.0,   # root
        Gumbel, 3.0,    # mid
        Clayton, 5.0,   # leaf
        n_groups, n_sectors, n_leaves,
    )
    key = jrandom.PRNGKey(42)

    print("  Sampling with Rosenblatt...")
    samples_r = np.asarray(model.sample(key, n, method="rosenblatt"))
    ok = check_samples(samples_r, "3-level", d)

    # Check hierarchical tau structure:
    # Same sector > same group diff sector > different group
    print("\n  Hierarchical tau structure:")
    # Same sector (dims 0,1)
    tau_same_sector, _ = stats.kendalltau(samples_r[:, 0], samples_r[:, 1])
    # Same group, different sector (dims 0,2)
    tau_same_group, _ = stats.kendalltau(samples_r[:, 0], samples_r[:, 2])
    # Different group (dims 0, n_sectors*n_leaves)
    tau_diff_group, _ = stats.kendalltau(
        samples_r[:, 0], samples_r[:, n_sectors * n_leaves])
    print(f"    same sector: {tau_same_sector:.4f}")
    print(f"    same group, diff sector: {tau_same_group:.4f}")
    print(f"    different group: {tau_diff_group:.4f}")
    print(f"    hierarchy ok: {tau_same_sector > tau_same_group > tau_diff_group}")

    return ok and (tau_same_sector > tau_same_group > tau_diff_group)


def main():
    results = {}

    # 1. Flat copula tests
    families = [
        ("Clayton", Clayton, 2.0),
        ("Gumbel", Gumbel, 2.0),
        ("Frank", Frank, 5.0),
        ("AMH", AMH, 0.8),
        ("Joe", Joe, 2.0),
    ]
    for name, cls, theta in families:
        try:
            ok = test_flat_copula(name, cls, theta, d=5, n=2000)
            results[f"flat_{name}"] = "PASS" if ok else "FAIL"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[f"flat_{name}"] = f"ERROR: {e}"

    # 2. Nested 2-level tests
    nested_cases = [
        ("Clayton>Clayton", Clayton, 1.0, Clayton, 5.0),
        ("Clayton>Gumbel", Clayton, 1.0, Gumbel, 3.0),
        ("Gumbel>Gumbel", Gumbel, 2.0, Gumbel, 4.0),
        ("Frank>Frank", Frank, 2.0, Frank, 8.0),
    ]
    for name, outer_cls, outer_theta, inner_cls, inner_theta in nested_cases:
        try:
            ok = test_nested_2level(name, outer_cls, outer_theta,
                                    inner_cls, inner_theta,
                                    groups=3, leaves_per_group=3, n=2000)
            results[f"nested_{name}"] = "PASS" if ok else "FAIL"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[f"nested_{name}"] = f"ERROR: {e}"

    # 3. Nested 3-level test
    try:
        ok = test_nested_3level(n=2000)
        results["nested_3level"] = "PASS" if ok else "FAIL"
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["nested_3level"] = f"ERROR: {e}"

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    n_pass = sum(1 for s in results.values() if s == "PASS")
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} passed")


if __name__ == "__main__":
    main()
