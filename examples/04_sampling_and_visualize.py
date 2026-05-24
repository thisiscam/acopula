"""Sample from a nested copula and visualize its structure.

Builds a two-level Clayton-over-Frank copula (one root, three sectors of two
leaves), draws samples with the Marshall-Olkin sampler, and writes two
figures:

  * ``acopula_tree.png``  - the copula tree (root / sectors / leaves)
  * ``acopula_pairs.png`` - a pairwise scatter of the sampled margins

Run:
    python examples/04_sampling_and_visualize.py [output_dir]
"""

import sys
from pathlib import Path

import jax.random as jrandom
import matplotlib
matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt
import numpy as np

from acopula import compile_model, copula, marginal


class Uniform:
    """Trivial marginal: data already on the copula (unit-cube) scale."""
    def quantile(self, u):
        return u

    def cdf(self, x):
        return x

    def log_prob(self, x):
        return 0.0


@copula
class Clayton:
    theta: float

    def generator(self, t):
        return (1.0 + t) ** (-1.0 / self.theta)


@copula
class Frank:
    theta: float

    def generator(self, t):
        import jax.numpy as jnp
        return -jnp.log1p(jnp.expm1(-self.theta) * jnp.exp(-t)) / self.theta


def model(params, u):
    """Clayton root over three Frank sectors, each with two uniform leaves."""
    root = Clayton(params[0])
    sector = Frank(params[1])
    return root(
        sector(marginal(Uniform(), obs=u[i, j]) for j in range(2))
        for i in range(3)
    )


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    import jax.numpy as jnp
    params = jnp.array([2.0, 5.0])  # theta_root (Clayton), theta_sector (Frank)
    cm = compile_model(model, template=params)

    # ---- draw the copula tree ----
    fig, ax = cm.visualize(include_leaves=True, layout="hierarchical")
    tree_path = out_dir / "acopula_tree.png"
    fig.savefig(tree_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {tree_path}")

    # ---- sample and plot pairwise margins ----
    samples = cm.sample(jrandom.PRNGKey(0), 2000, params, method="marshall_olkin")
    s = np.asarray(samples).reshape(samples.shape[0], -1)  # (n, 6)
    d = s.shape[1]

    fig, axes = plt.subplots(d, d, figsize=(10, 10))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(s[:, i], bins=30, color="tab:blue", alpha=0.7)
            else:
                ax.scatter(s[:, j], s[:, i], s=2, alpha=0.2, color="tab:blue")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pairwise margins (within-sector pairs are more dependent)")
    pairs_path = out_dir / "acopula_pairs.png"
    fig.savefig(pairs_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pairs_path}")


if __name__ == "__main__":
    main()
