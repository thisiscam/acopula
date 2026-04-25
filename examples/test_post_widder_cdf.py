#!/usr/bin/env python3
"""Test Post-Widder CDF approximation for Archimedean copula generators.

Supports:
- Clayton: psi(t) = (1 + theta * t)^(-1/theta), Laplace of Gamma(1/theta, 1)
- Frank: psi(t) = -ln(exp(-t) * (exp(-theta) - 1) + 1) / theta, discrete distribution
"""

import argparse
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import from our acopula package
from acopula.core import _post_widder_cdf

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def clayton_generator(t: jax.Array, theta: float) -> jax.Array:
    """Clayton generator psi(t) = (1 + theta * t)^(-1/theta)."""
    return (1.0 + t) ** (-1.0 / theta)


def frank_generator(u: jax.Array, theta: float) -> jax.Array:
    """Frank generator psi(t) = -ln(exp(-t) * (exp(-theta) - 1) + 1) / theta."""
    return -jnp.log(jnp.exp(-u) * (jnp.exp(-theta) - 1.0) + 1) / theta


def gamma_cdf(x: jax.Array, shape: float) -> jax.Array:
    """Gamma(shape, scale=1) CDF (value 0 for x<=0)."""
    cdf = jax.scipy.stats.gamma.cdf(x, a=shape, scale=1.0)
    return jnp.where(x > 0, cdf, 0.0)


def frank_exact_pmf(k: int, theta: float) -> float:
    """Exact PMF for Frank at integer k: P(V = k) = (1 - exp(-theta))^k / (k * theta) for k >= 1."""
    if k == 0:
        return 0.0
    return (1.0 - jnp.exp(-theta)) ** k / (k * theta)


def run_experiment(
    copula_type: str,
    theta: float,
    ks: Sequence[int],
    x_min: float,
    x_max: float,
    num_x: int,
    save_path: str | None = None,
) -> None:
    """Compute Post-Widder CDF estimates for a range of k and plot vs exact CDF.

    Args:
      copula_type: 'clayton' or 'frank'
      theta: Copula parameter (>0).
      ks: list of Post-Widder orders (positive integers).
      x_min, x_max: x-range for plotting (>0).
      num_x: number of x grid points.
      save_path: optional path to save the plot; if None, show interactively.
    """
    if copula_type == "clayton":

        def psi_with_theta(t):
            return clayton_generator(t, theta)

        # Exact CDF: Gamma(1/theta, 1)
        x_grid = jnp.linspace(x_min, x_max, num_x)
        shape = 1.0 / theta
        cdf_exact = gamma_cdf(x_grid, shape)
        title = f"Post-Widder CDF for Clayton generator (theta={theta:g})"

    elif copula_type == "frank":

        def psi_with_theta(t):
            return frank_generator(t, theta)

        # Exact CDF: discrete distribution over non-negative integers {0, 1, 2, ...}
        # PMF: P(V = k) = (1 - exp(-theta))^k / (k * theta) for k >= 1, P(V = 0) = 0
        x_grid = jnp.arange(0, int(x_max) + 1)
        pmf_vals = jnp.array([frank_exact_pmf(int(k_val), theta) for k_val in x_grid])
        cdf_exact = jnp.cumsum(pmf_vals)  # Convert PMF to CDF
        title = f"Post-Widder CDF for Frank generator (theta={theta:g})"

    else:
        raise ValueError(f"Unknown copula_type: {copula_type}")

    # Compute Post-Widder CDF curves
    pw_cdfs = []
    for k in ks:
        # vmap over x to compute F_k(x)
        fk = jax.vmap(lambda x: _post_widder_cdf(psi_with_theta, x, k))(x_grid)
        pw_cdfs.append(fk)

    # Plot with distinct linestyles per k
    linestyles = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    plt.figure(figsize=(7.2, 4.8))

    if copula_type == "frank":
        plt.step(
            x_grid, cdf_exact, "k-", lw=2.0, where="post", label="Exact (Discrete CDF)"
        )
    else:
        plt.plot(x_grid, cdf_exact, "k-", lw=2.0, label="Exact (Gamma CDF)")

    for idx, (k, fk) in enumerate(zip(ks, pw_cdfs)):
        ls = linestyles[idx % len(linestyles)]
        plt.plot(x_grid, fk, lw=1.6, linestyle=ls, label=f"Post-Widder k={k}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test Post-Widder CDF approximation for copula generators"
    )
    parser.add_argument(
        "--copula",
        type=str,
        default="clayton",
        choices=["clayton", "frank"],
        help="Copula type: clayton or frank",
    )
    parser.add_argument(
        "--theta", type=float, default=2.0, help="Copula parameter (>0)"
    )
    parser.add_argument("--kmin", type=int, default=1, help="Min Post-Widder order")
    parser.add_argument("--kmax", type=int, default=9, help="Max Post-Widder order")
    parser.add_argument(
        "--kstep", type=int, default=2, help="Step size for Post-Widder order"
    )
    parser.add_argument("--xmin", type=float, default=1e-3, help="Minimum x for plot")
    parser.add_argument("--xmax", type=float, default=8.0, help="Maximum x for plot")
    parser.add_argument("--nx", type=int, default=200, help="Number of x grid points")
    parser.add_argument(
        "--save", type=str, default="", help="If provided, save plot to this path"
    )
    args = parser.parse_args()

    ks = list(range(args.kmin, args.kmax + 1, args.kstep))
    save_path = args.save if args.save else None

    print("Testing Post-Widder CDF approximation:")
    print(f"  copula = {args.copula}")
    print(f"  theta = {args.theta}")
    print(f"  k values = {ks}")
    print(f"  x range = [{args.xmin}, {args.xmax}]")
    print()

    run_experiment(
        args.copula, args.theta, ks, args.xmin, args.xmax, args.nx, save_path
    )


if __name__ == "__main__":
    main()
