#!/usr/bin/env python3
"""Test Post-Widder inversion on a simple discontinuous function.

We test f(t) = indicator{t < 1} (i.e., f(t) = 1 if t < 1, else 0).

The Laplace transform is:
    F(s) = ∫₀^∞ f(t) e^(-st) dt = ∫₀^1 e^(-st) dt = (1 - e^(-s))/s

The Post-Widder formula inverts F(s) to approximate f(t):
    f_k(t) = (-1)^k * (k/t)^(k+1) / k! * F^(k)(k/t)

where F^(k) is the k-th derivative of F(s).
"""

import argparse
from typing import Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import factorial
import jet_array

# Import from our acopula package


def _kth_derivative(
    fun: Callable[[jax.Array], jax.Array], x: jax.Array, k: int
) -> jax.Array:
    """Compute the k-th derivative of a scalar function at x using jet.

    Returns d^k/dx^k fun(x) as a scalar.
    """
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return fun(x)

    # Build series input for jet: provide coefficients for derivatives of the inner identity h(z)=x+z
    # series_input[j-1] corresponds to the j-th directional derivative of h at 0 in direction v=1.
    series_input = jnp.zeros(k)
    series_input = series_input.at[0].set(1.0)

    _, series_out = jet_array.jet(fun, (x,), (series_input,))
    # jet returns [f'(x), f''(x), ..., f^{(k)}(x)] without factorial scaling
    return jnp.asarray(series_out[-1])


# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def indicator_function(t: jax.Array) -> jax.Array:
    """Original function: f(t) = 1 if t < 1, else 0."""
    return jnp.where(t < 1.0, 0.0, 1.0)


def laplace_transform(s: jax.Array) -> jax.Array:
    """Laplace transform of indicator function: F(s) = (1 - e^(-s))/s."""
    return jnp.exp(-s) / s


def post_widder_inversion(F: callable, t: jax.Array, k: int) -> jax.Array:
    """Post-Widder inversion formula to approximate f(t) from its Laplace transform F(s).

    f_k(t) = (-1)^k * (k/t)^(k+1) / k! * F^(k)(k/t)

    Args:
        F: Laplace transform function F(s)
        t: point at which to evaluate f(t)
        k: order of the Post-Widder approximation

    Returns:
        Approximation f_k(t)
    """
    tiny = jnp.finfo(float).eps
    t_safe = jnp.maximum(t, tiny)
    s = k / t_safe

    # Compute k-th derivative of F at s = k/t
    F_k = _kth_derivative(F, s, k)

    # Post-Widder formula
    sign = -1.0 if (k % 2 == 1) else 1.0
    log_coef = (k + 1) * jnp.log(s)  # - jnp.log(factorial(k))
    f_approx = sign * jnp.exp(log_coef) * F_k
    return f_approx
    # Handle edge cases
    f_approx = jnp.where(jnp.isfinite(f_approx), f_approx, 0.0)
    return jnp.where(t > 0, f_approx, 0.0)


def run_experiment(
    k_values: list[int],
    t_min: float,
    t_max: float,
    num_t: int,
    save_path: str | None = None,
) -> None:
    """Compute Post-Widder approximations for various orders and plot.

    Args:
        k_values: list of Post-Widder orders to test
        t_min, t_max: time range for plotting
        num_t: number of time grid points
        save_path: optional path to save the plot; if None, show interactively
    """
    # Create time grid
    t_grid = jnp.linspace(t_min, t_max, num_t)

    # Compute exact function values
    f_exact = jax.vmap(indicator_function)(t_grid)

    # Compute Post-Widder approximations for each k
    pw_approx = []
    for k in k_values:
        print(f"Computing Post-Widder approximation for k={k}...")
        f_k = jax.vmap(lambda t: post_widder_inversion(laplace_transform, t, k))(t_grid)
        pw_approx.append(f_k)

    # Plot results
    linestyles = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = plt.cm.viridis(jnp.linspace(0.2, 0.9, len(k_values)))

    plt.figure(figsize=(10, 6))

    # Plot exact function
    plt.plot(t_grid, f_exact, "k-", lw=2.5, label="Exact: f(t) = 𝟙{t < 1}", zorder=10)

    # Plot Post-Widder approximations
    for idx, (k, f_k) in enumerate(zip(k_values, pw_approx)):
        ls = linestyles[idx % len(linestyles)]
        plt.plot(
            t_grid,
            f_k,
            lw=1.8,
            linestyle=ls,
            color=colors[idx],
            label=f"Post-Widder k={k}",
            alpha=0.8,
        )

    # Add vertical line at discontinuity
    plt.axvline(
        x=1.0,
        color="red",
        linestyle=":",
        alpha=0.5,
        lw=1.5,
        label="Discontinuity at t=1",
    )

    plt.title(
        "Post-Widder Inversion of Indicator Function", fontsize=14, fontweight="bold"
    )
    plt.xlabel("t", fontsize=12)
    plt.ylabel("f(t)", fontsize=12)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"\nSaved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test Post-Widder inversion on indicator function"
    )
    parser.add_argument("--kmin", type=int, default=1, help="Minimum Post-Widder order")
    parser.add_argument(
        "--kmax", type=int, default=10, help="Maximum Post-Widder order"
    )
    parser.add_argument(
        "--kstep", type=int, default=1, help="Step size for Post-Widder order"
    )
    parser.add_argument("--tmin", type=float, default=0.01, help="Minimum t for plot")
    parser.add_argument("--tmax", type=float, default=2.5, help="Maximum t for plot")
    parser.add_argument("--nt", type=int, default=500, help="Number of t grid points")
    parser.add_argument(
        "--save", type=str, default="", help="If provided, save plot to this path"
    )
    args = parser.parse_args()

    k_values = list(range(args.kmin, args.kmax + 1, args.kstep))
    save_path = args.save if args.save else None

    print("=" * 60)
    print("Testing Post-Widder Inversion on Discontinuous Function")
    print("=" * 60)
    print(f"Function: f(t) = 1 if t < 1, else 0")
    print(f"Laplace transform: F(s) = (1 - e^(-s))/s")
    print(f"Post-Widder orders: k = {k_values}")
    print(f"Time range: [{args.tmin}, {args.tmax}]")
    print("=" * 60)
    print()

    run_experiment(k_values, args.tmin, args.tmax, args.nt, save_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
