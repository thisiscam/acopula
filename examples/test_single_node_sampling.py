#!/usr/bin/env python3
"""Test sampling V for a single Frank node with one Clayton ancestor.

This isolates the Post-Widder sampling to debug why bisection isn't working.
We assume V_clayton = 0.845 is already sampled, and we want to sample V_frank.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from oryx import core as oryx_core

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def clayton_generator(theta: float):
    """Return Clayton generator function."""

    def gen(u: jax.Array) -> jax.Array:
        return (1.0 + u) ** (-1.0 / theta)

    return gen


def clayton_generator_inv(theta: float):
    """Return Clayton generator inverse function using oryx."""
    gen = clayton_generator(theta)
    return oryx_core.inverse(gen)


def frank_generator(theta: float):
    """Return Frank generator function."""

    def gen(u: jax.Array) -> jax.Array:
        return -jnp.log1p(jnp.expm1(-theta) * jnp.exp(-u)) / theta

    return gen


def frank_generator_inv(theta: float):
    """Return Frank generator inverse using oryx."""
    gen = frank_generator(theta)
    return oryx_core.inverse(gen)


def build_modified_frank_generator(
    theta_clayton: float, theta_frank: float, v_clayton: float
):
    """Build the modified Frank generator: φ̂_frank(t) = exp(-V_clayton * φ_clayton^{-1}(φ_frank(t)))."""
    outer_gen_inv = clayton_generator_inv(theta_clayton)
    inner_gen = clayton_generator(theta_frank)

    def modified_gen(t: jax.Array) -> jax.Array:
        return jnp.exp(-v_clayton * outer_gen_inv(inner_gen(t)))

    return modified_gen


def main():
    # Parameters
    theta_clayton = 0.2
    theta_frank = 0.9
    v_clayton = 0.8  # Assumed already sampled

    print("=== Setup ===")
    print(f"theta_clayton = {theta_clayton}")
    print(f"theta_frank = {theta_frank}")
    print(f"V_clayton = {v_clayton}")
    print()

    # Build the modified Frank generator (this is what we'll use for Post-Widder)
    psi_modified = build_modified_frank_generator(theta_clayton, theta_frank, v_clayton)

    # Test: evaluate the modified generator at a few points
    print("=== Test modified generator ===")
    t_test = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
    psi_test = jax.vmap(psi_modified)(t_test)
    print(f"t: {t_test}")
    print(f"φ̂_frank(t): {psi_test}")
    print()

    # Now test the Post-Widder CDF
    print("=== Test Post-Widder CDF ===")
    from acopula.core import _post_widder_cdf

    k = 100  # Post-Widder order
    x_grid = jnp.linspace(1e-6, 1000, 1000)
    cdf_vals = jax.vmap(lambda x: _post_widder_cdf(psi_modified, x, k))(x_grid)

    print(f"Post-Widder order k = {k}")
    print(f"x range: [{float(x_grid[0]):.6f}, {float(x_grid[-1]):.2f}]")
    print(f"CDF range: [{float(cdf_vals.min()):.6f}, {float(cdf_vals.max()):.6f}]")
    print()

    # Sample using Post-Widder
    print("=== Sample V_frank ===")
    from acopula.core import _sample_frailty_via_post_widder

    key = jrandom.PRNGKey(42)
    v_frank = _sample_frailty_via_post_widder(key, psi_modified, k, max_cdf_x=1e30)

    print(f"Sampled V_frank = {float(v_frank)}")
    print()

    # Plot the CDF
    print("=== Plotting CDF ===")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot CDF
    ax1.plot(x_grid, cdf_vals, "b-", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("CDF")
    ax1.set_title(f"Post-Widder CDF for modified Frank generator (k={k})")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax1.axhline(y=1, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    # Plot derivative to check monotonicity
    cdf_diff = jnp.diff(cdf_vals)
    ax2.plot(x_grid[:-1], cdf_diff, "g-", linewidth=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("dCDF/dx")
    ax2.set_title("CDF derivative (should be positive for monotonic)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("debug_post_widder_cdf.png", dpi=160)
    print("Saved debug_post_widder_cdf.png")
    plt.show()


if __name__ == "__main__":
    main()
