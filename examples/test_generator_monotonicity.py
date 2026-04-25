#!/usr/bin/env python3
"""Test monotonicity of generators: Clayton, Frank, and modified Frank.

All generators should be strictly decreasing functions on (0, 1].
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from oryx import core as oryx_core

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def clayton_generator(theta: float):
    """Return Clayton generator function."""

    def gen(u: jax.Array) -> jax.Array:
        return (1 + u) ** (-1.0 / theta)

    return gen


def clayton_generator_inv(theta: float):
    """Return Clayton generator inverse function using oryx."""
    gen = clayton_generator(theta)
    return oryx_core.inverse(gen)


def frank_generator(theta: float):
    """Return Frank generator function."""

    def gen(u: jax.Array) -> jax.Array:
        arg = jnp.expm1(-theta) * jnp.exp(-u)  # stays accurate for large θ, u
        return -jnp.log1p(arg) / theta

    return gen


def gumbel_generator(theta: float):
    """Return Gumbel generator function."""

    def gen(u: jax.Array) -> jax.Array:
        return jnp.exp(-jnp.power(u, 1.0 / theta))

    return gen


def gumbel_generator_inv(theta: float):
    """Return Gumbel generator inverse using oryx."""
    gen = gumbel_generator(theta)
    return oryx_core.inverse(gen)


def frank_generator_inv(theta: float):
    """Return Frank generator inverse using oryx."""
    gen = frank_generator(theta)
    return oryx_core.inverse(gen)


def build_modified_frank_generator(
    theta_outer: float, theta_inner: float, v_outer: float
):
    """Build the modified Frank generator: φ̂_frank(t) = exp(-V_clayton * φ_clayton^{-1}(φ_frank(t)))."""
    outer_gen_inv = clayton_generator_inv(theta_outer)
    inner_gen = gumbel_generator(theta_inner)

    def modified_gen(t: jax.Array) -> jax.Array:
        return outer_gen_inv(inner_gen(t))

    return modified_gen


def main():
    # Parameters
    theta_outer = 0.2
    theta_inner = jnp.exp(theta_outer)
    v_outer = 2

    print("=== Testing Generator Monotonicity ===")
    print(f"theta_outer = {theta_outer}")
    print(f"theta_inner = {theta_inner}")
    print(f"V_outer = {v_outer}")
    print()

    # Build generators
    clayton_gen = clayton_generator(theta_outer)
    frank_gen = frank_generator(theta_inner)
    modified_frank_gen = build_modified_frank_generator(
        theta_outer, theta_inner, v_outer
    )

    # Test on domain (0, 1] for generators (input u)
    u_grid = jnp.linspace(0.01, 1.0, 200)

    clayton_vals = jax.vmap(clayton_gen)(u_grid)
    frank_vals = jax.vmap(frank_gen)(u_grid)

    # For modified generator, we need to test on [0, ∞) domain (input t)
    t_grid = jnp.linspace(0.0, 100.0, 200)
    modified_vals = jax.vmap(modified_frank_gen)(t_grid)

    # Check monotonicity by computing finite-difference first derivatives for u-based gens
    clayton_diff = jnp.diff(clayton_vals)
    frank_diff = jnp.diff(frank_vals)
    modified_diff = jnp.diff(modified_vals)

    # Also check first..fourth analytical derivatives of modified Frank via jax.grad
    def nth_derivative_fun(fun, n: int):
        g = fun
        for _ in range(n):
            g = jax.grad(g)
        return g

    # Avoid t=0 for higher-order derivatives
    t_grid_der = jnp.linspace(1e-6, 100.0, 200)
    d1_fun = nth_derivative_fun(modified_frank_gen, 1)
    d2_fun = nth_derivative_fun(modified_frank_gen, 2)
    d3_fun = nth_derivative_fun(modified_frank_gen, 3)
    d4_fun = nth_derivative_fun(modified_frank_gen, 4)
    d1_vals = jax.vmap(d1_fun)(t_grid_der)
    d2_vals = jax.vmap(d2_fun)(t_grid_der)
    d3_vals = jax.vmap(d3_fun)(t_grid_der)
    d4_vals = jax.vmap(d4_fun)(t_grid_der)

    print("=== Monotonicity Check ===")
    print(
        f"Clayton: all negative? {jnp.all(clayton_diff < 0)} (min: {float(clayton_diff.min()):.6f}, max: {float(clayton_diff.max()):.6f})"
    )
    print(
        f"Frank: all negative? {jnp.all(frank_diff < 0)} (min: {float(frank_diff.min()):.6f}, max: {float(frank_diff.max()):.6f})"
    )
    print(
        f"Modified Frank: all negative? {jnp.all(modified_diff < 0)} (min: {float(modified_diff.min()):.6f}, max: {float(modified_diff.max()):.6f})"
    )
    print()

    # Check alternating sign pattern for modified Frank derivatives: (-1)^k d^k φ >= 0
    def check_alt_sign(vals, k):
        signed = ((-1) ** k) * vals
        return bool(jnp.all(signed >= 0)), float(signed.min()), float(signed.max())

    ok1, mn1, mx1 = check_alt_sign(d1_vals, 1)
    ok2, mn2, mx2 = check_alt_sign(d2_vals, 2)
    ok3, mn3, mx3 = check_alt_sign(d3_vals, 3)
    ok4, mn4, mx4 = check_alt_sign(d4_vals, 4)
    print("=== Modified Frank higher derivatives (alternating signs) ===")
    print(f"d1: (-1)^1 d1 >= 0 ? {ok1} (min:{mn1:.6f}, max:{mx1:.6f})")
    print(f"d2: (-1)^2 d2 >= 0 ? {ok2} (min:{mn2:.6f}, max:{mx2:.6f})")
    print(f"d3: (-1)^3 d3 >= 0 ? {ok3} (min:{mn3:.6f}, max:{mx3:.6f})")
    print(f"d4: (-1)^4 d4 >= 0 ? {ok4} (min:{mn4:.6f}, max:{mx4:.6f})")
    print()

    # Plot all three generators
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Generator functions
    # Clayton
    axes[0, 0].plot(u_grid, clayton_vals, "b-", linewidth=2)
    axes[0, 0].set_xlabel("u")
    axes[0, 0].set_ylabel("φ_clayton(u)")
    axes[0, 0].set_title(f"Clayton Generator (θ={theta_outer})")
    axes[0, 0].grid(True, alpha=0.3)

    # Frank
    axes[0, 1].plot(u_grid, frank_vals, "g-", linewidth=2)
    axes[0, 1].set_xlabel("u")
    axes[0, 1].set_ylabel("φ_frank(u)")
    axes[0, 1].set_title(f"Frank Generator (θ={theta_inner})")
    axes[0, 1].grid(True, alpha=0.3)

    # Modified Frank
    axes[0, 2].plot(t_grid, modified_vals, "r-", linewidth=2)
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel("φ̂_frank(t)")
    axes[0, 2].set_title(f"Modified Frank Generator (V_c={v_outer})")
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Derivatives (should all be negative)
    # Clayton derivative
    axes[1, 0].plot(u_grid[:-1], clayton_diff, "b-", linewidth=2)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1, 0].set_xlabel("u")
    axes[1, 0].set_ylabel("dφ/du")
    axes[1, 0].set_title("Clayton Derivative (should be < 0)")
    axes[1, 0].grid(True, alpha=0.3)

    # Frank derivative
    axes[1, 1].plot(u_grid[:-1], frank_diff, "g-", linewidth=2)
    axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1, 1].set_xlabel("u")
    axes[1, 1].set_ylabel("dφ/du")
    axes[1, 1].set_title("Frank Derivative (should be < 0)")
    axes[1, 1].grid(True, alpha=0.3)

    # Modified Frank derivative
    axes[1, 2].plot(t_grid[:-1], modified_diff, "r-", linewidth=2)
    axes[1, 2].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1, 2].set_xlabel("t")
    axes[1, 2].set_ylabel("dφ̂/dt")
    axes[1, 2].set_title("Modified Frank Derivative (should be < 0)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("generator_monotonicity.png", dpi=160)
    print("Saved generator_monotonicity.png")
    plt.show()

    # Plot modified Frank first..fourth derivatives
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2[0, 0].plot(t_grid_der, d1_vals, "r-", linewidth=2)
    axes2[0, 0].axhline(0, color="k", ls="--", lw=0.5)
    axes2[0, 0].set_title("Modified Frank d1(t)")
    axes2[0, 1].plot(t_grid_der, d2_vals, "r-", linewidth=2)
    axes2[0, 1].axhline(0, color="k", ls="--", lw=0.5)
    axes2[0, 1].set_title("Modified Frank d2(t)")
    axes2[1, 0].plot(t_grid_der, d3_vals, "r-", linewidth=2)
    axes2[1, 0].axhline(0, color="k", ls="--", lw=0.5)
    axes2[1, 0].set_title("Modified Frank d3(t)")
    axes2[1, 1].plot(t_grid_der, d4_vals, "r-", linewidth=2)
    axes2[1, 1].axhline(0, color="k", ls="--", lw=0.5)
    axes2[1, 1].set_title("Modified Frank d4(t)")
    for ax in axes2.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("t")
    plt.tight_layout()
    plt.savefig("modified_frank_derivatives.png", dpi=160)
    print("Saved modified_frank_derivatives.png")
    plt.show()


if __name__ == "__main__":
    main()
