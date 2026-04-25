#!/usr/bin/env python3
"""Test Frank generator inversion using oryx.core.inverse.

Verifies that φ_frank^{-1}(φ_frank(u)) = u for the Frank copula.
"""

import jax
import jax.numpy as jnp
from oryx import core as oryx_core

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def frank_generator(theta: float):
    """Return Frank generator function."""

    def gen(u: jax.Array) -> jax.Array:
        return -jnp.log(jnp.exp(-u) * (jnp.exp(-theta) - 1.0) + 1) / theta

    return gen


def frank_generator_inv(theta: float):
    """Return Frank generator inverse using oryx."""
    gen = frank_generator(theta)
    return oryx_core.inverse(gen)


def main():
    theta = 2.0

    print("=== Testing Frank Generator Inversion ===")
    print(f"theta = {theta}")
    print()

    # Build generator and its inverse
    frank_gen = frank_generator(theta)
    frank_gen_inv = frank_generator_inv(theta)

    # Test on domain (0, 1]
    print("=== Test 1: Forward then inverse (u -> t -> u) ===")
    u_test = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    t_test = jax.vmap(frank_gen)(u_test)
    u_recovered = jax.vmap(frank_gen_inv)(t_test)

    print(f"u:          {u_test}")
    print(f"φ(u):       {t_test}")
    print(f"φ^{{-1}}(φ(u)): {u_recovered}")
    print(f"Error:      {jnp.abs(u_test - u_recovered)}")
    print(f"Max error:  {float(jnp.max(jnp.abs(u_test - u_recovered))):.2e}")
    print()

    # Test on range [0, ∞) for inverse
    print("=== Test 2: Inverse then forward (t -> u -> t) ===")
    t_test2 = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
    u_test2 = jax.vmap(frank_gen_inv)(t_test2)
    t_recovered = jax.vmap(frank_gen)(u_test2)

    print(f"t:          {t_test2}")
    print(f"φ^{{-1}}(t):    {u_test2}")
    print(f"φ(φ^{{-1}}(t)): {t_recovered}")
    print(f"Error:      {jnp.abs(t_test2 - t_recovered)}")
    print(f"Max error:  {float(jnp.max(jnp.abs(t_test2 - t_recovered))):.2e}")
    print()

    # Plot generator and inverse
    print("=== Plotting ===")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Generator φ(u)
    u_plot = jnp.linspace(0.01, 1.0, 200)
    phi_plot = jax.vmap(frank_gen)(u_plot)

    axes[0].plot(u_plot, phi_plot, "b-", linewidth=2, label="φ_frank(u)")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("φ(u)")
    axes[0].set_title(f"Frank Generator (θ={theta})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Inverse generator φ^{-1}(t)
    t_plot = jnp.linspace(0.0, 10.0, 200)
    phi_inv_plot = jax.vmap(frank_gen_inv)(t_plot)

    axes[1].plot(t_plot, phi_inv_plot, "r-", linewidth=2, label="φ_frank^{-1}(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("φ^{-1}(t)")
    axes[1].set_title(f"Frank Generator Inverse (θ={theta})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("frank_generator_inverse.png", dpi=160)
    print("Saved frank_generator_inverse.png")
    plt.show()

    # Check if inversion succeeded
    if jnp.max(jnp.abs(u_test - u_recovered)) < 1e-6:
        print("\n✓ Frank generator inversion PASSED")
    else:
        print("\n✗ Frank generator inversion FAILED")


if __name__ == "__main__":
    main()
