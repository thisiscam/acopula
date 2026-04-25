#!/usr/bin/env python3
"""Minimal test of oryx.core.inverse to diagnose the issue."""

import jax
import jax.numpy as jnp
from oryx import core as oryx_core

jax.config.update("jax_enable_x64", True)


def test_simple_exp():
    """Test if oryx can invert a simple exponential function."""
    print("=== Test 1: Simple exponential ===")

    def f(x):
        return jnp.exp(x)

    try:
        f_inv = oryx_core.inverse(f)
        x_test = jnp.array([0.5, 0.7, 0.9])
        y_test = jax.vmap(f)(x_test)
        x_recovered = jax.vmap(f_inv)(y_test)
        print(f"x:          {x_test}")
        print(f"f(x):       {y_test}")
        print(f"f^{{-1}}(f(x)): {x_recovered}")
        print("✓ Simple exp inversion works")
    except Exception as e:
        print(f"✗ Simple exp inversion failed: {e}")
    print()


def test_frank_simplified():
    """Test a simplified version of Frank generator."""
    print("=== Test 2: Simplified Frank (without division by theta) ===")

    theta = 2.0

    def f(u):
        # Simplified: -ln(exp(-u) * (exp(-theta) - 1) + 1)
        return -jnp.log(jnp.exp(-u) * (jnp.exp(-theta) - 1.0) + 1.0)

    try:
        f_inv = oryx_core.inverse(f)
        u_test = jnp.array([0.1, 0.5, 0.9])
        t_test = jax.vmap(f)(u_test)
        u_recovered = jax.vmap(f_inv)(t_test)
        print(f"u:          {u_test}")
        print(f"f(u):       {t_test}")
        print(f"f^{{-1}}(f(u)): {u_recovered}")
        print(f"Error:      {jnp.abs(u_test - u_recovered)}")
        print("✓ Simplified Frank inversion works")
    except Exception as e:
        print(f"✗ Simplified Frank inversion failed: {e}")
    print()


def test_frank_full():
    """Test full Frank generator with division."""
    print("=== Test 3: Full Frank generator ===")

    theta = 2.0

    def f(u):
        return -jnp.log(jnp.exp(-u) * (jnp.exp(-theta) - 1.0) + 1.0) / theta

    try:
        f_inv = oryx_core.inverse(f)
        u_test = jnp.array([0.1, 0.5, 0.9])
        t_test = jax.vmap(f)(u_test)
        u_recovered = jax.vmap(f_inv)(t_test)
        print(f"u:          {u_test}")
        print(f"f(u):       {t_test}")
        print(f"f^{{-1}}(f(u)): {u_recovered}")
        print(f"Error:      {jnp.abs(u_test - u_recovered)}")
        print("✓ Full Frank inversion works")
    except Exception as e:
        print(f"✗ Full Frank inversion failed: {e}")
    print()


def main():
    print("Testing oryx.core.inverse with different functions\n")

    test_simple_exp()
    test_frank_simplified()
    test_frank_full()

    print("\n=== Summary ===")
    print(
        "If any test fails, we need to provide explicit generator_inv implementations."
    )


if __name__ == "__main__":
    main()
