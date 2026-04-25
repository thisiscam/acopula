from __future__ import annotations

import jax.numpy as jnp
from oryx import core


def test_frank_f(u):
    theta = 2.0
    return -jnp.log(1.0 - (jnp.exp(-theta * u) - 1.0) / (jnp.exp(-theta) - 1.0))


def main():
    # Test values in the domain of φ: [0, ∞)
    # Note φ(1)=0 and φ is decreasing; larger t -> smaller u
    t_vals = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])

    # Test: Try inverting the standalone function
    print("=== Test: Invert standalone Frank function ===")

    try:
        test_frank_inv = core.inverse(test_frank_f)
        u_from_inv = test_frank_inv(t_vals)
        print("SUCCESS!")
        print("Inverted function result:", u_from_inv)

        # Verify by composing back
        t_back = test_frank_f(u_from_inv)
        print("φ(φ^{-1}(t)):", t_back)
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
