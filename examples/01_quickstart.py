"""Quickstart: a two-level nested Archimedean copula in ~20 lines.

Builds a Frank-over-Frank nested copula with four uniform leaves
(two sectors of two), then evaluates the log-likelihood and its exact
gradient w.r.t. the parameters via ``jax.grad``.

Run:
    python examples/01_quickstart.py
"""

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from acopula import compile_model, copula, marginal


@copula
class Frank:
    theta: float

    def generator(self, t):
        return -jnp.log1p(jnp.expm1(-self.theta) * jnp.exp(-t)) / self.theta


def model(params, obs):
    """Root Frank over two Frank sectors, each with two uniform leaves."""
    outer = Frank(params[0])
    inner = Frank(params[1])
    return outer(
        inner(marginal(tfp.distributions.Uniform(0.0, 1.0), obs=obs[i, j])
              for j in range(2))
        for i in range(2)
    )


def main():
    # `template` only conveys the parameter pytree structure to the tracer;
    # its values are placeholders.
    cm = compile_model(model, template=jnp.array([1.0, 1.0]))

    obs = jnp.array([[0.3, 0.7],
                     [0.4, 0.8]])
    params = jnp.array([2.0, 5.0])  # theta_outer, theta_inner

    ll = cm.eval(obs, params)
    print(f"log-likelihood:        {ll:.6f}")

    # Exact gradient of the log-likelihood w.r.t. the parameters.
    grad = jax.grad(cm.eval, argnums=1)(obs, params)
    print(f"d(ll)/d(theta_outer):  {grad[0]:.6f}")
    print(f"d(ll)/d(theta_inner):  {grad[1]:.6f}")


if __name__ == "__main__":
    main()
