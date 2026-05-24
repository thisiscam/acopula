"""Censored survival likelihood with Weibull marginals.

Models a bivariate survival time with a Frank copula and Weibull marginals,
where the second event time is right-censored. A censored leaf still enters
the copula argument (through its survival function) but is not differentiated
in the density — ``acopula`` assembles the correct mixed partial over only the
observed dimensions. Pass ``survival=True`` so leaf inversion uses
``F^{-1}(1 - u)`` (the survival convention).

Run:
    python examples/03_censored_survival.py
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


def make_model(censored_second: bool):
    """Bivariate Frank copula over two Weibull event times.

    `acopula` enables jax_enable_x64 at import, so use float64 distribution
    parameters to match.
    """
    weib = tfp.distributions.Weibull(
        concentration=jnp.float64(1.5), scale=jnp.float64(1.0))

    def model(theta, t):
        c = Frank(theta[0])
        return c([
            marginal(weib, obs=t[0]),
            marginal(weib, obs=t[1], censored=censored_second),
        ])

    return model


def main():
    theta = jnp.array([4.0])
    times = jnp.array([0.6, 1.2])  # second time is the (possibly) censored one

    # Same data, scored two ways: both observed vs. second right-censored.
    cm_obs = compile_model(make_model(censored_second=False),
                           template=theta, survival=True)
    cm_cens = compile_model(make_model(censored_second=True),
                            template=theta, survival=True)

    ll_obs = cm_obs.eval(times, theta)
    ll_cens = cm_cens.eval(times, theta)

    print(f"log-likelihood, both observed:        {ll_obs:.6f}")
    print(f"log-likelihood, 2nd right-censored:   {ll_cens:.6f}")

    # Gradient w.r.t. the dependence parameter still flows through the
    # censored contribution.
    g = jax.grad(cm_cens.eval, argnums=1)(times, theta)
    print(f"d(ll_censored)/d(theta):              {g[0]:.6f}")


if __name__ == "__main__":
    main()
