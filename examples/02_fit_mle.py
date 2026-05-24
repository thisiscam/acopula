"""Maximum-likelihood fitting of a nested copula by gradient descent.

Simulates data from a known two-level Clayton copula (d=10, two sectors of
five), then recovers the parameters by maximising the Bell-polynomial
log-likelihood with Adam. The whole likelihood is differentiable, so the fit
is plain ``jax.value_and_grad`` + ``optax`` — no bespoke EM or quadrature.

Run:
    python examples/02_fit_mle.py
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from acopula import compile_model, copula, marginal


class Uniform:
    """Trivial marginal: data already lives on the copula (unit-cube) scale."""
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
        return jnp.power(1.0 + t, -1.0 / self.theta)

    def generator_inv(self, u):
        # Closed form psi^{-1}(u) = u^{-theta} - 1, smooth on (0, 1);
        # avoids the oryx-derived inverse that can NaN under higher-order AD.
        return jnp.expm1(-self.theta * jnp.log(u))


def make_model(d=10, group_size=5):
    """Root Clayton over ``d/group_size`` Clayton sectors of uniform leaves."""
    n_sectors = d // group_size

    def model(params, u):
        root = Clayton(params["theta_root"])
        sector = Clayton(params["theta_sector"])
        sectors = []
        for s in range(n_sectors):
            leaves = [marginal(Uniform(), obs=u[s * group_size + j])
                      for j in range(group_size)]
            sectors.append(sector(leaves))
        return root(sectors)

    return model


def main():
    d, group_size = 10, 5
    true_params = {"theta_root": 2.0, "theta_sector": 5.0}
    model = make_model(d, group_size)

    # --- simulate 1000 observations from the true copula ---
    cm = compile_model(model, template=true_params, method="bell")
    key = jrandom.PRNGKey(0)
    data = cm.sample(key, 1000, true_params, method="rosenblatt")
    data = jnp.clip(data, 1e-3, 1 - 1e-3)
    print(f"simulated data: shape={data.shape}")

    # --- fit by maximising the mean log-likelihood (Adam on log-theta) ---
    # Optimise in log-space so theta stays positive.
    log_theta = {k: jnp.log(jnp.array(1.0)) for k in true_params}  # init theta=1

    def neg_mean_ll(log_theta):
        params = {k: jnp.exp(v) for k, v in log_theta.items()}
        flat = cm.flatten(params)
        lls = jax.vmap(lambda x: cm.ll_fn(x, flat))(data)
        return -jnp.mean(lls)

    opt = optax.adam(0.05)
    state = opt.init(log_theta)
    loss_and_grad = jax.jit(jax.value_and_grad(neg_mean_ll))

    for step in range(300):
        loss, grads = loss_and_grad(log_theta)
        updates, state = opt.update(grads, state)
        log_theta = optax.apply_updates(log_theta, updates)
        if step % 50 == 0:
            print(f"  step {step:3d}  NLL={float(loss):.4f}")

    fitted = {k: float(jnp.exp(v)) for k, v in log_theta.items()}
    print("\nparameter         true     fitted")
    for k in true_params:
        print(f"  {k:<14} {true_params[k]:>5.2f}   {fitted[k]:>8.3f}")


if __name__ == "__main__":
    main()
