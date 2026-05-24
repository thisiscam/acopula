# Quickstart

A two-level nested Archimedean copula with Frank inner and outer, four
uniform leaves, and gradient-based evaluation in twenty lines:

```python
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from acopula import compile_model, copula, marginal

@copula
class Frank:
    theta: float
    def generator(self, u):
        return -jnp.log1p(jnp.expm1(-self.theta) * jnp.exp(-u)) / self.theta

def model(params, obs):
    outer = Frank(params[0])
    inner = Frank(params[1])
    return outer(
        inner(marginal(tfp.distributions.Uniform(0.0, 1.0), obs=obs[i, j])
              for j in range(2))
        for i in range(2))

obs = jnp.array([[0.3, 0.7],
                 [0.4, 0.8]])

cm = compile_model(model, template=jnp.array([1.0, 1.0]))
params = jnp.array([2.0, 5.0])
print(cm.eval(obs, params))                     # scalar log-likelihood

grad = jax.grad(cm.eval, argnums=1)(obs, params)
print(grad)                                     # ∂ll/∂params, exact
```

## How the pieces fit

- `@copula` registers the parameters declared as type-annotated class
  attributes and derives the inverse generator symbolically via `oryx`.
- `compile_model` traces the model function into a copula tree, flattens
  parameters into a single array, and returns a [`CompiledModel`](api.md)
  exposing a jit'd log-likelihood that is `jax.grad` / `jax.vmap`-compatible.
- `marginal` pairs each leaf with a distribution and an optional
  per-observation censoring flag.

## With Weibull marginals and right-censoring

For survival data, swap the leaf distribution and mark censored observations.
Use `float64` distribution parameters since `acopula` enables
`jax_enable_x64` at import. Pass `survival=True` so leaf inversion uses
`F^{-1}(1 - u)`.

```python
def survival_model(params, obs):
    outer = Frank(params[0])
    inner = Frank(params[1])
    weib = tfp.distributions.Weibull(
        concentration=jnp.float64(1.5), scale=jnp.float64(1.0))
    return outer(
        inner(marginal(weib, obs=obs[i, j], censored=((i, j) == (1, 1)))
              for j in range(2))
        for i in range(2))

cm = compile_model(survival_model, template=jnp.array([1.0, 1.0]), survival=True)
```

See [`examples/03_censored_survival.py`](examples.md) for a runnable version.
