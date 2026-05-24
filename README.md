# acopula

Archimedean copula inference via Taylor-mode automatic differentiation in JAX.

`acopula` fits **nested Archimedean copulas** end-to-end with `jax.grad` —
including high-dimensional models, per-dimension censoring, and Bell-polynomial
densities derived automatically from the user-supplied generator. The
high-order derivatives that previously bottlenecked nested-Archimedean
likelihoods (limiting prior tools to roughly `d=10`) are computed in a single
forward pass via Taylor-mode AD, scaling polynomially in the dimension.

For background, see the paper *Archimedean Copula Inference via Taylor-Mode
AD* (arXiv:TBD).

**Documentation:** <https://thisiscam.github.io/acopula/> ·
**Examples:** [rendered notebooks](https://thisiscam.github.io/acopula/examples/01_quickstart/) ([source](docs/examples/))

## Install

```bash
pip install git+https://github.com/thisiscam/acopula
```

acopula depends on a patched [`oryx`](https://github.com/jax-ml/oryx) build
(pulled from git as part of the install) because stock `oryx` is incompatible
with jax 0.8. For development, clone and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/thisiscam/acopula
cd acopula
uv sync --extra examples
uv run jupyter lab docs/examples/   # the example notebooks
```

A PyPI `pip install acopula` is **not available yet**: the git-pinned `oryx`
dependency can't be uploaded to PyPI, so this is gated on an upstream `oryx`
release compatible with jax 0.8.

`acopula` pins `jax>=0.8,<0.9` because it relies on JAX-internal
APIs in the [jet-array](https://github.com/thisiscam/jet-array) backend.

`acopula` uses [`oryx`](https://github.com/jax-ml/oryx) for symbolic generator
inversion. At import time we register a missing ILDJ rule for the
`lax.copy_p` primitive; this no-ops once the rule is upstreamed.

## Quickstart

A two-level nested Archimedean copula with Frank inner and outer, four
Uniform leaves, and gradient-based MLE in twenty lines:

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

### With Weibull marginals and right-censoring

For survival data, swap the leaf distribution and mark censored observations.
Use `float64` for the distribution parameters since `acopula` enables
`jax_enable_x64` at import time.

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
```

`@copula` registers the parameters declared as type-annotated class
attributes and derives the inverse generator symbolically via `oryx`.
`compile_model` traces the model function into a copula tree, flattens
parameters into a single array, and returns a `CompiledModel` exposing
a jit'd log-likelihood that is `jax.grad`/`jax.vmap`-compatible.
`marginal` pairs each leaf with a distribution and an optional
per-observation censoring flag.

## Features

- **Nested Archimedean copulas** of arbitrary depth and arity.
- **Per-dimension censoring** — each leaf can be independently right-censored
  per observation; one XLA program handles all masks.
- **Density via Bell polynomials**, computed from a Taylor expansion of the
  generator rather than nested first-order AD.
- **Symbolic generator inversion** via `oryx`, with bisection + IFT fallback.
- **Validity diagnostic** — per-edge `d_c`-monotonicity check for
  cross-family nesting.

## Citation

```bibtex
@misc{yang2026copulaad,
  title={Archimedean Copula Inference via Taylor-Mode AD},
  author={Yang, Cambridge and Li, Dongdong},
  year={2026},
  note={arXiv preprint},
}
```

## License

Apache-2.0.
