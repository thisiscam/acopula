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

## Install

```bash
pip install acopula
```

`acopula` currently pins `jax>=0.8,<0.9` because it relies on JAX-internal
APIs in the [jet-array](https://github.com/thisiscam/jet-array) backend.

`acopula` uses [`oryx`](https://github.com/jax-ml/oryx) for symbolic generator
inversion. At import time we register a missing ILDJ rule for the
`lax.copy_p` primitive; this no-ops once the rule is upstreamed.

## Quickstart

```python
import jax
import jax.numpy as jnp
from acopula import copula, defmodel, marginal

@copula
class Clayton:
    theta: float
    def generator(self, u):
        return (1 + u)**(-1.0 / self.theta)

@copula
class AMH:
    theta: float
    def generator(self, u):
        return (1 - self.theta) / (jnp.exp(u) - self.theta)

@defmodel
def nested_amh_clayton(params, obs):
    outer = AMH(params[0])
    inner = Clayton(params[1])
    return outer(
        inner(marginal(Weibull(),
                       obs=obs[i, j],
                       censored=((i, j) == (1, 3)))
              for j in range(5))
        for i in range(4))

model = nested_amh_clayton
model.set_params(jnp.array([0.5, 2.0]))
ll = model.log_likelihood(observations)
gradient = jax.grad(ll)(model.params)
```

The `@copula` decorator registers parameters and derives the generator inverse
symbolically (falling back to bisection with implicit-function-theorem
gradients when symbolic inversion fails). The `@defmodel` decorator traces the
function into a copula tree, flattens parameters into a single array, and
exposes `log_likelihood`, `sample`, and `cdf` — all `jit`/`grad`-compatible.
The `marginal` primitive pairs each leaf with a distribution and an optional
per-observation censoring flag.

## Features

- **Nested Archimedean copulas** of arbitrary depth and arity.
- **Per-dimension censoring** — each leaf can be independently right-censored
  per observation; one XLA program handles all masks.
- **Density via Bell polynomials**, computed from a Taylor expansion of the
  generator rather than nested first-order AD.
- **Symbolic generator inversion** via `oryx`, with bisection + IFT fallback.
- **Conditional sampling** of nested Archimedean models, made practical by
  the same higher-order derivatives.
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
