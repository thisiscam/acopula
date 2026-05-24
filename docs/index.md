# acopula

**Archimedean copula inference via Taylor-mode automatic differentiation in JAX.**

`acopula` fits **nested Archimedean copulas** end-to-end with `jax.grad` —
including high-dimensional models, per-dimension censoring, and Bell-polynomial
densities derived automatically from the user-supplied generator. The
high-order derivatives that previously bottlenecked nested-Archimedean
likelihoods (limiting prior tools to roughly `d=10`) are computed in a single
forward pass via Taylor-mode AD, scaling polynomially in the dimension.

## Install

```bash
pip install acopula
```

`acopula` pins `jax>=0.8,<0.9` (it relies on JAX-internal APIs in the
[jet-array](https://github.com/thisiscam/jet-array) backend) and uses
[`oryx`](https://github.com/jax-ml/oryx) for symbolic generator inversion.

## Features

- **Nested Archimedean copulas** of arbitrary depth and arity.
- **Per-dimension censoring** — each leaf can be independently right-censored
  per observation; one XLA program handles all masks.
- **Density via Bell polynomials**, computed from a Taylor expansion of the
  generator rather than nested first-order AD.
- **Symbolic generator inversion** via `oryx`, with bisection + IFT fallback.
- **Sampling** via the Marshall-Olkin algorithm and the Rosenblatt transform.
- **Validity diagnostic** — per-edge `d_c`-monotonicity check for
  cross-family nesting.

## Where to go next

- [Quickstart](quickstart.md) — a working model in ~20 lines.
- [Examples](examples.md) — MLE fitting, censored survival, sampling & plots.
- [API reference](api.md) — the full public surface.

## Citation

```bibtex
@misc{yang2026copulaad,
  title={Archimedean Copula Inference via Taylor-Mode AD},
  author={Yang, Cambridge and Li, Dongdong},
  year={2026},
  note={arXiv preprint},
}
```
