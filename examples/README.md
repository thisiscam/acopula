# Examples

Runnable, self-contained scripts. Each uses small synthetic / copula-scale
data — no external datasets required.

| Script | What it shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Build a two-level nested copula, evaluate the log-likelihood and its exact `jax.grad`. |
| [`02_fit_mle.py`](02_fit_mle.py) | Simulate from a known `d=10` Clayton copula, then recover the parameters by gradient-descent MLE (`optax`). |
| [`03_censored_survival.py`](03_censored_survival.py) | Weibull marginals with right-censoring — the correct mixed partial over only the observed dimensions. |
| [`04_sampling_and_visualize.py`](04_sampling_and_visualize.py) | Marshall-Olkin sampling + a tree diagram and pairwise scatter of the margins. |

## Running

The plotting example needs the `examples` extra (matplotlib / networkx):

```bash
# from the repo root
uv run python examples/01_quickstart.py
uv run python examples/02_fit_mle.py
uv run python examples/03_censored_survival.py
uv run --extra examples python examples/04_sampling_and_visualize.py [output_dir]
```

(or `pip install -e ".[examples]"` then `python examples/<script>.py`).
