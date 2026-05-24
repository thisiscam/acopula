"""Pure-functional model compilation API.

Replaces the stateful ``Model.set_params`` / ``Model.make_ll_fn`` pattern
with a single ``compile_model`` entry point that builds the structural
graph + flatten function + jit'd log-likelihood once, then returns an
immutable ``CompiledModel`` that callers reuse across many parameter
values.

The old API rebuilt the graph object on every ``set_params`` call, even
when only theta values changed.  Because JAX's jit cache is keyed on
function-object identity rather than on the *contents* of captured
graphs, every rebuild invalidated the cache and triggered a fresh XLA
compile — turning a 5-minute sweep into hours.  ``compile_model``
performs the structural discovery exactly once, so identical-shape
parameter sweeps reuse the same compiled program.

Typical usage::

    from acopula import compile_model, marginal, copula

    @copula
    class Clayton:
        theta: float
        def generator(self, t):
            return jnp.power(1.0 + t, -1.0 / self.theta)

    def model(p, u):
        return Clayton(p["theta"])([
            marginal(Uniform(), obs=u[i]) for i in range(5)
        ])

    cm = compile_model(model, template={"theta": 1.0}, method="bell")
    ll = cm.eval(jnp.array([0.3, 0.5, 0.7, 0.2, 0.9]),
                 {"theta": 0.7})

The ``template`` argument fixes the parameter pytree structure; its
*values* are never used past the initial harvest, so any sentinel
floats work.  Theta enters the jit'd ``ll_fn`` as a regular tensor
argument via ``flatten``, so different ``params`` calls hit cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random as jrandom
from oryx import core as oryx_core


def _harvest_graph_and_params(
    fn: Callable, template: Any
) -> Tuple[Any, dict]:
    """Run the user function once under oryx harvest to extract the
    structural graph and the canonical (sorted-by-start) param dict.
    """
    # Imported here to avoid a circular import at module load time.
    from .core import (
        Node,
        RegistrationContext,
        UPlaceholder,
        _collect_leaves_in_order,
        _infer_obs_shape_and_validate_unique,
    )

    u_placeholder = UPlaceholder()

    def wrapped_fn(p):
        return fn(p, u_placeholder)

    with RegistrationContext():
        graph, harvested = oryx_core.harvest(wrapped_fn, tag="params")(
            {}, template
        )

    if not isinstance(graph, Node):
        raise TypeError(
            "Model function must return a Copula call producing a Node"
        )

    leaves_in_order = _collect_leaves_in_order(graph)
    obs_shape = _infer_obs_shape_and_validate_unique(leaves_in_order)
    if obs_shape is None:
        raise ValueError(
            "Model must specify marginal(...) leaves; flat-U is no longer supported."
        )

    return graph, harvested, obs_shape


def _make_flatten(model_fn: Callable, template: Any) -> Tuple[Callable, Tuple[int, ...]]:
    """Return a function that flattens a params pytree to the canonical
    flat array order (sorted by sow-name = start index), and the
    expected flat shape.

    The flatten function re-runs oryx harvest on each call.  That is
    cheap relative to the JAX compute that follows, and it lets users
    pass arbitrary pytree shapes (dicts, lists, scalars, jax arrays)
    without us caching a particular ravel function.
    """
    # Imported lazily to avoid the circular import.
    from .core import RegistrationContext, UPlaceholder

    u_placeholder = UPlaceholder()

    def wrapped_fn(p):
        return model_fn(p, u_placeholder)

    def flatten(params):
        with RegistrationContext():
            _graph, harvested = oryx_core.harvest(wrapped_fn, tag="params")(
                {}, params
            )
        sorted_items = sorted(harvested.items(), key=lambda kv: int(kv[0]))
        flat_parts = [jnp.asarray(v).reshape((-1,)) for _, v in sorted_items]
        if not flat_parts:
            return jnp.array([])
        return jnp.concatenate(flat_parts)

    # Compute the expected shape once using the template.
    flat_template = flatten(template)
    return flatten, flat_template.shape


def _build_ll_fn(
    graph,
    *,
    method: str,
    survival: bool,
    with_censored_mask: bool,
    ils_method: str,
    ils_params: Optional[dict],
    log_jet: bool = False,
) -> Callable:
    """Construct the raw (un-jit'd) log-likelihood evaluator that
    closes over the structural graph.

    Returned signature:
        with_censored_mask=False:  ll_fn(obs, flat_params) -> log_val
        with_censored_mask=True:   ll_fn(obs, flat_params, censored_mask) -> log_val
    """
    from .core import _depth, _log_likelihood_single_layer, _log_likelihood_nested

    depth = _depth(graph)

    if with_censored_mask:
        if method not in ("auto", "bell"):
            raise ValueError(
                f"with_censored_mask=True requires method='bell' or 'auto' "
                f"(got method={method!r})."
            )
        from .bell import _log_likelihood_bell

        def ll_fn(obs, flat_params, censored_mask):
            _, log_val, _ = _log_likelihood_bell(
                graph, obs, flat_params,
                censored_mask=censored_mask, survival=survival,
                log_jet=log_jet,
            )
            return log_val

        return ll_fn

    if method == "bell" or (method == "auto" and depth > 1):
        from .bell import _log_likelihood_bell

        def ll_fn(obs, flat_params):
            _, log_val, _ = _log_likelihood_bell(
                graph, obs, flat_params, survival=survival, log_jet=log_jet,
            )
            return log_val
    elif depth <= 1 and method != "integral":
        def ll_fn(obs, flat_params):
            _, log_val = _log_likelihood_single_layer(
                graph, obs, flat_params, survival=survival,
            )
            return log_val
    else:
        def ll_fn(obs, flat_params):
            _, log_val = _log_likelihood_nested(
                graph, obs, flat_params,
                ils_method=ils_method, ils_params=ils_params,
                survival=survival,
            )
            return log_val

    return ll_fn


@dataclass(frozen=True)
class CompiledModel:
    """Immutable compiled representation of a nested-copula model.

    Built by :func:`compile_model`.  Holds the structural graph, the
    flattening function for parameter pytrees, and a jit'd
    log-likelihood evaluator.  Reuse a single ``CompiledModel`` across
    parameter sweeps to avoid repeated XLA compilation; only the input
    *values* change, so the cache hits.

    Attributes:
        graph: The frozen :class:`Node` tree describing the copula
            composition.  Read-only handle for downstream tooling
            (visualisation, structural diagnostics).
        param_shape: Expected flat parameter shape, e.g. ``(2,)`` for a
            two-parameter model.  Useful for sanity checks before
            calling ``ll_fn`` with raw arrays.
        with_censored_mask: True iff this compilation supports a
            per-observation censoring mask as a third argument to
            ``ll_fn``.
        survival: True iff this compilation interprets observations as
            survival data — the copula models the joint survival
            ``S(T) = 1 - F(T)``. Stored here so ``sample()`` can apply
            the matching ``F^{-1}(1 - U)`` inversion at the leaves and
            return data drawn from the same joint distribution that
            the likelihood scores.
    """

    graph: Any
    param_shape: Tuple[int, ...]
    obs_shape: Tuple[int, ...]
    with_censored_mask: bool
    survival: bool
    _model_fn: Callable
    _flatten_fn: Callable
    _ll_fn_jit: Callable

    # ------------------------- core API -------------------------

    def flatten(self, params: Any) -> jax.Array:
        """Convert a parameter pytree into the canonical flat array."""
        return self._flatten_fn(params)

    def ll_fn(self, *args, **kwargs):
        """Direct handle on the jit'd log-likelihood.

        Signature matches the compilation:
            with_censored_mask=False: ``ll_fn(obs, flat_params)``
            with_censored_mask=True:  ``ll_fn(obs, flat_params, censored_mask)``

        Use this when you want raw control for ``jax.value_and_grad``,
        ``jax.hessian``, ``jax.vmap``, etc.  Prefer :meth:`eval` for
        one-shot calls with a parameter pytree.
        """
        return self._ll_fn_jit(*args, **kwargs)

    def eval(self, obs: jax.Array, params: Any, *, censored_mask=None) -> jax.Array:
        """Evaluate the log-likelihood at ``params`` for a single
        observation (or a leading-batched array).

        Convenience wrapper that flattens ``params`` and dispatches to
        the underlying jit'd ``ll_fn``.
        """
        flat = self._flatten_fn(params)
        if self.with_censored_mask:
            if censored_mask is None:
                raise ValueError(
                    "compile_model(..., with_censored_mask=True) requires "
                    "passing censored_mask to eval()."
                )
            return self._ll_fn_jit(obs, flat, censored_mask)
        return self._ll_fn_jit(obs, flat)

    # ------------------------- sampling -------------------------

    def sample(
        self,
        key: jax.Array,
        n: int,
        params: Any,
        *,
        post_widder_k: int = 8,
        max_cdf_x: float = 1e6,
        method: str = "marshall_olkin",
    ) -> jax.Array:
        """Generate ``n`` samples from the copula at ``params``.

        Respects the ``survival`` flag set at compile time — when
        ``survival=True`` the leaf inversion uses ``F^{-1}(1 - U)`` so
        the samples are drawn from the same joint distribution that
        the likelihood scores.
        """
        from .core import _sample_once, _sample_once_rosenblatt

        flat = self._flatten_fn(params)

        if method == "marshall_olkin":
            def sample_one(sample_key):
                return _sample_once(
                    self.graph, sample_key, flat, post_widder_k, max_cdf_x,
                    survival=self.survival,
                )
        elif method == "rosenblatt":
            def sample_one(sample_key):
                return _sample_once_rosenblatt(
                    self.graph, sample_key, flat,
                    survival=self.survival,
                )
        else:
            raise ValueError(
                f"Unknown sampling method '{method}'. "
                f"Choose 'marshall_olkin' or 'rosenblatt'."
            )

        sample_keys = jrandom.split(key, n)
        return jax.vmap(sample_one)(sample_keys)

    # ------------------------- introspection -------------------------

    def as_networkx(self, include_leaves: bool = True, annotations=None):
        from .visualize import to_networkx
        return to_networkx(self.graph, include_leaves=include_leaves,
                           annotations=annotations)

    def visualize(
        self,
        *,
        include_leaves: bool = True,
        layout: str = "hierarchical",
        with_labels: bool = True,
        graphviz_prog: str = "dot",
        annotations=None,
        ax=None,
    ):
        from .visualize import draw_networkx
        G = self.as_networkx(include_leaves=include_leaves, annotations=annotations)
        return draw_networkx(
            G,
            layout=layout,
            with_labels=with_labels,
            graphviz_prog=graphviz_prog,
            ax=ax,
        )


def compile_model(
    model_fn: Callable,
    *,
    template: Any,
    method: str = "auto",
    survival: bool = False,
    with_censored_mask: bool = False,
    ils_method: str = "cohen",
    ils_params: Optional[dict] = None,
    log_jet: bool = False,
) -> CompiledModel:
    """Compile a copula composition function into a reusable evaluator.

    The compilation runs the user function once under oryx harvest to
    discover the structural graph (with sentinel ``template`` values),
    builds a flattener that maps parameter pytrees to the canonical
    flat array, constructs the appropriate log-likelihood evaluator
    (``bell`` / ``single_layer`` / ``nested``), and JIT-compiles it.

    Reuse the returned :class:`CompiledModel` across many parameter
    sweeps — JAX caches the compiled program by function identity, so
    repeated calls with different *values* hit cache.  The historical
    pitfall (``set_params`` rebuilds the graph object every call,
    invalidating cache) is avoided by construction.

    Args:
        model_fn: A function ``(params, u) -> Node`` returning a Copula
            call that produces the structural graph.
        template: A parameter pytree whose structure matches what
            ``model_fn`` expects.  Values are used only to drive oryx's
            tracer; pass any sensible defaults (e.g.
            ``{"theta": 1.0}``).
        method: Density evaluation backend.  ``"bell"`` uses the
            polynomial-powering scan (recommended for nested >= 2
            levels), ``"single_layer"`` is the closed-form unnested
            formula, ``"nested"`` falls back to numerical integration
            via the implicit-LST solver.  ``"auto"`` picks ``"bell"``
            for depth > 1 else ``"single_layer"``.
        survival: Treat marginals as survival functions
            ``S(t) = 1 - F(t)`` rather than CDFs.  Required for the
            HACSurv-style joint survival likelihood.
        with_censored_mask: If True, the returned ``ll_fn`` takes a
            third ``censored_mask`` argument so callers can ``vmap``
            per-observation masks.  Forces ``method="bell"`` since the
            other backends do not support dynamic censoring.
        ils_method: Implicit-LST solver for ``method="nested"``.
        ils_params: Extra params for the implicit solver.

    Returns:
        A :class:`CompiledModel` holding the frozen graph, flattener,
        and jit'd log-likelihood.
    """
    graph, _harvested, obs_shape = _harvest_graph_and_params(model_fn, template)
    flatten, param_shape = _make_flatten(model_fn, template)

    raw_ll = _build_ll_fn(
        graph,
        method=method,
        survival=survival,
        with_censored_mask=with_censored_mask,
        ils_method=ils_method,
        ils_params=ils_params,
        log_jet=log_jet,
    )
    ll_fn_jit = jax.jit(raw_ll)

    return CompiledModel(
        graph=graph,
        param_shape=param_shape,
        obs_shape=obs_shape,
        with_censored_mask=with_censored_mask,
        survival=survival,
        _model_fn=model_fn,
        _flatten_fn=flatten,
        _ll_fn_jit=ll_fn_jit,
    )
