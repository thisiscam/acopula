from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import threading

import jax
import jax.numpy as jnp
import jax.scipy.special
from jax import lax
from jax import random as jrandom
from jax.flatten_util import ravel_pytree
import optimistix as optx
from oryx import core as oryx_core
from oryx.core import sow
import quadax

import jet_array

# use 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# =============================
# Core abstractions and tracing
# =============================

# Context for parameter registration/sowing
_REGISTRY = threading.local()


class RegistrationContext:
    def __enter__(self):
        _REGISTRY.active = True
        _REGISTRY.counter = 0
        _REGISTRY.flat_offset = 0
        return self

    def __exit__(self, *args):
        _REGISTRY.active = False
        _REGISTRY.counter = 0
        _REGISTRY.flat_offset = 0


class Copula:
    """Base class for Archimedean copulas in φ-notation.

    Subclasses must implement generator(u) which returns φ(u), where
    u ∈ (0, 1], φ: (0,1] → [0, ∞), strictly decreasing, convex, φ(1)=0.
    """

    def generator(self, u: jax.Array) -> jax.Array:  # pragma: no cover - abstract
        raise NotImplementedError

    # Inverse of generator: φ^{-1}(t)
    def generator_inv(self, t: jax.Array) -> jax.Array:
        return _invert_generator(self.generator, t)

    def log_generator_kth_derivative(self, t: jax.Array, k: int) -> jax.Array:
        """Return log |ψ^{(k)}(t)|, the log absolute k-th derivative.

        Override this with a closed-form expression to bypass jet for
        specific families.  For example, Clayton can use::

            lgamma(k + 1/theta) - lgamma(1/theta) - (k + 1/theta) * log1p(t)

        The default returns None, which tells the framework to use
        Taylor-mode AD via jet.
        """
        return None  # sentinel: use jet

    def generator_taylor_coefficients(
        self, t: jax.Array, k_max: int
    ) -> jax.Array:
        """Return Taylor coefficients ``[ψ(t), ψ'(t)/1!, …, ψ^{(k_max)}(t)/k_max!]``.

        Override this when the closed-form ψ has a numerically stable
        series representation that avoids the cancellations Taylor-mode
        AD on the closed form would produce at high derivative order.
        For example, AMH's ψ is the Laplace transform of Geometric(1−θ),
        so all Taylor coefficients are sums of same-sign terms (no
        cancellation):

            ψ^{(k)}(t) = (-1)^k Σ_{x=1}^∞ x^k (1−θ) θ^{x−1} e^{−tx}

        The framework consults this hook everywhere it would otherwise
        call ``jet_array.jet`` on ``self.generator`` to get a Taylor
        expansion: the root density (``bell._root_assembly``) and the
        nesting composition (``compose.compute_composition_taylor``).
        Returning ``None`` (the default) tells those call sites to
        fall back to the jet path, which is correct but loses
        precision when ψ's derivatives have large alternating-sign
        cancellations (e.g. AMH past d≈30 in float64).

        Args:
            t: scalar input point.
            k_max: highest derivative order to compute (inclusive); the
                returned array has shape ``(k_max + 1,)``.

        Returns:
            JAX array of shape ``(k_max + 1,)`` containing the Taylor
            coefficients, or ``None`` to fall back to jet.
        """
        return None  # sentinel: use jet

    # For tracing: combine children (nodes/leaves) into a Node
    def __call__(self, children: Iterable[Union["Node", "Leaf", jax.Array]]) -> "Node":
        child_nodes: List[Union[Node, Leaf]] = []
        for ch in children:
            if isinstance(ch, (Node, Leaf)):
                child_nodes.append(ch)
            else:
                # Treat raw scalars as leaves without indices (single-layer convenience)
                child_nodes.append(Leaf(index=None, subindex=None))
        return Node(copula=self, children=child_nodes)


def copula(cls):
    """Decorator to create a copula class with dataclass-like syntax.

    Usage:
        @copula
        class Clayton:
            theta: float

            def generator(self, u: jax.Array) -> jax.Array:
                return (1.0 + u) ** (-1.0 / self.theta)

        # Can instantiate with positional or keyword arguments
        c1 = Clayton(2.0)
        c2 = Clayton(theta=2.0)

    This automatically:
    - Makes the class inherit from Copula
    - Generates an __init__ method from type annotations
    - Preserves all user-defined methods
    - Handles parameter registration
    """
    # Get type annotations in definition order
    annotations = getattr(cls, "__annotations__", {})
    param_names = list(annotations.keys())

    # Create __init__ method that accepts both positional and keyword args
    def __init__(self, *args, **kwargs):
        # Handle positional arguments
        if len(args) > len(param_names):
            raise TypeError(
                f"{cls.__name__}() takes {len(param_names)} positional "
                f"argument(s) but {len(args)} were given"
            )

        # Assign positional arguments
        for i, (param_name, value) in enumerate(zip(param_names, args)):
            if param_name in kwargs:
                raise TypeError(
                    f"{cls.__name__}() got multiple values for argument '{param_name}'"
                )
            setattr(self, param_name, value)

        # Assign keyword arguments
        for param_name in param_names[len(args) :]:
            if param_name in kwargs:
                setattr(self, param_name, kwargs[param_name])
            else:
                raise TypeError(
                    f"{cls.__name__}() missing required argument: '{param_name}'"
                )

        # Check for unexpected keyword arguments
        unexpected = set(kwargs.keys()) - set(param_names)
        if unexpected:
            raise TypeError(
                f"{cls.__name__}() got unexpected keyword argument(s): "
                f"{', '.join(repr(k) for k in unexpected)}"
            )

        # Parameter registration via Oryx sow/harvest.
        #
        # We sow exactly once per copula instance, sowing a *flattened* vector of the
        # full parameter pytree for this instance. We also attach a static spec to
        # this instance so we can reconstruct the pytree later by slicing from the
        # global flat vector.
        if getattr(_REGISTRY, "active", False):
            params_pytree: Dict[str, Any] = {
                name: getattr(self, name) for name in param_names
            }
            flat_vec, unravel_fn = ravel_pytree(params_pytree)
            flat_size = int(flat_vec.size)

            start = int(getattr(_REGISTRY, "flat_offset", 0))
            _REGISTRY.flat_offset = start + flat_size

            # Sow the flattened vector; harvest(tag="params") returns a dict mapping
            # name -> flat_vec. We use `start` as the name so we can sort/concatenate in order.
            sow(flat_vec, tag="params", name=str(start))

            self._params_symbol = ParamsSymbol(
                start=start,
                size=flat_size,
                unravel_fn=unravel_fn,
            )

    # Create a new class that inherits from Copula
    new_class = type(
        cls.__name__,
        (Copula,),
        {
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__annotations__": annotations,
            "__init__": __init__,
            **{
                name: value
                for name, value in cls.__dict__.items()
                if not name.startswith("_")
            },
        },
    )

    return new_class


@dataclass(frozen=True)
class Leaf:
    index: Optional[int]
    subindex: Optional[int]
    # For marginal DSL: leaf corresponds to an observation index in `obs`
    obs_index: Optional[Tuple[int, ...]] = None
    # Optional marginal distribution type and its parameter spec.
    # We store the *type* (static) and a ParamsSymbol to recover its params from the flat vector.
    dist_type: Optional[type] = None
    dist_params_symbol: Optional["ParamsSymbol"] = None
    dist_static_kwargs: Optional[Dict[str, Any]] = None
    # Censoring support
    censored: bool = False


@dataclass(eq=False)
class Node:
    copula: Copula
    children: List[Union["Node", Leaf]]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class UPlaceholder:
    """Placeholder for `obs`/`u` inside model definition to build a computation graph."""

    def __getitem__(self, key):
        """Handle various indexing patterns.

        Supports:
        - Single integer: u[i] -> Leaf(index=i, subindex=None)
        - Tuple: u[i, j] -> Leaf(index=i, subindex=j)
        - Multiple indices: u[i, j, k, ...] -> Leaf with encoded indices
        - Slices and other types: converted to integers when possible
        """
        # Normalize to a full obs index tuple
        if isinstance(key, tuple):
            if len(key) == 0:
                raise IndexError("Empty index tuple not supported")
            idxs = tuple(self._to_int(k) for k in key)
        else:
            idxs = (self._to_int(key),)

        # Handle slice objects
        if isinstance(key, slice):
            # Convert simple slices to integers
            if key.start is not None and key.stop is None and key.step is None:
                # u[i:] -> treat as u[i]
                idxs = (self._to_int(key.start),)
                idx0 = idxs[0]
                return Leaf(index=idx0, subindex=None, obs_index=idxs)
            raise IndexError(f"Slice indexing not supported: {key}")

        # Handle lists/arrays (could return multiple Leaves)
        if isinstance(key, (list, jnp.ndarray)):
            raise IndexError("List/array indexing not supported. Use integer indices.")

        # Create leaf (keep legacy index/subindex for existing codepaths)
        idx0 = idxs[0]
        idx1 = idxs[1] if len(idxs) > 1 else None
        return Leaf(index=idx0, subindex=idx1, obs_index=idxs)

    @staticmethod
    def _to_int(value) -> int:
        """Convert a value to an integer, handling various types."""
        if isinstance(value, int):
            return value
        if isinstance(value, (jnp.ndarray, jax.Array)):
            return int(value.item())
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Index must be an integer, got {type(value).__name__}: {value}"
            ) from e


# Symbolic parameter system
@dataclass
class ParamsSymbol:
    """Symbolic descriptor for a copula instance's parameter pytree in a flat vector.

    - `start` / `size`: slice in the *global* flattened parameter vector
    - `unravel_fn`: maps a 1D vector (length=size) back to the parameter pytree
    """

    start: int
    size: int
    unravel_fn: Any  # Callable[[jax.Array], Any]

    def __repr__(self):
        return f"ParamsSymbol(start={self.start}, size={self.size})"


def marginal(dist: Any, *, obs: Any, censored: bool = False) -> Leaf:
    """DSL primitive for leaf marginals.

    Args:
        dist: A (tfp) distribution object with a `quantile` method.
        obs: An indexing expression into the obs placeholder, e.g. `obs[i, j, k]`.
        censored: If True, the variable is censored (contributes to the
            copula argument but the density does not differentiate it).

    Returns:
        A Leaf annotated with (dist, obs_index, censored).
    """

    if isinstance(obs, Leaf):
        obs_index = obs.obs_index
        if obs_index is None:
            # Fall back to legacy 1D/2D fields
            if obs.index is None:
                raise ValueError(
                    "marginal(..., obs=...) requires a concrete obs index."
                )
            if obs.subindex is None:
                obs_index = (int(obs.index),)
            else:
                obs_index = (int(obs.index), int(obs.subindex))
    elif isinstance(obs, tuple):
        obs_index = tuple(int(x) for x in obs)
    else:
        raise TypeError(
            "marginal(..., obs=...) expects obs to be an indexed obs placeholder (e.g. obs[i,j,k])."
        )

    if len(obs_index) == 0:
        raise ValueError("marginal(..., obs=...) got empty obs index.")

    # Store the full obs index; keep legacy index/subindex for convenience
    idx0 = obs_index[0]
    idx1 = obs_index[1] if len(obs_index) > 1 else None

    dist_type = type(dist)
    dist_static_kwargs: Dict[str, Any] = {}
    dist_dynamic_kwargs: Dict[str, Any] = {}

    # Try to extract constructor-like parameters from TFP distributions.
    # For TFP JAX substrate, `dist.parameters` is a dict of init args + metadata.
    params = getattr(dist, "parameters", None)
    if isinstance(params, dict):
        for k, v in params.items():
            if k in ("name", "validate_args", "allow_nan_stats", "dtype"):
                # treat as static configuration
                dist_static_kwargs[k] = v
                continue
            # Heuristic: if it's a JAX value/array, treat as dynamic; else static
            if isinstance(v, (jax.Array, jnp.ndarray)):
                dist_dynamic_kwargs[k] = v
            else:
                dist_static_kwargs[k] = v
    else:
        # Best effort: if no parameters dict, assume dist has no dynamic params
        dist_static_kwargs = {}
        dist_dynamic_kwargs = {}

    dist_params_symbol: Optional[ParamsSymbol] = None
    if getattr(_REGISTRY, "active", False) and dist_dynamic_kwargs:
        # Register dynamic distribution params into the same global flat params vector.
        flat_vec, unravel_fn = ravel_pytree(dist_dynamic_kwargs)
        flat_size = int(flat_vec.size)

        start = int(getattr(_REGISTRY, "flat_offset", 0))
        _REGISTRY.flat_offset = start + flat_size

        sow(flat_vec, tag="params", name=str(start))
        dist_params_symbol = ParamsSymbol(
            start=start, size=flat_size, unravel_fn=unravel_fn
        )

    return Leaf(
        index=idx0,
        subindex=idx1,
        obs_index=obs_index,
        dist_type=dist_type,
        dist_params_symbol=dist_params_symbol,
        dist_static_kwargs=dist_static_kwargs if dist_static_kwargs else None,
        censored=censored,
    )


def _collect_leaves_in_order(graph: Node) -> List[Leaf]:
    """Collect leaves in the same DFS order used by leaf transforms."""
    leaves: List[Leaf] = []

    def rec(n: Node):
        for ch in n.children:
            if isinstance(ch, Node):
                rec(ch)
            else:
                leaves.append(ch)

    rec(graph)
    return leaves


def _infer_obs_shape_and_validate_unique(
    leaves: List[Leaf],
) -> Optional[Tuple[int, ...]]:
    """Infer obs shape from marginal leaves and validate unique indexing.

    Returns inferred shape if marginal leaves exist, else None.
    """
    marg_leaves = [lf for lf in leaves if lf.dist_type is not None]
    if not marg_leaves:
        return None

    obs_indices: List[Tuple[int, ...]] = []
    for lf in marg_leaves:
        if lf.obs_index is None:
            raise ValueError("All marginal leaves must have an obs_index.")
        obs_indices.append(lf.obs_index)

    rank = len(obs_indices[0])
    if any(len(ix) != rank for ix in obs_indices):
        raise ValueError("All marginal obs indices must have the same rank.")

    seen = set()
    for ix in obs_indices:
        if ix in seen:
            raise ValueError(f"Duplicate obs index in marginals: {ix}")
        seen.add(ix)

    maxs = [0] * rank
    for ix in obs_indices:
        for d, v in enumerate(ix):
            if v < 0:
                raise ValueError(f"Negative obs index not allowed: {ix}")
            maxs[d] = max(maxs[d], v)

    return tuple(m + 1 for m in maxs)


def _marginal_quantiles_chunked(
    *,
    leaves_in_order: List[Leaf],
    params: jax.Array,
    U_leaves: jax.Array,
) -> Tuple[List[Tuple[int, ...]], jax.Array]:
    """Compute marginal quantiles for each marginal leaf, grouped by dist_type.

    Returns:
      obs_indices: list of obs_index tuples in the same order as returned values
      obs_values_arr: concatenated array of quantile values
    """
    obs_indices: List[Tuple[int, ...]] = []
    obs_values_list: List[jax.Array] = []

    # Group marginal leaves by distribution type (static)
    groups: Dict[type, List[int]] = {}
    for i, lf in enumerate(leaves_in_order):
        if lf.dist_type is None:
            continue
        groups.setdefault(lf.dist_type, []).append(i)

    for dist_type, leaf_is in groups.items():
        # Representative leaf provides static kwargs and param spec shape
        rep_leaf = leaves_in_order[leaf_is[0]]
        static_kwargs = rep_leaf.dist_static_kwargs or {}
        rep_spec = rep_leaf.dist_params_symbol

        starts: List[int] = []
        for li in leaf_is:
            lf = leaves_in_order[li]
            if lf.obs_index is None:
                raise ValueError(
                    "marginal leaf missing obs_index (should have been validated)."
                )
            obs_indices.append(lf.obs_index)
            starts.append(
                int(lf.dist_params_symbol.start)
                if lf.dist_params_symbol is not None
                else -1
            )

            # Validate grouping assumptions: same constructor kwargs keys and same param spec size
            if (lf.dist_static_kwargs or {}) != static_kwargs:
                raise ValueError(
                    "All marginal leaves in a dist_type group must share static kwargs."
                )
            if (lf.dist_params_symbol is None) != (rep_spec is None):
                raise ValueError(
                    "All marginal leaves in a dist_type group must either all have params or none."
                )
            if rep_spec is not None and lf.dist_params_symbol.size != rep_spec.size:
                raise ValueError(
                    "All marginal leaves in a dist_type group must share the same params size."
                )

        starts_arr = jnp.array(starts, dtype=jnp.int32)
        u_arr = U_leaves[jnp.array(leaf_is, dtype=jnp.int32)]

        if rep_spec is None:
            # No dynamic params: instantiate once, then vmap quantile
            dist_obj = dist_type(**static_kwargs)
            obs_vals = jax.vmap(dist_obj.quantile)(u_arr)
        else:
            # Dynamic params: vmap over leaves; each leaf slices params and rebuilds dist
            size = rep_spec.size
            unravel_fn = rep_spec.unravel_fn

            def quantile_one(start_i, u_i):
                flat_slice = lax.dynamic_slice(params, (start_i,), (size,))
                dyn_kwargs = unravel_fn(flat_slice)
                dist_obj = dist_type(**dyn_kwargs, **static_kwargs)
                return dist_obj.quantile(u_i)

            obs_vals = jax.vmap(quantile_one)(starts_arr, u_arr)

        obs_values_list.append(obs_vals)

    obs_values_arr = (
        jnp.concatenate(obs_values_list, axis=0) if obs_values_list else jnp.array([])
    )
    return obs_indices, obs_values_arr


def _scatter_obs_values(
    *,
    obs_shape: Tuple[int, ...],
    obs_indices: List[Tuple[int, ...]],
    obs_values_arr: jax.Array,
) -> jax.Array:
    """Scatter 1D values into an obs-shaped tensor at obs_indices."""
    out = jnp.zeros(obs_shape, dtype=obs_values_arr.dtype)

    # Scatter into output tensor (unique indices guaranteed)
    rank = len(obs_shape)
    idx_arrays = tuple(
        jnp.array([ix[d] for ix in obs_indices], dtype=jnp.int32) for d in range(rank)
    )
    return out.at[idx_arrays].set(obs_values_arr)


def _reconstruct_obs_from_marginals(
    *,
    graph: Node,
    params: jax.Array,
    U_leaves: jax.Array,
) -> jax.Array:
    """Reconstruct an obs-shaped tensor from marginal leaves."""
    leaves_in_order = _collect_leaves_in_order(graph)
    obs_shape = _infer_obs_shape_and_validate_unique(leaves_in_order)
    if obs_shape is None:
        raise ValueError(
            "No marginal(...) leaves found; sampling requires marginals for all leaves."
        )

    obs_indices, obs_values_arr = _marginal_quantiles_chunked(
        leaves_in_order=leaves_in_order,
        params=params,
        U_leaves=U_leaves,
    )
    return _scatter_obs_values(
        obs_shape=obs_shape,
        obs_indices=obs_indices,
        obs_values_arr=obs_values_arr,
    )


# =============================
# Model decorator and container
# =============================
#
# Removed in favour of the pure-functional compile_model API in
# acopula/compile.py.  The legacy Model class held mutable graph state
# and rebuilt the structural representation on every set_params call,
# which invalidated JAX's jit cache between parameter values and made
# parameter sweeps run thousands of times slower than necessary.
#
# Migrate Model() / set_params() / make_ll_fn() callers to:
#
#     from acopula import compile_model, defmodel
#
#     @defmodel
#     def m(p, u): ...
#
#     cm = compile_model(m, template={'theta': 1.0}, method='bell')
#     ll = cm.eval(obs, {'theta': 0.5})
#
# See compile.py for the full CompiledModel surface (sample, ll_fn,
# flatten, visualize, ...).


# =============================


def optimize_order(graph: Node) -> Node:
    """Reorder children of each node.

    Sorting criteria:
    1. Leaves come before Subnodes.
    2. Among Leaves: Censored (True) comes before Uncensored (False).
    3. Among Subnodes: Sort by number of uncensored leaves in their subtree (descending).
       More uncensored leaves -> earlier in list.
    """

    def count_uncensored_leaves(n: Union[Node, Leaf]) -> int:
        if isinstance(n, Leaf):
            return 0 if getattr(n, "censored", False) else 1
        return sum(count_uncensored_leaves(c) for c in n.children)

    # Recursively optimize children that are Nodes
    optimized_subnodes = []
    leaves = []

    for child in graph.children:
        if isinstance(child, Node):
            # Recursively optimize the subnode
            opt_child = optimize_order(child)
            # Annotate with uncensored count for sorting
            count = count_uncensored_leaves(opt_child)
            optimized_subnodes.append((count, opt_child))
        else:
            leaves.append(child)

    # Sort subnodes by count descending (more uncensored -> first)
    optimized_subnodes.sort(key=lambda x: x[0], reverse=True)
    sorted_subnodes = [node for _, node in optimized_subnodes]

    # Partition leaves: Uncensored first (for easier k-th order derivative slicing)
    uncensored_leaves = [lf for lf in leaves if not getattr(lf, "censored", False)]
    censored_leaves = [lf for lf in leaves if getattr(lf, "censored", False)]

    # Final Order: Uncensored Leaves, Censored Leaves, Sorted Subnodes
    sorted_children = uncensored_leaves + censored_leaves + sorted_subnodes

    return Node(copula=graph.copula, children=sorted_children)


def _structure_key(node: Union[Node, Leaf]) -> Tuple:
    """Generate a hashable key representing the computational structure of a node.

    Nodes with identical structure keys can be batched (vmapped).
    Structure includes:
    - Node type (Leaf vs Node)
    - For Leaf: censored status
    - For Node: Copula class and recursive structure of children
    """
    if isinstance(node, Leaf):
        return ("Leaf", getattr(node, "censored", False))

    # Node
    child_keys = tuple(_structure_key(c) for c in node.children)
    return ("Node", type(node.copula), child_keys)


def _group_children_by_structure(graph: Node) -> Dict[Tuple, List[int]]:
    """Group children indices by their structure key."""
    groups = {}
    for i, child in enumerate(graph.children):
        key = _structure_key(child)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return groups


def _instantiate_copula_from_flat(node: Node, params_flat: jax.Array) -> Copula:
    """Recreate a copula instance for `node` from the global flat params vector."""
    spec: ParamsSymbol = node.copula._params_symbol  # type: ignore[attr-defined]
    start = int(spec.start)
    size = int(spec.size)
    flat_slice = lax.dynamic_slice(params_flat, (start,), (size,))
    params_pytree = spec.unravel_fn(flat_slice)
    return type(node.copula)(**params_pytree)


def _leaf_value_from_obs(lf: Leaf, obs: jax.Array) -> jax.Array:
    if lf.obs_index is None:
        raise ValueError("Leaf missing obs_index; cannot map observation value.")
    return jnp.asarray(obs[lf.obs_index])


def _marginal_transforms_chunked(
    *,
    leaves_in_order: List[Leaf],
    params: jax.Array,
    obs: jax.Array,
    survival: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Compute copula inputs and marginal log-likelihoods for all leaves.

    Pre-builds a single advanced-indexing gather for ALL leaves (one HLO op
    independent of the number of leaves), buckets by (dist_type, is_censored)
    for vmapped CDF/log_prob calls, and scatters results back into a flat
    output via at[].set (one op per group).  This keeps the trace work
    constant in the leaf count, matching the schedule.py-driven design of
    the rest of the bell pipeline.

    When ``survival=True`` (right-censored survival data), the copula models
    the joint survival function S = C(S_1,...,S_d) and ALL leaves receive the
    survival probability u = 1 - F(t).
    """
    n_leaves = len(leaves_in_order)
    if n_leaves == 0:
        return jnp.zeros(0), jnp.zeros(0)

    # Validate every leaf has obs_index (Python-side, no JAX op).
    for lf in leaves_in_order:
        if lf.obs_index is None:
            raise ValueError("Leaf missing obs_index; cannot map observation value.")

    # Build batched advanced-indexing gather: one HLO op for all leaves.
    rank = len(leaves_in_order[0].obs_index)
    if rank == 1:
        all_idx = jnp.array(
            [lf.obs_index[0] for lf in leaves_in_order], dtype=jnp.int32)
        obs_arr_all = obs[all_idx]                                     # (n_leaves,)
    else:
        per_dim_idxs = tuple(
            jnp.array([lf.obs_index[k] for lf in leaves_in_order], dtype=jnp.int32)
            for k in range(rank))
        obs_arr_all = obs[per_dim_idxs]                                # (n_leaves,)

    # Bucket leaves Python-side.
    groups: Dict[Tuple[type, bool], List[int]] = {}
    no_dist_indices: List[int] = []
    for i, lf in enumerate(leaves_in_order):
        if lf.dist_type is None:
            no_dist_indices.append(i)
        else:
            groups.setdefault(
                (lf.dist_type, getattr(lf, "censored", False)), []).append(i)

    # Output accumulators: assemble via scatter, one op per group.
    u_vec = jnp.zeros(n_leaves, dtype=obs_arr_all.dtype)
    log_lik_vec = jnp.zeros(n_leaves, dtype=obs_arr_all.dtype)

    if no_dist_indices:
        # No distribution => obs already on copula scale, just copy through.
        idx_arr = jnp.array(no_dist_indices, dtype=jnp.int32)
        u_vec = u_vec.at[idx_arr].set(obs_arr_all[idx_arr])

    for (dist_type, is_censored), leaf_is in groups.items():
        rep_leaf = leaves_in_order[leaf_is[0]]
        static_kwargs = rep_leaf.dist_static_kwargs or {}
        rep_spec = rep_leaf.dist_params_symbol
        idx_arr = jnp.array(leaf_is, dtype=jnp.int32)
        obs_arr = obs_arr_all[idx_arr]                                 # (n_g,)

        if rep_spec is None:
            dist_obj = dist_type(**static_kwargs)
            cdf_vals = jax.vmap(dist_obj.cdf)(obs_arr)
            if survival:
                cdf_vals = 1.0 - cdf_vals
            if not is_censored:
                log_lik_vals = jax.vmap(dist_obj.log_prob)(obs_arr)
            else:
                log_lik_vals = jnp.zeros_like(cdf_vals)
        else:
            size = rep_spec.size
            unravel_fn = rep_spec.unravel_fn
            starts_arr = jnp.array(
                [int(leaves_in_order[li].dist_params_symbol.start) for li in leaf_is],
                dtype=jnp.int32)

            def transform_one(start_i, x_i):
                flat_slice = lax.dynamic_slice(params, (start_i,), (size,))
                dyn_kwargs = unravel_fn(flat_slice)
                dist_obj = dist_type(**dyn_kwargs, **static_kwargs)
                u = dist_obj.cdf(x_i)
                if survival:
                    u = 1.0 - u
                if not is_censored:
                    return u, dist_obj.log_prob(x_i)
                return u, jnp.array(0.0, dtype=u.dtype)

            cdf_vals, log_lik_vals = jax.vmap(transform_one)(starts_arr, obs_arr)

        u_vec = u_vec.at[idx_arr].set(cdf_vals)
        log_lik_vec = log_lik_vec.at[idx_arr].set(log_lik_vals)

    return u_vec, log_lik_vec


def _log_likelihood_single_layer(
    graph: Node, obs: jax.Array, params_flat: jax.Array,
    survival: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Log-likelihood for a single-layer Archimedean copula."""

    # 0. Reorder graph: Uncensored leaves first, then censored
    graph = optimize_order(graph)

    # 1. Compute marginal transforms (CDF -> u, log_prob -> marginal density)
    leaves = _collect_leaves_in_order(graph)

    # Identify number of uncensored leaves (k)
    # Since optimize_order puts uncensored first, we just count them.
    # We can trust they are at the beginning.
    k = 0
    for lf in leaves:
        if not getattr(lf, "censored", False):
            k += 1
        else:
            break

    # Verification: Ensure no uncensored leaves after index k
    # (optimize_order should guarantee this, but good to be safe/aware)

    u_vec, log_lik_vec = _marginal_transforms_chunked(
        leaves_in_order=leaves,
        params=params_flat,
        obs=obs,
        survival=survival,
    )

    d = u_vec.shape[0]
    if d == 0:
        return jnp.array(0.0), jnp.array(0.0)

    # 2. Instantiate copula
    copula_instance = _instantiate_copula_from_flat(graph, params_flat)
    # generator is phi (the Laplace transform [0, inf] -> [0,1])
    # generator_inv is phi^-1 (the map [0,1] -> [0, inf])
    # C(u) = phi( sum phi^-1(u_i) ).
    phi = copula_instance.generator
    phi_inv = copula_instance.generator_inv

    # 3. Compute mixed partial derivative
    # _full_mixed_partials(outer, inner, u_vec, k)
    # outer = phi, inner = phi_inv
    copula_log_lik = _full_mixed_partials(
        phi, phi_inv, u_vec, k, copula_instance=copula_instance
    )

    # 4. Marginal Log-likelihood
    # Only sum log_lik for Uncensored leaves.
    if k > 0:
        marginal_log_lik = jnp.sum(log_lik_vec[:k])
    else:
        marginal_log_lik = 0.0

    return phi(jnp.sum(jax.vmap(phi_inv)(u_vec))), copula_log_lik + marginal_log_lik


def _log_likelihood_nested(
    graph: Node,
    obs: jax.Array,
    params_flat: jax.Array,
    *,
    ils_method: str = "fixed_talbot",
    ils_params: Optional[dict] = None,
    post_widder_k: int = 8,
    survival: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Log-likelihood for nested copulas using integral method."""

    # 0. Reorder graph: Uncensored leaves first, etc.
    graph = optimize_order(graph)

    # 1. Compute marginal transforms
    leaves = _collect_leaves_in_order(graph)
    u_vec, log_lik_vec = _marginal_transforms_chunked(
        leaves_in_order=leaves,
        params=params_flat,
        obs=obs,
        survival=survival,
    )

    # Marginal LL (sum uncensored)
    marginal_log_lik = 0.0
    for i, lf in enumerate(leaves):
        if not getattr(lf, "censored", False):
            marginal_log_lik += log_lik_vec[i]

    # Root corresponds to integrating over frailty V0 ~ Mixing(phi_0)
    # where phi_0 is the Laplace transform [0, inf] -> [0, 1].
    # In acopula, phi_0 is generator.
    root_copula = _instantiate_copula_from_flat(graph, params_flat)
    phi_0 = root_copula.generator  # Laplace transform
    psi_0 = root_copula.generator_inv  # Inverse generator phi^-1

    # Log PDF of mixing density f_V(v) via Post-Widder
    # f_V(v) approx (-1)^k/k! * (k/v)^(k+1) * phi^(k)(k/v)
    # log f_V(v) = log(|phi^(k)|) + (k+1)log(k/v) - log(k!)

    ils_params_ = {} if ils_params is None else dict(ils_params)
    ils_method_ = str(ils_method).lower()

    if ils_method_ == "post_widder":
        k_pw = int(ils_params_.get("k", post_widder_k))

        def log_mixing_pdf(v):
            return _post_widder_log_pdf(phi_0, v, k_pw)

    elif ils_method_ == "fixed_talbot":
        M = int(ils_params_.get("M", 18))
        tmax = ils_params_.get("tmax", None)
        pdf_fun = _fixed_talbot(phi_0, M)

        def log_mixing_pdf(v):
            val = pdf_fun(v, tmax=tmax) if tmax is not None else pdf_fun(v)
            return jnp.log(val)

    elif ils_method_ == "cohen":
        degree = int(ils_params_.get("degree", 24))
        alpha = ils_params_.get("alpha", None)
        pdf_fun = _cohen(phi_0, degree=degree)

        def log_mixing_pdf(v):
            val = pdf_fun(v, alpha=alpha) if alpha is not None else pdf_fun(v)
            return jnp.log(val)

    else:
        raise ValueError(
            f"Unknown ils_method={ils_method!r}. Expected one of "
            "'post_widder', 'fixed_talbot', or 'cohen'."
        )

    # 3. Process Children
    # Group children by structure to vmap processing
    child_groups = _group_children_by_structure(graph)

    # Pre-process children
    # We need to map leaf indices in 'leaves' list to u_vec slices for each child group.
    # Group processing is needed for efficiency (vmap over similar children).

    # Map leaf id -> index in u_vec
    leaf_id_to_idx = {id(lf): i for i, lf in enumerate(leaves)}

    # For direct leaves, we can process them as a single group if they share censoring status
    # _group_children_by_structure already separates censored vs uncensored leaves.

    def process_child_group(key, indices, v0):
        # key is structure key: (type, ...)
        # indices is list of child indices in graph.children

        child_nodes = [graph.children[i] for i in indices]

        # Dispatch based on child type (Leaf vs Node)
        if key[0] == "Leaf":
            # Direct leaf children u_j
            # Gather u values
            u_indices = jnp.array([leaf_id_to_idx[id(n)] for n in child_nodes])
            u_vals = u_vec[u_indices]
            is_censored = key[1]

            def fn(u_j):
                return jnp.exp(-v0 * psi_0(u_j))

            fn_prime = jax.grad(fn)

            if is_censored:
                # k=0
                return jnp.sum(jnp.log(jax.vmap(fn)(u_vals)))
            else:
                jax.debug.print("fn_prime: {}", jax.vmap(fn_prime)(u_vals))
                return jnp.sum(jnp.log(jax.vmap(fn_prime)(u_vals)))

        else:
            # Node child C_i
            child_structure_key = key[2]

            # Determine if this is a single-layer or multi-layer child
            is_single_layer = all(k[0] == "Leaf" for k in child_structure_key)

            if is_single_layer:
                return _process_single_layer_child(
                    v0, child_nodes, params_flat, u_vec, leaf_id_to_idx, psi_0
                )
            else:
                return _process_multi_layer_child(
                    v0, child_nodes, params_flat, u_vec, leaf_id_to_idx, psi_0
                )

        return 0.0  # Should be unreachable given logic above

    # Define Integrand
    def integrand(v0):
        # Sum of log-contributions from all children
        logpdf = log_mixing_pdf(v0)
        log_val = logpdf
        # jax.debug.print("log_val: {} v0: {}", log_val, v0)
        # Iterate over groups
        gvs = []
        for key, indices in child_groups.items():
            gvs.append(process_child_group(key, indices, v0))
            log_val += gvs[-1]
        jax.debug.print(
            "gvs: {} log_val: {} v0: {} logpdf: {}", gvs, log_val, v0, logpdf
        )
        return jnp.exp(log_val)

    # 4. Integrate
    res, info = quadax.quadgk(
        integrand,
        [0, jnp.inf],
        # max_ninter=10000,
        # epsabs=1e-12,
        # epsrel=1e-10,
        # order=101,
    )

    final_log_lik = jnp.log(res) + marginal_log_lik

    # Dummy root value
    return jnp.array(0.0), final_log_lik


def _process_single_layer_child(
    v0: jax.Array,
    child_nodes: List[Node],
    params_flat: jax.Array,
    u_vec: jax.Array,
    leaf_id_to_idx: Dict[int, int],
    psi_root: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Compute log-contribution for a batch of single-layer child subtrees.

    Each child C_i corresponds to a conditional copula given v0.
    The conditional generator is Theta_i(t) = exp(-v0 * psi_root(phi_i(t))).
    We compute the mixed partial of Theta_i(sum psi_i(u)).
    """
    # All child_nodes have the same structure and copula type
    # We can vmap over them.
    # We need to gather:
    # 1. Copula parameters for each child node
    # 2. Leaves data for each child node (u values)

    # Instantiate copulas
    # Since they are same type, we can batch instantiation?
    # _instantiate_copula_from_flat works on a single node.
    # Let's assume we can vmap the computation over child indices.

    # But leaves might have different sizes or structure?
    # No, _group_children_by_structure ensures they have identical structure.
    # This means same number of children, same leaf types in same order.
    # So we can stack u values.

    # Structure of child_nodes[0]:
    # It has N leaves.
    # We need to collect u indices for these N leaves for EACH child node.
    # shape of u_indices: (num_child_nodes, num_leaves_per_child)

    representative = child_nodes[0]

    # Collect u indices
    # We iterate over children of the child nodes
    # For child j, leaf k is child_nodes[j].children[k]
    all_u_indices = []
    for node in child_nodes:
        node_indices = [leaf_id_to_idx[id(ch)] for ch in node.children]
        all_u_indices.append(node_indices)
    u_indices = jnp.array(all_u_indices)  # (M, N)

    batch_u = u_vec[u_indices]  # (M, N)

    # Identify uncensored count k
    # Structure key guarantees censoring pattern is identical
    k_unc = 0
    for ch in representative.children:
        if not getattr(ch, "censored", False):
            k_unc += 1
        else:
            # Assumes sorted by optimize_order (uncensored first)
            break

    # Gather start indices.
    starts = []
    size = int(representative.copula._params_symbol.size)
    unravel_fn = representative.copula._params_symbol.unravel_fn
    copula_type = type(representative.copula)

    for node in child_nodes:
        starts.append(int(node.copula._params_symbol.start))
    starts = jnp.array(starts)

    def compute_one_child(start_idx, u_vals):
        # Reconstruct copula
        flat_slice = lax.dynamic_slice(params_flat, (start_idx,), (size,))
        params_pytree = unravel_fn(flat_slice)
        copula = copula_type(**params_pytree)

        # In acopula:
        # generator is phi (outer)
        # generator_inv is phi^-1 (inner)
        phi_i = copula.generator
        phi_i_inv = copula.generator_inv

        # Define Tilted Generator Theta(t) = exp(-v0 * psi_root(phi_i(t)))
        # This is the outer function applied to the sum of phi_i_inv(u)
        def theta_generator(t):
            # phi_i(t) maps [0, inf] -> [0, 1]
            # psi_root (phi_root^-1) maps [0, 1] -> [0, inf]
            return jnp.exp(-v0 * psi_root(phi_i(t)))

        # Compute mixed partial using helper
        # _full_mixed_partials(outer, inner, u, k)
        # outer = theta_generator, inner = phi_i_inv
        return _full_mixed_partials(theta_generator, phi_i_inv, u_vals, k_unc)

    # Vmap over the batch of child nodes
    log_contribs = jax.vmap(compute_one_child)(starts, batch_u)

    return jnp.sum(log_contribs)


def _process_multi_layer_child(
    v0: jax.Array,
    child_nodes: List[Node],
    params_flat: jax.Array,
    u_vec: jax.Array,
    leaf_id_to_idx: Dict[int, int],
    psi_root: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Compute log-contribution for a batch of multi-layer child subtrees.

    Uses brute-force combinatorial sum of directional derivatives.
    L = 1/k! * sum_{S subset {1...k}} (-1)^(k-|S|) D_vS^k G(u)
    where G(u) = exp(-v0 * psi_root( C_child(u) )).
    """
    representative = child_nodes[0]

    # 1. Collect all leaves in subtree
    subtree_leaves = _collect_leaves_in_order(representative)
    num_leaves = len(subtree_leaves)

    # Identify uncensored indices among these leaves
    unc_indices = jnp.array(
        [i for i, lf in enumerate(subtree_leaves) if not getattr(lf, "censored", False)]
    )
    k = len(unc_indices)

    if k > 12:
        # Prevent exponential explosion for very large subtrees
        raise ValueError(
            f"Mixed partial for multi-layer subtree with k={k} uncensored variables is too expensive (2^{k} evaluations)."
        )

    # 2. Identify subnodes and their parameter info for batch reconstruction
    subnodes_metadata = []

    def collect_nodes(n):
        if isinstance(n, Node):
            subnodes_metadata.append(
                {
                    "spec": n.copula._params_symbol,
                    "type": type(n.copula),
                    "children": n.children,
                }
            )
            for ch in n.children:
                collect_nodes(ch)

    collect_nodes(representative)

    # Pre-collect all start indices for all child nodes in batch
    # shape: (num_child_nodes, num_subnodes)
    all_starts = []
    for node in child_nodes:
        starts = []

        def walk_starts(n):
            if isinstance(n, Node):
                starts.append(n.copula._params_symbol.start)
                for ch in n.children:
                    walk_starts(ch)

        walk_starts(node)
        all_starts.append(starts)
    all_starts = jnp.array(all_starts)

    # 3. Pre-collect leaf indices for each child
    # Each child node in batch has its own leaves mapping to u_vec
    all_batch_u_indices = []
    for node in child_nodes:
        node_leaves = _collect_leaves_in_order(node)
        all_batch_u_indices.append([leaf_id_to_idx[id(lf)] for lf in node_leaves])
    batch_u_indices = jnp.array(all_batch_u_indices)
    batch_u = u_vec[batch_u_indices]

    # 4. Combinatorial logic (Inclusion-Exclusion)
    bit_matrix = (
        (jnp.arange(1 << k, dtype=jnp.int32)[:, None] >> jnp.arange(k)) & 1
    ).astype(jnp.float64)
    masks = jnp.zeros((1 << k, num_leaves)).at[:, unc_indices].set(bit_matrix)
    subset_sizes = jnp.sum(bit_matrix, axis=1)
    parity = (-1.0) ** (k - subset_sizes)

    def compute_one_child(u_vals, starts):
        def subtree_fn(u_in):
            # Evaluate the subtree value C_child(u_in)
            # Recursively walk the representative structure, but use `starts`
            # and `u_in` for the current child in the batch.
            curr_node_idx = 0
            curr_leaf_idx = 0

            def walk(node_temp):
                nonlocal curr_node_idx, curr_leaf_idx
                if isinstance(node_temp, Leaf):
                    val = u_in[curr_leaf_idx]
                    curr_leaf_idx += 1
                    return val

                # Internal node
                info = subnodes_metadata[curr_node_idx]
                start = starts[curr_node_idx]
                curr_node_idx += 1

                child_vals_list = [walk(ch) for ch in info["children"]]
                child_vals = jnp.stack(child_vals_list)

                # Reconstruct copula
                spec = info["spec"]
                flat_slice = lax.dynamic_slice(params_flat, (start,), (int(spec.size),))
                params_pytree = spec.unravel_fn(flat_slice)
                cop = info["type"](**params_pytree)

                return cop.generator(jnp.sum(jax.vmap(cop.generator_inv)(child_vals)))

            return walk(representative)

        def target_fn(u_in):
            # G(u) = exp(-v0 * psi_root( C_child(u) ))
            c_val = subtree_fn(u_in)
            return jnp.exp(-v0 * psi_root(c_val))

        # Helper for k-th order directional derivative using jet
        def compute_directional(mask):
            # Directional derivative coefficient D_mask^k target_fn(u_vals) / k!
            # acopula.jet_array.jet expects a single array for series coefficients along axis 0.
            # shape: (k, num_leaves)
            series_x = jnp.stack([mask] + [jnp.zeros_like(mask)] * (k - 1), axis=0)
            _, series_out = jet_array.jet(target_fn, (u_vals,), (series_x,))
            # jet returns Taylor coefficients
            return series_out[-1]

        # Vmap over all 2^k masks
        d_k_vals = jax.vmap(compute_directional)(masks)

        # Mixed partial derivative = sum (parity * Taylor_coeff)
        # By inclusion-exclusion: mixed partial = 1/k! * sum (parity * D_mask^k)
        # Since Taylor_coeff = D_mask^k / k!, the sum is exactly the mixed partial.
        mixed_partial = jnp.sum(parity * d_k_vals)

        return jnp.log(mixed_partial)

    # Vmap over batch of child nodes
    log_contribs = jax.vmap(compute_one_child)(batch_u, all_starts)
    return jnp.sum(log_contribs)


# =============================
# Sampling helpers
# =============================


def _compute_sampling_schedule(graph: Node):
    """Compute the sampling schedule for a graph using ancestry-based colors.

    Returns a dict with:
        - stages_nodes: List[List[Node]] in top-down order
        - node_to_stage_col: dict[Node, Tuple[int, int]]
        - node_parent: dict[Node, Optional[Node]]
    """
    from .schedule import solve_scheduling_nodes

    # Build parent map
    node_parent: dict[Node, Optional[Node]] = {}

    def build_parent_map(n: Node, p: Optional[Node]):
        node_parent[n] = p
        for ch in n.children:
            if isinstance(ch, Node):
                build_parent_map(ch, n)

    build_parent_map(graph, None)

    # Build ancestry key: tuple of copula class names from root to node
    def ancestry_key(node: Node) -> Tuple[str, ...]:
        path: List[str] = []
        cur: Optional[Node] = node
        while cur is not None:
            path.append(cur.copula.__class__.__name__)
            cur = node_parent.get(cur)
        path.reverse()
        return tuple(path)

    # Solve scheduling
    result = solve_scheduling_nodes(graph, ancestry_key)
    stages_nodes: List[List[Node]] = result["stages"]

    # Reverse to get top-down order (root first)
    stages_nodes = list(reversed(stages_nodes))

    # Build node -> (stage_idx, col_idx) lookup
    node_to_stage_col: dict[Node, Tuple[int, int]] = {}
    for si, nodes_in_stage in enumerate(stages_nodes):
        for cj, node_obj in enumerate(nodes_in_stage):
            node_to_stage_col[node_obj] = (si, cj)

    return {
        "stages_nodes": stages_nodes,
        "node_to_stage_col": node_to_stage_col,
        "node_parent": node_parent,
    }


def _build_sampling_metadata(schedule_result: dict) -> dict:
    """Build index mappings and ancestry information for sampling.

    Returns dict with:
        - total_nodes: int
        - node_to_idx: dict[Node, int]
        - node_ancestry_indices: dict[Node, List[int]]
    """
    stages_nodes = schedule_result["stages_nodes"]
    node_parent = schedule_result["node_parent"]

    # Build node -> flat index mapping
    total_nodes = sum(len(stage) for stage in stages_nodes)
    node_to_idx: dict[Node, int] = {}
    idx = 0
    for stage in stages_nodes:
        for node_obj in stage:
            node_to_idx[node_obj] = idx
            idx += 1

    # Build ancestry chain lookup
    def get_ancestry_chain(node: Node) -> List[Node]:
        chain: List[Node] = []
        cur: Optional[Node] = node
        while cur is not None:
            chain.append(cur)
            cur = node_parent.get(cur)
        chain.reverse()
        return chain

    # Pre-compute ancestry indices
    node_ancestry_indices: dict[Node, List[int]] = {}
    for stage in stages_nodes:
        for node_obj in stage:
            ancestry = get_ancestry_chain(node_obj)
            node_ancestry_indices[node_obj] = [
                node_to_idx[anc] for anc in ancestry[:-1]
            ]

    return {
        "total_nodes": total_nodes,
        "node_to_idx": node_to_idx,
        "node_ancestry_indices": node_ancestry_indices,
        "get_ancestry_chain": get_ancestry_chain,
    }


def _sample_v_values_topdown(
    stages_nodes: List[List[Node]],
    metadata: dict,
    params_flat: jax.Array,
    node_param_starts: jax.Array,
    parent_lookup: jax.Array,
    key: jax.Array,
    post_widder_k: int,
    max_cdf_x: float,
) -> jax.Array:
    """Sample V values for all nodes top-down using stages.

    Returns v_vals array of shape (total_nodes,).
    """
    total_nodes = metadata["total_nodes"]
    node_to_idx = metadata["node_to_idx"]
    node_ancestry_indices = metadata["node_ancestry_indices"]
    get_ancestry_chain = metadata["get_ancestry_chain"]

    # Prepare RNG keys per stage/node
    keys_per_stage: List[List[jax.Array]] = []
    k = key
    for nodes_in_stage in stages_nodes:
        k, *stage_keys = jrandom.split(k, 1 + len(nodes_in_stage))
        keys_per_stage.append(stage_keys)

    # Storage: flat V values array
    v_vals = jnp.zeros(total_nodes)

    # Sample each stage
    v_vals = _sample_stages_topdown(
        stages_nodes,
        keys_per_stage,
        v_vals,
        node_to_idx,
        node_ancestry_indices,
        get_ancestry_chain,
        params_flat,
        node_param_starts,
        parent_lookup,
        post_widder_k,
        max_cdf_x,
    )

    return v_vals


def _get_copula_instance_factory(
    params_flat: jax.Array,
    node_param_starts: jax.Array,
):
    """Factory to create a get_copula_instance function.

    Returns a function that retrieves a copula instance for a given node index.
    """

    def get_copula_instance(representative_node: Node, node_idx: int):
        """Reconstruct full params pytree and instantiate copula for node_idx."""
        copula_type = type(representative_node.copula)
        spec: ParamsSymbol = representative_node.copula._params_symbol
        start = node_param_starts[node_idx]
        flat_slice = lax.dynamic_slice(params_flat, (start,), (spec.size,))
        params_pytree = spec.unravel_fn(flat_slice)
        return copula_type(**params_pytree)

    return get_copula_instance


def _make_connecting_lt_factory(
    get_copula_instance: Callable,
    get_ancestry_chain: Callable,
    parent_lookup: jax.Array,
    v_vals: jax.Array,
):
    """Factory to create connecting Laplace transforms for frailty sampling.

    For each node, produces a function gen(i, u) that is the Laplace transform
    of the node's mixing distribution:
      - Root: gen(i, u) = phi_root(u)  (plain generator)
      - Non-root: gen(i, u) = exp(-V_parent * phi_parent^{-1}(phi_child(u)))

    Uses only plain generator inverses (not recursively-built modified
    generators), so oryx_core.inverse only needs to invert simple closed-form
    generators. This enables nesting at arbitrary depth.

    The mathematical equivalence follows from the cancellation identity:
      [psi_tilde_parent]^{-1} o psi_tilde_child = phi_parent^{-1} o phi_child
    where psi_tilde denotes the recursively-modified generator.
    """

    def make_connecting_lt(representative_node: Node):
        """Create connecting Laplace transform gen(i, u) for a node.

        Args:
            representative_node: Representative node (all same copula type in stage)

        Returns:
            A function gen(i, u) where i is the global node index
        """
        ancestry = get_ancestry_chain(representative_node)
        is_root = len(ancestry) == 1

        if is_root:
            # Root: gen(i, u) = phi(u)
            def gen(i, u):
                copula_instance = get_copula_instance(representative_node, i)
                return copula_instance.generator(u)

            return gen
        else:
            # Non-root: gen(i, u) = exp(-V_parent * phi_parent^{-1}(phi_child(u)))
            # Uses plain generator inverse — works at any depth.
            parent_node = ancestry[-2]

            def gen(i, u):
                parent_node_idx = parent_lookup[i]
                v_parent = v_vals[parent_node_idx]

                # Plain child generator
                copula_instance = get_copula_instance(representative_node, i)
                phi_child_u = copula_instance.generator(u)

                # Plain parent generator inverse
                parent_instance = get_copula_instance(parent_node, parent_node_idx)
                composition = parent_instance.generator_inv(phi_child_u)

                return jnp.exp(-v_parent * composition)

            return gen

    return make_connecting_lt


def _sample_stages_topdown(
    stages_nodes: List[List[Node]],
    keys_per_stage: List[List[jax.Array]],
    v_vals: jax.Array,
    node_to_idx: dict,
    node_ancestry_indices: dict,
    get_ancestry_chain: Callable,
    params_flat: jax.Array,
    node_param_starts: jax.Array,
    parent_lookup: jax.Array,
    post_widder_k: int,
    max_cdf_x: float,
) -> jax.Array:
    """Process all stages top-down, sampling V values.

    For each node, we sample from a modified generator:
    - Root: psi(t) (standard generator)
    - Child: psi_child(t; v_parent) = exp[-v_parent * psi_parent^{-1}(psi_child(t))]

    This function uses vectorization (vmap) within each stage.
    """
    # Create helper functions using factories
    get_copula_instance = _get_copula_instance_factory(params_flat, node_param_starts)

    # Sample each stage sequentially (must be sequential since later stages depend on earlier v values)
    for stage_idx, nodes_in_stage in enumerate(stages_nodes):

        def sample_at_stage(stage, v):
            """Sample V values for all nodes in a stage."""
            stage_keys = jnp.stack(keys_per_stage[stage_idx])
            node_indices = jnp.array(
                [node_to_idx[node] for node in stage], dtype=jnp.int32
            )

            representative_node = stage[0]

            # Create connecting Laplace transform factory for this stage
            make_connecting_lt = _make_connecting_lt_factory(
                get_copula_instance, get_ancestry_chain, parent_lookup, v
            )

            def sample_i(i, key):
                """Sample V for node at stage position i (0-indexed within stage)."""
                # Get global node index
                node_idx = node_indices[i]

                # Create connecting Laplace transform
                gen = make_connecting_lt(representative_node)

                # Partially apply node_idx to gen to get psi(u)
                def psi(u):
                    return gen(node_idx, u)

                return _sample_frailty_via_post_widder(
                    key, psi, post_widder_k, max_cdf_x
                )

            # Vmap over all nodes in stage
            indices = jnp.arange(len(stage))
            return jax.vmap(sample_i)(indices, stage_keys), node_indices

        # Sample all V values for this stage
        v_sampled, node_indices = sample_at_stage(nodes_in_stage, v_vals)

        # Update v_vals array
        v_vals = v_vals.at[node_indices].set(v_sampled)

    return v_vals


def _transform_leaves_bottomup(
    graph: Node,
    stages_nodes: List[List[Node]],
    v_vals: jax.Array,
    node_to_idx: dict,
    get_ancestry_chain: Callable,
    params_flat: jax.Array,
    node_param_starts: jax.Array,
    parent_lookup: jax.Array,
    key: jax.Array,
    total_nodes: int,
) -> jax.Array:
    """Transform uniform leaf samples using the Marshall-Olkin algorithm.

    For each leaf with immediate parent node p:
      U = phi_p(-log(X) / V_p)
    where phi_p is the plain generator of the parent and V_p is the
    parent's sampled frailty value.

    This is equivalent to the standard formula U = psi(E / V) from
    McNeil (2008), where E = -log(X) ~ Exp(1).

    Groups leaves by their immediate parent's copula type for efficient
    vectorization via vmap.

    Returns U_leaves array of shape (num_leaves,).
    """
    # Collect leaves with their immediate parent info
    leaf_data = []  # List of (leaf_idx, immediate_parent_node)
    leaf_idx = 0

    def collect_leaves(n: Node, path: List[Node]):
        nonlocal leaf_idx
        for ch in n.children:
            if isinstance(ch, Node):
                collect_leaves(ch, path + [ch])
            else:
                # Leaf: immediate parent is the last node in the path
                leaf_data.append((leaf_idx, path[-1]))
                leaf_idx += 1

    collect_leaves(graph, [graph])
    num_leaves = leaf_idx

    # Sample uniform for all leaves
    k_e = jrandom.fold_in(key, total_nodes)
    X_vals = jrandom.uniform(k_e, (num_leaves,))

    # Group leaves by immediate parent copula type for vectorization
    leaf_groups = {}  # parent_copula_type_name -> [(leaf_idx, parent_node), ...]
    for lidx, parent_node in leaf_data:
        key_name = type(parent_node.copula).__name__
        if key_name not in leaf_groups:
            leaf_groups[key_name] = []
        leaf_groups[key_name].append((lidx, parent_node))

    get_copula_instance = _get_copula_instance_factory(params_flat, node_param_starts)

    # Process each group of leaves with the same parent copula type
    for group_key, group_leaves in leaf_groups.items():
        representative_parent = group_leaves[0][1]
        leaf_indices = jnp.array([lidx for lidx, _ in group_leaves], dtype=jnp.int32)
        parent_indices = jnp.array(
            [node_to_idx[pnode] for _, pnode in group_leaves], dtype=jnp.int32
        )

        def transform_leaf(i, X_in):
            parent_node_idx = parent_indices[i]
            V_parent = v_vals[parent_node_idx]
            parent_copula = get_copula_instance(representative_parent, parent_node_idx)
            return parent_copula.generator(-jnp.log(X_in) / V_parent)

        indices = jnp.arange(len(group_leaves))
        X_current = X_vals[leaf_indices]
        X_transformed = jax.vmap(transform_leaf)(indices, X_current)
        X_vals = X_vals.at[leaf_indices].set(X_transformed)

    return X_vals


def _sample_once(
    graph: Node,
    key: jax.Array,
    params: jax.Array,
    post_widder_k: int,
    max_cdf_x: float,
) -> jax.Array:
    """Sample one observation using Marshall-Olkin algorithm.

    High-level algorithm:
    1. Compute schedule (stages with indices)
    2. Build sampling metadata (index mappings, ancestry)
    3. Build global parameter indexers
    4. Sample V values top-down (internal nodes)
    5. Transform leaves bottom-up (from uniform to final)
    """
    # Step 1: Compute schedule
    schedule_result = _compute_sampling_schedule(graph)
    stages_nodes = schedule_result["stages_nodes"]

    # Step 2: Build metadata
    metadata = _build_sampling_metadata(schedule_result)

    # Step 3: Build global parameter indexers (used by both top-down and bottom-up passes)
    total_nodes = metadata["total_nodes"]
    node_to_idx = metadata["node_to_idx"]
    node_ancestry_indices = metadata["node_ancestry_indices"]

    # Build parent_lookup array
    parent_lookup_list = [-1] * total_nodes
    for stage_nodes in stages_nodes:
        for node in stage_nodes:
            node_idx = node_to_idx[node]
            ancestry = node_ancestry_indices[node]
            parent_lookup_list[node_idx] = ancestry[-1] if ancestry else -1
    parent_lookup = jnp.array(parent_lookup_list, dtype=jnp.int32)

    # Build node -> start index into the global flat params vector.
    node_param_starts_list = [0] * total_nodes
    for stage_nodes in stages_nodes:
        for node in stage_nodes:
            node_idx = node_to_idx[node]
            spec: ParamsSymbol = node.copula._params_symbol
            node_param_starts_list[node_idx] = int(spec.start)
    node_param_starts = jnp.array(node_param_starts_list, dtype=jnp.int32)

    # Step 4: Sample V values top-down
    k_v, k_leaves = jrandom.split(key)
    v_vals = _sample_v_values_topdown(
        stages_nodes,
        metadata,
        params,
        node_param_starts,
        parent_lookup,
        k_v,
        post_widder_k,
        max_cdf_x,
    )

    # Step 5: Transform leaves bottom-up (copula uniforms at leaves)
    U_leaves = _transform_leaves_bottomup(
        graph,
        stages_nodes,
        v_vals,
        node_to_idx,
        metadata["get_ancestry_chain"],
        params,
        node_param_starts,
        parent_lookup,
        k_leaves,
        total_nodes,
    )
    return _reconstruct_obs_from_marginals(
        graph=graph, params=params, U_leaves=U_leaves
    )


# =============================
# Utilities
# =============================


def _depth(node: Node) -> int:
    if not node.children:
        return 0
    child_depths = []
    for ch in node.children:
        if isinstance(ch, Leaf):
            child_depths.append(0)
        else:
            child_depths.append(_depth(ch))
    return 1 + (0 if not child_depths else max(child_depths))


def _kth_derivative(
    fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    k: int,
    scale: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute the k-th derivative of a scalar function at x using jet.

    Returns the k-th Taylor coefficient of fun(x + scale*t) at t=0.
    This corresponds to (fun^{(k)}(x) / k!) * scale^k.

    If scale is None, defaults to 1.0.
    """
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return fun(x)

    # Build series input for jet: provide coefficients for derivatives of the inner identity h(t)=x + scale*t
    # series_input[0] corresponds to h'(0) = scale.
    series_input = jnp.zeros(k)
    s = 1.0 if scale is None else scale
    series_input = series_input.at[0].set(s)

    _, series_out = jet_array.jet(fun, (x,), (series_input,))
    # jet returns Taylor coefficients: [f'(x)*s, f''(x)*s^2 / 2!, ..., f^{(k)}(x)*s^k / k!]
    return jnp.asarray(series_out[-1])


def _full_mixed_partials(
    outer_fun: Callable[[jax.Array], jax.Array],
    inner_fun: Callable[[jax.Array], jax.Array],
    u_vals: jax.Array,
    k: int,
    copula_instance: Optional["Copula"] = None,
) -> jax.Array:
    """Compute log of the mixed partial derivative d^k/du1...duk F(sum(G(u))).

    L = (-1)^k * F^{(k)}(S) * prod(G'(u_i))
    where S = sum(G(u_i)).

    If copula_instance provides log_generator_kth_derivative, uses that
    closed-form expression instead of jet for computing |F^{(k)}(S)|.
    """
    # 1. Compute inner values and sum (using ALL u_vals)
    g_vals = jax.vmap(inner_fun)(u_vals)
    S = jnp.sum(g_vals)

    if k == 0:
        return jnp.log(outer_fun(S))

    # 2. Compute derivatives for the first k components (uncensored)
    # Assumes u_vals is sorted such that first k are uncensored.
    u_uncensored = u_vals[:k]
    g_primes = jax.vmap(jax.grad(inner_fun))(u_uncensored)

    # Check if copula provides a closed-form k-th derivative
    custom_log_kth = None
    if copula_instance is not None:
        custom_log_kth = copula_instance.log_generator_kth_derivative(S, k)

    if custom_log_kth is not None:
        # Fast path: use closed-form log |ψ^{(k)}(S)| + sum log |g'(u_i)|
        log_abs_g_primes = jnp.sum(jnp.log(jnp.abs(g_primes)))
        T_k_log = custom_log_kth + log_abs_g_primes
        return T_k_log

    # 3. Default path: use jet with scaling for stability
    # We choose scaling such that scaling^k = prod(|g'|) * k!
    log_abs_g_primes = jnp.log(jnp.abs(g_primes))
    mean_log_g_prime = jnp.mean(log_abs_g_primes)

    log_factorial_term = jax.scipy.special.gammaln(k + 1) / k
    log_scaling = mean_log_g_prime + log_factorial_term
    scaling_factor = jnp.exp(log_scaling)

    # 4. Compute k-th Taylor coefficient of outer function with scaling
    # T_k = (F^{(k)}(S) / k!) * scaling^k
    #     = (F^{(k)}(S) / k!) * prod(|g'|) * k!
    #     = F^{(k)}(S) * prod(|g'|) = Likelihood!
    T_k = _kth_derivative(outer_fun, S, k, scale=scaling_factor)

    # 5. Assemble log-likelihood (mixed partial)
    log_derivative_term = jnp.log(T_k)
    jax.debug.print(
        "log_derivative_term: {} T_k: {} S: {} scaling_factor: {} g_primes: {}",
        log_derivative_term,
        T_k,
        S,
        scaling_factor,
        g_primes,
    )

    return log_derivative_term


def _invert_generator(phi: Callable[[jax.Array], jax.Array], t: jax.Array) -> jax.Array:
    """Inverse of φ using oryx.core.inverse; supports scalar or vector t."""
    t = jnp.asarray(t)
    phi_inv = oryx_core.inverse(phi)
    if t.ndim == 0:
        return phi_inv(t)
    return jax.vmap(phi_inv)(t)


def _fixed_talbot(
    phi: Callable[[jax.Array], jax.Array], M: int = 6
) -> Callable[[jax.Array], jax.Array]:
    """Fixed-point Talbot approximation to the inverse Laplace CDF with transform φ.

    Returns callable that approximates the inverse Laplace CDF with transform φ.
    """
    r = 2 * M / 5
    krange = jnp.arange(1, M)
    theta = krange * jnp.pi / M
    # cot(theta) = cos/sin is typically a bit more stable than 1/tan
    cot_theta = jnp.cos(theta) / jnp.sin(theta)
    # Following mpmath's FixedTalbot notation:
    # delta_k = r * theta_k * (cot(theta_k) + i),  k = 1..M-1
    # w_k     = 1 + i*theta_k*(1 + cot(theta_k)^2) - i*cot(theta_k), k = 1..M-1
    delta_rest = r * theta * (cot_theta + 1j)
    w_rest = 1 + 1j * theta * (1 + cot_theta**2) - 1j * cot_theta

    def wrapped(t: jax.Array, *, tmax: jax.Array | None = None) -> jax.Array:
        """Approximate inverse Laplace evaluation at time t.

        Args:
            t: time at which to evaluate.
            tmax: optional scaling time (mpmath's tmax). Defaults to t.
        """
        t = jnp.asarray(t)
        tmax_ = t if tmax is None else jnp.asarray(tmax)
        # Guard against division by 0; Talbot is defined for t>0.
        tiny = jnp.finfo(jnp.result_type(t, jnp.float64)).tiny
        t_safe = jnp.maximum(t, tiny)
        tmax_safe = jnp.maximum(tmax_, tiny)

        # Laplace parameters p_k = delta_k / tmax (mpmath uses tmax scaling).
        p0 = r / tmax_safe
        p_rest = delta_rest / tmax_safe
        p = jnp.concatenate([jnp.atleast_1d(p0), p_rest], axis=0)

        # Talbot weights gamma_k are independent of the Laplace-space evaluations.
        # Numerical stabilization: rescale exp(delta) by subtracting a common real shift.
        # This reduces overflow/underflow in exp(delta) (but cannot fully fix cancellation
        # error inherent to Talbot at large M in fixed precision).
        shift = jnp.max(
            jnp.concatenate([jnp.atleast_1d(jnp.array(r)), jnp.real(delta_rest)])
        )
        gamma0 = 0.5 * jnp.exp((r - shift))
        gamma_rest = jnp.exp(delta_rest - shift) * w_rest
        gamma = jnp.concatenate([jnp.atleast_1d(gamma0), gamma_rest], axis=0)

        # f(t) = (2/(5t)) * Re[ sum_k gamma_k * phi(p_k) ]
        terms = gamma * phi(p)
        s = jnp.sum(jnp.real(terms))
        out = (2 / (5 * t_safe)) * jnp.exp(shift) * s
        return out

    return wrapped


def _euler_ilt(
    phi: Callable[[jax.Array], jax.Array], M: int = 11
) -> Callable[[jax.Array], jax.Array]:
    """Euler (Abate-Whitt) inverse Laplace transform.

    Uses the Fourier-series representation with Euler acceleration
    (binomial averaging) to suppress Gibbs oscillations.  More stable
    than Talbot for discrete mixing distributions evaluated at
    half-integers.

    Args:
        phi: Laplace-domain function  F(s).
        M:   Euler acceleration order.  The method evaluates 2M+1 terms
             of the Fourier series.

    Returns:
        Callable that maps a real t > 0 to the approximate inverse.
    """
    n_terms = 2 * M + 1
    # Precompute binomial coefficients C(M, j) / 2^M
    bc = [0.0] * (M + 1)
    bc[0] = 1.0
    for j in range(1, M + 1):
        bc[j] = bc[j - 1] * (M - j + 1) / j
    denom = 2.0 ** M
    binom_coeffs = jnp.array([v / denom for v in bc])
    k_arr = jnp.arange(n_terms, dtype=jnp.float64)
    signs = jnp.where(k_arr % 2 == 0, 1.0, -1.0)
    # Abscissa shift (controls truncation error vs round-off)
    A = jnp.array(18.4, dtype=jnp.float64)

    def wrapped(t: jax.Array) -> jax.Array:
        tiny = jnp.finfo(jnp.float64).tiny
        t_safe = jnp.maximum(jnp.asarray(t, dtype=jnp.float64), tiny)
        c = A / (2.0 * t_safe)
        s = c + 1j * k_arr * jnp.pi / t_safe
        Fs = phi(s)
        a_k = jnp.real(Fs)
        terms = signs * a_k
        partial_sums = jnp.cumsum(terms)
        euler_avg = jnp.sum(binom_coeffs * partial_sums[M:])
        result = euler_avg - terms[0] / 2.0
        return jnp.exp(A / 2.0) / t_safe * result

    return wrapped


def _sample_frailty_via_talbot(
    key: jax.Array,
    psi: Callable[[jax.Array], jax.Array],
    max_cdf_x: float,
    discrete: bool = False,
    M: int = 12,
) -> jax.Array:
    """Sample from mixing distribution via ILT inverse-CDF.

    For continuous distributions, uses Fixed Talbot CDF + bisection.
    For discrete distributions, evaluates the Euler ILT CDF at half-integers
    to build a PMF, then samples via inverse-CDF on the discrete distribution.
    This avoids bisection over an oscillatory CDF.

    Args:
        key:        JAX PRNG key.
        psi:        Laplace transform (generator) of the mixing distribution.
        max_cdf_x:  Upper bracket for bisection (continuous) or max integer (discrete).
        discrete:   If True, use half-integer PMF sampling.
        M:          ILT order (Talbot for continuous, Euler for discrete).

    Returns:
        A single frailty sample.
    """
    u = jrandom.uniform(key, dtype=jnp.float64)

    # Continuous: Talbot CDF + bisection
    cdf_fn = _fixed_talbot(lambda s: psi(s) / s, M=M)

    lo = jnp.finfo(jnp.float64).tiny
    hi = jnp.array(max_cdf_x, dtype=jnp.float64)

    def cond_fn(state):
        lo_s, hi_s, n = state
        gap = hi_s - lo_s
        return (n < 150) & (gap > jnp.maximum(hi_s * 1e-10, 1e-15))

    def body_fn(state):
        lo_s, hi_s, n = state
        mid = lo_s + (hi_s - lo_s) / 2.0
        cdf_mid = jnp.clip(cdf_fn(mid), 0.0, 1.0)
        lo_s = jnp.where(cdf_mid < u, mid, lo_s)
        hi_s = jnp.where(cdf_mid >= u, mid, hi_s)
        return lo_s, hi_s, n + 1

    lo_f, hi_f, _ = lax.while_loop(cond_fn, body_fn, (lo, hi, jnp.array(0)))
    x = (lo_f + hi_f) / 2.0
    x = jnp.where(discrete, jnp.maximum(jnp.round(x), 1.0), x)
    return x


def _cohen(
    phi: Callable[[jax.Array], jax.Array], degree: int = 22
) -> Callable[[jax.Array], jax.Array]:
    """Cohen (CRVZ) inverse Laplace approximation.

    This follows mpmath's implementation of the Cohen algorithm:
    - Abscissa: p_k = alpha/(2t) + i*pi*k/t,  k=0..M-1 where M = degree+1
    - Result:
        f(t) = exp(alpha/2)/t * (A0/2 - s/d)
      where A_k = Re(F(p_k)) and s/d are coefficients from the CRVZ acceleration.

    Notes:
    - In mpmath, `alpha` is chosen based on the working precision. Here we mimic
      the same heuristic using a "dps_goal" derived from `degree`.
    - This is intended for smooth, non-oscillatory transforms. As with all
      numerical ILT methods, singularities to the right of the Bromwich contour
      will break accuracy.
    """

    n = int(degree)
    if n < 4:
        raise ValueError("degree must be >= 4 for Cohen inversion to be meaningful")

    # mpmath uses dps_goal ~ 1.5*degree when `degree` is specified.
    dps_goal = 1.5 * n
    log10 = jnp.log(jnp.array(10.0, dtype=jnp.float64))

    # Number of Laplace samples
    M = n + 1
    k = jnp.arange(M, dtype=jnp.float64)

    # Precompute constant for d
    three_plus_sqrt8 = jnp.array(3.0 + jnp.sqrt(8.0), dtype=jnp.float64)
    d_const = (three_plus_sqrt8**n + three_plus_sqrt8 ** (-n)) / 2.0

    def wrapped(t: jax.Array, *, alpha: jax.Array | None = None) -> jax.Array:
        t = jnp.asarray(t, dtype=jnp.float64)
        tiny = jnp.finfo(jnp.float64).tiny
        t_safe = jnp.maximum(t, tiny)

        # Heuristic alpha, mirroring mpmath:
        # alpha = (2/3) * (dps_goal*log(10) + log(2t))
        alpha_ = (
            (2.0 / 3.0) * (dps_goal * log10 + jnp.log(2.0 * t_safe))
            if alpha is None
            else jnp.asarray(alpha, dtype=jnp.float64)
        )

        a_t = alpha_ / (2.0 * t_safe)
        p_t = (jnp.pi * 1j) / t_safe
        p = a_t + k * p_t  # complex

        fp = phi(p)
        A = jnp.real(fp)  # vector length M

        # CRVZ acceleration recurrence (mpmath Cohen.calc_time_domain_solution)
        # b=-1; c=-d; s=0
        def step(carry, i_f):
            b, c, s = carry
            c_new = b - c
            i_i = i_f.astype(jnp.int32)
            s_new = s + c_new * A[i_i + 1]
            # i_f is float64 index in [0, n-1]
            b_new = 2.0 * (i_f + n) * (i_f - n) * b / ((2.0 * i_f + 1.0) * (i_f + 1.0))
            return (b_new, c_new, s_new), None

        b0 = jnp.array(-1.0, dtype=jnp.float64)
        c0 = -d_const
        s0 = jnp.array(0.0, dtype=jnp.float64)
        idx = jnp.arange(n, dtype=jnp.float64)
        (bN, cN, sN), _ = lax.scan(step, (b0, c0, s0), idx)
        s = sN

        core = (A[0] / 2.0) - (s / d_const)
        # f(t) = exp(alpha/2)/t * core
        out = jnp.exp(alpha_ / 2.0 - jnp.log(t_safe)) * core
        jax.debug.print(
            "out: {} alpha_: {} t_safe: {} core: {}", out, alpha_, t_safe, core
        )
        return out

    return wrapped


def _post_widder_cdf(
    psi: Callable[[jax.Array], jax.Array], x: jax.Array, k: int
) -> jax.Array:
    """Post–Widder approximation to the inverse Laplace CDF with transform ψ.

    F_k(x) = (-1)^k/k! * (k/x)^{k+1} * (ψ/s)^{(k)}(k/x).

    Note: We compute the derivative of ψ(s)/s, not just ψ(s).
    """
    # TODO: safe scaling?
    tiny = jnp.finfo(float).eps
    x_safe = jnp.maximum(x, tiny)
    t = k / x_safe

    # Compute k-th derivative of psi(s)/s at s=t
    psi_k = _kth_derivative(lambda s: psi(s) / s, t, k)

    sign = -1.0 if (k % 2 == 1) else 1.0
    log_coef = (k + 1) * jnp.log(t)
    log_fact = jax.scipy.special.gammaln(k + 1)
    out = sign * jnp.exp(log_coef - log_fact) * psi_k

    out = jnp.where(jnp.isnan(out), 0.0, out)
    out = jnp.clip(out, 0.0, 1.0)
    return jnp.where(x > 0, out, 0.0)


def _post_widder_log_pdf(
    phi: Callable[[jax.Array], jax.Array], v: jax.Array, k: int
) -> jax.Array:
    """Post-Widder approximation to the inverse Laplace Log-PDF.

    Given Laplace transform phi(t), approximates log f(v).
    Formula: f_k(v) = (-1)^k * (k/v)^{k+1} * (phi^{(k)}(k/v) / k!)

    We choose scale s such that s^k = (k/v)^{k+1} to absorb the power term
    into the jet computation for numerical stability.
    log(s) = (k+1)/k * log(k/v)
    """
    val = k / v
    log_scale = ((k + 1) / k) * jnp.log(val)
    scaling_factor = jnp.exp(log_scale)

    # phi_k_scaled = (phi^{(k)}(val) / k!) * scaling_factor^k
    #              = (phi^{(k)}(val) / k!) * (k/v)^{k+1} = f_k(v)
    phi_k_scaled = _kth_derivative(phi, val, k, scale=scaling_factor)

    # Sign check: phi is CM, (-1)^k phi^{(k)} >= 0. Abs value handles sign correctly.
    sign = -1.0 if (k % 2 == 1) else 1.0
    return jnp.log(phi_k_scaled * sign)


def _sample_frailty_via_post_widder(
    key: jax.Array,
    psi: Callable[[jax.Array], jax.Array],
    k: int,
    max_cdf_x: float,
) -> jax.Array:
    """Sample from mixing distribution with Laplace transform ψ via inverse-CDF.

    Uses Fixed Talbot CDF approximation directly, avoiding expensive
    integration.  Bisection runs in a manual ``lax.while_loop`` so that
    Talbot's small numerical excursions outside ``[0, 1]`` (which fire on
    discrete-supported frailties such as Frank/AMH/Joe) are clipped
    rather than raised — optimistix's ``Bisection`` checks the bracket at
    init time and raises before ``throw=False`` has a chance to suppress
    it, so we reimplement the loop directly.
    """
    u = jrandom.uniform(key, dtype=jnp.float64)

    cdf_fun = _fixed_talbot(lambda s: psi(s) / s, 6)

    lo = jnp.array(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    hi = jnp.array(max_cdf_x, dtype=jnp.float64)

    def cond_fn(state):
        lo_s, hi_s, n = state
        gap = hi_s - lo_s
        return (n < 150) & (gap > jnp.maximum(hi_s * 1e-10, 1e-15))

    def body_fn(state):
        lo_s, hi_s, n = state
        mid = lo_s + (hi_s - lo_s) / 2.0
        # Talbot inversion can return slightly outside [0, 1] on some
        # transforms; clip so monotone bisection still converges.
        cdf_mid = jnp.clip(cdf_fun(mid), 0.0, 1.0)
        lo_s = jnp.where(cdf_mid < u, mid, lo_s)
        hi_s = jnp.where(cdf_mid >= u, mid, hi_s)
        return lo_s, hi_s, n + 1

    lo_f, hi_f, _ = lax.while_loop(cond_fn, body_fn, (lo, hi, jnp.array(0)))
    return (lo_f + hi_f) / 2.0


# =============================
# Rosenblatt transform sampling
# =============================


def _rosenblatt_conditional_cdf(
    psi: Callable,
    t_prev: jax.Array,
    t_new: jax.Array,
    order: int,
) -> jax.Array:
    """Compute the conditional CDF for the j-th leaf (non-nested case).

    C_{j|1..j-1}(u_j) = psi^{(j-1)}(t_new) / psi^{(j-1)}(t_prev)

    where t_new = t_prev + psi^{-1}(u_j), and the derivatives are computed
    via jet (Taylor-mode AD).

    Args:
        psi: generator function
        t_prev: running sum of psi^{-1}(u_i) for i < j
        t_new: t_prev + psi^{-1}(u_j)
        order: derivative order (= j-1, the number of previously processed leaves)

    Returns:
        Conditional CDF value in [0, 1].
    """
    if order == 0:
        return psi(t_new)

    series_in = jnp.zeros(order).at[0].set(1.0)
    _, series_new = jet_array.jet(psi, (t_new,), (series_in,))
    _, series_old = jet_array.jet(psi, (t_prev,), (series_in,))
    # series[order-1] = psi^{(order)}(t) / order!
    return jnp.asarray(series_new[order - 1]) / jnp.asarray(series_old[order - 1])


def _rosenblatt_bisect_uj(
    psi: Callable,
    psi_inv: Callable,
    t_prev: jax.Array,
    v_j: jax.Array,
    order: int,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> jax.Array:
    """Find u_j such that C_{j|1..j-1}(u_j) = v_j via bisection (non-nested case).

    Uses lax.while_loop for JIT compatibility.
    """
    def cond_fn(state):
        lo, hi, _ = state
        return (hi - lo) > tol

    def body_fn(state):
        lo, hi, _ = state
        mid = (lo + hi) / 2.0
        t_new = t_prev + psi_inv(mid)
        cdf_val = _rosenblatt_conditional_cdf(psi, t_prev, t_new, order)
        lo = jnp.where(cdf_val < v_j, mid, lo)
        hi = jnp.where(cdf_val < v_j, hi, mid)
        return lo, hi, mid

    init = (jnp.float64(1e-14), jnp.float64(1.0 - 1e-14), jnp.float64(0.5))
    lo, hi, _ = lax.while_loop(cond_fn, body_fn, init)
    return (lo + hi) / 2.0


def _rosenblatt_h_taylor(
    parent_cop: "Copula",
    child_cop: "Copula",
    t_child: jax.Array,
    order: int,
) -> jax.Array:
    """Compute Taylor coefficients of h(t) = psi_parent^{-1}(psi_child(t)) at t_child.

    Returns array of length `order` where p[j] = h^{(j+1)}(t_child) / (j+1)!.
    """
    def h_vc(t):
        return parent_cop.generator_inv(child_cop.generator(t))

    series_in = jnp.zeros(order).at[0].set(1.0)
    _, series_out = jet_array.jet(h_vc, (t_child,), (series_in,))
    return jnp.asarray(series_out)


def _build_alpha_from_h(h_p: jax.Array, order: int) -> jax.Array:
    """Build alpha-coefficients from h-Taylor coefficients via polynomial powering.

    For a group child with `order` direct leaves (all uncensored), the internal
    beta is [0,...,0,1] at index `order`. The alpha formula from Faa di Bruno is:

        alpha[k] = order! * (p^{*k})[order] / k!

    where p_padded[0]=0, p_padded[j]=h_p[j-1] for j>=1.

    Args:
        h_p: h-Taylor coefficients of length >= order
        order: number of leaves processed (determines polynomial length)

    Returns:
        alpha array of length order+1, with alpha[0] = 0.
    """
    n = order + 1
    p_padded = jnp.zeros(n, dtype=jnp.float64)
    p_padded = p_padded.at[1:n].set(h_p[:order])

    # Precompute factorial ratio: order! / k!
    log_facts = jax.scipy.special.gammaln(jnp.arange(n, dtype=jnp.float64) + 1)
    log_order_fact = log_facts[order]

    alpha = jnp.zeros(n, dtype=jnp.float64)
    power = p_padded.copy()
    for k in range(1, n):
        fact_ratio = jnp.exp(log_order_fact - log_facts[k])
        alpha = alpha.at[k].set(power[order] * fact_ratio)
        if k < order:
            power = jnp.convolve(power, p_padded)[:n]

    return alpha


def _build_alpha_general(h_p: jax.Array, child_beta: jax.Array, d_c: int) -> jax.Array:
    """Build alpha-coefficients from h-Taylor coefficients and arbitrary child beta.

    Generalization of _build_alpha_from_h for arbitrary-depth nesting.
    Uses the same formula as bell.py's _poly_power_alpha:

        alpha[k] = (1/k!) * sum_j (j! * child_beta[j]) * (p^{*k})[j]

    Args:
        h_p: h-Taylor coefficients (jet output), length >= d_c
        child_beta: beta polynomial of the child subtree, length d_c + 1
        d_c: number of uncensored leaves in the child subtree

    Returns:
        alpha array of length d_c + 1, with alpha[0] = 0.
    """
    n = d_c + 1
    p_padded = jnp.zeros(n, dtype=jnp.float64)
    p_padded = p_padded.at[1:n].set(h_p[:d_c])

    # beta_tilde[j] = j! * child_beta[j], inv_factorials[k] = 1/k!
    log_facts = jax.scipy.special.gammaln(jnp.arange(n, dtype=jnp.float64) + 1)
    beta_tilde = child_beta[:n] * jnp.exp(log_facts)
    inv_factorials = jnp.exp(-log_facts)

    alpha = jnp.zeros(n, dtype=jnp.float64)
    power = p_padded.copy()  # p^{*1}
    for k in range(1, n):
        # alpha[k] = dot(beta_tilde, power) / k!
        alpha_k = jnp.dot(beta_tilde, power) * inv_factorials[k]
        alpha = alpha.at[k].set(alpha_k)
        if k < d_c:
            power = jnp.convolve(power, p_padded)[:n]

    return alpha


def _cauchy_product_simple(
    alpha_list: list,
    total_order: int,
) -> jax.Array:
    """Combine alpha-polynomials via truncated convolution (Cauchy product).

    Args:
        alpha_list: list of alpha arrays (each with alpha[0] = 0 for uncensored)
        total_order: total derivative order

    Returns:
        Combined beta array of length total_order + 1.
    """
    n = total_order + 1
    if len(alpha_list) == 0:
        return jnp.array([1.0])

    result = jnp.zeros(n, dtype=jnp.float64)
    a0 = alpha_list[0]
    result = result.at[:min(len(a0), n)].set(a0[:min(len(a0), n)])

    for a in alpha_list[1:]:
        a_padded = jnp.zeros(n, dtype=jnp.float64)
        a_padded = a_padded.at[:min(len(a), n)].set(a[:min(len(a), n)])
        result = jnp.convolve(result, a_padded)[:n]

    return result


def _rosenblatt_assembly(
    psi: Callable,
    S: jax.Array,
    beta: jax.Array,
    total_order: int,
) -> jax.Array:
    """Compute sum_k beta[k] * psi^{(k)}(S) for k=0..total_order.

    Uses jet Taylor coefficients (taylor[k] = psi^{(k)}(S)/k!) and multiplies
    by k! to recover the actual derivative. This matches bell.py's convention
    where alpha includes the d_c!/k! factor.
    """
    if total_order == 0:
        return psi(S) * beta[0]

    series_in = jnp.zeros(total_order).at[0].set(1.0)
    primal, series = jet_array.jet(psi, (S,), (series_in,))

    log_facts = jax.scipy.special.gammaln(
        jnp.arange(total_order + 1, dtype=jnp.float64) + 1
    )

    result = beta[0] * primal
    for k in range(1, total_order + 1):
        k_fact = jnp.exp(log_facts[k])
        result = result + beta[k] * jnp.asarray(series[k - 1]) * k_fact
    return result


def _sample_once_rosenblatt(
    graph: Node,
    key: jax.Array,
    params: jax.Array,
) -> jax.Array:
    """Sample one observation using the Rosenblatt transform.

    For non-nested copulas, uses the efficient jet-based Rosenblatt.
    For nested copulas (any depth), uses the Bell polynomial-based approach
    that correctly computes conditional CDFs via Faa di Bruno.

    The Bell polynomial approach works by maintaining state at each internal
    node (running generator-inverse sum, derivative order, completed child
    alphas) and computing the root-level assembly ratio for each leaf's
    conditional CDF. The alpha at each node is computed recursively using
    the child subtree's beta polynomial and h-Taylor coefficients.
    """
    leaves = _collect_leaves_in_order(graph)
    num_leaves = len(leaves)
    leaf_id_to_idx = {id(lf): i for i, lf in enumerate(leaves)}

    U_leaves = jnp.zeros(num_leaves, dtype=jnp.float64)

    # Check if this is a non-nested copula (all children are leaves)
    all_leaves_direct = all(isinstance(ch, Leaf) for ch in graph.children)

    if all_leaves_direct:
        # Non-nested: use the efficient jet-based Rosenblatt
        cop = _instantiate_copula_from_flat(graph, params)
        psi = cop.generator
        psi_inv = cop.generator_inv
        m = len(graph.children)

        v_arr = jrandom.uniform(key, shape=(m,), dtype=jnp.float64)

        w = jnp.zeros(m, dtype=jnp.float64)
        w = w.at[0].set(v_arr[0])
        t_prev = psi_inv(v_arr[0])

        for j_idx in range(1, m):
            w_j = _rosenblatt_bisect_uj(psi, psi_inv, t_prev, v_arr[j_idx], j_idx)
            w = w.at[j_idx].set(w_j)
            t_prev = t_prev + psi_inv(w_j)

        for i, ch in enumerate(graph.children):
            leaf_idx = leaf_id_to_idx[id(ch)]
            U_leaves = U_leaves.at[leaf_idx].set(w[i])

        return _reconstruct_obs_from_marginals(
            graph=graph, params=params, U_leaves=U_leaves
        )

    # ---- Nested copula: Bell polynomial-based Rosenblatt (arbitrary depth) ----
    # State per internal node: {node_id: {t, order, prior_alphas, cop}}
    node_state = {}
    node_copulas = {}

    def _init_node(node):
        cop = _instantiate_copula_from_flat(node, params)
        node_copulas[id(node)] = cop
        node_state[id(node)] = {
            't': jnp.array(0.0, dtype=jnp.float64),
            'order': 0,
            'prior_alphas': [],
        }
        for ch in node.children:
            if isinstance(ch, Node):
                _init_node(ch)

    _init_node(graph)

    # Build parent map: child_id -> parent Node
    parent_map = {}

    def _build_parent_map(node):
        for ch in node.children:
            if isinstance(ch, Node):
                parent_map[id(ch)] = node
                _build_parent_map(ch)

    _build_parent_map(graph)

    root_cop = node_copulas[id(graph)]
    root_psi = root_cop.generator

    # Pre-split keys
    all_keys = jrandom.split(key, num_leaves)
    global_order = [0]  # mutable counter

    def _process_subtree(node):
        """Process all leaves in a subtree in DFS order, sampling each."""
        nonlocal U_leaves

        for child in node.children:
            if isinstance(child, Leaf):
                leaf_idx = leaf_id_to_idx[id(child)]
                v_j = jrandom.uniform(all_keys[leaf_idx], dtype=jnp.float64)

                if global_order[0] == 0:
                    u_j = v_j
                else:
                    u_j = _bisect_for_leaf(child, node, v_j)

                U_leaves = U_leaves.at[leaf_idx].set(u_j)

                # Update immediate parent state
                ns = node_state[id(node)]
                cop = node_copulas[id(node)]
                ns['t'] = ns['t'] + cop.generator_inv(u_j)
                ns['order'] += 1
                global_order[0] += 1

                # Add trivial alpha for this leaf to parent's prior_alphas
                ns['prior_alphas'].append(jnp.array([0.0, 1.0]))

            elif isinstance(child, Node):
                # Recursively process subtree
                _process_subtree(child)

                # After completing child subtree, compute its alpha and add
                # to parent's state
                ns = node_state[id(node)]
                cop = node_copulas[id(node)]
                child_cop = node_copulas[id(child)]
                child_ns = node_state[id(child)]
                m_child = child_ns['order']  # total leaves processed in child

                if m_child == 0:
                    # Empty subtree — skip
                    continue

                # Compute child's final alpha at parent level
                t_child = child_ns['t']
                h_p = _rosenblatt_h_taylor(cop, child_cop, t_child, m_child)
                child_beta = _cauchy_product_simple(
                    child_ns['prior_alphas'], m_child
                )
                alpha = _build_alpha_general(h_p, child_beta, m_child)

                # Update parent: add h(t_child) to parent's t-sum and
                # add child's leaf count to parent's order
                ns['t'] = ns['t'] + cop.generator_inv(child_cop.generator(t_child))
                ns['order'] += m_child
                ns['prior_alphas'].append(alpha)

    def _bisect_for_leaf(leaf, parent_node, v_j):
        """Find u_j via bisection using the Bell polynomial conditional CDF.

        The conditional CDF is:
            F(u_j | u_1,...,u_{j-1}) = assembly_new / assembly_old

        where assembly = sum_k beta_root[k] * psi_root^{(k)}(S_root)
        and the beta/S change when we include leaf j.

        Both denominator and numerator trace from leaf_parent up to root,
        differing only in whether the candidate leaf is included.
        """
        par_ns = node_state[id(parent_node)]
        par_cop = node_copulas[id(parent_node)]

        # --- Denominator: current state at parent_node, trace to root ---
        # The conditional CDF F(u_j | u_1,...,u_{j-1}) is the ratio of the
        # (j-1)-th mixed partial at two evaluation points: with u_j included
        # vs u_j = 1. Both use the same total_order = go (number of previously
        # processed leaves). The candidate leaf only changes the evaluation
        # point, not the derivative order.
        den_beta = _cauchy_product_simple(par_ns['prior_alphas'], par_ns['order'])
        den = _trace_assembly(
            parent_node, par_ns['t'], par_ns['order'], den_beta
        )

        # --- Bisection ---
        def _cond(state):
            lo, hi, _ = state
            return (hi - lo) > 1e-10

        def _body(state):
            lo, hi, _ = state
            mid = (lo + hi) / 2.0

            # Numerator: same total_order but with modified evaluation point.
            # Adding the candidate leaf changes t at parent_node (and thus
            # the h-Taylor coefficients and alphas at all ancestor levels).
            # But it does NOT change the derivative order at any level.
            new_t = par_ns['t'] + par_cop.generator_inv(mid)
            num = _trace_assembly(
                parent_node, new_t, par_ns['order'], den_beta
            )
            cdf_val = num / den
            lo = jnp.where(cdf_val < v_j, mid, lo)
            hi = jnp.where(cdf_val < v_j, hi, mid)
            return lo, hi, mid

        init = (jnp.float64(1e-14), jnp.float64(1.0 - 1e-14),
                jnp.float64(0.5))
        lo_f, hi_f, _ = lax.while_loop(_cond, _body, init)
        return (lo_f + hi_f) / 2.0

    def _trace_assembly(start_node, start_t, start_order, start_beta):
        """Compute root assembly by tracing from start_node up to root.

        Given a starting node with specified (t, order, beta), traces up
        through all ancestors, computing alphas and t-sums at each level,
        until reaching the root where the final assembly is computed.

        This is used for both denominator (current state) and numerator
        (state + candidate leaf). The caller prepares the starting state
        at the leaf's immediate parent.

        The total_order at each level is computed by accumulating the
        child's order with the parent's order (from completed children).

        Args:
            start_node: the node where we begin tracing (leaf's parent)
            start_t: t-sum at start_node
            start_order: derivative order at start_node
            start_beta: beta polynomial at start_node

        Returns:
            Assembly value at root.
        """
        if start_node is graph:
            # start_node IS the root — direct assembly
            return _rosenblatt_assembly(root_psi, start_t, start_beta, start_order)

        # Trace from start_node up to root
        child_t = start_t
        child_order = start_order
        child_beta = start_beta
        current = start_node

        while current is not graph:
            par = parent_map[id(current)]
            par_cop = node_copulas[id(par)]
            child_cop = node_copulas[id(current)]
            par_ns = node_state[id(par)]

            # Compute alpha for current node at parent level
            if child_order == 0:
                # No processed leaves in this subtree — identity alpha
                alpha_active = jnp.array([1.0])
            else:
                h_p = _rosenblatt_h_taylor(par_cop, child_cop, child_t, child_order)
                alpha_active = _build_alpha_general(h_p, child_beta, child_order)

            # Combine with parent's prior_alphas
            combined_alphas = par_ns['prior_alphas'] + [alpha_active]
            new_order_par = par_ns['order'] + child_order
            new_beta_par = _cauchy_product_simple(combined_alphas, new_order_par)

            # Compute new t at parent level
            new_t_par = par_ns['t'] + par_cop.generator_inv(
                child_cop.generator(child_t)
            )

            # Move up
            child_t = new_t_par
            child_order = new_order_par
            child_beta = new_beta_par
            current = par

        # current is now root — child_order is the total derivative order
        return _rosenblatt_assembly(root_psi, child_t, child_beta, child_order)

    # Run the recursive sampling
    _process_subtree(graph)

    return _reconstruct_obs_from_marginals(
        graph=graph, params=params, U_leaves=U_leaves
    )


# =============================
# Visualization helpers (NetworkX)
# =============================
