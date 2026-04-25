"""Bell polynomial approach (Approach B) for nested Archimedean copula density.

Computes the nested copula log-likelihood without frailty integration,
using generator derivatives, Bell polynomials (via polynomial powering),
and Cauchy products in a bottom-up tree traversal.

Uses the scheduling infrastructure (schedule.py) to process nodes in
stage order rather than via Python recursion, producing a shallow XLA
graph that LLVM can compile even at high nesting depths (d=8+).

Works for ALL copula families (including AMH, Frank, Joe with discrete mixing).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

import jet_array
from ._stable_log import safe_log
from .core import (
    Copula,
    Leaf,
    Node,
    _collect_leaves_in_order,
    _group_children_by_structure,
    _instantiate_copula_from_flat,
    _marginal_transforms_chunked,
    optimize_order,
)
from .schedule import solve_scheduling_nodes


# ---------------------------------------------------------------------------
# Higher-order-stable log option
# ---------------------------------------------------------------------------
#
# The Bell-polynomial density pipeline takes ``jnp.log`` of Taylor coefficients
# that, for nested compositions at extreme arguments (e.g. Clayton at the
# Kaplan-Meier floor), can legitimately reach ~1e-200.  The log is fine, but
# JAX's default second-derivative rule materialises ``-1/x**2``, which
# overflows float64 once ``|x| < 2e-154``.  The true
# ``d**2 log|x(theta)|/dtheta**2`` is bounded because the relative derivatives
# ``x'/x`` and ``x''/x`` remain O(1); only the naive chain rule blows up.
#
# Opting into :func:`set_stable_log` swaps the three log sites in
# :func:`_poly_power_alpha` for :func:`acopula._stable_log.safe_log`, whose
# JVP rule restructures the higher-order derivatives to avoid the overflow.
# Forward value and first derivative are unchanged; only second and higher
# derivatives differ.  Default is ``False`` for backward compatibility.

_USE_STABLE_LOG = False


def set_stable_log(enabled: bool) -> None:
    """Enable higher-order-stable ``log`` for the Bell polynomial density.

    When ``enabled=True``, :func:`_poly_power_alpha` routes its three ``jnp.log``
    calls through :func:`acopula._stable_log.safe_log`, which uses a custom JVP
    rule to keep ``jax.hessian`` / ``jacfwd(jacrev)`` finite for Taylor
    coefficients that reach ~1e-200 (e.g. nested Clayton at Kaplan-Meier
    pseudo-observations near the marginal floor).  Forward value and first
    derivative are unchanged.

    Default is ``False``.  Call once at startup, before constructing the model.
    """
    global _USE_STABLE_LOG
    _USE_STABLE_LOG = bool(enabled)


def _log(x):
    """Dispatch to ``safe_log`` or ``jnp.log`` based on the module toggle."""
    if _USE_STABLE_LOG:
        return safe_log(x)
    return jnp.log(x)


# ---------------------------------------------------------------------------
# Data structures for the bottom-up pass
# ---------------------------------------------------------------------------

_SQRT_TINY = jnp.sqrt(jnp.finfo(jnp.float64).tiny)  # ~1.49e-154


def _normalize_poly(arr: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Normalize polynomial coefficients to prevent underflow/overflow.

    Returns (normalized_arr, log_scale) where arr = exp(log_scale) * normalized_arr.

    The floor on safe_max prevents NaN gradients: the quotient rule for
    arr/safe_max has safe_max**2 in the denominator, so when
    safe_max < sqrt(tiny) the square underflows to zero and the gradient
    becomes 0/0 = NaN.  Clamping to sqrt(tiny) keeps safe_max**2 >= tiny.
    """
    max_abs = jnp.max(jnp.abs(arr))
    safe_max = jnp.maximum(max_abs, _SQRT_TINY)
    log_scale = _log(safe_max)
    return arr / safe_max, log_scale


@dataclass
class _ChildInfo:
    """Per-child data computed during the tree-value pass."""
    child: Union[Node, Leaf]
    C_child: jax.Array         # copula value (u_l for leaf, psi_c(t_c) for node)
    psi_inv_C: jax.Array       # psi_parent^{-1}(C_child) -- contribution to parent's t_v
    d_child: int               # uncensored leaf descendants (static), or total leaves (dynamic)
    is_leaf: bool
    is_censored: bool          # only meaningful for leaves (static path)
    leaf_idx: int = -1         # index of this leaf in the global leaf ordering (dynamic path)
    # For internal children, the recursive result:
    t_child: Optional[jax.Array] = None  # t_c (only for internal children)


@dataclass
class _NodeInfo:
    """Per-node data computed during the tree-value pass."""
    t_v: jax.Array             # sum of psi_v^{-1}(C_child) over children
    C_v: jax.Array             # psi_v(t_v)
    d_v: int                   # uncensored leaf descendants (static), or total leaves (dynamic)
    children_info: List[_ChildInfo]
    copula_instance: Copula    # instantiated copula for this node


# ---------------------------------------------------------------------------
# Schedule computation for bell polynomial bottom-up traversal
# ---------------------------------------------------------------------------

def _count_uncensored_leaves(node: Union[Node, Leaf]) -> int:
    """Count uncensored leaves in a subtree."""
    if isinstance(node, Leaf):
        return 0 if getattr(node, "censored", False) else 1
    return sum(_count_uncensored_leaves(c) for c in node.children)


def _count_total_leaves(node: Union[Node, Leaf]) -> int:
    """Count ALL leaves in a subtree (censored and uncensored)."""
    if isinstance(node, Leaf):
        return 1
    return sum(_count_total_leaves(c) for c in node.children)


def _build_bell_schedule(graph: Node) -> List[Node]:
    """Build a bottom-up processing order for internal nodes using the scheduler.

    Returns a list of Node objects in bottom-up order: children before parents.
    Nodes whose dependencies are all satisfied appear earlier in the list.
    """
    # Build parent map (needed for ancestry key)
    node_parent: Dict[int, Optional[Node]] = {}

    def build_parent_map(n: Node, p: Optional[Node]):
        node_parent[id(n)] = p
        for ch in n.children:
            if isinstance(ch, Node):
                build_parent_map(ch, n)

    build_parent_map(graph, None)

    # Color function: use the copula class name as the color (same as sampling)
    def color_fn(node: Node):
        return type(node.copula).__name__

    # Solve scheduling -- returns stages in bottom-up order (leaves first)
    result = solve_scheduling_nodes(graph, color_fn)
    stages: List[List[Node]] = result["stages"]

    # Flatten stages into a single bottom-up ordered list
    bottom_up_order: List[Node] = []
    for stage in stages:
        bottom_up_order.extend(stage)

    return bottom_up_order


# ---------------------------------------------------------------------------
# Step 1: Bottom-up tree value computation (stage-based, non-recursive)
# ---------------------------------------------------------------------------

def _compute_tree_values_staged(
    graph: Node,
    u_vec: jax.Array,
    leaf_indices: Dict[int, int],
    params_flat: jax.Array,
    bottom_up_order: List[Node],
    dynamic_censoring: bool = False,
) -> Dict[int, _NodeInfo]:
    """Bottom-up pass computing t_v, C_v, d_v at each internal node.

    Processes nodes in schedule order (bottom-up) rather than via recursion.
    This produces a shallow XLA graph: each node's computation reads from
    arrays rather than composing nested function calls.

    Args:
        graph: root Node of the copula tree
        u_vec: array of marginal CDF values (copula-scale) for all leaves
        leaf_indices: mapping from id(leaf) -> index into u_vec
        params_flat: flat parameter vector for copula instantiation
        bottom_up_order: nodes in bottom-up processing order from scheduler
        dynamic_censoring: if True, d_child/d_v count ALL leaves (not just
            uncensored).  Censoring is decided at runtime via censored_mask.

    Returns:
        Dict mapping id(node) -> _NodeInfo
    """
    info: Dict[int, _NodeInfo] = {}

    for node in bottom_up_order:
        cop = _instantiate_copula_from_flat(node, params_flat)
        children_info = []
        psi_inv_sum = jnp.array(0.0)

        for child in node.children:
            if isinstance(child, Leaf):
                u_val = u_vec[leaf_indices[id(child)]]
                psi_inv_val = cop.generator_inv(u_val)
                psi_inv_sum = psi_inv_sum + psi_inv_val
                if dynamic_censoring:
                    children_info.append(_ChildInfo(
                        child=child, C_child=u_val, psi_inv_C=psi_inv_val,
                        d_child=1,  # always 1 in dynamic mode
                        is_leaf=True, is_censored=child.censored,
                        leaf_idx=leaf_indices[id(child)],
                    ))
                else:
                    children_info.append(_ChildInfo(
                        child=child, C_child=u_val, psi_inv_C=psi_inv_val,
                        d_child=0 if child.censored else 1,
                        is_leaf=True, is_censored=child.censored,
                    ))
            else:
                # Internal child -- already processed in an earlier stage
                child_info = info[id(child)]
                C_child = child_info.C_v
                psi_inv_val = cop.generator_inv(C_child)
                psi_inv_sum = psi_inv_sum + psi_inv_val
                children_info.append(_ChildInfo(
                    child=child, C_child=C_child, psi_inv_C=psi_inv_val,
                    d_child=child_info.d_v, is_leaf=False, is_censored=False,
                    t_child=child_info.t_v,
                ))

        t_v = psi_inv_sum
        C_v = cop.generator(t_v)
        d_v = sum(ci.d_child for ci in children_info)

        info[id(node)] = _NodeInfo(
            t_v=t_v, C_v=C_v, d_v=d_v,
            children_info=children_info,
            copula_instance=cop,
        )

    return info


# ---------------------------------------------------------------------------
# Step 2: Polynomial powering (Bell polynomial step)
# ---------------------------------------------------------------------------

def _truncated_convolve(a: jax.Array, b: jax.Array) -> jax.Array:
    """Convolve two equal-length 1D arrays, truncated to that length.

    Uses lax.conv_general_dilated for XLA compatibility (no Python loops).
    Both inputs must have the same length (as guaranteed by all callers
    which pre-pad to a common size).
    """
    n = a.shape[0]
    # 1D convolution: flip b, slide over a padded
    a_pad = jnp.pad(a, (n - 1, 0))
    out = lax.conv_general_dilated(
        a_pad[None, None, :],
        b[::-1][None, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCW", "IOW", "NCW"),
    )[0, 0]
    return out[:n]


def _convolve_short_kernel(a: jax.Array, b: jax.Array) -> jax.Array:
    """Convolve array a (length n) with shorter kernel b (length m <= n).

    Returns truncated polynomial product of length n. The kernel b has
    length m, so the convolution costs O(n*m) instead of O(n^2).
    """
    n = a.shape[0]
    m = b.shape[0]
    a_pad = jnp.pad(a, (m - 1, 0))
    out = lax.conv_general_dilated(
        a_pad[None, None, :],
        b[::-1][None, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCW", "IOW", "NCW"),
    )[0, 0]
    return out[:n]


def _poly_power_alpha(
    p: jax.Array,
    beta: jax.Array,
    d_c: int,
) -> jax.Array:
    """Compute per-child alpha-coefficients via polynomial powering.

    Per-index rescaling removes the geometric decay: p_hat[j-1] = p[j-1] / p[0]^j
    so P_hat(z) = P(z/p0).  Then [z^j] P(z)^k = p0^j * [z^j] P_hat(z)^k.

    The dot product alpha[k] = (1/k!) * sum_j (j! * beta[j] * p0^j) * q_hat[j]
    is computed in log domain (signed logsumexp) to handle the p0^j factor
    that spans too many orders of magnitude for float64.

    Uses lax.scan for the polynomial powering recurrence.

    Args:
        p: array of length d_c, where p[j-1] = h^{(j)}(t_c) / j!
        beta: array of length d_c + 1
        d_c: number of uncensored leaves in child subtree

    Returns:
        (alpha, log_scale) where alpha is normalized (max |alpha| ~ 1) and
        alpha_actual = exp(log_scale) * alpha.
    """
    n = d_c + 1
    _LOG_TINY = jnp.log(jnp.finfo(jnp.float64).tiny)  # ~-708

    # Per-index rescaling: p_hat[j-1] = p[j-1] / p[0]^j
    # This gives P_hat(z) = P(z/p0), removing geometric decay.
    p0 = p[0]
    abs_p0 = jnp.abs(p0)
    safe_abs_p0 = jnp.where(abs_p0 > 0, abs_p0, 1.0)
    log_abs_p0 = jnp.where(abs_p0 > 0, _log(safe_abs_p0), _LOG_TINY)
    sign_p0 = jnp.sign(p0)

    js = jnp.arange(1, d_c + 1, dtype=jnp.float64)
    abs_p = jnp.abs(p[:d_c])
    safe_abs_p = jnp.where(abs_p > 0, abs_p, 1.0)
    log_abs_p = jnp.where(abs_p > 0, _log(safe_abs_p), _LOG_TINY)
    log_rescaled = log_abs_p - js * log_abs_p0
    p_hat = jnp.sign(p[:d_c]) * (sign_p0 ** js) * jnp.exp(
        jnp.clip(log_rescaled, -500, 500))
    # Zero out entries where original p was zero
    p_hat = jnp.where(abs_p > 0, p_hat, 0.0)

    p_hat_padded = jnp.zeros(n).at[1:n].set(p_hat)

    # Polynomial powering: q_hat_k = P_hat^k via scan
    # [z^j] P(z)^k = p0^j * q_hat_k[j]
    # Collect all q_hat arrays (need them for the log-domain dot product)
    def body(carry, _k):
        q_hat = carry
        q_hat_next = _truncated_convolve(q_hat, p_hat_padded)
        return q_hat_next, q_hat_next  # emit P_hat^{_k+2}

    final_q, q_hat_scanned = lax.scan(body, p_hat_padded, jnp.arange(d_c - 1))
    # q_hat_scanned[i] = P_hat^{i+2} for i=0..d_c-2
    # Prepend P_hat^1 = p_hat_padded
    q_hat_all = jnp.concatenate([p_hat_padded[None, :], q_hat_scanned], axis=0)
    # q_hat_all[k-1] = P_hat^k for k=1..d_c  (shape: d_c x n)

    # Log-domain dot product for each k:
    # alpha[k] = (1/k!) * sum_j  [j! * beta[j]] * [p0^j * q_hat_k[j]]
    #          = (1/k!) * sum_j  sign_term * exp(log_abs_term)
    # where log_abs_term = gammaln(j+1) + log|beta[j]| + j*log|p0| + log|q_hat_k[j]|
    js_full = jnp.arange(n, dtype=jnp.float64)
    log_facts = jax.scipy.special.gammaln(js_full + 1)
    abs_beta = jnp.abs(beta)
    safe_abs_beta = jnp.where(abs_beta > 0, abs_beta, 1.0)
    log_abs_beta = jnp.where(abs_beta > 0, _log(safe_abs_beta), _LOG_TINY)
    sign_beta = jnp.sign(beta)
    # Per-j log weight: gammaln(j+1) + log|beta[j]| + j*log|p0|
    log_w = log_facts + log_abs_beta + js_full * log_abs_p0
    sign_w = sign_beta * (sign_p0 ** js_full)
    # Mask: beta[j] must be nonzero and j > 0 for meaningful contribution
    w_valid = (beta != 0.0) & (js_full > 0)

    def compute_alpha_k(k_minus_1):
        """Compute alpha[k] for k = k_minus_1 + 1."""
        k = k_minus_1 + 1
        q_hat_k = q_hat_all[k_minus_1]  # P_hat^k, shape (n,)

        abs_q = jnp.abs(q_hat_k)
        safe_abs_q = jnp.where(abs_q > 0, abs_q, 1.0)
        log_abs_q = jnp.where(abs_q > 0, _log(safe_abs_q), _LOG_TINY)
        sign_q = jnp.sign(q_hat_k)

        # log|term_j| = log_w[j] + log|q_hat_k[j]|
        log_abs_term = log_w + log_abs_q
        sign_term = sign_w * sign_q
        # Mask invalid terms
        valid = w_valid & (q_hat_k != 0.0)
        log_abs_term = jnp.where(valid, log_abs_term, _LOG_TINY)

        # Signed logsumexp
        max_log = jnp.max(log_abs_term)
        sum_signed = jnp.sum(
            jnp.where(valid, sign_term * jnp.exp(log_abs_term - max_log), 0.0))

        # alpha[k] = sum / k!
        abs_sum = jnp.abs(sum_signed)
        safe_abs_sum = jnp.where(abs_sum > 0, abs_sum, 1.0)
        log_abs_alpha_k = jnp.where(
            abs_sum > 0,
            max_log + _log(safe_abs_sum) - log_facts[k],
            _LOG_TINY)
        sign_alpha_k = jnp.sign(sum_signed)

        return log_abs_alpha_k, sign_alpha_k

    log_abs_alpha_arr, sign_alpha_arr = jax.vmap(compute_alpha_k)(jnp.arange(d_c))
    # These are for k=1..d_c

    # Find global reference scale and build normalized alpha
    ref_log = jnp.max(log_abs_alpha_arr)
    alpha_values = sign_alpha_arr * jnp.exp(
        jnp.clip(log_abs_alpha_arr - ref_log, -500, 500))

    alpha = jnp.zeros(n).at[1:n].set(alpha_values)

    # d_c-monotone penalty: (-1)^m * p[m] should be >= 0 for m = 0,...,d_c-1
    # p[m] = h^{(m+1)}(t_c) / (m+1)!, sign of p[m] = sign of h^{(m+1)}
    # d_c-monotone requires (-1)^m h^{(m+1)} >= 0, i.e. (-1)^m p[m] >= 0
    # Use modular arithmetic: even indices keep sign, odd indices flip
    signed_p = jnp.where(jnp.arange(d_c) % 2 == 0, p, -p)
    nesting_pen = jnp.sum(jnp.minimum(signed_p, 0.0) ** 2)

    return alpha, ref_log, nesting_pen


# ---------------------------------------------------------------------------
# Step 3: Per-child alpha computation
# ---------------------------------------------------------------------------

def _alpha_for_child(
    parent_cop: Copula,
    ci: _ChildInfo,
    child_beta: Optional[jax.Array],
    params_flat: jax.Array,
    effective_order=None,
) -> jax.Array:
    """Compute alpha-coefficients for one child of a parent node.

    Returns (alpha, log_scale, nesting_penalty).

    Args:
        effective_order: optional JAX scalar. When provided, jet only
            computes Taylor coefficients up to this order.
    """
    if ci.is_leaf:
        if ci.is_censored:
            return jnp.array([1.0]), jnp.array(0.0), jnp.array(0.0)
        else:
            return jnp.array([0.0, 1.0]), jnp.array(0.0), jnp.array(0.0)

    # Internal child: compute node composition Taylor coefficients via jet
    child_node = ci.child
    child_cop = _instantiate_copula_from_flat(child_node, params_flat)
    d_c = ci.d_child

    if d_c == 0:
        return jnp.array([1.0]), jnp.array(0.0), jnp.array(0.0)

    # Compute Taylor coefficients of h(t) = psi_parent^{-1}(psi_child(t)) at t_child
    from .compose import compute_composition_taylor
    p = compute_composition_taylor(parent_cop, child_cop, ci.t_child, d_c,
                                   effective_order=effective_order)
    return _poly_power_alpha(p, child_beta, d_c)  # returns (alpha, log_scale, nesting_pen)


def _pad_to(arr: jax.Array, length: int) -> jax.Array:
    """Pad array with zeros to target length."""
    if len(arr) >= length:
        return arr[:length]
    return jnp.concatenate([arr, jnp.zeros(length - len(arr))])


def _collect_leaf_indices_for_subtree(node: Union[Node, Leaf],
                                      leaf_indices: Dict[int, int]) -> List[int]:
    """Collect global leaf indices for all leaves in a subtree."""
    if isinstance(node, Leaf):
        return [leaf_indices[id(node)]]
    result = []
    for c in node.children:
        result.extend(_collect_leaf_indices_for_subtree(c, leaf_indices))
    return result


# ---------------------------------------------------------------------------
# Step 4: Cauchy product
# ---------------------------------------------------------------------------

def _cauchy_product(alpha_list: List[jax.Array], d_total: int) -> jax.Array:
    """Combine alpha-polynomials from multiple children via polynomial multiplication.

    Each alpha_list[i] has alpha[0] = 0 (for uncensored children) or alpha[0] = 1
    (for censored-only children). The convolution naturally enforces k_i >= 1
    since the z^0 coefficient is 0 for uncensored children.

    The carry (running product) has length d_total+1, but each alpha kernel is
    padded only to max(len(alpha_i)), not d_total+1.  Each scan step convolves
    the long carry with the short kernel via _convolve_short_kernel, reducing
    the cost from O(M * d_total^2) to O(M * d_total * max_child_degree).

    Args:
        alpha_list: list of per-child alpha arrays
        d_total: total uncensored dimension at this node

    Returns:
        beta: array of length d_total + 1
    """
    if len(alpha_list) == 0:
        return jnp.array([1.0]), jnp.array(0.0)

    n = d_total + 1

    if len(alpha_list) == 1:
        return _pad_to(alpha_list[0], n), jnp.array(0.0)

    # Pad first alpha to carry length (d_total + 1)
    init = jnp.zeros(n).at[:len(alpha_list[0])].set(alpha_list[0])

    # Pad remaining alphas to common kernel length (max child degree + 1)
    max_kernel_len = max(len(a) for a in alpha_list[1:])
    kernels = jnp.zeros((len(alpha_list) - 1, max_kernel_len))
    for i, a in enumerate(alpha_list[1:]):
        kernels = kernels.at[i, :len(a)].set(a)

    # Sequential convolution via lax.scan with intermediate normalization
    # to prevent overflow when multiplying many alpha vectors.
    # carry = (result, log_scale)
    def body(carry, alpha_row):
        result, ls = carry
        conv = _convolve_short_kernel(result, alpha_row)
        conv_norm, step_ls = _normalize_poly(conv)
        return (conv_norm, ls + step_ls), None

    (result, total_log_scale), _ = lax.scan(body, (init, jnp.array(0.0)), kernels)
    return result, total_log_scale


def _cauchy_product_dynamic(
    alpha_stack: jax.Array,
    item_types: jax.Array,
    d_total: int,
) -> jax.Array:
    """Cauchy product with dynamic dispatch via lax.switch.

    All alpha polynomials are pre-padded to the same static shape (d_total+1),
    so the scan body has fixed shapes. Each item is dispatched by type:
      0 = censored leaf  -> identity (beta unchanged)
      1 = uncensored leaf -> shift (multiply by z)
      2 = child subtree  -> full truncated convolution

    Args:
        alpha_stack: (n_items, d_total+1) padded alpha polynomials
        item_types: (n_items,) int array -- 0=censored, 1=uncensored, 2=child
        d_total: total number of leaves (static)

    Returns:
        beta: array of length d_total + 1
    """
    n = d_total + 1
    init_beta = jnp.zeros(n).at[0].set(1.0)

    def body(carry, item):
        beta, ls = carry
        alpha = item[0]
        itype = item[1]

        def _identity(b):
            return b

        def _shift(b):
            # Multiply polynomial by z: shift coefficients right by 1
            return jnp.concatenate([jnp.zeros(1), b[:-1]])

        def _convolve(b):
            return _truncated_convolve(b, alpha)

        new_beta = lax.switch(itype, [_identity, _shift, _convolve], beta)
        # Normalize to prevent overflow during sequential convolutions
        new_beta_norm, step_ls = _normalize_poly(new_beta)
        return (new_beta_norm, ls + step_ls), None

    (beta, total_log_scale), _ = lax.scan(
        body, (init_beta, jnp.array(0.0)), (alpha_stack, item_types))
    return beta, total_log_scale


# ---------------------------------------------------------------------------
# Step 5: Stage-based beta computation (replaces recursive version)
# ---------------------------------------------------------------------------

def _compute_beta_for_node(
    node: Node,
    node_info_dict: Dict[int, _NodeInfo],
    node_beta_dict: Dict[int, jax.Array],
    node_log_scale_dict: Dict[int, jax.Array],
    params_flat: jax.Array,
    censored_mask: Optional[jax.Array] = None,
    leaf_indices: Optional[Dict[int, int]] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute beta-coefficients for a single node.

    All child betas must already be in node_beta_dict.
    Groups same-structure children and vmaps the jet + polynomial powering
    over each group for efficiency.

    When censored_mask is provided (dynamic censoring), all polynomials are
    padded to d_v+1 and the Cauchy product uses lax.switch dispatch so that
    XLA shapes are static regardless of censoring pattern.

    Returns (beta, log_scale) where beta is normalized and log_scale tracks
    the cumulative normalization factor.
    """
    ni = node_info_dict[id(node)]
    d_v = ni.d_v
    dynamic = censored_mask is not None

    if d_v == 0:
        return jnp.array([1.0]), jnp.array(0.0), jnp.array(0.0)

    # Collect child betas from the dict (already computed in earlier stages)
    child_betas = {}  # child index -> beta array
    for idx, ci in enumerate(ni.children_info):
        if not ci.is_leaf:
            child_betas[idx] = node_beta_dict[id(ci.child)]

    # Group children by structure for batching
    groups = _group_children_by_structure(node)

    n = d_v + 1  # static polynomial length for this node

    if dynamic:
        # --- Dynamic censoring path ---
        alpha_list_padded = [None] * len(ni.children_info)
        item_types_list = [None] * len(ni.children_info)
        cumulative_log_scale = jnp.array(0.0)
        cumulative_nesting_penalty = jnp.array(0.0)

        for struct_key, child_indices in groups.items():
            first_ci = ni.children_info[child_indices[0]]

            if first_ci.is_leaf:
                for idx in child_indices:
                    ci = ni.children_info[idx]
                    is_cens = censored_mask[ci.leaf_idx]
                    # Both branches produce shape (n,)
                    alpha_unc = jnp.zeros(n).at[1].set(1.0)   # [0, 1, 0, ...]
                    alpha_cens = jnp.zeros(n).at[0].set(1.0)  # [1, 0, 0, ...]
                    alpha_list_padded[idx] = jnp.where(is_cens, alpha_cens, alpha_unc)
                    # item_type: 0 if censored, 1 if uncensored -- decided at
                    # trace time via jnp.where so the scan body uses lax.switch
                    item_types_list[idx] = jnp.where(is_cens, 0, 1).astype(jnp.int32)

            elif len(child_indices) == 1:
                idx = child_indices[0]
                ci = ni.children_info[idx]
                # Compute effective_order for this child subtree
                child_leaf_idxs = _collect_leaf_indices_for_subtree(
                    ci.child, leaf_indices)
                eff_order = jnp.sum(
                    ~censored_mask[jnp.array(child_leaf_idxs, dtype=jnp.int32)])
                alpha_norm, poly_log_scale, nest_pen = _alpha_for_child(
                    ni.copula_instance, ci, child_betas[idx], params_flat,
                    effective_order=eff_order,
                )
                # When eff_order == 0 (child subtree fully censored) the
                # child's beta = [1, 0, ...] (identity), but _poly_power_alpha
                # then returns alpha = [0, 0, ...] because its (j>0) mask
                # filters out beta[0].  Override alpha to the identity
                # polynomial [1, 0, ..., 0] so the Cauchy product convolve
                # branch treats this child as a no-op.  Also zero out the
                # log-scale and nesting penalty contributions.  Mirrors the
                # static path short-circuit at _alpha_for_child line 416.
                is_zero_eff = eff_order == 0
                alpha_padded = _pad_to(alpha_norm, n)
                identity_padded = jnp.zeros(n).at[0].set(1.0)
                alpha_list_padded[idx] = jnp.where(
                    is_zero_eff, identity_padded, alpha_padded)
                poly_log_scale = jnp.where(is_zero_eff, 0.0, poly_log_scale)
                nest_pen = jnp.where(is_zero_eff, 0.0, nest_pen)
                child_log_scale = node_log_scale_dict.get(id(ci.child), jnp.array(0.0))
                cumulative_log_scale = cumulative_log_scale + child_log_scale + poly_log_scale
                cumulative_nesting_penalty = cumulative_nesting_penalty + nest_pen
                item_types_list[idx] = jnp.int32(2)

            else:
                # Multiple same-structure internal children -- vmap
                d_c = first_ci.d_child

                t_c_batch = jnp.stack([
                    ni.children_info[idx].t_child for idx in child_indices
                ])
                beta_batch = jnp.stack([
                    _pad_to(child_betas[idx], d_c + 1) for idx in child_indices
                ])

                rep_child_node = ni.children_info[child_indices[0]].child
                child_param_starts = jnp.array([
                    int(ni.children_info[idx].child.copula._params_symbol.start)
                    for idx in child_indices
                ], dtype=jnp.int32)
                child_param_size = int(rep_child_node.copula._params_symbol.size)
                child_cop_cls = type(rep_child_node.copula)
                child_unravel = rep_child_node.copula._params_symbol.unravel_fn

                parent_cop = ni.copula_instance

                # Compute per-child effective_order
                child_eff_orders = []
                for idx in child_indices:
                    ci_inner = ni.children_info[idx]
                    child_leaf_idxs = _collect_leaf_indices_for_subtree(
                        ci_inner.child, leaf_indices)
                    eff = jnp.sum(
                        ~censored_mask[jnp.array(child_leaf_idxs, dtype=jnp.int32)])
                    child_eff_orders.append(eff)
                eff_order_batch = jnp.stack(child_eff_orders)

                def single_alpha_dyn(t_c, beta, param_start, eff_order):
                    flat_slice = jax.lax.dynamic_slice(
                        params_flat, (param_start,), (child_param_size,)
                    )
                    child_params = child_unravel(flat_slice)
                    child_cop = child_cop_cls(**child_params)

                    from .compose import compute_composition_taylor
                    p = compute_composition_taylor(
                        parent_cop, child_cop, t_c, d_c,
                        effective_order=eff_order)
                    return _poly_power_alpha(p, beta, d_c)

                alpha_batch, poly_ls_batch, nest_pen_batch = jax.vmap(single_alpha_dyn)(
                    t_c_batch, beta_batch, child_param_starts, eff_order_batch
                )

                # Per-child eff_order==0 masking: when a child subtree is
                # fully censored, _poly_power_alpha returns alpha = [0, ..., 0]
                # because child_beta = [1, 0, ...] fails the (j>0) mask.
                # Override alpha to the identity polynomial [1, 0, ..., 0]
                # so the Cauchy product convolve branch treats this child as
                # a no-op (convolving with [1, 0, ...] is identity).
                is_zero_batch = eff_order_batch == 0
                _alpha_dim = alpha_batch.shape[-1]
                identity_alpha = jnp.zeros(_alpha_dim).at[0].set(1.0)
                alpha_batch = jnp.where(
                    is_zero_batch[:, None],
                    identity_alpha[None, :],
                    alpha_batch,
                )
                poly_ls_batch = jnp.where(is_zero_batch, 0.0, poly_ls_batch)
                nest_pen_batch = jnp.where(is_zero_batch, 0.0, nest_pen_batch)

                for i, idx in enumerate(child_indices):
                    alpha_list_padded[idx] = _pad_to(alpha_batch[i], n)
                    item_types_list[idx] = jnp.int32(2)
                    child_log_scale = node_log_scale_dict.get(
                        id(ni.children_info[idx].child), jnp.array(0.0))
                    cumulative_log_scale = cumulative_log_scale + child_log_scale + poly_ls_batch[i]
                cumulative_nesting_penalty = cumulative_nesting_penalty + jnp.sum(nest_pen_batch)

        alpha_stack = jnp.stack(alpha_list_padded)
        item_types = jnp.stack(item_types_list)
        beta_raw, cauchy_log_scale = _cauchy_product_dynamic(alpha_stack, item_types, d_v)
        beta_norm, beta_log_scale = _normalize_poly(beta_raw)
        total_log_scale = cumulative_log_scale + cauchy_log_scale + beta_log_scale
        return beta_norm, total_log_scale, cumulative_nesting_penalty

    else:
        # --- Static censoring path (original) ---
        alpha_list = [None] * len(ni.children_info)
        cumulative_nesting_penalty = jnp.array(0.0)
        cumulative_log_scale = jnp.array(0.0)

        for struct_key, child_indices in groups.items():
            first_ci = ni.children_info[child_indices[0]]

            if first_ci.is_leaf:
                # Leaf group -- trivial alpha, no normalization needed
                for idx in child_indices:
                    ci = ni.children_info[idx]
                    if ci.is_censored:
                        alpha_list[idx] = jnp.array([1.0])
                    else:
                        alpha_list[idx] = jnp.array([0.0, 1.0])

            elif len(child_indices) == 1:
                # Single internal child -- no batching benefit
                idx = child_indices[0]
                ci = ni.children_info[idx]
                alpha_norm, poly_log_scale, nest_pen = _alpha_for_child(
                    ni.copula_instance, ci, child_betas[idx], params_flat
                )
                child_log_scale = node_log_scale_dict.get(id(ci.child), jnp.array(0.0))
                cumulative_log_scale = cumulative_log_scale + child_log_scale + poly_log_scale
                cumulative_nesting_penalty = cumulative_nesting_penalty + nest_pen
                alpha_list[idx] = alpha_norm

            else:
                # Multiple same-structure internal children -- vmap!
                d_c = first_ci.d_child
                if d_c == 0:
                    for idx in child_indices:
                        alpha_list[idx] = jnp.array([1.0])
                    continue

                # Stack t_c and beta arrays for the batch
                t_c_batch = jnp.stack([
                    ni.children_info[idx].t_child for idx in child_indices
                ])
                beta_batch = jnp.stack([
                    _pad_to(child_betas[idx], d_c + 1) for idx in child_indices
                ])

                # Stack child param starts for per-child copula instantiation
                rep_child_node = ni.children_info[child_indices[0]].child
                child_param_starts = jnp.array([
                    int(ni.children_info[idx].child.copula._params_symbol.start)
                    for idx in child_indices
                ], dtype=jnp.int32)
                child_param_size = int(rep_child_node.copula._params_symbol.size)
                child_cop_cls = type(rep_child_node.copula)
                child_unravel = rep_child_node.copula._params_symbol.unravel_fn

                parent_cop = ni.copula_instance

                def single_alpha(t_c, beta, param_start):
                    flat_slice = jax.lax.dynamic_slice(
                        params_flat, (param_start,), (child_param_size,)
                    )
                    child_params = child_unravel(flat_slice)
                    child_cop = child_cop_cls(**child_params)

                    from .compose import compute_composition_taylor
                    p = compute_composition_taylor(
                        parent_cop, child_cop, t_c, d_c)
                    return _poly_power_alpha(p, beta, d_c)

                alpha_batch, poly_ls_batch, nest_pen_batch = jax.vmap(single_alpha)(
                    t_c_batch, beta_batch, child_param_starts
                )

                for i, idx in enumerate(child_indices):
                    alpha_list[idx] = alpha_batch[i]
                    child_log_scale = node_log_scale_dict.get(
                        id(ni.children_info[idx].child), jnp.array(0.0))
                    cumulative_log_scale = cumulative_log_scale + child_log_scale + poly_ls_batch[i]
                cumulative_nesting_penalty = cumulative_nesting_penalty + jnp.sum(nest_pen_batch)

        beta_raw, cauchy_log_scale = _cauchy_product(alpha_list, d_v)
        beta_norm, beta_log_scale = _normalize_poly(beta_raw)
        total_log_scale = cumulative_log_scale + cauchy_log_scale + beta_log_scale
        return beta_norm, total_log_scale, cumulative_nesting_penalty


def _compute_beta_staged(
    bottom_up_order: List[Node],
    node_info_dict: Dict[int, _NodeInfo],
    params_flat: jax.Array,
    censored_mask: Optional[jax.Array] = None,
    leaf_indices: Optional[Dict[int, int]] = None,
) -> Dict[int, jax.Array]:
    """Bottom-up beta computation using stage-based scheduling.

    Processes nodes in bottom-up order so that all children's betas are
    available when computing a parent's beta. This replaces the recursive
    _compute_beta_recursive, producing a shallow XLA graph that LLVM can
    compile even at high nesting depths.

    The key difference from recursion: each node's computation reads child
    betas from a dictionary (simple value lookups) rather than through
    nested function compositions. This prevents the JAX tracer from
    building a deeply nested expression tree.

    Args:
        bottom_up_order: nodes ordered bottom-up from the scheduler
        node_info_dict: per-node info (t_v, children, etc.) from tree-value pass
        params_flat: flat parameter vector
        censored_mask: optional (d,) bool array for dynamic censoring
        leaf_indices: mapping from leaf id to global index (needed for effective_order)

    Returns:
        Tuple of (Dict mapping id(node) -> beta array,
                  Dict mapping id(node) -> log_scale scalar)
    """
    node_beta_dict: Dict[int, jax.Array] = {}
    node_log_scale_dict: Dict[int, jax.Array] = {}
    total_nesting_penalty = jnp.array(0.0)

    for node in bottom_up_order:
        beta, log_scale, nest_pen = _compute_beta_for_node(
            node, node_info_dict, node_beta_dict, node_log_scale_dict,
            params_flat,
            censored_mask=censored_mask,
            leaf_indices=leaf_indices,
        )
        node_beta_dict[id(node)] = beta
        node_log_scale_dict[id(node)] = log_scale
        total_nesting_penalty = total_nesting_penalty + nest_pen

    return node_beta_dict, node_log_scale_dict, total_nesting_penalty


# ---------------------------------------------------------------------------
# Step 6: Root assembly + leaf product
# ---------------------------------------------------------------------------

def _root_assembly(
    beta: jax.Array,
    root_cop: Copula,
    t_r: jax.Array,
    d: int,
    effective_order=None,
) -> jax.Array:
    """Compute log|sum_k beta[k] * psi_r^{(k)}(t_r)|.

    Uses jet to get all root generator Taylor coefficients in one call.

    Args:
        effective_order: optional JAX scalar. When provided, jet only
            computes Taylor coefficients up to this order; higher entries
            are left as zero.  Saves O((d/d_unc)^2) work when many
            leaves are censored.  When None, computes all d coefficients.
    """
    if d == 0:
        return _log(jnp.abs(root_cop.generator(t_r)))

    # Get Taylor coefficients of psi_r at t_r (jet returns primal + series)
    series_in = jnp.zeros(d).at[0].set(1.0)
    primal_out, series_out = jet_array.jet(
        root_cop.generator, (t_r,), (series_in,),
        effective_order=effective_order,
    )
    # series_out[k-1] = psi_r^{(k)}(t_r) / k!

    # Work in log domain to avoid overflow for large d.
    # We need log|sum_k beta[k] * psi_r^{(k)}(t_r)| where
    # psi_r^{(k)}(t_r) = taylor_with_primal[k] * k!
    ks = jnp.arange(d + 1, dtype=jnp.float64)
    log_factorials = jax.scipy.special.gammaln(ks + 1)
    taylor_with_primal = jnp.concatenate([
        jnp.array([primal_out]),
        jnp.asarray(series_out),
    ])

    beta_slice = beta[:d + 1]
    twp_slice = taylor_with_primal[:d + 1]

    # Mask out zero terms to avoid log(0) = -inf gradient issues
    nonzero = (beta_slice != 0) & (twp_slice != 0)
    safe_beta = jnp.where(nonzero, jnp.abs(beta_slice), 1.0)
    safe_twp = jnp.where(nonzero, jnp.abs(twp_slice), 1.0)

    # log|term_k| = log|beta[k]| + log|taylor[k]| + gammaln(k+1)
    log_abs_terms = _log(safe_beta) + _log(safe_twp) + log_factorials
    _LOG_TINY_LOCAL = jnp.log(jnp.finfo(jnp.float64).tiny)
    # Zero out masked terms so they contribute nothing to logsumexp
    log_abs_terms = jnp.where(nonzero, log_abs_terms, _LOG_TINY_LOCAL)
    signs = jnp.sign(beta_slice) * jnp.sign(twp_slice)

    # Signed log-sum-exp: shift by max for numerical stability
    max_log = jnp.max(log_abs_terms)
    sum_signed = jnp.sum(signs * jnp.exp(log_abs_terms - max_log))
    safe_abs_sum = jnp.maximum(jnp.abs(sum_signed), jnp.finfo(jnp.float64).tiny)
    return max_log + _log(safe_abs_sum)


def _leaf_log_product(
    graph: Node,
    u_vec: jax.Array,
    leaf_indices: Dict[int, int],
    node_info_dict: Dict[int, _NodeInfo],
    bottom_up_order: List[Node],
    censored_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute sum of log|(psi_{parent}^{-1})'(u_l)| over uncensored leaves.

    Groups leaves by parent copula and vmaps the gradient computation.
    Uses the bottom-up order to iterate without recursion.

    When censored_mask is provided, computes the gradient for ALL leaves
    and masks out censored contributions with jnp.where (JAX-traceable).
    """
    if censored_mask is not None:
        # --- Dynamic censoring path ---
        # Compute log|psi_inv'(u)| for ALL leaves, then mask.
        # Group ALL leaf children by parent copula for vmapping.
        groups: Dict[int, Tuple[Copula, List[jax.Array], List[int]]] = {}

        for node in bottom_up_order:
            ni = node_info_dict[id(node)]
            cop = ni.copula_instance
            for ci in ni.children_info:
                if ci.is_leaf:
                    u_val = u_vec[leaf_indices[id(ci.child)]]
                    cop_id = id(cop)
                    if cop_id not in groups:
                        groups[cop_id] = (cop, [], [])
                    groups[cop_id][1].append(u_val)
                    groups[cop_id][2].append(ci.leaf_idx)

        log_sum = jnp.array(0.0)
        for cop_id, (cop, u_vals, leaf_idxs) in groups.items():
            u_batch = jnp.stack(u_vals)
            grad_vals = jax.vmap(jax.grad(cop.generator_inv))(u_batch)
            log_derivs = _log(jnp.abs(grad_vals))

            # Mask: only include uncensored leaves
            leaf_idx_arr = jnp.array(leaf_idxs, dtype=jnp.int32)
            mask_vals = censored_mask[leaf_idx_arr]
            log_sum = log_sum + jnp.where(~mask_vals, log_derivs, 0.0).sum()

        return log_sum

    else:
        # --- Static censoring path (original) ---
        groups_static: Dict[int, Tuple[Copula, List[jax.Array]]] = {}

        for node in bottom_up_order:
            ni = node_info_dict[id(node)]
            cop = ni.copula_instance
            for ci in ni.children_info:
                if ci.is_leaf and not ci.is_censored:
                    u_val = u_vec[leaf_indices[id(ci.child)]]
                    cop_id = id(cop)
                    if cop_id not in groups_static:
                        groups_static[cop_id] = (cop, [])
                    groups_static[cop_id][1].append(u_val)

        log_sum = jnp.array(0.0)
        for cop_id, (cop, u_vals) in groups_static.items():
            if len(u_vals) == 1:
                grad_val = jax.grad(cop.generator_inv)(u_vals[0])
                log_sum = log_sum + _log(jnp.abs(grad_val))
            else:
                u_batch = jnp.stack(u_vals)
                grad_vals = jax.vmap(jax.grad(cop.generator_inv))(u_batch)
                log_sum = log_sum + jnp.sum(_log(jnp.abs(grad_vals)))

        return log_sum


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _log_likelihood_bell(
    graph: Node,
    obs: jax.Array,
    params_flat: jax.Array,
    censored_mask: Optional[jax.Array] = None,
    survival: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Log-likelihood for nested copulas using Bell polynomial approach.

    This eliminates the frailty integral entirely, working for all copula
    families including those with discrete mixing distributions.

    Uses schedule-based bottom-up traversal instead of Python recursion,
    producing a shallow XLA graph that LLVM can compile at high nesting
    depths (d=8+).

    Args:
        graph: root Node of the copula tree
        obs: observation array
        params_flat: flat parameter vector
        censored_mask: optional (d,) bool array where True = censored.
            When provided, enables dynamic censoring: all polynomial
            shapes are static (padded to d_total+1), and censoring
            decisions use lax.switch inside lax.scan so that one XLA
            compilation handles all censoring patterns.  When None,
            falls back to the original static censoring via Leaf.censored.

    Returns:
        (cdf_value, log_likelihood)
    """
    dynamic = censored_mask is not None

    # 1. Marginal transforms (reuse existing infrastructure)
    graph = optimize_order(graph)
    leaves_in_order = _collect_leaves_in_order(graph)

    u_vec, log_lik_vec = _marginal_transforms_chunked(
        leaves_in_order=leaves_in_order,
        params=params_flat,
        obs=obs,
        survival=survival,
    )

    # Sum marginal log-likelihoods for uncensored leaves
    if dynamic:
        # Dynamic: mask via JAX array ops for traceability
        marginal_ll = jnp.where(~censored_mask, log_lik_vec, 0.0).sum()
    else:
        # Static: Python-level loop (original behavior)
        marginal_ll = 0.0
        for i, lf in enumerate(leaves_in_order):
            if not getattr(lf, "censored", False):
                marginal_ll += log_lik_vec[i]

    # Build leaf id -> index mapping
    leaf_indices = {id(lf): i for i, lf in enumerate(leaves_in_order)}

    # 2. Compute bottom-up schedule (nodes ordered children-before-parents)
    bottom_up_order = _build_bell_schedule(graph)

    # 3. Bottom-up tree values (stage-based, non-recursive)
    node_info_dict = _compute_tree_values_staged(
        graph, u_vec, leaf_indices, params_flat, bottom_up_order,
        dynamic_censoring=dynamic,
    )
    root_info = node_info_dict[id(graph)]

    # 4. Bottom-up beta coefficients (stage-based, non-recursive)
    node_beta_dict, node_log_scale_dict, nesting_penalty = _compute_beta_staged(
        bottom_up_order, node_info_dict, params_flat,
        censored_mask=censored_mask,
        leaf_indices=leaf_indices if dynamic else None,
    )
    beta = node_beta_dict[id(graph)]
    root_log_scale = node_log_scale_dict[id(graph)]

    # 5. Root assembly: log|sum_k beta[k] * psi_r^{(k)}(t_r)|
    root_eff_order = None
    if dynamic:
        # Dynamic: jet only needs d_uncensored coefficients; the rest are
        # multiplied by zero betas and contribute nothing.
        root_eff_order = jnp.sum(~censored_mask).astype(jnp.int32)
    log_root = _root_assembly(
        beta, root_info.copula_instance, root_info.t_v, root_info.d_v,
        effective_order=root_eff_order,
    ) + root_log_scale

    # 6. Leaf product: sum log|(psi_parent^{-1})'(u_l)|
    log_leaves = _leaf_log_product(
        graph, u_vec, leaf_indices, node_info_dict, bottom_up_order,
        censored_mask=censored_mask,
    )

    # 7. Combine
    log_lik = log_root + log_leaves + marginal_ll
    cdf_val = root_info.C_v

    return cdf_val, log_lik, nesting_penalty
