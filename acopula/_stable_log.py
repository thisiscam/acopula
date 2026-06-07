"""Numerically stable log for higher-order AD through very small inputs.

Problem
-------
The Bell-polynomial copula density pipeline computes ``log|p[j]|`` where
``p[j]`` is the ``j``-th Taylor coefficient of a composition.  For analytic
compositions at large arguments (e.g. nested Clayton with one uniform margin
at the Kaplan-Meier floor), these coefficients span hundreds of orders of
magnitude; the smallest legitimately reach ~1e-200.  Their log is a normal
float, but the *second* derivative of ``log|p|`` via JAX's default chain rule
produces the intermediate ``1/p^2``, which overflows float64 once
``|p| < 2e-154``.  Even though the true second derivative of
``log|p(theta)|`` with respect to a parameter ``theta`` is bounded
(because the relative derivatives ``p'/p`` and ``p''/p`` are O(1)), JAX's
native AD does not see the cancellation and returns NaN.

Fix
---
We introduce ``_log_deriv(x, y) = y / x`` as a :class:`jax.extend.core.Primitive`
with explicit forward, JVP, and transpose rules.  The JVP rule restructures
the second-order derivative as

    d/dtheta (y/x) = (y_dot/x) - (y/x) * (x_dot/x)
                   = _log_deriv(x, y_dot) - primal * _log_deriv(x, x_dot)

Every intermediate here is a bounded *relative* derivative; ``1/x^2`` is
never materialized.  The transpose rule stays inside the same primitive, so
``jax.hessian`` (= ``jacfwd(jacrev)``) and ``jacrev(jacrev)`` both produce
finite values.

The patched :func:`safe_log` wraps ``jnp.log`` with a :func:`jax.custom_jvp`
rule whose tangent goes through ``_log_deriv``.  Drop-in replacement for
``jnp.log`` wherever the input may be very close to zero.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
import jax._src.core as _jc

ShapedArray = _jc.ShapedArray

__all__ = ["safe_log", "log_deriv"]


# ---------------------------------------------------------------------------
# _log_deriv primitive: y / x with higher-order-stable derivatives
# ---------------------------------------------------------------------------

log_deriv_p = Primitive("acopula_log_deriv")


def log_deriv(x, y):
    """Compute ``y / x`` as a primitive with stable higher-order AD.

    Linear in ``y`` (with ``x`` held as a non-linear input).  Forward value
    is identical to plain division.  Differs from ``jnp.divide`` only in
    its JVP / transpose rules, which avoid materialising ``1/x^2`` in the
    second-order computation.
    """
    return log_deriv_p.bind(x, y)


# Eager implementation
log_deriv_p.def_impl(lambda x, y: y / x)


# Abstract eval (tracing / JIT)
def _log_deriv_abstract_eval(x, y):
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    return ShapedArray(shape, x.dtype)


log_deriv_p.def_abstract_eval(_log_deriv_abstract_eval)


# MLIR lowering: identical to plain y/x at the XLA level
mlir.register_lowering(
    log_deriv_p,
    mlir.lower_fun(lambda x, y: y / x, multiple_results=False),
)


# JVP rule
#   d/dθ (y/x) = y_dot/x - (y/x) * (x_dot/x)
#              = log_deriv(x, y_dot) - primal * log_deriv(x, x_dot)
# Every intermediate is a bounded ratio; no 1/x^2 is ever formed.
def _log_deriv_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    # Materialise symbolic zeros (JAX's ad.Zero) so arithmetic works below.
    if type(x_dot) is ad.Zero:
        x_dot = jnp.zeros_like(x)
    if type(y_dot) is ad.Zero:
        y_dot = jnp.zeros_like(y)
    primal = log_deriv(x, y)
    tangent = log_deriv(x, y_dot) - primal * log_deriv(x, x_dot)
    return primal, tangent


ad.primitive_jvps[log_deriv_p] = _log_deriv_jvp


# Transpose rule (for reverse-mode AD).  The primitive is linear in y, with
# x held as a residual.  Given output cotangent ``ct``, the transpose yields
# ``(None, ct / x)`` -- expressed via log_deriv again so jacfwd through the
# backward pass also stays stable.
def _log_deriv_transpose(ct, x, y):
    if ad.is_undefined_primal(y):
        return None, log_deriv(x, ct)
    raise NotImplementedError("log_deriv is only linear in its second argument")


ad.primitive_transposes[log_deriv_p] = _log_deriv_transpose


# Batching rule (for vmap)
def _log_deriv_batching(batched_args, batch_dims):
    size = next(
        (arg.shape[bdim] for arg, bdim in zip(batched_args, batch_dims) if bdim is not None),
        None,
    )
    if size is None:
        return log_deriv(*batched_args), None
    x, y = batched_args
    x_bdim, y_bdim = batch_dims
    x_bc = batching.bdim_at_front(x, x_bdim, size)
    y_bc = batching.bdim_at_front(y, y_bdim, size)
    return log_deriv(x_bc, y_bc), 0


batching.primitive_batchers[log_deriv_p] = _log_deriv_batching


# ---------------------------------------------------------------------------
# safe_log: drop-in jnp.log replacement with higher-order-stable AD
# ---------------------------------------------------------------------------

@jax.custom_jvp
def safe_log(x):
    """``jnp.log(x)`` with a higher-order-stable JVP rule.

    Use this instead of ``jnp.log`` whenever the argument may legitimately
    reach values below ``~2e-154`` AND the caller wants the second (or
    higher) derivative of the result with respect to an upstream parameter.

    Forward value is identical to ``jnp.log``.  First derivative is
    identical.  Second and higher derivatives go through :func:`log_deriv`,
    which avoids the ``1/x^2`` overflow that causes JAX's default rule to
    produce NaN for tiny ``x``.
    """
    return jnp.log(x)


# First-order reciprocal floor.  ``log``'s first derivative is ``1/x``, which
# overflows float64 once ``|x| < 1 / max_float ~ 5.6e-309``.  XLA:GPU *preserves*
# subnormals while XLA:CPU flushes them to exactly 0, so the unclamped
# ``x_dot / x`` produces ``inf`` -> NaN on GPU even though the forward ``log(x)``
# is finite.  We set the floor to the smallest *normal* float (``finfo.tiny``):
# below it a value is subnormal, which is exactly the range XLA:CPU flushes to 0
# (and routes through the ``where(abs>0, .., _LOG_TINY)`` guard to a zero
# tangent).  Clamping at ``tiny`` therefore makes the GPU reproduce the CPU
# gradient bit-for-bit, while every *normal* coefficient (``>= tiny``, including
# the ~1e-200 range the second-order Hessian path uses) is kept and differentiated
# exactly -- its ``1/x <= 1/tiny ~ 4.5e307`` never overflows.  We clamp the
# *denominator* so the reciprocal is never formed for subnormal ``x``.
import numpy as _np
_RECIP_FLOOR = float(_np.finfo(_np.float64).tiny)  # 2.2250738585072014e-308


@safe_log.defjvp
def _safe_log_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    if type(x_dot) is ad.Zero:
        x_dot = jnp.zeros_like(x)
    # Forward value is exact log(x) (finite even for subnormal x); only the
    # first-order tangent's 1/x is clamped to avoid overflow.
    safe = jnp.abs(x) >= _RECIP_FLOOR
    x_clamped = jnp.where(safe, x, jnp.ones_like(x))
    tangent = jnp.where(safe, log_deriv(x_clamped, x_dot), jnp.zeros_like(x_dot))
    return jnp.log(x), tangent
