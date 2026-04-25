"""Register an ILDJ rule for ``lax.copy_p`` if oryx does not provide one.

Some JAX trace paths emit ``lax.copy_p``, which oryx's inverse interpreter
does not handle out of the box. The rule below is a trivial identity:
``copy`` is its own inverse and contributes 0 to the log-determinant.

This shim is a no-op once the rule is upstreamed to ``jax-ml/oryx``.
"""

from jax import lax

try:
    from oryx.core.interpreters.inverse import rules as _rules
except ImportError:
    _rules = None


def _register_copy_p_rule() -> None:
    if _rules is None:
        return
    if lax.copy_p in _rules.ildj_registry:
        return

    def copy_ildj(incells, outcells, **params):
        incell, = incells
        outcell, = outcells
        if incell.top() and not outcell.top():
            outcells = [_rules.InverseAndILDJ.new(lax.copy_p.bind(incell.val, **params))]
        elif outcell.top() and not incell.top():
            incells = [_rules.InverseAndILDJ.new(outcell.val)]
        return incells, outcells, None

    _rules.ildj_registry[lax.copy_p] = copy_ildj


_register_copy_p_rule()
