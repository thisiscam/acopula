"""Persistent JAX compilation cache helper.

Nested Archimedean density compilation is expensive for families with
deeply-nested transcendental generators (Frank is the worst — the val+grad
JIT for the full MIMIC-IV ICU lab cohort takes ~30 minutes on an
RTX 3090 Ti, and the Hessian JIT pushes ~2.5 hours).  That cost is
structural: ``jax.hessian`` of an O(d_c^2) jet recurrence unrolled at
d_c=14 produces an XLA graph that LLVM's optimisation passes grind
through slowly.

The runtime itself is fine once compiled — the hotspot is strictly
the one-time compile.  JAX's persistent compilation cache writes the
compiled HLO + device-specific binary to disk keyed on a hash of the
program and compile configuration.  Re-running the same experiment in
a new Python process finds the cache entry and reuses it, turning a
30-minute compile into a sub-second cache hit.

Call :func:`set_compile_cache_dir` once, before any ``jax.jit``
operation, to enable the cache for the whole process.
"""
from __future__ import annotations

import os
from pathlib import Path

import jax


def set_compile_cache_dir(
    path: str | os.PathLike,
    *,
    min_compile_time_seconds: float = 1.0,
) -> str:
    """Enable JAX persistent compile cache at ``path``.

    Creates the directory if it does not exist.  Sets the three JAX
    config knobs required for the cache to actually capture the slow
    compiles:

        jax_compilation_cache_dir                    (the location)
        jax_persistent_cache_min_entry_size_bytes    (0: cache everything)
        jax_persistent_cache_min_compile_time_secs   (only compiles
            slower than this threshold get written)

    The default threshold of 1 second keeps the cache small while
    still capturing the Frank/MIMIC compiles we care about.

    Must be called before the first ``jax.jit`` / ``jax.pmap`` /
    ``jax.vmap`` traces through your code; call it at program
    start-up, right after importing jax.

    Returns:
        The absolute cache directory path (useful for logging).
    """
    cache_path = Path(path).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_time_seconds),
    )
    return str(cache_path)
