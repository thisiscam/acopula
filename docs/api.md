# API reference

Generated from the docstrings. The everyday surface is small: define a copula
family, compile a model, then evaluate or sample.

## Defining a copula

A family subclasses `Copula` — usually via the `@copula` decorator for
dataclass-style parameters — and implements `generator`. Everything else
(density, inverse, sampling) is derived automatically. Call a copula instance
on its children to build the model tree.

::: acopula.copula

::: acopula.Copula
    options:
      members:
        - generator

::: acopula.marginal

## Compiling and evaluating

::: acopula.compile_model

::: acopula.CompiledModel
    options:
      members:
        - eval
        - sample
        - flatten
        - ll_fn
        - visualize
        - as_networkx

## Composition registry

For nested copulas that mix families, register a closed-form composition
`ψ_outer⁻¹ ∘ ψ_inner`, or choose the fallback strategy used when none is
registered.

::: acopula.register_composition

::: acopula.set_composition_fallback

## Configuration

::: acopula.set_stable_log

::: acopula.set_compile_cache_dir

## Advanced: custom generator precision

Most families never touch these — they are optional override hooks on `Copula`
for families whose high-order derivatives suffer float64 cancellation (e.g. AMH
past d≈30). Each returns `None` by default, which selects Taylor-mode AD
(`jet`). `generator_taylor_coefficients` (and its log-space variant
`log_generator_taylor_coefficients`) is the recommended, general hook, used by
the Bell density and the nesting composition. `log_generator_kth_derivative` is
a legacy hook consulted only by the flat single-layer likelihood path.

::: acopula.Copula.generator_inv
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: acopula.Copula.generator_taylor_coefficients
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: acopula.Copula.log_generator_taylor_coefficients
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: acopula.Copula.log_generator_kth_derivative
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
