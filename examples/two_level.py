from acopula import Copula
import jax
import jax.numpy as jnp


class Clayton(Copula):
    theta: float

    def generator(self, u: jax.Array) -> jax.Array:
        return -(jax.log(u) ** self.theta)


class Frank(Copula):
    theta: float

    def generator(self, u: jax.Array) -> jax.Array:
        return -jax.log(1 - (jax.exp(-self.theta * u) - 1) / (jax.exp(-self.theta) - 1))


def model(params, u):
    f = Frank(params[0])
    c = Clayton(params[1])
    return c(f(u[i, j] for j in range(20)) for i in range(10))


# ================================================
# sample
model.sample(
    jax.random.PRNGKey(0), 100
)  # this will generate 100 samples from the model
# under the hood
# compile_model traces the user-defined `model` and builds a complete graph
# then, we need to find a "plan" that traverses the graph
# In particular, a plan is a sequence of stages, where each stage is a set of nodes of the same family and all ancestor families.
# For each stage, all the nodes that are inputs to the current staged are already computed and thus available
# Then, the stage stores statically the indices into the existing computed outputs to be used as inputs to the current stage.
# Now to sample from the model, we use the Marshall-Olkin algorithm:
# 1. sample a random variable V from the outer copula's FS-1
# 2. for each child copula, recursively sample X assuming the outer generator now becomes outer^-1 compose inner
# 3. return exp(-X / V)
# To compute the FS-1 and sample from it, we use the Post-Widder approximation
# The entire MO algorithms is ran on the sequence of stages, and each stage is a single call using jax.vmap
# To find the plan, it suffices to look for each layer (e.g. nodes of the same height from the root), and arrange them in groups that are of the same family.

# ================================================
# log likelihood
model.log_likelihood(
    jnp.ones((10, 20))
)  # this will compute the log likelihood of the model
# under the hood
# compile_model traces the user-defined `model` and builds a complete graph
# if the graph is a single layer tree (there's only one node of copula),
# then the log likelihood is simply found by finding the d^th order mixed derivative of the outer generator, multiplied by the product of the derivatives of the inner generators.
# if the graph is a multi-layer tree,
# then we use the algorithm in @playground/copulaAD/integral_mle.py to compute the log likelihood.
