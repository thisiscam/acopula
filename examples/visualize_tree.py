from acopula import Copula, defmodel
import jax
import jax.numpy as jnp


class Clayton(Copula):
    theta: float

    def __init__(self, theta: float):
        self.theta = float(theta)

    def generator(self, u: jax.Array) -> jax.Array:
        return -(jnp.log(u) ** self.theta)


class Frank(Copula):
    theta: float

    def __init__(self, theta: float):
        self.theta = float(theta)

    def generator(self, u: jax.Array) -> jax.Array:
        return -jnp.log(
            1.0 - (jnp.exp(-self.theta * u) - 1.0) / (jnp.exp(-self.theta) - 1.0)
        )


@defmodel
def model(params, u):
    f = Frank(params[0])
    c = Clayton(params[1])
    # 10 groups, each with 5 leaves
    return c(f(u[i, j] for j in range(5)) for i in range(10))


def main():
    # Set some params and build visualization
    params = jnp.array([2.0, 1.5])
    model.set_params(params)

    # Create and show the graph (top-down hierarchical layout)
    fig, ax, G = model.visualize(
        include_leaves=True,
        layout="hierarchical",
        graphviz_prog="dot",
        with_labels=True,
    )

    # Display in GUI
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
