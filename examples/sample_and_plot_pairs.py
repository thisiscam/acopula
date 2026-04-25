"""Generate samples using vmap and plot pairwise distributions."""

from acopula import copula, defmodel, marginal
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.substrates import jax as tfp


@copula
class Clayton:
    theta: float

    def generator(self, u: jax.Array) -> jax.Array:
        return (1.0 + u) ** (-1.0 / self.theta)


@copula
class Frank:
    theta: float

    def generator(self, u: jax.Array) -> jax.Array:
        return -jnp.log1p(jnp.expm1(-self.theta) * jnp.exp(-u)) / self.theta


@defmodel
def model(params, obs):
    c1 = Frank(params[0])
    c2 = Frank(params[1])
    c3 = Frank(params[2])
    # 6 groups in one layer
    # return f(u[i] for i in range(6))
    # 3 groups, each with 2 leaves
    return c1(
        c2(
            c3(
                marginal(tfp.distributions.Uniform(0, 1), obs=obs[i, j, k])
                for k in range(2)
            )
            for j in range(2)
        )
        for i in range(2)
    )


def main():
    # Set parameters
    params = jnp.array([1.0, 2.0, 3.0])
    model.set_params(params)

    # Generate many samples
    print("Generating samples...")
    n_samples = 5000
    key = jrandom.PRNGKey(42)

    # Generate all samples at once (sample method uses vmap internally)
    samples = jax.jit(model.sample, static_argnums=(1,))(key, n_samples)

    print(f"Generated {n_samples} samples")
    print(f"Sample shape: {samples.shape}")
    print("Sample statistics:")
    print(f"  Mean: {jnp.mean(samples, axis=0)}")
    print(f"  Std: {jnp.std(samples, axis=0)}")

    # Convert to numpy for plotting (flatten obs dims -> variables)
    samples_np = np.array(samples).reshape((n_samples, -1))
    n_vars = samples_np.shape[1]

    # Create pairwise scatter plots
    print("\nCreating pairwise scatter plots...")
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(
                    samples_np[:, i],
                    bins=50,
                    density=True,
                    alpha=0.7,
                    color="steelblue",
                )
                ax.set_yticks([])
            else:
                # Off-diagonal: scatter plot for all pairs
                ax.scatter(
                    samples_np[:, j],
                    samples_np[:, i],
                    alpha=0.15,
                    s=1,
                    color="steelblue",
                )
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.2, linewidth=0.5)

                # Calculate and display correlation
                corr = np.corrcoef(samples_np[:, i], samples_np[:, j])[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"ρ={corr:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=7,
                )

            # Set labels for all cells (both diagonal and off-diagonal)
            if i == 0:
                ax.set_title(f"$U_{{{j}}}$", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$U_{{{i}}}$")
            if i == n_vars - 1:
                ax.set_xlabel(f"$U_{{{j}}}$")

            # Remove tick labels except for edges
            if i < n_vars - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.suptitle(
        f"Pairwise Distributions (Clayton θ₁={params[0]}, θ₂={params[1]})",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("pairwise_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved pairwise_distributions.png")
    plt.close()

    # Create a simplified version with just scatter plots
    print("\nCreating simplified scatter plot matrix...")
    fig2, axes2 = plt.subplots(n_vars - 1, n_vars - 1, figsize=(10, 10))

    for i in range(n_vars - 1):
        for j in range(i + 1, n_vars):
            ax = axes2[i, j - 1] if n_vars > 2 else axes2
            ax.scatter(
                samples_np[:, j], samples_np[:, i], alpha=0.2, s=2, color="steelblue"
            )
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f"$U_{{{j}}}$")
            if j == n_vars - 1:
                ax.set_ylabel(f"$U_{{{i}}}$")
            if i == n_vars - 2:
                ax.set_xlabel(f"$U_{{{j}}}$")

            # Calculate and display correlation
            corr = np.corrcoef(samples_np[:, i], samples_np[:, j])[0, 1]
            ax.text(
                0.05,
                0.95,
                f"ρ={corr:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

    # Remove unused subplots
    for i in range(n_vars - 1):
        for j in range(i + 1):
            if n_vars > 2:
                axes2[i, j].axis("off")

    plt.suptitle(f"Scatter Plot Matrix (n={n_samples})", fontsize=14)
    plt.tight_layout()
    plt.savefig("scatter_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved scatter_matrix.png")
    plt.close()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
