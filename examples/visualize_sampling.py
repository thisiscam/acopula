"""Example showing step-by-step visualization of sampling computation."""

from acopula import copula, defmodel
import jax
import jax.numpy as jnp


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
def model(params, u):
    f = Clayton(params[0])
    c = Clayton(params[1])
    # 3 groups, each with 2 leaves
    return c(f(u[i, j] for j in range(2)) for i in range(3))


def main():
    import matplotlib.pyplot as plt

    params = jnp.array([2.0, 1.5])
    model.set_params(params)

    # Get the graph
    graph = model.graph()

    # Example 1: Visualize structure
    print("=== Model Structure ===")
    fig1, ax1, G1 = model.visualize(include_leaves=True)
    plt.savefig("model_structure.png", dpi=160)
    print("Saved model_structure.png")
    plt.close()

    # Example 2: Visualize top-down pass with actual V values
    print("\n=== Top-down pass (actual V values) ===")
    from acopula.core import (
        _compute_sampling_schedule,
        _build_sampling_metadata,
        _sample_v_values_topdown,
    )
    from acopula.visualize import to_networkx, draw_networkx
    import jax.random as jrandom

    schedule_result = _compute_sampling_schedule(graph)
    stages_nodes = schedule_result["stages_nodes"]
    metadata = _build_sampling_metadata(schedule_result)
    node_to_idx = metadata["node_to_idx"]
    total_nodes = metadata["total_nodes"]
    node_ancestry_indices = metadata["node_ancestry_indices"]

    # Build parent_lookup array
    from acopula.core import _resolve_param, Param

    parent_lookup_list = [-1] * total_nodes
    for stage_nodes in stages_nodes:
        for node in stage_nodes:
            node_idx = node_to_idx[node]
            ancestry = node_ancestry_indices[node]
            parent_lookup_list[node_idx] = ancestry[-1] if ancestry else -1
    parent_lookup = jnp.array(parent_lookup_list, dtype=jnp.int32)

    # Build parameter indexers (same as in _sample_one_observation)
    constants = []
    base_params_len = params.shape[0]
    param_indexer_list = []
    node_param_start_dict = {}

    for stage_nodes in stages_nodes:
        for node in stage_nodes:
            node_idx = node_to_idx[node]
            node_param_start_dict[node_idx] = len(param_indexer_list)

            param_names = sorted(vars(node.copula).keys())

            for attr_name in param_names:
                val = getattr(node.copula, attr_name)
                if isinstance(val, Param):
                    if val.path and len(val.path) == 1 and isinstance(val.path[0], int):
                        param_indexer_list.append(val.path[0])
                    elif val.path and val.path[0] == "__const__":
                        constants.append(float(val.path[1]))
                        param_indexer_list.append(base_params_len + len(constants) - 1)
                    else:
                        constants.append(_resolve_param(val, params))
                        param_indexer_list.append(base_params_len + len(constants) - 1)
                else:
                    constants.append(float(val))
                    param_indexer_list.append(base_params_len + len(constants) - 1)

    param_indexer = jnp.array(param_indexer_list, dtype=jnp.int32)
    node_param_start_indices = jnp.array(
        [node_param_start_dict[i] for i in range(total_nodes)], dtype=jnp.int32
    )
    params_extended = (
        jnp.concatenate([params, jnp.array(constants)]) if constants else params
    )

    # Sample actual V values (one observation)
    key_v = jrandom.PRNGKey(42)
    v_vals = _sample_v_values_topdown(
        stages_nodes,
        metadata,
        params,
        param_indexer,
        node_param_start_indices,
        params_extended,
        parent_lookup,
        key_v,
        post_widder_k=8,
        max_cdf_x=10,
    )

    # Build annotations using the SAME graph
    annotations_v = {
        node_obj: f"V={float(v_vals[idx]):.3f}" for node_obj, idx in node_to_idx.items()
    }

    # Use the same graph object for visualization
    G2 = to_networkx(graph, include_leaves=False, annotations=annotations_v)
    fig2, ax2 = draw_networkx(G2)
    plt.savefig("model_top_down.png", dpi=160)
    print("Saved model_top_down.png")
    plt.close()

    # Example 3: Visualize stage coloring
    print("\n=== Stage Coloring ===")
    stages_nodes = schedule_result["stages_nodes"]

    # Color nodes by stage
    stage_colors_map = {}
    for si, nodes_in_stage in enumerate(stages_nodes):
        for node_obj in nodes_in_stage:
            stage_colors_map[node_obj] = si

    # Build graph with stage annotations
    stage_annotations = {
        node_obj: f"Stage {si}" for node_obj, si in stage_colors_map.items()
    }

    G3 = to_networkx(graph, include_leaves=False, annotations=stage_annotations)

    # Use stage index for coloring
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("viridis", len(stages_nodes))

    # Map graph node ids to colors
    node_colors = []
    for gnode_id in G3.nodes:
        node_obj = G3.nodes[gnode_id].get("node_obj")
        if node_obj in stage_colors_map:
            node_colors.append(cmap(stage_colors_map[node_obj]))
        else:
            node_colors.append("gray")

    fig3, ax3 = draw_networkx(G3, node_colors=node_colors)
    ax3.set_title(f"Stages: {len(stages_nodes)} total")
    plt.savefig("model_stages.png", dpi=160)
    print("Saved model_stages.png")
    plt.close()

    # Example 4: Visualize bottom-up pass with actual sampled U values
    print("\n=== Bottom-up pass (actual sampled U values) ===")
    from acopula.core import _transform_leaves_bottomup, Node as _Node

    # Use a fresh key for the leaf uniforms
    key_u = jrandom.PRNGKey(123)
    U_leaves = _transform_leaves_bottomup(
        graph,
        stages_nodes,
        v_vals,
        node_to_idx,
        metadata["get_ancestry_chain"],
        params,
        param_indexer,
        node_param_start_indices,
        params_extended,
        parent_lookup,
        key_u,
        metadata["total_nodes"],
    )

    # Build a consistent leaf ordering identical to the transform function
    leaves_in_order = []

    def _collect_leaves_indexed(n):
        for ch in n.children:
            if isinstance(ch, _Node):
                _collect_leaves_indexed(ch)
            else:
                leaves_in_order.append(ch)

    _collect_leaves_indexed(graph)

    # Annotations: internal nodes with V, leaves with sampled U
    annotations_bottom_up = {
        node_obj: f"V={float(v_vals[idx]):.3f}" for node_obj, idx in node_to_idx.items()
    }
    for i, leaf in enumerate(leaves_in_order):
        annotations_bottom_up[leaf] = f"U={float(U_leaves[i]):.3f}"

    G4 = to_networkx(graph, include_leaves=True, annotations=annotations_bottom_up)
    fig4, ax4 = draw_networkx(G4)
    plt.savefig("model_bottom_up.png", dpi=160)
    print("Saved model_bottom_up.png")
    plt.close()

    print("\n=== Summary ===")
    print(f"Total stages: {len(stages_nodes)}")
    for si, stage in enumerate(stages_nodes):
        print(f"  Stage {si}: {len(stage)} nodes")
    print(f"Total leaves: {len(U_leaves)}")
    print(f"Sampled U values: {U_leaves}")

    print("\nVisualization complete! Check the PNG files.")


if __name__ == "__main__":
    main()
