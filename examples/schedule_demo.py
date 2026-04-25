from __future__ import annotations

from typing import Any, Dict, List

# Prefer module mode: python -m examples.schedule_demo
from acopula.schedule import solve_scheduling_nodes
from acopula.core import Copula, Node
import jax
import jax.numpy as jnp


class Red(Copula):
    def generator(self, u: jax.Array) -> jax.Array:
        return -jnp.log(u)


class Green(Copula):
    def generator(self, u: jax.Array) -> jax.Array:
        return -jnp.log(u)


class Blue(Copula):
    def generator(self, u: jax.Array) -> jax.Array:
        return -jnp.log(u)


def build_demo_tree() -> Node:
    # Build Node-based tree matching the earlier logical structure
    c3 = Node(copula=Blue(), children=[])
    c4 = Node(copula=Blue(), children=[])
    c5 = Node(copula=Blue(), children=[])
    b1 = Node(copula=Green(), children=[c3, c4])
    b2 = Node(copula=Green(), children=[c5])
    a0 = Node(copula=Red(), children=[b1, b2])
    return a0


def simple_hierarchy_layout(G):
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        roots = [next(iter(G.nodes))]
    depths: Dict[Any, int] = {}

    def compute_depth(n):
        if n in depths:
            return depths[n]
        preds = list(G.predecessors(n))
        if not preds:
            depths[n] = 0
        else:
            depths[n] = 1 + max(compute_depth(p) for p in preds)
        return depths[n]

    for n in G.nodes:
        compute_depth(n)

    layers: Dict[int, List[Any]] = {}
    for n, d in depths.items():
        layers.setdefault(d, []).append(n)

    max_width = max(len(v) for v in layers.values()) if layers else 1
    node_gap = 1.5
    layer_gap = 2.0

    pos = {}
    for depth_level in sorted(layers.keys()):
        nodes = layers[depth_level]
        nodes_sorted = sorted(nodes)
        width = len(nodes_sorted)
        total_width = (max_width - 1) * node_gap
        row_width = (width - 1) * node_gap if width > 1 else 0.0
        x_start = -total_width / 2.0 + (total_width - row_width) / 2.0
        for i, n in enumerate(nodes_sorted):
            x = x_start + i * node_gap
            y = -depth_level * layer_gap
            pos[n] = (x, y)
    return pos


def visualize(root: Node, stages_nodes, stage_colors):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        print("Visualization dependencies missing; skipping plot.")
        return

    G = nx.DiGraph()

    # Traverse Node tree and add only Node children edges
    def add_nodes(n: Node):
        G.add_node(id(n), label=n.copula.__class__.__name__)
        for ch in n.children:
            if isinstance(ch, Node):
                G.add_edge(id(n), id(ch))
                add_nodes(ch)

    add_nodes(root)

    pos = simple_hierarchy_layout(G)

    node_to_stage = {}
    for i, slist in enumerate(stages_nodes):
        for node in slist:
            node_to_stage[id(node)] = i
    cmap = plt.cm.get_cmap("viridis", max(1, len(stages_nodes)))
    node_colors = [cmap(node_to_stage.get(v, 0)) for v in G.nodes]

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_color=node_colors,
        node_size=900,
        arrows=True,
    )
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    ax.set_title(f"Stage colors: {stage_colors}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def key_fn(node: Node) -> str:
    return node.copula.__class__.__name__


def main():
    root = build_demo_tree()
    result = solve_scheduling_nodes(root, key_fn)
    print("min_stages:", result["min_stages"])
    print("stage_colors:", result["stage_colors"])
    print("stages (node counts):", [len(s) for s in result["stages"]])

    visualize(root, result["stages"], result["stage_colors"])


if __name__ == "__main__":
    main()
