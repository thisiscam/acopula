"""Visualization utilities for copula models."""

from __future__ import annotations

from typing import Optional, Dict, Any

from .core import Node, Leaf


def to_networkx(node: Node, include_leaves: bool = True, annotations: Optional[Dict[Node, str]] = None):
    """Build a NetworkX DiGraph from a Node tree.
    
    Args:
        node: Root node of the tree
        include_leaves: Whether to include Leaf nodes
        annotations: Optional dict mapping Node -> annotation string to display
    
    Returns:
        NetworkX DiGraph
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError(
            "networkx is required for visualization. Please install 'networkx'."
        ) from e

    G = nx.DiGraph()
    node_to_graph_id: Dict[int, str] = {}  # Use id(node) as key

    def node_label(n: Node | Leaf) -> str:
        base_label = ""
        if isinstance(n, Node):
            base_label = n.copula.__class__.__name__
            # Add annotation if provided
            if annotations and n in annotations:
                base_label += f"\n{annotations[n]}"
        else:
            # Leaf
            if n.index is None and n.subindex is None:
                base_label = "u"
            elif n.subindex is None:
                base_label = f"u[{n.index}]"
            else:
                base_label = f"u[{n.index},{n.subindex}]"
            # Add annotation for leaves if provided
            if annotations and n in annotations:
                base_label += f"\n{annotations[n]}"
        return base_label

    counter = {"i": 0}

    def add(n: Node | Leaf) -> str:
        nid = f"N{counter['i']}"
        counter["i"] += 1
        G.add_node(
            nid,
            label=node_label(n),
            kind=("leaf" if isinstance(n, Leaf) else "copula"),
            node_obj=n,  # Store reference for later lookup
        )
        node_to_graph_id[id(n)] = nid
        return nid

    def walk(n: Node | Leaf) -> str:
        cur = add(n)
        if isinstance(n, Node):
            for ch in n.children:
                if isinstance(ch, Leaf) and not include_leaves:
                    continue
                cid = walk(ch)
                G.add_edge(cur, cid)
        return cur

    walk(node)
    return G


def draw_networkx(
    G,
    *,
    layout: str = "hierarchical",
    with_labels: bool = True,
    graphviz_prog: str = "dot",
    node_colors: Optional[list] = None,
    ax=None,
):
    """Draw a NetworkX graph.
    
    Args:
        G: NetworkX DiGraph
        layout: 'hierarchical' or 'spring'
        with_labels: Whether to show node labels
        graphviz_prog: Graphviz program ('dot', 'twopi', etc.)
        node_colors: Optional list of colors for nodes (overrides default coloring)
        ax: Optional matplotlib axes
    
    Returns:
        (fig, ax)
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError(
            "networkx and matplotlib are required to draw the graph."
        ) from e

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Choose layout
    if layout == "hierarchical":
        try:
            from networkx.drawing.nx_pydot import graphviz_layout

            pos = graphviz_layout(G, prog=graphviz_prog)
        except Exception:
            pos = simple_hierarchy_layout(G)
    else:
        pos = nx.spring_layout(G, seed=0)

    labels = nx.get_node_attributes(G, "label") if with_labels else None
    
    # Use provided colors or default based on kind
    if node_colors is None:
        kinds = nx.get_node_attributes(G, "kind")
        node_colors = [
            "tab:blue" if kinds[n] == "copula" else "tab:orange" for n in G.nodes
        ]

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_color=node_colors,
        node_size=800,
        arrows=True,
    )
    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def simple_hierarchy_layout(G):
    """Simple top-down hierarchical layout without requiring Graphviz."""
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        roots = [next(iter(G.nodes))]
    depths = {}

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

    layers = {}
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

