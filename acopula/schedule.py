from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

from .core import Node, ParamsSymbol


# -----------------------------------------------------------------------------
# Single-Kind-Per-Stage Scheduling on a Tree — Symmetry-Aware Optimal Solver
# -----------------------------------------------------------------------------

CONFIG = {
    "MAX_EXACT_CHILD_TYPES": 6,
    "DP_STATE_CAP": 5_000_000,
    "DETERMINISTIC_TIEBREAK": True,
}


def solve_scheduling(nodes: Dict[Any, Dict[str, Any]], root_id: Any):
    """
    Computes an optimal schedule under 'single-kind-per-stage'.
    Returns min_stages, stage_colors, and per-stage node lists (S_i).
    """
    parent = build_parent_map(nodes)
    indeg0 = {v: len(nodes[v]["children"]) for v in nodes}

    # 1) Canonicalize colored rooted subtrees (AHU) -> type ids
    type_of_node, type_attrs = build_types_AHU(nodes, root_id)

    # 2) Bottom-up optimal sequence per type (memoized; merges cached by child-type SET)
    memo_type_seq: Dict[int, List[Any]] = {}
    memo_type_len: Dict[int, int] = {}
    memo_merge: Dict[Tuple[int, ...], List[Any]] = {}

    root_type = type_of_node[root_id]
    solve_type(root_type, type_attrs, memo_type_seq, memo_type_len, memo_merge)

    stage_colors = memo_type_seq[root_type]
    min_stages = len(stage_colors)

    # 3) Simulate the stage colors against the actual tree (snapshot semantics)
    stages = build_stages(nodes, parent, indeg0, stage_colors)

    return {
        "min_stages": min_stages,
        "stage_colors": stage_colors,
        "stages": stages,
    }


def solve_scheduling_nodes(root: Node, key_fn: Callable[[Node], Any]):
    """Node-based entry point: accepts a `Node` tree and a key function mapping Node -> color/type.

    Returns the same dictionary as solve_scheduling, but with stages as lists of Node objects.
    """
    nodes_dict, root_id, id_to_node = _node_tree_to_dict(root, key_fn)
    result = solve_scheduling(nodes_dict, root_id)
    # Map stage ids back to Node objects
    node_stages: List[List[Node]] = [
        [id_to_node[nid] for nid in stage] for stage in result["stages"]
    ]
    return {
        "min_stages": result["min_stages"],
        "stage_colors": result["stage_colors"],
        "stages": node_stages,
    }


def _node_tree_to_dict(root: Node, key_fn: Callable[[Node], Any]):
    """Convert a Node tree to the dictionary format expected by solve_scheduling.

    Only Node children are considered (Leaf children are ignored for scheduling).
    """
    id_to_node: Dict[str, Node] = {}
    nodes: Dict[str, Dict[str, Any]] = {}

    # Assign stable ids via DFS
    counter = {"i": 0}

    def new_id() -> str:
        nid = f"N{counter['i']}"
        counter["i"] += 1
        return nid

    def is_node_child(ch) -> bool:
        return isinstance(ch, Node)

    def build(n: Node) -> str:
        nid = new_id()
        id_to_node[nid] = n
        color = key_fn(n)
        child_ids: List[str] = []
        for ch in n.children:
            if is_node_child(ch):
                cid = build(ch)
                child_ids.append(cid)
        nodes[nid] = {"color": color, "children": child_ids}
        return nid

    root_id = build(root)
    return nodes, root_id, id_to_node


# --------------------------- DATA PREPARATION --------------------------------


def build_parent_map(nodes: Dict[Any, Dict[str, Any]]):
    parent = {v: None for v in nodes}
    for p in nodes:
        for u in nodes[p]["children"]:
            parent[u] = p
    return parent


def postorder(root_id: Any, nodes: Dict[Any, Dict[str, Any]]):
    """Iterative postorder to avoid recursion limits; returns list of node_ids."""
    out = []
    stack = [(root_id, 0)]
    seen = set()
    while stack:
        v, state = stack.pop()
        if state == 0:
            stack.append((v, 1))
            for u in nodes[v]["children"]:
                if u not in seen:
                    stack.append((u, 0))
        else:
            out.append(v)
            seen.add(v)
    return out


def build_types_AHU(nodes: Dict[Any, Dict[str, Any]], root_id: Any):
    """
    AHU-style canonicalization with colors:
      sig(v) = ( color(v), sorted list of child-type-ids with multiplicities )

    Produces:
      - type_of_node: node_id -> type_id
      - type_attrs: {
            "color": dict[type_id] -> color,
            "kids_multiset": dict[type_id] -> List[type_id] (sorted, with duplicates),
            "kids_distinct": dict[type_id] -> List[type_id] (sorted, no duplicates)
        }
    """
    order = postorder(root_id, nodes)
    sig_to_type: Dict[Tuple[Any, Tuple[int, ...]], int] = {}
    type_of_node: Dict[Any, int] = {}
    type_color: Dict[int, Any] = {}
    type_kids_multi: Dict[int, List[int]] = {}
    type_kids_distinct: Dict[int, List[int]] = {}
    next_type_id = 0

    for v in order:  # children first
        child_types = [type_of_node[u] for u in nodes[v]["children"]]
        child_types.sort()  # permutation-invariance
        sig = (nodes[v]["color"], tuple(child_types))  # includes multiplicities
        if sig in sig_to_type:
            t = sig_to_type[sig]
        else:
            t = next_type_id
            sig_to_type[sig] = t
            next_type_id += 1
            type_color[t] = nodes[v]["color"]
            type_kids_multi[t] = list(child_types)
            type_kids_distinct[t] = sorted(set(child_types))
        type_of_node[v] = t

    type_attrs = {
        "color": type_color,
        "kids_multiset": type_kids_multi,
        "kids_distinct": type_kids_distinct,
    }
    return type_of_node, type_attrs


# --------------------------- TYPE SOLVER (DP) --------------------------------


def solve_type(
    tau: int,
    type_attrs: Dict[str, Dict[int, Any]],
    memo_seq: Dict[int, List[Any]],
    memo_len: Dict[int, int],
    memo_merge: Dict[Tuple[int, ...], List[Any]],
):
    """
    Compute σ*(τ) and its length once, cached. Uses:
      - Duplicate child types collapsed at merge time.
      - Merges cached by the set of child types (sorted tuple key).
      - Exact SCS (with LCS specialization for r=2) or greedy fallback.
    """
    if tau in memo_len:
        return

    color = type_attrs["color"][tau]
    kids_distinct = type_attrs["kids_distinct"][tau]

    if not kids_distinct:
        memo_seq[tau] = [color]
        memo_len[tau] = 1
        return

    # Solve children first
    for sigma in kids_distinct:
        solve_type(sigma, type_attrs, memo_seq, memo_len, memo_merge)

    key = tuple(kids_distinct)
    if key not in memo_merge:
        child_strings = [memo_seq[sigma] for sigma in kids_distinct]
        w = multi_string_scs(child_strings)
        memo_merge[key] = w
    w = memo_merge[key]

    # Parent needs +1 stage for its own color (cannot execute same stage its kids finish)
    seq = list(w) + [color]
    memo_seq[tau] = seq
    memo_len[tau] = len(seq)


# --------------------------- MERGE: SCS UTILITIES -----------------------------


def multi_string_scs(strings: List[List[Any]]) -> List[Any]:
    strings = [s for s in strings if len(s) > 0]
    r = len(strings)
    if r == 0:
        return []
    if r == 1:
        return list(strings[0])
    if r == 2:
        return scs_via_lcs(strings[0], strings[1])

    # Try exact DP if r is small; else greedy fallback
    if r <= CONFIG["MAX_EXACT_CHILD_TYPES"]:
        est_states = 1
        for s in strings:
            est_states *= len(s) + 1
            if est_states > CONFIG["DP_STATE_CAP"]:
                break
        if est_states <= CONFIG["DP_STATE_CAP"]:
            return scs_exact_dp(strings)

    return scs_greedy_frontier(strings)


def scs_via_lcs(a: List[Any], b: List[Any]) -> List[Any]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    i, j = 0, 0
    out: List[Any] = []
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        else:
            if dp[i + 1][j] >= dp[i][j + 1]:
                out.append(a[i])
                i += 1
            else:
                out.append(b[j])
                j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def scs_exact_dp(strings: List[List[Any]]) -> List[Any]:
    r = len(strings)
    lengths = [len(s) for s in strings]
    from functools import lru_cache

    k = common_prefix_length(strings)
    if k > 0:
        prefix = strings[0][:k]
        suffixes = [s[k:] for s in strings]
        return prefix + scs_exact_dp(suffixes)

    @lru_cache(maxsize=None)
    def solve(state: Tuple[int, ...]):
        done = True
        heads: List[Any] = []
        for i in range(r):
            if state[i] < lengths[i]:
                done = False
                heads.append(strings[i][state[i]])
        if done:
            return []

        unique_heads = set(heads)
        if len(unique_heads) == 1:
            c = heads[0]
            new_state = list(state)
            for i in range(r):
                if state[i] < lengths[i] and strings[i][state[i]] == c:
                    new_state[i] += 1
            return [c] + solve(tuple(new_state))

        best = None
        candidates = (
            sorted(unique_heads) if CONFIG["DETERMINISTIC_TIEBREAK"] else unique_heads
        )
        for c in candidates:
            new_state = list(state)
            for i in range(r):
                if state[i] < lengths[i] and strings[i][state[i]] == c:
                    new_state[i] += 1
            tail = solve(tuple(new_state))
            cand = [c] + tail
            if (
                best is None
                or len(cand) < len(best)
                or (
                    CONFIG["DETERMINISTIC_TIEBREAK"]
                    and len(cand) == len(best)
                    and cand < best
                )
            ):
                best = cand
        return best  # type: ignore

    start = tuple(0 for _ in range(r))
    return solve(start)  # type: ignore


def common_prefix_length(strings: List[List[Any]]) -> int:
    if not strings:
        return 0
    k = 0
    while True:
        if any(k >= len(s) for s in strings):
            return k
        c0 = strings[0][k]
        for s in strings[1:]:
            if s[k] != c0:
                return k
        k += 1


def scs_greedy_frontier(strings: List[List[Any]]) -> List[Any]:
    strings = [list(s) for s in strings if len(s) > 0]
    out: List[Any] = []
    while True:
        strings = [s for s in strings if len(s) > 0]
        if not strings:
            break
        heads = [s[0] for s in strings]
        uniq = set(heads)
        if len(uniq) == 1:
            c = heads[0]
        else:
            freq: Dict[Any, int] = {}
            for h in heads:
                freq[h] = freq.get(h, 0) + 1
            best_c, best_f = None, -1
            for c in sorted(freq) if CONFIG["DETERMINISTIC_TIEBREAK"] else freq.keys():
                if freq[c] > best_f:
                    best_c, best_f = c, freq[c]
            c = best_c  # type: ignore
        out.append(c)
        for s in strings:
            if s and s[0] == c:
                s.pop(0)
    return out


# --------------------------- STAGE SIMULATION --------------------------------


def build_stages(
    nodes: Dict[Any, Dict[str, Any]],
    parent: Dict[Any, Any],
    indeg0: Dict[Any, int],
    stage_colors: List[Any],
) -> List[List[Any]]:
    """
    Simulate executing all ready nodes of the stage color, snapshot-at-stage-start.
    """
    ready: Dict[Any, set] = {}
    for v in nodes:
        if indeg0[v] == 0:
            c = nodes[v]["color"]
            if c not in ready:
                ready[c] = set()
            ready[c].add(v)

    executed = set()
    indeg = dict(indeg0)
    stages: List[List[Any]] = []

    for color in stage_colors:
        if color not in ready:
            ready[color] = set()
        start_snapshot = [v for v in ready[color] if v not in executed]
        if CONFIG["DETERMINISTIC_TIEBREAK"]:
            start_snapshot.sort()
        stages.append(start_snapshot)

        for v in start_snapshot:
            executed.add(v)
            p = parent[v]
            if p is not None:
                indeg[p] -= 1
                if indeg[p] == 0:
                    cp = nodes[p]["color"]
                    if cp not in ready:
                        ready[cp] = set()
                    ready[cp].add(p)

    return stages


# --------------------------- Stage dataclass for indices ----------------------


@dataclass
class Stage:
    color: Any
    nodes: List[Any]
    param_refs: List[List[ParamsSymbol]]  # One ParamsSymbol list per node
    input_indices: List[
        int
    ]  # Flat indices into V_values array for parent, one per node (-1 if root)
    output_indices: List[int]  # Flat indices where this stage writes its outputs


def build_stage_objects(
    nodes: Dict[Any, Dict[str, Any]],
    stages: List[List[Any]],
    parent: Dict[Any, Any],
) -> List[Stage]:
    stage_objs: List[Stage] = []

    # Build node -> output index mapping across all stages
    node_to_output_idx: Dict[Any, int] = {}
    output_idx = 0
    for node_list in stages:
        for nid in node_list:
            node_to_output_idx[nid] = output_idx
            output_idx += 1

    for i, node_list in enumerate(stages):
        # Extract param_refs if nodes are Node objects
        param_refs_list: List[List[ParamsSymbol]] = []
        for nid in node_list:
            if isinstance(nid, Node):
                spec = getattr(nid.copula, "_params_symbol", None)
                param_refs_list.append([spec] if spec is not None else [])
            else:
                param_refs_list.append([])

        # Build flat input indices: for each node, find its parent's output index
        input_indices: List[int] = []
        for nid in node_list:
            pid = parent.get(nid)
            if pid is not None:
                input_indices.append(node_to_output_idx[pid])
            else:
                input_indices.append(-1)  # No parent (root)

        # Build output indices for this stage
        output_indices = [node_to_output_idx[nid] for nid in node_list]

        color = nodes[node_list[0]]["color"] if node_list else None
        stage_objs.append(
            Stage(
                color=color,
                nodes=node_list,
                param_refs=param_refs_list,
                input_indices=input_indices,
                output_indices=output_indices,
            )
        )

    return stage_objs


# --------------------------- End of library code ------------------------------
