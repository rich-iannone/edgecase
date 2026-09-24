#!/usr/bin/env python3
"""SAGE — SVG Art Gallery for EdgeCase.

Produces a single HTML page with many graph × option combinations and
a pill-button filter bar for quick visual verification.

Usage:
    python scripts/svg_gallery.py [--out path/to/gallery.html]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Direct-import svg_renderer (bypass edgecase __init__ which needs Python 3.10+)
# ---------------------------------------------------------------------------

_mod_name = "edgecase.svg_renderer"
spec = importlib.util.spec_from_file_location(
    _mod_name,
    str(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "python"
        / "edgecase"
        / "svg_renderer.py"
    ),
)
svg_renderer = importlib.util.module_from_spec(spec)
sys.modules[_mod_name] = svg_renderer
svg_renderer.__name__ = _mod_name
spec.loader.exec_module(svg_renderer)
render_plot = svg_renderer.render_plot
render_traversal = svg_renderer.render_traversal
render_traversal_chain = svg_renderer.render_traversal_chain


# ═══════════════════════════════════════════════════════════════════════════
# Mock Graph classes (matches the Graph API that svg_renderer expects)
# ═══════════════════════════════════════════════════════════════════════════


class MockNode:
    def __init__(self, attrs: Optional[Dict] = None):
        self.attributes = attrs or {}


class MockEdge:
    def __init__(self, attrs: Optional[Dict] = None, source: Any = None, target: Any = None):
        self.attributes = attrs or {}
        self.source = source
        self.target = target


class MockGraph:
    def __init__(self, directed: bool = False):
        self._directed = directed
        self._nodes: Dict[str, Dict] = {}
        self._edges_list: List[Tuple[str, str, int, Dict]] = []
        self._edge_counts: Dict[Tuple[str, str], int] = {}
        self._adj: Dict[str, List[str]] = {}

    def add_node(self, nid: str, **attrs: Any) -> None:
        self._nodes[nid] = attrs
        if nid not in self._adj:
            self._adj[nid] = []

    def add_edge(self, src: str, tgt: str, **attrs: Any) -> None:
        key = (src, tgt)
        idx = self._edge_counts.get(key, 0)
        self._edge_counts[key] = idx + 1
        self._edges_list.append((src, tgt, idx, attrs))
        if src not in self._adj:
            self._adj[src] = []
        self._adj[src].append(tgt)
        if not self._directed:
            if tgt not in self._adj:
                self._adj[tgt] = []
            self._adj[tgt].append(src)

    def get_node_ids(self, limit: Optional[int] = None) -> List[str]:
        return list(self._nodes.keys())

    def get_node(self, nid: str) -> MockNode:
        return MockNode(self._nodes.get(nid, {}))

    def is_directed(self) -> bool:
        return self._directed

    def neighbors(self, nid: str) -> List[str]:
        return self._adj.get(nid, [])

    def find_edge_ids(self, src: str, tgt: str) -> List:
        results = []
        count = self._edge_counts.get((src, tgt), 0)
        for i in range(count):
            results.append((src, tgt, i))
        if not self._directed:
            count2 = self._edge_counts.get((tgt, src), 0)
            for i in range(count2):
                results.append((tgt, src, i))
        return results

    def get_edge(self, eid: Any) -> MockEdge:
        if isinstance(eid, tuple) and len(eid) >= 3:
            src, tgt, idx = eid[0], eid[1], eid[2]
            for s, t, i, attrs in self._edges_list:
                if s == src and t == tgt and i == idx:
                    return MockEdge(attrs, source=src, target=tgt)
            return MockEdge({}, source=src, target=tgt)
        src = eid[0] if isinstance(eid, tuple) else None
        tgt = eid[1] if isinstance(eid, tuple) else None
        for s, t, _i, attrs in self._edges_list:
            if s == src and t == tgt:
                return MockEdge(attrs, source=src, target=tgt)
        return MockEdge({}, source=src, target=tgt)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Factories
# ═══════════════════════════════════════════════════════════════════════════

COMMUNITIES = {
    "A": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "B": ["Frank", "Grace", "Heidi", "Ivan"],
    "C": ["Judy", "Karl", "Leo"],
}


def _graph_social_small() -> MockGraph:
    """13 nodes, 3 communities, undirected — the 'karate club lite'."""
    g = MockGraph(directed=False)
    for comm, names in COMMUNITIES.items():
        for name in names:
            g.add_node(name, community=comm, role="member")
    g.add_node("Mallory", community="A", role="leader")
    for a, b in [
        ("Alice", "Bob"),
        ("Bob", "Carol"),
        ("Carol", "Dave"),
        ("Dave", "Eve"),
        ("Alice", "Carol"),
        ("Bob", "Dave"),
    ]:
        g.add_edge(a, b)
    for a, b in [
        ("Frank", "Grace"),
        ("Grace", "Heidi"),
        ("Heidi", "Ivan"),
        ("Frank", "Ivan"),
    ]:
        g.add_edge(a, b)
    for a, b in [("Judy", "Karl"), ("Karl", "Leo"), ("Judy", "Leo")]:
        g.add_edge(a, b)
    g.add_edge("Mallory", "Alice")
    g.add_edge("Mallory", "Bob")
    g.add_edge("Mallory", "Frank")
    g.add_edge("Mallory", "Judy")
    g.add_edge("Mallory", "Eve")
    g.add_edge("Eve", "Frank")
    g.add_edge("Ivan", "Judy")
    return g


def _graph_tiny() -> MockGraph:
    """3 nodes — triangle."""
    g = MockGraph(directed=False)
    for n in ["X", "Y", "Z"]:
        g.add_node(n, group="core")
    g.add_edge("X", "Y")
    g.add_edge("Y", "Z")
    g.add_edge("Z", "X")
    return g


def _graph_single() -> MockGraph:
    """1 node — edge case."""
    g = MockGraph(directed=False)
    g.add_node("Solo", group="alone")
    return g


def _graph_pair() -> MockGraph:
    """2 nodes, 1 edge."""
    g = MockGraph(directed=False)
    g.add_node("Left", group="a")
    g.add_node("Right", group="b")
    g.add_edge("Left", "Right")
    return g


def _graph_disconnected() -> MockGraph:
    """Two disconnected components."""
    g = MockGraph(directed=False)
    for n in ["A1", "A2", "A3"]:
        g.add_node(n, component="alpha")
    g.add_edge("A1", "A2")
    g.add_edge("A2", "A3")
    for n in ["B1", "B2", "B3", "B4"]:
        g.add_node(n, component="beta")
    g.add_edge("B1", "B2")
    g.add_edge("B2", "B3")
    g.add_edge("B3", "B4")
    g.add_edge("B4", "B1")
    g.add_node("C1", component="gamma")
    return g


def _graph_star() -> MockGraph:
    """Star graph with 8 leaves."""
    g = MockGraph(directed=False)
    g.add_node("Hub", role="center")
    for i in range(1, 9):
        name = f"L{i}"
        g.add_node(name, role="leaf")
        g.add_edge("Hub", name)
    return g


def _graph_directed_small() -> MockGraph:
    """5-node directed cycle with shortcuts."""
    g = MockGraph(directed=True)
    for n in ["A", "B", "C", "D", "E"]:
        g.add_node(n, type="node")
    g.add_edge("A", "B", rel="follows")
    g.add_edge("A", "C", rel="follows")
    g.add_edge("B", "D", rel="follows")
    g.add_edge("C", "D", rel="follows")
    g.add_edge("D", "E", rel="follows")
    g.add_edge("E", "A", rel="follows")
    return g


def _graph_directed_dag() -> MockGraph:
    """A directed acyclic graph (pipeline-like)."""
    g = MockGraph(directed=True)
    nodes = ["Source", "Filter", "Transform", "Validate", "Load", "Archive", "Report"]
    for n in nodes:
        g.add_node(n, stage=n.lower())
    g.add_edge("Source", "Filter", rel="pipes")
    g.add_edge("Filter", "Transform", rel="pipes")
    g.add_edge("Filter", "Validate", rel="pipes")
    g.add_edge("Transform", "Load", rel="pipes")
    g.add_edge("Validate", "Load", rel="pipes")
    g.add_edge("Load", "Archive", rel="stores")
    g.add_edge("Load", "Report", rel="emits")
    return g


def _graph_bipartite() -> MockGraph:
    """Bipartite graph — users and items."""
    g = MockGraph(directed=False)
    users = ["User1", "User2", "User3", "User4"]
    items = ["ItemA", "ItemB", "ItemC", "ItemD", "ItemE"]
    for u in users:
        g.add_node(u, partition="user")
    for it in items:
        g.add_node(it, partition="item")
    connections = [
        ("User1", "ItemA"),
        ("User1", "ItemC"),
        ("User2", "ItemB"),
        ("User2", "ItemD"),
        ("User2", "ItemE"),
        ("User3", "ItemA"),
        ("User3", "ItemB"),
        ("User4", "ItemC"),
        ("User4", "ItemD"),
    ]
    for u, it in connections:
        g.add_edge(u, it)
    return g


def _graph_tree() -> MockGraph:
    """Binary tree, depth 3 (15 nodes)."""
    g = MockGraph(directed=True)
    nodes = []
    for i in range(15):
        name = f"N{i}"
        depth = int(math.log2(i + 1))
        g.add_node(name, depth=str(depth))
        nodes.append(name)
    for i in range(7):
        g.add_edge(nodes[i], nodes[2 * i + 1], rel="child")
        g.add_edge(nodes[i], nodes[2 * i + 2], rel="child")
    return g


def _graph_medium_random() -> MockGraph:
    """25 nodes with random edges and 4 communities."""
    random.seed(42)
    g = MockGraph(directed=False)
    comms = ["alpha", "beta", "gamma", "delta"]
    for i in range(25):
        comm = comms[i % len(comms)]
        g.add_node(f"n{i}", community=comm, weight=str(random.randint(1, 10)))
    for i in range(25):
        for j in range(i + 1, 25):
            ci, cj = comms[i % len(comms)], comms[j % len(comms)]
            p = 0.35 if ci == cj else 0.06
            if random.random() < p:
                g.add_edge(f"n{i}", f"n{j}")
    return g


def _graph_large_random() -> MockGraph:
    """50 nodes with random edges and 5 communities."""
    random.seed(99)
    g = MockGraph(directed=False)
    comms = ["c1", "c2", "c3", "c4", "c5"]
    for i in range(50):
        comm = comms[i % len(comms)]
        g.add_node(f"v{i}", community=comm, importance=str(random.randint(1, 20)))
    for i in range(50):
        for j in range(i + 1, 50):
            ci, cj = comms[i % len(comms)], comms[j % len(comms)]
            p = 0.25 if ci == cj else 0.03
            if random.random() < p:
                g.add_edge(f"v{i}", f"v{j}")
    return g


def _graph_path() -> MockGraph:
    """Linear path of 8 nodes."""
    g = MockGraph(directed=False)
    for i in range(8):
        g.add_node(f"P{i}", position=str(i))
    for i in range(7):
        g.add_edge(f"P{i}", f"P{i + 1}")
    return g


def _graph_complete_5() -> MockGraph:
    """Complete graph K5."""
    g = MockGraph(directed=False)
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    for n in names:
        g.add_node(n, group="all")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            g.add_edge(a, b)
    return g


def _graph_wheel() -> MockGraph:
    """Wheel graph — hub + cycle of 10."""
    g = MockGraph(directed=False)
    g.add_node("Center", role="hub")
    rim = [f"R{i}" for i in range(10)]
    for r in rim:
        g.add_node(r, role="rim")
        g.add_edge("Center", r)
    for i in range(len(rim)):
        g.add_edge(rim[i], rim[(i + 1) % len(rim)])
    return g


def _graph_directed_weighted() -> MockGraph:
    """Directed graph with weighted edges."""
    g = MockGraph(directed=True)
    cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix", "Philly"]
    for c in cities:
        g.add_node(c, type="city")
    routes = [
        ("NYC", "LA", "2800"),
        ("NYC", "Chicago", "790"),
        ("LA", "Phoenix", "370"),
        ("Chicago", "Houston", "1090"),
        ("Houston", "Phoenix", "1180"),
        ("Philly", "NYC", "95"),
        ("Chicago", "Philly", "760"),
        ("Phoenix", "LA", "370"),
    ]
    for src, tgt, dist in routes:
        g.add_edge(src, tgt, distance=dist, rel="route")
    return g


def _stochastic_block_model(
    community_sizes: List[int],
    p_intra: float = 0.3,
    p_inter: float = 0.01,
    seed: int = 42,
) -> MockGraph:
    """Generate a stochastic block model graph with clear community structure."""
    rng = random.Random(seed)
    g = MockGraph(directed=False)
    comm_names = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
    ]
    node_idx = 0
    nodes_by_comm: Dict[str, List[str]] = {}
    for ci, size in enumerate(community_sizes):
        comm = comm_names[ci % len(comm_names)]
        nodes_by_comm[comm] = []
        for _ in range(size):
            nid = f"n{node_idx}"
            g.add_node(nid, community=comm)
            nodes_by_comm[comm].append(nid)
            node_idx += 1

    all_nodes = list(g._nodes.keys())
    node_comm = {nid: attrs["community"] for nid, attrs in g._nodes.items()}

    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            ci = node_comm[all_nodes[i]]
            cj = node_comm[all_nodes[j]]
            p = p_intra if ci == cj else p_inter
            if rng.random() < p:
                g.add_edge(all_nodes[i], all_nodes[j])
    return g


def _graph_sbm_100() -> MockGraph:
    """100 nodes, 5 communities (20 each), stochastic block model."""
    return _stochastic_block_model([20, 20, 20, 20, 20], p_intra=0.3, p_inter=0.015, seed=42)


def _graph_sbm_150_uneven() -> MockGraph:
    """150 nodes, 6 uneven communities."""
    return _stochastic_block_model([40, 30, 25, 25, 15, 15], p_intra=0.25, p_inter=0.006, seed=55)


def _graph_sbm_200() -> MockGraph:
    """200 nodes, 8 communities."""
    return _stochastic_block_model(
        [30, 30, 25, 25, 25, 25, 20, 20], p_intra=0.2, p_inter=0.004, seed=77
    )


def _graph_sbm_300() -> MockGraph:
    """300 nodes, 10 communities."""
    return _stochastic_block_model(
        [40, 35, 35, 30, 30, 30, 25, 25, 25, 25],
        p_intra=0.15,
        p_inter=0.002,
        seed=101,
    )


def _graph_sbm_500() -> MockGraph:
    """500 nodes, 12 communities."""
    return _stochastic_block_model(
        [55, 50, 50, 45, 45, 40, 40, 40, 35, 35, 35, 30],
        p_intra=0.1,
        p_inter=0.001,
        seed=202,
    )


def _graph_sbm_100_dense() -> MockGraph:
    """100 nodes, 4 communities, dense intra-community edges."""
    return _stochastic_block_model([25, 25, 25, 25], p_intra=0.5, p_inter=0.01, seed=33)


def _graph_sbm_200_sparse() -> MockGraph:
    """200 nodes, 7 communities, very sparse inter-community."""
    return _stochastic_block_model(
        [35, 30, 30, 30, 25, 25, 25],
        p_intra=0.2,
        p_inter=0.002,
        seed=88,
    )


def _graph_sbm_1000() -> MockGraph:
    """1000 nodes, 15 communities."""
    sizes = [80, 75, 75, 70, 70, 70, 65, 65, 65, 60, 60, 55, 55, 50, 85]
    return _stochastic_block_model(sizes, p_intra=0.08, p_inter=0.0005, seed=303)


def _graph_tree_deep(depth: int = 6, branching: int = 3, seed: int = 42) -> MockGraph:
    """Generate a tree graph with variable branching."""
    rng = random.Random(seed)
    g = MockGraph(directed=False)
    node_id = 0
    g.add_node(f"t{node_id}", depth="0")
    queue = [(f"t{node_id}", 0)]
    node_id += 1
    while queue:
        parent, d = queue.pop(0)
        if d >= depth:
            continue
        n_children = rng.randint(max(1, branching - 1), branching + 1)
        for _ in range(n_children):
            child = f"t{node_id}"
            g.add_node(child, depth=str(d + 1))
            g.add_edge(parent, child)
            queue.append((child, d + 1))
            node_id += 1
    return g


def _graph_hub_spoke(n_hubs: int = 8, spokes_per_hub: int = 12, seed: int = 42) -> MockGraph:
    """Hub-and-spoke topology: central hubs with radiating leaf nodes."""
    rng = random.Random(seed)
    g = MockGraph(directed=False)
    nid = 0
    hubs = []
    for h in range(n_hubs):
        hub = f"h{nid}"
        g.add_node(hub, role="hub", community=str(h))
        hubs.append(hub)
        nid += 1
        n_sp = rng.randint(spokes_per_hub - 3, spokes_per_hub + 3)
        for _ in range(n_sp):
            spoke = f"s{nid}"
            g.add_node(spoke, role="spoke", community=str(h))
            g.add_edge(hub, spoke)
            nid += 1
    for i in range(len(hubs)):
        for j in range(i + 1, len(hubs)):
            if rng.random() < 0.4:
                g.add_edge(hubs[i], hubs[j])
    return g


# ═══════════════════════════════════════════════════════════════════════════
# Gallery Spec: each entry is (graph_factory, render_kwargs, tags)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GalleryItem:
    title: str
    description: str
    graph_fn: Any
    render_kwargs: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    render_fn: Any = None  # defaults to render_plot
    code: str = ""  # EdgeCase code snippet for this example


def _generate_code(item: "GalleryItem") -> str:
    """Auto-generate EdgeCase Python code from a gallery item."""
    g = item.graph_fn()
    lines: List[str] = ["import edgecase", ""]

    # Graph construction
    directed = g.is_directed()
    lines.append(f"G = edgecase.Graph(directed={directed})")
    lines.append("")

    # Nodes
    node_ids = g.get_node_ids()
    for nid in node_ids:
        obj = g.get_node(nid)
        attrs = dict(obj.attributes) if obj and obj.attributes else {}
        if attrs:
            kw = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
            lines.append(f"G.add_node({nid!r}, {kw})")
        else:
            lines.append(f"G.add_node({nid!r})")

    # Edges
    if hasattr(g, "_edges_list"):
        edge_data = [(s, t, a) for s, t, _i, a in g._edges_list]
    else:
        edge_data = []
    if edge_data:
        lines.append("")
        for src, tgt, attrs in edge_data:
            if attrs:
                kw = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
                lines.append(f"G.add_edge({src!r}, {tgt!r}, {kw})")
            else:
                lines.append(f"G.add_edge({src!r}, {tgt!r})")

    lines.append("")

    # Render call
    rk = dict(item.render_kwargs)
    rk.pop("width", None)
    rk.pop("height", None)

    if item.render_fn is not None:
        # Check if it's a chain renderer (traversal chain)
        fn = item.render_fn
        chain_steps = getattr(fn, "__closure__", None)
        if chain_steps and hasattr(fn, "__code__") and "chain_steps" in (fn.__code__.co_freevars if hasattr(fn.__code__, "co_freevars") else ()):
            # Traversal chain — show record_traversal pattern
            # Extract steps from closure
            steps = None
            for cell in fn.__closure__ or []:
                try:
                    val = cell.cell_contents
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        steps = val
                        break
                except ValueError:
                    pass
            if steps:
                lines.append("rec = G.record_traversal()")
                for step in steps:
                    op = step.get("op", "")
                    display = step.get("op_display", op + "()")
                    lines.append(f"rec.{display}")
                kw_str = ", ".join(f"{k}={v!r}" for k, v in rk.items() if v is not None and v is not False)
                call = f"rec.plot({kw_str})" if kw_str else "rec.plot()"
                lines.append(call)
            else:
                kw_str = ", ".join(f"{k}={v!r}" for k, v in rk.items() if v is not None and v is not False)
                lines.append(f"G.plot({kw_str})" if kw_str else "G.plot()")
        else:
            kw_str = ", ".join(f"{k}={v!r}" for k, v in rk.items() if v is not None and v is not False)
            lines.append(f"G.plot({kw_str})" if kw_str else "G.plot()")
    elif "bfs" in item.tags or "dfs" in item.tags:
        algo = "dfs" if "dfs" in item.tags else "bfs"
        start = node_ids[0] if node_ids else "1"
        kw_str = ", ".join(f"{k}={v!r}" for k, v in rk.items() if v is not None and v is not False)
        extra = f", {kw_str}" if kw_str else ""
        lines.append(f'G.plot_traversal(start={start!r}, algorithm="{algo.upper()}"{extra})')
    else:
        kw_str = ", ".join(f"{k}={v!r}" for k, v in rk.items() if v is not None and v is not False)
        lines.append(f"G.plot({kw_str})" if kw_str else "G.plot()")

    return "\n".join(lines)


def build_gallery_items() -> List[GalleryItem]:
    items = []

    # ------------------------------------------------------------------
    # 1. Graph-type showcase (circular layout, default options)
    # ------------------------------------------------------------------
    graph_showcase = [
        ("Single node", "Edge case: 1 node, 0 edges", _graph_single, ["tiny"]),
        ("Pair", "2 nodes, 1 edge", _graph_pair, ["tiny"]),
        ("Triangle", "3 nodes, 3 edges — minimal cycle", _graph_tiny, ["tiny"]),
        ("Path (8)", "Linear chain of 8 nodes", _graph_path, ["small"]),
        ("Star (8)", "Hub + 8 leaves", _graph_star, ["small"]),
        ("Complete K5", "Fully connected 5-node graph", _graph_complete_5, ["small"]),
        ("Wheel (10)", "Hub + cycle of 10 rim nodes", _graph_wheel, ["small"]),
        (
            "Social (13)",
            "3 communities with a hub — karate-club-lite",
            _graph_social_small,
            ["medium"],
        ),
        ("Bipartite", "Users ↔ Items, 9 nodes", _graph_bipartite, ["small"]),
        ("Disconnected", "3 components (triangle, quad, isolate)", _graph_disconnected, ["small"]),
        ("Directed cycle", "5-node directed graph", _graph_directed_small, ["small", "directed"]),
        (
            "DAG pipeline",
            "7-node directed acyclic graph",
            _graph_directed_dag,
            ["small", "directed"],
        ),
        (
            "Binary tree",
            "Depth-3 binary tree, 15 nodes, directed",
            _graph_tree,
            ["medium", "directed"],
        ),
        (
            "Directed cities",
            "6 cities with weighted routes",
            _graph_directed_weighted,
            ["small", "directed"],
        ),
        (
            "Medium random (25)",
            "25 nodes, 4 communities, random edges",
            _graph_medium_random,
            ["medium"],
        ),
        (
            "Large random (50)",
            "50 nodes, 5 communities, random edges",
            _graph_large_random,
            ["large"],
        ),
    ]

    for title, desc, fn, extra_tags in graph_showcase:
        items.append(
            GalleryItem(
                title=f"{title} — circular",
                description=desc,
                graph_fn=fn,
                render_kwargs=dict(layout="circular", width=500, height=400, show_labels=True),
                tags=["circular", "light", "straight"] + extra_tags,
            )
        )

    # ------------------------------------------------------------------
    # 2. Layout comparison (social graph across all layouts)
    # ------------------------------------------------------------------
    for layout in ["circular", "random"]:
        items.append(
            GalleryItem(
                title=f"Social — {layout} layout",
                description=f"13-node social graph with {layout} layout",
                graph_fn=_graph_social_small,
                render_kwargs=dict(
                    layout=layout, color_by="community", width=500, height=400, show_labels=True
                ),
                tags=[layout, "light", "straight", "medium", "color_by"],
            )
        )

    # ------------------------------------------------------------------
    # 3. Curved vs. straight edges
    # ------------------------------------------------------------------
    for curved, label in [(False, "straight"), (True, "curved")]:
        items.append(
            GalleryItem(
                title=f"Social — {label} edges",
                description=f"Community-colored with {label} edges",
                graph_fn=_graph_social_small,
                render_kwargs=dict(
                    layout="circular",
                    color_by="community",
                    curved_edges=curved,
                    width=500,
                    height=400,
                    show_labels=True,
                ),
                tags=["circular", "light", label, "medium", "color_by"],
            )
        )

    # ------------------------------------------------------------------
    # 4. Theme comparison
    # ------------------------------------------------------------------
    for theme in ["light", "dark"]:
        items.append(
            GalleryItem(
                title=f"Social — {theme} theme",
                description=f"Community colors, {theme} background",
                graph_fn=_graph_social_small,
                render_kwargs=dict(
                    layout="circular",
                    color_by="community",
                    theme=theme,
                    width=500,
                    height=400,
                    show_labels=True,
                ),
                tags=["circular", theme, "straight", "medium", "color_by"],
            )
        )

    # ------------------------------------------------------------------
    # 5. Curved edges on various graph types
    # ------------------------------------------------------------------
    curved_graphs = [
        ("Triangle — curved", _graph_tiny, ["tiny"]),
        ("Star — curved", _graph_star, ["small"]),
        ("Complete K5 — curved", _graph_complete_5, ["small"]),
        ("Wheel — curved", _graph_wheel, ["small"]),
        ("Medium random — curved", _graph_medium_random, ["medium"]),
        ("Directed cycle — curved", _graph_directed_small, ["small", "directed"]),
    ]
    for title, fn, extra_tags in curved_graphs:
        items.append(
            GalleryItem(
                title=title,
                description="Curved Bezier edges",
                graph_fn=fn,
                render_kwargs=dict(
                    layout="circular", curved_edges=True, width=500, height=400, show_labels=True
                ),
                tags=["circular", "light", "curved"] + extra_tags,
            )
        )

    # ------------------------------------------------------------------
    # 6. Edge labels
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Directed — edge labels (straight)",
            description="Edge labels on straight directed edges",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="rel",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "small", "directed", "edge_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="Directed — edge labels (curved)",
            description="Edge labels on curved directed edges via textPath",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="rel",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "curved", "small", "directed", "edge_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="DAG — edge labels",
            description="Pipeline DAG with relationship labels",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="rel",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "small", "directed", "edge_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="Cities — distance labels",
            description="Directed city routes with distance edge labels",
            graph_fn=_graph_directed_weighted,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="distance",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "curved", "small", "directed", "edge_labels"],
        )
    )

    # ------------------------------------------------------------------
    # 7. Color-by attribute
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Bipartite — color by partition",
            description="Users vs items colored by partition attribute",
            graph_fn=_graph_bipartite,
            render_kwargs=dict(
                layout="circular",
                color_by="partition",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "small", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Disconnected — color by component",
            description="3 components colored by attribute",
            graph_fn=_graph_disconnected,
            render_kwargs=dict(
                layout="circular",
                color_by="component",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "small", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Tree — color by depth",
            description="Binary tree colored by depth level",
            graph_fn=_graph_tree,
            render_kwargs=dict(
                layout="circular",
                color_by="depth",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "directed", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Medium random — communities (curved)",
            description="25-node graph, community colors, curved edges",
            graph_fn=_graph_medium_random,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                curved_edges=True,
                width=600,
                height=500,
                show_labels=True,
            ),
            tags=["circular", "light", "curved", "medium", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Large random — communities",
            description="50 nodes, 5 communities, community coloring",
            graph_fn=_graph_large_random,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                width=700,
                height=600,
                show_labels=True,
                node_size=10.0,
            ),
            tags=["circular", "light", "straight", "large", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Large random — communities, curved",
            description="50 nodes, curved community-tinted edges",
            graph_fn=_graph_large_random,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                curved_edges=True,
                width=700,
                height=600,
                show_labels=True,
                node_size=10.0,
            ),
            tags=["circular", "light", "curved", "large", "color_by"],
        )
    )

    # ------------------------------------------------------------------
    # 8. Labels off
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Social — no labels",
            description="Community colors, labels hidden",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                show_labels=False,
                width=500,
                height=400,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "no_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="Large random — no labels",
            description="50 nodes without labels, cleaner at scale",
            graph_fn=_graph_large_random,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                show_labels=False,
                width=600,
                height=500,
                node_size=10.0,
            ),
            tags=["circular", "light", "straight", "large", "color_by", "no_labels"],
        )
    )

    # ------------------------------------------------------------------
    # 9. Node size variations
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Social — large nodes",
            description="node_size=24",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                node_size=24.0,
                width=600,
                height=500,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "sizing"],
        )
    )
    items.append(
        GalleryItem(
            title="Social — small nodes",
            description="node_size=8",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                node_size=8.0,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "sizing"],
        )
    )

    # ------------------------------------------------------------------
    # 10. Edge width / opacity variations
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Social — thick edges",
            description="edge_width=3.0, edge_opacity=0.6",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                edge_width=3.0,
                edge_opacity=0.6,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "styling"],
        )
    )
    items.append(
        GalleryItem(
            title="Social — thin faint edges",
            description="edge_width=0.5, edge_opacity=0.2",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                edge_width=0.5,
                edge_opacity=0.2,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "styling"],
        )
    )

    # ------------------------------------------------------------------
    # 11. Dark theme on various graphs
    # ------------------------------------------------------------------
    dark_graphs = [
        ("Triangle — dark", _graph_tiny, ["tiny"]),
        ("Star — dark", _graph_star, ["small"]),
        ("Directed cycle — dark", _graph_directed_small, ["small", "directed"]),
        ("Medium random — dark", _graph_medium_random, ["medium"]),
        ("Large random — dark", _graph_large_random, ["large"]),
    ]
    for title, fn, extra_tags in dark_graphs:
        items.append(
            GalleryItem(
                title=title,
                description="Dark theme",
                graph_fn=fn,
                render_kwargs=dict(
                    layout="circular",
                    color_by="community" if "community" in str(fn) else None,
                    theme="dark",
                    width=500,
                    height=400,
                    show_labels=True,
                ),
                tags=["circular", "dark", "straight"] + extra_tags,
            )
        )

    # Fix: the lambda check doesn't work; set color_by properly for graphs that have communities
    # Re-do the dark theme items with correct color_by
    items = [it for it in items if not it.title.endswith("— dark")]
    dark_items = [
        ("Triangle — dark", _graph_tiny, None, ["tiny"]),
        ("Star — dark", _graph_star, None, ["small"]),
        ("Directed cycle — dark", _graph_directed_small, None, ["small", "directed"]),
        ("Medium random — dark", _graph_medium_random, "community", ["medium"]),
        ("Large random — dark", _graph_large_random, "community", ["large"]),
        ("Social — dark, curved", _graph_social_small, "community", ["medium"]),
        ("Bipartite — dark", _graph_bipartite, "partition", ["small"]),
    ]
    for title, fn, color_by, extra_tags in dark_items:
        kw = dict(layout="circular", theme="dark", width=500, height=400, show_labels=True)
        if color_by:
            kw["color_by"] = color_by
            extra_tags = extra_tags + ["color_by"]
        items.append(
            GalleryItem(
                title=title,
                description="Dark theme",
                graph_fn=fn,
                render_kwargs=kw,
                tags=["circular", "dark", "straight"] + extra_tags,
            )
        )
    items.append(
        GalleryItem(
            title="Social — dark, curved",
            description="Dark theme with curved Bezier edges",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                theme="dark",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "dark", "curved", "medium", "color_by"],
        )
    )

    # ------------------------------------------------------------------
    # 12. Canvas size variations
    # ------------------------------------------------------------------
    for w, h, label in [
        (300, 200, "small canvas"),
        (800, 600, "large canvas"),
        (600, 300, "wide"),
        (300, 500, "tall"),
    ]:
        items.append(
            GalleryItem(
                title=f"Social — {label} ({w}×{h})",
                description=f"Canvas {w}×{h}px",
                graph_fn=_graph_social_small,
                render_kwargs=dict(
                    layout="circular",
                    color_by="community",
                    width=w,
                    height=h,
                    show_labels=True,
                ),
                tags=["circular", "light", "straight", "medium", "color_by", "canvas_size"],
            )
        )

    # ------------------------------------------------------------------
    # 13. Palette variations
    # ------------------------------------------------------------------
    palettes = ["tableau", "Set1", "Set2", "Pastel1", "Dark2"]
    for pal in palettes:
        items.append(
            GalleryItem(
                title=f"Social — palette '{pal}'",
                description=f"Community colors using {pal} palette",
                graph_fn=_graph_social_small,
                render_kwargs=dict(
                    layout="circular",
                    color_by="community",
                    palette=pal,
                    width=500,
                    height=400,
                    show_labels=True,
                ),
                tags=["circular", "light", "straight", "medium", "color_by", "palette"],
            )
        )

    # ------------------------------------------------------------------
    # 14. Combined options (stress tests)
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Large — dark, curved, no labels",
            description="50 nodes, dark, curved, labels off",
            graph_fn=_graph_large_random,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                theme="dark",
                curved_edges=True,
                show_labels=False,
                node_size=8.0,
                width=600,
                height=500,
            ),
            tags=["circular", "dark", "curved", "large", "color_by", "no_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="DAG — dark, curved, edge labels",
            description="Pipeline DAG: dark theme, curves, relationship labels",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="rel",
                theme="dark",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "dark", "curved", "small", "directed", "edge_labels"],
        )
    )
    items.append(
        GalleryItem(
            title="Cities — dark, curved, distance labels",
            description="Dark theme city routes with distance labels",
            graph_fn=_graph_directed_weighted,
            render_kwargs=dict(
                layout="circular",
                edge_label_by="distance",
                theme="dark",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "dark", "curved", "small", "directed", "edge_labels"],
        )
    )

    # ------------------------------------------------------------------
    # 15. Large community-structured graphs (force-directed layout)
    # ------------------------------------------------------------------

    # SBM 100 nodes — the baseline large graph
    items.append(
        GalleryItem(
            title="SBM 100 — force, communities",
            description="100 nodes, 5 equal communities, force-directed layout",
            graph_fn=_graph_sbm_100,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=700,
                height=600,
                show_labels=False,
                node_size=6.0,
                edge_width=0.8,
                edge_opacity=0.25,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 100 — force, communities, curved",
            description="100 nodes, curved community-tinted edges",
            graph_fn=_graph_sbm_100,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                curved_edges=True,
                width=700,
                height=600,
                show_labels=False,
                node_size=6.0,
                edge_width=0.8,
                edge_opacity=0.2,
            ),
            tags=["force", "light", "curved", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 100 — force, with labels",
            description="100 nodes with labels shown",
            graph_fn=_graph_sbm_100,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=700,
                show_labels=True,
                node_size=5.0,
                edge_width=0.6,
                edge_opacity=0.15,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )

    # SBM 100 dense — tighter clusters
    items.append(
        GalleryItem(
            title="SBM 100 dense — force, 4 communities",
            description="100 nodes, 4 dense communities (p_intra=0.5)",
            graph_fn=_graph_sbm_100_dense,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=700,
                height=600,
                show_labels=False,
                node_size=6.0,
                edge_width=0.6,
                edge_opacity=0.15,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )

    # SBM 150 uneven — varying community sizes
    items.append(
        GalleryItem(
            title="SBM 150 uneven — force, 6 communities",
            description="150 nodes, 6 uneven communities (40/30/25/25/15/15)",
            graph_fn=_graph_sbm_150_uneven,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=650,
                show_labels=False,
                node_size=5.0,
                edge_width=0.6,
                edge_opacity=0.2,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 150 uneven — dark, curved",
            description="150 nodes, dark theme, curved edges",
            graph_fn=_graph_sbm_150_uneven,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                theme="dark",
                curved_edges=True,
                width=800,
                height=650,
                show_labels=False,
                node_size=5.0,
                edge_width=0.6,
                edge_opacity=0.2,
            ),
            tags=["force", "dark", "curved", "xlarge", "color_by", "community_structure"],
        )
    )

    # SBM 200 — 8 communities
    items.append(
        GalleryItem(
            title="SBM 200 — force, 8 communities",
            description="200 nodes, 8 communities, force-directed",
            graph_fn=_graph_sbm_200,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=900,
                height=700,
                show_labels=False,
                node_size=4.0,
                edge_width=0.5,
                edge_opacity=0.15,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 200 sparse — force, 7 communities",
            description="200 nodes, very sparse inter-community (p_inter=0.003)",
            graph_fn=_graph_sbm_200_sparse,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=900,
                height=700,
                show_labels=False,
                node_size=4.0,
                edge_width=0.5,
                edge_opacity=0.15,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 200 — dark, 8 communities",
            description="200 nodes, dark theme",
            graph_fn=_graph_sbm_200,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                theme="dark",
                width=900,
                height=700,
                show_labels=False,
                node_size=4.0,
                edge_width=0.5,
                edge_opacity=0.15,
            ),
            tags=["force", "dark", "straight", "xlarge", "color_by", "community_structure"],
        )
    )

    # SBM 300 — 10 communities
    items.append(
        GalleryItem(
            title="SBM 300 — force, 10 communities",
            description="300 nodes, 10 communities, force-directed",
            graph_fn=_graph_sbm_300,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=1000,
                height=800,
                show_labels=False,
                node_size=3.5,
                edge_width=0.4,
                edge_opacity=0.1,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 300 — dark, curved",
            description="300 nodes, dark theme, curved edges",
            graph_fn=_graph_sbm_300,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                theme="dark",
                curved_edges=True,
                width=1000,
                height=800,
                show_labels=False,
                node_size=3.5,
                edge_width=0.4,
                edge_opacity=0.1,
            ),
            tags=["force", "dark", "curved", "xlarge", "color_by", "community_structure"],
        )
    )

    # SBM 500 — 12 communities (stress test)
    items.append(
        GalleryItem(
            title="SBM 500 — force, 12 communities",
            description="500 nodes, 12 communities — stress test",
            graph_fn=_graph_sbm_500,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=1100,
                height=900,
                show_labels=False,
                node_size=3.0,
                edge_width=0.5,
                edge_opacity=0.08,
            ),
            tags=["force", "light", "straight", "xlarge", "color_by", "community_structure"],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 500 — dark",
            description="500 nodes, dark theme — stress test",
            graph_fn=_graph_sbm_500,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                theme="dark",
                width=1100,
                height=900,
                show_labels=False,
                node_size=3.0,
                edge_width=0.5,
                edge_opacity=0.08,
            ),
            tags=["force", "dark", "straight", "xlarge", "color_by", "community_structure"],
        )
    )

    # ------------------------------------------------------------------
    # 16. Force-directed layout on smaller graphs (comparison)
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="Social — force layout",
            description="13-node social graph, force-directed vs circular",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["force", "light", "straight", "medium", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Medium random — force layout",
            description="25 nodes, 4 communities, force-directed",
            graph_fn=_graph_medium_random,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=600,
                height=500,
                show_labels=True,
                node_size=10.0,
            ),
            tags=["force", "light", "straight", "medium", "color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Large random — force layout",
            description="50 nodes, 5 communities, force-directed",
            graph_fn=_graph_large_random,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=700,
                height=600,
                show_labels=False,
                node_size=8.0,
            ),
            tags=["force", "light", "straight", "large", "color_by"],
        )
    )

    # ------------------------------------------------------------------
    # 17. BFS / DFS traversal visualizations (visit-order slider)
    # ------------------------------------------------------------------

    # BFS on Star graph: Hub first, then all leaves
    items.append(
        GalleryItem(
            title="BFS on Star",
            description="BFS from Hub — radial expansion",
            graph_fn=_graph_star,
            render_kwargs=dict(
                visit_order=["Hub", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
                algorithm="BFS",
                layout="circular",
                width=500,
                height=400,
                node_size=18.0,
            ),
            tags=["bfs", "small", "light"],
            render_fn=render_traversal,
        )
    )

    # BFS on social graph: Mallory first (hub), then communities
    items.append(
        GalleryItem(
            title="BFS on Social graph",
            description="BFS from Mallory — crosses community boundaries",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                visit_order=[
                    "Mallory", "Alice", "Bob", "Frank", "Judy", "Eve",
                    "Carol", "Grace", "Karl", "Leo", "Dave", "Heidi", "Ivan",
                ],
                algorithm="BFS",
                layout="force",
                width=600,
                height=450,
                node_size=16.0,
            ),
            tags=["bfs", "medium", "light"],
            render_fn=render_traversal,
        )
    )

    # DFS on social graph: deeper exploration
    items.append(
        GalleryItem(
            title="DFS on Social graph",
            description="DFS from Mallory — dives into communities",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                visit_order=[
                    "Mallory", "Alice", "Bob", "Carol", "Dave", "Eve",
                    "Frank", "Grace", "Heidi", "Ivan", "Judy", "Karl", "Leo",
                ],
                algorithm="DFS",
                layout="force",
                width=600,
                height=450,
                node_size=16.0,
            ),
            tags=["dfs", "medium", "light"],
            render_fn=render_traversal,
        )
    )

    # BFS on social graph — dark theme
    items.append(
        GalleryItem(
            title="BFS on Social — dark",
            description="BFS traversal, dark theme",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                visit_order=[
                    "Mallory", "Alice", "Bob", "Frank", "Judy", "Eve",
                    "Carol", "Grace", "Karl", "Leo", "Dave", "Heidi", "Ivan",
                ],
                algorithm="BFS",
                layout="force",
                theme="dark",
                width=600,
                height=450,
                node_size=16.0,
            ),
            tags=["bfs", "medium", "dark"],
            render_fn=render_traversal,
        )
    )

    # DFS on binary tree: depth-first branch diving
    items.append(
        GalleryItem(
            title="DFS on Binary tree",
            description="DFS from root — depth-first branch diving",
            graph_fn=_graph_tree,
            render_kwargs=dict(
                visit_order=[
                    "N0", "N1", "N3", "N7", "N8", "N4", "N9", "N10",
                    "N2", "N5", "N11", "N12", "N6", "N13", "N14",
                ],
                algorithm="DFS",
                layout="force",
                width=700,
                height=500,
                node_size=16.0,
            ),
            tags=["dfs", "medium", "light"],
            render_fn=render_traversal,
        )
    )

    # ------------------------------------------------------------------
    # 18. Traversal chain visualizations (trav_* selection-based)
    # ------------------------------------------------------------------

    def _make_chain_renderer(chain_steps):
        def _render(graph, **kwargs):
            return render_traversal_chain(graph, chain_steps, **kwargs)
        return _render

    # --- trav_out: forward hop on directed DAG ---
    # DAG: Source→Filter→{Transform,Validate}→Load→{Archive,Report}
    items.append(
        GalleryItem(
            title="trav_out — DAG pipeline",
            description="select Source → trav_out × 2 → trav_out_edge — forward traversal",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", width=600, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Source])",
                 "selection_type": "nodes", "ids": ["Source"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Filter"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges",
                 "ids": [("Transform", "Load"), ("Validate", "Load")]},
            ]),
        )
    )

    # --- trav_in: backward hop on directed cycle ---
    # Directed small: A→B, A→C, B→D, C→D, D→E, E→A
    items.append(
        GalleryItem(
            title="trav_in — reverse on directed cycle",
            description="select D → trav_in — finds B and C (who feed into D)",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([D])",
                 "selection_type": "nodes", "ids": ["D"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["B", "C"]},
            ]),
        )
    )

    # --- trav_in chained: two backward hops ---
    items.append(
        GalleryItem(
            title="trav_in — two backward hops",
            description="select E → trav_in → trav_in — walks the DAG backward",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", width=600, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Report])",
                 "selection_type": "nodes", "ids": ["Report"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Load"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
            ]),
        )
    )

    # --- trav_both: undirected neighbor expansion ---
    items.append(
        GalleryItem(
            title="trav_both — star bounce",
            description="Leaf → trav_both (hub) → trav_both (all leaves)",
            graph_fn=_graph_star,
            render_kwargs=dict(layout="circular", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "light", "circular"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([L1])",
                 "selection_type": "nodes", "ids": ["L1"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes", "ids": ["Hub"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes",
                 "ids": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]},
            ]),
        )
    )

    # --- trav_both on directed graph: bidirectional ---
    items.append(
        GalleryItem(
            title="trav_both — bidirectional on directed",
            description="select D → trav_both — finds B, C (in) and E (out)",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([D])",
                 "selection_type": "nodes", "ids": ["D"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes", "ids": ["B", "C", "E"]},
            ]),
        )
    )

    # --- trav_out_edge: node → edge selection ---
    items.append(
        GalleryItem(
            title="trav_out_edge — social hub expansion",
            description="Select Mallory → trav_both → trav_out_edge — node to edge",
            graph_fn=_graph_social_small,
            render_kwargs=dict(layout="force", width=600, height=450, show_labels=True),
            tags=["traversal", "medium", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Mallory])",
                 "selection_type": "nodes", "ids": ["Mallory"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes",
                 "ids": ["Alice", "Bob", "Frank", "Judy", "Eve"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges",
                 "ids": [("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Carol"),
                         ("Bob", "Dave"), ("Frank", "Grace"), ("Frank", "Ivan"),
                         ("Judy", "Karl"), ("Judy", "Leo"), ("Eve", "Frank"),
                         ("Dave", "Eve"), ("Ivan", "Judy")]},
            ]),
        )
    )

    # --- trav_in_edge: inbound edges of selected nodes ---
    items.append(
        GalleryItem(
            title="trav_in_edge — inbound edges",
            description="select B, C → trav_in_edge — reveals A→B, A→C",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([B, C])",
                 "selection_type": "nodes", "ids": ["B", "C"]},
                {"op": "trav_in_edge", "op_display": "trav_in_edge()",
                 "selection_type": "edges", "ids": [("A", "B"), ("A", "C")]},
            ]),
        )
    )

    # --- trav_out then trav_in_edge combo ---
    items.append(
        GalleryItem(
            title="trav_out → trav_in_edge — round trip",
            description="select A → trav_out → trav_in_edge — forward then back-edges",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([A])",
                 "selection_type": "nodes", "ids": ["A"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["B", "C"]},
                {"op": "trav_in_edge", "op_display": "trav_in_edge()",
                 "selection_type": "edges", "ids": [("A", "B"), ("A", "C")]},
            ]),
        )
    )

    # --- trav_reverse_edge: edge direction flip ---
    # Directed small: A→B, A→C, B→D, C→D, D→E, E→A
    # Select edges A→B, A→C then reverse → should find E→A (only reverse that exists)
    # Actually: reverse of A→B is B→A (doesn't exist), reverse of A→C is C→A (doesn't exist)
    # Better: select D→E, then reverse → E→A (which does exist: E→A)
    # Actually E→A is the reverse of A→E (which doesn't exist).
    # D→E reverse = E→D (doesn't exist). E→A reverse = A→E (doesn't exist).
    # The graph edges are: A→B, A→C, B→D, C→D, D→E, E→A
    # Reverse of E→A = A→E (doesn't exist). Reverse of D→E = E→D (doesn't exist).
    # Reverse of A→B = B→A (no). Reverse of B→D = D→B (no).
    # For this graph, no edge has its reverse present except maybe...
    # None of the edges have reverses. Let's use a different graph with bidirectional edges.

    # --- trav_reverse_edge on a graph with reciprocal edges ---
    def _graph_reciprocal() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["X", "Y", "Z", "W"]:
            g.add_node(n)
        g.add_edge("X", "Y", rel="friend")
        g.add_edge("Y", "X", rel="friend")
        g.add_edge("Y", "Z", rel="follows")
        g.add_edge("Z", "Y", rel="follows")
        g.add_edge("Z", "W", rel="follows")
        # W→Z does NOT exist — one-way
        g.add_edge("X", "W", rel="admin")
        return g

    items.append(
        GalleryItem(
            title="trav_reverse_edge — reciprocal check",
            description="select X→Y, Y→Z edges → trav_reverse_edge — finds Y→X, Z→Y",
            graph_fn=_graph_reciprocal,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force", "multi_edge"],
            render_fn=_make_chain_renderer([
                {"op": "select_edges_by_id", "op_display": "select_edges_by_id([X→Y, Y→Z])",
                 "selection_type": "edges", "ids": [("X", "Y"), ("Y", "Z")]},
                {"op": "trav_reverse_edge", "op_display": "trav_reverse_edge()",
                 "selection_type": "edges", "ids": [("Y", "X"), ("Z", "Y")]},
            ]),
        )
    )

    # --- Long chain: DAG full forward walk ---
    items.append(
        GalleryItem(
            title="trav_out — DAG full walk (6 steps)",
            description="Source → Filter → branch → Load → sinks — complete pipeline",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", width=600, height=450, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Source])",
                 "selection_type": "nodes", "ids": ["Source"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Filter"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Load"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["Archive", "Report"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": []},
            ]),
        )
    )

    # --- trav_both multi-hop on social — community spread ---
    items.append(
        GalleryItem(
            title="trav_both — 3-hop community spread",
            description="Judy → neighbors → 2nd hop → 3rd hop — ripple across communities",
            graph_fn=_graph_social_small,
            render_kwargs=dict(layout="force", width=600, height=450, show_labels=True),
            tags=["traversal", "medium", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Judy])",
                 "selection_type": "nodes", "ids": ["Judy"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes", "ids": ["Karl", "Leo", "Mallory", "Ivan"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes",
                 "ids": ["Judy", "Leo", "Alice", "Bob", "Frank", "Eve",
                         "Heidi", "Grace"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes",
                 "ids": ["Karl", "Mallory", "Ivan", "Carol", "Dave",
                         "Leo", "Judy"]},
            ]),
        )
    )

    # --- trav_in on DAG: backward from sinks ---
    items.append(
        GalleryItem(
            title="trav_in — backward from DAG sinks",
            description="select Archive, Report → trav_in → trav_in — back to sources",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", width=600, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id",
                 "op_display": "select_nodes_by_id([Archive, Report])",
                 "selection_type": "nodes", "ids": ["Archive", "Report"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Load"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
                {"op": "trav_in_edge", "op_display": "trav_in_edge()",
                 "selection_type": "edges",
                 "ids": [("Filter", "Transform"), ("Filter", "Validate")]},
            ]),
        )
    )

    # --- Mixed node/edge oscillation ---
    items.append(
        GalleryItem(
            title="Node ↔ edge oscillation",
            description="select → trav_out_edge → (edges) → select connected → repeat",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([A])",
                 "selection_type": "nodes", "ids": ["A"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": [("A", "B"), ("A", "C")]},
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([D])",
                 "selection_type": "nodes", "ids": ["D"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": [("D", "E")]},
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([E])",
                 "selection_type": "nodes", "ids": ["E"]},
            ]),
        )
    )

    # --- trav_out on social: from Eve (cross-community bridge) ---
    items.append(
        GalleryItem(
            title="trav_both — bridge node",
            description="Eve sits between communities A and B — trav_both shows the bridge",
            graph_fn=_graph_social_small,
            render_kwargs=dict(layout="force", width=600, height=450, show_labels=True),
            tags=["traversal", "medium", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Eve])",
                 "selection_type": "nodes", "ids": ["Eve"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes", "ids": ["Dave", "Mallory", "Frank"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges",
                 "ids": [("Carol", "Dave"), ("Bob", "Dave"), ("Mallory", "Alice"),
                         ("Mallory", "Bob"), ("Mallory", "Frank"), ("Mallory", "Judy"),
                         ("Mallory", "Eve"), ("Frank", "Grace"), ("Frank", "Ivan")]},
            ]),
        )
    )

    # --- Dark theme: trav_both community ripple ---
    items.append(
        GalleryItem(
            title="trav_both — dark theme ripple",
            description="Judy → trav_both × 2 → trav_out_edge — dark theme",
            graph_fn=_graph_social_small,
            render_kwargs=dict(layout="force", theme="dark", width=600, height=450,
                               show_labels=True),
            tags=["traversal", "medium", "dark", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Judy])",
                 "selection_type": "nodes", "ids": ["Judy"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes", "ids": ["Karl", "Leo", "Mallory", "Ivan"]},
                {"op": "trav_both", "op_display": "trav_both()",
                 "selection_type": "nodes",
                 "ids": ["Judy", "Leo", "Alice", "Bob", "Frank", "Eve",
                         "Heidi", "Grace"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges",
                 "ids": [("Judy", "Karl"), ("Judy", "Leo"), ("Alice", "Bob"),
                         ("Alice", "Carol"), ("Bob", "Carol"), ("Bob", "Dave"),
                         ("Frank", "Grace"), ("Frank", "Ivan"), ("Eve", "Frank"),
                         ("Heidi", "Ivan"), ("Grace", "Heidi")]},
            ]),
        )
    )

    # --- Dark theme: trav_in on DAG ---
    items.append(
        GalleryItem(
            title="trav_in — DAG backward (dark)",
            description="Report → trav_in × 3 — reverse pipeline walk, dark",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", theme="dark", width=600, height=400,
                               show_labels=True),
            tags=["traversal", "small", "directed", "dark", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Report])",
                 "selection_type": "nodes", "ids": ["Report"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Load"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
                {"op": "trav_in", "op_display": "trav_in()",
                 "selection_type": "nodes", "ids": ["Filter"]},
            ]),
        )
    )

    # --- trav_in_edge then back to nodes ---
    items.append(
        GalleryItem(
            title="trav_in_edge → nodes — edge then source",
            description="select Load → trav_in_edge → see who feeds Load",
            graph_fn=_graph_directed_dag,
            render_kwargs=dict(layout="force", width=600, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([Load])",
                 "selection_type": "nodes", "ids": ["Load"]},
                {"op": "trav_in_edge", "op_display": "trav_in_edge()",
                 "selection_type": "edges",
                 "ids": [("Transform", "Load"), ("Validate", "Load")]},
                {"op": "select_nodes_by_id",
                 "op_display": "select_nodes_by_id([Transform, Validate])",
                 "selection_type": "nodes", "ids": ["Transform", "Validate"]},
            ]),
        )
    )

    # --- Circular layout: trav_out on directed cycle ---
    items.append(
        GalleryItem(
            title="trav_out — directed cycle (circular)",
            description="A → B → D → E → A — chasing the cycle",
            graph_fn=_graph_directed_small,
            render_kwargs=dict(layout="circular", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "circular"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([A])",
                 "selection_type": "nodes", "ids": ["A"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["B", "C"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["D"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["E"]},
                {"op": "trav_out", "op_display": "trav_out()",
                 "selection_type": "nodes", "ids": ["A"]},
            ]),
        )
    )

    # --- trav_reverse_edge with add_to_selection ---
    items.append(
        GalleryItem(
            title="trav_reverse_edge(add=True)",
            description="select X→Y → reverse(add=True) — accumulates both directions",
            graph_fn=_graph_reciprocal,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force", "multi_edge"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([X])",
                 "selection_type": "nodes", "ids": ["X"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": [("X", "Y"), ("X", "W")]},
                {"op": "trav_reverse_edge",
                 "op_display": "trav_reverse_edge(add_to_selection=True)",
                 "selection_type": "edges",
                 "ids": [("X", "Y"), ("X", "W"), ("Y", "X")]},
            ]),
        )
    )

    # --- select_edges_by_id → trav_reverse_edge on reciprocal ---
    items.append(
        GalleryItem(
            title="Edge select → reverse — dark",
            description="Start from edges, find reverses — dark theme",
            graph_fn=_graph_reciprocal,
            render_kwargs=dict(layout="force", theme="dark", width=500, height=400,
                               show_labels=True),
            tags=["traversal", "small", "directed", "dark", "force", "multi_edge"],
            render_fn=_make_chain_renderer([
                {"op": "select_edges_by_id", "op_display": "select_edges_by_id([Y→Z, Z→W])",
                 "selection_type": "edges", "ids": [("Y", "Z"), ("Z", "W")]},
                {"op": "trav_reverse_edge", "op_display": "trav_reverse_edge()",
                 "selection_type": "edges", "ids": [("Z", "Y")]},
            ]),
        )
    )

    # --- Self-loop examples ---
    def _graph_with_loops() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["A", "B", "C"]:
            g.add_node(n)
        g.add_edge("A", "B", rel="sends")
        g.add_edge("B", "C", rel="sends")
        g.add_edge("A", "A", rel="self-ref")
        g.add_edge("C", "C", rel="retry")
        return g

    items.append(
        GalleryItem(
            title="Self-loop edges",
            description="Graph with self-loops (A→A, C→C) rendered as arcs",
            graph_fn=_graph_with_loops,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force", "multi_edge"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([A])",
                 "selection_type": "nodes", "ids": ["A"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": [("A", "B"), ("A", "A")]},
            ]),
        )
    )

    # Reciprocal + self-loop combined
    def _graph_reciprocal_with_loop() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["P", "Q", "R"]:
            g.add_node(n)
        g.add_edge("P", "Q", rel="link")
        g.add_edge("Q", "P", rel="link")
        g.add_edge("Q", "R", rel="link")
        g.add_edge("P", "P", rel="self")
        return g

    items.append(
        GalleryItem(
            title="Reciprocal + self-loop",
            description="P↔Q reciprocal pair with P→P self-loop — all rendered as curves/arcs",
            graph_fn=_graph_reciprocal_with_loop,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["traversal", "small", "directed", "light", "force", "multi_edge"],
            render_fn=_make_chain_renderer([
                {"op": "select_nodes_by_id", "op_display": "select_nodes_by_id([P])",
                 "selection_type": "nodes", "ids": ["P"]},
                {"op": "trav_out_edge", "op_display": "trav_out_edge()",
                 "selection_type": "edges", "ids": [("P", "Q"), ("P", "P")]},
                {"op": "trav_reverse_edge", "op_display": "trav_reverse_edge()",
                 "selection_type": "edges", "ids": [("Q", "P"), ("P", "P")]},
            ]),
        )
    )

    # ------------------------------------------------------------------
    # Multi-edge static examples (non-traversal)
    # ------------------------------------------------------------------

    # Reciprocal pair — static plot
    def _graph_reciprocal_static() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["A", "B", "C", "D"]:
            g.add_node(n)
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        g.add_edge("B", "C")
        g.add_edge("C", "B")
        g.add_edge("C", "D")
        g.add_edge("A", "D")
        return g

    items.append(
        GalleryItem(
            title="Reciprocal edges — static",
            description="Directed graph with A↔B and B↔C reciprocal pairs, auto-curved",
            graph_fn=_graph_reciprocal_static,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["small", "directed", "light", "force", "multi_edge"],
        )
    )

    items.append(
        GalleryItem(
            title="Reciprocal edges — dark",
            description="Same reciprocal graph in dark theme",
            graph_fn=_graph_reciprocal_static,
            render_kwargs=dict(layout="force", theme="dark", width=500, height=400,
                               show_labels=True),
            tags=["small", "directed", "dark", "force", "multi_edge"],
        )
    )

    # Triple edges between same pair
    def _graph_triple_edge() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["X", "Y", "Z"]:
            g.add_node(n)
        g.add_edge("X", "Y", label="email")
        g.add_edge("X", "Y", label="chat")
        g.add_edge("X", "Y", label="call")
        g.add_edge("Y", "Z", label="link")
        return g

    items.append(
        GalleryItem(
            title="Triple parallel edges",
            description="Three X→Y edges fan out as curves; Y→Z stays straight",
            graph_fn=_graph_triple_edge,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["tiny", "directed", "light", "force", "multi_edge"],
        )
    )

    # Self-loop — static
    def _graph_self_loop_static() -> MockGraph:
        g = MockGraph(directed=True)
        for n in ["S", "T", "U"]:
            g.add_node(n)
        g.add_edge("S", "T")
        g.add_edge("T", "U")
        g.add_edge("S", "S")
        g.add_edge("U", "U")
        return g

    items.append(
        GalleryItem(
            title="Self-loops — static",
            description="Directed graph with S→S and U→U self-loop arcs",
            graph_fn=_graph_self_loop_static,
            render_kwargs=dict(layout="force", width=500, height=400, show_labels=True),
            tags=["tiny", "directed", "light", "force", "multi_edge"],
        )
    )

    # Stress test: dense multi-edge graph
    def _graph_multi_edge_stress() -> MockGraph:
        g = MockGraph(directed=True)
        nodes = ["N1", "N2", "N3", "N4", "N5"]
        for n in nodes:
            g.add_node(n)
        # 5 edges between N1↔N2
        for i in range(3):
            g.add_edge("N1", "N2", idx=i)
        for i in range(2):
            g.add_edge("N2", "N1", idx=i)
        # 4 edges between N3↔N4
        for i in range(2):
            g.add_edge("N3", "N4", idx=i)
        for i in range(2):
            g.add_edge("N4", "N3", idx=i)
        # Regular edges
        g.add_edge("N2", "N3")
        g.add_edge("N4", "N5")
        g.add_edge("N5", "N1")
        # Self-loop
        g.add_edge("N3", "N3")
        return g

    items.append(
        GalleryItem(
            title="Multi-edge stress — 5 nodes",
            description="Dense multi-edge: 5 edges N1↔N2, 4 edges N3↔N4, self-loop on N3",
            graph_fn=_graph_multi_edge_stress,
            render_kwargs=dict(layout="force", width=600, height=500, show_labels=True),
            tags=["small", "directed", "light", "force", "multi_edge"],
        )
    )

    items.append(
        GalleryItem(
            title="Multi-edge stress — dark",
            description="Same dense multi-edge graph in dark theme",
            graph_fn=_graph_multi_edge_stress,
            render_kwargs=dict(layout="force", theme="dark", width=600, height=500,
                               show_labels=True),
            tags=["small", "directed", "dark", "force", "multi_edge"],
        )
    )

    # Large multi-edge stress: many pairs with multiple edges
    def _graph_multi_edge_large() -> MockGraph:
        g = MockGraph(directed=True)
        import random as _rng
        _rng.seed(42)
        nodes = [f"v{i}" for i in range(12)]
        for n in nodes:
            g.add_node(n)
        pairs = [
            ("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v4"),
            ("v4", "v5"), ("v5", "v0"), ("v0", "v3"),
            ("v6", "v7"), ("v7", "v8"), ("v8", "v6"),
            ("v9", "v10"), ("v10", "v11"), ("v11", "v9"),
            ("v2", "v7"), ("v5", "v9"),
        ]
        for s, t in pairs:
            n_edges = _rng.choice([1, 1, 1, 2, 2, 3])
            for _ in range(n_edges):
                g.add_edge(s, t)
            if _rng.random() < 0.3:
                g.add_edge(t, s)
        # A few self-loops
        g.add_edge("v0", "v0")
        g.add_edge("v6", "v6")
        return g

    items.append(
        GalleryItem(
            title="Multi-edge stress — 12 nodes",
            description="12 nodes with random multi-edges (1-3 per pair) and self-loops",
            graph_fn=_graph_multi_edge_large,
            render_kwargs=dict(layout="force", width=700, height=500, show_labels=True),
            tags=["medium", "directed", "light", "force", "multi_edge"],
        )
    )

    items.append(
        GalleryItem(
            title="Multi-edge circular layout",
            description="Same 12-node multi-edge graph in circular layout",
            graph_fn=_graph_multi_edge_large,
            render_kwargs=dict(layout="circular", width=700, height=500, show_labels=True),
            tags=["medium", "directed", "light", "circular", "multi_edge"],
        )
    )

    # ------------------------------------------------------------------
    # Edge color by community
    # ------------------------------------------------------------------
    items.append(
        GalleryItem(
            title="SBM 100 — edge_color_by community",
            description="Intra-community edges match node color, cross-community edges in red",
            graph_fn=_graph_sbm_100,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                edge_color_by="community",
                width=700,
                height=500,
                show_labels=False,
            ),
            tags=[
                "force",
                "light",
                "straight",
                "large",
                "color_by",
                "edge_color_by",
                "community_structure",
            ],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 100 — edge_color_by (dark)",
            description="Community-colored edges on dark background",
            graph_fn=_graph_sbm_100,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                edge_color_by="community",
                theme="dark",
                width=700,
                height=500,
                show_labels=False,
            ),
            tags=[
                "force",
                "dark",
                "straight",
                "large",
                "color_by",
                "edge_color_by",
                "community_structure",
            ],
        )
    )
    items.append(
        GalleryItem(
            title="Social — edge_color_by community",
            description="Small social graph with community-colored edges",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                edge_color_by="community",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "straight", "medium", "color_by", "edge_color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Social — edge_color_by, curved",
            description="Curved community-colored edges",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="circular",
                color_by="community",
                edge_color_by="community",
                curved_edges=True,
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["circular", "light", "curved", "medium", "color_by", "edge_color_by"],
        )
    )
    items.append(
        GalleryItem(
            title="Medium random — edge_color_by",
            description="25-node graph with community-colored edges",
            graph_fn=_graph_medium_random,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                edge_color_by="community",
                width=600,
                height=450,
                show_labels=False,
            ),
            tags=[
                "force",
                "light",
                "straight",
                "medium",
                "color_by",
                "edge_color_by",
                "community_structure",
            ],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 150 — edge_color_by (dark, curved)",
            description="Large uneven SBM with community-colored curved edges",
            graph_fn=_graph_sbm_150_uneven,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                edge_color_by="community",
                curved_edges=True,
                theme="dark",
                width=700,
                height=500,
                show_labels=False,
            ),
            tags=[
                "force",
                "dark",
                "curved",
                "large",
                "color_by",
                "edge_color_by",
                "community_structure",
            ],
        )
    )

    # ------------------------------------------------------------------
    # Overlap-free layouts (showcasing overlap removal at various scales)
    # ------------------------------------------------------------------

    # Small: 13-node social graph — force layout, default node size
    items.append(
        GalleryItem(
            title="Social — force, overlap-free",
            description="13-node social graph with overlap removal",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=500,
                height=400,
                show_labels=True,
            ),
            tags=["force", "light", "straight", "small", "color_by", "overlap_free"],
        )
    )

    # Medium: 25-node random graph
    items.append(
        GalleryItem(
            title="Medium random — force, overlap-free",
            description="25 nodes, 4 communities, overlap removal spreads clusters",
            graph_fn=_graph_medium_random,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=600,
                height=450,
                show_labels=True,
            ),
            tags=["force", "light", "straight", "medium", "color_by", "overlap_free"],
        )
    )

    # Large: hub-and-spoke topology (~100 nodes)
    items.append(
        GalleryItem(
            title="Hub-spoke — force, overlap-free",
            description="8 hubs with ~12 spokes each, overlap removal separates clusters",
            graph_fn=_graph_hub_spoke,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=700,
                show_labels=False,
            ),
            tags=["force", "light", "straight", "large", "color_by", "overlap_free"],
        )
    )
    items.append(
        GalleryItem(
            title="Hub-spoke — dark, overlap-free",
            description="Hub-and-spoke on dark background",
            graph_fn=_graph_hub_spoke,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                theme="dark",
                width=800,
                height=700,
                show_labels=False,
            ),
            tags=["force", "dark", "straight", "large", "color_by", "overlap_free"],
        )
    )

    # Tree: ~300+ nodes, branching tree structure
    items.append(
        GalleryItem(
            title="Tree (depth 5) — force, overlap-free",
            description="Branching tree (~300 nodes), overlap removal preserves hierarchy",
            graph_fn=lambda: _graph_tree_deep(depth=5, branching=3, seed=42),
            render_kwargs=dict(
                layout="force",
                width=900,
                height=700,
                show_labels=False,
            ),
            tags=["force", "light", "straight", "xlarge", "overlap_free"],
        )
    )
    items.append(
        GalleryItem(
            title="Tree (depth 5) — dark, overlap-free",
            description="Branching tree on dark background",
            graph_fn=lambda: _graph_tree_deep(depth=5, branching=3, seed=42),
            render_kwargs=dict(
                layout="force",
                theme="dark",
                width=900,
                height=700,
                show_labels=False,
            ),
            tags=["force", "dark", "straight", "xlarge", "overlap_free"],
        )
    )

    # XLarge: SBM 500
    items.append(
        GalleryItem(
            title="SBM 500 — force, overlap-free",
            description="500 nodes, 12 communities, overlap removal at scale",
            graph_fn=_graph_sbm_500,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=1000,
                height=800,
                show_labels=False,
            ),
            tags=[
                "force",
                "light",
                "straight",
                "xlarge",
                "color_by",
                "overlap_free",
                "community_structure",
            ],
        )
    )
    items.append(
        GalleryItem(
            title="SBM 500 — dark, curved, overlap-free",
            description="500 nodes with curved edges and community colors",
            graph_fn=_graph_sbm_500,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                curved_edges=True,
                theme="dark",
                width=1000,
                height=800,
                show_labels=False,
            ),
            tags=[
                "force",
                "dark",
                "curved",
                "xlarge",
                "color_by",
                "overlap_free",
                "community_structure",
            ],
        )
    )

    # Massive: SBM 1000
    items.append(
        GalleryItem(
            title="SBM 1000 — force, overlap-free",
            description="1000 nodes, 15 communities, stress test for overlap removal",
            graph_fn=_graph_sbm_1000,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=1200,
                height=1000,
                show_labels=False,
            ),
            tags=[
                "force",
                "light",
                "straight",
                "xlarge",
                "color_by",
                "overlap_free",
                "community_structure",
            ],
        )
    )

    # ------------------------------------------------------------------
    # Overlap / routing option comparisons
    # ------------------------------------------------------------------

    # Same graph with node_overlap="allow" (raw layout, no overlap removal)
    items.append(
        GalleryItem(
            title="Hub-spoke — node_overlap='allow'",
            description="Raw force layout — no overlap removal applied",
            graph_fn=_graph_hub_spoke,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=700,
                show_labels=False,
                node_overlap="allow",
                edge_routing="straight",
            ),
            tags=["force", "light", "straight", "large", "color_by", "comparison"],
        )
    )

    # Same graph with overlap removal but no edge routing
    items.append(
        GalleryItem(
            title="Hub-spoke — edge_routing='straight'",
            description="Overlap removal ON but edges go straight (no routing)",
            graph_fn=_graph_hub_spoke,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=700,
                show_labels=False,
                node_overlap="remove",
                edge_routing="straight",
            ),
            tags=["force", "light", "straight", "large", "color_by", "comparison"],
        )
    )

    # Full: overlap removal + edge routing (for comparison)
    items.append(
        GalleryItem(
            title="Hub-spoke — full (remove + around)",
            description="Both overlap removal and edge routing enabled (default)",
            graph_fn=_graph_hub_spoke,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=800,
                height=700,
                show_labels=False,
            ),
            tags=["force", "light", "straight", "large", "color_by", "comparison", "overlap_free"],
        )
    )

    # Custom padding: wide spacing between nodes
    items.append(
        GalleryItem(
            title="Social — node_overlap=8.0 (wide)",
            description="Extra-wide 8px padding between node circles",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=600,
                height=500,
                show_labels=True,
                node_overlap=8.0,
            ),
            tags=["force", "light", "straight", "small", "color_by", "comparison"],
        )
    )

    # Custom edge routing margin
    items.append(
        GalleryItem(
            title="Social — edge_routing=10.0 (wide)",
            description="Wide 10px clearance margin for edge routing around nodes",
            graph_fn=_graph_social_small,
            render_kwargs=dict(
                layout="force",
                color_by="community",
                width=600,
                height=500,
                show_labels=True,
                edge_routing=10.0,
            ),
            tags=["force", "light", "straight", "small", "color_by", "comparison"],
        )
    )

    return items


# ═══════════════════════════════════════════════════════════════════════════
# HTML Generation
# ═══════════════════════════════════════════════════════════════════════════

FILTER_AXES = {
    "Graph Size": ["tiny", "small", "medium", "large", "xlarge"],
    "Layout": ["circular", "random", "force"],
    "Theme": ["light", "dark"],
    "Edges": ["straight", "curved"],
    "Features": [
        "directed",
        "color_by",
        "edge_color_by",
        "community_structure",
        "overlap_free",
        "comparison",
        "edge_labels",
        "no_labels",
        "sizing",
        "styling",
        "canvas_size",
        "palette",
        "traversal",
        "bfs",
        "dfs",
        "multi_edge",
    ],
}

AXIS_COLORS = {
    "Graph Size": "#6366f1",
    "Layout": "#0891b2",
    "Theme": "#7c3aed",
    "Edges": "#059669",
    "Features": "#d97706",
}

TAG_COLORS = {
    "tiny": "#a5b4fc",
    "small": "#818cf8",
    "medium": "#6366f1",
    "large": "#4f46e5",
    "xlarge": "#3730a3",
    "circular": "#22d3ee",
    "random": "#06b6d4",
    "force": "#0e7490",
    "light": "#e2e8f0",
    "dark": "#334155",
    "straight": "#34d399",
    "curved": "#10b981",
    "directed": "#fb923c",
    "color_by": "#fbbf24",
    "edge_color_by": "#f59e0b",
    "edge_labels": "#f97316",
    "no_labels": "#f59e0b",
    "sizing": "#d97706",
    "styling": "#b45309",
    "canvas_size": "#92400e",
    "palette": "#78350f",
    "community_structure": "#dc2626",
    "overlap_free": "#16a34a",
    "comparison": "#0ea5e9",
    "traversal": "#7c3aed",
    "bfs": "#8b5cf6",
    "dfs": "#6d28d9",
    "multi_edge": "#e11d48",
}


def generate_gallery_html(items: List[GalleryItem]) -> str:
    t0 = time.time()

    # Generate code for each item and render all SVGs
    rendered = []
    for i, item in enumerate(items):
        if not item.code:
            try:
                item.code = _generate_code(item)
            except Exception:
                item.code = "# Code generation failed"
        g = item.graph_fn()
        try:
            fn = item.render_fn or render_plot
            result = fn(g, **item.render_kwargs)
            svg_html = result._html
            status = "ok"
        except Exception as e:
            svg_html = f'<div style="color:red;padding:20px;">Error: {e}</div>'
            status = "error"
        rendered.append((item, svg_html, status))

    elapsed = time.time() - t0

    # Count stats
    ok_count = sum(1 for _, _, s in rendered if s == "ok")
    err_count = sum(1 for _, _, s in rendered if s == "error")

    # All unique tags
    all_tags = sorted(set(t for item, _, _ in rendered for t in item.tags))

    # Build cards HTML
    cards_html = []
    for idx, (item, svg_html, status) in enumerate(rendered):
        tags_json = json.dumps(item.tags)
        dark_bg = "background:#1e293b;" if "dark" in item.tags else ""
        border_color = "#ef4444" if status == "error" else "#e2e8f0"

        tag_badges = ""
        for t in item.tags:
            tc = TAG_COLORS.get(t, "#94a3b8")
            text_col = "#1e293b" if tc > "#888888" else "#f8fafc"
            tag_badges += (
                f'<span style="display:inline-block;padding:2px 8px;border-radius:9999px;'
                f"font-size:11px;font-weight:500;background:{tc};color:{text_col};"
                f'margin:2px 2px 0 0;">{t}</span>'
            )

        kw_display = ", ".join(
            f"{k}={v!r}"
            for k, v in item.render_kwargs.items()
            if k not in ("width", "height") and v is not None and v is not False
        )

        code_escaped = (
            item.code
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        cards_html.append(f"""
<div class="card" data-tags='{tags_json}' data-idx="{idx}"
     style="border-color:{border_color};">
  <div class="card-header">
    <span class="card-num">#{idx + 1:03d}</span>
    <span class="card-title">{item.title}</span>
    <button class="code-btn" onclick="showCode(this)" data-code="{code_escaped}"
            title="Show EdgeCase code">&lt;/&gt;</button>
  </div>
  <div class="card-desc">{item.description}</div>
  <div class="card-kwargs">{kw_display}</div>
  <div class="card-tags">{tag_badges}</div>
  <div class="card-svg" style="{dark_bg}">
    {svg_html}
  </div>
</div>""")

    cards_joined = "\n".join(cards_html)

    # Build filter buttons
    filter_buttons = '<button class="filter-btn active" data-filter="all">All</button>\n'
    for axis, tags in FILTER_AXES.items():
        ac = AXIS_COLORS.get(axis, "#64748b")
        filter_buttons += f'<span class="axis-label" style="color:{ac};">{axis}:</span>\n'
        for t in tags:
            count = sum(1 for item, _, _ in rendered if t in item.tags)
            if count > 0:
                filter_buttons += (
                    f'<button class="filter-btn" data-filter="{t}">'
                    f'{t} <span class="count">({count})</span></button>\n'
                )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAGE — SVG Art Gallery for EdgeCase</title>
<style>
  :root {{
    --bg: #0f172a;
    --surface: #1e293b;
    --surface-hover: #334155;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
    padding: 32px 24px 24px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .header h1 span {{ color: var(--accent-light); }}
  .header .stats {{
    display: flex;
    gap: 24px;
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }}
  .header .stats strong {{ color: var(--text); }}
  .filter-bar {{
    background: var(--surface);
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .axis-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-left: 12px;
    margin-right: 2px;
  }}
  .filter-btn {{
    padding: 5px 14px;
    border: 1px solid var(--border);
    border-radius: 9999px;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  .filter-btn:hover {{
    background: var(--surface-hover);
    color: var(--text);
  }}
  .filter-btn.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
  .filter-btn .count {{
    font-size: 11px;
    opacity: 0.7;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
    gap: 16px;
    padding: 24px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }}
  .card-header {{
    padding: 12px 16px 4px;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }}
  .card-num {{
    font-size: 12px;
    color: var(--accent-light);
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-weight: 600;
  }}
  .card-title {{
    font-size: 15px;
    font-weight: 600;
  }}
  .card-desc {{
    padding: 0 16px;
    font-size: 13px;
    color: var(--text-muted);
  }}
  .card-kwargs {{
    padding: 4px 16px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'SF Mono', 'Fira Code', monospace;
    opacity: 0.7;
  }}
  .card-tags {{
    padding: 4px 16px 8px;
  }}
  .card-svg {{
    border-top: 1px solid var(--border);
    padding: 12px;
    background: white;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .card-svg svg {{
    max-width: 100%;
    height: auto;
  }}
  .hidden {{ display: none !important; }}
  .no-results {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
    font-size: 16px;
  }}
  .code-btn {{
    margin-left: auto;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: transparent;
    color: var(--text-muted);
    font-size: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }}
  .code-btn:hover {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
  .code-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 1000;
    align-items: center;
    justify-content: center;
  }}
  .code-overlay.visible {{
    display: flex;
  }}
  .code-panel {{
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 12px;
    width: min(720px, 90vw);
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 24px 80px rgba(0,0,0,0.5);
  }}
  .code-panel-header {{
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #45475a;
    gap: 12px;
  }}
  .code-panel-title {{
    font-size: 14px;
    font-weight: 600;
    color: #cdd6f4;
    flex: 1;
  }}
  .code-panel-copy {{
    padding: 4px 12px;
    border: 1px solid #45475a;
    border-radius: 6px;
    background: transparent;
    color: #a6adc8;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .code-panel-copy:hover {{
    background: #45475a;
    color: #cdd6f4;
  }}
  .code-panel-close {{
    padding: 4px 10px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #a6adc8;
    font-size: 18px;
    cursor: pointer;
    line-height: 1;
  }}
  .code-panel-close:hover {{
    background: #45475a;
    color: #cdd6f4;
  }}
  .code-panel-body {{
    overflow: auto;
    padding: 0;
  }}
  .code-panel-body pre {{
    margin: 0;
    padding: 16px 20px;
    font-size: 13px;
    line-height: 1.6;
    tab-size: 4;
  }}
  .code-panel-body code {{
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace;
  }}
  /* Syntax colors (Catppuccin Mocha inspired) */
  .code-panel-body .kw {{ color: #cba6f7; }}
  .code-panel-body .fn {{ color: #89b4fa; }}
  .code-panel-body .str {{ color: #a6e3a1; }}
  .code-panel-body .num {{ color: #fab387; }}
  .code-panel-body .cmt {{ color: #6c7086; font-style: italic; }}
  .code-panel-body .op {{ color: #89dceb; }}
  .code-panel-body .bool {{ color: #fab387; }}
</style>
</head>
<body>
<div class="header">
  <h1><span>SAGE</span> — SVG Art Gallery for EdgeCase</h1>
  <div class="stats">
    <span>Total: <strong>{len(rendered)}</strong></span>
    <span>OK: <strong>{ok_count}</strong></span>
    {"<span>Failed: <strong>" + str(err_count) + "</strong></span>" if err_count else ""}
    <span>Rendered in <strong>{elapsed:.1f}s</strong></span>
  </div>
</div>

<div class="filter-bar" id="filter-bar">
  {filter_buttons}
</div>

<div class="grid" id="gallery">
  {cards_joined}
  <div class="no-results hidden" id="no-results">No examples match this filter.</div>
</div>

<div class="code-overlay" id="code-overlay" onclick="if(event.target===this)closeCode()">
  <div class="code-panel">
    <div class="code-panel-header">
      <span class="code-panel-title" id="code-panel-title">EdgeCase Code</span>
      <button class="code-panel-copy" onclick="copyCode()">Copy</button>
      <button class="code-panel-close" onclick="closeCode()">&times;</button>
    </div>
    <div class="code-panel-body">
      <pre><code id="code-content"></code></pre>
    </div>
  </div>
</div>

<script>
(function() {{
  const bar = document.getElementById('filter-bar');
  const cards = document.querySelectorAll('.card');
  const noResults = document.getElementById('no-results');

  bar.addEventListener('click', function(e) {{
    const btn = e.target.closest('.filter-btn');
    if (!btn) return;
    bar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const filter = btn.dataset.filter;
    let visible = 0;

    cards.forEach(card => {{
      if (filter === 'all') {{
        card.classList.remove('hidden');
        visible++;
      }} else {{
        const tags = JSON.parse(card.dataset.tags);
        if (tags.includes(filter)) {{
          card.classList.remove('hidden');
          visible++;
        }} else {{
          card.classList.add('hidden');
        }}
      }}
    }});

    if (visible === 0) {{
      noResults.classList.remove('hidden');
    }} else {{
      noResults.classList.add('hidden');
    }}
  }});
}})();

var _rawCode = '';

function highlightPython(code) {{
  var re = /(&quot;[^&]*?&quot;|'[^'\\n]*?')|(#[^\\n]*)|(\\b(?:import|from|as|def|class|return|if|elif|else|for|in|while|with|try|except|raise|pass|break|continue|and|or|not|is|lambda|yield|global|nonlocal|assert|del)\\b)|(\\b(?:True|False|None)\\b)|(\\b\\d+\\.?\\d*\\b)|([a-zA-Z_]\\w*(?=\\s*\\())/g;
  return code.replace(re, function(m,s,c,k,b,n,f) {{
    if (s) return '<span class="str">'+m+'</span>';
    if (c) return '<span class="cmt">'+m+'</span>';
    if (k) return '<span class="kw">'+m+'</span>';
    if (b) return '<span class="bool">'+m+'</span>';
    if (n) return '<span class="num">'+m+'</span>';
    if (f) return '<span class="fn">'+m+'</span>';
    return m;
  }});
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function showCode(btn) {{
  var raw = btn.getAttribute('data-code');
  _rawCode = raw;
  var card = btn.closest('.card');
  var title = card ? card.querySelector('.card-title') : null;
  document.getElementById('code-panel-title').textContent =
    title ? title.textContent + ' — EdgeCase Code' : 'EdgeCase Code';
  document.getElementById('code-content').innerHTML = highlightPython(escHtml(raw));
  document.getElementById('code-overlay').classList.add('visible');
}}

function closeCode() {{
  document.getElementById('code-overlay').classList.remove('visible');
}}

function copyCode() {{
  navigator.clipboard.writeText(_rawCode).then(function() {{
    var btn = document.querySelector('.code-panel-copy');
    btn.textContent = 'Copied!';
    setTimeout(function() {{ btn.textContent = 'Copy'; }}, 1500);
  }});
}}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeCode();
}});
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Generate SVG renderer gallery")
    parser.add_argument(
        "--out",
        default="svg_gallery.html",
        help="Output HTML file path (default: svg_gallery.html in project root)",
    )
    args = parser.parse_args()

    print("Building gallery items...")
    items = build_gallery_items()
    print(f"  {len(items)} examples to render")

    print("Rendering SVGs and generating HTML...")
    html = generate_gallery_html(items)

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gallery written to {out_path}")
    print("Serve with: python3 -m http.server 8765")


if __name__ == "__main__":
    main()
