"""Node metrics, local and path-based variants: a score for every node.

Same recipe as `node_metrics.py` (template `compute`): left, the graph with
every node shaded by its score (`heat`: darker = higher) and the top node
`accent`-outlined; right, a table of node → score sorted high to low, top row
highlighted, tied nodes sharing a row.

All whole-graph measures here reuse the shared strip graph from
`node_metrics.py` (`STRIP_*`), so they can be compared with degree,
betweenness and the rest. Scores that can be negative (leverage) are shaded
from the minimum up. The two clustering methods differ only for nodes of
degree < 2: `local_clustering_coefficient` returns None (node `muted`, a
`None` row), `clustering_coefficient` returns 0.0 (tied with F).

The three `bipartite_*` centralities share ONE small bipartite graph
(`BIP`) drawn as two rows: the opposite side `g2` (as in `is_bipartite`),
the scored side (`which=0`, the bottom row on this graph) heat-shaded. Every
number is what edgecase returns for the depicted graph (rounded to 2 decimals).
"""

from core import SIZE_NOTE, Box, Scene, compute, figure, place

from figures.node_metrics import STRIP_COORDS, STRIP_EDGES, UNIT, _rows


def _heat(scores: dict[str, float | None]) -> dict[str, float]:
    """Heat in [0, 1]; shaded from the minimum when scores go negative."""
    vals = [v for v in scores.values() if v is not None]
    lo, hi = min(min(vals), 0), max(vals)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items() if v is not None}


def _local(s: Scene, call: str, header: str, scores: dict[str, float | None],
           box: Box | None = None) -> None:
    """Heat-shaded strip graph → sorted table; None scores get a muted node and a row."""
    left, right = compute(s, "g", call)
    pos = place(STRIP_COORDS, box or left, unit=UNIT)
    top = max(v for v in scores.values() if v is not None)
    styles = {k: "accent" for k, v in scores.items() if v == top}
    styles.update({k: "muted" for k, v in scores.items() if v is None})
    s.graph(pos, STRIP_EDGES, heat=_heat(scores), styles=styles)
    rows = [["node", header]] + _rows({k: v for k, v in scores.items() if v is not None})
    rows += [[k, "None"] for k, v in sorted(scores.items()) if v is None]
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- whole-graph measures on the shared strip graph -------------------------------


@figure("constraint", "node-metrics", "Burt's constraint",
        "The shared seven-node strip of triangles with a tail, shaded by Burt's constraint. "
        "Hub C, whose contacts are all linked to each other, is the most constrained at "
        "1.20; the tail node F, the only link between E and G, sits in a structural hole "
        "and has the lowest constraint, 0.50.")
def constraint(s: Scene) -> None:
    _local(s, "g.constraint()", "C_i",
           {"A": 1.13, "B": 1.14, "C": 1.2, "D": 1.14, "E": 0.81, "F": 0.5, "G": 1.0})


@figure("leverage_centrality", "node-metrics", "Leverage centrality",
        "The shared seven-node graph shaded by leverage centrality, which compares each "
        "node's degree with its neighbors' degrees. Hub C out-connects its neighbors and "
        "tops the table at 0.19, while A and the pendant G have better-connected neighbors "
        "and score negative, -0.27 and -0.33.")
def leverage_centrality(s: Scene) -> None:
    _local(s, "g.leverage_centrality()", "score",
           {"A": -0.27, "B": 0.02, "C": 0.19, "D": -0.05, "E": 0.02, "F": 0.07, "G": -0.33})


@figure("load_centrality", "node-metrics", "Load centrality",
        "The shared seven-node graph shaded by load centrality, the share of shortest-path "
        "traffic routed through each node. Bridge node E carries the most load at 0.53, "
        "then C and F at 0.33; on this graph the scores match betweenness centrality "
        "exactly.")
def load_centrality(s: Scene) -> None:
    _local(s, "g.load_centrality()", "load",
           {"A": 0.0, "B": 0.03, "C": 0.33, "D": 0.1, "E": 0.53, "F": 0.33, "G": 0.0})


@figure("local_clustering_coefficient", "node-metrics", "Local clustering coefficient",
        "The shared seven-node graph shaded by local clustering, the fraction of each "
        "node's neighbor pairs that are linked. A's two neighbors are linked, so A scores "
        "1.00, down to F at 0; the pendant G has a single neighbor, so its value is "
        "undefined, drawn faded and listed as None.")
def local_clustering_coefficient(s: Scene) -> None:
    _local(s, "g.local_clustering_coefficient()", "C(v)",
           {"A": 1.0, "B": 0.67, "C": 0.5, "D": 0.67, "E": 0.33, "F": 0.0, "G": None})


@figure("clustering_coefficient", "node-metrics", "Clustering coefficient",
        "The shared seven-node graph shaded by each node's clustering coefficient. A's two "
        "neighbors are linked, so A scores 1.00, then B and D 0.67, C 0.50 and E 0.33. "
        "Nodes with fewer than two neighbors get 0 rather than None, so the pendant G ties "
        "with F at 0.")
def clustering_coefficient(s: Scene) -> None:
    _local(s, "g.clustering_coefficient()", "C(v)",
           {"A": 1.0, "B": 0.67, "C": 0.5, "D": 0.67, "E": 0.33, "F": 0.0, "G": 0.0})


@figure("bridging_centrality", "node-metrics", "Bridging centrality",
        "The shared seven-node graph shaded by bridging centrality, betweenness weighted "
        "by how much a node links otherwise separate parts. Bridge node E scores 1.00 "
        "after normalizing, the tail node F 0.76 and the hub C only 0.34, while A and G "
        "score 0.")
def bridging_centrality(s: Scene) -> None:
    _local(s, "g.bridging_centrality()", "score",
           {"A": 0.0, "B": 0.06, "C": 0.34, "D": 0.22, "E": 1.0, "F": 0.76, "G": 0.0})


@figure("closeness_vitality", "node-metrics", "Closeness vitality",
        "The shared seven-node graph shaded by closeness vitality, how much the total "
        "pairwise distance drops when a node is removed. Removing the bridge E cuts off "
        "the tail, so E tops the table at 66, while the hub C, which the others can route "
        "around, scores only 12.")
def closeness_vitality(s: Scene) -> None:
    # pairs left unreachable by the removal stop contributing, so cut nodes score high
    _local(s, "g.closeness_vitality()", "vitality",
           {"A": 26, "B": 24, "C": 12, "D": 20, "E": 66, "F": 56, "G": 34})


@figure("percolation_centrality", "node-metrics", "Percolation centrality",
        "The shared seven-node graph with the two percolated nodes, A and G, outlined; "
        "every other node has state 0. Only paths that start or end at A or G count, so "
        "the tail node F, the sole way to G, tops the table at 12.00, ahead of C 9.50 and "
        "E 8.00.")
def percolation_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.percolation_centrality(states)")
    s.text(left.cx, left.y0 + 8, "states = {A: 1, G: 1, others: 0}", size=SIZE_NOTE,
           opacity=0.6)
    gb = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    pos = place(STRIP_COORDS, gb, unit=UNIT)
    scores = {"A": 0.0, "B": 1.0, "C": 9.5, "D": 0.5, "E": 8.0, "F": 12.0, "G": 0.0}
    top = max(scores.values())
    styles = {"A": "soft", "G": "soft", "F": "accent"}
    s.graph(pos, STRIP_EDGES, heat={k: v / top for k, v in scores.items()}, styles=styles)
    rows = [["node", "score"]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- bipartite centralities on one shared two-row graph ----------------------------

# top row A-D is partition 1, bottom row E-G partition 0 (edgecase's 2-coloring
# on this graph), so the default which=0 scores the bottom row
BIP = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0),
       "E": (0.5, 1.2), "F": (2, 1.2), "G": (3, 1.2)}
BIP_EDGES = [("A", "E"), ("B", "E"), ("B", "F"), ("C", "F"), ("D", "F"), ("D", "G")]


def _bipartite(s: Scene, call: str, header: str, scores: dict[str, float]) -> None:
    left, right = compute(s, "g", call)
    pos = place(BIP, left, unit=UNIT)
    top = max(scores.values())
    styles = {k: "g2" for k in "ABCD"}
    styles.update({k: "accent" for k, v in scores.items() if v == top})
    s.graph(pos, BIP_EDGES, heat={k: v / top for k, v in scores.items()}, styles=styles)
    # name the two sides at the left end of each row
    x = min(p[0] for p in pos.values()) - 30
    s.text(x, pos["A"][1], "side 1", size=SIZE_NOTE, anchor="end", color="g2")
    s.text(x, pos["E"][1], "side 0", size=SIZE_NOTE, anchor="end", opacity=0.6)
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("bipartite_degree_centrality", "node-metrics", "Bipartite degree centrality",
        "A bipartite graph drawn as two rows: side 1, A to D, on top and side 0, E to G, "
        "below, which the default which=0 scores. Each degree is divided by the four "
        "nodes on the other side, so F, linked to B, C and D, scores 0.75, E 0.50 and G "
        "0.25.")
def bipartite_degree_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_degree_centrality()", "score",
               {"E": 0.5, "F": 0.75, "G": 0.25})


@figure("bipartite_closeness_centrality", "node-metrics", "Bipartite closeness centrality",
        "The same two-row bipartite graph, with the bottom side E to G shaded by closeness "
        "to the top side A to D. F reaches B, C and D in one hop and A in three, so it "
        "scores 0.67, ahead of E 0.50 and G 0.33.")
def bipartite_closeness_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_closeness_centrality()", "score",
               {"E": 0.5, "F": 0.67, "G": 0.33})


@figure("bipartite_betweenness_centrality", "node-metrics", "Bipartite betweenness centrality",
        "The same two-row bipartite graph, with the bottom side E to G shaded by how often "
        "each lies on shortest paths between top-side nodes. F joins B, C and D and scores "
        "2.50, E, the only way out of A, scores 1.33, and G, which only touches D, scores 0.")
def bipartite_betweenness_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_betweenness_centrality()", "score",
               {"E": 1.33, "F": 2.5, "G": 0.0})
