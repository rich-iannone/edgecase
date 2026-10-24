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
the scored side (`which=1`, the bottom row, since partition 0 holds A, the lowest id) heat-shaded. Every
number is what edgecase returns for the depicted graph (rounded to 2 decimals).
The two-mode measures from the docstrings (latapy_clustering, node_redundancy,
birank) draw people / events or customers / products with `two_rows`: empty
circles named outside their row, events `g2`, scored nodes heat-shaded, each
figure asserting its values with edgecase.

`effective_size` (the flip side of `constraint`) and `local_constraint` (one
term of it, the queried pair `soft`, the mutual contact and the paths through
it `accent`, a card with the arithmetic) also use the strip graph, as does
`betweenness_centrality_subset`, which outlines its sources and targets
(`soft`, tagged with `s` / `t` badges) like `percolation_centrality` outlines its
percolated nodes.

Directed measures use small hierarchies. `local_reaching_centrality` uses the
five-node out-tree `TREE` (also used by `global_reaching_centrality` in
graph_metrics.py). The trophic methods share ONE food web (`WEB`: grass,
rabbit, mouse, fox, owl, drawn as G, R, M, F, O with a key line) laid out
left to right by trophic level against faint level guides; `trophic_levels`
heat-shades it, `trophic_differences` labels each edge with its level
difference (largest `accent`), and `trophic_incoherence_parameter` (in
graph_metrics.py) reuses that labelled drawing.

`square_clustering` needs squares, not triangles, so it uses a 2 x 3 grid
(`GRID`) and adds a third table column with the triangle clustering
coefficient, 0 for every node, to show what squares catch that triangles miss.

`butterflies` draws its docstring shoppers and products with `two_rows`,
heat-shaded by count, the two nodes in both butterflies `accent`.
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
        "The pendant G, with a single contact, is fully constrained at 1.00, and A, whose two "
        "contacts know each other, follows at 0.84. The bridge E, linking the triangles to "
        "the tail, spans a structural hole and has the lowest constraint, 0.48.")
def constraint(s: Scene) -> None:
    _local(s, "g.constraint()", "C_i",
           {"A": 0.84, "B": 0.72, "C": 0.61, "D": 0.66, "E": 0.48, "F": 0.5, "G": 1.0})


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
        "every other node has state 0. Only paths that start at A or G count, so the nodes "
        "linking A to G matter most: E and F share the top at 0.60, then C 0.50, with B and "
        "D 0.05 and the endpoints A and G 0.")
def percolation_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.percolation_centrality(states)")
    s.text(left.cx, left.y0 + 8, "states = {A: 1, G: 1, others: 0}", size=SIZE_NOTE,
           opacity=0.6)
    gb = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    pos = place(STRIP_COORDS, gb, unit=UNIT)
    scores = {"A": 0.0, "B": 0.05, "C": 0.5, "D": 0.05, "E": 0.6, "F": 0.6, "G": 0.0}
    top = max(scores.values())
    styles = {"A": "soft", "G": "soft", "E": "accent", "F": "accent"}
    s.graph(pos, STRIP_EDGES, heat={k: v / top for k, v in scores.items()}, styles=styles)
    rows = [["node", "score"]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- bipartite centralities on one shared two-row graph ----------------------------

# top row A-D is partition 0 (it holds A, the lowest node id) and the bottom row
# E-G partition 1, so which=1 scores the bottom row
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
    s.text(x, pos["A"][1], "side 0", size=SIZE_NOTE, anchor="end", color="g2")
    s.text(x, pos["E"][1], "side 1", size=SIZE_NOTE, anchor="end", opacity=0.6)
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("bipartite_degree_centrality", "node-metrics", "Bipartite degree centrality",
        "A bipartite graph drawn as two rows: side 0, A to D, on top and side 1, E to G, "
        "below, which which=1 scores. Each degree is divided by the four "
        "nodes on the other side, so F, linked to B, C and D, scores 0.75, E 0.50 and G "
        "0.25.")
def bipartite_degree_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_degree_centrality(which=1)", "score",
               {"E": 0.5, "F": 0.75, "G": 0.25})


@figure("bipartite_closeness_centrality", "node-metrics", "Bipartite closeness centrality",
        "The same two-row bipartite graph, with the bottom side E to G shaded by closeness "
        "to the top side A to D. F reaches B, C and D in one hop and A in three, so it "
        "scores 0.67, ahead of E 0.50 and G 0.33.")
def bipartite_closeness_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_closeness_centrality(which=1)", "score",
               {"E": 0.5, "F": 0.67, "G": 0.33})


@figure("bipartite_betweenness_centrality", "node-metrics", "Bipartite betweenness centrality",
        "The same two-row bipartite graph, with the bottom side E to G shaded by how often "
        "each lies on shortest paths between top-side nodes. F joins B, C and D and scores "
        "2.50, E, the only way out of A, scores 1.33, and G, which only touches D, scores 0.")
def bipartite_betweenness_centrality(s: Scene) -> None:
    _bipartite(s, "g.bipartite_betweenness_centrality(1)", "score",
               {"E": 1.33, "F": 2.5, "G": 0.0})


# --- two-mode metrics on the docstrings' people / events graphs ----------------------
#
# Long names do not fit in a circle, so these graphs draw empty circles in two rows
# with each name set outside its row (above the top row, below the bottom row) and
# the row's side named at its left end. `two_rows` is shared with the other
# bipartite figures (robins_alexander_clustering, biadjacency_matrix,
# is_bipartite_node_set).

PEOPLE = ["ann", "bob", "cy"]
EVENTS = ["party", "talk", "game"]
ATTENDS = [("ann", "party"), ("ann", "talk"), ("bob", "party"), ("bob", "talk"), ("cy", "talk"),
           ("cy", "game")]


def two_rows(s: Scene, box: Box, top: list[str], bottom: list[str], edges: list, *,
             styles: dict | None = None, heat: dict | None = None, prefix: str = "",
             sides: tuple[str, str] | None = None, labels: bool = False,
             unit: float = UNIT) -> dict:
    """A bipartite graph as two rows, `top` above `bottom`, names outside the rows."""
    from figures.similarity import _name

    styles = styles or {}
    coords = {k: (i * 1.4 - (len(top) - 1) * 0.7, 0) for i, k in enumerate(top)}
    coords.update({k: (i * 1.4 - (len(bottom) - 1) * 0.7, 1.25) for i, k in enumerate(bottom)})
    gb = box if labels else Box(box.x0, box.y0 + 16, box.x1, box.y1 - 16)
    pos = place(coords, gb, unit=unit)
    s.graph(pos, edges, styles=styles, heat=heat, labels=labels, prefix=prefix)
    if not labels:
        for k in coords:
            st = styles.get(k, "default")
            color = {"g2": "g2", "g3": "g3", "g4": "g4", "default": "ink", "muted": "ink"}.get(st, "accent")
            _name(s, prefix + k, k, "n" if k in top else "s", color)
    if sides:
        x = min(p[0] for p in pos.values()) - 34
        for name, k in zip(sides, (top[0], bottom[0])):
            s.text(x, pos[k][1], name, size=SIZE_NOTE, anchor="end", opacity=0.6)
    return pos


def _attendance():
    import edgecase as ec

    g = ec.Graph().add_nodes(PEOPLE + EVENTS)
    g.add_edges(ATTENDS)
    return g


@figure("latapy_clustering", "node-metrics", "Latapy bipartite clustering",
        "People ann, bob and cy on top and the events party, talk and game below, all "
        "teal. Each person is shaded by the mean Jaccard overlap of their events with "
        "those of the people two steps away. ann and bob attended the same two events "
        "(overlap 1) and share one of cy's three (1/3), so both score 0.67; cy overlaps each "
        "of them by 1/3 and scores 0.33.")
def latapy_clustering(s: Scene) -> None:
    g = _attendance()
    cc = {g.get_label_from_id(n): round(c, 3)
          for n, c in g.latapy_clustering(nodes=PEOPLE).items()}
    assert cc == {"ann": 0.667, "bob": 0.667, "cy": 0.333}
    left, right = compute(s, "g", "g.latapy_clustering(nodes=people)")
    two_rows(s, left, PEOPLE, EVENTS, ATTENDS,
             styles={**dict.fromkeys(EVENTS, "g2"), "ann": "accent", "bob": "accent"},
             heat={k: v / 0.667 for k, v in cc.items()}, sides=("people", "events"))
    rows = [["node", "overlaps", "score"], ["ann, bob", "1, 1/3", "0.67"],
            ["cy", "1/3, 1/3", "0.33"]]
    _, y0, _, _ = s.table_centered(Box(right.x0, right.y0 + 12, right.x1, right.y1), rows,
                                   [72, 72, 56], highlight={(1, j): "accent" for j in range(3)})
    s.text(right.cx, y0 - 12, "mean Jaccard at distance 2", size=SIZE_NOTE, opacity=0.6)


@figure("node_redundancy", "node-metrics", "Node redundancy",
        "The same people and events graph. ann attended party and talk, and bob attended "
        "both too, so the pair stays linked without ann: ann scores 1.0, and bob, covered "
        "by ann, also 1.0. cy is the only one at both talk and game, so the edges joining "
        "them, in the accent color, are the pair's sole link and cy scores 0.0.")
def node_redundancy(s: Scene) -> None:
    g = _attendance()
    red = {g.get_label_from_id(n): r for n, r in g.node_redundancy(nodes=PEOPLE).items()}
    assert red == {"ann": 1.0, "bob": 1.0, "cy": 0.0}
    left, right = compute(s, "g", "g.node_redundancy(nodes=people)")
    broker = {("cy", "talk"), ("cy", "game")}
    two_rows(s, left, PEOPLE, EVENTS,
             [(a, b, {"style": "accent"}) if (a, b) in broker else (a, b) for a, b in ATTENDS],
             styles={**dict.fromkeys(EVENTS, "g2"), "ann": "accent", "bob": "accent"},
             heat=red, sides=("people", "events"))
    rows = [["node", "event pair", "via", "score"], ["ann", "party, talk", "bob", "1.0"],
            ["bob", "party, talk", "ann", "1.0"], ["cy", "talk, game", "none", "0.0"]]
    hl = {(i, j): "accent" for i in (1, 2) for j in (0, 3)}
    _, y0, _, _ = s.table_centered(Box(right.x0, right.y0 + 12, right.x1, right.y1), rows,
                                   [44, 88, 44, 44], highlight=hl)
    s.text(right.cx, y0 - 12, "pairs also linked by another", size=SIZE_NOTE, opacity=0.6)


# birank docstring: ann and bob each bought tea plus one other product, cy only tea.
# Customers run ann, cy, bob so that no two edges cross.
CUSTOMERS = ["ann", "cy", "bob"]
PRODUCTS = ["jam", "tea", "bread"]
BOUGHT = [("ann", "tea"), ("ann", "jam"), ("bob", "tea"), ("bob", "bread"), ("cy", "tea")]


@figure("birank", "node-metrics", "BiRank",
        "Customers ann, cy and bob on top and products jam, tea and bread below, every "
        "node shaded by its BiRank score; each side sums to 1. tea, bought by all three, "
        "is the top product at 0.464, ahead of jam and bread at 0.268, and ann and bob, who "
        "bought tea plus one more, lead the customers at 0.369, ahead of cy at 0.261.")
def birank(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(["ann", "bob", "cy", "tea", "jam", "bread"])
    g.add_edges(BOUGHT)
    sc = {g.get_label_from_id(n): round(v, 3)
          for n, v in g.birank(top_nodes=["ann", "bob", "cy"]).items()}
    assert sc == {"ann": 0.369, "bob": 0.369, "cy": 0.261, "tea": 0.464, "jam": 0.268,
                  "bread": 0.268}
    left, right = compute(s, "g", "g.birank(top_nodes=customers)")
    two_rows(s, left, CUSTOMERS, PRODUCTS, BOUGHT,
             styles={"ann": "accent", "bob": "accent", "tea": "accent"},
             heat={k: v / 0.464 for k, v in sc.items()}, sides=("customers", "products"))
    rows = [["customer", "score", "product", "score"], ["ann, bob", "0.369", "tea", "0.464"],
            ["cy", "0.261", "jam, bread", "0.268"]]
    s.table_centered(right, rows, [64, 48, 76, 48], highlight={(1, j): "accent" for j in range(4)})


# --- structural holes on the shared strip graph --------------------------------------


@figure("effective_size", "node-metrics", "Effective size",
        "The shared seven-node strip of triangles with a tail, shaded by Burt's effective "
        "size: the number of contacts minus how redundant they are with one another. Hub C "
        "has four contacts that partly know each other and scores 2.50, the bridge E 2.33, "
        "and F, whose two contacts E and G are unlinked, a full 2.00; A and G score 1.00.")
def effective_size(s: Scene) -> None:
    _local(s, "g.effective_size()", "size",
           {"A": 1.0, "B": 1.6667, "C": 2.5, "D": 1.6667, "E": 2.3333, "F": 2.0, "G": 1.0})


@figure("local_constraint", "node-metrics", "Local constraint",
        "The shared seven-node graph with the queried pair A and B outlined. A puts half its "
        "ties into B directly (0.50) and reaches B again through their mutual contact C, "
        "drawn in accent: A's half share in C times C's quarter share in B adds 0.125. The "
        "card squares the sum, (0.50 + 0.125)² = 0.39.")
def local_constraint(s: Scene) -> None:
    left, right = compute(s, "g", 'g.local_constraint("A", "B")')
    pos = place(STRIP_COORDS, left, unit=UNIT)
    paths = {("A", "B"), ("A", "C"), ("B", "C")}
    s.graph(pos, [(a, b, {"style": "accent"}) if (a, b) in paths else (a, b)
                  for a, b in STRIP_EDGES],
            styles={"A": "soft", "B": "soft", "C": "accent"})
    # p_ij = 1 / degree(i) without weights: p_AB = 1/2, p_AC = 1/2, p_CB = 1/4
    s.card_centered(right, ["direct  p(A,B)        0.50", "via C   p(A,C)·p(C,B)  0.125",
                            ("(0.50 + 0.125)²       0.39", "accent")], title="c(A, B)")


# --- traffic between chosen sources and targets ------------------------------------


@figure("betweenness_centrality_subset", "node-metrics", "Subset betweenness centrality",
        "The shared seven-node graph with the sources A and B and the target G outlined; only "
        "shortest paths from A or B to G count. They all leave the triangles through E and "
        "F, which share the top at 1.00, while C scores 0.75 and D 0.25; the endpoints "
        "themselves score 0.")
def betweenness_centrality_subset(s: Scene) -> None:
    left, right = compute(s, "g", "g.betweenness_centrality_subset(S, T)")
    s.text(left.cx, left.y0 + 8, "S = [A, B]    T = [G]", size=SIZE_NOTE, opacity=0.6)
    gb = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    pos = place(STRIP_COORDS, gb, unit=UNIT)
    # raw values (normalized=False), halved on undirected graphs as in NetworkX
    scores = {"A": 0.0, "B": 0.0, "C": 0.75, "D": 0.25, "E": 1.0, "F": 1.0, "G": 0.0}
    styles = {"A": "soft", "B": "soft", "G": "soft", "E": "accent", "F": "accent"}
    s.graph(pos, STRIP_EDGES, heat=scores, styles=styles)
    s.badge("A", "s", where="sw")
    s.badge("B", "s", where="nw")
    s.badge("G", "t", where="ne")
    rows = [["node", "score"]] + _rows(scores)
    s.table_centered(right, rows, [80, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- reach in a directed hierarchy ---------------------------------------------------

# a five-node out-tree (a reporting line): A heads B and C, B heads D and E
TREE = {"A": (1, 0), "B": (0.4, 1), "C": (1.6, 1), "D": (0, 2), "E": (0.8, 2)}
TREE_ARCS = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")]


@figure("local_reaching_centrality", "node-metrics", "Local reaching centrality",
        "A five-node directed out-tree, like an org chart: A heads B and C, and B heads D "
        "and E. Each node is shaded by the share of the other four nodes it can reach. A "
        "reaches all of them and scores 1.00, B reaches D and E for 0.50, and the leaves C, "
        "D and E reach no one and score 0.")
def local_reaching_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.local_reaching_centrality()")
    pos = place(TREE, left, unit=UNIT)
    scores = {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0, "E": 0.0}
    s.graph(pos, TREE_ARCS, directed=True, heat=scores, styles={"A": "accent"})
    rows = [["node", "score"]] + _rows(scores)
    s.table_centered(right, rows, [80, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- trophic levels in a small food web -----------------------------------------------

# grass, rabbit, mouse, fox, owl; arcs point from prey to predator. The x
# coordinate is the trophic level, so the web climbs left to right.
WEB_LEVEL = {"G": 1.0, "R": 2.0, "M": 2.0, "F": 3.0, "O": 3.5}
WEB = {"G": (1, 0.5), "R": (2, 0), "M": (2, 1), "F": (3, 0), "O": (3.5, 1)}
WEB_ARCS = [("G", "R"), ("G", "M"), ("R", "F"), ("M", "F"), ("M", "O"), ("F", "O")]
WEB_DIFF = {("G", "R"): 1.0, ("G", "M"): 1.0, ("R", "F"): 1.0, ("M", "F"): 1.0,
            ("M", "O"): 1.5, ("F", "O"): 0.5}
WEB_KEY = "G grass · R rabbit · M mouse · F fox · O owl"
WEB_UNIT = 96


def food_web(s: Scene, box: Box, *, heat: dict[str, float] | None = None,
             styles: dict | None = None, diffs: bool = False, prefix: str = "") -> None:
    """The shared food web: level ticks on top, web by level, species key below."""
    gb = Box(box.x0, box.y0 + 16, box.x1, box.y1 - 16)
    pos = place(WEB, gb, unit=WEB_UNIT)
    top = max(WEB_DIFF.values())
    arcs = []
    for a, b in WEB_ARCS:
        kw: dict = {}
        if diffs:
            d = WEB_DIFF[(a, b)]
            kw["weight"] = f"{d:g}"
            if d == top:
                kw["style"] = "accent"
        arcs.append((a, b, kw))
    # faint vertical guide per level, its value on top
    ys = [y for _, y in pos.values()]
    y0, y1 = min(ys) - 24, max(ys) + 24
    for lv in sorted(set(WEB_LEVEL.values())):
        x = next(pos[k][0] for k, v in WEB_LEVEL.items() if v == lv)
        s.line(x, y0, x, y1, opacity=0.18, width=1, dash="3 4", role="rule")
        s.text(x, box.y0 + 8, f"{lv:g}", size=SIZE_NOTE, opacity=0.6)
    gx = pos["G"][0]
    s.text(gx - 22, box.y0 + 8, "level", size=SIZE_NOTE, anchor="end", opacity=0.6)
    s.text(box.cx, box.y1 - 6, WEB_KEY, size=SIZE_NOTE, opacity=0.6)
    s.graph(pos, arcs, directed=True, heat=heat, styles=styles, prefix=prefix)


@figure("trophic_levels", "node-metrics", "Trophic levels",
        "A food web drawn left to right by trophic level: grass G feeds rabbit R and mouse "
        "M, both feed fox F, and mouse and fox feed owl O. Grass is basal at level 1, the "
        "rabbit and mouse sit at 2, the fox at 3, and the owl, eating at levels 2 and 3, at "
        "3.5, the top of the table.")
def trophic_levels(s: Scene) -> None:
    left, right = compute(s, "g", "g.trophic_levels()")
    food_web(s, left, heat={k: v / 3.5 for k, v in WEB_LEVEL.items()}, styles={"O": "accent"})
    rows = [["node", "level"]] + _rows(WEB_LEVEL)
    s.table_centered(right, rows, [80, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("trophic_differences", "node-metrics", "Trophic differences",
        "The same food web, laid out by trophic level, with each edge labelled by the level "
        "of the predator minus the level of the prey. Mouse to owl jumps 1.5 levels and is "
        "drawn in accent, fox to owl only 0.5, and the other four edges exactly 1.")
def trophic_differences(s: Scene) -> None:
    left, right = compute(s, "g", "g.trophic_differences()")
    food_web(s, left, diffs=True)
    s.card_centered(right, [("M→O  3.5 − 2  1.5", "accent"), "F→O  3.5 − 3  0.5",
                            "others        1.0"], title="level(head) − level(tail)")


# --- squares instead of triangles ------------------------------------------------------

# A 2 x 3 grid (a ladder of two squares): no triangles at all, so ordinary
# clustering is 0 everywhere and only square clustering tells nodes apart.
GRID = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (0, 1), "E": (1, 1), "F": (2, 1)}
GRID_EDGES = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("A", "D"), ("B", "E"),
              ("C", "F")]


@figure("square_clustering", "node-metrics", "Square clustering",
        "A 2 by 3 grid, two squares side by side, has no triangles, so the ordinary "
        "clustering coefficient, listed in the last column, is 0 for every node. Square "
        "clustering sees the squares: the corners A, C, D and F score 0.50, and the middle "
        "nodes B and E 0.33, since one of their three neighbor pairs closes no square.")
def square_clustering(s: Scene) -> None:
    left, right = compute(s, "g", "g.square_clustering()")
    scores = {"A": 0.5, "B": 0.3333, "C": 0.5, "D": 0.5, "E": 0.3333, "F": 0.5}
    s.graph(place(GRID, left, unit=UNIT), GRID_EDGES, heat={k: v / 0.5 for k, v in scores.items()},
            styles={k: "accent" for k, v in scores.items() if v == 0.5})
    rows = [["node", "square", "C(v)"]] + [r + ["0"] for r in _rows(scores)]
    s.table_centered(right, rows, [84, 64, 52], highlight={(1, 0): "accent", (1, 1): "accent"})


# butterflies docstring: shoppers and the products they buy. The two butterflies
# (4-cycles) are ann, bob x tea, jam and bob, cy x jam, oat.
SHOPPERS = ["ann", "bob", "cy"]
GOODS = ["tea", "jam", "oat"]
BASKETS = [("ann", "tea"), ("ann", "jam"), ("bob", "tea"), ("bob", "jam"), ("bob", "oat"),
           ("cy", "oat"), ("cy", "jam")]


@figure("butterflies", "node-metrics", "Butterflies",
        "Shoppers ann, bob and cy on top and the products tea, jam and oat below. A butterfly "
        "is a 4-cycle, two shoppers who both bought the same two products: ann and bob with "
        "tea and jam, and bob and cy with jam and oat. Nodes are shaded by how many "
        "butterflies they belong to; bob and jam, in the accent color, are in both and count "
        "2, every other node counts 1.")
def butterflies(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(SHOPPERS + GOODS).add_edges(BASKETS)
    counts = {g.get_label_from_id(n): c for n, c in g.butterflies().items()}
    assert counts == {"ann": 1, "bob": 2, "cy": 1, "tea": 1, "jam": 2, "oat": 1}
    left, right = compute(s, "g", "g.butterflies()")
    two_rows(s, left, SHOPPERS, GOODS, BASKETS, styles={"bob": "accent", "jam": "accent"},
             heat={k: v / 2 for k, v in counts.items()}, sides=("shoppers", "products"))
    s.table_centered(right, [["node", "butterflies"]] + _rows(counts, digits=0), [120, 96],
                     highlight={(1, 0): "accent", (1, 1): "accent"})
