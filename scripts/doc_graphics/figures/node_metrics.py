"""Node metrics: a score for every node.

Template: `compute`. Left: the graph, every node shaded by its score (`heat`:
darker = higher), the top node also `accent`-outlined. Right: a small table of
node → score sorted high to low, top row highlighted; tied nodes share a row.
The shading and the table always agree, so a reader can check one against the
other.

All whole-graph metrics use ONE shared graph (`STRIP_*`: a strip of three
triangles A–E with a two-node tail F–G) so sibling measures can be compared
directly: degree, harmonic and PageRank favor the hub C; betweenness and
radiality favor the bridge E; average neighbor degree favors A; eccentricity
is largest at the periphery (A, B, G).

Single-node getters (`get_degree`, `get_in_degree`, `get_out_degree`,
`get_node_vitality`) show the queried node `soft`, the counted elements
`accent`, and a card with the value. Every number is what edgecase returns
for the depicted graph (rounded to 2 decimals).
"""

from core import Scene, compute, figure, place

UNIT = 80  # one scale for the whole family

# --- the shared undirected graph --------------------------------------------

STRIP_COORDS = {"A": (0, 1), "B": (0.6, 0), "C": (1.2, 1), "D": (1.8, 0), "E": (2.4, 1),
                "F": (3.4, 0.5), "G": (4.4, 0.5)}
STRIP_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
               ("D", "E"), ("E", "F"), ("F", "G")]


def _fmt(v: float) -> str:
    return str(v) if isinstance(v, int) else f"{v:.2f}"


def _rows(scores: dict[str, float], max_rows: int = 7) -> list[list[str]]:
    """Table body sorted high→low; ties share a row; long tails end in '…'.

    Seven rows (every node of the shared graph) still fit the content area.
    """
    groups: dict[str, list[str]] = {}
    for k, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        groups.setdefault(_fmt(v), []).append(k)
    rows = [[", ".join(ks), v] for v, ks in groups.items()]
    if len(rows) > max_rows:
        rows = rows[:max_rows - 1] + [["…", "…"]]
    return rows


def _metric(s: Scene, call: str, header: str, scores: dict[str, float],
            col_w: tuple[float, float] = (64, 64)) -> None:
    """The family template: heat-shaded strip graph → sorted score table."""
    left, right = compute(s, "g", call)
    pos = place(STRIP_COORDS, left, unit=UNIT)
    top = max(scores.values())
    s.graph(pos, STRIP_EDGES, heat={k: v / top for k, v in scores.items()},
            styles={k: "accent" for k, v in scores.items() if v == top})
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, list(col_w), highlight={(1, 0): "accent", (1, 1): "accent"})


# --- whole-graph node metrics on the shared graph ------------------------------


@figure("degrees", "node-metrics", "Degree of every node",
        "A seven-node graph, a strip of three triangles A to E with a tail F, G, each node "
        "shaded by its degree. Hub C has the most neighbors and the darkest shading, and the "
        "table lists C 4, then B, D, E 3, A, F 2, and G 1.")
def degrees(s: Scene) -> None:
    _metric(s, "g.degrees()", "degree",
            {"A": 2, "B": 3, "C": 4, "D": 3, "E": 3, "F": 2, "G": 1})


@figure("betweenness_centrality", "node-metrics", "Betweenness centrality",
        "The same seven-node strip of triangles with a tail, shaded by betweenness. Bridge "
        "node E lies on every shortest path between the triangles and the tail F, G, so it "
        "tops the table at 0.53, ahead of the hub C and the tail node F at 0.33; A and G "
        "score 0.")
def betweenness_centrality(s: Scene) -> None:
    # normalized by default
    _metric(s, "g.betweenness_centrality()", "score",
            {"A": 0.0, "B": 0.03, "C": 0.33, "D": 0.1, "E": 0.53, "F": 0.33, "G": 0.0})


@figure("harmonic_centrality", "node-metrics", "Harmonic centrality",
        "The shared seven-node graph shaded by harmonic centrality, the sum of 1 over the "
        "distance to every other node. Hub C is closest to everyone and scores 4.83, "
        "followed by E 4.50, D 4.33 and B 4.08, while the pendant node G is "
        "palest.")
def harmonic_centrality(s: Scene) -> None:
    _metric(s, "g.harmonic_centrality()", "score",
            {"A": 3.58, "B": 4.08, "C": 4.83, "D": 4.33, "E": 4.5, "F": 3.67, "G": 2.67})


@figure("pagerank", "node-metrics", "PageRank",
        "The shared seven-node graph shaded by PageRank. Hub C, linked from four neighbors, "
        "collects the most rank at 0.20, followed by E 0.17, D 0.16 and B 0.15, while the "
        "pendant node G gets the least.")
def pagerank(s: Scene) -> None:
    _metric(s, "g.pagerank()", "rank",
            {"A": 0.11, "B": 0.15, "C": 0.2, "D": 0.16, "E": 0.17, "F": 0.14, "G": 0.08})


@figure("radiality", "node-metrics", "Radiality",
        "The shared seven-node graph shaded by radiality, which rises as eccentricity falls. "
        "Bridge node E reaches every node within two hops and scores 0.50; C, D and F score "
        "0.33; the outer nodes A, B and G score 0.17.")
def radiality(s: Scene) -> None:
    _metric(s, "g.radiality()", "score",
            {"A": 0.17, "B": 0.17, "C": 0.33, "D": 0.33, "E": 0.5, "F": 0.33, "G": 0.17})


@figure("eccentricity", "node-metrics", "Eccentricity",
        "The shared seven-node graph shaded by eccentricity, the distance to the farthest "
        "node. The outer nodes A, B and G are four hops from their farthest node and are "
        "darkest; C, D and F have eccentricity 3, and the bridge E only 2.")
def eccentricity(s: Scene) -> None:
    _metric(s, "g.eccentricity()", "ecc",
            {"A": 4, "B": 4, "C": 3, "D": 3, "E": 2, "F": 3, "G": 4})


@figure("average_neighbor_degree", "node-metrics", "Average neighbor degree",
        "The shared seven-node graph shaded by the mean degree of each node's neighbors. "
        "A has only two neighbors, but they are B (degree 3) and the hub C (degree 4), so A "
        "tops the table at 3.50, ahead of D 3.33, while the hub C itself scores 2.75.")
def average_neighbor_degree(s: Scene) -> None:
    _metric(s, "g.average_neighbor_degree()", "avg",
            {"A": 3.5, "B": 3.0, "C": 2.75, "D": 3.33, "E": 3.0, "F": 2.0, "G": 2.0})


@figure("local_efficiency", "node-metrics", "Local efficiency",
        "The shared seven-node graph shaded by each node's local efficiency, how well its "
        "neighbors stay connected without it: A 1.00, B and D 0.83, C 0.72, E 0.33, F and G "
        "0. The method returns the mean over all nodes, 0.53, highlighted at the bottom of "
        "the table.")
def local_efficiency(s: Scene) -> None:
    left, right = compute(s, "g", "g.local_efficiency()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    per_node = {"A": 1.0, "B": 0.83, "C": 0.72, "D": 0.83, "E": 0.33, "F": 0.0, "G": 0.0}
    # per-node values are context; the returned scalar is their mean, so only
    # the mean row is accented (no single node is "the result")
    s.graph(pos, STRIP_EDGES, heat=per_node)
    rows = [["node", "E_loc"]] + _rows(per_node) + [["mean", "0.53"]]
    n = len(rows) - 1
    s.table_centered(right, rows, [64, 64], highlight={(n, 0): "accent", (n, 1): "accent"})


# --- single-node getters ----------------------------------------------------------


@figure("get_degree", "node-metrics", "Degree of one node",
        "The shared seven-node graph with the queried node C outlined and its four incident "
        "edges, to A, B, D and E, drawn in accent. The card lists those neighbors and the "
        "result, degree 4.")
def get_degree(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_degree("C")')
    pos = place(STRIP_COORDS, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "accent"}) if "C" in (a, b) else (a, b)
                  for a, b in STRIP_EDGES], styles={"C": "soft"})
    s.card_centered(right, ["neighbors  A, B, D, E", ("degree     4", "accent")])


# directed five-node graph for in/out degree; B and C form a reciprocal pair
DIR_COORDS = {"A": (0, 0), "B": (0, 1.2), "C": (1.2, 0.6), "D": (2.4, 0), "E": (2.4, 1.2)}
DIR_IN = [("A", "C"), ("B", "C", 9), ("E", "C")]
DIR_OUT = [("C", "D"), ("C", "B", 9)]
DIR_OTHER = [("A", "B"), ("D", "E")]


def _directed(s: Scene, call: str, counted: list, card: list) -> None:
    left, right = compute(s, "g", call)
    pos = place(DIR_COORDS, left, unit=UNIT)
    s.graph(pos, [], styles={"C": "soft"})
    for e in DIR_IN + DIR_OUT + DIR_OTHER:
        s.edge(e[0], e[1], directed=True, bend=e[2] if len(e) > 2 else 0.0,
               style="accent" if e in counted else "default")
    s.card_centered(right, card)


@figure("get_in_degree", "node-metrics", "In-degree of one node",
        "A five-node directed graph with the queried node C outlined. The three arrows "
        "pointing into C, from A, B and E, are drawn in accent; the arrows C sends to B and "
        "D are not counted. The card reads in-degree 3.")
def get_in_degree(s: Scene) -> None:
    _directed(s, 'g.get_in_degree("C")', DIR_IN,
              ["from       A, B, E", ("in-degree  3", "accent")])


@figure("get_out_degree", "node-metrics", "Out-degree of one node",
        "The same five-node directed graph with the queried node C outlined. The two arrows "
        "leaving C, to B and D, are drawn in accent; the three arrows arriving at C are not "
        "counted. The card reads out-degree 2.")
def get_out_degree(s: Scene) -> None:
    _directed(s, 'g.get_out_degree("C")', DIR_OUT,
              ["to          B, D", ("out-degree  2", "accent")])


@figure("get_node_vitality", "node-metrics", "Node vitality",
        "The shared seven-node graph with the queried hub C outlined and its four edges "
        "dashed, as if removed. The neighbors can still reach each other around C, so the "
        "total pairwise distance only drops from 82 to 70 and the card reads vitality 12.")
def get_node_vitality(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_node_vitality("C")')
    pos = place(STRIP_COORDS, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "removed"}) if "C" in (a, b) else (a, b)
                  for a, b in STRIP_EDGES], styles={"C": "soft"})
    # W = sum of shortest-path distances over ordered pairs (Wiener index)
    s.card_centered(right, ["W(G)       82", "W(G − C)   70", ("vitality   12", "accent")])


# --- per-edge metric ------------------------------------------------------------


@figure("get_edge_clustering_coefficient", "node-metrics", "Edge clustering coefficient",
        "The shared seven-node graph with edge C-D in accent and the two triangles through "
        "it, B-C-D and C-D-E, shaded. With endpoint degrees 4 and 3 at most 2 triangles are "
        "possible, so the default score is (2 + 1) / 2 = 1.5, shown in the card.")
def get_edge_clustering_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_clustering_coefficient()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    s.hull(["B", "C", "D"])
    s.hull(["C", "D", "E"])
    s.graph(pos, [(a, b, {"style": "accent"}) if (a, b) == ("C", "D") else (a, b)
                  for a, b in STRIP_EDGES], styles={"C": "soft", "D": "soft"})
    # offset=True by default: (z + 1) / s with s = min(d_C - 1, d_D - 1)
    s.card_centered(right, ["triangles z  2", "possible s   2",
                            ("(z + 1) / s  1.5", "accent")], title="edge C–D")
