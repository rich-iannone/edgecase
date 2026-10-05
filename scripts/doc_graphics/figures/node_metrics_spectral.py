"""Node metrics, spectral and walk-based: eigenvector-family, HITS and friends.

Template: `compute`, the same recipe as `node_metrics.py`. Left: the graph,
every node shaded by its score (`heat`: darker = higher), the top node also
`accent`-outlined. Right: a small table of node → score sorted from most to
least central, top row highlighted; tied nodes share a row.

Undirected measures reuse the shared strip graph from `node_metrics.py`
(three triangles A–E with a tail F–G), so they can be compared directly with
degree, betweenness and PageRank: the eigenvector family (eigen, Bonacich,
Laplacian) and DomiRank favor the hub C, while communicability betweenness
and second-order centrality favor the bridge E, like betweenness does.
Subgraph centrality (closed walks) favors the hub C; the electrical
current-flow measures favor the bridge E, current-flow closeness only just
(its table uses 3 decimals so E and C do not round to a tie).
`second_order_centrality` is inverted (lower = more central), so its table is
sorted low → high and the lowest value is the top node.

Measures that only differ on directed graphs share ONE small digraph
(`HITS_*`: A points to B, C and D, B and C point to D, and D points to E):
A is the best hub, D the best authority; alpha centrality counts outgoing
walks (A highest) and Katz incoming walks (D highest).

`personalized_pagerank` draws its seed `soft`, as the diffusion family does;
seeds are keyed by node id in the API, so the caption shows the real call
(`{1: 1.0}`) and the seed carries an `id 1` pill. `knotty_centrality` returns a set, not scores, so it
shows the connective core in `accent` inside a hull with a card of its score.
`fiedler_vector` has signed values, so it uses the signed heat: positive
entries shaded accent, negative ones g2 (darker = larger magnitude), on
graph_metrics' PAIR (two triangles joined by C–D) like the other split
methods; its table cells take the same two colors. `spectral_ordering`
returns an order, so it uses traversal-style filled order badges (1 = first)
on a path whose node ids were shuffled, with a card mapping ids to labels.
Every number is what edgecase returns for the depicted graph (2 decimals).
"""

from core import Scene, compute, figure, place

from figures.graph_metrics import PAIR, PAIR_EDGES
from figures.node_metrics import STRIP_COORDS, STRIP_EDGES, UNIT, _metric, _rows

# --- the shared directed graph ---------------------------------------------------

HITS_COORDS = {"A": (0, 0.6), "B": (1.2, 0), "C": (1.2, 1.2), "D": (2.4, 0.6), "E": (3.6, 0.6)}
HITS_ARCS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D"), ("D", "E")]


def _directed(s: Scene, call: str, header: str, scores: dict[str, float]) -> None:
    """`_metric` on the shared digraph."""
    left, right = compute(s, "g", call)
    pos = place(HITS_COORDS, left, unit=UNIT)
    top = max(scores.values())
    s.graph(pos, HITS_ARCS, directed=True, heat={k: v / top for k, v in scores.items()},
            styles={k: "accent" for k, v in scores.items() if v == top})
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- eigenvector family on the shared strip graph ----------------------------------


@figure("eigen_centrality", "node-metrics", "Eigenvector centrality",
        "The shared seven-node strip of three triangles with a tail, each node shaded by "
        "eigenvector centrality. Hub C sits among the best-connected nodes and scores 0.55, "
        "ahead of D 0.46 and B 0.45, while the tail fades from F 0.15 to G 0.05.")
def eigen_centrality(s: Scene) -> None:
    _metric(s, "g.eigen_centrality()", "score",
            {"A": 0.3335, "B": 0.4497, "C": 0.5467, "D": 0.4630, "E": 0.3868, "F": 0.1458,
             "G": 0.0488})


@figure("bonacich_power_centrality", "node-metrics", "Bonacich power centrality",
        "The shared seven-node graph shaded by Bonacich power centrality with the default "
        "beta 0.1, where ties to powerful neighbors add power. Hub C scores 1.45, then D "
        "1.14, B 1.11 and E 1.10, down to the pendant node G at 0.32.")
def bonacich_power_centrality(s: Scene) -> None:
    _metric(s, "g.bonacich_power_centrality()", "power",
            {"A": 0.7740, "B": 1.1133, "C": 1.4488, "D": 1.1431, "E": 1.1019, "F": 0.6605,
             "G": 0.3249})


@figure("laplacian_centrality", "node-metrics", "Laplacian centrality",
        "The shared seven-node graph shaded by Laplacian centrality, the share of the "
        "graph's Laplacian energy lost when a node is removed. Removing hub C costs the most, "
        "0.24, then D 0.18, B and E 0.17, while the pendant node G costs only 0.03.")
def laplacian_centrality(s: Scene) -> None:
    _metric(s, "g.laplacian_centrality()", "score",
            {"A": 0.1149, "B": 0.1724, "C": 0.2414, "D": 0.1839, "E": 0.1724, "F": 0.0805,
             "G": 0.03448})


@figure("domirank_centrality", "node-metrics", "DomiRank centrality",
        "The shared seven-node graph shaded by DomiRank, which rewards nodes that dominate "
        "their neighborhood. Hub C dominates at 1.17, and the tail node F, whose neighbors E "
        "and G it outranks, comes second at 0.71, ahead of B 0.60; A ends lowest at 0.10.")
def domirank_centrality(s: Scene) -> None:
    _metric(s, "g.domirank_centrality()", "score",
            {"A": 0.1049, "B": 0.6002, "C": 1.1707, "D": 0.4141, "E": 0.3252, "F": 0.7054,
             "G": 0.1349})


@figure("subgraph_centrality", "node-metrics", "Subgraph centrality",
        "The shared seven-node strip of triangles with a tail, shaded by subgraph "
        "centrality, which counts closed walks that start and end at each node. Hub C sits "
        "in three triangles and scores 6.17, ahead of D 4.74 and B 4.67, while the tail "
        "nodes F 2.32 and G 1.59, in no triangle, score least.")
def subgraph_centrality(s: Scene) -> None:
    _metric(s, "g.subgraph_centrality()", "score",
            {"A": 3.1799, "B": 4.6658, "C": 6.1744, "D": 4.7442, "E": 4.0391, "F": 2.3162,
             "G": 1.5929})


# --- walk-based betweenness on the shared strip graph ------------------------------


@figure("communicability_betweenness_centrality", "node-metrics",
        "Communicability betweenness centrality",
        "The shared seven-node graph shaded by communicability betweenness, the share of "
        "walk-based communication between other pairs that runs through each node. The "
        "bridge E scores 0.63 and the hub C 0.56, while A 0.12 and the pendant G 0.02 carry "
        "almost nothing.")
def communicability_betweenness_centrality(s: Scene) -> None:
    _metric(s, "g.communicability_betweenness_centrality()", "score",
            {"A": 0.1173, "B": 0.2669, "C": 0.5603, "D": 0.3764, "E": 0.6293, "F": 0.3556,
             "G": 0.0201})


@figure("second_order_centrality", "node-metrics", "Second-order centrality",
        "The shared seven-node graph shaded by second-order centrality, the spread of a "
        "random walker's return times, where lower means more central. The table runs low to "
        "high: the bridge E has the steadiest return times at 8.59, then C 9.04, while the "
        "pendant G is least central at 22.84.")
def second_order_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.second_order_centrality()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    sigma = {"A": 14.2060, "B": 11.6251, "C": 9.0449, "D": 10.1559, "E": 8.5912, "F": 15.5502,
             "G": 22.8432}
    # inverted scale: the lowest sigma is the most central node, so shade by
    # min/σ and list low → high
    best = min(sigma.values())
    s.graph(pos, STRIP_EDGES, heat={k: best / v for k, v in sigma.items()},
            styles={k: "accent" for k, v in sigma.items() if v == best})
    body = [[k, f"{v:.2f}"] for k, v in sorted(sigma.items(), key=lambda kv: kv[1])]
    s.table_centered(right, [["node", "σ"]] + body, [64, 64],
                     highlight={(1, 0): "accent", (1, 1): "accent"})


# --- current flow (the graph as a resistor network) on the shared strip graph ------


@figure("current_flow_betweenness_centrality", "node-metrics",
        "Current-flow betweenness centrality",
        "The shared seven-node graph shaded by current-flow betweenness, the share of "
        "electrical current through each node when every pair in turn injects and draws a "
        "unit of current. The bridge E carries the most at 0.58, then the hub C 0.43 and "
        "the tail node F 0.33; the pendant G carries none.")
def current_flow_betweenness_centrality(s: Scene) -> None:
    _metric(s, "g.current_flow_betweenness_centrality()", "score",
            {"A": 0.0889, "B": 0.219, "C": 0.4254, "D": 0.2952, "E": 0.5841, "F": 0.3333,
             "G": 0.0})


@figure("current_flow_closeness_centrality", "node-metrics",
        "Current-flow closeness centrality",
        "The shared seven-node graph shaded by current-flow closeness, based on the "
        "effective resistance from each node to all others. The bridge E edges out the hub "
        "C, 0.159 to 0.156, ahead of D 0.147, while the pendant G, behind a single edge, "
        "scores 0.070.")
def current_flow_closeness_centrality(s: Scene) -> None:
    _metric(s, "g.current_flow_closeness_centrality()", "score",
            {"A": 0.1167, "B": 0.1355, "C": 0.1556, "D": 0.1469, "E": 0.1591, "F": 0.1077,
             "G": 0.07}, digits=3)


# --- seeded walk ------------------------------------------------------------------


@figure("personalized_pagerank", "node-metrics", "Personalized PageRank",
        "The shared seven-node graph with the seed A (node id 1) outlined, every node "
        "shaded by PageRank whose random surfer always restarts at A. Rank stays near the "
        "seed, A 0.26, C 0.23 and B 0.20, and fades along the strip to F 0.05 and G 0.02.")
def personalized_pagerank(s: Scene) -> None:
    left, right = compute(s, "g", "g.personalized_pagerank({1: 1.0})")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    scores = {"A": 0.25546, "B": 0.1969, "C": 0.2338, "D": 0.1363, "E": 0.1088, "F": 0.0483,
              "G": 0.0205}
    top = max(scores.values())
    # the seed is what you pass in (soft); it is also the top node here
    s.graph(pos, STRIP_EDGES, heat={k: v / top for k, v in scores.items()},
            styles={"A": "soft"})
    s.badge("A", "id 1")
    s.table_centered(right, [["node", "rank"]] + _rows(scores), [64, 64],
                     highlight={(1, 0): "accent", (1, 1): "accent"})


# --- connective core --------------------------------------------------------------


@figure("knotty_centrality", "node-metrics", "Knotty centrality",
        "The shared seven-node graph with its connective core, the hub C and the bridge E, "
        "in accent inside a shaded hull, joined by the accent edge C-E. The pair is fully "
        "connected (density 1) and holds 0.25 and 0.40 of all betweenness, so the card "
        "reads a knotty score of 0.325.")
def knotty_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.knotty_centrality()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    core = {"C", "E"}
    s.hull(sorted(core))
    s.graph(pos, [(a, b, {"style": "accent"}) if {a, b} == core else (a, b)
                  for a, b in STRIP_EDGES], styles={k: "accent" for k in core})
    # KC(S) = density(S) × mean betweenness share; the share sums to 1 over all nodes
    s.card_centered(right, ["core      C, E", "density   1.00", "bc share  0.25, 0.40",
                            ("score     0.325", "accent")])


# --- HITS and walk counts on the shared digraph -------------------------------------


@figure("authority_centrality", "node-metrics", "HITS authority scores",
        "A five-node directed graph: A points to B, C and D, B and C point to D, and D "
        "points to E. Each node is shaded by its authority score. D is pointed to by all "
        "three hubs and scores 0.82, B and C 0.41, while the source A and E, fed only by D, "
        "score 0.")
def authority_centrality(s: Scene) -> None:
    _directed(s, "g.authority_centrality()", "auth",
              {"A": 0.0, "B": 0.4082, "C": 0.4082, "D": 0.8165, "E": 0.0})


@figure("hub_centrality", "node-metrics", "HITS hub scores",
        "The same five-node directed graph shaded by hub score. A points to B, C and the "
        "top authority D and scores 0.82; B and C each point to D and score 0.41; D and E "
        "point to no authority and score 0.")
def hub_centrality(s: Scene) -> None:
    _directed(s, "g.hub_centrality()", "hub",
              {"A": 0.8165, "B": 0.4082, "C": 0.4082, "D": 0.0, "E": 0.0})


@figure("alpha_centrality", "node-metrics", "Alpha centrality",
        "The same five-node directed graph shaded by alpha centrality with alpha 0.8, which "
        "follows outgoing edges. The source A, whose arcs reach every other node, scores "
        "0.45, then B and C 0.17, D 0.13, and the sink E, which reaches nothing, only its "
        "exogenous share, 0.07.")
def alpha_centrality(s: Scene) -> None:
    _directed(s, "g.alpha_centrality(alpha=0.8)", "score",
              {"A": 0.4524, "B": 0.174, "C": 0.174, "D": 0.1284, "E": 0.0713})


@figure("katz_centrality", "node-metrics", "Katz centrality",
        "The same five-node directed graph shaded by Katz centrality with alpha 0.5, which "
        "counts incoming walks. D receives walks from A, B and C and scores 0.32, E, fed "
        "through D, 0.26, B and C 0.16, and the source A, which nothing reaches, 0.11.")
def katz_centrality(s: Scene) -> None:
    _directed(s, "g.katz_centrality(alpha=0.5)", "score",
              {"A": 0.1053, "B": 0.1579, "C": 0.1579, "D": 0.3158, "E": 0.2632})


# --- Fiedler vector: signed node values on the two-triangle graph ---------------

# PAIR from graph_metrics.py: triangles A, B, C and D, E, F joined by C–D. The
# same graph hosts modularity_spectrum and spectral_bisection.
FIEDLER = {"A": 0.4647, "B": 0.4647, "C": 0.2610, "D": -0.2610, "E": -0.4647, "F": -0.4647}


@figure("fiedler_vector", "node-metrics", "Fiedler vector",
        "Two triangles joined by the edge C–D, each node shaded by its Fiedler vector "
        "entry: positive in the accent color, negative in teal, darker for larger values. "
        "The values change smoothly, A and B 0.46, C 0.26, then across the bridge D -0.26 "
        "and E and F -0.46, so their sign splits the graph at C–D.")
def fiedler_vector(s: Scene) -> None:
    left, right = compute(s, "g", "g.fiedler_vector()")
    top = max(abs(v) for v in FIEDLER.values())
    s.graph(place(PAIR, left, unit=UNIT), PAIR_EDGES,
            heat={k: v / top for k, v in FIEDLER.items()})
    rows = [["node", "value"]] + _rows(FIEDLER)
    hl = {(i, j): "accent" if float(r[1]) > 0 else "g2"
          for i, r in enumerate(rows[1:], 1) for j in (0, 1)}
    s.table_centered(right, rows, [64, 64], highlight=hl)


# --- spectral ordering: a path whose node ids are shuffled ------------------------

ZIGZAG = {"A": (0, 1), "B": (1, 0), "C": (2, 1), "D": (3, 0), "E": (4, 1)}
ZIGZAG_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]


@figure("spectral_ordering", "node-metrics", "Spectral ordering",
        "A five-node path A–B–C–D–E whose nodes were added in the shuffled order D, B, E, "
        "A, C, so their ids are out of path order. Sorting by the Fiedler vector returns "
        "the ids [4, 2, 5, 1, 3], which are A, B, C, D, E: the filled order badges run "
        "straight along the path.")
def spectral_ordering(s: Scene) -> None:
    left, right = compute(s, "g", "g.spectral_ordering()")
    s.graph(place(ZIGZAG, left, unit=UNIT), ZIGZAG_EDGES)
    for i, k in enumerate("ABCDE", 1):
        s.badge(k, str(i), where="n" if ZIGZAG[k][1] == 0 else "s", filled=True)
    s.card_centered(right, ["added  D, B, E, A, C", "ids    [4, 2, 5, 1, 3]",
                            ("labels A, B, C, D, E", "accent")], title="returns node ids")
