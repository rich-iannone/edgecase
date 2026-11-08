"""Node metrics: a score for every node.

Template: `compute`. Left: the graph, every node shaded by its score (`heat`:
darker = higher), the top node also `accent`-outlined. Right: a small table of
node → score sorted high to low, top row highlighted; tied nodes share a row.
The shading and the table always agree, so a reader can check one against the
other.

All whole-graph metrics use ONE shared graph (`STRIP_*`: a strip of three
triangles A–E with a two-node tail F–G) so sibling measures can be compared
directly: degree, harmonic and PageRank favor the hub C; betweenness and
radiality favor the bridge E; closeness ties C and E; average neighbor degree
favors A; eccentricity is largest at the periphery (A, B, G).

Per-edge metrics (`edge_betweenness_centrality`) use the same graph with
edge heat: stroke width grows with the score, the top edge is `accent`, and
the table lists edge → score sorted high to low.

Group centralities (`group_degree_centrality`, `group_closeness_centrality`,
`group_betweenness_centrality`) score a node SET, so they use `contrast` on
the strip graph with the same two groups every time: the spread pair {C, F}
(left, its score in accent) beside the bunched pair {A, B}. Members are
`soft` (the group you pass in, named in a note line `S = [...]`); reached
outside nodes are `accent` (degree) or carry their distance to the group as a
badge (closeness).

Single-node getters (`get_degree`, `get_in_degree`, `get_out_degree`,
`node_vitality`) show the queried node `soft`, the counted elements
`accent`, and a card with the value. Every number is what edgecase returns
for the depicted graph (rounded to 2 decimals).

`incremental_closeness_centrality` scores a hypothetical edge, so it draws
that edge dashed (`ghost`) on a six-node path bent into a ring, the nodes it
recomputes in accent, and a before / after table in place of the heat shading.
"""

from core import SIZE_NOTE, Box, Scene, compute, contrast, figure, place, ring, text_width

UNIT = 80  # one scale for the whole family

# --- the shared undirected graph --------------------------------------------

STRIP_COORDS = {"A": (0, 1), "B": (0.6, 0), "C": (1.2, 1), "D": (1.8, 0), "E": (2.4, 1),
                "F": (3.4, 0.5), "G": (4.4, 0.5)}
STRIP_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
               ("D", "E"), ("E", "F"), ("F", "G")]


def _fmt(v: float, digits: int = 2) -> str:
    return str(v) if isinstance(v, int) else f"{v:.{digits}f}"


def _rows(scores: dict[str, float], max_rows: int = 7, digits: int = 2) -> list[list[str]]:
    """Table body sorted high→low; ties share a row; long tails end in '…'.

    Seven rows (every node of the shared graph) still fit the content area.
    """
    groups: dict[str, list[str]] = {}
    for k, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        groups.setdefault(_fmt(v, digits), []).append(k)
    rows = [[", ".join(ks), v] for v, ks in groups.items()]
    if len(rows) > max_rows:
        rows = rows[:max_rows - 1] + [["…", "…"]]
    return rows


def _metric(s: Scene, call: str, header: str, scores: dict[str, float],
            col_w: tuple[float, float] = (64, 64), digits: int = 2) -> None:
    """The family template: heat-shaded strip graph → sorted score table."""
    left, right = compute(s, "g", call)
    pos = place(STRIP_COORDS, left, unit=UNIT)
    top = max(scores.values())
    s.graph(pos, STRIP_EDGES, heat={k: v / top for k, v in scores.items()},
            styles={k: "accent" for k, v in scores.items() if v == top})
    rows = [["node", header]] + _rows(scores, digits=digits)
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


@figure("degree_centrality", "node-metrics", "Degree centrality",
        "The shared seven-node graph shaded by degree centrality, each degree divided by "
        "n - 1 = 6. Hub C touches four of the six other nodes and tops the table at 0.67, "
        "followed by B, D and E at 0.50, A and F at 0.33, and the pendant node G at 0.17.")
def degree_centrality(s: Scene) -> None:
    _metric(s, "g.degree_centrality()", "score",
            {"A": 0.33, "B": 0.5, "C": 0.67, "D": 0.5, "E": 0.5, "F": 0.33, "G": 0.17})


@figure("closeness_centrality", "node-metrics", "Closeness centrality",
        "The shared seven-node graph shaded by closeness, the reciprocal of the mean distance "
        "to every other node. Hub C and bridge E are both an average of 1.5 hops from the "
        "rest and tie at 0.67, followed by D 0.60, B and F 0.50, A 0.46, and the pendant "
        "node G at 0.35.")
def closeness_centrality(s: Scene) -> None:
    _metric(s, "g.closeness_centrality()", "score",
            {"A": 0.46, "B": 0.5, "C": 0.67, "D": 0.6, "E": 0.67, "F": 0.5, "G": 0.35})


@figure("incremental_closeness_centrality", "node-metrics", "Incremental closeness centrality",
        "A six-node path 1 to 6 drawn as a ring, with the edge e = (1, 6) that would close "
        "it dashed; cc, the current closeness scores, is passed in. Nodes 1, 2, 5 and 6 (accent) are at least two hops closer to one end of "
        "the new edge than the other, so their closeness is recomputed; nodes 3 and 4 are "
        "not, and keep their previous score. The table gives each node's score before and "
        "after: 1 and 6 rise from 0.33 to 0.56, 2 and 5 from 0.45 to 0.56, and 3 and 4 stay "
        "at 0.56. The graph itself is not changed.")
def incremental_closeness_centrality(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_path_graph(6)
    lab = g.get_label_from_id
    before = g.closeness_centrality(weighted=False, return_dict=True)
    after = g.incremental_closeness_centrality((1, 6), prev_cc=before)
    assert g.count_edges() == 5
    rnd = {lab(n): (round(before[n], 2), round(after[n], 2)) for n in after}
    assert rnd == {"1": (0.33, 0.56), "2": (0.45, 0.56), "3": (0.56, 0.56),
                   "4": (0.56, 0.56), "5": (0.45, 0.56), "6": (0.33, 0.56)}
    # a node is recomputed unless its hop distances to 1 and to 6 differ by at most 1
    redo = [k for k in "123456" if abs((int(k) - 1) - (6 - int(k))) > 1]
    assert redo == ["1", "2", "5", "6"]
    left, right = compute(s, "g", "g.incremental_closeness_centrality(e, cc)")
    s.text(left.cx, left.y0 + 8, "e = (1, 6)    cc = scores before", size=SIZE_NOTE, opacity=0.6)
    pos = place(ring(list("123456"), start_deg=-60), Box(left.x0, left.y0 + 22, left.x1, left.y1),
                unit=UNIT * 0.9)
    path = [(str(i), str(i + 1)) for i in range(1, 6)]
    s.graph(pos, path + [("1", "6", {"style": "ghost"})], styles=dict.fromkeys(redo, "accent"))
    rows = [["node", "before", "after"]]
    for ks in (["1", "6"], ["2", "5"], ["3", "4"]):
        b, a = rnd[ks[0]]
        rows.append([", ".join(ks), f"{b:.2f}", f"{a:.2f}" if ks[0] in redo else f"{a:.2f} kept"])
    s.table_centered(right, rows, [56, 58, 78],
                     highlight={(i, 2): "accent" for i in (1, 2)})


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


@figure("node_vitality", "node-metrics", "Node vitality",
        "The shared seven-node graph with the queried hub C outlined and its four edges "
        "dashed, as if removed. The neighbors can still reach each other around C, so the "
        "total pairwise distance only drops from 82 to 70 and the card reads vitality 12.")
def node_vitality(s: Scene) -> None:
    left, right = compute(s, "g", 'g.node_vitality("C")')
    pos = place(STRIP_COORDS, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "removed"}) if "C" in (a, b) else (a, b)
                  for a, b in STRIP_EDGES], styles={"C": "soft"})
    # W = sum of shortest-path distances over ordered pairs (Wiener index)
    s.card_centered(right, ["W(G)       82", "W(G − C)   70", ("vitality   12", "accent")])


# --- per-edge metrics -----------------------------------------------------------

EDGE_W_MIN, EDGE_W_MAX = 1.5, 6.0  # edge heat: stroke width range


@figure("edge_betweenness_centrality", "node-metrics", "Edge betweenness centrality",
        "The shared seven-node graph with each edge drawn thicker the more shortest paths "
        "run through it. The bridge E-F, which every path from the triangles to the tail "
        "must cross, is thickest and in accent at 0.48, ahead of C-E at 0.36 and F-G at "
        "0.29; the triangle edges A-B and C-D carry the least, 0.07.")
def edge_betweenness_centrality(s: Scene) -> None:
    left, right = compute(s, "g", "g.edge_betweenness_centrality()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    # normalized by n(n - 1) = 42 ordered pairs, as edgecase returns by default
    scores = {("A", "B"): 0.07, ("A", "C"): 0.21, ("B", "C"): 0.12, ("B", "D"): 0.14,
              ("C", "D"): 0.07, ("C", "E"): 0.36, ("D", "E"): 0.21, ("E", "F"): 0.48,
              ("F", "G"): 0.29}
    top = max(scores.values())
    s.graph(pos, [])
    for (a, b), v in scores.items():
        s.edge(a, b, style="accent" if v == top else "default",
               width=EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * v / top)
    rows = [["edge", "score"]] + _rows({f"{a}–{b}": v for (a, b), v in scores.items()})
    s.table_centered(right, rows, [84, 64], highlight={(1, 0): "accent", (1, 1): "accent"})



@figure("edge_clustering_coefficient", "node-metrics", "Edge clustering coefficient",
        "The shared seven-node graph with edge C-D in accent and the two triangles through "
        "it, B-C-D and C-D-E, shaded. With endpoint degrees 4 and 3 at most 2 triangles are "
        "possible, so the default score is (2 + 1) / 2 = 1.5, shown in the card.")
def edge_clustering_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.edge_clustering_coefficient()")
    pos = place(STRIP_COORDS, left, unit=UNIT)
    s.hull(["B", "C", "D"])
    s.hull(["C", "D", "E"])
    s.graph(pos, [(a, b, {"style": "accent"}) if (a, b) == ("C", "D") else (a, b)
                  for a, b in STRIP_EDGES], styles={"C": "soft", "D": "soft"})
    # offset=True by default: (z + 1) / s with s = min(d_C - 1, d_D - 1)
    s.card_centered(right, ["triangles z  2", "possible s   2",
                            ("(z + 1) / s  1.5", "accent")], title="edge C–D")


# --- group centralities: one node set, two contrasting groups ------------------------

GROUP_SPREAD, GROUP_BUNCHED = ["C", "F"], ["A", "B"]


def _group_panel(s: Scene, box: Box, group: list[str], prefix: str, *,
                 accent: set[str] = frozenset(), badges: dict[str, int] | None = None) -> None:
    s.text(box.cx, box.y0 + 8, f"S = [{', '.join(group)}]", size=SIZE_NOTE, opacity=0.6)
    gb = Box(box.x0, box.y0 + 22, box.x1, box.y1)
    pos = place(STRIP_COORDS, gb, unit=UNIT)
    styles = {k: "accent" for k in accent} | {k: "soft" for k in group}
    edges = [(a, b, {"style": "accent"}) if ({a, b} & set(group) and {a, b} & accent) else (a, b)
             for a, b in STRIP_EDGES]
    s.graph(pos, edges, styles=styles, prefix=prefix)
    for k, v in (badges or {}).items():
        s.badge(prefix + k, str(v))


def _outside_neighbors(group: list[str]) -> set[str]:
    return {b if a in group else a for a, b in STRIP_EDGES
            if (a in group) != (b in group)}


@figure("group_degree_centrality", "node-metrics", "Group degree centrality",
        "The shared seven-node graph twice. Left, the group C, F: between them they touch "
        "all five outside nodes A, B, D, E and G, drawn in accent, so the score is 5 / 5 = "
        "1.0. Right, the group A, B reaches only C and D, 2 of 5, for 0.4.")
def group_degree_centrality(s: Scene) -> None:
    left, right = contrast(s, "g.group_degree_centrality(S)", "g.group_degree_centrality(S)",
                           "1.0", "0.4")
    _group_panel(s, left, GROUP_SPREAD, "l.", accent=_outside_neighbors(GROUP_SPREAD))
    _group_panel(s, right, GROUP_BUNCHED, "r.", accent=_outside_neighbors(GROUP_BUNCHED))


@figure("group_closeness_centrality", "node-metrics", "Group closeness centrality",
        "The shared seven-node graph twice, each outside node badged with its distance to "
        "the nearest group member. Left, every outside node is one hop from C or F, so the "
        "score is 5 / 5 = 1.0. Right, from A, B the distances are 1, 1, 2, 3 and 4, so the "
        "score is 5 / 11 = 0.455.")
def group_closeness_centrality(s: Scene) -> None:
    left, right = contrast(s, "g.group_closeness_centrality(S)",
                           "g.group_closeness_centrality(S)", "1.0", "0.455")
    _group_panel(s, left, GROUP_SPREAD, "l.",
                 badges={"A": 1, "B": 1, "D": 1, "E": 1, "G": 1})
    _group_panel(s, right, GROUP_BUNCHED, "r.",
                 badges={"C": 1, "D": 1, "E": 2, "F": 3, "G": 4})


@figure("group_betweenness_centrality", "node-metrics", "Group betweenness centrality",
        "The shared seven-node graph twice. Left, the group C, F intercepts 0.6 of the "
        "shortest-path traffic between the five outside nodes, including every path from the "
        "triangles to the pendant G. Right, the group A, B sits at the end of the strip, no shortest path "
        "between other nodes needs it, and it scores 0.0.")
def group_betweenness_centrality(s: Scene) -> None:
    left, right = contrast(s, "g.group_betweenness_centrality(S)",
                           "g.group_betweenness_centrality(S)", "0.6", "0.0")
    _group_panel(s, left, GROUP_SPREAD, "l.")
    _group_panel(s, right, GROUP_BUNCHED, "r.")


# --- later additions: per-edge variants and the best group ----------------------------
#
# The per-edge variants (`edge_current_flow_betweenness_centrality`, its `_subset`,
# `edge_betweenness_centrality_subset`, `edge_load_centrality`) follow
# `edge_betweenness_centrality`: edge width grows with the score, the top edge is
# `accent`, and the table lists edge → score. Subset variants outline their sources and
# targets `soft` with `s` / `t` badges, as `betweenness_centrality_subset` does; calls too
# long for the result caption sit under the graph instead. `prominent_group` uses
# `contrast` like the group centralities: the group it finds (left, accent hull) beside the
# pair a ranking by individual betweenness suggests (right), which covers the same paths
# twice. Each figure computes its numbers with edgecase and asserts them.


def strip_graph():
    """The shared strip graph as an edgecase Graph (node ids 1-7 are A-G)."""
    import edgecase as ec

    return ec.Graph().add_nodes(list(STRIP_COORDS)).add_edges(STRIP_EDGES)


def edge_scores(g, scores: dict[int, float], digits: int = 2) -> dict[tuple[str, str], float]:
    """Edge-ID scores keyed by the drawn (label, label) pair, rounded."""
    out = {}
    for eid, v in scores.items():
        a, b = (g.get_label_from_id(n) for n in g.get_node_ids_by_edge_id(eid))
        out[(a, b)] = round(v, digits)
    return out


def _edge_metric(s: Scene, captions: tuple[str, str], scores: dict[tuple[str, str], float],
                 *, digits: int = 2, header: str = "score", ends: dict[str, str] | None = None,
                 box: Box | None = None) -> None:
    """Edge widths by score on the strip graph → edge table (top edge accent)."""
    left, right = compute(s, *captions)
    gbox = left if box is None else box
    if ends:
        note = "    ".join(f"{t} = [{', '.join(k for k, v in ends.items() if v == t.lower())}]"
                           for t in ("S", "T"))
        s.text(left.cx, left.y0 + 8, note, size=SIZE_NOTE, opacity=0.6)
        gbox = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    pos = place(STRIP_COORDS, gbox, unit=UNIT)
    top = max(scores.values())
    s.graph(pos, [], styles={k: "soft" for k in (ends or {})})
    for (a, b), v in scores.items():
        s.edge(a, b, style="accent" if v == top else "muted" if v == 0 else "default",
               width=EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * v / top)
    for k, t in (ends or {}).items():
        s.badge(k, t, where={"A": "sw", "B": "nw", "G": "ne"}.get(k, "auto"))
    rows = [["edge", header]] + _rows({f"{a}–{b}": v for (a, b), v in scores.items()},
                                      digits=digits)
    first = max(84, max(text_width(r[0], SIZE_NOTE) for r in rows) + 16)
    s.table_centered(right, rows, [first, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("edge_current_flow_betweenness_centrality", "node-metrics",
        "Edge current-flow betweenness centrality",
        "The shared seven-node graph with each edge drawn thicker the more electrical "
        "current it carries when every pair of nodes in turn sends a unit of current. The "
        "bridge E-F carries the most, 0.33 in accent, then C-E 0.23, D-E 0.22 and F-G 0.20; "
        "the triangle edges B-C and C-D, with parallel routes beside them, carry the least, "
        "0.12.")
def edge_current_flow_betweenness_centrality(s: Scene) -> None:
    g = strip_graph()
    scores = edge_scores(g, g.edge_current_flow_betweenness_centrality(return_dict=True))
    assert scores[("E", "F")] == 0.33 and scores[("C", "D")] == 0.12
    _edge_metric(s, ("g.edge_current_flow_betweenness_centrality()", ""), scores)


@figure("edge_current_flow_betweenness_centrality_subset", "node-metrics",
        "Edge current-flow betweenness on a subset",
        "The shared seven-node graph with the sources A and B and the target G outlined; "
        "only current sent from A or B to G counts. Edges are drawn thicker the more of it "
        "they carry: all of it crosses E-F and F-G, which share the top at 0.033, while "
        "inside the triangles the current spreads out, C-E 0.018 down to C-D 0.003.")
def edge_current_flow_betweenness_centrality_subset(s: Scene) -> None:
    g = strip_graph()
    raw = g.edge_current_flow_betweenness_centrality_subset(["A", "B"], ["G"], return_dict=True)
    scores = edge_scores(g, raw, digits=3)
    assert scores[("E", "F")] == scores[("F", "G")] == 0.033 and scores[("C", "D")] == 0.003
    _edge_metric(s, ("g.edge_current_flow_betweenness_centrality_subset(S, T)", ""), scores,
                 digits=3, ends={"A": "s", "B": "s", "G": "t"})


@figure("edge_betweenness_centrality_subset", "node-metrics",
        "Edge betweenness on a subset",
        "The shared seven-node graph with the sources A and B and the target G outlined; "
        "only shortest paths from A or B to G count. Every one ends along E-F and F-G "
        "(accent, 1.00). A's single path and one of B's two run through C-E, 0.75, B's "
        "other path takes B-D-E, and A-B and C-D carry none.")
def edge_betweenness_centrality_subset(s: Scene) -> None:
    g = strip_graph()
    raw = g.edge_betweenness_centrality_subset(["A", "B"], ["G"], return_dict=True)
    scores = edge_scores(g, raw)
    assert scores[("E", "F")] == scores[("F", "G")] == 1.0 and scores[("C", "E")] == 0.75
    assert scores[("A", "B")] == scores[("C", "D")] == 0.0
    _edge_metric(s, ("g", "g.edge_betweenness_centrality_subset(S, T)"), scores,
                 ends={"A": "s", "B": "s", "G": "t"})


@figure("edge_load_centrality", "node-metrics", "Edge load centrality",
        "The shared seven-node graph with each edge drawn thicker the more load it carries "
        "when every node sends one unit to every other node, split equally at each hop. The "
        "bridge E-F carries the most, 22 in accent, then C-E 18, and D-E and F-G 13; the "
        "triangle edges A-B and C-D carry the least, 8.")
def edge_load_centrality(s: Scene) -> None:
    g = strip_graph()
    scores = {k: int(v) for k, v in edge_scores(g, g.edge_load_centrality(return_dict=True)).items()}
    assert scores[("E", "F")] == 22 and scores[("C", "E")] == 18 and scores[("F", "G")] == 13
    assert scores[("A", "B")] == scores[("C", "D")] == 8
    _edge_metric(s, ("g", "g.edge_load_centrality()"), scores, header="load")


@figure("prominent_group", "node-metrics", "Most prominent group",
        "The shared seven-node graph twice. Left, prominent_group(2) finds the pair E, B "
        "(accent), which together lie on 0.65 of the shortest paths between the other "
        "five nodes. Right, E and F, two of the three nodes with the highest individual "
        "betweenness, cover much of the same traffic, so as a group they score only 0.40.")
def prominent_group(s: Scene) -> None:
    g = strip_graph()
    value, group = g.prominent_group(2)
    best = [g.get_label_from_id(n) for n in group]
    assert best == ["E", "B"] and round(value, 2) == 0.65
    rival = round(g.group_betweenness_centrality(["E", "F"]), 2)
    assert rival == 0.4
    left, right = contrast(s, "g.prominent_group(2)", "g.group_betweenness_centrality(S)",
                           "(0.65, [E, B])", "0.4")
    s.text(left.cx, left.y0 + 8, "k = 2", size=SIZE_NOTE, opacity=0.6)
    gb = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    s.graph(place(STRIP_COORDS, gb, unit=UNIT), STRIP_EDGES,
            styles={"E": "accent", "B": "accent"}, prefix="l.")
    _group_panel(s, right, ["E", "F"], "r.")
