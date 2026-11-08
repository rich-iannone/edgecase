"""Set algebra on two graphs.

Template: `binary`. G and H on the left and middle, the result on the right.
Edges only in G are `accent`, edges only in H are `g2` (teal), edges in both
are plain ink. The result keeps those colors, so the reader can trace every
edge back to its source. Edges the operation drops appear `removed`, and so
do nodes it prunes (intersection and both differences drop nodes left with
no edges).

union / intersection / difference / symmetric_difference share one G and one
H so their results can be compared directly. compose reuses them with one H
label changed to show whose attributes win; disjoint_union uses integer ids
because renumbering is its whole point.

Graph products use the same G ∘ H = result layout: G's edges `accent`, H's
`g2`, and each product edge colored by the factor it copies (ink when both
coordinates move). Product nodes (g, h) are labeled compactly as gh. The two
joins (corona_product, full_join) draw H's nodes `g2` and the joining edges
`accent`. The *_all folds show three inputs, the operator between them, and
the result. Every later addition asserts its docstring example in edgecase.
"""

from core import (
    ARROW_Y,
    CONTENT_BOTTOM,
    CONTENT_TOP,
    MARGIN,
    R_SMALL,
    SIZE_NOTE,
    Box,
    Scene,
    W,
    binary,
    figure,
    place,
    ring,
)

UNIT = 80  # one density for the whole family

NODES = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}

# G = {A–B, A–C, B–D}, H = {A–B, B–D, C–D}: two shared edges, one per side.
SHARED = [("A", "B"), ("B", "D")]
G_ONLY = [("A", "C")]
H_ONLY = [("C", "D")]


def _styled(edges, style: str):
    return [(a, b, {"style": style}) for a, b in edges]


def _inputs(s: Scene, a, b) -> None:
    s.graph(place(NODES, a, unit=UNIT), SHARED + _styled(G_ONLY, "accent"), prefix="g.")
    s.graph(place(NODES, b, unit=UNIT), SHARED + _styled(H_ONLY, "g2"), prefix="h.")


@figure("union", "set-ops", "Union",
        "G has edges A–B and B–D, which H shares, plus A–C (only in G, accent). H adds C–D "
        "(only in H, teal). The union keeps all four nodes and all four edges, each in the "
        "color of the graph it came from.")
def union(s: Scene) -> None:
    a, b, c = binary(s, "∪", ("G", "H", "G.union(H)"))
    _inputs(s, a, b)
    s.graph(place(NODES, c, unit=UNIT),
            SHARED + _styled(G_ONLY, "accent") + _styled(H_ONLY, "g2"), prefix="r.")


@figure("intersection", "set-ops", "Intersection",
        "G and H share edges A–B and B–D; A–C is only in G (accent) and C–D only in H (teal). "
        "The intersection keeps just the shared edges A–B and B–D; A–C and C–D are dashed, "
        "and node C, left with no edges, is dropped too.")
def intersection(s: Scene) -> None:
    a, b, c = binary(s, "∩", ("G", "H", "G.intersection(H)"))
    _inputs(s, a, b)
    s.graph(place(NODES, c, unit=UNIT), SHARED + _styled(G_ONLY + H_ONLY, "removed"),
            styles={"C": "removed"}, prefix="r.")


@figure("difference", "set-ops", "Difference",
        "G has edges A–B, B–D, and A–C; H has A–B, B–D, and C–D. G minus H keeps only A–C "
        "(accent), the one edge of G that H lacks. The shared edges A–B and B–D are dashed, "
        "and nodes B and D, left with no edges, are dropped.")
def difference(s: Scene) -> None:
    a, b, c = binary(s, "−", ("G", "H", "G.difference(H)"))
    _inputs(s, a, b)
    # C–D is not drawn: it was never in G, so nothing was removed.
    s.graph(place(NODES, c, unit=UNIT), _styled(G_ONLY, "accent") + _styled(SHARED, "removed"),
            styles={"B": "removed", "D": "removed"}, prefix="r.")


@figure("symmetric_difference", "set-ops", "Symmetric difference",
        "G and H share edges A–B and B–D; A–C is only in G (accent) and C–D only in H (teal). "
        "The symmetric difference keeps the edges in exactly one graph, A–C and C–D; the "
        "shared edges are dashed and node B, left with no edges, is dropped.")
def symmetric_difference(s: Scene) -> None:
    a, b, c = binary(s, "△", ("G", "H", "G.symmetric_difference(H)"))
    _inputs(s, a, b)
    s.graph(place(NODES, c, unit=UNIT),
            _styled(G_ONLY, "accent") + _styled(H_ONLY, "g2") + _styled(SHARED, "removed"),
            styles={"B": "removed"}, prefix="r.")


@figure("compose", "set-ops", "Compose",
        "The same G and H as for union, except that the shared fourth node is labeled D in G "
        "(accent) and D′ in H (teal). Compose merges nodes by id and keeps all four edges like "
        "union, but H wins the clash, so the result shows D′.")
def compose(s: Scene) -> None:
    a, b, c = binary(s, "∪", ("G", "H", "G.compose(H)"))
    # Node D carries a different label in each graph; color says whose label it is.
    s.graph(place(NODES, a, unit=UNIT), SHARED + _styled(G_ONLY, "accent"), prefix="g.",
            styles={"D": "accent"})
    s.graph(place(NODES, b, unit=UNIT), SHARED + _styled(H_ONLY, "g2"), prefix="h.",
            labels={"D": "D′"}, styles={"D": "g2"})
    s.graph(place(NODES, c, unit=UNIT),
            SHARED + _styled(G_ONLY, "accent") + _styled(H_ONLY, "g2"), prefix="r.",
            labels={"D": "D′"}, styles={"D": "g2"})


# Integer ids: G is a triangle 1, 2, 3; H is a single edge 1–2.
TRI = {"1": (0.5, 0), "2": (0, 0.87), "3": (1, 0.87)}
TRI_EDGES = [("1", "2"), ("2", "3"), ("1", "3")]
PAIR = {"1": (0, 0), "2": (0, 0.87)}


@figure("disjoint_union", "set-ops", "Disjoint union",
        "G is a triangle on nodes 1, 2, 3 (accent); H is a single edge between its own nodes 1 "
        "and 2 (teal). The disjoint union never merges ids: H's nodes are renumbered 4 and 5, "
        "giving five nodes in two separate components.")
def disjoint_union(s: Scene) -> None:
    a, b, c = binary(s, "+", ("G", "H", "G.disjoint_union(H)"))
    s.graph(place(TRI, a, unit=UNIT), _styled(TRI_EDGES, "accent"),
            styles=dict.fromkeys(TRI, "accent"), prefix="g.")
    s.graph(place(PAIR, b, unit=UNIT), _styled([("1", "2")], "g2"),
            styles=dict.fromkeys(PAIR, "g2"), prefix="h.")
    coords = {**TRI, "4": (1.9, 0), "5": (1.9, 0.87)}
    s.graph(place(coords, c, unit=UNIT), _styled(TRI_EDGES, "accent") + _styled([("4", "5")], "g2"),
            styles={**dict.fromkeys(TRI, "accent"), "4": "g2", "5": "g2"}, prefix="r.")


# --- later additions: graph products ------------------------------------------
#
# Each product figure rebuilds its docstring example in edgecase, reads the
# returned edges back, and asserts them, so a library change breaks the build
# instead of the picture. Product nodes are labeled "(g, h)" by edgecase; the
# figures write them compactly as gh (A1, b2) inside the node, and the drawn
# edges are the ones edgecase returned. Edge colors say which factor moved:
# G's coordinate only = `accent` (a copy of a G edge), H's coordinate only =
# `g2` (a copy of an H edge), both at once = ink, like the shared edges above.
# Joins (corona_product, full_join) instead keep both graphs' own edges ink,
# draw H's nodes and their copies `g2`, and make the joining edges `accent`.

PRODUCT_UNIT = 64  # products are denser than the set-algebra panels


def _pair(label: str) -> tuple[str, str]:
    """'(A, 1)' -> ('A', '1')."""
    g, h = label.strip("()").split(", ")
    return g, h


def _label_edges(g) -> set[tuple[str, str]]:
    """Edges of an edgecase graph as sorted (label, label) pairs (ids if unlabeled)."""
    def lab(n):
        x = g.get_label_from_id(n)
        return str(n) if x is None else x
    return {tuple(sorted((lab(a), lab(b)))) for _, a, b in g.get_edges_as_tuples()}


def _factor(nodes: list[str], edges) -> object:
    import edgecase as ec

    return ec.Graph().add_nodes(nodes).add_edges(list(edges))


def _product_edges(product, short) -> list[tuple[str, str, dict]]:
    """Product edges keyed by short ids, styled by which coordinate moves."""
    out = []
    for a, b in sorted(_label_edges(product)):
        (ga, ha), (gb, hb) = _pair(a), _pair(b)
        style = "g2" if ga == gb else "accent" if ha == hb else "default"
        out.append((short(ga, ha), short(gb, hb), {"style": style}))
    return out


def _grid(gs: str, hs: str) -> dict[str, tuple[float, float]]:
    """G's nodes across, H's nodes down: the product drawn as a grid."""
    return {g + h: (i, j) for i, g in enumerate(gs) for j, h in enumerate(hs)}


def _factors(s: Scene, a, b, gpos, gedges, hpos, hedges, **kw) -> None:
    s.graph(place(gpos, a, unit=PRODUCT_UNIT), _styled(gedges, "accent"), prefix="g.", **kw)
    s.graph(place(hpos, b, unit=PRODUCT_UNIT), _styled(hedges, "g2"), prefix="h.")


ROW3 = {"A": (0, 0), "B": (1, 0), "C": (2, 0)}
P3_EDGES = [("A", "B"), ("B", "C")]
TRI_ABC = {"A": (0.5, 0), "B": (0, 0.87), "C": (1, 0.87)}


@figure("cartesian_product", "set-ops", "Cartesian product",
        "G is a path A–B–C (accent edges) and H is a single edge 1–2 (teal). Their Cartesian "
        "product has a node for every pair, drawn as a grid with labels A1 to C2 for the pairs "
        "(A, 1) to (C, 2): 6 nodes and 7 edges, a 3 by 2 ladder. The four accent rungs along "
        "the rows copy G's edges, and the three teal edges down the columns copy H's edge.")
def cartesian_product(s: Scene) -> None:
    a, b, c = binary(s, "□", ("G", "H", "G.cartesian_product(H)"))
    g = _factor(list("ABC"), P3_EDGES)
    h = _factor(["1", "2"], [("1", "2")])
    product = g.cartesian_product(h)
    edges = _product_edges(product, lambda x, y: x + y)
    assert product.count_nodes() == 6 and len(edges) == 7
    assert sorted(e[2]["style"] for e in edges) == ["accent"] * 4 + ["g2"] * 3
    _factors(s, a, b, ROW3, P3_EDGES, {"1": (0, 0), "2": (0, 1)}, [("1", "2")])
    s.graph(place(_grid("ABC", "12"), c, unit=PRODUCT_UNIT), edges, prefix="r.")


@figure("tensor_product", "set-ops", "Tensor product",
        "G is a triangle A, B, C (accent edges) and H is a single edge 0–1 (teal). In the tensor "
        "product both coordinates must move along an edge at once, so every product edge is "
        "ink: (A, 0)–(B, 1), (B, 1)–(C, 0), and so on, labeled A0 to C1. The result has 6 "
        "nodes and 6 edges forming a single 6-cycle, the bipartite double cover of the "
        "triangle.")
def tensor_product(s: Scene) -> None:
    import edgecase as ec

    a, b, c = binary(s, "×", ("G", "H", "G.tensor_product(H)"))
    tri = [("A", "B"), ("B", "C"), ("A", "C")]
    g = _factor(list("ABC"), tri)
    h = _factor(["0", "1"], [("0", "1")])
    product = g.tensor_product(h)
    edges = _product_edges(product, lambda x, y: x + y)
    assert product.is_isomorphic(ec.Graph().add_cycle_graph(6)) and len(edges) == 6
    assert {e[2]["style"] for e in edges} == {"default"}
    _factors(s, a, b, TRI_ABC, tri, {"0": (0, 0), "1": (0, 1)}, [("0", "1")])
    cycle = ring(["A0", "B1", "C0", "A1", "B0", "C1"], radius=1.0, start_deg=-120)
    s.graph(place(cycle, c, unit=PRODUCT_UNIT), edges, prefix="r.")



@figure("strong_product", "set-ops", "Strong product",
        "G is a path a–b–c (accent edges) and H a path 1–2–3 (teal). Their strong product is "
        "the 3 by 3 king's graph, labeled a1 to c3: accent edges along the rows copy G, teal "
        "edges down the columns copy H, and ink diagonals move both coordinates at once. It "
        "has 9 nodes and 20 edges, and the center (b, 2), outlined, reaches all 8 others.")
def strong_product(s: Scene) -> None:
    a, b, c = binary(s, "⊠", ("G", "H", "G.strong_product(H)"))
    gedges, hedges = [("a", "b"), ("b", "c")], [("1", "2"), ("2", "3")]
    g = _factor(list("abc"), gedges)
    h = _factor(["1", "2", "3"], hedges)
    product = g.strong_product(h)
    edges = _product_edges(product, lambda x, y: x + y)
    assert product.count_edges() == 20 and product.get_degree("(b, 2)") == 8
    assert sorted(e[2]["style"] for e in edges) == ["accent"] * 6 + ["default"] * 8 + ["g2"] * 6
    _factors(s, a, b, {"a": (0, 0), "b": (1, 0), "c": (2, 0)}, gedges,
             {"1": (0, 0), "2": (0, 1), "3": (0, 2)}, hedges)
    s.graph(place(_grid("abc", "123"), c, unit=PRODUCT_UNIT), edges, prefix="r.",
            styles={"b2": "soft"})


@figure("lexicographic_product", "set-ops", "Lexicographic product",
        "G is a single edge between departments sales (S) and ops (O), accent; H is a team, an "
        "edge between lead (L) and rep (R), teal. The lexicographic product puts a copy of the "
        "team in each department (the shaded groups, L and R in each): the teal edges copy H "
        "inside each copy, and "
        "because sales and ops are linked, four accent edges join every member of one team to "
        "every member of the other. The result is a complete graph on 4 nodes with 6 edges.")
def lexicographic_product(s: Scene) -> None:
    import edgecase as ec

    a, b, c = binary(s, "•", ("G", "H", "G.lexicographic_product(H)"))
    g = _factor(["sales", "ops"], [("sales", "ops")])
    h = _factor(["lead", "rep"], [("lead", "rep")])
    product = g.lexicographic_product(h)
    # every cross-department edge comes from G's edge, whatever the roles
    edges = [(x, y, {"style": "g2" if x[0] == y[0] else "accent"})
             for x, y, _ in _product_edges(product, lambda d, r: d[0] + r[0])]
    assert product.is_isomorphic(ec.Graph().add_full_graph(4)) and len(edges) == 6
    names = {"s": "sales", "o": "ops", "l": "lead", "r": "rep"}
    gp = place({"S": (0, 0), "O": (1.6, 0)}, a, unit=PRODUCT_UNIT)
    s.graph(gp, _styled([("S", "O")], "accent"), prefix="g.")
    for k, (x, y) in gp.items():
        s.text(x, y + 30, names[k.lower()], size=SIZE_NOTE, opacity=0.6)
    hp = place({"L": (0, 0), "R": (0, 1)}, b, unit=PRODUCT_UNIT)
    s.graph(hp, _styled([("L", "R")], "g2"), prefix="h.")
    for k, (x, y) in hp.items():
        s.text(x + 24, y, names[k.lower()], size=SIZE_NOTE, anchor="start", opacity=0.6)
    pos = {f"{d}{r}": (i * 1.6, j) for i, d in enumerate("so") for j, r in enumerate("lr")}
    box = Box(c.x0, c.y0 + 18, c.x1, c.y1)  # room for the department names
    s.graph(place(pos, box, unit=PRODUCT_UNIT), edges, prefix="r.",
            labels={k: k[1].upper() for k in pos})
    for d in "so":
        s.hull([f"r.{d}l", f"r.{d}r"], "g2")
        top = s.nodes[f"r.{d}l"]
        s.text(top.x, top.y - 36, names[d], size=SIZE_NOTE, opacity=0.6)


@figure("rooted_product", "set-ops", "Rooted product",
        "G is a path 1–2–3–4 (accent edges) and H a tooth, an edge from top to tip (teal), "
        "rooted at top (outlined). The rooted product hangs a copy of the tooth from every node "
        "of G: the top row holds the roots (1, top) to (4, top), wired like G by accent edges, "
        "and each teal edge drops to that column's tip. The result is a comb with 8 nodes and "
        "7 edges, a tree.")
def rooted_product(s: Scene) -> None:
    import edgecase as ec

    a, b, c = binary(s, "∗", ("G", "H", 'G.rooted_product(H, "top")'))
    spine = ec.Graph().add_path_graph(4)
    h = _factor(["top", "tip"], [("top", "tip")])
    comb = spine.rooted_product(h, root="top")
    edges = _product_edges(comb, lambda x, y: x + y[1])
    assert comb.count_nodes() == 8 and len(edges) == 7 and comb.is_tree()
    assert sorted(e[2]["style"] for e in edges) == ["accent"] * 3 + ["g2"] * 4
    gpos = {k: (i, 0) for i, k in enumerate("1234")}
    s.graph(place(gpos, a, unit=PRODUCT_UNIT), _styled(list(zip("123", "234")), "accent"),
            prefix="g.")
    s.graph(place({"top": (0, 0), "tip": (0, 1)}, b, unit=PRODUCT_UNIT),
            _styled([("top", "tip")], "g2"), prefix="h.", styles={"top": "soft"})
    grid = {f"{k}{r}": (i, j) for i, k in enumerate("1234") for j, r in enumerate("oi")}
    pos = place(grid, Box(c.x0 + 34, c.y0, c.x1, c.y1), unit=54)
    s.graph(pos, edges, prefix="r.", labels={k: k[0] for k in grid},
            styles={f"{k}o": "soft" for k in "1234"})
    for r, name in (("o", "top"), ("i", "tip")):
        s.text(pos["1" + r][0] - 26, pos["1" + r][1], name, size=SIZE_NOTE, anchor="end",
               opacity=0.6)


@figure("corona_product", "set-ops", "Corona product",
        "G is a triangle A, B, C and H a single node x (teal). The corona product keeps the "
        "triangle and gives each corner its own copy of H, the teal leaves Ax, Bx, and Cx for "
        "(A, x), (B, x), and (C, x), each joined to its corner by an accent edge. The result is "
        "the sunlet graph with 6 nodes and 6 edges.")
def corona_product(s: Scene) -> None:
    a, b, c = binary(s, "⊙", ("G", "H", "G.corona_product(H)"))
    tri = [("A", "B"), ("B", "C"), ("A", "C")]
    g = _factor(list("ABC"), tri)
    leaf = _factor(["x"], [])
    sunlet = g.corona_product(leaf)
    got = {tuple(sorted(lab.replace("(", "").replace(", ", "").replace(")", "") for lab in e))
           for e in _label_edges(sunlet)}
    spokes = [(k, k + "x") for k in "ABC"]
    assert sunlet.count_nodes() == 6
    assert got == {tuple(sorted(e)) for e in tri + spokes}
    s.graph(place(TRI_ABC, a, unit=PRODUCT_UNIT), tri, prefix="g.")
    s.graph(place({"x": (0, 0)}, b), [], prefix="h.", styles={"x": "g2"})
    inner = ring(["A", "C", "B"], radius=1.0, start_deg=-90)  # B left, C right, as in G
    outer = {k + "x": (2.35 * x, 2.35 * y) for k, (x, y) in inner.items()}
    s.graph(place(inner | outer, c, unit=PRODUCT_UNIT), tri + _styled(spokes, "accent"),
            prefix="r.", styles=dict.fromkeys(outer, "g2"))


@figure("modular_product", "set-ops", "Modular product",
        "G is a paw: a triangle A, B, C with a tail C–D. H is a 4-cycle 1, 2, 3, 4. The path "
        "D–C–B in G and the path 1–2–3 in H (accent; the rest muted) are the same induced "
        "subgraph. Their modular product has 16 nodes and 40 edges, and its largest clique, "
        "drawn on the right as D1, C2, B3 for the pairs (D, 1), (C, 2), (B, 3), has 3 nodes: "
        "the largest induced subgraph common to both graphs.")
def modular_product(s: Scene) -> None:
    import edgecase as ec

    a, b, c = binary(s, "◇", ("G", "H", "G.modular_product(H)"))
    paw_edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")]
    g = _factor(list("ABCD"), paw_edges)
    h = ec.Graph().add_cycle_graph(4)
    product = g.modular_product(h)
    clique = {product.get_label_from_id(n) for n in product.maximum_clique()}
    assert product.count_nodes() == 16 and product.count_edges() == 40
    assert clique == {"(D, 1)", "(C, 2)", "(B, 3)"}
    sq_edges = [("1", "2"), ("2", "3"), ("3", "4"), ("1", "4")]
    assert _label_edges(h) == {tuple(sorted(e)) for e in sq_edges}
    paw = {"D": (0, 0), "C": (1, 0), "B": (1, 1), "A": (1.87, 0.5)}
    sq = {"1": (0, 0), "2": (1, 0), "3": (1, 1), "4": (0, 1)}

    def common(edges, keep):
        return [(x, y, {"style": "accent" if x in keep and y in keep else "muted"})
                for x, y in edges]

    s.graph(place(paw, a, unit=PRODUCT_UNIT), common(paw_edges, "BCD"), prefix="g.",
            styles={"A": "muted", **dict.fromkeys("BCD", "accent")})
    s.graph(place(sq, b, unit=PRODUCT_UNIT), common(sq_edges, "123"), prefix="h.",
            styles={"4": "muted", **dict.fromkeys("123", "accent")})
    top = Box(c.x0, c.y0, c.x1, c.y1 - 66)
    tri = {"D1": (0, 0), "C2": (1, 0), "B3": (1, 1)}
    s.graph(place(tri, top, unit=52), _styled([("D1", "C2"), ("C2", "B3"), ("D1", "B3")], "accent"),
            prefix="r.", styles=dict.fromkeys(tri, "accent"))
    s.card_centered(Box(c.x0, c.y1 - 64, c.x1, c.y1),
                    [f"{product.count_nodes()} nodes, {product.count_edges()} edges",
                     (f"max clique: {len(clique)}", "accent")])


@figure("full_join", "set-ops", "Full join",
        "G is a 5-cycle on nodes 1 to 5 and H is a single node, hub (teal). The full join keeps "
        "both graphs and adds an accent edge from every node of G to every node of H: five "
        "spokes from the hub to the rim. The result is a wheel with 6 nodes and 10 edges, and "
        "hub has degree 5.")
def full_join(s: Scene) -> None:
    import edgecase as ec

    a, b, c = binary(s, "+", ("G", "H", "G.full_join(H)"))
    rim = ec.Graph().add_cycle_graph(5)
    hub = _factor(["hub"], [])
    wheel = rim.full_join(hub)
    cycle = [(str(i), str(i % 5 + 1)) for i in range(1, 6)]
    spokes = [(str(i), "hub") for i in range(1, 6)]
    assert _label_edges(wheel) == {tuple(sorted(e)) for e in cycle + spokes}
    assert wheel.get_degree("hub") == 5
    pos = ring([str(i) for i in range(1, 6)], radius=1.0)
    s.graph(place(pos, a, unit=PRODUCT_UNIT), cycle, prefix="g.")
    s.graph(place({"hub": (0, 0)}, b), [], prefix="h.", styles={"hub": "g2"})
    s.graph(place(pos | {"hub": (0, 0)}, c, unit=PRODUCT_UNIT),
            cycle + _styled(spokes, "accent"), prefix="r.", styles={"hub": "g2"})


# --- later additions: folds over a list of graphs ----------------------------
#
# union_all / compose_all / disjoint_union_all / intersection_all fold the
# binary operation over a list. The figures draw three small inputs, the
# operator between them, then `=` and the result, all from the docstring
# examples (asserted in edgecase). Coloring follows the binary figures: an
# edge found in only one input takes that input's color (accent, teal,
# violet), an edge in several inputs is ink, and the result keeps the colors.

FOLD_INPUTS = (Box(MARGIN, CONTENT_TOP, 140, CONTENT_BOTTOM),
               Box(166, CONTENT_TOP, 294, CONTENT_BOTTOM),
               Box(320, CONTENT_TOP, 448, CONTENT_BOTTOM))
FOLD_RESULT = Box(486, CONTENT_TOP, W - MARGIN, CONTENT_BOTTOM)
FOLD_COLORS = ("accent", "g2", "g3")


def _fold(s: Scene, op: str, captions: tuple[str, str, str], result: str):
    for x in (153, 307):
        s.symbol(x, ARROW_Y, op)
    s.symbol(467, ARROW_Y, "=")
    for box, cap in zip(FOLD_INPUTS, captions):
        s.caption(box.cx, cap, max_w=box.w)
    s.caption(FOLD_RESULT.cx, result, max_w=FOLD_RESULT.w)
    return FOLD_INPUTS, FOLD_RESULT


def _fold_styles(inputs: list[list[tuple[str, str]]]) -> dict[frozenset, str]:
    """Edge -> style: the input's color if only one input has it, else ink."""
    seen: dict[frozenset, list[int]] = {}
    for i, edges in enumerate(inputs):
        for e in edges:
            seen.setdefault(frozenset(e), []).append(i)
    return {e: FOLD_COLORS[ix[0]] if len(ix) == 1 else "default" for e, ix in seen.items()}


def _snapshot(pairs):
    import edgecase as ec

    g = ec.Graph()
    for _ in range(4):
        g.add_node()
    return g.add_edges(pairs)


SNAP = {"1": (0, 0), "2": (1, 0), "3": (1, 1), "4": (0, 1)}  # path 1–2–3–4 runs clockwise


def _snapshots(s: Scene, days: list[list[tuple[int, int]]], styles) -> None:
    for i, (box, pairs) in enumerate(zip(FOLD_INPUTS, days)):
        edges = [(str(a), str(b)) for a, b in pairs]
        s.graph(place(SNAP, box, unit=64, r=R_SMALL),
                [(a, b, {"style": styles[frozenset((a, b))]}) for a, b in edges],
                r=R_SMALL, prefix=f"i{i}.")


@figure("union_all", "set-ops", "Union of many graphs",
        "Three snapshots of contacts among the same four people, nodes 1 to 4: the first has "
        "edge 1–2, the second 2–3 (teal), the third 3–4 (violet) and 1–2 again. Edge 1–2 is "
        "in two snapshots, so it is ink. union_all merges nodes by id and keeps every edge: "
        "the path 1–2–3–4 with 3 edges, each in the color of the snapshot it came from.")
def union_all(s: Scene) -> None:
    import edgecase as ec

    days = [[(1, 2)], [(2, 3)], [(3, 4), (1, 2)]]
    _, out = _fold(s, "∪", ("days[0]", "days[1]", "days[2]"), "Graph.union_all(days)")
    result = ec.Graph.union_all([_snapshot(d) for d in days])
    edges = _label_edges(result)
    assert result.count_nodes() == 4 and edges == {("1", "2"), ("2", "3"), ("3", "4")}
    styles = _fold_styles([[(str(a), str(b)) for a, b in d] for d in days])
    _snapshots(s, days, styles)
    s.graph(place(SNAP, out, unit=UNIT), [(a, b, {"style": styles[frozenset((a, b))]})
                                          for a, b in sorted(edges)], prefix="r.")


@figure("compose_all", "set-ops", "Compose many graphs",
        "Three graphs on the same nodes A, B, C: base has edge A–B (accent), update1 adds B–C "
        "(teal), and update2 adds C–A (violet). compose_all merges nodes and edges by id from "
        "left to right, later graphs winning any attribute clash, so the result is the "
        "triangle A, B, C with 3 edges, each in the color of the graph that supplied it.")
def compose_all(s: Scene) -> None:
    import edgecase as ec

    steps = [[("A", "B")], [("B", "C")], [("A", "C")]]
    ins, out = _fold(s, "∪", ("base", "update1", "update2"), "Graph.compose_all([base, …])")
    result = ec.Graph.compose_all([_factor(list("ABC"), e) for e in steps])
    edges = _label_edges(result)
    assert edges == {("A", "B"), ("B", "C"), ("A", "C")}
    styles = _fold_styles(steps)
    for i, (box, e) in enumerate(zip(ins, steps)):
        s.graph(place(TRI_ABC, box, unit=64, r=R_SMALL),
                [(a, b, {"style": styles[frozenset((a, b))]}) for a, b in e],
                r=R_SMALL, prefix=f"i{i}.")
    s.graph(place(TRI_ABC, out, unit=UNIT),
            [(a, b, {"style": styles[frozenset((a, b))]}) for a, b in sorted(edges)], prefix="r.")


# disjoint_union_all pieces: a triangle, a path of 4 (drawn upright), a star of 5.
PIECES = [
    ({"1": (0.5, 0), "2": (0, 0.87), "3": (1, 0.87)}, [("1", "2"), ("2", "3"), ("1", "3")]),
    ({str(i + 1): (0, i) for i in range(4)}, [("1", "2"), ("2", "3"), ("3", "4")]),
    ({"1": (0, 0), "2": (-0.7, -0.7), "3": (0.7, -0.7), "4": (0.7, 0.7), "5": (-0.7, 0.7)},
     [("1", "2"), ("1", "3"), ("1", "4"), ("1", "5")]),
]


@figure("disjoint_union_all", "set-ops", "Disjoint union of many graphs",
        "Three separate pieces, each numbered from 1: a triangle (accent), a path of 4 nodes "
        "(teal), and a star of 5 nodes (violet). disjoint_union_all never merges ids: the "
        "triangle keeps ids 1 to 3, the path is renumbered 4 to 7 and the star 8 to 12, giving "
        "12 nodes, 10 edges, and 3 connected components.")
def disjoint_union_all(s: Scene) -> None:
    import edgecase as ec

    pieces = [ec.Graph().add_cycle_graph(3), ec.Graph().add_path_graph(4),
              ec.Graph().add_star_graph(5)]
    for g, (pos, e) in zip(pieces, PIECES):
        assert _label_edges(g) == {tuple(sorted(x)) for x in e}
    ins, out = _fold(s, "+", ("pieces[0]", "pieces[1]", "pieces[2]"), "Graph.disjoint_union_all(pieces)")
    combined = ec.Graph.disjoint_union_all(pieces)
    by_id = {tuple(sorted((a, b))) for _, a, b in combined.get_edges_as_tuples()}
    offsets, want, coords, styles = (0, 3, 7), set(), {}, {}
    for (pos, e), off, color, dx in zip(PIECES, offsets, FOLD_COLORS, (0, 1.8, 3.4)):
        want |= {tuple(sorted((int(a) + off, int(b) + off))) for a, b in e}
        dy = 1.5 if color == "g3" else 0 if color == "g2" else 1.07
        coords |= {str(int(k) + off): (x + dx, y + dy) for k, (x, y) in pos.items()}
        styles |= {str(int(k) + off): color for k in pos}
    assert by_id == want and combined.count_weakly_connected_components() == 3
    assert combined.count_nodes() == 12
    for i, (box, (pos, e), color) in enumerate(zip(ins, PIECES, FOLD_COLORS)):
        s.graph(place(pos, box, unit=46, r=R_SMALL), _styled(e, color), r=R_SMALL,
                prefix=f"i{i}.", styles=dict.fromkeys(pos, color))
    s.graph(place(coords, out, unit=46, r=R_SMALL),
            [(str(a), str(b), {"style": styles[str(a)]}) for a, b in sorted(by_id)],
            r=R_SMALL, prefix="r.", styles=styles)


@figure("intersection_all", "set-ops", "Intersection of many graphs",
        "Three snapshots of contacts among nodes 1 to 4: the first has edges 1–2 and 2–3, the "
        "second 1–2 and 3–4 (3–4 in teal, the only edge in just one snapshot), the third 1–2 "
        "and 2–3 again. intersection_all keeps only edges present in every snapshot: just "
        "1–2, so the result has 1 edge and 2 nodes; 2–3 and 3–4 are dashed, and nodes 3 and "
        "4, left with no edges, are dropped.")
def intersection_all(s: Scene) -> None:
    import edgecase as ec

    days = [[(1, 2), (2, 3)], [(1, 2), (3, 4)], [(1, 2), (2, 3)]]
    _, out = _fold(s, "∩", ("days[0]", "days[1]", "days[2]"), "Graph.intersection_all(days)")
    core = ec.Graph.intersection_all([_snapshot(d) for d in days])
    assert _label_edges(core) == {("1", "2")} and core.count_nodes() == 2
    styles = _fold_styles([[(str(a), str(b)) for a, b in d] for d in days])
    _snapshots(s, days, styles)
    s.graph(place(SNAP, out, unit=UNIT),
            [("1", "2", {"style": styles[frozenset(("1", "2"))]})]
            + [(a, b, {"style": "removed"}) for a, b in (("2", "3"), ("3", "4"))],
            prefix="r.", styles={"3": "removed", "4": "removed"})
