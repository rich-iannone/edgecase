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
"""

from core import Scene, binary, figure, place

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
