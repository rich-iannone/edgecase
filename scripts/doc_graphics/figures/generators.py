"""Generators: methods that add a whole family of graphs.

Template: `multiples`. The same generator at three growing parameter values,
so the reader sees the rule rather than one instance. Every panel uses small
unlabeled nodes (`R_SMALL`) at one density (`UNIT`), and new nodes are drawn
`default`: nothing is highlighted because everything is new. Exceptions that
carry the point of the method: Johnson nodes are labeled with their subsets,
preferential-attachment hubs are `accent`, rewired small-world shortcuts are
`accent`, and the geometric graph shows its connection radius around one
`soft` node.

Shared layouts so siblings can be compared directly:

- Rings (`_ring_xy`) for cycle, complete, star spokes, Johnson, and the four
  random models (G(n,p), G(n,m), preferential attachment, small world). Ring
  nodes are numbered clockwise from the top, matching edgecase's ids 1..n.
- Random generators are drawn from real seeded edgecase output (seed noted
  next to each edge list); the seed is left out of the captions as plumbing.
"""

from core import R_SMALL, Box, Scene, figure, multiples, place, ring

UNIT = 62  # one density for the whole family


def _ring_xy(n: int, radius: float = 1.0) -> dict[str, tuple[float, float]]:
    """Abstract ring coords for nodes "1".."n"; squares sit flat like the cycle."""
    ids = [str(i) for i in range(1, n + 1)]
    return ring(ids, radius, start_deg=-135 if n == 4 else -90)


def _draw(s: Scene, box: Box, coords, edges, *, prefix: str, unit: float = UNIT,
          styles=None, labels=False, edge_style=None, pad: float = 6) -> dict:
    """Place `coords` in `box` and draw the graph with small nodes.

    `edges` are (a, b) pairs of ints or strings; `edge_style` maps a
    frozenset pair to a style for the few edges that carry the point.
    """
    pos = place(coords, box, unit=unit, r=R_SMALL, pad=pad)
    edge_style = edge_style or {}
    es = []
    for a, b in edges:
        a, b = str(a), str(b)
        st = edge_style.get(frozenset((a, b)))
        es.append((a, b, {"style": st}) if st else (a, b))
    s.graph(pos, es, labels=labels, prefix=prefix, r=R_SMALL, styles=styles)
    return pos


def _cycle_edges(n: int, offset: int = 0) -> list[tuple[int, int]]:
    return [(offset + i, offset + i % n + 1) for i in range(1, n + 1)]


# --- deterministic shapes ------------------------------------------------------


@figure("add_path_graph", "generators", "Path graph",
        "Three path graphs side by side: two nodes joined by one edge for n=2, a chain of three "
        "nodes for n=3, and a chain of four nodes for n=4. Each is a single line in which the two "
        "end nodes have one neighbor and every inner node has two.")
def add_path_graph(s: Scene) -> None:
    ns = (2, 3, 4)
    boxes = multiples(s, tuple(f"g.add_path_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        coords = {str(i): (i, 0) for i in range(1, n + 1)}
        _draw(s, box, coords, [(i, i + 1) for i in range(1, n)], prefix=f"n{n}.")


@figure("add_cycle_graph", "generators", "Cycle graph",
        "Three cycle graphs side by side: a triangle for n=3, a square for n=4, and a hexagon "
        "for n=6. Each is a single closed loop in which every node has exactly two neighbors.")
def add_cycle_graph(s: Scene) -> None:
    ns = (3, 4, 6)
    boxes = multiples(s, tuple(f"g.add_cycle_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _draw(s, box, _ring_xy(n), _cycle_edges(n), prefix=f"n{n}.")


@figure("add_star_graph", "generators", "Star graph",
        "Three star graphs side by side: a hub with three spokes for n=4, four spokes for n=5, "
        "and six spokes for n=7. The hub is joined to every other node and the spokes are not "
        "joined to each other, so n nodes give n - 1 edges.")
def add_star_graph(s: Scene) -> None:
    ns = (4, 5, 7)
    boxes = multiples(s, tuple(f"g.add_star_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        # hub "1" in the middle, spokes on the same rings as the cycle figure
        spokes = {str(int(k) + 1): xy for k, xy in _ring_xy(n - 1).items()}
        coords = {"1": (0.0, 0.0), **spokes}
        _draw(s, box, coords, [(1, i) for i in range(2, n + 1)], prefix=f"n{n}.")


@figure("add_full_graph", "generators", "Complete graph",
        "Three complete graphs side by side: a triangle for n=3, a square with both diagonals for "
        "n=4, and a hexagon with every chord for n=6. Every pair of nodes is joined, giving 3, 6, "
        "and 15 edges.")
def add_full_graph(s: Scene) -> None:
    ns = (3, 4, 6)
    boxes = multiples(s, tuple(f"g.add_full_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        edges = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        _draw(s, box, _ring_xy(n), edges, prefix=f"n{n}.")


@figure("add_balanced_tree", "generators", "Balanced tree",
        "Three binary trees side by side for k=2 and heights h=1, 2, and 3: a root with two "
        "children, then two more levels of doubling, ending with eight leaves. Every internal "
        "node has exactly two children and all leaves sit at the same depth.")
def add_balanced_tree(s: Scene) -> None:
    hs = (1, 2, 3)
    boxes = multiples(s, tuple(f"g.add_balanced_tree(k=2, h={h})" for h in hs))
    for box, h in zip(boxes, hs):
        # heap numbering (node i has children 2i, 2i+1); leaves 0.5 units apart
        # at h=3, so each level up doubles the spacing (same scale in all panels)
        coords, edges = {}, []
        for d in range(h + 1):
            step = 0.5 * 2 ** (3 - d)
            for j in range(2 ** d):
                i = 2 ** d + j
                coords[str(i)] = ((j - (2 ** d - 1) / 2) * step, d * 0.75)
                if d:
                    edges.append((i // 2, i))
        # the h=3 row of eight leaves needs the panel's full width
        _draw(s, box, coords, edges, prefix=f"h{h}.", pad=2)


@figure("add_grid_2d", "generators", "2D grid",
        "Three lattices side by side: 2 by 3, 3 by 3, and 3 by 4. Each node is joined only to "
        "its horizontal and vertical neighbors, so corners have two neighbors, border nodes "
        "three, and interior nodes four.")
def add_grid_2d(s: Scene) -> None:
    dims = ((2, 3), (3, 3), (3, 4))
    boxes = multiples(s, tuple(f"g.add_grid_2d(rows={r}, cols={c})" for r, c in dims))
    for box, (rows, cols) in zip(boxes, dims):
        idx = {(r, c): r * cols + c + 1 for r in range(rows) for c in range(cols)}
        coords = {str(i): (c, r) for (r, c), i in idx.items()}
        edges = [(idx[r, c], idx[r, c + 1]) for r in range(rows) for c in range(cols - 1)]
        edges += [(idx[r, c], idx[r + 1, c]) for r in range(rows - 1) for c in range(cols)]
        _draw(s, box, coords, edges, prefix=f"g{rows}{cols}.")


@figure("add_prism_graph", "generators", "Prism graph",
        "Three prism graphs side by side for n=3, 4, and 5, each drawn as an outer n-cycle around "
        "an inner n-cycle with n rungs joining matching nodes. That gives 2n nodes and 3n edges: "
        "6 and 9, 8 and 12, 10 and 15.")
def add_prism_graph(s: Scene) -> None:
    ns = (3, 4, 5)
    boxes = multiples(s, tuple(f"g.add_prism_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        # edgecase: cycle 1..n, cycle n+1..2n, rung i -- i+n
        # inner ring at half size; a little smaller for n=3 so the inner
        # corners keep clear of the outer triangle's sides
        outer = _ring_xy(n)
        inner = {str(int(k) + n): xy for k, xy in _ring_xy(n, 0.4 if n == 3 else 0.5).items()}
        edges = _cycle_edges(n) + _cycle_edges(n, n) + [(i, i + n) for i in range(1, n + 1)]
        _draw(s, box, {**outer, **inner}, edges, prefix=f"n{n}.")


@figure("add_johnson_graph", "generators", "Johnson graph",
        "Three Johnson graphs side by side, each node labeled with its subset. J(3, 1) is a "
        "triangle on 0, 1, 2 and J(4, 1) is a complete graph on four singletons; J(4, 2) is a "
        "hexagon of the six pairs from 0 to 3, where two pairs are joined when they share one "
        "element, so only complementary pairs such as 0,1 and 2,3 stay unjoined.")
def add_johnson_graph(s: Scene) -> None:
    params = ((3, 1), (4, 1), (4, 2))
    boxes = multiples(s, tuple(f"g.add_johnson_graph(n={n}, k={k})" for n, k in params))
    for box, (n, k) in zip(boxes, params):
        if k == 1:
            subsets = [str(i) for i in range(n)]
        else:
            # hexagon order puts complementary pairs opposite each other, so the
            # 12 edges are the six sides plus two triangles of short chords
            subsets = ["0,1", "0,2", "0,3", "2,3", "1,3", "1,2"]
        sets = [set(t.split(",")) for t in subsets]
        ids = [str(i) for i in range(1, len(subsets) + 1)]
        edges = [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))
                 if len(sets[a] & sets[b]) == k - 1]
        _draw(s, box, _ring_xy(len(ids)), edges, prefix=f"j{n}{k}.",
              labels=dict(zip(ids, subsets)))


# --- random models -----------------------------------------------------------
# Edge lists are edgecase's own output for the seed given; nodes sit on the
# shared 8-node ring in id order.

GNP = {  # g.add_gnp_graph(n=8, p=..., seed=39); each graph contains the previous one
    "0.1": [(1, 5), (3, 4), (6, 8)],
    "0.3": [(1, 5), (2, 8), (3, 4), (3, 8), (4, 7), (5, 6), (5, 7), (6, 8)],
    "0.5": [(1, 5), (1, 7), (2, 4), (2, 6), (2, 8), (3, 4), (3, 8), (4, 5), (4, 7), (4, 8),
            (5, 6), (5, 7), (6, 7), (6, 8)],
}


@figure("add_gnp_graph", "generators", "G(n, p) random graph",
        "Three random graphs on the same ring of eight nodes for edge probability p=0.1, 0.3, "
        "and 0.5. Each possible edge appears independently, so the drawings gain edges as p grows "
        "(3, 8, and 14 here) with no pattern to which pairs are joined.")
def add_gnp_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"g.add_gnp_graph(n=8, p={p})" for p in GNP))
    for i, (box, edges) in enumerate(zip(boxes, GNP.values())):
        _draw(s, box, _ring_xy(8), edges, prefix=f"p{i}.")


GNM = {  # g.add_gnm_graph(n=8, m=..., seed=39); each graph contains the previous one
    4: [(2, 3), (3, 6), (3, 8), (4, 8)],
    8: [(1, 8), (2, 3), (2, 8), (3, 6), (3, 8), (4, 8), (6, 7), (7, 8)],
    12: [(1, 6), (1, 8), (2, 3), (2, 6), (2, 8), (3, 4), (3, 6), (3, 8), (4, 8), (5, 6),
         (6, 7), (7, 8)],
}


@figure("add_gnm_graph", "generators", "G(n, m) random graph",
        "Three random graphs on the same ring of eight nodes with exactly m=4, 8, and 12 edges. "
        "The edge count is fixed by m while which pairs get joined is random.")
def add_gnm_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"g.add_gnm_graph(n=8, m={m})" for m in GNM))
    for box, (m, edges) in zip(boxes, GNM.items()):
        _draw(s, box, _ring_xy(8), edges, prefix=f"m{m}.")


PA = {  # g.add_pa_graph(n=8, m=..., seed=2); node 1 is the top-degree hub in all three
    1: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (2, 7)],
    2: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 6), (3, 7), (3, 8), (4, 5),
        (4, 7), (6, 8)],
    3: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (3, 4), (3, 5),
        (3, 6), (3, 7), (4, 5), (4, 6), (4, 8), (6, 7), (6, 8)],
}


@figure("add_pa_graph", "generators", "Preferential attachment graph",
        "Three preferential attachment graphs on eight nodes, placed on a ring in arrival order, "
        "for m=1, 2, and 3 edges per new node. The first node, in the accent color, collects the "
        "most links in every panel (degree 6, 5, and 7), showing how early, well-connected nodes "
        "become hubs.")
def add_pa_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"g.add_pa_graph(n=8, m={m})" for m in PA))
    for box, (m, edges) in zip(boxes, PA.items()):
        _draw(s, box, _ring_xy(8), edges, prefix=f"m{m}.", styles={"1": "accent"})


SW_LATTICE = [(i, j) for i in range(1, 9) for j in (i % 8 + 1, (i + 1) % 8 + 1)]
SW = {  # n=8, k=4, seed=2; p=0 is the ring lattice itself
    "0": SW_LATTICE,
    "0.2": [(1, 2), (1, 3), (1, 6), (1, 7), (2, 3), (2, 4), (2, 8), (3, 4), (3, 5), (4, 5),
            (4, 7), (4, 8), (5, 6), (5, 7), (6, 7), (7, 8)],
    "1": [(1, 2), (1, 4), (1, 5), (1, 6), (1, 8), (2, 4), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7),
          (4, 5), (5, 8), (6, 7), (6, 8), (7, 8)],
}


@figure("add_smallworld_graph", "generators", "Small-world graph",
        "Three Watts-Strogatz graphs on a ring of eight nodes with k=4 and rewiring probability "
        "p=0, 0.2, and 1. At p=0 every node links to its two nearest neighbors on each side; at "
        "p=0.2 three edges are rewired into accent-colored shortcuts across the ring; at p=1 most "
        "edges are rewired and the lattice is gone.")
def add_smallworld_graph(s: Scene) -> None:
    # The full call (g.add_smallworld_graph(n=8, k=4, p=0.2)) is wider than a
    # multiples panel, so the captions give just the arguments.
    boxes = multiples(s, tuple(f"n=8, k=4, p={p}" for p in SW))
    lattice = {frozenset(map(str, e)) for e in SW_LATTICE}
    for i, (box, edges) in enumerate(zip(boxes, SW.values())):
        shortcuts = {frozenset(map(str, e)): "accent" for e in edges
                     if frozenset(map(str, e)) not in lattice}
        _draw(s, box, _ring_xy(8), edges, prefix=f"p{i}.", edge_style=shortcuts)


# g.add_random_geometric_graph(n=7, radius=..., seed=270): node positions in
# the unit square (the same for every radius) and the edges at each radius
GEO_POS = {"1": (0.023185, 0.136684), "2": (0.311760, 0.739284), "3": (0.766448, 0.322370),
           "4": (0.930185, 0.490077), "5": (0.746901, 0.828389), "6": (0.491660, 0.550297),
           "7": (0.014941, 0.705845)}
GEO = {
    "0.25": [(3, 4)],
    "0.4": [(2, 6), (2, 7), (3, 4), (3, 6), (4, 5), (5, 6)],
    "0.55": [(2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6), (6, 7)],
}
GEO_SCALE = 150  # px per unit of the unit square, so the radius circle is true to scale
GEO_HUB = "6"  # the node nearest the middle carries the radius circle


@figure("add_random_geometric_graph", "generators", "Random geometric graph",
        "The same seven random points in the unit square joined at radius=0.25, 0.4, and 0.55. "
        "A faint dashed circle of that radius surrounds the central node, and two points are "
        "joined exactly when they lie within the radius of each other, so edges fill in as the "
        "radius grows (1, 6, and 10 edges).")
def add_random_geometric_graph(s: Scene) -> None:
    # The full call is wider than a multiples panel; captions give the arguments.
    boxes = multiples(s, tuple(f"n=7, radius={r}" for r in GEO))
    hx, hy = GEO_POS[GEO_HUB]
    for i, (box, (r, edges)) in enumerate(zip(boxes, GEO.items())):
        # true-scale positions, shifted so the circled node sits mid-panel
        pos = {k: (box.cx + (x - hx) * GEO_SCALE, box.cy + (y - hy) * GEO_SCALE)
               for k, (x, y) in GEO_POS.items()}
        rad = float(r) * GEO_SCALE
        cx, cy = pos[GEO_HUB]
        s.path(f"M {cx - rad:.1f} {cy:.1f} a {rad:.1f} {rad:.1f} 0 1 0 {2 * rad:.1f} 0 "
               f"a {rad:.1f} {rad:.1f} 0 1 0 {-2 * rad:.1f} 0", color="accent", opacity=0.45,
               width=1, dash="4 4")
        s.graph(pos, [(str(a), str(b)) for a, b in edges], labels=False, prefix=f"r{i}.",
                r=R_SMALL, styles={GEO_HUB: "soft"})
