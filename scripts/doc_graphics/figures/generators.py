"""Generators: methods that add a whole family of graphs.

Template: `multiples`. The same generator at three growing parameter values,
so the reader sees the rule rather than one instance. Every panel uses small
unlabeled nodes (`R_SMALL`) at one density (`UNIT`), and new nodes are drawn
`default`: nothing is highlighted because everything is new. Exceptions that
carry the point of the method: Johnson nodes are labeled with their subsets,
preferential-attachment hubs are `accent`, rewired small-world shortcuts are
`accent`, and the geometric graph shows its connection radius around one
`soft` node. The degree-driven models (configuration model, expected degree)
put each node's requested degree or weight in a pill, and the stochastic block
model colors its two blocks `accent` / `g2`. Those three return a new Graph
rather than adding to `g`, so their captions give just the argument.

Shared layouts so siblings can be compared directly:

- Rings (`_ring_xy`) for cycle, complete, star spokes, Johnson, and the four
  random models (G(n,p), G(n,m), preferential attachment, small world). Ring
  nodes are numbered clockwise from the top, matching edgecase's ids 1..n.
- Random generators are drawn from real seeded edgecase output (seed noted
  next to each edge list); the seed is left out of the captions as plumbing.
- The classic shapes, lattices, named graphs, and community benchmarks further
  down run the generator live (`_built`) instead of copying an edge list, so
  their figures can't drift from the code. Partitioned graphs color their
  parts, and community benchmarks color each ground-truth `community`, in group
  order `accent`, `g2`, `g3`, `g4`; edges inside a group take its color and
  edges between groups stay plain.
- Community benchmarks share `_cluster_layout`: each group on a small ring,
  members ordered to keep edges clear of nodes, with a small bow on any edge
  that still grazes one. Their seeds were picked for a lint-clean layout.
- The later random, degree-sequence, directed-growth, and geometric models also
  run live, from the seeds in `SEEDS`. Directed models draw arrows, repeated
  pairs bow apart, trees hang from their center (growing networks from their
  first node), degree-driven models put each node's degree in a pill (`in/out`
  when directed), and geometric models place nodes at their `pos`: the soft
  random geometric graph circles its radius around a `soft` node, and the
  geographical threshold graph shades each node by its weight (heat).
- Bipartite generators draw part 0 as an `accent` top row and part 1 as a
  teal bottom row (`_rows`), with degree pills above and below the rows for
  the degree-driven ones.
- Expanders and the duplication, kernel, and clustered models sit on the ring.
  Self-loops and repeated pairs are part of the picture: a pair repeated twice
  bows apart by the first of `_pair_bends` that clears every node. The
  Margulis-Gabber-Galil n=4 panel and the Sudoku graph are too dense to draw
  whole, so they show their grid with only one `accent` node's edges and its
  neighbors `soft`. Paley graphs at p = 3 mod 4 are drawn directed.
- Furniture placed first: visibility graphs draw dashed stems up from a
  baseline, interval graphs draw each interval as a faint `accent` bar, and
  both then hand their placed coords to `_live(pos=...)`. The prefix tree
  hangs from its root with the NIL sink `soft` and pilled below.
- The Internet AS graph uses the `single` layout: tier-1 nodes on a ring, the
  middle and customer tiers in rows sorted under their upstream links, each
  type in its own color.
- Enumerations (atlas slices, non-isomorphic trees) are galleries
  (`_gallery`): one small graph per grid cell at a smaller unit, atlas graphs
  on rings and trees by `_spine_layout`, their longest path as a row.
"""

import itertools
import math
from collections import Counter

from core import R_SMALL, Box, Scene, compute, figure, multiples, place, ring, single

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
    "1": [(1, 2), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7),
          (4, 5), (4, 8), (5, 6), (5, 7), (6, 7), (7, 8)],
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


# --- degree-driven and block models ------------------------------------------
# These three return a new Graph instead of adding to `g`. The ring is a bit
# smaller (BADGE_RING) so a value pill fits outside every node; pills sit on
# the side facing away from the ring's center.

BADGE_RING = 0.8


def _ring_badges(s: Scene, pos: dict, prefix: str, values: dict[str, str]) -> None:
    """Put `values[k]` in a pill on the outward side of each ring node."""
    cx = sum(x for x, _ in pos.values()) / len(pos)
    cy = sum(y for _, y in pos.values()) / len(pos)
    dirs = ["e", "se", "s", "sw", "w", "nw", "n", "ne"]
    for k, (x, y) in pos.items():
        ang = math.degrees(math.atan2(y - cy, x - cx)) % 360
        s.badge(prefix + k, values[k], where=dirs[round(ang / 45) % 8])


CONFIG = {  # g.configuration_model(degree_sequence=..., seed=3)
    (1, 1, 1, 1): [(1, 4), (2, 3)],
    (2, 2, 2, 2): [(1, 3), (1, 4), (2, 3), (2, 4)],
    (3, 2, 2, 2, 1): [(1, 3), (1, 3), (1, 4), (2, 4), (2, 5)],  # a real multi-edge 1-3
}


@figure("configuration_model", "generators", "Configuration model",
        "Three random graphs built to match a degree sequence, each node's target degree in a "
        "pill: 1, 1, 1, 1 gives two separate edges; 2, 2, 2, 2 gives a four-cycle; and 3, 2, 2, "
        "2, 1 on five nodes includes a doubled edge, because the model pairs edge stubs at "
        "random and may create multi-edges. Every node ends with exactly the degree in its pill.")
def configuration_model(s: Scene) -> None:
    boxes = multiples(s, tuple(f"degree_sequence=[{', '.join(map(str, d))}]" for d in CONFIG))
    for i, (box, (seq, edges)) in enumerate(zip(boxes, CONFIG.items())):
        coords = _ring_xy(len(seq), BADGE_RING)
        pos = place(coords, box, unit=UNIT, r=R_SMALL)
        es, seen = [], set()
        for a, b in edges:
            a, b = str(a), str(b)
            if (a, b) in seen:  # second copy of a multi-edge: bow both copies apart
                es = [e for e in es if (e[0], e[1]) != (a, b)]
                es += [(a, b, {"bend": 10}), (b, a, {"bend": 10})]
            else:
                es.append((a, b))
            seen.add((a, b))
        s.graph(pos, es, labels=False, prefix=f"c{i}.", r=R_SMALL)
        _ring_badges(s, pos, f"c{i}.", {str(k): str(d) for k, d in enumerate(seq, 1)})


CHUNG_LU = {  # g.expected_degree_graph(weights=..., seed=3)
    (1, 1, 1, 1, 1, 1): [(1, 3), (1, 5)],
    (2, 2, 2, 2, 2, 2): [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (2, 6), (3, 4)],
    (4, 2, 2, 1, 1, 1): [(1, 2), (1, 3), (1, 4), (1, 5)],
}


@figure("expected_degree_graph", "generators", "Expected degree graph",
        "Three Chung-Lu random graphs on a ring of six nodes, each node's weight, its expected "
        "degree, in a pill. All weights 1 give 2 edges and all weights 2 give 7; with weights "
        "4, 2, 2, 1, 1, 1 the heavy top node gets 4 links. Degrees match the weights only on "
        "average, so individual nodes can land above or below their pill.")
def expected_degree_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"weights=[{', '.join(map(str, w))}]" for w in CHUNG_LU))
    for i, (box, (w, edges)) in enumerate(zip(boxes, CHUNG_LU.items())):
        pos = _draw(s, box, _ring_xy(len(w), BADGE_RING), edges, prefix=f"w{i}.")
        _ring_badges(s, pos, f"w{i}.", {str(k): str(x) for k, x in enumerate(w, 1)})


# g.generate_stochastic_block_model(sizes=[4, 4], p_matrix=..., seed=16). Both
# blocks come out complete in all three panels; only the cross edges change.
SBM_K4 = [(a, b) for blk in ((1, 2, 3, 4), (5, 6, 7, 8)) for i, a in enumerate(blk) for b in blk[i + 1:]]
SBM = {
    "0": [],
    "0.1": [(4, 5), (4, 8)],
    "0.5": [(1, 7), (2, 6), (2, 7), (2, 8), (3, 6), (4, 5), (4, 8)],
}


@figure("generate_stochastic_block_model", "generators", "Stochastic block model",
        "Three stochastic block models with two blocks of four nodes, block one in the accent "
        "color on the left and block two in teal on the right, with edge probability 0.9 inside "
        "a block and 0, 0.1, and 0.5 between blocks. Both blocks come out fully connected in every "
        "panel while the plain cross-block edges grow from none to 2 to 7, blurring the split.")
def generate_stochastic_block_model(s: Scene) -> None:
    boxes = multiples(s, tuple(f"p_matrix=[0.9, {p}, {p}, 0.9]" for p in SBM))
    # ids 1..8 clockwise on a slightly wide ring: block 1 fills the left half
    # (bottom to top), block 2 the right half (top to bottom)
    coords = {k: (x * 1.25, y) for k, (x, y) in
              ring([str(i) for i in range(1, 9)], start_deg=112.5).items()}
    block = {str(i): "accent" if i <= 4 else "g2" for i in range(1, 9)}
    for i, (box, cross) in enumerate(zip(boxes, SBM.values())):
        pos = place(coords, box, unit=UNIT, r=R_SMALL)
        es = [(str(a), str(b), {"style": block[str(a)]}) for a, b in SBM_K4]
        s.graph(pos, es + [(str(a), str(b)) for a, b in cross], labels=False, prefix=f"b{i}.",
                r=R_SMALL, styles=block)


# --- classic shapes, lattices, and named graphs --------------------------------
# These draw edgecase's own output live (`_built`), so a figure can't drift from
# the generator. Every graph starts empty, so node ids are 1..n in creation
# order and the layouts below key on them. Partitioned graphs (bipartite,
# multipartite, Turán) color their parts in group order: `accent`, `g2`, `g3`,
# `g4`.

GROUP_STYLES = ("accent", "g2", "g3", "g4")


def _built(call: str, directed: bool = False):
    """Run `g.<call>` on an empty edgecase graph; return (graph, edges as str pairs)."""
    import edgecase as ec

    g = eval(f"g.{call}", {"g": ec.Graph(directed=directed)})
    return g, [(str(a), str(b)) for _, a, b in g.get_edges_as_tuples()]


def _labels(g) -> dict[str, str]:
    return {str(nid): g.get_label_from_id(nid) for nid in sorted(g.get_node_ids())}


def _group_styles(g, attr: str) -> dict[str, str]:
    """Node id -> group style, groups numbered in order of their smallest member."""
    values = g.get_node_attrs(attr)
    order = []
    for nid in sorted(values):
        if values[nid] not in order:
            order.append(values[nid])
    return {str(nid): GROUP_STYLES[order.index(v)] for nid, v in values.items()}


def _colored_edges(edges, styles) -> list:
    """Edges inside a group take the group's color; edges between groups stay plain."""
    return [(a, b, {"style": styles[a]}) if styles[a] == styles[b] else (a, b)
            for a, b in edges]


def _clique_ring(ids: list[str], center: tuple[float, float], radius: float,
                 face_deg: float) -> dict[str, tuple[float, float]]:
    """`ids` on a small ring around `center`, the last id pointing at `face_deg`."""
    n = len(ids)
    return ring(ids, radius, start_deg=face_deg - 360 * (n - 1) / n, center=center)


def _tidy_tree(children: dict[str, list[str]], root: str, dy: float = 1.0) -> dict:
    """Leaves on consecutive slots in depth-first order, parents centered above."""
    coords: dict[str, tuple[float, float]] = {}
    slot = [0]

    def visit(v: str, depth: int) -> float:
        kids = children.get(v, [])
        if not kids:
            x = slot[0]
            slot[0] += 1
        else:
            xs = [visit(c, depth + 1) for c in kids]
            x = (xs[0] + xs[-1]) / 2
        coords[v] = (x, depth * dy)
        return x

    visit(root, 0)
    return coords


def _children(edges) -> dict[str, list[str]]:
    kids: dict[str, list[str]] = {}
    for a, b in edges:
        kids.setdefault(a, []).append(b)
    return kids


@figure("add_complete_bipartite_graph", "generators", "Complete bipartite graph",
        "Three complete bipartite graphs side by side for sizes 2 and 2, 2 and 3, and 3 and 4. "
        "The first part sits in a top row in the accent color and the second in a teal bottom "
        "row; every top node is joined to every bottom node and no two nodes in the same row "
        "are joined, giving 4, 6, and 12 edges.")
def add_complete_bipartite_graph(s: Scene) -> None:
    params = ((2, 2), (2, 3), (3, 4))
    boxes = multiples(s, tuple(f"n1={a}, n2={b}" for a, b in params))
    for box, (n1, n2) in zip(boxes, params):
        g, edges = _built(f"add_complete_bipartite_graph({n1}, {n2})")
        coords = {str(i + 1): (i - (n1 - 1) / 2, 0) for i in range(n1)}
        coords |= {str(n1 + i + 1): (i - (n2 - 1) / 2, 1.5) for i in range(n2)}
        styles = _group_styles(g, "part")
        _draw(s, box, coords, edges, prefix=f"b{n1}{n2}.", styles=styles)


@figure("add_complete_multipartite_graph", "generators", "Complete multipartite graph",
        "Three complete multipartite graphs on a ring of six nodes, each part in its own color: "
        "sizes 1, 2, 3; sizes 2, 2, 2 (the octahedron); and sizes 1, 1, 2, 2. Two nodes are "
        "joined exactly when they belong to different parts, so the only missing pairs are "
        "inside a part.")
def add_complete_multipartite_graph(s: Scene) -> None:
    params = ((1, 2, 3), (2, 2, 2), (1, 1, 2, 2))
    boxes = multiples(s, tuple(f"sizes=[{', '.join(map(str, p))}]" for p in params))
    for i, (box, sizes) in enumerate(zip(boxes, params)):
        g, edges = _built(f"add_complete_multipartite_graph({list(sizes)})")
        _draw(s, box, _ring_xy(6), edges, prefix=f"m{i}.", styles=_group_styles(g, "part"))


@figure("add_turan_graph", "generators", "Turán graph",
        "Three Turán graphs on a ring of seven nodes for r=2, 3, and 4 parts, each part in its "
        "own color. The seven nodes are split into parts as equal as possible (4 and 3; 3, 2, "
        "and 2; 2, 2, 2, and 1) and every pair of nodes in different parts is joined, so the "
        "graph gains edges as r grows (12, 16, and 18).")
def add_turan_graph(s: Scene) -> None:
    rs = (2, 3, 4)
    boxes = multiples(s, tuple(f"g.add_turan_graph(n=7, r={r})" for r in rs))
    for box, r in zip(boxes, rs):
        g, edges = _built(f"add_turan_graph(7, {r})")
        _draw(s, box, _ring_xy(7), edges, prefix=f"r{r}.", styles=_group_styles(g, "part"))


@figure("add_wheel_graph", "generators", "Wheel graph",
        "Three wheel graphs side by side for n=4, 5, and 7: a hub in the middle joined to every "
        "node of a rim cycle of three, four, and six nodes. The hub has n - 1 neighbors and "
        "every rim node has three.")
def add_wheel_graph(s: Scene) -> None:
    ns = (4, 5, 7)
    boxes = multiples(s, tuple(f"g.add_wheel_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _, edges = _built(f"add_wheel_graph({n})")
        rim = {str(int(k) + 1): xy for k, xy in _ring_xy(n - 1).items()}
        _draw(s, box, {"1": (0.0, 0.0), **rim}, edges, prefix=f"n{n}.")


@figure("add_ladder_graph", "generators", "Ladder graph",
        "Three ladder graphs side by side for n=2, 3, and 4: two parallel paths of n nodes, the "
        "rails, joined by n rungs between matching nodes. The ends have two neighbors and the "
        "inner nodes three, for 3n - 2 edges.")
def add_ladder_graph(s: Scene) -> None:
    ns = (2, 3, 4)
    boxes = multiples(s, tuple(f"g.add_ladder_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _, edges = _built(f"add_ladder_graph({n})")
        coords = {str(i + 1): (i, 0) for i in range(n)}
        coords |= {str(n + i + 1): (i, 1) for i in range(n)}
        _draw(s, box, coords, edges, prefix=f"n{n}.")


@figure("add_circulant_graph", "generators", "Circulant graph",
        "Three circulant graphs on the same ring of eight nodes. With offsets 1 each node joins "
        "its two ring neighbors, giving a cycle; offsets 1 and 2 add the chords that skip one "
        "node; offsets 1 and 3 instead add chords that skip two. Every node has the same "
        "number of neighbors, twice the number of offsets.")
def add_circulant_graph(s: Scene) -> None:
    params = ((1,), (1, 2), (1, 3))
    boxes = multiples(s, tuple(f"n=8, offsets=[{', '.join(map(str, p))}]" for p in params))
    for i, (box, offsets) in enumerate(zip(boxes, params)):
        _, edges = _built(f"add_circulant_graph(8, {list(offsets)})")
        _draw(s, box, _ring_xy(8), edges, prefix=f"c{i}.")


def _clique_coords(ids: list[str], joint_x: float, face_deg: float, radius: float) -> dict:
    """A clique ring whose last id (the joint) sits at (joint_x, 0) facing `face_deg`."""
    cx = joint_x - radius * math.cos(math.radians(face_deg))
    return _clique_ring(ids, (cx, 0.0), radius, face_deg)


def _clique_radius(m: int) -> float:
    return {3: 0.5, 4: 0.6, 5: 0.75}[m]


@figure("add_barbell_graph", "generators", "Barbell graph",
        "Three barbell graphs side by side: two triangles joined by one edge for m1=3 and m2=0, "
        "two triangles joined through a path of two nodes for m2=2, and two complete graphs on "
        "four nodes joined through one path node for m1=4 and m2=1. The cliques are the bells "
        "and the path is the bar.")
def add_barbell_graph(s: Scene) -> None:
    params = ((3, 0), (3, 2), (4, 1))
    boxes = multiples(s, tuple(f"g.add_barbell_graph(m1={a}, m2={b})" for a, b in params))
    for box, (m1, m2) in zip(boxes, params):
        _, edges = _built(f"add_barbell_graph({m1}, {m2})")
        rad = _clique_radius(m1)
        coords = _clique_coords([str(i) for i in range(1, m1 + 1)], 0.0, 0, rad)
        coords |= {str(m1 + j): (float(j), 0.0) for j in range(1, m2 + 1)}
        # the right bell's joint (its first id) faces left, so list it last
        right = [str(m1 + m2 + i) for i in range(2, m1 + 1)] + [str(m1 + m2 + 1)]
        coords |= _clique_coords(right, m2 + 1.0, 180, rad)
        _draw(s, box, coords, edges, prefix=f"b{m1}{m2}.", pad=2)


@figure("add_lollipop_graph", "generators", "Lollipop graph",
        "Three lollipop graphs side by side: a triangle with a one-node tail for m=3 and n=1, a "
        "complete graph on four nodes with a two-node tail, and a complete graph on five nodes "
        "with a two-node tail. The tail is a path hanging off one node of the clique.")
def add_lollipop_graph(s: Scene) -> None:
    params = ((3, 1), (4, 2), (5, 2))
    boxes = multiples(s, tuple(f"g.add_lollipop_graph(m={m}, n={n})" for m, n in params))
    for box, (m, n) in zip(boxes, params):
        _, edges = _built(f"add_lollipop_graph({m}, {n})")
        coords = _clique_coords([str(i) for i in range(1, m + 1)], 0.0, 0, _clique_radius(m))
        coords |= {str(m + j): (float(j), 0.0) for j in range(1, n + 1)}
        _draw(s, box, coords, edges, prefix=f"l{m}{n}.", pad=2)


@figure("add_tadpole_graph", "generators", "Tadpole graph",
        "Three tadpole graphs side by side: a triangle with a one-node tail for m=3 and n=1, a "
        "square with a two-node tail, and a hexagon with a two-node tail. Unlike the lollipop, "
        "the head is a plain cycle, so every head node but the one holding the tail has two "
        "neighbors.")
def add_tadpole_graph(s: Scene) -> None:
    params = ((3, 1), (4, 2), (6, 2))
    boxes = multiples(s, tuple(f"g.add_tadpole_graph(m={m}, n={n})" for m, n in params))
    for box, (m, n) in zip(boxes, params):
        _, edges = _built(f"add_tadpole_graph({m}, {n})")
        rad = {3: 0.5, 4: 0.6, 6: 0.8}[m]
        coords = _clique_coords([str(i) for i in range(1, m + 1)], 0.0, 0, rad)
        coords |= {str(m + j): (float(j), 0.0) for j in range(1, n + 1)}
        _draw(s, box, coords, edges, prefix=f"t{m}{n}.", pad=2)


@figure("add_windmill_graph", "generators", "Windmill graph",
        "Three windmill graphs side by side, each a hub shared by several blades: three "
        "triangles (the friendship graph) for n=3 and k=3, four triangles for n=4, and three "
        "complete graphs on four nodes for k=4. Blades meet only at the hub, so removing it "
        "leaves n separate pieces.")
def add_windmill_graph(s: Scene) -> None:
    params = ((3, 3), (4, 3), (3, 4))
    boxes = multiples(s, tuple(f"g.add_windmill_graph(n={n}, k={k})" for n, k in params))
    for box, (n, k) in zip(boxes, params):
        _, edges = _built(f"add_windmill_graph({n}, {k})")
        coords = {"1": (0.0, 0.0)}
        spread = 24 if n == 3 else 19
        for b in range(n):
            theta = -90 + 360 * b / n
            blade = [str(2 + b * (k - 1) + j) for j in range(k - 1)]
            # two blade nodes flank the blade's direction; a third sits farther out on it
            spots = [(1.0, theta - spread), (1.0, theta + spread), (1.6, theta)][: k - 1]
            for nid, (r, deg) in zip(blade, spots):
                coords[nid] = (r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)))
        _draw(s, box, coords, edges, prefix=f"w{n}{k}.", pad=2)


def _cube_xy(coord: tuple[int, ...]) -> tuple[float, float]:
    """Oblique projection: axis 0 across, axis 1 down, axis 2 back (up and to the right)."""
    x, y = float(coord[0]), float(coord[1]) if len(coord) > 1 else 0.0
    if len(coord) > 2:
        x, y = x + 0.5 * coord[2], y - 0.4 * coord[2]
    return x, y


def _coord(label: str) -> tuple[int, ...]:
    """"(1,0,2)" -> (1, 0, 2)."""
    return tuple(int(v) for v in label.strip("()").split(","))


@figure("add_hypercube_graph", "generators", "Hypercube graph",
        "Three hypercube graphs side by side for d=1, 2, and 3: a single edge, a square, and a "
        "cube drawn in perspective. Each dimension doubles the graph, joining every node to its "
        "copy, so the d-cube has 2^d nodes and every node has d neighbors.")
def add_hypercube_graph(s: Scene) -> None:
    ds = (1, 2, 3)
    boxes = multiples(s, tuple(f"g.add_hypercube_graph(d={d})" for d in ds))
    for box, d in zip(boxes, ds):
        g, edges = _built(f"add_hypercube_graph({d})")
        # labels are bit strings; bit i picks the i-th axis of the projection
        coords = {k: _cube_xy(tuple(int(b) for b in reversed(lab)))
                  for k, lab in _labels(g).items()}
        _draw(s, box, coords, edges, prefix=f"d{d}.")


@figure("add_grid_graph", "generators", "Grid graph",
        "Three grid graphs side by side: a 3 by 3 grid; a three-dimensional 3 by 2 by 2 grid "
        "drawn in perspective, in which each node also joins its neighbor along the depth axis; "
        "and a 6 by 2 grid with periodic=True, drawn as two rings: the axis of length 6 wraps "
        "around into a cycle, while the axis of length 2 is too short to wrap, so its two "
        "positions are just joined by rungs.")
def add_grid_graph(s: Scene) -> None:
    calls = ("dims=[3, 3]", "dims=[3, 2, 2]", "dims=[6, 2], periodic=True")
    boxes = multiples(s, calls)
    for i, (box, call) in enumerate(zip(boxes, calls)):
        g, edges = _built(f"add_grid_graph({call})")
        if "periodic" in call:
            # the wrapped axis runs around a ring, the short one picks the radius
            coords = {}
            for k, lab in _labels(g).items():
                a, b = _coord(lab)
                theta = math.radians(-90 + 60 * a)
                coords[k] = ((1 - 0.5 * b) * math.cos(theta), (1 - 0.5 * b) * math.sin(theta))
        else:
            # dims[0] runs across and dims[1] down, like the 2D grid figure
            coords = {k: _cube_xy(_coord(lab)) for k, lab in _labels(g).items()}
        _draw(s, box, coords, edges, prefix=f"g{i}.")


@figure("add_triangular_lattice_graph", "generators", "Triangular lattice graph",
        "Three triangular lattices side by side: one row of three triangles, two rows of four, "
        "and three rows of five. Every other row of nodes is shifted half a step, each node "
        "joins its neighbors in its own row and the two nodes touching it in the next row, and "
        "all the faces are triangles.")
def add_triangular_lattice_graph(s: Scene) -> None:
    dims = ((1, 3), (2, 4), (3, 5))
    boxes = multiples(s, tuple(f"rows={r}, cols={c}" for r, c in dims))
    for box, (rows, cols) in zip(boxes, dims):
        g, edges = _built(f"add_triangular_lattice_graph({rows}, {cols})")
        coords = {}
        for k, lab in _labels(g).items():
            r, c = _coord(lab)
            coords[k] = (c + 0.5 * (r % 2), -r * math.sqrt(3) / 2)
        _draw(s, box, coords, edges, prefix=f"t{rows}{cols}.")


@figure("add_hexagonal_lattice_graph", "generators", "Hexagonal lattice graph",
        "Three honeycomb lattices side by side: a single hexagon, one row of three hexagons, "
        "and two rows of two. Neighboring hexagons share an edge, every node has two or three "
        "neighbors, and there are no triangles.")
def add_hexagonal_lattice_graph(s: Scene) -> None:
    dims = ((1, 1), (1, 3), (2, 2))
    boxes = multiples(s, tuple(f"rows={r}, cols={c}" for r, c in dims))
    for box, (rows, cols) in zip(boxes, dims):
        g, edges = _built(f"add_hexagonal_lattice_graph({rows}, {cols})")
        coords = {}
        for k, lab in _labels(g).items():
            r, c = _coord(lab)
            # NetworkX's honeycomb embedding: columns zigzag, rows climb
            coords[k] = (0.5 + c + c // 2 + (r % 2) * ((c % 2) - 0.5), -r * math.sqrt(3) / 2)
        _draw(s, box, coords, edges, prefix=f"h{rows}{cols}.")


KNESER_XY = {
    # K(4, 2): each pair above its complement
    (4, 2): {"0,1": (0, 0), "0,2": (1, 0), "0,3": (2, 0),
             "2,3": (0, 1), "1,3": (1, 1), "1,2": (2, 1)},
    # K(5, 2): the Petersen drawing; spokes join each outer pair to the inner pair
    # disjoint from it that isn't on the outer cycle
    (5, 2): {**ring(["0,1", "2,3", "0,4", "1,2", "3,4"], 1.0),
             **ring(["2,4", "1,4", "1,3", "0,3", "0,2"], 0.5)},
}


@figure("add_kneser_graph", "generators", "Kneser graph",
        "Three Kneser graphs side by side, each node labeled with its subset, joined when the "
        "subsets are disjoint. K(4, 1) is the complete graph on four singletons; K(4, 2) is "
        "three separate edges, each pair joined to its complement; K(5, 2) is the Petersen "
        "graph on the ten pairs from 0 to 4.")
def add_kneser_graph(s: Scene) -> None:
    params = ((4, 1), (4, 2), (5, 2))
    boxes = multiples(s, tuple(f"g.add_kneser_graph(n={n}, k={k})" for n, k in params))
    for box, (n, k) in zip(boxes, params):
        g, edges = _built(f"add_kneser_graph({n}, {k})")
        labels = _labels(g)
        layout = KNESER_XY.get((n, k)) or dict(zip(labels.values(), _ring_xy(n).values()))
        coords = {nid: layout[lab] for nid, lab in labels.items()}
        _draw(s, box, coords, edges, prefix=f"k{n}{k}.", labels=labels)


def _cycle_order(edges, start: str) -> list[str]:
    """Walk a graph that is a single cycle from `start`."""
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    order = [start]
    while len(order) < len(adj):
        order.append(next(v for v in adj[order[-1]] if v not in order))
    return order


@figure("add_mycielski_graph", "generators", "Mycielski graph",
        "Three Mycielski graphs side by side: a single edge for n=2, a five-cycle for n=3, and "
        "the 11-node Grötzsch graph for n=4, drawn as the five-cycle on the outside, a shadow "
        "copy of each of its nodes inside, and a hub in the middle joined to every shadow. "
        "None of them contains a triangle, yet each needs one more color than the last.")
def add_mycielski_graph(s: Scene) -> None:
    ns = (2, 3, 4)
    boxes = multiples(s, tuple(f"g.add_mycielski_graph(n={n})" for n in ns))
    _, m3 = _built("add_mycielski_graph(3)")
    outer = _cycle_order(m3, "1")
    for box, n in zip(boxes, ns):
        _, edges = _built(f"add_mycielski_graph({n})")
        if n == 2:
            coords = {"1": (0.0, 0.0), "2": (1.0, 0.0)}
        else:
            coords = dict(zip(outer, _ring_xy(5).values()))
        if n == 4:
            # shadow of node i is node i + 5, at the same angle; the hub is node 11
            coords |= {str(int(k) + 5): (0.5 * x, 0.5 * y) for k, (x, y) in coords.items()}
            coords["11"] = (0.0, 0.0)
        _draw(s, box, coords, edges, prefix=f"m{n}.")


@figure("add_binomial_tree", "generators", "Binomial tree",
        "Three binomial trees side by side for orders 1, 2, and 3, roots at the top. Each is "
        "two copies of the tree before it with one root hung under the other, so the size "
        "doubles from 2 to 4 to 8 nodes and the root gains one child per order.")
def add_binomial_tree(s: Scene) -> None:
    orders = (1, 2, 3)
    boxes = multiples(s, tuple(f"g.add_binomial_tree(order={k})" for k in orders))
    for box, k in zip(boxes, orders):
        _, edges = _built(f"add_binomial_tree({k})", directed=True)
        # largest subtree on the left, as binomial trees are usually drawn
        kids = {p: sorted(c, key=int, reverse=True) for p, c in _children(edges).items()}
        _draw(s, box, _tidy_tree(kids, "1"), edges, prefix=f"b{k}.", pad=2)


@figure("add_full_rary_tree", "generators", "Full r-ary tree",
        "Three full r-ary trees side by side, roots at the top: a complete binary tree of seven "
        "nodes; a binary tree of ten nodes, whose last level is filled from the left and stops "
        "partway; and a ternary tree of eight nodes, where the root's first child takes three "
        "children and the second child takes the one node left.")
def add_full_rary_tree(s: Scene) -> None:
    params = ((2, 7), (2, 10), (3, 8))
    boxes = multiples(s, tuple(f"g.add_full_rary_tree(r={r}, n={n})" for r, n in params))
    for box, (r, n) in zip(boxes, params):
        _, edges = _built(f"add_full_rary_tree({r}, {n})", directed=True)
        kids = {p: sorted(c, key=int) for p, c in _children(edges).items()}
        _draw(s, box, _tidy_tree(kids, "1"), edges, prefix=f"r{r}{n}.", pad=2)


@figure("add_lcf_graph", "generators", "LCF graph",
        "Three LCF graphs, each a ring of nodes in Hamiltonian-cycle order plus one chord per "
        "node set by the shift list: shift 3 on six nodes gives the utility graph K(3,3); "
        "shifts 3 and -3 on eight nodes give the cube; shifts 5 and -5 on fourteen nodes give "
        "the Heawood graph. Every node has exactly three neighbors.")
def add_lcf_graph(s: Scene) -> None:
    params = ((6, [3], 6), (8, [3, -3], 4), (14, [5, -5], 7))
    boxes = multiples(s, tuple(f"n={n}, shifts={sh}, repeats={rep}" for n, sh, rep in params))
    for box, (n, shifts, rep) in zip(boxes, params):
        _, edges = _built(f"add_lcf_graph({n}, {shifts}, {rep})")
        # fourteen nodes need a little more than UNIT to keep the ring spaced out
        _draw(s, box, _ring_xy(n), edges, prefix=f"l{n}.", unit=70, pad=2)


NAMED_XY = {
    # nodes 0, 1 along the floor, 2, 3 under the eaves, 4 at the peak
    "house": {"1": (0, 1.8), "2": (1.2, 1.8), "3": (0, 0.6), "4": (1.2, 0.6), "5": (0.6, 0)},
    # outer five-cycle 0..4 and inner pentagram 5..9, spoke i -- i + 5
    "petersen": {**_ring_xy(5), **{str(int(k) + 5): (0.5 * x, 0.5 * y)
                                   for k, (x, y) in _ring_xy(5).items()}},
    # a hexagon with each node opposite the one it misses (0-5, 1-4, 2-3)
    "octahedral": dict(zip(["1", "2", "3", "6", "5", "4"], _ring_xy(6).values())),
}


@figure("add_named_graph", "generators", "Named graph",
        "Three graphs from the named-graph registry side by side: the house, a square with a "
        "triangular roof; the Petersen graph, an outer five-cycle joined by spokes to an inner "
        "five-pointed star; and the octahedral graph, a hexagon in which every node is joined to "
        "all the others except the one opposite it.")
def add_named_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f'g.add_named_graph("{name}")' for name in NAMED_XY))
    for box, (name, coords) in zip(boxes, NAMED_XY.items()):
        _, edges = _built(f'add_named_graph("{name}")')
        _draw(s, box, coords, edges, prefix=f"{name}.")


# --- community benchmarks ------------------------------------------------------
# Each group sits on its own small ring and the rings are spread around a larger
# one. Nodes are colored by their ground-truth `community` attribute, edges
# inside a group take its color, and edges between groups stay plain ink (as in
# the stochastic block model figure), so mixing shows up as plain edges. Random
# models use a fixed seed, left out of the captions as plumbing.

CLUSTER_GAP = 0.55  # neighbor spacing on a group's ring, abstract units


def _cluster_radius(k: int) -> float:
    if k == 1:
        return 0.0
    if k == 2:
        return CLUSTER_GAP / 2
    return CLUSTER_GAP / (2 * math.sin(math.pi / k))


CLEARANCE = 0.27  # closest an edge may pass a node, abstract units (lint: r + 3 px)


def _seg_dist(p, a, b) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / ((dx * dx + dy * dy) or 1.0)
    t = min(1.0, max(0.0, t))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _grazes(coords: dict, edges) -> int:
    """Number of (edge, node) pairs where the edge passes too close to the node."""
    return sum(_seg_dist(coords[v], coords[a], coords[b]) < CLEARANCE
               for a, b in edges for v in coords if v not in (a, b))


def _cluster_layout(groups: list[list[str]], edges, gap: float = 0.6) -> dict:
    """Each group on a small ring, arranged so edges between groups are short and clear.

    The groups' rings are spread around a larger ring. Each group then tries every
    order and a few rotations of its members, keeping the one with the fewest edges
    grazing a node and, among those, the shortest edges to other groups.
    """
    radii = [_cluster_radius(len(m)) for m in groups]
    count = len(groups)
    if count == 1:
        centers = [(0.0, 0.0)]
    else:
        big = max((radii[i] + radii[(i + 1) % count] + gap) / (2 * math.sin(math.pi / count))
                  for i in range(count))
        start = {2: 180, 4: -135}.get(count, -90)
        centers = list(ring([str(i) for i in range(count)], big, start_deg=start).values())
    home = {v: c for c, members in enumerate(groups) for v in members}
    coords = {v: centers[home[v]] for v in home}
    for sweep in range(3):  # later sweeps see where the other groups' members ended up
        for c, members in enumerate(groups):
            outside = [(a, b) if home[a] == c else (b, a) for a, b in edges
                       if (home[a] == c) != (home[b] == c)]
            best, best_cost = None, (math.inf, math.inf)
            k = len(members)
            orders = itertools.permutations(members) if k <= 6 else [members]
            for order in orders:
                for step in range(4):
                    xy = ring(list(order), radii[c], start_deg=-90 + 90 * step / k,
                              center=centers[c])
                    length = sum(math.dist(xy[a], coords[b]) for a, b in outside)
                    if length >= best_cost[1] + 1e-9 and sweep == 0:
                        continue
                    cost = (_grazes(coords | xy, edges) if sweep else 0, length)
                    if cost < (best_cost[0], best_cost[1] - 1e-9):
                        best, best_cost = xy, cost
            coords |= best
    return coords


def _bend_clears(pos: dict, a: str, b: str, bend: float, gap: float = 4) -> bool:
    """True if edge a-b bowed by `bend` px passes every other node by at least `gap` px."""
    (ax, ay), (bx, by) = pos[a], pos[b]
    dist = math.hypot(bx - ax, by - ay)
    px, py = -(by - ay) / dist, (bx - ax) / dist
    others = [xy for v, xy in pos.items() if v not in (a, b)]
    cx, cy = (ax + bx) / 2 + px * bend * 2, (ay + by) / 2 + py * bend * 2
    pts = [((1 - t) ** 2 * ax + 2 * (1 - t) * t * cx + t * t * bx,
            (1 - t) ** 2 * ay + 2 * (1 - t) * t * cy + t * t * by)
           for t in (i / 40 for i in range(41))]
    return all(_seg_dist(o, p, q) >= R_SMALL + gap for o in others for p, q in zip(pts, pts[1:]))


def _clear_bend(pos: dict, a: str, b: str) -> float:
    """Smallest bow (px) that keeps edge a-b clear of every other node; 0 if straight is."""
    for bend in (0, *(sign * size for size in range(8, 21, 4) for sign in (1, -1))):
        if _bend_clears(pos, a, b, bend):
            return bend
    return 0


def _pair_bends(pos: dict, a: str, b: str) -> tuple[float, float]:
    """Bows for two copies of a-b: either side of the line, or both to one side of a node."""
    for pair in ((10, -10), (12, 24), (-12, -24), (16, 30), (-16, -30)):
        if all(_bend_clears(pos, a, b, bend, gap=3) for bend in pair):
            return pair
    return (10, -10)


def _community_panel(s: Scene, box: Box, call: str, prefix: str, gap: float = 0.6) -> None:
    g, edges = _built(call)
    styles = _group_styles(g, "community")
    groups: dict[str, list[str]] = {}
    for nid in sorted(styles, key=int):
        groups.setdefault(styles[nid], []).append(nid)
    coords = _cluster_layout(list(groups.values()), edges, gap=gap)
    pos = place(coords, box, unit=UNIT, r=R_SMALL)
    drawn = []
    for a, b, *opts in _colored_edges(edges, styles):
        opts = dict(*opts)
        if bend := _clear_bend(pos, a, b):
            opts["bend"] = bend
        drawn.append((a, b, opts))
    s.graph(pos, drawn, labels=False, prefix=prefix, r=R_SMALL, styles=styles)


@figure("add_caveman_graph", "generators", "Caveman graph",
        "Three caveman graphs side by side, each clique in its own color: two triangles, three "
        "triangles, and three complete graphs on four nodes. The cliques are completely "
        "separate, with no edges between them, so each one is its own community.")
def add_caveman_graph(s: Scene) -> None:
    params = ((2, 3), (3, 3), (3, 4))
    boxes = multiples(s, tuple(f"num_cliques={c}, clique_size={k}" for c, k in params))
    for box, (c, k) in zip(boxes, params):
        _community_panel(s, box, f"add_caveman_graph({c}, {k})", f"c{c}{k}.")


@figure("add_connected_caveman_graph", "generators", "Connected caveman graph",
        "Three connected caveman graphs side by side, each clique in its own color: three "
        "triangles, three complete graphs on four nodes, and four of them. In each clique one "
        "internal edge is moved to point at the next clique, so the cliques form a ring joined "
        "by single plain edges and every node keeps its original degree in total.")
def add_connected_caveman_graph(s: Scene) -> None:
    params = ((3, 3), (3, 4), (4, 4))
    boxes = multiples(s, tuple(f"num_cliques={c}, clique_size={k}" for c, k in params))
    for box, (c, k) in zip(boxes, params):
        _community_panel(s, box, f"add_connected_caveman_graph({c}, {k})", f"c{c}{k}.")


@figure("add_ring_of_cliques", "generators", "Ring of cliques",
        "Three rings of cliques side by side, each clique in its own color: three triangles, "
        "four triangles, and four complete graphs on four nodes. Every clique stays complete "
        "and is joined to the next one around the ring by a single plain edge.")
def add_ring_of_cliques(s: Scene) -> None:
    params = ((3, 3), (4, 3), (4, 4))
    boxes = multiples(s, tuple(f"num_cliques={c}, clique_size={k}" for c, k in params))
    for box, (c, k) in zip(boxes, params):
        _community_panel(s, box, f"add_ring_of_cliques({c}, {k})", f"r{c}{k}.")


RELAXED_SEED = 6


@figure("add_relaxed_caveman_graph", "generators", "Relaxed caveman graph",
        "Three relaxed caveman graphs built from three complete graphs on four nodes, each in "
        "its own color, at rewiring probability p=0, 0.2, and 0.5. At p=0 the cliques are "
        "separate; as p grows, more edges are rewired to random nodes and show up as plain "
        "edges between the cliques (3 at p=0.2 and 6 at p=0.5), while the total stays 18.")
def add_relaxed_caveman_graph(s: Scene) -> None:
    ps = ("0", "0.2", "0.5")
    boxes = multiples(s, tuple(f"clique_size=4, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        call = f"add_relaxed_caveman_graph(3, 4, {p}, seed={RELAXED_SEED})"
        _community_panel(s, box, call, f"p{i}.", gap=1.0)


PLANTED_SEED = 14


@figure("add_planted_partition_graph", "generators", "Planted partition graph",
        "Three planted partition graphs with three groups of four nodes, each group in its own "
        "color, edge probability 0.9 inside a group and 0, 0.1, and 0.2 between groups. The "
        "groups stay dense in every panel, while the plain edges between them grow from none "
        "to 3 to 9 and blur the split.")
def add_planted_partition_graph(s: Scene) -> None:
    p_outs = ("0", "0.1", "0.2")
    boxes = multiples(s, tuple(f"p_in=0.9, p_out={p}" for p in p_outs))
    for i, (box, p) in enumerate(zip(boxes, p_outs)):
        call = f"add_planted_partition_graph(3, 4, 0.9, {p}, seed={PLANTED_SEED})"
        _community_panel(s, box, call, f"p{i}.", gap=1.0)


PARTITION_SEED = 1


@figure("add_random_partition_graph", "generators", "Random partition graph",
        "Three random partition graphs with groups of the sizes given, each group in its own "
        "color: two groups of four, groups of two, three, and four, and groups of two, two, "
        "three, and four. Edges appear with probability 0.9 inside a group and 0.1 between "
        "groups, so each group is dense and the few plain edges link them.")
def add_random_partition_graph(s: Scene) -> None:
    params = ((4, 4), (2, 3, 4), (2, 2, 3, 4))
    boxes = multiples(s, tuple(f"sizes=[{', '.join(map(str, p))}]" for p in params))
    for i, (box, sizes) in enumerate(zip(boxes, params)):
        call = f"add_random_partition_graph({list(sizes)}, 0.9, 0.1, seed={PARTITION_SEED})"
        _community_panel(s, box, call, f"s{i}.", gap=1.0)


GAUSSIAN_SEED = 13
GAUSSIAN_GAP = 0.6


@figure("add_gaussian_random_partition_graph", "generators", "Gaussian random partition graph",
        "Three Gaussian random partition graphs on twelve nodes with mean group size 3, 4, and "
        "5, each group in its own color. Group sizes are drawn at random around the mean, here "
        "4, 3, 3, and 2, then 5, 4, and 3, then 6, 5, and 1, so a larger mean gives fewer, "
        "bigger groups; edges are dense inside a group and sparse between groups.")
def add_gaussian_random_partition_graph(s: Scene) -> None:
    means = (3, 4, 5)
    boxes = multiples(s, tuple(f"n=12, mean_size={m}" for m in means))
    for box, m in zip(boxes, means):
        call = f"add_gaussian_random_partition_graph(12, {m}, 2, 0.8, 0.1, seed={GAUSSIAN_SEED})"
        _community_panel(s, box, call, f"m{m}.", gap=GAUSSIAN_GAP)


LFR_SEED = 4
LFR_GAP = 0.6


@figure("add_lfr_graph", "generators", "LFR benchmark graph",
        "Three LFR benchmark graphs on fifteen nodes in three communities of five, each "
        "community in its own color, at mixing parameter mu=0.1, 0.25, and 0.4. The mixing "
        "parameter is the share of each node's edges that leave its community, so the plain "
        "edges between communities grow from 2 to 5 to 9 and the communities get harder to "
        "tell apart.")
def add_lfr_graph(s: Scene) -> None:
    mus = ("0.1", "0.25", "0.4")
    boxes = multiples(s, tuple(f"n=15, mu={mu}, average_degree=3" for mu in mus))
    for i, (box, mu) in enumerate(zip(boxes, mus)):
        call = (f"add_lfr_graph(15, {mu}, 3, max_degree=4, min_community=5, max_community=5, "
                f"seed={LFR_SEED})")
        _community_panel(s, box, call, f"mu{i}.", gap=LFR_GAP)


# --- random models, degree sequences, directed growth, geometry ---------------
# Drawn live like the shapes above, from a fixed seed per figure (in SEEDS, left
# out of the captions as plumbing; each was picked for a lint-clean drawing).
# Most sit on the shared 8-node ring in id order. `_live` handles what these
# models can produce that the earlier figures never needed: arrows, parallel
# edges (bowed apart), self-loops, and edges that would graze a node off the
# ring (bowed clear with `_clear_bend`).

SEEDS = {
    "add_random_regular_graph": 0,
    "add_powerlaw_cluster_graph": 11,
    "add_newman_watts_strogatz_graph": 0,
    "add_connected_watts_strogatz_graph": 0,
    "add_extended_barabasi_albert_graph": 1,
    "add_dual_barabasi_albert_graph": 0,
    "add_random_lobster": 60,
    "add_random_labeled_tree": 0,
    "add_random_powerlaw_tree": 95,
    "add_directed_configuration_model": 52,
    "add_random_degree_sequence_graph": 0,
    "add_joint_degree_graph": 0,
    "add_gn_graph": 1,
    "add_gnr_graph": 1,
    "add_gnc_graph": 0,
    "add_scale_free_graph": 1,
    "add_random_k_out_graph": 18,
    "add_waxman_graph": 41,
    "add_soft_random_geometric_graph": 289,
    "add_geographical_threshold_graph": 139,
    "add_navigable_small_world_graph": 5,
    "add_bipartite_configuration_model": 0,
    "add_bipartite_preferential_attachment_graph": 0,
    "add_bipartite_gnp_graph": 0,
    "add_bipartite_gnm_graph": 0,
    "add_maybe_regular_expander": 1,
    "add_random_regular_expander_graph": 0,
    "add_random_kernel_graph": 0,
    "add_random_clustered_graph": 0,
    "add_duplication_divergence_graph": 23,
    "add_partial_duplication_graph": 0,
    "add_random_internet_as_graph": 9,
}


def _seeded(method: str, args: str, directed: bool = False):
    """`_built` for `method(args)` with the figure's seed."""
    return _built(f"{method}({args}, seed={SEEDS[method]})", directed)


def _live(s: Scene, box: Box, coords, edges, *, prefix: str, directed: bool = False,
          styles=None, edge_style=None, labels=False, unit: float = UNIT, pad: float = 6,
          pos=None) -> dict:
    """Like `_draw`, for generator output that may hold arrows, repeats, or self-loops.

    Copies of the same pair bow apart (two copies to either side, a third down
    the middle); any other edge that would graze a node is bowed clear of it.
    `pos` skips the placement, for callers that place extra furniture with it.
    """
    pos = pos or place(coords, box, unit=unit, r=R_SMALL, pad=pad)
    edge_style = edge_style or {}
    count = Counter(frozenset((a, b)) for a, b in edges)
    seen: Counter = Counter()
    es = []
    for a, b in edges:
        key = frozenset((a, b))
        opts = {}
        if st := edge_style.get(key):
            opts["style"] = st
        if a != b and count[key] > 1:
            # bends are set for the low -> high direction and flipped for the reverse
            lo, hi = sorted((a, b), key=int)
            pair = _pair_bends(pos, lo, hi) if count[key] == 2 else (0, 14, -14)
            bend = pair[seen[key]]
            seen[key] += 1
            opts["bend"] = bend if int(a) < int(b) else -bend
        elif a != b and (bend := _clear_bend(pos, a, b)):
            opts["bend"] = bend
        es.append((a, b, opts))
    s.graph(pos, es, labels=labels, prefix=prefix, r=R_SMALL, styles=styles, directed=directed)
    return pos


def _bfs_children(edges, root: str) -> dict[str, list[str]]:
    """Children of each node in a tree hung from `root`, smaller ids first."""
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    kids: dict[str, list[str]] = {}
    seen, queue = {root}, [root]
    for v in queue:
        for w in sorted(adj.get(v, []), key=int):
            if w not in seen:
                seen.add(w)
                kids.setdefault(v, []).append(w)
                queue.append(w)
    return kids


def _tree_center(edges) -> str:
    """The node whose farthest node is nearest (smallest id on ties), for a low tree."""
    nodes = sorted({v for e in edges for v in e}, key=int)

    def height(kids, v):
        return 1 + max((height(kids, c) for c in kids.get(v, [])), default=0)

    return min(nodes, key=lambda v: height(_bfs_children(edges, v), v))


def _tree_panel(s: Scene, box: Box, edges, prefix: str, *, root: str | None = None,
                directed: bool = False, styles=None) -> None:
    root = root or _tree_center(edges)
    coords = _tidy_tree(_bfs_children(edges, root), root)
    _live(s, box, coords, edges, prefix=prefix, directed=directed, styles=styles, pad=2)


def _new_edges(edges, base) -> dict:
    """`accent` style for each edge whose pair is not among the `base` pairs."""
    base = {frozenset(e) for e in base}
    return {frozenset(e): "accent" for e in edges if frozenset(e) not in base}


def _ring_lattice(n: int, k: int) -> list[tuple[str, str]]:
    """Each node joined to its k // 2 clockwise neighbors: the model's p=0 starting ring."""
    return [(str(i), str((i + j - 1) % n + 1))
            for i in range(1, n + 1) for j in range(1, k // 2 + 1)]


def _seq(values, tight: bool = False) -> str:
    """A list as it reads in a caption; `tight` drops the spaces for long ones."""
    return f"[{(',' if tight else ', ').join(map(str, values))}]"


@figure("add_random_regular_graph", "generators", "Random regular graph",
        "Three random regular graphs on the same ring of eight nodes for degree d=2, 3, and 4. "
        "Every node has exactly d neighbors, so the graphs have 8, 12, and 16 edges; which pairs "
        "get joined is random, and the d=2 graph here splits into a triangle and a five-cycle "
        "rather than one long cycle.")
def add_random_regular_graph(s: Scene) -> None:
    ds = (2, 3, 4)
    boxes = multiples(s, tuple(f"n=8, d={d}" for d in ds))
    for box, d in zip(boxes, ds):
        _, edges = _seeded("add_random_regular_graph", f"8, {d}")
        _live(s, box, _ring_xy(8), edges, prefix=f"d{d}.")


@figure("add_powerlaw_cluster_graph", "generators", "Powerlaw cluster graph",
        "Three Holme-Kim graphs on a ring of eight nodes, each new node attaching to m=2 others "
        "by preferential attachment, with triad-formation probability p=0, 0.5, and 1. Every "
        "panel has 12 edges, but a higher p closes more of them into triangles: 1, 4, and 5 "
        "triangles.")
def add_powerlaw_cluster_graph(s: Scene) -> None:
    ps = ("0", "0.5", "1")
    boxes = multiples(s, tuple(f"n=8, m=2, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_powerlaw_cluster_graph", f"8, 2, {p}")
        _live(s, box, _ring_xy(8), edges, prefix=f"p{i}.")


@figure("add_newman_watts_strogatz_graph", "generators", "Newman-Watts-Strogatz graph",
        "Three Newman-Watts-Strogatz graphs on a ring of eight nodes with k=2 and shortcut "
        "probability p=0, 0.3, and 0.6. The ring is never broken: accent-colored shortcuts are "
        "added on top of it, 1 at p=0.3 and 4 at p=0.6, instead of replacing ring edges as in the "
        "Watts-Strogatz model.")
def add_newman_watts_strogatz_graph(s: Scene) -> None:
    ps = ("0", "0.3", "0.6")
    boxes = multiples(s, tuple(f"n=8, k=2, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_newman_watts_strogatz_graph", f"8, 2, {p}")
        _live(s, box, _ring_xy(8), edges, prefix=f"p{i}.",
              edge_style=_new_edges(edges, _ring_lattice(8, 2)))


@figure("add_connected_watts_strogatz_graph", "generators", "Connected Watts-Strogatz graph",
        "Three connected Watts-Strogatz graphs on a ring of eight nodes with k=4 and rewiring "
        "probability p=0, 0.2, and 1. At p=0 every node links to its two nearest neighbors on "
        "each side; rewired edges show as accent-colored shortcuts, 3 at p=0.2 and 9 at p=1. "
        "Every panel keeps all 16 edges and is connected, because a disconnected draw is retried.")
def add_connected_watts_strogatz_graph(s: Scene) -> None:
    ps = ("0", "0.2", "1")
    boxes = multiples(s, tuple(f"n=8, k=4, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_connected_watts_strogatz_graph", f"8, 4, {p}")
        _live(s, box, _ring_xy(8), edges, prefix=f"p{i}.",
              edge_style=_new_edges(edges, _ring_lattice(8, 4)))


@figure("add_extended_barabasi_albert_graph", "generators", "Extended Barabási-Albert graph",
        "Three extended Barabási-Albert graphs on a ring of eight nodes with m=1. With p=0 and "
        "q=0 it is plain preferential attachment, a tree of 7 edges; with p=0.4 some steps add "
        "edges between existing nodes instead, giving 12 edges and 7 triangles; with q=0.4 some "
        "steps rewire an existing edge instead, so the count stays at 7 and here one node is left "
        "with no edges.")
def add_extended_barabasi_albert_graph(s: Scene) -> None:
    params = (("0", "0"), ("0.4", "0"), ("0", "0.4"))
    boxes = multiples(s, tuple(f"n=8, m=1, p={p}, q={q}" for p, q in params))
    for i, (box, (p, q)) in enumerate(zip(boxes, params)):
        _, edges = _seeded("add_extended_barabasi_albert_graph", f"8, 1, {p}, {q}")
        _live(s, box, _ring_xy(8), edges, prefix=f"e{i}.")


@figure("add_dual_barabasi_albert_graph", "generators", "Dual Barabási-Albert graph",
        "Three dual Barabási-Albert graphs on a ring of eight nodes, where each new node attaches "
        "to m1=1 existing node with probability p and to m2=3 otherwise, for p=1, 0.5, and 0. At "
        "p=1 every new node brings one edge and the graph is a tree of 7 edges; the mixed p=0.5 "
        "gives 13 edges; at p=0 every new node brings three, for 15.")
def add_dual_barabasi_albert_graph(s: Scene) -> None:
    ps = ("1", "0.5", "0")
    boxes = multiples(s, tuple(f"n=8, m1=1, m2=3, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_dual_barabasi_albert_graph", f"8, 1, 3, {p}")
        _live(s, box, _ring_xy(8), edges, prefix=f"p{i}.")


def _lobster_coords(edges) -> dict:
    """Backbone (ids 1..L, a path) in a row; legs below it, leaves below their legs."""
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    backbone = ["1"]
    while str(len(backbone) + 1) in adj.get(backbone[-1], ()):
        backbone.append(str(len(backbone) + 1))
    # a stand-in root over the backbone lets `_tidy_tree` space the legs
    kids = {"root": backbone}
    for b in backbone:
        kids[b] = sorted((v for v in adj[b] if int(v) > len(backbone)), key=int)
        for leg in kids[b]:
            kids[leg] = sorted((v for v in adj[leg] if int(v) > int(leg)), key=int)
    coords = _tidy_tree(kids, "root")
    del coords["root"]
    return coords


@figure("add_random_lobster", "generators", "Random lobster",
        "Three random lobsters grown from the same backbone, a path of four nodes drawn as the "
        "top row. With p1=0 and p2=0 nothing grows and the lobster is just the path; with p1=0.6 "
        "five legs hang below the backbone, making a caterpillar; with p2=0.6 as well, two of the "
        "legs grow three leaves between them, so every node stays within two steps of the "
        "backbone.")
def add_random_lobster(s: Scene) -> None:
    params = (("0", "0"), ("0.6", "0"), ("0.6", "0.6"))
    boxes = multiples(s, tuple(f"n=4, p1={p1}, p2={p2}" for p1, p2 in params))
    for i, (box, (p1, p2)) in enumerate(zip(boxes, params)):
        _, edges = _seeded("add_random_lobster", f"4, {p1}, {p2}")
        coords = _lobster_coords(edges) if edges else {"1": (0.0, 0.0)}
        _live(s, box, coords, edges, prefix=f"l{i}.", pad=2)


@figure("add_random_labeled_tree", "generators", "Random labeled tree",
        "Three random labeled trees on 5, 7, and 9 nodes, each hung from its most central node. "
        "Every tree on n labeled nodes is equally likely, so the shapes vary freely, but each one "
        "has n - 1 edges and no cycles.")
def add_random_labeled_tree(s: Scene) -> None:
    ns = (5, 7, 9)
    boxes = multiples(s, tuple(f"g.add_random_labeled_tree(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _, edges = _seeded("add_random_labeled_tree", str(n))
        _tree_panel(s, box, edges, f"n{n}.")


@figure("add_random_powerlaw_tree", "generators", "Random power-law tree",
        "Three random power-law trees on eight nodes for exponent gamma=2, 3, and 5, each hung "
        "from its most central node. The degrees are drawn from a power law, so a small gamma "
        "gives a heavy tail: the biggest hub has 6 neighbors at gamma=2, 5 at gamma=3, and 3 at "
        "gamma=5.")
def add_random_powerlaw_tree(s: Scene) -> None:
    gammas = ("2", "3", "5")
    boxes = multiples(s, tuple(f"n=8, gamma={gm}" for gm in gammas))
    for i, (box, gm) in enumerate(zip(boxes, gammas)):
        _, edges = _seeded("add_random_powerlaw_tree", f"8, {gm}")
        _tree_panel(s, box, edges, f"g{i}.")


def _badged_ring(s: Scene, box: Box, edges, values: list[str], prefix: str, *,
                 directed: bool = False) -> None:
    """Nodes on the smaller badge ring, each with `values[i]` in a pill."""
    pos = _live(s, box, _ring_xy(len(values), BADGE_RING), edges, prefix=prefix,
                directed=directed)
    _ring_badges(s, pos, prefix, {str(k): v for k, v in enumerate(values, 1)})


@figure("add_havel_hakimi_graph", "generators", "Havel-Hakimi graph",
        "Three Havel-Hakimi graphs on five nodes, each node's requested degree in a pill: 3, 2, "
        "2, 2, 1; 3, 3, 2, 2, 2; and 4, 3, 3, 2, 2. The construction is deterministic, joining "
        "the node with the most degree left to the next-highest nodes, and every node ends with "
        "exactly the degree in its pill, using 5, 6, and 7 edges.")
def add_havel_hakimi_graph(s: Scene) -> None:
    seqs = ((3, 2, 2, 2, 1), (3, 3, 2, 2, 2), (4, 3, 3, 2, 2))
    boxes = multiples(s, tuple(f"degree_sequence={_seq(d)}" for d in seqs))
    for i, (box, seq) in enumerate(zip(boxes, seqs)):
        _, edges = _built(f"add_havel_hakimi_graph({list(seq)})")
        _badged_ring(s, box, edges, [str(d) for d in seq], f"h{i}.")


@figure("add_random_degree_sequence_graph", "generators", "Random degree sequence graph",
        "Three random graphs on a ring of six nodes built to match a degree sequence, each node's "
        "degree in a pill: all ones gives three separate edges, all twos gives a six-cycle, and "
        "3, 3, 2, 2, 1, 1 gives six edges with one triangle. Unlike the configuration model, the "
        "result is always simple, with no doubled edges or self-loops.")
def add_random_degree_sequence_graph(s: Scene) -> None:
    seqs = ((1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2), (3, 3, 2, 2, 1, 1))
    boxes = multiples(s, tuple(f"degree_sequence={_seq(d, tight=True)}" for d in seqs))
    for i, (box, seq) in enumerate(zip(boxes, seqs)):
        _, edges = _seeded("add_random_degree_sequence_graph", f"{list(seq)}, tries=100")
        _badged_ring(s, box, edges, [str(d) for d in seq], f"r{i}.")


DIRECTED_SEQS = (  # (in-degrees, out-degrees)
    ((1, 1, 1, 1), (1, 1, 1, 1)),
    ((0, 1, 1, 2), (2, 1, 1, 0)),
    ((2, 1, 1, 1, 1), (0, 1, 2, 1, 2)),
)


def _io_caption(ins, outs) -> str:
    return f"in={_seq(ins, tight=True)}, out={_seq(outs, tight=True)}"


@figure("add_directed_configuration_model", "generators", "Directed configuration model",
        "Three directed configuration models, each node's in-degree and out-degree in a pill "
        "written in/out. Four nodes at 1/1 form a directed four-cycle; 0/2, 1/1, 1/1, and 2/0 "
        "send every arc from the source toward the sink; on five nodes two pairs of opposite arcs "
        "appear, drawn bowed apart. Out-stubs are paired with in-stubs at random, so every node "
        "meets its pill exactly; self-loops and repeated arcs are possible, though this draw has "
        "none.")
def add_directed_configuration_model(s: Scene) -> None:
    boxes = multiples(s, tuple(_io_caption(i, o) for i, o in DIRECTED_SEQS))
    for i, (box, (ins, outs)) in enumerate(zip(boxes, DIRECTED_SEQS)):
        _, edges = _seeded("add_directed_configuration_model", f"{list(ins)}, {list(outs)}",
                           directed=True)
        _badged_ring(s, box, edges, [f"{a}/{b}" for a, b in zip(ins, outs)], f"c{i}.",
                     directed=True)


@figure("add_directed_havel_hakimi_graph", "generators", "Directed Havel-Hakimi graph",
        "Three directed Havel-Hakimi graphs for the same degrees as the directed configuration "
        "model figure, each node's in-degree and out-degree in a pill written in/out. The "
        "construction is deterministic and never repeats an arc: four nodes at 1/1 pair up into "
        "two 2-cycles, drawn as bowed opposite arcs; 0/2, 1/1, 1/1, and 2/0 give two two-step "
        "paths from the source to the sink; and the five-node sequence uses one pair of opposite "
        "arcs.")
def add_directed_havel_hakimi_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(_io_caption(i, o) for i, o in DIRECTED_SEQS))
    for i, (box, (ins, outs)) in enumerate(zip(boxes, DIRECTED_SEQS)):
        _, edges = _built(f"add_directed_havel_hakimi_graph({list(ins)}, {list(outs)})",
                          directed=True)
        _badged_ring(s, box, edges, [f"{a}/{b}" for a, b in zip(ins, outs)], f"h{i}.",
                     directed=True)


JOINT = (  # a path of four, a diamond, and two joined stars; diagonals count edges twice
    {1: {2: 2}, 2: {1: 2, 2: 2}},
    {2: {3: 4}, 3: {2: 4, 3: 2}},
    {1: {3: 4}, 3: {1: 4, 3: 2}},
)


@figure("add_joint_degree_graph", "generators", "Joint degree graph",
        "Three joint degree graphs, each node's degree in a pill, with captions giving how many "
        "edges join each pair of degrees, where pairs of equal degree count twice. Two edges from "
        "degree 1 to degree 2 plus one between the degree-2 nodes give a path of four; four edges "
        "from degree 2 to degree 3 plus one between the degree-3 nodes give a diamond; and two "
        "joined degree-3 hubs with two leaves each give a double star.")
def add_joint_degree_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(str(j).replace(" ", "") for j in JOINT))
    for i, (box, joint) in enumerate(zip(boxes, JOINT)):
        _, edges = _seeded("add_joint_degree_graph", str(joint))
        degree = Counter(v for e in edges for v in e)
        ids = sorted(degree, key=int)
        _badged_ring(s, box, edges, [str(degree[v]) for v in ids], f"j{i}.")


GN_KERNELS = ("lambda k: 1", "None", "lambda k: k**2")


@figure("add_gn_graph", "generators", "Growing network",
        "Three growing networks on eight nodes, drawn as trees with the first node at the top and "
        "every arc pointing from a newer node to the older node it attached to. With a constant "
        "kernel the target is uniform and the first node gets 3 arcs; with the default linear "
        "kernel it gets 5; with the kernel k**2 the rich get richer so fast that all 7 later "
        "nodes point at it.")
def add_gn_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"n=8, kernel={k}" for k in GN_KERNELS))
    for i, (box, kernel) in enumerate(zip(boxes, GN_KERNELS)):
        _, edges = _seeded("add_gn_graph", f"8, kernel={kernel}", directed=True)
        _tree_panel(s, box, edges, f"k{i}.", root="1", directed=True)


@figure("add_gnr_graph", "generators", "Growing network with redirection",
        "Three growing networks with redirection on eight nodes, drawn as trees with the first "
        "node at the top and every arc pointing from a newer node to an older one. Each new node "
        "picks a random target and, with probability p, links to the node that target points at "
        "instead. At p=0 the tree is deep and stringy, at p=0.5 the first node collects 5 arcs, "
        "and at p=1 every arc is redirected to it, giving a star.")
def add_gnr_graph(s: Scene) -> None:
    ps = ("0", "0.5", "1")
    boxes = multiples(s, tuple(f"g.add_gnr_graph(n=8, p={p})" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_gnr_graph", f"8, {p}", directed=True)
        _tree_panel(s, box, edges, f"p{i}.", root="1", directed=True)


@figure("add_gnc_graph", "generators", "Growing network with copying",
        "Three growing networks with copying on 4, 5, and 6 nodes, placed on a ring in arrival "
        "order. Each new node links to a random existing node and also copies all of that node's "
        "arcs, so arcs pile onto the earliest nodes: the first node receives 3, 4, and 5 arcs, "
        "and the graphs have 4, 7, and 11 arcs in all.")
def add_gnc_graph(s: Scene) -> None:
    ns = (4, 5, 6)
    boxes = multiples(s, tuple(f"g.add_gnc_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _, edges = _seeded("add_gnc_graph", str(n), directed=True)
        _live(s, box, _ring_xy(n), edges, prefix=f"n{n}.", directed=True)


@figure("add_scale_free_graph", "generators", "Scale-free directed graph",
        "Three scale-free directed graphs on 4, 6, and 8 nodes on a ring, grown from a three- "
        "cycle by the default mix of steps that add a node with an arc out, add an arc between "
        "existing nodes, or add a node with an arc in. One node quickly becomes the hub, touching "
        "7 of the 10 arcs at n=8; it and one neighbor have arcs both ways, plus a repeated arc at "
        "n=8, drawn bowed apart, because the model keeps parallel arcs.")
def add_scale_free_graph(s: Scene) -> None:
    ns = (4, 6, 8)
    boxes = multiples(s, tuple(f"g.add_scale_free_graph(n={n})" for n in ns))
    for box, n in zip(boxes, ns):
        _, edges = _seeded("add_scale_free_graph", str(n), directed=True)
        _live(s, box, _ring_xy(n), edges, prefix=f"n{n}.", directed=True)


@figure("add_random_k_out_graph", "generators", "Random k-out graph",
        "Three random k-out graphs on a ring of eight nodes in which every node sends k=2 arcs, "
        "for alpha=10, 1, and 0.2. Each arc picks its target with weight alpha plus the target's "
        "current in-degree, so a smaller alpha concentrates arcs on a few popular nodes: the most "
        "popular node receives 5, 6, and 7 arcs. Repeated arcs and opposite arcs are drawn bowed "
        "apart; self-loops are turned off.")
def add_random_k_out_graph(s: Scene) -> None:
    alphas = ("10", "1", "0.2")
    boxes = multiples(s, tuple(f"n=8, k=2, alpha={a}" for a in alphas))
    for i, (box, a) in enumerate(zip(boxes, alphas)):
        _, edges = _seeded("add_random_k_out_graph", f"8, 2, {a}, self_loops=False",
                           directed=True)
        _live(s, box, _ring_xy(8), edges, prefix=f"a{i}.", directed=True)


def _positions(g, attr: str = "pos") -> dict[str, tuple[float, float]]:
    return {str(k): (x, y) for k, (x, y) in g.get_node_attrs(attr).items()}


def _radius_circle(s: Scene, pos: dict, coords: dict, hub: str, radius: float) -> None:
    """Faint dashed circle of abstract `radius` around `hub`, at the placed scale."""
    a, b = sorted(set(coords) - {hub})[:2]
    scale = math.dist(pos[a], pos[b]) / math.dist(coords[a], coords[b])
    rad = radius * scale
    cx, cy = pos[hub]
    s.path(f"M {cx - rad:.1f} {cy:.1f} a {rad:.1f} {rad:.1f} 0 1 0 {2 * rad:.1f} 0 "
           f"a {rad:.1f} {rad:.1f} 0 1 0 {-2 * rad:.1f} 0", color="accent", opacity=0.45,
           width=1, dash="4 4")


def _central(coords: dict) -> str:
    """The node nearest the middle of the drawing."""
    xs, ys = [x for x, _ in coords.values()], [y for _, y in coords.values()]
    mid = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
    return min(coords, key=lambda k: math.dist(coords[k], mid))


GEO_UNIT = 150  # px per unit of the unit square, as in the random geometric graph


@figure("add_waxman_graph", "generators", "Waxman graph",
        "Three Waxman graphs on the same seven random points in the unit square, with beta=1 and "
        "alpha=0.1, 0.25, and 1. Two points are joined with a probability that decays with their "
        "distance, and alpha sets how slowly: at 0.1 only the closest pair is joined, at 0.25 "
        "there are 3 edges, and at 1 longer links appear for 8 edges in all. A few edges bow "
        "slightly to clear other nodes.")
def add_waxman_graph(s: Scene) -> None:
    alphas = ("0.1", "0.25", "1")
    boxes = multiples(s, tuple(f"n=7, beta=1, alpha={a}" for a in alphas))
    for i, (box, a) in enumerate(zip(boxes, alphas)):
        g, edges = _seeded("add_waxman_graph", f"7, beta=1, alpha={a}")
        _live(s, box, _positions(g), edges, prefix=f"a{i}.", unit=GEO_UNIT)


@figure("add_soft_random_geometric_graph", "generators", "Soft random geometric graph",
        "The same seven random points in the unit square at radius=0.25, 0.35, and 0.45, with a "
        "faint dashed circle of that radius around the central node, drawn in the accent outline. "
        "Unlike the random geometric graph, a pair within the radius is only joined with a "
        "probability that falls with distance, so some nodes inside the circle stay unjoined to "
        "the central one; the graphs have 2, 7, and 8 edges.")
def add_soft_random_geometric_graph(s: Scene) -> None:
    radii = ("0.25", "0.35", "0.45")
    boxes = multiples(s, tuple(f"n=7, radius={r}" for r in radii))
    for i, (box, r) in enumerate(zip(boxes, radii)):
        g, edges = _seeded("add_soft_random_geometric_graph", f"7, {r}")
        coords = _positions(g)
        hub = _central(coords)
        pos = place(coords, box, unit=GEO_UNIT, r=R_SMALL)
        _radius_circle(s, pos, coords, hub, float(r))
        _live(s, box, coords, edges, prefix=f"r{i}.", unit=GEO_UNIT, styles={hub: "soft"})


@figure("add_geographical_threshold_graph", "generators", "Geographical threshold graph",
        "Three geographical threshold graphs on the same seven random points, each node shaded by "
        "its random weight, darker for heavier, at threshold theta=80, 30, and 10. Two nodes are "
        "joined when their combined weight divided by their squared distance reaches theta, so "
        "heavy nodes reach far: at theta=80 only the heaviest node has edges, 2 of them, and at "
        "theta=10 it has 6 of the 13.")
def add_geographical_threshold_graph(s: Scene) -> None:
    thetas = ("80", "30", "10")
    boxes = multiples(s, tuple(f"n=7, theta={t}" for t in thetas))
    for i, (box, t) in enumerate(zip(boxes, thetas)):
        g, edges = _seeded("add_geographical_threshold_graph", f"7, {t}")
        weights = {str(k): w for k, w in g.get_node_attrs("weight").items()}
        top = max(weights.values())
        _live(s, box, _positions(g), edges, prefix=f"t{i}.", unit=GEO_UNIT)
        for k, w in weights.items():
            s.nodes[f"t{i}.{k}"].heat = w / top


@figure("add_navigable_small_world_graph", "generators", "Navigable small-world graph",
        "Three navigable small-world graphs on a 3 by 3 grid with p=1 and r=2, for q=0, 1, and 2 "
        "long-range contacts per node. At q=0 it is just the grid; each node then adds q contacts "
        "chosen with probability falling off as the squared grid distance, so most land on "
        "neighbors already joined, and the accent edges are the ones that reach farther: 1 at q=1 "
        "and 5 at q=2. Some bow to clear the nodes they would pass through.")
def add_navigable_small_world_graph(s: Scene) -> None:
    qs = (0, 1, 2)
    boxes = multiples(s, tuple(f"n=3, p=1, q={q}, r=2" for q in qs))
    for box, q in zip(boxes, qs):
        g, edges = _seeded("add_navigable_small_world_graph", f"3, 1, {q}")
        coords = _positions(g)
        long = {frozenset((a, b)): "accent" for a, b in edges
                if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) > 1}
        _live(s, box, coords, edges, prefix=f"q{q}.", edge_style=long)


# --- bipartite generators -------------------------------------------------------
# Part 0 in a top row in `accent`, part 1 in a teal bottom row, as in the
# complete bipartite figure. Degree-driven models put each node's degree in a
# pill outside its row (above the top row, below the bottom one).


def _rows(g) -> dict[str, tuple[float, float]]:
    """Abstract coords: each part centered in its own row, ids in order."""
    parts = g.get_node_attrs("part")
    coords = {}
    for part in (0, 1):
        ids = sorted(k for k, v in parts.items() if v == part)
        coords |= {str(k): (i - (len(ids) - 1) / 2, 1.5 * part) for i, k in enumerate(ids)}
    return coords


def _bipartite_panel(s: Scene, box: Box, g, edges, prefix: str, *,
                     degrees: bool = False) -> None:
    coords = _rows(g)
    if degrees:  # leave room for the pills above and below the rows
        box = Box(box.x0, box.y0 + 14, box.x1, box.y1 - 14)
    _live(s, box, coords, edges, prefix=prefix, styles=_group_styles(g, "part"),
          directed=g.is_directed())
    if degrees:
        degree = Counter(v for e in edges for v in e)
        parts = g.get_node_attrs("part")
        for k in coords:
            s.badge(prefix + k, str(degree[k]), where="n" if parts[int(k)] == 0 else "s")


BIPARTITE_SEQS = (((1, 1, 1), (1, 1, 1)), ((2, 2, 2), (3, 3)), ((3, 2, 1), (2, 2, 1, 1)))


@figure("add_bipartite_configuration_model", "generators", "Bipartite configuration model",
        "Three bipartite configuration models, the first part in an accent top row and the "
        "second in a teal bottom row, each node's degree in a pill. Three ones on each side give "
        "a perfect matching; three twos against two threes give six edges in which the "
        "stubs happened to pair into two doubled edges, drawn bowed apart; and 3, 2, 1 against "
        "2, 2, 1, 1 gives six edges with one doubled. Every node meets its degree exactly, and "
        "no edge joins two nodes in the same row.")
def add_bipartite_configuration_model(s: Scene) -> None:
    boxes = multiples(s, tuple(f"aseq={_seq(a, True)}, bseq={_seq(b, True)}"
                               for a, b in BIPARTITE_SEQS))
    for i, (box, (a, b)) in enumerate(zip(boxes, BIPARTITE_SEQS)):
        g, edges = _seeded("add_bipartite_configuration_model", f"{list(a)}, {list(b)}")
        _bipartite_panel(s, box, g, edges, f"c{i}.", degrees=True)


HH_VARIANTS = ("largest", "smallest", "alternating")


@figure("add_bipartite_havel_hakimi_graph", "generators", "Bipartite Havel-Hakimi graph",
        "Three bipartite Havel-Hakimi graphs for the same degrees, 3, 2, 1 in the accent top row "
        "and 2, 2, 1, 1 in the teal bottom row, each node's degree in a pill. Each top node in "
        "turn is joined to the bottom nodes with the most degree left, the fewest, or "
        "alternately the most and the fewest, so the three variants wire the same six edges "
        "differently while every node still meets its pill and no pair is joined twice.")
def add_bipartite_havel_hakimi_graph(s: Scene) -> None:
    a, b = BIPARTITE_SEQS[2]
    boxes = multiples(s, tuple(f'variant="{v}"' for v in HH_VARIANTS))
    for i, (box, v) in enumerate(zip(boxes, HH_VARIANTS)):
        g, edges = _built(f"add_bipartite_havel_hakimi_graph({list(a)}, {list(b)}, "
                          f"variant={v!r})")
        _bipartite_panel(s, box, g, edges, f"h{i}.", degrees=True)


@figure("add_bipartite_preferential_attachment_graph", "generators",
        "Bipartite preferential attachment graph",
        "Three bipartite preferential attachment graphs in which three accent top nodes bring "
        "two edges each, and each edge makes a new teal bottom node with probability p or else "
        "picks an existing one in proportion to its degree. At p=1 every edge gets its own "
        "node, six in all; at p=0.5 the six edges share three bottom nodes, the busiest taking "
        "three; and at p=0 every "
        "edge lands on the single node the first edge made, so each top node reaches it twice, "
        "drawn as bowed pairs.")
def add_bipartite_preferential_attachment_graph(s: Scene) -> None:
    ps = ("1", "0.5", "0")
    boxes = multiples(s, tuple(f"aseq=[2, 2, 2], p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        g, edges = _seeded("add_bipartite_preferential_attachment_graph", f"[2, 2, 2], {p}")
        _bipartite_panel(s, box, g, edges, f"p{i}.")


@figure("add_bipartite_gnp_graph", "generators", "Bipartite G(n1, n2, p) random graph",
        "Three random bipartite graphs with three accent top nodes and four teal bottom nodes, "
        "where each of the 12 top-bottom pairs is joined independently with probability p=0.2, "
        "0.5, and 0.8. Edges only ever run between the rows, and their number grows with p, "
        "from 2 to 4 to 8.")
def add_bipartite_gnp_graph(s: Scene) -> None:
    ps = ("0.2", "0.5", "0.8")
    boxes = multiples(s, tuple(f"n1=3, n2=4, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        g, edges = _seeded("add_bipartite_gnp_graph", f"3, 4, {p}")
        _bipartite_panel(s, box, g, edges, f"p{i}.")


@figure("add_bipartite_gnm_graph", "generators", "Bipartite G(n1, n2, m) random graph",
        "Three random bipartite graphs with three accent top nodes and four teal bottom nodes "
        "and exactly m=3, 6, and 9 edges, each set of m top-bottom pairs equally likely. Edges "
        "only ever run between the rows.")
def add_bipartite_gnm_graph(s: Scene) -> None:
    ms = (3, 6, 9)
    boxes = multiples(s, tuple(f"n1=3, n2=4, m={m}" for m in ms))
    for box, m in zip(boxes, ms):
        g, edges = _seeded("add_bipartite_gnm_graph", f"3, 4, {m}")
        _bipartite_panel(s, box, g, edges, f"m{m}.")


# --- expanders ------------------------------------------------------------------
# On the shared ring in id order (the Margulis-Gabber-Galil graph on its torus
# grid). These are regular multigraphs or near-regular graphs, so self-loops
# and repeated pairs are part of the picture.


MGG_CELL = "(1,1)"  # the node whose eight edges the n=4 panel shows


@figure("add_margulis_gabber_galil_graph", "generators", "Margulis-Gabber-Galil graph",
        "Margulis-Gabber-Galil graphs on the n by n grid of coordinate pairs modulo n. Each point "
        "(x, y) is joined to eight images under fixed affine maps, so every node has degree 8 "
        "counting self-loops, drawn as small loops, and repeated pairs, drawn bowed apart: n=2 "
        "and n=3 on a ring, and for n=4 the 4 by 4 grid with only the eight edges of the point "
        "(1, 1) shown, in the accent color, reaching its images such as (1, 3) and (3, 1).")
def add_margulis_gabber_galil_graph(s: Scene) -> None:
    boxes = multiples(s, ("n=2", "n=3", f"n=4, edges of {MGG_CELL.replace(',', ', ')}"))
    for box, n in zip(boxes[:2], (2, 3)):
        _, edges = _built(f"add_margulis_gabber_galil_graph({n})")
        _live(s, box, _ring_xy(n * n), edges, prefix=f"n{n}.")
    g, edges = _built("add_margulis_gabber_galil_graph(4)")
    cells = {lab: k for k, lab in _labels(g).items()}
    coords = {k: eval(lab)[::-1] for lab, k in cells.items()}  # x across, y down
    hub = cells[MGG_CELL]
    mine = [e for e in edges if hub in e]
    nbrs = {v for e in mine for v in e} - {hub}
    _live(s, boxes[2], coords, mine, prefix="n4.", unit=46,
          styles={hub: "accent", **{v: "soft" for v in nbrs}},
          edge_style={frozenset(e): "accent" for e in mine})


@figure("add_chordal_cycle_graph", "generators", "Chordal cycle graph",
        "Three chordal cycle graphs on rings of p=5, 7, and 11 nodes, the integers modulo p. "
        "Each x is joined to x - 1 and x + 1 around the ring and by a chord to its inverse "
        "modulo p, shown in the accent color; 0, 1, and p - 1 are their own inverses, so they "
        "get self-loops. Where an inverse is a ring neighbor, as 2 and 3 at p=5 and 3 and 4 or 7 "
        "and 8 at p=11, the chord doubles a ring edge, drawn bowed apart.")
def add_chordal_cycle_graph(s: Scene) -> None:
    ps = (5, 7, 11)
    boxes = multiples(s, tuple(f"g.add_chordal_cycle_graph(p={p})" for p in ps))
    for box, p in zip(boxes, ps):
        _, edges = _built(f"add_chordal_cycle_graph({p})")
        ring_pairs = {frozenset(e) for e in _cycle_edges(p)}
        chords = {frozenset(e): "accent" for e in edges[p:] if frozenset(e) not in ring_pairs}
        _live(s, box, _ring_xy(p), edges, prefix=f"p{p}.", edge_style=chords)


@figure("add_paley_graph", "generators", "Paley graph",
        "Three Paley graphs on the integers modulo a prime p, on a ring. At p=5 two numbers are "
        "joined when their difference is a nonzero square, 1 or 4, which gives a five-cycle. At "
        "p=3 and p=7, where -1 is not a square, the graph is directed: each pair gets exactly "
        "one arc, from x to x plus a nonzero square, making a tournament, a directed triangle at "
        "p=3 and 21 arcs at p=7.")
def add_paley_graph(s: Scene) -> None:
    params = ((3, True), (5, False), (7, True))
    boxes = multiples(s, ("p=3, directed", "g.add_paley_graph(p=5)", "p=7, directed"))
    for box, (p, directed) in zip(boxes, params):
        _, edges = _built(f"add_paley_graph({p})", directed=directed)
        _live(s, box, _ring_xy(p), edges, prefix=f"p{p}.", directed=directed)


@figure("add_maybe_regular_expander", "generators", "Random regular expander candidate",
        "Three random graphs on a ring of eight nodes built as unions of d / 2 random "
        "Hamiltonian cycles that share no edge, for d=2, 4, and 6. Every node has exactly d "
        "neighbors; d=2 is a single cycle visiting the nodes in random order. The graph is "
        "only likely to be a good expander and is not checked, which is what the maybe means.")
def add_maybe_regular_expander(s: Scene) -> None:
    ds = (2, 4, 6)
    boxes = multiples(s, tuple(f"n=8, d={d}" for d in ds))
    for box, d in zip(boxes, ds):
        _, edges = _seeded("add_maybe_regular_expander", f"8, {d}")
        _live(s, box, _ring_xy(8), edges, prefix=f"d{d}.")


@figure("add_random_regular_expander_graph", "generators", "Random regular expander graph",
        "Three random regular expanders on a ring of nine nodes for d=2, 4, and 6. Candidates "
        "are built like the maybe-regular expander and kept only once their second eigenvalue "
        "meets the Ramanujan bound, so each graph is d-regular and a proven expander: a "
        "nine-cycle in shuffled order, then 18 and 27 edges.")
def add_random_regular_expander_graph(s: Scene) -> None:
    ds = (2, 4, 6)
    boxes = multiples(s, tuple(f"n=9, d={d}" for d in ds))
    for box, d in zip(boxes, ds):
        _, edges = _seeded("add_random_regular_expander_graph", f"9, {d}")
        _live(s, box, _ring_xy(9), edges, prefix=f"d{d}.")


# --- special generators -----------------------------------------------------------


@figure("add_random_kernel_graph", "generators", "Random kernel graph",
        "Three random kernel graphs on a ring of eight nodes with a constant kernel c, given "
        "through its integral c times (z - w), for c=1, 2, and 4. A constant kernel joins each "
        "pair with probability about c / n, so the graph fills in as c grows, from one edge to "
        "four to twelve.")
def add_random_kernel_graph(s: Scene) -> None:
    cs = (1, 2, 4)
    boxes = multiples(s, tuple(f"lambda u, w, z: {c} * (z - w)" for c in cs))
    for box, c in zip(boxes, cs):
        _, edges = _seeded("add_random_kernel_graph", f"8, lambda u, w, z: {c} * (z - w)")
        _live(s, box, _ring_xy(8), edges, prefix=f"c{c}.")


CLUSTERED = ((1, 0), (0, 1), (1, 1))


@figure("add_random_clustered_graph", "generators", "Random clustered graph",
        "Three random clustered graphs on a ring of six nodes, where each node asks for a number "
        "of plain edges and a number of triangles. One edge each pairs the nodes into three "
        "separate edges; one triangle each groups them into two triangles; and one of each gives "
        "two triangles plus three more edges, two of which happen to repeat a triangle side and "
        "are drawn bowed apart.")
def add_random_clustered_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"[{c}] * 6".replace(" ", "", 1) for c in CLUSTERED))
    for i, (box, c) in enumerate(zip(boxes, CLUSTERED)):
        _, edges = _seeded("add_random_clustered_graph", f"[{c}] * 6")
        _live(s, box, _ring_xy(6), edges, prefix=f"c{i}.")


@figure("add_duplication_divergence_graph", "generators", "Duplication-divergence graph",
        "Three duplication-divergence graphs on a ring of eight nodes, each grown from a single "
        "edge by copying a random node and keeping each of the copy's edges with probability p, "
        "for p=0.3, 0.6, and 0.9. A copy that keeps no edge is discarded, so the graph stays "
        "connected; a higher retention p gives denser graphs with more shared neighbors, 7, 11, "
        "and 15 edges here.")
def add_duplication_divergence_graph(s: Scene) -> None:
    ps = ("0.3", "0.6", "0.9")
    boxes = multiples(s, tuple(f"n=8, p={p}" for p in ps))
    for i, (box, p) in enumerate(zip(boxes, ps)):
        _, edges = _seeded("add_duplication_divergence_graph", f"8, {p}")
        _live(s, box, _ring_xy(8), edges, prefix=f"p{i}.")


@figure("add_partial_duplication_graph", "generators", "Partial duplication graph",
        "Three partial duplication graphs on a ring of eight nodes, grown from a triangle on the "
        "first three nodes. Each new node copies a random node, keeps each of its edges with "
        "probability p=0.5, and links to the original with probability q=0, 0.5, and 1. With "
        "q=0 a copy that keeps no edge is left isolated, as three are here; with q=1 every copy "
        "stays attached to its original.")
def add_partial_duplication_graph(s: Scene) -> None:
    qs = ("0", "0.5", "1")
    boxes = multiples(s, tuple(f"n=8, n_initial=3, p=0.5, q={q}" for q in qs))
    for i, (box, q) in enumerate(zip(boxes, qs)):
        _, edges = _seeded("add_partial_duplication_graph", f"8, 3, 0.5, {q}")
        _live(s, box, _ring_xy(8), edges, prefix=f"q{i}.")


INTERNET_TYPES = {"T": "accent", "M": "g2", "CP": "g3", "C": "g4"}


@figure("add_random_internet_as_graph", "generators", "Random Internet AS graph",
        "One random Internet autonomous-system graph on 20 nodes in three tiers. The five tier-1 "
        "providers, in the accent color, form a full clique at the top; the mid-level providers, "
        "in teal, and the content provider, in a third color, sit in the middle row; and the "
        "customers, in a fourth color, sit along the bottom. Customers buy transit from the "
        "providers above them, and edges between two nodes of the same kind, the peering links, "
        "take its color.")
def add_random_internet_as_graph(s: Scene) -> None:
    box = single(s, "g.add_random_internet_as_graph(n=20)")
    g, edges = _seeded("add_random_internet_as_graph", "20")
    kinds = {str(k): v for k, v in g.get_node_attrs("type").items()}
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    coords = ring(sorted((k for k, v in kinds.items() if v == "T"), key=int), 0.85)
    for row, tier in ((2.0, ("M", "CP")), (3.3, ("C",))):
        # each row sorted under the mean position of its links to the rows above
        ids = [k for k, v in kinds.items() if v in tier]
        above = {k: [coords[v][0] for v in adj.get(k, []) if v in coords] or [0.0] for k in ids}
        ids.sort(key=lambda k: (sum(above[k]) / len(above[k]), int(k)))
        step = 1.2 if tier == ("C",) else 2.2
        coords |= {k: ((j - (len(ids) - 1) / 2) * step, row) for j, k in enumerate(ids)}
    styles = {k: INTERNET_TYPES[v] for k, v in kinds.items()}
    _live(s, box, coords, edges, prefix="as.", styles=styles, unit=44,
          edge_style={frozenset((a, b)): styles[a] for a, b in edges if kinds[a] == kinds[b]})


VISIBILITY = ((1, 3, 2), (3, 1, 2, 1, 3), (2, 1, 5, 1, 3, 1))


@figure("add_visibility_graph", "generators", "Visibility graph",
        "Three natural visibility graphs of short time series, each value drawn as a dashed "
        "stem rising from a baseline with its node on top. Two values are joined when the "
        "straight line between their tops clears every value in between, so neighbors in time "
        "are always joined and a tall value such as the 5 in the last series sees far in both "
        "directions.")
def add_visibility_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"series={_seq(v)}" for v in VISIBILITY))
    for i, (box, series) in enumerate(zip(boxes, VISIBILITY)):
        _, edges = _built(f"add_visibility_graph({list(series)})")
        coords = {str(k): (k, -0.5 * v) for k, v in enumerate(series, 1)}
        coords |= {f"base{k}": (k, 0.0) for k in range(1, len(series) + 1)}
        pos = place(coords, box, unit=UNIT, r=R_SMALL)
        for k in range(1, len(series) + 1):
            (x, y), (_, base) = pos[str(k)], pos.pop(f"base{k}")
            s.line(x, y + R_SMALL, x, base, opacity=0.35, width=1, dash="3 3")
        x0, x1 = pos["1"][0], pos[str(len(series))][0]
        s.line(x0 - 14, base, x1 + 14, base, opacity=0.35, width=1, role="axis")
        _live(s, box, {}, edges, prefix=f"v{i}.", pos=pos)


INTERVALS = (
    ((0, 2), (1, 3), (4, 5)),
    ((0, 3), (1, 2), (2.5, 4), (3.5, 5)),
    ((0, 1.5), (1, 4), (2, 3), (3.5, 5), (4.5, 6)),
)


@figure("add_interval_graph", "generators", "Interval graph",
        "Three interval graphs, each interval drawn as a faint bar on its own row with its node "
        "at the middle. Two nodes are joined when their intervals overlap, touching endpoints "
        "included: the third interval of the first set meets neither of the others, and the long "
        "second interval of the last set overlaps three of the other four.")
def add_interval_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"{len(iv)} intervals" for iv in INTERVALS))
    for i, (box, intervals) in enumerate(zip(boxes, INTERVALS)):
        _, edges = _built(f"add_interval_graph({[tuple(map(float, iv)) for iv in intervals]})")
        coords = {}
        for k, (lo, hi) in enumerate(intervals, 1):
            y = k - 1.0
            coords |= {str(k): ((lo + hi) / 2, y), f"lo{k}": (lo, y), f"hi{k}": (hi, y)}
        pos = place(coords, box, unit=40, r=R_SMALL)
        for k in range(1, len(intervals) + 1):
            (x0, y), (x1, _) = pos.pop(f"lo{k}"), pos.pop(f"hi{k}")
            s.line(x0, y, x1, y, color="accent", opacity=0.35, width=5)
        _live(s, box, {}, edges, prefix=f"i{i}.", pos=pos)


SUDOKU_CELLS = ((0, 0), (1, 2), (3, 3))


@figure("add_sudoku_graph", "generators", "Sudoku graph",
        "The 4 by 4 Sudoku graph for n=2 three times, drawn as its grid with a small gap between "
        "the 2 by 2 boxes and, in each panel, only the edges of one cell shown, in the accent "
        "color. Each cell is joined to the other cells in its row, its column, and its box, "
        "seven in all, so a valid Sudoku solution is a proper four-coloring of the graph. Edges "
        "along a row or column bow around the cells in between.")
def add_sudoku_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"n=2, neighbors of ({r}, {c})" for r, c in SUDOKU_CELLS))
    g, edges = _built("add_sudoku_graph(2)")
    cells = {lab: k for k, lab in _labels(g).items()}
    coords = {k: (c + 0.3 * (c // 2), r + 0.3 * (r // 2))
              for k, (r, c) in ((k, eval(lab)) for lab, k in cells.items())}
    for i, (box, cell) in enumerate(zip(boxes, SUDOKU_CELLS)):
        hub = cells[f"({cell[0]},{cell[1]})"]
        mine = [e for e in edges if hub in e]
        nbrs = {v for e in mine for v in e} - {hub}
        _live(s, box, coords, mine, prefix=f"s{i}.", unit=38,
              styles={hub: "accent", **{v: "soft" for v in nbrs}},
              edge_style={frozenset(e): "accent" for e in mine})


PREFIX_PATHS = (["ab", "ac"], ["ab", "ac", "b"], ["cat", "car", "dog"])


@figure("add_prefix_tree", "generators", "Prefix tree",
        "Three prefix trees (tries), each node labeled with the character it stands for and "
        "hung from an unlabeled root at the top. Paths that share a prefix share its nodes, so "
        "ab and ac branch below a single a, and every path ends at the one NIL node at the "
        "bottom, as in NetworkX.")
def add_prefix_tree(s: Scene) -> None:
    boxes = multiples(s, tuple(f"paths={_seq([repr(p) for p in ps])}".replace("'", '"')
                               for ps in PREFIX_PATHS))
    for i, (box, paths) in enumerate(zip(boxes, PREFIX_PATHS)):
        g, edges = _built(f"add_prefix_tree({paths})")
        source = {str(k): v for k, v in g.get_node_attrs("source").items()}
        nil = next(k for k, v in source.items() if v == "NIL")
        trie = [e for e in edges if nil not in e]
        coords = _tidy_tree(_bfs_children(trie, "1"), "1")
        xs = [x for x, _ in coords.values()]
        coords[nil] = ((min(xs) + max(xs)) / 2, 1 + max(y for _, y in coords.values()))
        labels = {k: "" if source.get(k) in (None, "NIL") else source[k] for k in coords}
        box = Box(box.x0, box.y0, box.x1, box.y1 - 14)  # room for the NIL pill below
        _live(s, box, coords, edges, prefix=f"t{i}.", labels=labels, styles={nil: "soft"},
              pad=2)
        s.badge(f"t{i}.{nil}", "NIL", where="s")


# --- small graph enumeration ------------------------------------------------------
# Several small graphs per panel, each in its own cell of a grid, at a smaller
# unit so they read as a catalog.


def _gallery(s: Scene, box: Box, graphs, prefix: str, *, cols: int, unit: float,
             layout) -> None:
    """Draw `graphs` (lists of str edge pairs plus node ids) in a `cols`-wide grid."""
    rows = math.ceil(len(graphs) / cols)
    w, h = box.w / cols, box.h / rows
    for j, (ids, edges) in enumerate(graphs):
        r, c = divmod(j, cols)
        cell = Box(box.x0 + c * w, box.y0 + r * h, box.x0 + (c + 1) * w, box.y0 + (r + 1) * h)
        _live(s, cell, layout(ids, edges), edges, prefix=f"{prefix}{j}.", unit=unit, pad=2)


def _graph_parts(g) -> tuple[list[str], list[tuple[str, str]]]:
    ids = [str(k) for k in sorted(g.get_node_ids(limit=None))]
    return ids, [(str(a), str(b)) for _, a, b in g.get_edges_as_tuples()]


def _ring_layout(ids, edges) -> dict:
    return _ring_xy(len(ids))


def _spine_layout(ids, edges) -> dict:
    """A tree's longest path as a row, the other nodes hanging below it."""
    def far(v):
        kids = _bfs_children(edges, v)
        order = [v] + [c for p in kids.values() for c in p]
        return order[-1], kids

    end, _ = far(ids[0])
    other, kids = far(end)
    spine = [other]
    parent = {c: p for p, cs in kids.items() for c in cs}
    while spine[-1] != end:
        spine.append(parent[spine[-1]])
    on_spine = set(spine)
    tree = {"root": spine}
    for v in spine:
        hang = _bfs_children([e for e in edges if not set(e) <= on_spine], v)
        tree[v] = hang.get(v, [])
        tree |= {k: c for k, c in hang.items() if k != v}
    coords = _tidy_tree(tree, "root")
    del coords["root"]
    return coords


ATLAS_INDEXES = (16, 38, 52)


@figure("add_atlas_graph", "generators", "Atlas graph",
        "Three graphs from An Atlas of Graphs, on a ring in node order: index 16 is the "
        "four-cycle, index 38 the five-cycle, and index 52 the complete graph K5, the last "
        "graph on five nodes. "
        "The atlas lists all 1253 graphs on up to seven nodes once each, up to isomorphism, "
        "ordered by node count, then edge count, then degree sequence.")
def add_atlas_graph(s: Scene) -> None:
    boxes = multiples(s, tuple(f"g.add_atlas_graph(index={i})" for i in ATLAS_INDEXES))
    for box, index in zip(boxes, ATLAS_INDEXES):
        g, edges = _built(f"add_atlas_graph({index})")
        _live(s, box, _ring_xy(g.count_nodes()), edges, prefix=f"a{index}.")


ATLAS_SLICES = ((4, 8), (8, 12), (15, 19))


@figure("atlas_graphs", "generators", "Atlas graphs",
        "Three slices of the list of all 1253 atlas graphs, four small graphs each, in atlas "
        "order. Items 4 to 7 are every graph on three nodes, from no edges to the triangle; "
        "items 8 to 11 start the four-node graphs with zero, one, and two edges; items 15 to 18 "
        "end them with four, five, and six edges, from the paw to the complete graph.")
def atlas_graphs(s: Scene) -> None:
    import edgecase as ec

    atlas = ec.Graph.atlas_graphs()
    boxes = multiples(s, tuple(f"Graph.atlas_graphs()[{a}:{b}]" for a, b in ATLAS_SLICES))
    for i, (box, (a, b)) in enumerate(zip(boxes, ATLAS_SLICES)):
        _gallery(s, box, [_graph_parts(g) for g in atlas[a:b]], f"s{i}.", cols=2, unit=34,
                 layout=_ring_layout)


TREE_ORDERS = (3, 4, 5)


@figure("iter_nonisomorphic_trees", "generators", "Non-isomorphic trees",
        "The trees yielded for order 3, 4, and 5, each drawn with its longest path as a row and "
        "any other nodes hanging below. There is one tree on three nodes, the path; two on four "
        "nodes, the path and the star; and three on five nodes, the path, the chair, and the "
        "star. Every tree on that many nodes looks like exactly one of them.")
def iter_nonisomorphic_trees(s: Scene) -> None:
    import edgecase as ec

    boxes = multiples(s, tuple(f"iter_nonisomorphic_trees({n})" for n in TREE_ORDERS))
    for box, n in zip(boxes, TREE_ORDERS):
        trees = [_graph_parts(g) for g in ec.Graph.iter_nonisomorphic_trees(n)]
        _gallery(s, box, trees, f"n{n}.", cols=1, unit=34, layout=_spine_layout)


COUNT_ORDERS = range(3, 10)
COUNT_SHOWN = 5  # the order whose trees are drawn and whose count is highlighted


@figure("count_nonisomorphic_trees", "generators", "Count non-isomorphic trees",
        "Left: the three trees on five nodes, the path, the chair, and the star, each drawn with "
        "its longest path as a row and any other nodes hanging below. Right: a table of how many "
        "trees there are on 3 to 9 nodes, 1, 2, 3, 6, 11, 23, and 47, with the count 3 for five "
        "nodes highlighted. The count comes from a formula, without building the trees, and "
        "roughly doubles with each added node at these sizes.")
def count_nonisomorphic_trees(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, f"iter_nonisomorphic_trees({COUNT_SHOWN})",
                          f"Graph.count_nonisomorphic_trees({COUNT_SHOWN})")
    trees = [_graph_parts(g) for g in ec.Graph.iter_nonisomorphic_trees(COUNT_SHOWN)]
    _gallery(s, left, trees, "t.", cols=1, unit=34, layout=_spine_layout)
    counts = [ec.Graph.count_nonisomorphic_trees(n) for n in COUNT_ORDERS]
    assert counts[COUNT_SHOWN - COUNT_ORDERS.start] == len(trees)
    col = 1 + COUNT_SHOWN - COUNT_ORDERS.start
    s.table_centered(right, [["order", *map(str, COUNT_ORDERS)], ["count", *map(str, counts)]],
                     [48] + [26] * len(counts), header=False,
                     highlight={(0, col): "accent", (1, col): "accent"})
