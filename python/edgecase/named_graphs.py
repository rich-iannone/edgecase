"""Registry of small named graphs for `Graph.add_named_graph()`.

Each entry gives a node count and an edge list over nodes `0..n-1`. Hamiltonian
cubic graphs are written in LCF notation (a Hamiltonian cycle `0, 1, ..., n-1`
plus one chord per node); the rest are listed edge by edge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedGraph:
    description: str
    n: int
    edges: tuple[tuple[int, int], ...]


def lcf_edges(n: int, shifts: list[int], repeats: int) -> list[tuple[int, int]]:
    """Edges of the LCF graph `[shifts]^repeats` on `n` nodes, without repeats."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []

    def add(a: int, b: int) -> None:
        key = (min(a, b), max(a, b))
        if a != b and key not in seen:
            seen.add(key)
            out.append(key)

    for i in range(n):
        add(i, (i + 1) % n)
    for k in range(len(shifts) * repeats):
        i = k % n
        add(i, (i + shifts[k % len(shifts)]) % n)
    return out


def mycielski_edges(n: int) -> tuple[int, list[tuple[int, int]]]:
    """Node count and edges of the Mycielski graph M_n (M_1 = K1, M_2 = K2)."""
    size, edges = 1, []
    for _ in range(n - 1):
        if size == 1:
            size, edges = 2, [(0, 1)]
            continue
        # shadow u_i = size + i copies each neighborhood of v_i; hub w joins every u_i
        edges = edges + [(a, size + b) for a, b in edges] + [(b, size + a) for a, b in edges]
        hub = 2 * size
        edges += [(size + i, hub) for i in range(size)]
        size = 2 * size + 1
    return size, edges


def _hoffman_singleton() -> list[tuple[int, int]]:
    # pentagons P_h (nodes 5h + i) and pentagrams Q_j (nodes 25 + 5j + i);
    # P_h,i joins Q_j,(h*j + i) mod 5
    edges = []
    for h in range(5):
        for i in range(5):
            edges.append((5 * h + i, 5 * h + (i + 1) % 5))
            edges.append((25 + 5 * h + i, 25 + 5 * h + (i + 2) % 5))
            for j in range(5):
                edges.append((5 * h + i, 25 + 5 * j + (h * j + i) % 5))
    return edges


def _entry(description: str, n: int, edges) -> NamedGraph:
    return NamedGraph(description, n, tuple(sorted((min(a, b), max(a, b)) for a, b in edges)))


_GROTZSCH = mycielski_edges(4)

NAMED_GRAPHS: dict[str, NamedGraph] = {
    "bull": _entry("A triangle with two pendant horns", 5,
                   [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)]),
    "butterfly": _entry("Two triangles sharing one node (the bowtie)", 5,
                        [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]),
    "chvatal": _entry("Smallest triangle-free 4-regular graph with chromatic number 4", 12,
                      [(0, 1), (0, 4), (0, 6), (0, 9), (1, 2), (1, 5), (1, 7), (2, 3), (2, 6),
                       (2, 8), (3, 4), (3, 7), (3, 9), (4, 5), (4, 8), (5, 10), (5, 11), (6, 10),
                       (6, 11), (7, 8), (7, 11), (8, 10), (9, 10), (9, 11)]),
    "cubical": _entry("Skeleton of the cube, the 3-dimensional hypercube", 8,
                      lcf_edges(8, [3, -3], 4)),
    "desargues": _entry("Cubic symmetric graph on 20 nodes, the Levi graph of the Desargues "
                        "configuration", 20, lcf_edges(20, [5, -5, 9, -9], 5)),
    "diamond": _entry("Complete graph K4 with one edge removed", 4,
                      [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
    "dodecahedral": _entry("Skeleton of the dodecahedron", 20,
                           lcf_edges(20, [10, 7, 4, -4, -7, 10, -4, 7, -7, 4], 2)),
    "franklin": _entry("Cubic bipartite graph on 12 nodes that embeds on the Klein bottle", 12,
                       lcf_edges(12, [5, -5], 6)),
    "frucht": _entry("Smallest cubic graph with no symmetry (trivial automorphism group)", 12,
                     lcf_edges(12, [-5, -2, -4, 2, 5, -2, 2, 5, -2, -5, 4, 2], 1)),
    "grotzsch": _entry("Smallest triangle-free graph with chromatic number 4 (Mycielski M4)",
                       _GROTZSCH[0], _GROTZSCH[1]),
    "heawood": _entry("The (3,6)-cage, Levi graph of the Fano plane", 14,
                      lcf_edges(14, [5, -5], 7)),
    "hoffman_singleton": _entry("The (7,5)-cage, a 7-regular Moore graph of diameter 2", 50,
                                _hoffman_singleton()),
    "house": _entry("A square with a triangular roof", 5,
                    [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
    "house_x": _entry("The house graph with both diagonals of the square", 5,
                      [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
    "icosahedral": _entry("Skeleton of the icosahedron", 12,
                          [(0, 1), (0, 5), (0, 7), (0, 8), (0, 11), (1, 2), (1, 5), (1, 6),
                           (1, 8), (2, 3), (2, 6), (2, 8), (2, 9), (3, 4), (3, 6), (3, 9),
                           (3, 10), (4, 5), (4, 6), (4, 10), (4, 11), (5, 6), (5, 11), (7, 8),
                           (7, 9), (7, 10), (7, 11), (8, 9), (9, 10), (10, 11)]),
    "krackhardt_kite": _entry("Krackhardt's social network whose most central node differs by "
                              "degree, betweenness, and closeness", 10,
                              [(0, 1), (0, 2), (0, 3), (0, 5), (1, 3), (1, 4), (1, 6), (2, 3),
                               (2, 5), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 7), (6, 7),
                               (7, 8), (8, 9)]),
    "mcgee": _entry("The (3,7)-cage, the smallest cubic graph of girth 7", 24,
                    lcf_edges(24, [12, 7, -7], 8)),
    "moebius_kantor": _entry("Cubic symmetric graph on 16 nodes, the generalized Petersen "
                             "graph GP(8,3)", 16, lcf_edges(16, [5, -5], 8)),
    "nauru": _entry("Cubic symmetric graph on 24 nodes, the generalized Petersen graph GP(12,5)",
                    24, lcf_edges(24, [5, -9, 7, -7, 9, -5], 4)),
    "octahedral": _entry("Skeleton of the octahedron, the complete multipartite graph "
                         "K(2,2,2)", 6,
                         [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5),
                          (3, 4), (3, 5), (4, 5)]),
    "pappus": _entry("Cubic symmetric graph on 18 nodes, the Levi graph of the Pappus "
                     "configuration", 18, lcf_edges(18, [5, 7, -7, 7, -7, -5], 3)),
    "petersen": _entry("The (3,5)-cage, a common counterexample in graph theory", 10,
                       [(0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8),
                        (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9)]),
    "sedgewick_maze": _entry("Small maze from Sedgewick's Algorithms, a traversal test case", 8,
                             [(0, 2), (0, 5), (0, 7), (1, 7), (2, 6), (3, 4), (3, 5), (4, 5),
                              (4, 6), (4, 7)]),
    "tetrahedral": _entry("Skeleton of the tetrahedron, the complete graph K4", 4,
                          [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    "truncated_cube": _entry("Skeleton of the truncated cube", 24,
                             lcf_edges(24, [2, 9, -2, 2, -9, -2], 4)),
    "truncated_tetrahedron": _entry("Skeleton of the truncated tetrahedron", 12,
                                    lcf_edges(12, [2, 6, -2], 4)),
    "tutte": _entry("Cubic, planar, 3-connected graph with no Hamiltonian cycle", 46,
                    [(0, 1), (0, 2), (0, 3), (1, 4), (1, 26), (2, 10), (2, 11), (3, 18), (3, 19),
                     (4, 5), (4, 33), (5, 6), (5, 29), (6, 7), (6, 27), (7, 8), (7, 14), (8, 9),
                     (8, 38), (9, 10), (9, 37), (10, 39), (11, 12), (11, 39), (12, 13), (12, 35),
                     (13, 14), (13, 15), (14, 34), (15, 16), (15, 22), (16, 17), (16, 44),
                     (17, 18), (17, 43), (18, 45), (19, 20), (19, 45), (20, 21), (20, 41),
                     (21, 22), (21, 23), (22, 40), (23, 24), (23, 27), (24, 25), (24, 32),
                     (25, 26), (25, 31), (26, 33), (27, 28), (28, 29), (28, 32), (29, 30),
                     (30, 31), (30, 33), (31, 32), (34, 35), (34, 38), (35, 36), (36, 37),
                     (36, 39), (37, 38), (40, 41), (40, 44), (41, 42), (42, 43), (42, 45),
                     (43, 44)]),
    "tutte_coxeter": _entry("The (3,8)-cage, the Levi graph of the Cremona-Richmond "
                            "configuration", 30, lcf_edges(30, [-13, -9, 7, -7, 9, 13], 5)),
}
