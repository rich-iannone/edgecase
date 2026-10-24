"""Tests for the small structural functions added in NetworkX 3.6/3.7: `is_perfect_graph`,
`hyper_wiener_index`, `antichain_width`, bipartite `butterflies`, planar faces and
`is_valid_tree_degree_sequence`.

The installed NetworkX may be older and lack them, so the references below are brute force or
re-implement NetworkX 3.7's definitions from functions older releases have.
"""

import itertools
import math
import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _build(G):
    """An edgecase graph with node labels str(v) for NetworkX's nodes."""
    g = ec.Graph(directed=G.is_directed()).add_nodes([str(v) for v in G])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges()])
    return g


def _by_label(g, values):
    return {g.get_label_from_id(n): v for n, v in values.items()}


def _random_graphs(count, seed, n_range=(0, 11)):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(*n_range)
        yield nx.gnp_random_graph(n, rng.choice([0.2, 0.35, 0.5, 0.7]), seed=rng.randrange(2**31))


def _random_dags(count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(0, 10)
        p = rng.choice([0.1, 0.25, 0.4, 0.7])
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        order = list(range(n))
        rng.shuffle(order)
        G.add_edges_from(
            (order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < p
        )
        yield G


# =============================================================================
# Perfect graphs
# =============================================================================


def _ref_perfect(G):
    """NetworkX 3.7: no odd hole in G or its complement (holes via chordless_cycles)."""

    def has_odd_hole(H):
        return any(len(c) >= 5 and len(c) % 2 == 1 for c in nx.chordless_cycles(H))

    return not has_odd_hole(G) and not has_odd_hole(nx.complement(G))


class TestIsPerfectGraph:
    def test_matches_reference(self):
        for G in _random_graphs(250, seed=1):
            assert _build(G).is_perfect_graph() == _ref_perfect(G), sorted(G.edges)

    def test_cycles(self):
        for n in range(3, 12):
            expected = n < 5 or n % 2 == 0
            assert ec.Graph().add_cycle_graph(n).is_perfect_graph() is expected

    def test_odd_antihole(self):
        for n in (7, 9):
            antihole = ec.Graph().add_cycle_graph(n).complement()
            assert not antihole.is_perfect_graph()
        # The complement of C5 is C5 again
        assert not ec.Graph().add_cycle_graph(5).complement().is_perfect_graph()

    def test_known_families(self):
        assert ec.Graph().add_complete_bipartite_graph(4, 5).is_perfect_graph()
        assert ec.Graph().add_full_graph(7).is_perfect_graph()
        assert ec.Graph().add_grid_2d(4, 5).is_perfect_graph()
        assert not ec.Graph().add_named_graph("petersen").is_perfect_graph()
        # A wheel with an odd rim of length 5 or more contains its rim as an odd hole
        assert not ec.Graph().add_wheel_graph(6).is_perfect_graph()
        assert ec.Graph().add_wheel_graph(7).is_perfect_graph() == _ref_perfect(nx.wheel_graph(7))

    def test_small_and_empty(self):
        assert ec.Graph().is_perfect_graph()
        assert ec.Graph().add_nodes(list("ABCD")).is_perfect_graph()

    def test_hole_beyond_64_nodes(self):
        # Bitsets span more than one word
        G = nx.cycle_graph(71)
        assert not _build(G).is_perfect_graph()
        G.add_edge(0, 2)
        assert _build(G).is_perfect_graph() == _ref_perfect(G)

    def test_loops_and_parallel_edges_ignored(self):
        g = ec.Graph().add_cycle_graph(5)
        assert not g.is_perfect_graph()
        _, s, t = g.get_edges_as_tuples()[0]
        g.add_edges_by_id([(s, s), (s, t)])
        assert not g.is_perfect_graph()

    def test_directed_raises(self):
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).is_perfect_graph()

    def test_step(self):
        g = ec.Graph().add_cycle_graph(5)
        g.is_perfect_graph()
        assert g.steps[-1].code == "G.is_perfect_graph()"
        assert g.steps[-1].narration == "is_perfect_graph(): False."


# =============================================================================
# Hyper-Wiener index
# =============================================================================


def _ref_hyper_wiener(G, weight=None):
    """Randić's definition over unordered pairs (half of NetworkX 3.7's value)."""
    if len(G) < 2:
        return 0.0
    if not nx.is_connected(G):
        return math.inf
    total = sum(
        d + d * d
        for u, lengths in nx.shortest_path_length(G, weight=weight)
        for v, d in lengths.items()
        if u < v
    )
    return total / 2


class TestHyperWienerIndex:
    def test_known_values(self):
        # Standard values; NetworkX 3.7 returns twice these (30, 20, 210)
        assert ec.Graph().add_path_graph(4).get_hyper_wiener_index() == 15.0
        assert ec.Graph().add_cycle_graph(4).get_hyper_wiener_index() == 10.0
        assert ec.Graph().add_named_graph("petersen").get_hyper_wiener_index() == 105.0
        # Complete graph: every pair at distance 1
        assert ec.Graph().add_full_graph(6).get_hyper_wiener_index() == 15.0

    def test_matches_reference(self):
        for G in _random_graphs(80, seed=2, n_range=(0, 14)):
            ours = _build(G).get_hyper_wiener_index()
            assert ours == pytest.approx(_ref_hyper_wiener(G)), sorted(G.edges)

    def test_weighted(self):
        rng = random.Random(5)
        G = nx.connected_watts_strogatz_graph(15, 4, 0.3, seed=5)
        for u, v in G.edges:
            G[u][v]["weight"] = rng.choice([0.5, 1.0, 2.5])
        g = ec.Graph(weighted=True).add_nodes([str(v) for v in G])
        for u, v, w in G.edges(data="weight"):
            g.add_edge(str(u), str(v), weight=w)
        assert g.get_hyper_wiener_index(weight="weight") == pytest.approx(
            _ref_hyper_wiener(G, weight="weight")
        )
        assert g.get_hyper_wiener_index() == pytest.approx(_ref_hyper_wiener(G))

    def test_disconnected_and_tiny(self):
        assert ec.Graph().get_hyper_wiener_index() == 0.0
        assert ec.Graph().add_nodes(["A"]).get_hyper_wiener_index() == 0.0
        assert ec.Graph().add_nodes(["A", "B"]).get_hyper_wiener_index() == math.inf

    def test_directed_raises(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="undirected"):
            g.get_hyper_wiener_index()


# =============================================================================
# Antichain width
# =============================================================================


class TestAntichainWidth:
    def test_matches_largest_antichain(self):
        for G in _random_dags(150, seed=3):
            expected = max(len(a) for a in nx.antichains(G))
            assert _build(G).antichain_width() == expected, sorted(G.edges)

    def test_diamond(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        assert g.antichain_width() == 2

    def test_divisibility_order(self):
        n = 100
        g = ec.Graph(directed=True).add_nodes([str(i) for i in range(1, n + 1)])
        g.add_edges([(str(a), str(b)) for a in range(1, n + 1) for b in range(2 * a, n + 1, a)])
        assert g.antichain_width() == n // 2

    def test_chain_and_edgeless(self):
        chain = ec.Graph(directed=True).add_nodes(list("ABCDE"))
        chain.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        assert chain.antichain_width() == 1
        assert ec.Graph(directed=True).add_nodes(list("ABCDE")).antichain_width() == 5
        assert ec.Graph(directed=True).antichain_width() == 0

    def test_large_closure(self):
        # A long chain plus a parallel one: more than 64 nodes, so bitsets span several words
        G = nx.DiGraph()
        G.add_edges_from((i, i + 1) for i in range(79))
        G.add_edges_from((100 + i, 100 + i + 1) for i in range(49))
        G.add_edges_from((i, 100 + i) for i in range(0, 50, 7))
        assert _build(G).antichain_width() == 2

    def test_parallel_edges(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC"))
        g.add_edges([("A", "B"), ("A", "B"), ("A", "C")])
        assert g.antichain_width() == 2

    def test_errors(self):
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_nodes(["A"]).antichain_width()
        cyc = ec.Graph(directed=True).add_nodes(list("AB")).add_edges([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="acyclic"):
            cyc.antichain_width()
        loop = ec.Graph(directed=True).add_nodes(["A"]).add_edge("A", "A")
        with pytest.raises(ValueError, match="acyclic"):
            loop.antichain_width()

    def test_step(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC")).add_edge("A", "B")
        g.antichain_width()
        assert g.steps[-1].code == "G.antichain_width()"
        assert g.steps[-1].narration == "Width of the partial order: 2."


# =============================================================================
# Butterflies
# =============================================================================


def _ref_butterflies(G):
    """Number of 4-cycles (on four distinct nodes) through each node, by brute force."""
    count = dict.fromkeys(G, 0)
    for quad in itertools.combinations(G, 4):
        a, b, c, d = quad
        # The three distinct cyclic orders of four nodes
        for cyc in ((a, b, c, d), (a, b, d, c), (a, c, b, d)):
            if all(G.has_edge(cyc[i], cyc[(i + 1) % 4]) for i in range(4)):
                for v in quad:
                    count[v] += 1
    return count


class TestButterflies:
    def test_bipartite_matches_reference(self):
        for seed in range(25):
            B = nx.bipartite.random_graph(6, 7, 0.45, seed=seed)
            g = _build(B)
            ref = {str(v): c for v, c in _ref_butterflies(B).items()}
            assert _by_label(g, g.butterflies()) == ref

    def test_general_graphs_count_four_cycles(self):
        for G in _random_graphs(40, seed=4):
            g = _build(G)
            ref = {str(v): c for v, c in _ref_butterflies(G).items()}
            assert _by_label(g, g.butterflies()) == ref

    def test_complete_bipartite(self):
        for a, b in ((2, 2), (3, 3), (3, 5), (4, 6)):
            counts = ec.Graph().add_complete_bipartite_graph(a, b).butterflies()
            assert sum(counts.values()) == 4 * math.comb(a, 2) * math.comb(b, 2)
            assert sorted(set(counts.values())) == sorted(
                {(a - 1) * math.comb(b, 2), (b - 1) * math.comb(a, 2)}
            )

    def test_hub_heavy(self):
        # A high-degree hub exercises the degree priority
        B = nx.Graph()
        B.add_edges_from(("h", f"r{i}") for i in range(30))
        B.add_edges_from((f"t{j}", f"r{i}") for j in range(4) for i in range(j, 30, 3))
        g = _build(B)
        assert _by_label(g, g.butterflies()) == {
            str(v): c for v, c in _ref_butterflies(B).items()
        }

    def test_nodes_filter(self):
        g = ec.Graph().add_nodes(["a", "b", "x", "y", "z"])
        g.add_edges([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("b", "z")])
        sub = g.butterflies(nodes=["a", "z"])
        assert _by_label(g, sub) == {"a": 1, "z": 0}
        with pytest.raises(ValueError):
            g.butterflies(nodes=["nobody"])

    def test_loops_and_parallel_edges_ignored(self):
        g = ec.Graph().add_nodes(["a", "b", "x", "y"])
        g.add_edges([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")])
        before = g.butterflies()
        g.add_edges([("a", "x"), ("a", "a")])
        assert g.butterflies() == before == {n: 1 for n in g.get_node_ids()}

    def test_empty_and_directed(self):
        assert ec.Graph().butterflies() == {}
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).butterflies()

    def test_step(self):
        g = ec.Graph().add_complete_bipartite_graph(2, 3)
        g.butterflies()
        step = g.steps[-1]
        assert step.code == "G.butterflies()"
        assert step.narration == "Butterfly counts of 5 nodes (3 butterflies in all)."
        g.butterflies(nodes=[g.get_node_ids()[0]])
        assert g.steps[-1].code == "G.butterflies(nodes=nodes)"


# =============================================================================
# Planar faces
# =============================================================================


def _check_faces(g, faces):
    """Every face walk follows edges, every half-edge is used once, and Euler's formula holds."""
    nbrs = {n: set() for n in g.get_node_ids(limit=None)}
    for _, s, t in g.get_edges_as_tuples():
        if s != t:
            nbrs[s].add(t)
            nbrs[t].add(s)
    half_edges = []
    for face in faces:
        assert face
        for i, a in enumerate(face):
            b = face[(i + 1) % len(face)]
            assert b in nbrs[a]
            half_edges.append((a, b))
    expected = sorted((a, b) for a in nbrs for b in nbrs[a])
    assert sorted(half_edges) == expected
    G = nx.Graph()
    G.add_edges_from((a, b) for a in nbrs for b in nbrs[a])
    euler = sum(
        G.subgraph(c).number_of_edges() - len(c) + 2 for c in nx.connected_components(G)
    )
    assert len(faces) == euler


class TestPlanarFaces:
    def test_random_planar_graphs(self):
        rng = random.Random(6)
        checked = 0
        for _ in range(200):
            n = rng.randint(1, 14)
            G = nx.gnp_random_graph(n, rng.choice([0.15, 0.3, 0.5]), seed=rng.randrange(2**31))
            if not nx.is_planar(G):
                continue
            g = _build(G)
            _check_faces(g, g.planar_faces())
            checked += 1
        assert checked > 100

    def test_faces_follow_check_planarity_embedding(self):
        # Each step a -> b continues to the neighbor of b just before a in clockwise order
        for G in (nx.wheel_graph(7), nx.grid_2d_graph(3, 4), nx.random_labeled_tree(9, seed=1)):
            g = _build(G)
            _, emb = g.check_planarity()
            for face in g.planar_faces():
                for i, a in enumerate(face):
                    b = face[(i + 1) % len(face)]
                    c = face[(i + 2) % len(face)]
                    around = emb[b]
                    assert around[(around.index(a) - 1) % len(around)] == c

    def test_square_with_diagonal(self):
        g = ec.Graph().add_nodes(list("ABCD"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])
        faces = [[g.get_label_from_id(n) for n in f] for f in g.planar_faces()]
        assert sorted(len(f) for f in faces) == [3, 3, 4]
        assert sorted(sorted(f) for f in faces if len(f) == 3) == [["A", "B", "C"], ["A", "C", "D"]]

    def test_polyhedra(self):
        cube = ec.Graph().add_hypercube_graph(3)
        assert sorted(len(f) for f in cube.planar_faces()) == [4] * 6
        dodeca = ec.Graph().add_named_graph("dodecahedral")
        assert sorted(len(f) for f in dodeca.planar_faces()) == [5] * 12

    def test_tree_and_isolated_nodes(self):
        g = ec.Graph().add_nodes(list("ABCDE")).add_edges([("A", "B"), ("B", "C"), ("B", "D")])
        faces = g.planar_faces()
        assert len(faces) == 1 and len(faces[0]) == 6
        _check_faces(g, faces)
        assert ec.Graph().add_nodes(["A"]).planar_faces() == []
        assert ec.Graph().planar_faces() == []

    def test_two_components(self):
        G = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(4))
        g = _build(G)
        faces = g.planar_faces()
        # Each cycle has an inside and an outside
        assert sorted(len(f) for f in faces) == [3, 3, 4, 4]

    def test_ignores_direction_loops_and_parallel_edges(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("A", "B"), ("B", "B")])
        assert sorted(len(f) for f in g.planar_faces()) == [3, 3]

    def test_non_planar_raises(self):
        with pytest.raises(ValueError, match="not planar"):
            ec.Graph().add_named_graph("petersen").planar_faces()
        with pytest.raises(ValueError, match="not planar"):
            ec.Graph().add_full_graph(5).planar_faces()


# =============================================================================
# Tree degree sequences
# =============================================================================


class TestIsValidTreeDegreeSequence:
    def test_matches_enumeration(self):
        for n in range(1, 8):
            tree_seqs = {
                tuple(sorted(d for _, d in T.degree())) for T in nx.nonisomorphic_trees(n)
            } if n > 1 else {(0,)}
            for seq in itertools.combinations_with_replacement(range(n), n):
                assert ec.Graph.is_valid_tree_degree_sequence(list(seq)) == (seq in tree_seqs)

    def test_examples(self):
        valid = ec.Graph.is_valid_tree_degree_sequence
        assert valid([3, 1, 1, 1])
        assert valid([1, 2, 2, 1])
        assert valid([0])
        assert valid([1, 1])
        assert not valid([2, 2, 2, 2])
        assert not valid([2, 2, 0, 1, 1])
        assert not valid([])
        assert not valid([1])
        assert not valid([5, 4, -1, 1, 1, 1, 1])  # right sum, negative degree

    def test_any_order_and_iterables(self):
        assert ec.Graph.is_valid_tree_degree_sequence((1, 3, 1, 1))
        assert ec.Graph.is_valid_tree_degree_sequence(iter([1, 1, 2]))

    def test_random_trees(self):
        for seed in range(10):
            t = ec.Graph().add_random_labeled_tree(15, seed=seed)
            degrees = [t.get_degree(n) for n in t.get_node_ids()]
            assert ec.Graph.is_valid_tree_degree_sequence(degrees)
