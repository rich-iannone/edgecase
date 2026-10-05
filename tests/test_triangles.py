import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_METRIC


class TestTriangleCounting:
    """Test triangle counting functions."""

    def test_count_triangles_empty_graph(self):
        """Empty graph has no triangles."""
        g = ec.Graph()
        counts = g.count_triangles()
        assert len(counts) == 0

    def test_count_triangles_single_node(self):
        """Single node has no triangles."""
        g = ec.Graph()
        g.add_node(label="A")
        counts = g.count_triangles()
        a_id = g.get_id_from_label("A")
        assert counts[a_id] == 0

    def test_count_triangles_single_triangle(self):
        """Single triangle: each node participates in one triangle."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        counts = g.count_triangles()
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        assert counts[a_id] == 1
        assert counts[b_id] == 1
        assert counts[c_id] == 1

    def test_count_triangles_chain(self):
        """Chain graph has no triangles."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        counts = g.count_triangles()
        assert all(count == 0 for count in counts.values())

    def test_count_triangles_multiple(self):
        """Graph with multiple triangles."""
        g = ec.Graph()
        # Two triangles sharing an edge B-C
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Triangle 1
        g.add_edge("B", "D")
        g.add_edge("D", "C")  # Triangle 2

        counts = g.count_triangles()
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        d_id = g.get_id_from_label("D")
        assert counts[a_id] == 1  # One triangle
        assert counts[b_id] == 2  # Two triangles
        assert counts[c_id] == 2  # Two triangles
        assert counts[d_id] == 1  # One triangle

    def test_count_triangles_complete_graph_k4(self):
        """Complete graph K4 has 4 triangles per node."""
        g = ec.Graph()
        # Complete graph on 4 nodes
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            g.add_node(label=node)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        counts = g.count_triangles()
        # Each node in K4 participates in C(3,2) = 3 triangles
        for node in nodes:
            node_id = g.get_id_from_label(node)
            assert counts[node_id] == 3

    @pytest.mark.parametrize("ca_edge", [("C", "A"), ("A", "C")])
    def test_count_triangles_independent_of_edge_direction(self, ca_edge):
        """Undirected counts don't depend on the order edge endpoints were given."""
        g = ec.Graph()
        for label in "ABCD":
            g.add_node(label=label)
        for source, target in [("A", "B"), ("B", "C"), ca_edge, ("C", "D"), ("A", "D")]:
            g.add_edge(source, target)

        counts = g.count_triangles()
        by_label = {label: counts[g.get_id_from_label(label)] for label in "ABCD"}
        assert by_label == {"A": 2, "B": 1, "C": 2, "D": 1}

    def test_count_triangles_parallel_edges_and_self_loops(self):
        """Parallel edges, reversed duplicates, and self-loops don't inflate counts."""
        g = ec.Graph()
        for label in "ABC":
            g.add_node(label=label)
        for source, target in [("A", "B"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "A"),
                               ("A", "A")]:
            g.add_edge(source, target)

        counts = g.count_triangles()
        assert all(count == 1 for count in counts.values())

    def test_count_triangles_matches_networkx_random_orientation(self):
        """Per-node counts match NetworkX when edges are added in random orientations."""
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(0)
        for seed in range(50):
            n = rng.randint(3, 20)
            reference = nx.gnp_random_graph(n, 0.4, seed=seed)
            g = ec.Graph()
            for i in range(n):
                g.add_node(label=str(i))
            for u, v in reference.edges():
                if rng.random() < 0.5:
                    u, v = v, u
                g.add_edge(str(u), str(v))

            counts = g.count_triangles()
            got = {i: counts[g.get_id_from_label(str(i))] for i in range(n)}
            assert got == nx.triangles(reference)

    def test_count_triangles_directed(self):
        """Test triangle counting in directed graph."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        counts = g.count_triangles()
        # In directed graph, each node should still count the cycle
        assert all(count >= 0 for count in counts.values())


class TestListTriangles:
    """Test list_triangles function."""

    def test_list_triangles_empty(self):
        """Empty graph has no triangles to list."""
        g = ec.Graph()
        triangles = g.list_triangles()
        assert len(triangles) == 0

    def test_list_triangles_single(self):
        """List single triangle."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        triangles = g.list_triangles()
        assert len(triangles) == 1

        # Check triangle contains all three nodes (as IDs)
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        tri_set = set(triangles[0])
        assert tri_set == {a_id, b_id, c_id}

    def test_list_triangles_multiple(self):
        """List multiple triangles."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Triangle 1
        g.add_edge("B", "D")
        g.add_edge("D", "C")  # Triangle 2

        triangles = g.list_triangles()
        assert len(triangles) == 2

        # Get node IDs
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")
        d_id = g.get_id_from_label("D")

        # Convert to sets for easier comparison
        tri_sets = [set(tri) for tri in triangles]
        assert {a_id, b_id, c_id} in tri_sets
        assert {b_id, c_id, d_id} in tri_sets

    def test_list_triangles_no_triangles(self):
        """Chain graph has no triangles to list."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        triangles = g.list_triangles()
        assert len(triangles) == 0


class TestTransitivity:
    """Test global transitivity (clustering coefficient)."""

    def test_transitivity_perfect_triangle(self):
        """Perfect triangle has transitivity 1.0."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        trans = g.get_transitivity()
        assert trans is not None
        assert abs(trans - 1.0) < 1e-10

    def test_transitivity_chain(self):
        """Chain graph has transitivity 0.0."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        trans = g.get_transitivity()
        assert trans is not None
        assert abs(trans - 0.0) < 1e-10

    def test_transitivity_partial(self):
        """Graph with some triangles has partial transitivity."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Triangle
        g.add_edge("A", "D")  # Extra edge creating open triple

        trans = g.get_transitivity()
        assert trans is not None
        assert 0.0 < trans < 1.0

    def test_transitivity_empty(self):
        """Empty graph returns None."""
        g = ec.Graph()
        trans = g.get_transitivity()
        assert trans is None

    def test_transitivity_no_triples(self):
        """Graph with no connected triples returns None."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")

        trans = g.get_transitivity()
        assert trans is None


class TestLocalClustering:
    """Test local clustering coefficient."""

    def test_local_clustering_triangle(self):
        """Triangle has clustering coefficient of 1.0 for all nodes."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        clustering = g.local_clustering_coefficient(return_dict=True)
        node_a_id = g.get_id_from_label("A")
        node_b_id = g.get_id_from_label("B")
        node_c_id = g.get_id_from_label("C")
        assert abs(clustering[node_a_id] - 1.0) < 1e-10
        assert abs(clustering[node_b_id] - 1.0) < 1e-10
        assert abs(clustering[node_c_id] - 1.0) < 1e-10

    def test_local_clustering_chain(self):
        """Middle node of chain has clustering 0.0."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        clustering = g.local_clustering_coefficient(return_dict=True)
        node_b_id = g.get_id_from_label("B")
        assert clustering[node_b_id] is not None
        assert abs(clustering[node_b_id] - 0.0) < 1e-10

    def test_local_clustering_partial(self):
        """Node with some connected neighbors has partial clustering."""
        g = ec.Graph()
        # A connected to B, C, D
        # B and C are connected (triangle A-B-C)
        # D is not connected to B or C
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Triangle
        g.add_edge("A", "D")  # Extra connection

        clustering = g.local_clustering_coefficient(return_dict=True)
        # A has 3 neighbors, 1 triangle out of C(3,2)=3 possible
        # So clustering = 1/3
        node_a_id = g.get_id_from_label("A")
        assert clustering[node_a_id] is not None
        expected = 1.0 / 3.0
        assert abs(clustering[node_a_id] - expected) < 1e-10

    def test_local_clustering_degree_zero(self):
        """Isolated node has None clustering."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("B", "C")

        clustering = g.local_clustering_coefficient(return_dict=True)
        node_a_id = g.get_id_from_label("A")
        assert clustering[node_a_id] is None

    def test_local_clustering_degree_one(self):
        """Node with degree 1 has None clustering."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        clustering = g.local_clustering_coefficient(return_dict=True)
        node_a_id = g.get_id_from_label("A")
        node_b_id = g.get_id_from_label("B")
        node_c_id = g.get_id_from_label("C")
        assert clustering[node_a_id] is None  # Degree 1
        assert clustering[node_b_id] is not None  # Degree 2
        assert clustering[node_c_id] is None  # Degree 1

    def test_local_clustering_star_graph(self):
        """Hub of star graph has clustering 0.0."""
        g = ec.Graph()
        # Central hub connected to 4 leaves
        g.add_node(label="Hub")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("Hub", "A")
        g.add_edge("Hub", "B")
        g.add_edge("Hub", "C")
        g.add_edge("Hub", "D")

        clustering = g.local_clustering_coefficient(return_dict=True)
        hub_id = g.get_id_from_label("Hub")
        assert clustering[hub_id] is not None
        assert abs(clustering[hub_id] - 0.0) < 1e-10

        # Leaves have degree 1, so None
        for leaf in ["A", "B", "C", "D"]:
            leaf_id = g.get_id_from_label(leaf)
            assert clustering[leaf_id] is None

    def test_local_clustering_complete_graph(self):
        """Complete graph has clustering 1.0 for all nodes."""
        g = ec.Graph()
        # Complete graph K4
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            g.add_node(label=node)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        clustering = g.local_clustering_coefficient(return_dict=True)
        for node in nodes:
            node_id = g.get_id_from_label(node)
            assert abs(clustering[node_id] - 1.0) < 1e-10


class TestDirectedGraphs:
    """Test triangle functions on directed graphs."""

    def test_directed_triangle_counting(self):
        """Test counting directed triangles."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        counts = g.count_triangles()
        # Should detect directed cycle
        assert all(count > 0 for count in counts.values())

    def test_directed_partial_triangle(self):
        """Incomplete directed cycle has no triangle."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        # Missing C -> A

        counts = g.count_triangles()
        assert all(count == 0 for count in counts.values())


class TestDirectedTransitivity:
    """Directed transitivity: closed fraction of directed 2-paths through distinct nodes."""

    @staticmethod
    def _build(n, edges):
        g = ec.Graph(directed=True)
        for i in range(n):
            g.add_node(label=str(i))
        for u, v in edges:
            g.add_edge(str(u), str(v))
        return g

    def test_reciprocated_triangle_is_fully_transitive(self):
        """Paths that return to their start (u -> v -> u) aren't triples."""
        edges = [(u, v) for u in range(3) for v in range(3) if u != v]
        assert self._build(3, edges).get_transitivity() == pytest.approx(1.0)

    def test_reciprocal_pair_has_no_triples(self):
        """A lone A <-> B pair has no triple through three distinct nodes."""
        assert self._build(2, [(0, 1), (1, 0)]).get_transitivity() is None

    def test_matches_brute_force(self):
        """Matches a direct count of closed / all distinct-node directed 2-paths."""
        import random

        rng = random.Random(3)
        for _ in range(50):
            n = rng.randint(3, 10)
            edges = {
                (u, v) for u in range(n) for v in range(n) if u != v and rng.random() < 0.35
            }
            paths = [(u, v, w) for (u, v) in edges for (x, w) in edges if x == v and w != u]
            expected = (
                sum((w, u) in edges for u, v, w in paths) / len(paths) if paths else None
            )

            result = self._build(n, sorted(edges)).get_transitivity()
            if expected is None:
                assert result is None
            else:
                assert result == pytest.approx(expected)


class TestParallelEdgesAndSelfLoops:
    """Triangle-based measures are computed on the simple graph."""

    BASE = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]

    def _build(self, directed, extra=()):
        g = ec.Graph(directed=directed)
        for label in "ABCD":
            g.add_node(label=label)
        for source, target in [*self.BASE, *extra]:
            g.add_edge(source, target)
        return g

    @pytest.mark.parametrize(
        "directed, extra",
        [
            (False, [("A", "B"), ("B", "A"), ("C", "C"), ("D", "C")]),
            (True, [("A", "B"), ("B", "C"), ("C", "C")]),
        ],
    )
    def test_multigraph_matches_simple_graph(self, directed, extra):
        """Parallel edges, reversed duplicates, and self-loops don't change any measure."""
        simple = self._build(directed)
        multi = self._build(directed, extra)

        assert multi.count_triangles() == simple.count_triangles()
        assert sorted(multi.list_triangles()) == sorted(simple.list_triangles())
        assert multi.get_transitivity() == pytest.approx(simple.get_transitivity())
        assert multi.clustering_coefficient(return_dict=True) == pytest.approx(
            simple.clustering_coefficient(return_dict=True)
        )
        assert multi.get_avg_clustering_coefficient() == pytest.approx(
            simple.get_avg_clustering_coefficient()
        )

    def test_undirected_values_with_parallel_edges(self):
        """A doubled edge keeps the triangle-with-tail values of the simple graph."""
        g = self._build(False, [("A", "B"), ("B", "A"), ("C", "C")])
        ids = {label: g.get_id_from_label(label) for label in "ABCD"}

        assert g.get_transitivity() == pytest.approx(0.6)
        local = g.clustering_coefficient(return_dict=True)
        assert local[ids["A"]] == pytest.approx(1.0)
        assert local[ids["C"]] == pytest.approx(1 / 3)
        assert len(g.list_triangles()) == 1

    def test_matches_networkx_with_random_parallel_edges(self):
        """Transitivity and local clustering match NetworkX on the underlying simple graph."""
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(0)
        for seed in range(30):
            n = rng.randint(3, 20)
            reference = nx.gnp_random_graph(n, 0.4, seed=seed)
            g = ec.Graph()
            for i in range(n):
                g.add_node(label=str(i))
            for u, v in reference.edges():
                g.add_edge(str(u), str(v))
                if rng.random() < 0.3:
                    g.add_edge(str(v), str(u))
            for i in range(n):
                if rng.random() < 0.1:
                    g.add_edge(str(i), str(i))

            assert (g.get_transitivity() or 0.0) == pytest.approx(nx.transitivity(reference))
            local = g.clustering_coefficient(return_dict=True)
            expected = nx.clustering(reference)
            for i in range(n):
                assert (local[g.get_id_from_label(str(i))] or 0.0) == pytest.approx(expected[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Square clustering and generalized degree, validated against NetworkX.


def _cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G0 = nx.gnp_random_graph(
                14, rnd.choice([0.15, 0.25, 0.4]), seed=seed, directed=directed
            )
            G = (nx.DiGraph if directed else nx.Graph)()
            G.add_nodes_from(range(14))
            for u, v in G0.edges:
                G.add_edge(u, v, weight=rnd.choice([0.5, 1, 2, 3]))
            g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(14)])
            gw = ec.Graph(directed=directed, weighted=True).add_nodes([str(v) for v in range(14)])
            for u, v, d in G.edges(data=True):
                g.add_edge(str(u), str(v))
                gw.add_edge(str(u), str(v), weight=d["weight"])
            yield G, g, gw, rnd


def _two_triangles():
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _lab(g, nid):
    return int(g.get_label_from_id(nid))


class TestSquareClusteringGeneralizedDegree:
    def test_square_clustering_and_generalized_degree(self):
        for G, g, _, _ in _cases():
            ours = {_lab(g, k): v for k, v in g.square_clustering(return_dict=True).items()}
            assert ours == pytest.approx(nx.square_clustering(G))
            if not G.is_directed():
                gd = {_lab(g, k): v for k, v in g.generalized_degree().items()}
                assert gd == {k: dict(v) for k, v in nx.generalized_degree(G).items()}

    def test_examples(self):
        square = ec.Graph().add_cycle_graph(4)
        assert set(square.square_clustering(return_dict=True).values()) == {1.0}
        g = (
            ec.Graph()
            .add_nodes(list("ABCD"))
            .add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
        )
        assert g.generalized_degree()[g.get_id_from_label("C")] == {0: 1, 1: 2}
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).generalized_degree()

    def test_steps(self):
        g = _two_triangles()
        g.square_clustering()
        assert g.steps[-1].kind == KIND_METRIC
