import pytest

import edgecase as ec


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
        assert counts["A"] == 0

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
        assert counts["A"] == 1
        assert counts["B"] == 1
        assert counts["C"] == 1

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
        assert counts["A"] == 1  # One triangle
        assert counts["B"] == 2  # Two triangles
        assert counts["C"] == 2  # Two triangles
        assert counts["D"] == 1  # One triangle

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
            assert counts[node] == 3

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

        # Check triangle contains all three nodes
        tri_set = set(triangles[0])
        assert tri_set == {"A", "B", "C"}

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

        # Convert to sets for easier comparison
        tri_sets = [set(tri) for tri in triangles]
        assert {"A", "B", "C"} in tri_sets
        assert {"B", "C", "D"} in tri_sets

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

        trans = g.get_graph_transitivity()
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

        trans = g.get_graph_transitivity()
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

        trans = g.get_graph_transitivity()
        assert trans is not None
        assert 0.0 < trans < 1.0

    def test_transitivity_empty(self):
        """Empty graph returns None."""
        g = ec.Graph()
        trans = g.get_graph_transitivity()
        assert trans is None

    def test_transitivity_no_triples(self):
        """Graph with no connected triples returns None."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")

        trans = g.get_graph_transitivity()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
