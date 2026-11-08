import pytest
from edgecase import Graph


class TestAverageNeighborDegreeBasic:
    """Basic functionality tests for average_neighbor_degree."""

    def test_path_graph_undirected(self):
        """Test average neighbor degree on a path graph."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i + 1))

        and_values = g.average_neighbor_degree(return_dict=True)

        # Node 0: neighbor 1 has degree 2 -> average = 2.0
        node_0 = g.get_id_from_label("0")

        assert abs(and_values[node_0] - 2.0) < 1e-6

        # Node 1: neighbors 0,2 have degrees 1,2 -> average = 1.5
        node_1 = g.get_id_from_label("1")

        assert abs(and_values[node_1] - 1.5) < 1e-6

        # Node 2: neighbors 1,3 have degrees 2,2 -> average = 2.0
        node_2 = g.get_id_from_label("2")

        assert abs(and_values[node_2] - 2.0) < 1e-6

        # Node 3: neighbors 2,4 have degrees 2,1 -> average = 1.5
        node_3 = g.get_id_from_label("3")

        assert abs(and_values[node_3] - 1.5) < 1e-6

        # Node 4: neighbor 3 has degree 2 -> average = 2.0
        node_4 = g.get_id_from_label("4")

        assert abs(and_values[node_4] - 2.0) < 1e-6

    def test_star_graph_undirected(self):
        """Test average neighbor degree on a star graph."""
        g = Graph(directed=False)
        g.add_node(label="hub")
        for i in range(4):
            g.add_node(label=f"leaf_{i}")
            g.add_edge("hub", f"leaf_{i}")

        and_values = g.average_neighbor_degree(return_dict=True)

        hub_id = g.get_id_from_label("hub")

        # Hub has 4 neighbors, each with degree 1 -> average = 1.0
        assert abs(and_values[hub_id] - 1.0) < 1e-6

        # Each leaf has 1 neighbor (hub) with degree 4 -> average = 4.0
        for i in range(4):
            leaf_id = g.get_id_from_label(f"leaf_{i}")
            assert abs(and_values[leaf_id] - 4.0) < 1e-6

    def test_complete_graph_undirected(self):
        """Test average neighbor degree on a complete graph."""
        g = Graph(directed=False)
        n = 5
        for i in range(n):
            g.add_node(label=str(i))

        # Create complete graph K5
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(str(i), str(j))

        and_values = g.average_neighbor_degree(return_dict=True)

        # In K5, every node has degree 4, and all neighbors have degree 4
        # So average neighbor degree = 4.0 for all nodes
        for i in range(n):
            node_id = g.get_id_from_label(str(i))
            assert abs(and_values[node_id] - 4.0) < 1e-6

    def test_directed_cycle(self):
        """Test average neighbor degree on a directed cycle."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Default: source="out", target="out"
        # Each node has 1 out-neighbor with out-degree 1
        and_out_out = g.average_neighbor_degree(
            source_mode="out", target_mode="out", return_dict=True
        )
        for label in ["A", "B", "C"]:
            node_id = g.get_id_from_label(label)
            assert abs(and_out_out[node_id] - 1.0) < 1e-6

        # source="in", target="in"
        # Each node has 1 in-neighbor with in-degree 1
        and_in_in = g.average_neighbor_degree(source_mode="in", target_mode="in", return_dict=True)
        for label in ["A", "B", "C"]:
            node_id = g.get_id_from_label(label)
            assert abs(and_in_in[node_id] - 1.0) < 1e-6

    def test_directed_star(self):
        """Test average neighbor degree on a directed star graph."""
        g = Graph(directed=True)
        g.add_node(label="center")
        for i in range(3):
            g.add_node(label=f"leaf_{i}")
            g.add_edge("center", f"leaf_{i}")

        # Out-neighbors of center have out-degree 0
        and_out_out = g.average_neighbor_degree(
            source_mode="out", target_mode="out", return_dict=True
        )
        center_id = g.get_id_from_label("center")
        assert abs(and_out_out[center_id] - 0.0) < 1e-6

        # In-neighbors of leaves have in-degree 0
        and_in_in = g.average_neighbor_degree(source_mode="in", target_mode="in", return_dict=True)
        for i in range(3):
            leaf_id = g.get_id_from_label(f"leaf_{i}")
            assert abs(and_in_in[leaf_id] - 0.0) < 1e-6


class TestAverageNeighborDegreeEdgeCases:
    """Edge case tests for average_neighbor_degree."""

    def test_empty_graph(self):
        """Test on empty graph."""
        g = Graph(directed=False)
        with pytest.raises(ValueError, match="empty graph"):
            g.average_neighbor_degree(return_dict=True)

    def test_single_node(self):
        """Test on graph with single node (no neighbors)."""
        g = Graph(directed=False)
        g.add_node(label="A")
        and_values = g.average_neighbor_degree(return_dict=True)

        node_id = g.get_id_from_label("A")

        # Node with no neighbors should have average = 0
        assert abs(and_values[node_id] - 0.0) < 1e-6

    def test_isolated_nodes(self):
        """Test graph with isolated nodes."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")

        and_values = g.average_neighbor_degree(return_dict=True)

        # A and B have neighbors with degree 1
        node_a = g.get_id_from_label("A")
        node_b = g.get_id_from_label("B")

        assert abs(and_values[node_a] - 1.0) < 1e-6
        assert abs(and_values[node_b] - 1.0) < 1e-6

        # C is isolated, has no neighbors
        node_c = g.get_id_from_label("C")

        assert abs(and_values[node_c] - 0.0) < 1e-6

    def test_two_nodes_no_edge(self):
        """Test two nodes with no edge."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")

        and_values = g.average_neighbor_degree(return_dict=True)

        # Both nodes have no neighbors
        node_a = g.get_id_from_label("A")
        node_b = g.get_id_from_label("B")

        assert abs(and_values[node_a] - 0.0) < 1e-6
        assert abs(and_values[node_b] - 0.0) < 1e-6


class TestAverageNeighborDegreeDirected:
    """Tests for directed graphs with different source/target combinations."""

    def test_directed_source_target_combinations(self):
        """Test all combinations of source/target parameters."""
        # Create a simple directed graph
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        # Test source="out", target="out"
        out_out = g.average_neighbor_degree(source_mode="out", target_mode="out", return_dict=True)

        assert len(out_out) == 4

        # Test source="in", target="in"
        in_in = g.average_neighbor_degree(source_mode="in", target_mode="in", return_dict=True)

        assert len(in_in) == 4

        # Test source="in+out", target="in+out"
        inout_inout = g.average_neighbor_degree(
            source_mode="in+out", target_mode="in+out", return_dict=True
        )

        assert len(inout_inout) == 4

    def test_directed_mixed_source_target(self):
        """Test mixed source and target parameters."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # source="out" (successors), target="in" (measure their in-degree)
        out_in = g.average_neighbor_degree(source_mode="out", target_mode="in", return_dict=True)

        assert len(out_in) == 3

        # source="in" (predecessors), target="out" (measure their out-degree)
        in_out = g.average_neighbor_degree(source_mode="in", target_mode="out", return_dict=True)

        assert len(in_out) == 3


class TestAverageNeighborDegreeAPI:
    """Tests for API behavior (chaining, attributes, return types)."""

    def test_return_dict_true(self):
        """Test return_dict=True returns dictionary."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        result = g.average_neighbor_degree(return_dict=True)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_return_dict_false_returns_self(self):
        """Test return_dict=False returns self for chaining."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        result = g.average_neighbor_degree(return_dict=False)

        assert result is g

    def test_method_chaining(self):
        """Test method chaining works."""
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .average_neighbor_degree(attr_name="knn")
        )

        assert g is not None

        # Verify computation was tracked
        assert any("average_neighbor_degree" in str(c) for c in g._computations)

    def test_custom_attr_name(self):
        """Test custom attribute name."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        g.average_neighbor_degree(attr_name="custom_knn")

        # Verify computation was tracked with custom name
        computations = [c for c in g._computations if c[0] == "average_neighbor_degree"]

        assert len(computations) == 1
        assert computations[0][2] == "custom_knn"

    def test_default_attr_name(self):
        """Test default attribute name."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        g.average_neighbor_degree()

        # Verify computation was tracked with default name
        computations = [c for c in g._computations if c[0] == "average_neighbor_degree"]

        assert len(computations) == 1
        assert computations[0][2] == "average_neighbor_degree"


class TestAverageNeighborDegreeErrors:
    """Tests for error handling."""

    def test_weighted_not_implemented(self):
        """Test that weighted version raises error."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(Exception):
            g.average_neighbor_degree(weight="weight", return_dict=True)

    def test_invalid_source_parameter(self):
        """Test invalid source parameter raises error."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(Exception):
            g.average_neighbor_degree(source_mode="invalid", return_dict=True)

    def test_invalid_target_parameter(self):
        """Test invalid target parameter raises error."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(Exception):
            g.average_neighbor_degree(target_mode="invalid", return_dict=True)


class TestAverageNeighborDegreeNetworkX:
    """Validation tests against NetworkX implementation."""

    def test_validation_undirected_path(self):
        """Validate against NetworkX for path graph."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")

        # Create EdgeCase graph
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i + 1))

        ec_and = g.average_neighbor_degree(return_dict=True)

        # Create NetworkX graph
        nx_g = nx.path_graph(5)
        nx_and = nx.average_neighbor_degree(nx_g)

        # Compare results
        for i in range(5):
            ec_node_id = g.get_id_from_label(str(i))
            # NetworkX uses node labels directly
            assert abs(ec_and[ec_node_id] - nx_and[i]) < 1e-6

    def test_validation_undirected_star(self):
        """Validate against NetworkX for star graph."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")

        # Create EdgeCase graph
        g = Graph(directed=False)
        g.add_node(label="0")
        for i in range(1, 5):
            g.add_node(label=str(i))
            g.add_edge("0", str(i))

        ec_and = g.average_neighbor_degree(return_dict=True)

        # Create NetworkX graph
        nx_g = nx.star_graph(4)
        nx_and = nx.average_neighbor_degree(nx_g)

        # Compare results
        for i in range(5):
            ec_node_id = g.get_id_from_label(str(i))
            assert abs(ec_and[ec_node_id] - nx_and[i]) < 1e-6

    def test_validation_directed_cycle(self):
        """Validate against NetworkX for directed cycle."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")

        # Create EdgeCase graph
        g = Graph(directed=True)
        for i in range(4):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str((i + 1) % 4))

        ec_and = g.average_neighbor_degree(source_mode="out", target_mode="out", return_dict=True)

        # Create NetworkX graph
        nx_g = nx.DiGraph()
        nx_g.add_edges_from([(i, (i + 1) % 4) for i in range(4)])
        nx_and = nx.average_neighbor_degree(nx_g, source="out", target="out")

        # Compare results
        for i in range(4):
            ec_node_id = g.get_id_from_label(str(i))
            assert abs(ec_and[ec_node_id] - nx_and[i]) < 1e-6

    def test_validation_directed_source_in(self):
        """Validate against NetworkX with source='in'."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")

        # Create EdgeCase graph
        g = Graph(directed=True)
        for i in range(4):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("0", "2")
        g.add_edge("1", "3")
        g.add_edge("2", "3")

        ec_and = g.average_neighbor_degree(source_mode="in", target_mode="in", return_dict=True)

        # Create NetworkX graph
        nx_g = nx.DiGraph()
        nx_g.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("2", "3")])
        nx_and = nx.average_neighbor_degree(nx_g, source="in", target="in")

        # Compare results (convert labels back)
        for i in range(4):
            ec_node_id = g.get_id_from_label(str(i))
            assert abs(ec_and[ec_node_id] - nx_and[str(i)]) < 1e-6

    def test_validation_complete_graph(self):
        """Validate against NetworkX for complete graph."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")

        n = 6
        # Create EdgeCase graph
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(label=str(i))
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(str(i), str(j))

        ec_and = g.average_neighbor_degree(return_dict=True)

        # Create NetworkX graph
        nx_g = nx.complete_graph(n)
        nx_and = nx.average_neighbor_degree(nx_g)

        # Compare results
        for i in range(n):
            ec_node_id = g.get_id_from_label(str(i))
            assert abs(ec_and[ec_node_id] - nx_and[i]) < 1e-6


class TestAverageNeighborDegreeSpecialCases:
    """Tests for special graph structures."""

    def test_disconnected_components(self):
        """Test graph with multiple disconnected components."""
        g = Graph(directed=False)

        # Component 1: triangle
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Component 2: edge
        g.add_node(label="D")
        g.add_node(label="E")
        g.add_edge("D", "E")

        and_values = g.average_neighbor_degree(return_dict=True)

        # Triangle nodes: each has 2 neighbors with degree 2
        for label in ["A", "B", "C"]:
            node_id = g.get_id_from_label(label)
            assert abs(and_values[node_id] - 2.0) < 1e-6

        # Edge nodes: each has 1 neighbor with degree 1
        for label in ["D", "E"]:
            node_id = g.get_id_from_label(label)
            assert abs(and_values[node_id] - 1.0) < 1e-6

    def test_self_loops(self):
        """Test graph with self-loops."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # Self-loop

        and_values = g.average_neighbor_degree(return_dict=True)

        # Should handle self-loops appropriately
        assert len(and_values) == 2

    def test_bipartite_graph(self):
        """Test on bipartite graph."""
        g = Graph(directed=False)
        # Set 1
        for i in range(3):
            g.add_node(label=f"U{i}")
        # Set 2
        for i in range(2):
            g.add_node(label=f"V{i}")

        # Connect all U to all V
        for i in range(3):
            for j in range(2):
                g.add_edge(f"U{i}", f"V{j}")

        and_values = g.average_neighbor_degree(return_dict=True)

        # Each U node has 2 neighbors (all V nodes), each V has degree 3
        for i in range(3):
            u_id = g.get_id_from_label(f"U{i}")
            assert abs(and_values[u_id] - 3.0) < 1e-6

        # Each V node has 3 neighbors (all U nodes), each U has degree 2
        for i in range(2):
            v_id = g.get_id_from_label(f"V{i}")
            assert abs(and_values[v_id] - 2.0) < 1e-6
