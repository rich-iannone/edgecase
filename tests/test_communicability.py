import math
import pytest
from edgecase import Graph


class TestCommunicability:
    """Tests for communicability() method."""

    def test_empty_graph(self):
        """Test communicability on empty graph."""
        graph = Graph(directed=False)

        result = graph.communicability()

        assert result == {}

    def test_single_node(self):
        """Test communicability on single isolated node."""
        graph = Graph(directed=False)
        graph.add_node(label="A")

        result = graph.communicability()
        node_id = graph.get_id_from_label("A")

        # Self-communicability of isolated node = exp(0) = 1
        assert abs(result[node_id][node_id] - 1.0) < 1e-10

    def test_two_connected_nodes(self):
        """Test communicability on two connected nodes."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        result = graph.communicability()
        id_a = graph.get_id_from_label("A")
        id_b = graph.get_id_from_label("B")

        # For two connected nodes with adjacency [[0,1],[1,0]]:
        # eigenvalues: 1, -1
        # exp(A)_11 = exp(A)_22 = cosh(1)
        # exp(A)_12 = exp(A)_21 = sinh(1)
        assert abs(result[id_a][id_a] - math.cosh(1)) < 1e-10
        assert abs(result[id_b][id_b] - math.cosh(1)) < 1e-10
        assert abs(result[id_a][id_b] - math.sinh(1)) < 1e-10
        assert abs(result[id_b][id_a] - math.sinh(1)) < 1e-10

    def test_symmetric(self):
        """Test that communicability is symmetric."""
        graph = Graph(directed=False)
        graph.add_path_graph(5)

        result = graph.communicability()
        node_ids = graph.get_node_ids()

        for u in node_ids:
            for v in node_ids:
                assert abs(result[u][v] - result[v][u]) < 1e-10

    def test_positive_values(self):
        """Test that all communicability values are positive."""
        graph = Graph(directed=False)
        graph.add_star_graph(5)

        result = graph.communicability()

        for u, inner in result.items():
            for v, comm in inner.items():
                assert comm > 0, f"Communicability({u}, {v}) should be positive"

    def test_adjacent_higher(self):
        """Test that adjacent nodes have higher communicability than non-adjacent."""
        graph = Graph(directed=False)
        graph.add_node(label="1")
        graph.add_node(label="2")
        graph.add_node(label="3")
        graph.add_edge("1", "2")
        graph.add_edge("2", "3")

        result = graph.communicability()
        id1 = graph.get_id_from_label("1")
        id2 = graph.get_id_from_label("2")
        id3 = graph.get_id_from_label("3")

        # Adjacent pair (1,2) should have higher communicability than (1,3)
        assert result[id1][id2] > result[id1][id3]

    def test_triangle(self):
        """Test communicability on a triangle."""
        graph = Graph(directed=False)
        for i in range(1, 4):
            graph.add_node(label=str(i))
        graph.add_edge("1", "2")
        graph.add_edge("2", "3")
        graph.add_edge("3", "1")

        result = graph.communicability()
        ids = [graph.get_id_from_label(str(i)) for i in range(1, 4)]

        # In a triangle, all nodes should have equal communicability with each other
        comm_12 = result[ids[0]][ids[1]]
        comm_23 = result[ids[1]][ids[2]]
        comm_31 = result[ids[2]][ids[0]]

        assert abs(comm_12 - comm_23) < 1e-10
        assert abs(comm_23 - comm_31) < 1e-10

    def test_directed_error(self):
        """Test that directed graphs raise error."""
        graph = Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        with pytest.raises(ValueError, match="undirected"):
            graph.communicability()

    def test_self_loop_error(self):
        """Test that graphs with self-loops raise error."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_edge("A", "A")  # Self-loop

        with pytest.raises(ValueError, match="self-loops"):
            graph.communicability()

    def test_disconnected_pairs_are_exactly_zero(self):
        """No walk joins two components, so their communicability is exactly 0 (not noise)."""
        graph = Graph(directed=False)
        graph.add_nodes(["a", "b", "c", "x", "y"])
        graph.add_edges([("a", "b"), ("b", "c"), ("x", "y")])

        result = graph.communicability()
        ids = {label: graph.get_id_from_label(label) for label in "abcxy"}

        for u in "abc":
            for v in "xy":
                assert result[ids[u]][ids[v]] == 0.0
                assert result[ids[v]][ids[u]] == 0.0


class TestCommunicabilityBetweennessCentrality:
    """Tests for communicability_betweenness_centrality() method."""

    def test_empty_graph(self):
        """Test on empty graph."""
        graph = Graph(directed=False)
        result = graph.communicability_betweenness_centrality(return_dict=True)

        assert result == {}

    def test_single_node(self):
        """Test on single node."""
        graph = Graph(directed=False)
        graph.add_node(label="A")

        result = graph.communicability_betweenness_centrality(return_dict=True)
        node_id = graph.get_id_from_label("A")

        assert result[node_id] == 0.0

    def test_two_nodes(self):
        """Test on two connected nodes."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        result = graph.communicability_betweenness_centrality(return_dict=True)

        # With only two nodes, neither is "between" the other
        for value in result.values():
            assert value >= 0.0
            assert value <= 1.0

    def test_path_graph_middle_highest(self):
        """Test that middle node has highest betweenness in path graph."""
        graph = Graph(directed=False)

        for i in range(1, 6):
            graph.add_node(label=str(i))

        for i in range(1, 5):
            graph.add_edge(str(i), str(i + 1))

        result = graph.communicability_betweenness_centrality(return_dict=True)

        # Middle node (3) should have highest betweenness
        id3 = graph.get_id_from_label("3")
        id1 = graph.get_id_from_label("1")
        id5 = graph.get_id_from_label("5")

        assert result[id3] > result[id1]
        assert result[id3] > result[id5]

    def test_star_center_highest(self):
        """Test that center node has highest betweenness in star graph."""
        graph = Graph(directed=False)
        graph.add_node(label="center")

        for i in range(4):
            graph.add_node(label=f"leaf_{i}")
            graph.add_edge("center", f"leaf_{i}")

        result = graph.communicability_betweenness_centrality(return_dict=True)

        center_id = graph.get_id_from_label("center")
        center_cbc = result[center_id]

        for i in range(4):
            leaf_id = graph.get_id_from_label(f"leaf_{i}")
            assert center_cbc >= result[leaf_id]

    def test_normalized_range(self):
        """Test that values are in [0, 1]."""
        graph = Graph(directed=False)
        graph.add_cycle_graph(6)

        result = graph.communicability_betweenness_centrality(return_dict=True)

        for value in result.values():
            assert 0.0 <= value <= 1.0

    def test_directed_error(self):
        """Test that directed graphs raise error."""
        graph = Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        with pytest.raises(ValueError, match="undirected"):
            graph.communicability_betweenness_centrality(return_dict=True)

    def test_disconnected_graph_symmetric_and_per_component(self):
        """A second component must not add noise: pairs across components count 0."""
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("x", "y"), ("y", "z")]
        graph = Graph(directed=False)
        graph.add_nodes(["a", "b", "c", "d", "e", "x", "y", "z"])
        graph.add_edges(edges)

        result = graph.communicability_betweenness_centrality(return_dict=True)
        cbc = {graph.get_label_from_id(n): v for n, v in result.items()}

        # The two ends of the path x-y-z are symmetric, as are a and b
        assert cbc["x"] == pytest.approx(cbc["z"], abs=1e-12)
        assert cbc["a"] == pytest.approx(cbc["b"], abs=1e-12)

        # Each component scores as it would alone, rescaled from its own normalization
        # (m - 1)^2 - (m - 1) to the whole graph's (n - 1)^2 - (n - 1)
        n = 8
        for labels in (["a", "b", "c", "d", "e"], ["x", "y", "z"]):
            alone = Graph(directed=False)
            alone.add_nodes(labels)
            alone.add_edges([e for e in edges if e[0] in labels])
            ref = alone.communicability_betweenness_centrality(return_dict=True)
            m = len(labels)
            scale = ((m - 1) ** 2 - (m - 1)) / ((n - 1) ** 2 - (n - 1))
            for nid, value in ref.items():
                label = alone.get_label_from_id(nid)
                assert cbc[label] == pytest.approx(value * scale, abs=1e-12)

    def test_return_self_for_chaining(self):
        """Test that return_dict=False returns self for chaining."""
        graph = Graph(directed=False)
        graph.add_path_graph(4)

        result = graph.communicability_betweenness_centrality(return_dict=False)
        assert result is graph

    def test_stores_attribute(self):
        """Test that result is stored as node attribute."""
        graph = Graph(directed=False)
        graph.add_path_graph(4)

        graph.communicability_betweenness_centrality(attr_name="cbc")

        # Verify computation was tracked
        assert any("communicability_betweenness" in str(c) for c in graph._computations)

    def test_custom_attr_name(self):
        """Test custom attribute name."""
        graph = Graph(directed=False)
        graph.add_path_graph(4)

        graph.communicability_betweenness_centrality(attr_name="my_cbc")

        # Verify computation was tracked with custom name
        assert any("my_cbc" in str(c) for c in graph._computations)


class TestCommunicabilityComparison:
    """Tests comparing communicability with expected mathematical properties."""

    def test_complete_graph_uniform(self):
        """In a complete graph, all pairs should have similar communicability."""
        graph = Graph(directed=False)
        graph.add_full_graph(4)

        result = graph.communicability()
        node_ids = list(graph.get_node_ids())

        # All off-diagonal entries should be equal
        off_diag_values = []
        for i, u in enumerate(node_ids):
            for j, v in enumerate(node_ids):
                if i != j:
                    off_diag_values.append(result[u][v])

        # Check all off-diagonal values are the same
        first_val = off_diag_values[0]
        for val in off_diag_values:
            assert abs(val - first_val) < 1e-8

    def test_cycle_graph_symmetric(self):
        """In a cycle, opposite nodes should have same communicability."""
        graph = Graph(directed=False)
        graph.add_cycle_graph(6)

        result = graph.communicability()
        node_ids = list(graph.get_node_ids())

        # Node 0 with node 3 should equal node 1 with node 4 (opposite pairs)
        comm_03 = result[node_ids[0]][node_ids[3]]
        comm_14 = result[node_ids[1]][node_ids[4]]
        comm_25 = result[node_ids[2]][node_ids[5]]

        assert abs(comm_03 - comm_14) < 1e-8
        assert abs(comm_14 - comm_25) < 1e-8

    def test_hub_centrality(self):
        """Test that hub nodes have high communicability betweenness."""
        # Create a barbell graph: two cliques connected by a bridge
        graph = Graph(directed=False)

        # First clique: nodes 1-4
        for i in range(1, 5):
            graph.add_node(label=str(i))
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_edge(str(i), str(j))

        # Second clique: nodes 5-8
        for i in range(5, 9):
            graph.add_node(label=str(i))
        for i in range(5, 9):
            for j in range(i + 1, 9):
                graph.add_edge(str(i), str(j))

        # Bridge: connect node 4 to node 5
        graph.add_edge("4", "5")

        result = graph.communicability_betweenness_centrality(return_dict=True)

        # Bridge nodes (4 and 5) should have higher betweenness
        id4 = graph.get_id_from_label("4")
        id5 = graph.get_id_from_label("5")
        id1 = graph.get_id_from_label("1")
        id8 = graph.get_id_from_label("8")

        assert result[id4] > result[id1]
        assert result[id5] > result[id8]


class TestDocstringExamples:
    """Test the examples from the docstrings."""

    def test_communicability_example_1(self):
        """Test first docstring example for communicability."""
        graph = Graph(directed=False)
        graph.add_node(label="1")
        graph.add_node(label="2")
        graph.add_node(label="3")
        graph.add_edge("1", "2")
        graph.add_edge("2", "3")

        comm = graph.communicability()
        id1 = graph.get_id_from_label("1")
        id2 = graph.get_id_from_label("2")
        id3 = graph.get_id_from_label("3")

        # Adjacent nodes have higher communicability
        assert comm[id1][id2] > comm[id1][id3]

    def test_communicability_example_2(self):
        """Test second docstring example for communicability."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("B", "D")

        comm = graph.communicability()

        # Node B is a hub - check its communicability with others
        b_id = graph.get_id_from_label("B")
        total_comm = sum(comm[b_id].values())

        assert total_comm > 0

    def test_cbc_example_1(self):
        """Test first docstring example for communicability_betweenness_centrality."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")

        cbc = graph.communicability_betweenness_centrality(return_dict=True)

        # Middle nodes B and C have higher betweenness
        id_b = graph.get_id_from_label("B")
        id_c = graph.get_id_from_label("C")
        id_a = graph.get_id_from_label("A")
        id_d = graph.get_id_from_label("D")

        assert cbc[id_b] > cbc[id_a]
        assert cbc[id_c] > cbc[id_d]

    def test_cbc_example_star(self):
        """Test star graph example for communicability_betweenness_centrality."""
        star = Graph(directed=False)
        star.add_node(label="center")

        for i in range(4):
            star.add_node(label=f"leaf_{i}")
            star.add_edge("center", f"leaf_{i}")

        cbc = star.communicability_betweenness_centrality(return_dict=True)

        # Center node should have highest communicability betweenness
        center_id = star.get_id_from_label("center")

        # Should be close to 1.0 for a star
        assert cbc[center_id] > 0.9
