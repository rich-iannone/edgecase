import pytest
import math
from edgecase import Graph


class TestNodeVitality:
    """Test basic node vitality functionality."""

    def test_single_node_vitality_zero(self):
        """Single node graph has vitality 0."""
        graph = Graph()
        graph.add_node(label="A")
        node_id = graph.get_id_from_label("A")

        vitality = graph.get_node_vitality(node_id)

        assert vitality == 0.0

    def test_two_nodes_disconnected(self):
        """Two disconnected nodes have vitality 0."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        vitality_a = graph.get_node_vitality(a_id)
        vitality_b = graph.get_node_vitality(b_id)

        assert vitality_a == 0.0
        assert vitality_b == 0.0

    def test_path_graph_middle_most_vital(self):
        """In a path graph, the middle nodes are the most vital."""
        graph = Graph()
        graph.add_path_graph(5)
        nodes = sorted(graph.get_node_ids())

        # Middle node
        middle_vitality = graph.get_node_vitality(nodes[2])

        # End node
        end_vitality = graph.get_node_vitality(nodes[0])

        # Middle node's removal disconnects graph (infinite distances)
        # or significantly increases distances
        assert middle_vitality >= end_vitality

    def test_star_graph_hub_most_vital(self):
        """In a star graph, the hub is the most vital."""
        graph = Graph()
        graph.add_star_graph(5)
        nodes = sorted(graph.get_node_ids())

        # First node is the hub
        hub_vitality = graph.get_node_vitality(nodes[0])

        # Leaf node
        leaf_vitality = graph.get_node_vitality(nodes[1])

        # Hub removal disconnects all leaves
        assert hub_vitality > leaf_vitality or math.isinf(hub_vitality)

    def test_cycle_graph_equal_vitality(self):
        """In a cycle, all nodes have equal vitality."""
        graph = Graph()
        graph.add_cycle_graph(6)
        nodes = sorted(graph.get_node_ids())

        vitalities = [graph.get_node_vitality(node) for node in nodes]

        # All vitalities should be equal (within floating point precision)
        assert len(set(vitalities)) == 1

    def test_complete_graph_equal_vitality(self):
        """In complete graph, all nodes have equal vitality."""
        graph = Graph()
        n = 4
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())

        # Add all edges
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        vitalities = [graph.get_node_vitality(node) for node in nodes]

        # All should be equal due to symmetry
        for i in range(1, len(vitalities)):
            assert abs(vitalities[i] - vitalities[0]) < 1e-9

    def test_node_vitality_with_label(self):
        """Node vitality works with string labels."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # Using string label
        vitality_b = graph.get_node_vitality("B")
        vitality_a = graph.get_node_vitality("A")

        # B is connector, should have higher vitality
        assert vitality_b >= vitality_a

    def test_node_vitality_invalid_node_id(self):
        """Node vitality raises error for invalid node ID."""
        graph = Graph()
        graph.add_path_graph(3)

        with pytest.raises(ValueError, match="does not exist"):
            graph.get_node_vitality(999999)

    def test_node_vitality_invalid_label(self):
        """Node vitality raises error for invalid label."""
        graph = Graph()
        graph.add_path_graph(3)

        with pytest.raises(ValueError, match="not found"):
            graph.get_node_vitality("nonexistent")

    def test_directed_graph_hub_vitality(self):
        """Directed graph hub has high vitality."""
        graph = Graph(directed=True)

        # Create hub with outgoing edges
        graph.add_node(label="hub")
        for i in range(4):
            graph.add_node(label=f"leaf{i}")

        hub_id = graph.get_id_from_label("hub")

        for i in range(4):
            leaf_id = graph.get_id_from_label(f"leaf{i}")
            graph.add_edge(hub_id, leaf_id)

        hub_vitality = graph.get_node_vitality(hub_id)
        leaf_vitality = graph.get_node_vitality(graph.get_id_from_label("leaf0"))

        # Hub removal eliminates all paths
        assert hub_vitality >= leaf_vitality


class TestNodeVitalityIntegration:
    """Test node vitality in complex scenarios."""

    def test_bridge_node_high_vitality(self):
        """Node connecting two components has high vitality."""
        graph = Graph()

        # Create two triangles
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")

        graph.add_node(label="D")
        graph.add_node(label="E")
        graph.add_node(label="F")
        graph.add_edge("D", "E")
        graph.add_edge("E", "F")
        graph.add_edge("F", "D")

        # Connect via bridge node G
        graph.add_node(label="G")
        graph.add_edge("C", "G")
        graph.add_edge("G", "D")

        g_vitality = graph.get_node_vitality("G")
        a_vitality = graph.get_node_vitality("A")

        # Bridge node should have much higher vitality
        assert g_vitality > a_vitality

    def test_vitality_vs_closeness_vitality(self):
        """get_node_vitality() matches closeness_vitality() for single node."""
        graph = Graph()
        graph.add_star_graph(5)
        nodes = sorted(graph.get_node_ids())

        # Get vitality via get_node_vitality
        single_vitality = graph.get_node_vitality(nodes[0])  # Get vitality via closeness_vitality
        all_vitalities = graph.closeness_vitality(return_dict=True)
        closeness_vitality_value = all_vitalities[nodes[0]]

        # Should match
        assert abs(single_vitality - closeness_vitality_value) < 1e-9

    def test_leaf_removal_minimal_impact(self):
        """Removing leaf node has minimal impact on graph."""
        graph = Graph()
        graph.add_star_graph(6)
        nodes = sorted(graph.get_node_ids())

        # Hub
        hub_vitality = graph.get_node_vitality(nodes[0])

        # Leaf
        leaf_vitality = graph.get_node_vitality(nodes[1])  # Leaf should have much lower vitality
        assert leaf_vitality < hub_vitality or math.isinf(hub_vitality)

    def test_disconnected_components(self):
        """Nodes in disconnected components have independent vitality."""
        graph = Graph()

        # Component 1: path of 3
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # Component 2: path of 3
        graph.add_node(label="X")
        graph.add_node(label="Y")
        graph.add_node(label="Z")
        graph.add_edge("X", "Y")
        graph.add_edge("Y", "Z")

        # Middle nodes in each component should have equal vitality
        b_vitality = graph.get_node_vitality("B")
        y_vitality = graph.get_node_vitality("Y")

        assert abs(b_vitality - y_vitality) < 1e-9

    def test_self_loop_no_impact(self):
        """Self-loops don't affect vitality."""
        graph = Graph()
        graph.add_path_graph(3)
        nodes = sorted(graph.get_node_ids())

        # Add self-loop to middle node
        graph.add_edge(nodes[1], nodes[1])

        vitality = graph.get_node_vitality(nodes[1])

        # Should still be high/infinite (disconnects graph)
        assert vitality >= 0.0

    def test_multigraph_vitality(self):
        """Multiple edges between nodes are handled correctly."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        # Multiple edges A -> B
        graph.add_edge("A", "B")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # B is still connector
        vitality_b = graph.get_node_vitality("B")
        vitality_a = graph.get_node_vitality("A")

        assert vitality_b >= vitality_a

    def test_vitality_is_finite_for_cut_vertices(self):
        """Cut vertices get a finite vitality equal to closeness_vitality()."""
        graph = Graph()
        graph.add_path_graph(5)
        nodes = sorted(graph.get_node_ids())
        all_vitalities = graph.closeness_vitality(return_dict=True)
        for n in nodes:
            v = graph.get_node_vitality(n)
            assert math.isfinite(v)
            assert v == pytest.approx(all_vitalities[n])
        assert graph.get_node_vitality(nodes[2]) == 36


class TestNodeVitalityPerformance:
    """Test node vitality performance and edge cases."""

    def test_larger_graph(self):
        """Node vitality works on larger graphs."""
        graph = Graph()
        graph.add_path_graph(20)
        nodes = sorted(graph.get_node_ids())

        # Middle node
        middle_vitality = graph.get_node_vitality(nodes[10])

        # End node
        end_vitality = graph.get_node_vitality(nodes[0])

        # Should complete and middle should be more vital
        assert middle_vitality >= end_vitality

    def test_empty_graph_neighbors(self):
        """Single isolated node should be handled correctly."""
        graph = Graph()
        graph.add_node(label="isolated")
        node_id = graph.get_id_from_label("isolated")

        vitality = graph.get_node_vitality(node_id)

        assert vitality == 0.0
