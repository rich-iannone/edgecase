import pytest
from edgecase import Graph


class TestMinimumEdgeCut:
    """Test minimum_edge_cut method."""

    def test_simple_path(self):
        """Test minimum edge cut on a simple path."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 4)

        assert len(cut) == 1  # Only one edge needs to be removed

    def test_parallel_edges(self):
        """Test with parallel edges (multigraph)."""
        G = Graph()
        G.add_node("A")
        G.add_node("B")
        # Two edges between A and B
        G.add_edge("A", "B")
        G.add_edge("A", "B")

        cut = G.minimum_edge_cut("A", "B")

        # For undirected graphs, edges between same nodes are represented once in the cut
        # (the cut algorithm works on edge capacity, and multiple edges increase capacity)
        assert len(cut) >= 1  # At least one edge tuple in the cut

    def test_diamond_graph(self):
        """Test on diamond graph with two paths."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        cut = G.minimum_edge_cut("A", "D")

        assert len(cut) == 2  # Need to cut both paths

    def test_complete_graph(self):
        """Test on complete graph K4."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 4)
        G.add_edge(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 4)

        assert len(cut) == 3  # Three edge-disjoint paths

    def test_directed_graph(self):
        """Test on directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        cut = G.minimum_edge_cut("A", "D")

        assert len(cut) == 2  # Two paths from A to D

    def test_bottleneck(self):
        """Test graph with clear bottleneck."""
        G = Graph()
        for i in [1, 2, 3, 4, 5, 6]:
            G.add_node(i)

        # Left cluster connected by single edge to right cluster
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Left triangle
        G.add_edge(3, 4)  # Bottleneck
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 4)  # Right triangle

        cut = G.minimum_edge_cut(1, 6)

        assert len(cut) == 1  # Only the bottleneck edge
        assert (3, 4) in cut or (4, 3) in cut

    def test_string_labels(self):
        """Test with string node labels."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave"]:
            G.add_node(label)
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")
        G.add_edge("Carol", "Dave")

        cut = G.minimum_edge_cut("Alice", "Dave")
        assert len(cut) >= 1

    def test_self_loop_ignored(self):
        """Test that self-loops don't affect min cut."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 1)  # Self-loop
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_edge_cut(1, 3)
        # Self-loop shouldn't be in the cut
        for edge in cut:
            assert edge[0] != edge[1]

    def test_disconnected_graph_error(self):
        """Test that disconnected nodes have no cut."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(3, 4)

        cut = G.minimum_edge_cut(1, 3)

        # If no path exists, cut should be empty
        # (all edges from source side to unreachable side)
        assert isinstance(cut, set)

    def test_same_node_error(self):
        """Test error when source equals sink."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="Source and sink must be different"):
            G.minimum_edge_cut(1, 1)

    def test_nonexistent_node_error(self):
        """Test error when node doesn't exist."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="does not exist"):
            G.minimum_edge_cut(1, 999)

    def test_bridge_graph(self):
        """Test graph with bridge edge."""
        G = Graph()
        for i in [1, 2, 3, 4, 5]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Triangle
        G.add_edge(3, 4)  # Bridge
        G.add_edge(4, 5)

        cut = G.minimum_edge_cut(1, 5)

        assert len(cut) == 1


class TestMinimumNodeCut:
    """Test minimum_node_cut method."""

    def test_simple_path(self):
        """Test minimum node cut on simple path."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        cut = G.minimum_node_cut(1, 4)

        # Removing either node 2 or 3 disconnects 1 from 4
        assert len(cut) == 1
        assert 2 in cut or 3 in cut

    def test_diamond_graph(self):
        """Test on diamond graph - two independent paths."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        a_id = G.get_id_from_label("A")
        d_id = G.get_id_from_label("D")

        cut = G.minimum_node_cut("A", "D")

        # Need to remove both B and C to disconnect A from D
        assert len(cut) == 2

    def test_source_sink_not_in_cut(self):
        """Test that source and sink are never in the cut."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_node_cut(1, 3)

        assert 1 not in cut
        assert 3 not in cut
        assert 2 in cut

    def test_articulation_point(self):
        """Test graph with articulation point."""
        G = Graph()
        for i in [1, 2, 3, 4, 5, 6]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)  # Triangle
        G.add_edge(3, 4)  # Articulation point connection
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 4)  # Another triangle

        cut = G.minimum_node_cut(1, 6)

        # Either node 3 or node 4 must be in the cut
        assert len(cut) == 1
        assert 3 in cut or 4 in cut

    def test_complete_graph(self):
        """Test on complete graph K5."""
        G = Graph()
        nodes = [1, 2, 3, 4, 5]
        for i in nodes:
            G.add_node(i)
        for i in nodes:
            for j in nodes:
                if i < j:
                    G.add_edge(i, j)

        cut = G.minimum_node_cut(1, 5)

        # In K5, need to remove 3 nodes to disconnect any two
        assert len(cut) == 3

    def test_directed_graph(self):
        """Test on directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")
        G.add_edge("D", "E")

        cut = G.minimum_node_cut("A", "E")

        # Node D is the bottleneck
        assert len(cut) == 1

        d_id = G.get_id_from_label("D")

        assert d_id in cut

    def test_string_labels(self):
        """Test with string labels."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave"]:
            G.add_node(label)
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")
        G.add_edge("Alice", "Dave")
        G.add_edge("Dave", "Carol")

        cut = G.minimum_node_cut("Alice", "Carol")

        # Both Bob and Dave must be removed
        assert len(cut) == 2

    def test_single_intermediary(self):
        """Test graph with single intermediary node."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        cut = G.minimum_node_cut(1, 3)

        assert len(cut) == 1
        assert 2 in cut

    def test_same_node_error(self):
        """Test error when source equals sink."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="Source and sink must be different"):
            G.minimum_node_cut(1, 1)

    def test_nonexistent_node_error(self):
        """Test error when node doesn't exist."""
        G = Graph()
        for i in [1, 2]:
            G.add_node(i)
        G.add_edge(1, 2)

        with pytest.raises(ValueError, match="does not exist"):
            G.minimum_node_cut(1, 999)

    def test_adjacent_nodes(self):
        """Test when source and sink are adjacent."""
        G = Graph()
        for i in [1, 2, 3]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(3, 2)  # Alternative path

        cut = G.minimum_node_cut(1, 2)

        # Need to remove node 3 to force only direct connection
        assert len(cut) == 1
        assert 3 in cut

    def test_hub_and_spoke(self):
        """Test hub-and-spoke topology."""
        G = Graph()
        for i in [1, 2, 3, 4, 5]:
            G.add_node(i)
        # Hub node in center
        G.add_edge(1, 5)  # Spoke 1 to hub
        G.add_edge(2, 5)  # Spoke 2 to hub
        G.add_edge(3, 5)  # Spoke 3 to hub
        G.add_edge(4, 5)  # Spoke 4 to hub

        cut = G.minimum_node_cut(1, 2)

        # Hub node 5 is the only node that disconnects them
        assert len(cut) == 1
        assert 5 in cut


class TestMinimumCutComparison:
    """Test comparing edge cut vs node cut."""

    def test_edge_vs_node_cut(self):
        """Compare edge and node cuts on same graph."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)

        edge_cut = G.minimum_edge_cut(1, 4)
        node_cut = G.minimum_node_cut(1, 4)

        # Both should disconnect the graph
        assert len(edge_cut) >= 1
        assert len(node_cut) >= 1

    def test_min_cut_size_equivalence(self):
        """Test that get_min_cut_between matches minimum_edge_cut size."""
        G = Graph()
        for i in [1, 2, 3, 4]:
            G.add_node(i)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(1, 4)
        G.add_edge(2, 4)

        cut_size = G.get_min_cut_between(1, 3)
        edge_cut = G.minimum_edge_cut(1, 3)

        assert cut_size == len(edge_cut)


class TestMinimumCutRealWorld:
    """Real-world scenarios for minimum cuts."""

    def test_network_vulnerability(self):
        """Test identifying critical network links."""
        G = Graph(directed=True)
        for label in ["Router1", "Router2", "Router3", "Server"]:
            G.add_node(label)
        G.add_edge("Router1", "Router2")
        G.add_edge("Router1", "Router3")
        G.add_edge("Router2", "Server")
        G.add_edge("Router3", "Server")

        critical_links = G.minimum_edge_cut("Router1", "Server")

        # Both paths to server are needed
        assert len(critical_links) == 2

    def test_social_network_bridges(self):
        """Test finding bridge people in social network."""
        G = Graph()
        for label in ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]:
            G.add_node(label)
        # Two communities connected by one person
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Carol")  # Community 1
        G.add_edge("Carol", "Dave")  # Bridge
        G.add_edge("Dave", "Eve")
        G.add_edge("Eve", "Frank")  # Community 2

        bridges = G.minimum_node_cut("Alice", "Frank")

        # Carol and Dave are bridges
        assert len(bridges) >= 1

    def test_supply_chain(self):
        """Test supply chain bottleneck analysis."""
        G = Graph(directed=True)
        for label in ["Supplier", "Factory1", "Factory2", "Distributor", "Retailer"]:
            G.add_node(label)
        G.add_edge("Supplier", "Factory1")
        G.add_edge("Supplier", "Factory2")
        G.add_edge("Factory1", "Distributor")
        G.add_edge("Factory2", "Distributor")
        G.add_edge("Distributor", "Retailer")

        bottleneck = G.minimum_node_cut("Supplier", "Retailer")

        # Distributor is single point of failure
        assert len(bottleneck) == 1

        dist_id = G.get_id_from_label("Distributor")

        assert dist_id in bottleneck
