import pytest
import edgecase as ec


class TestNodeConnectivity:
    """Test node connectivity functions."""

    def test_node_connectivity_path_graph(self):
        """Path graph has node connectivity 1."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Node connectivity between endpoints
        conn = G._graph.get_node_connectivity_between(1, 5)
        assert conn == 1  # Removing any intermediate node disconnects path

    def test_node_connectivity_cycle_graph(self):
        """Cycle graph has node connectivity 2."""
        G = ec.Graph()
        G.add_cycle_graph(5)

        # Any two nodes in a cycle
        conn = G._graph.get_node_connectivity_between(1, 3)
        assert conn == 2  # Need to remove 2 nodes to disconnect

    def test_node_connectivity_complete_graph(self):
        """Complete graph K_n has node connectivity n-1."""
        G = ec.Graph()
        # Build K_5 (complete graph with 5 nodes)
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        conn = G._graph.get_node_connectivity()
        assert conn == 4  # K_5 has connectivity 4

    def test_node_connectivity_star_graph(self):
        """Star graph has node connectivity 1."""
        G = ec.Graph()
        G.add_star_graph(5)  # 1 central node + 5 peripheral

        conn = G._graph.get_node_connectivity()
        assert conn == 1  # Removing center disconnects everything

    def test_node_connectivity_disconnected(self):
        """Disconnected graph has connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")
        # C is isolated

        conn = G._graph.get_node_connectivity()
        assert conn == 0


class TestEdgeConnectivity:
    """Test edge connectivity functions."""

    def test_edge_connectivity_path(self):
        """Path graph has edge connectivity 1."""
        G = ec.Graph()
        G.add_path_graph(5)

        conn = G._graph.get_edge_connectivity()
        assert conn == 1

    def test_edge_connectivity_cycle(self):
        """Cycle graph has edge connectivity 2."""
        G = ec.Graph()
        G.add_cycle_graph(5)

        conn = G._graph.get_edge_connectivity()
        assert conn == 2

    def test_edge_connectivity_complete(self):
        """Complete graph K_n has high edge connectivity."""
        G = ec.Graph()
        # Build K_5
        for i in range(1, 6):
            G.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(str(i), str(j))

        conn = G._graph.get_edge_connectivity()
        # For K_5, edge connectivity equals vertex connectivity = 4
        assert conn == 4

    def test_edge_connectivity_star(self):
        """Star graph has edge connectivity 1."""
        G = ec.Graph()
        G.add_star_graph(5)

        conn = G._graph.get_edge_connectivity()
        assert conn == 1

    def test_edge_connectivity_disconnected(self):
        """Disconnected graph has edge connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge("A", "B")

        conn = G._graph.get_edge_connectivity()
        assert conn == 0


class TestConnectivityEdgeCases:
    """Test edge cases for connectivity."""

    def test_empty_graph(self):
        """Empty graph has connectivity 0."""
        G = ec.Graph()

        assert G._graph.get_node_connectivity() == 0
        assert G._graph.get_edge_connectivity() == 0

    def test_single_node(self):
        """Single node graph has connectivity 0."""
        G = ec.Graph()
        G.add_node(label="A")

        assert G._graph.get_node_connectivity() == 0
        assert G._graph.get_edge_connectivity() == 0

    def test_two_nodes_connected(self):
        """Two connected nodes have connectivity 1."""
        G = ec.Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_edge("A", "B")

        # Node connectivity between same nodes should error
        with pytest.raises(Exception):
            G._graph.get_node_connectivity_between(1, 1)

        # Global connectivity
        assert G._graph.get_node_connectivity() == 1
        assert G._graph.get_edge_connectivity() == 1
