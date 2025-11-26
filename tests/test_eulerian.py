from edgecase import Graph


class TestIsEulerian:
    """Tests for is_eulerian method."""

    def test_empty_graph_is_eulerian(self):
        """Empty graph is considered Eulerian."""
        graph = Graph()

        assert graph.is_eulerian()

    def test_single_node_is_eulerian(self):
        """Single node with no edges is Eulerian."""
        graph = Graph()
        graph.add_node(0)

        assert graph.is_eulerian()

    def test_cycle_graph_is_eulerian(self):
        """Cycle graph has Eulerian circuit (all even degrees)."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

    def test_path_graph_not_eulerian(self):
        """Path graph is not Eulerian (endpoints have odd degree)."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

    def test_star_graph_not_eulerian(self):
        """Star graph is not Eulerian (center and leaves have odd degree)."""
        graph = Graph()
        graph.add_star_graph(4)

        # Star: center has degree 4 (even), but leaves have degree 1 (odd)
        # More than 2 odd nodes -> not Eulerian
        assert not graph.is_eulerian()

    def test_grid_2x2_is_eulerian(self):
        """2x2 grid is Eulerian."""
        graph = Graph()
        graph.add_grid_2d(2, 2)

        # 2x2 grid: corners have degree 2, but forms a cycle -> Eulerian
        assert graph.is_eulerian()

    def test_disconnected_graph_not_eulerian(self):
        """Disconnected graph is not Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(3)  # First component (nodes 1,2,3)

        # Add disconnected path
        before = set(graph.get_node_ids())
        graph.add_node(label="d1")
        graph.add_node(label="d2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])

        assert not graph.is_eulerian()

    def test_directed_cycle_is_eulerian(self):
        """Directed cycle is Eulerian (in-degree = out-degree)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

    def test_directed_path_not_eulerian(self):
        """Directed path is not Eulerian."""
        graph = Graph(directed=True)
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

    def test_directed_balanced_graph(self):
        """Directed graph with balanced degrees is Eulerian."""
        graph = Graph(directed=True)

        # Create directed cycle with balanced degrees
        graph.add_cycle_graph(4)

        assert graph.is_eulerian()


class TestIsSemiEulerian:
    """Tests for is_semieulerian method."""

    def test_empty_graph_not_semieulerian(self):
        """Empty graph is not semi-Eulerian."""
        graph = Graph()

        assert not graph.is_semieulerian()

    def test_single_node_not_semieulerian(self):
        """Single node is not semi-Eulerian."""
        graph = Graph()
        graph.add_node(0)

        assert not graph.is_semieulerian()

    def test_path_graph_is_semieulerian(self):
        """Path graph is semi-Eulerian (exactly 2 odd-degree nodes)."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_semieulerian()

    def test_cycle_graph_not_semieulerian(self):
        """Cycle graph is not semi-Eulerian (it's Eulerian)."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert not graph.is_semieulerian()

    def test_star_graph_not_semieulerian(self):
        """Star graph is not semi-Eulerian (center has odd degree, leaves have degree 1)."""
        graph = Graph()
        graph.add_star_graph(5)

        # Star: center has degree 5 (odd), 5 leaves have degree 1 (odd)
        # More than 2 odd-degree nodes -> not semi-Eulerian
        assert not graph.is_semieulerian()

    def test_directed_path_is_semieulerian(self):
        """Directed path is semi-Eulerian (one source, one sink)."""
        graph = Graph(directed=True)
        graph.add_path_graph(5)

        assert graph.is_semieulerian()

    def test_directed_cycle_not_semieulerian(self):
        """Directed cycle is not semi-Eulerian (it's Eulerian)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert not graph.is_semieulerian()

    def test_graph_with_two_odd_degrees(self):
        """Graph with exactly 2 odd-degree vertices is semi-Eulerian."""
        # Path graph has exactly 2 odd-degree nodes (the endpoints)
        graph = Graph()
        graph.add_path_graph(4)

        assert graph.is_semieulerian()


class TestEulerianCircuit:
    """Tests for eulerian_circuit method."""

    def test_empty_graph_returns_empty_circuit(self):
        """Empty graph returns empty circuit."""
        graph = Graph()
        circuit = graph.eulerian_circuit()

        assert circuit == []

    def test_single_node_returns_single_node_circuit(self):
        """Single node returns circuit with just that node."""
        graph = Graph()
        node_id = graph.add_node(label="only").get_node_ids()[0]
        circuit = graph.eulerian_circuit()

        assert circuit == [node_id]

    def test_path_graph_no_circuit(self):
        """Path graph has no Eulerian circuit."""
        graph = Graph()
        graph.add_path_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_cycle_graph_has_circuit(self):
        """Cycle graph has Eulerian circuit."""
        graph = Graph()
        graph.add_cycle_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is not None

        # Circuit should have exactly n+1 nodes for n-cycle (returns to start)
        assert len(circuit) == 6  # 5 edges + return to start
        assert circuit[0] == circuit[-1]  # Starts and ends at same node

    def test_circuit_visits_all_edges(self):
        """Circuit visits all edges."""
        graph = Graph()
        graph.add_cycle_graph(4)
        circuit = graph.eulerian_circuit()

        # Circuit should visit all 4 edges
        assert circuit is not None
        assert len(circuit) >= 5  # At least 4 edges + return

    def test_directed_cycle_has_circuit(self):
        """Directed cycle has Eulerian circuit."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)
        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) >= 6  # At least 5 edges + return
        assert circuit[0] == circuit[-1]

    def test_circuit_respects_direction(self):
        """Directed circuit follows edge directions."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(3)

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) == 4  # 3 edges + return
        assert circuit[0] == circuit[-1]

        # Verify edges exist in the direction traveled
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            neighbors = graph.neighbors(u)
            assert v in neighbors

    def test_disconnected_graph_no_circuit(self):
        """Disconnected graph has no Eulerian circuit."""
        graph = Graph()
        graph.add_cycle_graph(3)

        # Add disconnected path
        before = set(graph.get_node_ids())
        graph.add_node(label="d1")
        graph.add_node(label="d2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])

        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_grid_2x2_has_circuit(self):
        """2x2 grid has Eulerian circuit."""
        graph = Graph()
        graph.add_grid_2d(2, 2)
        circuit = graph.eulerian_circuit()

        assert circuit is not None

        # 2x2 grid: 4 nodes, but edges form a cycle
        assert len(circuit) >= 5  # At least visit all edges + return


class TestEulerianIntegration:
    """Integration tests combining multiple Eulerian methods."""

    def test_eulerian_implies_circuit_exists(self):
        """If is_eulerian is True, eulerian_circuit should return a circuit."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None

    def test_not_eulerian_implies_no_circuit(self):
        """If is_eulerian is False, eulerian_circuit should return None."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is None

    def test_semieulerian_not_eulerian(self):
        """Semi-Eulerian graph is not Eulerian."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_semieulerian()
        assert not graph.is_eulerian()

    def test_eulerian_not_semieulerian(self):
        """Eulerian graph is not semi-Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_eulerian()
        assert not graph.is_semieulerian()

    def test_neither_eulerian_nor_semieulerian(self):
        """Graph can be neither Eulerian nor semi-Eulerian."""
        graph = Graph()
        graph.add_star_graph(5)

        assert not graph.is_eulerian()
        assert not graph.is_semieulerian()


class TestEulerianEdgeCases:
    """Edge cases and special scenarios."""

    def test_single_isolated_node(self):
        """Single isolated node is Eulerian."""
        graph = Graph()
        graph.add_node(label="solo")

        # Single node with no edges is Eulerian
        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) == 1

    def test_multiple_edges_between_nodes(self):
        """Multiple edges between same nodes (multigraph)."""
        graph = Graph()
        graph.add_path_graph(2)  # Two nodes with one edge
        node_ids = sorted(graph.get_node_ids())

        # Add more edges to make even degree
        graph.add_edge(node_ids[0], node_ids[1])  # Now 2 edges
        graph.add_edge(node_ids[0], node_ids[1])  # Now 3 edges (odd!)

        # Each node has degree 3 (odd) -> not Eulerian, but is semi-Eulerian
        assert not graph.is_eulerian()
        assert graph.is_semieulerian()

    def test_very_small_cycle(self):
        """Triangle is Eulerian."""
        graph = Graph()
        graph.add_cycle_graph(3)

        assert graph.is_eulerian()

        circuit = graph.eulerian_circuit()

        assert circuit is not None
        assert len(circuit) >= 4  # At least 3 edges + return

    def test_directed_strongly_connected_components(self):
        """Directed graph with multiple SCCs is not Eulerian."""
        graph = Graph(directed=True)

        # Create first cycle
        graph.add_cycle_graph(3)

        # Add disconnected cycle
        before = set(graph.get_node_ids())
        graph.add_node(label="x1")
        graph.add_node(label="x2")
        after = set(graph.get_node_ids())
        new_nodes = sorted(list(after - before))
        graph.add_edge(new_nodes[0], new_nodes[1])
        graph.add_edge(new_nodes[1], new_nodes[0])

        # Not weakly connected -> not Eulerian
        assert not graph.is_eulerian()
