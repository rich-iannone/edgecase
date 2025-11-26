from edgecase import Graph


class TestIsRegular:
    """Test basic is_regular functionality."""

    def test_empty_graph_is_regular(self):
        """Empty graph is trivially regular."""
        graph = Graph()

        assert graph.is_regular()

    def test_single_node_is_regular(self):
        """Single node graph is regular."""
        graph = Graph()
        graph.add_node(label="A")

        assert graph.is_regular()

    def test_two_isolated_nodes_regular(self):
        """Two isolated nodes are regular (both have degree 0)."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        assert graph.is_regular()

    def test_cycle_is_regular(self):
        """Cycle graphs are regular (all degree 2)."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_regular()

    def test_complete_graph_is_regular(self):
        """Complete graphs are regular."""
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        assert graph.is_regular()

    def test_star_not_regular(self):
        """Star graphs are NOT regular (hub has different degree)."""
        graph = Graph()
        graph.add_star_graph(4)

        assert not graph.is_regular()

    def test_path_not_regular(self):
        """Path graphs are NOT regular (end nodes differ from middle)."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_regular()

    def test_directed_cycle_is_regular(self):
        """Directed cycle is regular (all in-degree 1, all out-degree 1)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert graph.is_regular()

    def test_directed_star_not_regular(self):
        """Directed star is NOT regular."""
        graph = Graph(directed=True)
        graph.add_star_graph(4)

        assert not graph.is_regular()

    def test_bidirectional_edges_regular(self):
        """Graph with bidirectional edges between all pairs."""
        graph = Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")

        # Add bidirectional edges
        graph.add_edge(a_id, b_id)
        graph.add_edge(b_id, a_id)
        graph.add_edge(b_id, c_id)
        graph.add_edge(c_id, b_id)
        graph.add_edge(a_id, c_id)
        graph.add_edge(c_id, a_id)

        # All nodes have in-degree 2 and out-degree 2
        assert graph.is_regular()


class TestIsKRegular:
    """Test is_k_regular functionality."""

    def test_empty_graph_0_regular(self):
        """Empty graph is 0-regular."""
        graph = Graph()

        assert graph.is_k_regular(0)
        assert not graph.is_k_regular(1)

    def test_isolated_nodes_0_regular(self):
        """Isolated nodes form 0-regular graph."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        assert graph.is_k_regular(0)
        assert not graph.is_k_regular(1)

    def test_cycle_is_2_regular(self):
        """Cycle graphs are 2-regular."""
        graph = Graph()
        graph.add_cycle_graph(6)

        assert graph.is_k_regular(2)
        assert not graph.is_k_regular(1)
        assert not graph.is_k_regular(3)

    def test_complete_graph_k_regular(self):
        """Complete graph K_n is (n-1)-regular."""
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        # K_5 is 4-regular
        assert graph.is_k_regular(4)
        assert not graph.is_k_regular(3)
        assert not graph.is_k_regular(5)

    def test_path_not_k_regular(self):
        """Path graphs are not k-regular for any k."""
        graph = Graph()
        graph.add_path_graph(5)

        assert not graph.is_k_regular(1)  # End nodes have degree 1
        assert not graph.is_k_regular(2)  # Middle nodes have degree 2

    def test_star_not_k_regular(self):
        """Star graphs are not k-regular for any k."""
        graph = Graph()
        graph.add_star_graph(4)

        # Hub has degree 3, leaves have degree 1
        assert not graph.is_k_regular(1)
        assert not graph.is_k_regular(3)

    def test_single_edge_not_1_regular(self):
        """Two nodes with one edge: both have degree 1, so 1-regular."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        graph.add_edge(a_id, b_id)

        assert graph.is_k_regular(1)
        assert not graph.is_k_regular(0)
        assert not graph.is_k_regular(2)

    def test_directed_cycle_1_regular(self):
        """Directed cycle is 1-regular (in-degree 1, out-degree 1)."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        assert graph.is_k_regular(1)
        assert not graph.is_k_regular(0)
        assert not graph.is_k_regular(2)

    def test_self_loops_affect_regularity(self):
        """Self-loops affect degree calculation."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # Add self-loops (each counts as degree 2 in undirected)
        graph.add_edge(a_id, a_id)
        graph.add_edge(b_id, b_id)

        # Both nodes have degree 2
        assert graph.is_k_regular(2)


class TestRegularityIntegration:
    """Test regularity in complex scenarios."""

    def test_is_regular_implies_some_k_regular(self):
        """If graph is regular, it must be k-regular for some k."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert graph.is_regular()
        assert graph.is_k_regular(2)

    def test_k_regular_implies_regular(self):
        """If graph is k-regular, it must be regular."""
        graph = Graph()
        graph.add_cycle_graph(6)

        assert graph.is_k_regular(2)
        assert graph.is_regular()

    def test_multigraph_regularity(self):
        """Multiple edges between same nodes affect regularity."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # Add multiple edges
        graph.add_edge(a_id, b_id)
        graph.add_edge(a_id, b_id)

        # Both nodes have degree 2 (each edge counted)
        assert graph.is_k_regular(2)
        assert graph.is_regular()

    def test_complete_bipartite_not_regular(self):
        """Complete bipartite K_{m,n} is regular only if m=n."""
        graph = Graph()

        # K_{2,3} - not regular
        for i in range(2):
            graph.add_node(label=f"A{i}")
        for i in range(3):
            graph.add_node(label=f"B{i}")

        nodes = sorted(graph.get_node_ids())
        a_nodes = nodes[:2]
        b_nodes = nodes[2:]

        for a in a_nodes:
            for b in b_nodes:
                graph.add_edge(a, b)

        # A nodes have degree 3, B nodes have degree 2
        assert not graph.is_regular()

    def test_petersen_graph_3_regular(self):
        """Petersen graph is 3-regular (a well-known example)."""
        graph = Graph()

        # Create 10 nodes
        for i in range(10):
            graph.add_node(label=f"v{i}")

        nodes = sorted(graph.get_node_ids())

        # Outer pentagon (0-4)
        for i in range(5):
            graph.add_edge(nodes[i], nodes[(i + 1) % 5])

        # Inner pentagram (5-9)
        for i in range(5):
            graph.add_edge(nodes[5 + i], nodes[5 + (i + 2) % 5])

        # Spokes connecting outer to inner
        for i in range(5):
            graph.add_edge(nodes[i], nodes[5 + i])

        assert graph.is_regular()
        assert graph.is_k_regular(3)

    def test_disconnected_regular_components(self):
        """Disconnected graph can still be regular."""
        graph = Graph()

        # First component: triangle
        graph.add_node(label="A1")
        graph.add_node(label="A2")
        graph.add_node(label="A3")
        a1 = graph.get_id_from_label("A1")
        a2 = graph.get_id_from_label("A2")
        a3 = graph.get_id_from_label("A3")
        graph.add_edge(a1, a2)
        graph.add_edge(a2, a3)
        graph.add_edge(a3, a1)

        # Second component: triangle
        graph.add_node(label="B1")
        graph.add_node(label="B2")
        graph.add_node(label="B3")
        b1 = graph.get_id_from_label("B1")
        b2 = graph.get_id_from_label("B2")
        b3 = graph.get_id_from_label("B3")
        graph.add_edge(b1, b2)
        graph.add_edge(b2, b3)
        graph.add_edge(b3, b1)

        # Both components are 2-regular
        assert graph.is_regular()
        assert graph.is_k_regular(2)

    def test_directed_regular_in_out_different(self):
        """Directed graph can be regular with different in/out degrees."""
        # Actually, for is_regular, in-degree and out-degree must each be uniform
        # but they don't need to equal each other
        graph = Graph(directed=True)

        # Create a tournament (complete directed graph)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")

        # A -> B, B -> C, C -> A (directed cycle)
        graph.add_edge(a_id, b_id)
        graph.add_edge(b_id, c_id)
        graph.add_edge(c_id, a_id)

        # All have in-degree 1 and out-degree 1
        assert graph.is_regular()
        assert graph.is_k_regular(1)


class TestRegularityEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_large_k(self):
        """Testing with k larger than possible."""
        graph = Graph()
        graph.add_cycle_graph(5)

        # Cycle is 2-regular, not 100-regular
        assert not graph.is_k_regular(100)

    def test_negative_k_invalid(self):
        """Negative k doesn't make sense but should not crash."""
        graph = Graph()
        graph.add_cycle_graph(5)

        # Negative degree is impossible
        # Python will likely reject negative usize, but let's be safe
        # This might raise an error depending on binding
        # For now, just ensure it doesn't crash
        try:
            result = graph.is_k_regular(-1)
            # If it doesn't error, result should be False
            assert result is False
        except (ValueError, OverflowError, TypeError):
            # Expected for negative values
            pass
