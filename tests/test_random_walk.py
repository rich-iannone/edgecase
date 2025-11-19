import pytest
from edgecase import Graph


class TestRandomWalk:
    """Test random_walk() method."""

    def test_basic_walk(self):
        """Test basic random walk on a cycle."""
        g = Graph()
        for i in range(1, 7):
            g.add_node(label=str(i))
        for i in range(1, 7):
            g.add_edge(str(i), str((i % 6) + 1))

        path = g.random_walk("1", steps=10, seed=42, return_list=True)
        assert len(path) == 11  # steps + 1
        assert path[0] == g.get_id_from_label("1")  # starts at specified vertex

        # All vertices in path should exist
        for v in path:
            assert v in list(range(1, 7))

    def test_walk_reproducibility(self):
        """Test that seed makes walks reproducible."""
        g = Graph()
        for i in range(1, 7):
            g.add_node(label=str(i))
        for i in range(1, 7):
            g.add_edge(str(i), str((i % 6) + 1))

        path1 = g.random_walk("1", steps=100, seed=42, return_list=True)
        path2 = g.random_walk("1", steps=100, seed=42, return_list=True)
        assert path1 == path2

    def test_walk_randomness(self):
        """Test that walks without seed are different."""
        g = Graph()
        for i in range(1, 7):
            g.add_node(label=str(i))
        for i in range(1, 7):
            g.add_edge(str(i), str((i % 6) + 1))

        path1 = g.random_walk("1", steps=100, return_list=True)
        path2 = g.random_walk("1", steps=100, return_list=True)
        # Very unlikely to be identical for 100 steps
        assert path1 != path2

    def test_mode_out(self):
        """Test mode='out' follows edge directions."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        # Walking forward from A should only reach B, then get stuck
        path = g.random_walk("A", steps=10, mode="out", stuck="return", seed=42, return_list=True)
        assert len(path) <= 4  # A -> B -> C -> D
        assert path[0] == g.get_id_from_label("A")
        if len(path) > 1:
            assert path[1] == g.get_id_from_label("B")

    def test_mode_in(self):
        """Test mode='in' reverses edge directions."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        # Walking backward from C should reach B, then A
        path = g.random_walk("C", steps=10, mode="in", stuck="return", seed=42, return_list=True)
        assert path[0] == g.get_id_from_label("C")
        if len(path) > 1:
            assert path[1] == g.get_id_from_label("B")
        if len(path) > 2:
            assert path[2] == g.get_id_from_label("A")

    def test_mode_all(self):
        """Test mode='all' ignores edge direction."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # mode='all' treats as undirected, can go both ways
        path = g.random_walk("B", steps=20, mode="all", seed=42, return_list=True)
        assert len(path) == 21
        assert path[0] == g.get_id_from_label("B")
        # Should visit both A and C eventually
        assert g.get_id_from_label("A") in path or "C" in path

    def test_stuck_return(self):
        """Test stuck='return' returns partial walk."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        path = g.random_walk("A", steps=10, stuck="return", return_list=True)
        assert len(path) == 3  # A -> B -> C (stuck)
        a_id, b_id, c_id = (
            g.get_id_from_label("A"),
            g.get_id_from_label("B"),
            g.get_id_from_label("C"),
        )
        assert path == [a_id, b_id, c_id]

    def test_stuck_error(self):
        """Test stuck='error' raises exception."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        with pytest.raises(RuntimeError, match="stuck"):
            g.random_walk("A", steps=10, stuck="error", return_list=True)

    def test_invalid_start(self):
        """Test error on invalid start vertex."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(ValueError, match="not found"):
            g.random_walk("Z", steps=10, return_list=True)

    def test_invalid_mode(self):
        """Test error on invalid mode."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(ValueError, match="mode"):
            g.random_walk("A", steps=10, mode="invalid", return_list=True)

    def test_invalid_stuck(self):
        """Test error on invalid stuck parameter."""
        g = Graph()
        g.add_node(label="A")

        with pytest.raises(ValueError, match="stuck"):
            g.random_walk("A", steps=10, stuck="invalid", return_list=True)

    def test_zero_steps(self):
        """Test walk with zero steps."""
        g = Graph()
        g.add_node(label="A")

        path = g.random_walk("A", steps=0, return_list=True)
        assert len(path) == 1
        assert path == [g.get_id_from_label("A")]

    def test_isolated_vertex(self):
        """Test walk on isolated vertex."""
        g = Graph()
        g.add_node(label="A")

        # Should get stuck immediately
        path = g.random_walk("A", steps=10, stuck="return", return_list=True)
        assert len(path) == 1
        assert path == [g.get_id_from_label("A")]

    def test_star_graph(self):
        """Test walk on star graph."""
        g = Graph()
        g.add_node(label="center")
        for i in range(1, 6):
            g.add_node(label=f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        # Walk from center should alternate center/leaf
        path = g.random_walk("center", steps=10, seed=42, return_list=True)
        assert len(path) == 11
        assert path[0] == g.get_id_from_label("center")
        # Every other vertex should be center
        center_id = g.get_id_from_label("center")
        for i in range(0, len(path), 2):
            if i < len(path):
                assert path[i] == center_id


class TestRandomEdgeWalk:
    """Test random_edge_walk() method."""

    def test_basic_edge_walk(self):
        """Test basic random edge walk."""
        g = Graph()
        for i in range(1, 4):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")

        edges = g.random_edge_walk("1", steps=10, seed=42, return_list=True)
        assert len(edges) == 10  # exactly 'steps' edges

        # Each edge should be a tuple of (edge_id, source_id, target_id)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3
            edge_id, source_id, target_id = edge
            assert isinstance(edge_id, int)
            assert isinstance(source_id, int)
            assert isinstance(target_id, int)

    def test_edge_walk_validity(self):
        """Test that edge walk returns valid edges."""
        g = Graph()
        for i in range(1, 4):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")

        edge_path = g.random_edge_walk("1", steps=10, seed=42, return_list=True)

        # Should have 10 edges
        assert len(edge_path) == 10

        # Each edge should be valid
        for edge_id, source_id, target_id in edge_path:
            assert isinstance(edge_id, int)
            assert isinstance(source_id, int)
            assert isinstance(target_id, int)
            # Node IDs should be valid (1, 2, or 3)
            assert source_id in [1, 2, 3]
            assert target_id in [1, 2, 3]

    def test_edge_walk_mode(self):
        """Test edge walk with different modes."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # Forward
        forward = g.random_edge_walk(
            "A", steps=5, mode="out", stuck="return", seed=42, return_list=True
        )
        assert len(forward) <= 2
        # Edges should connect A->B and B->C

        # Backward
        backward = g.random_edge_walk(
            "C", steps=5, mode="in", stuck="return", seed=42, return_list=True
        )
        assert len(backward) <= 2
        # When walking backward from C, we traverse edges in reverse
        # Edge tuples still show original direction (source, target)
        # but walk goes: C -> B -> A

    def test_edge_walk_stuck(self):
        """Test edge walk stuck handling."""
        g = Graph(directed=True)
        for node in ["A", "B"]:
            g.add_node(label=node)
        g.add_edge("A", "B")

        # Return partial
        edges = g.random_edge_walk("A", steps=10, stuck="return", return_list=True)
        assert len(edges) == 1  # Only one edge before stuck

        # Raise error
        with pytest.raises(RuntimeError, match="stuck"):
            g.random_edge_walk("A", steps=10, stuck="error", return_list=True)

    def test_edge_walk_zero_steps(self):
        """Test edge walk with zero steps."""
        g = Graph()
        g.add_node(label="A")

        edges = g.random_edge_walk("A", steps=0, return_list=True)
        assert len(edges) == 0

    def test_edge_walk_reproducibility(self):
        """Test that seed makes edge walks reproducible."""
        g = Graph()
        for i in range(1, 7):
            g.add_node(label=str(i))
        for i in range(1, 7):
            g.add_edge(str(i), str((i % 6) + 1))

        edges1 = g.random_edge_walk("1", steps=100, seed=42, return_list=True)
        edges2 = g.random_edge_walk("1", steps=100, seed=42, return_list=True)
        assert edges1 == edges2


class TestRandomWalkIntegration:
    """Integration tests for random walks."""

    def test_disconnected_graph(self):
        """Test walk on disconnected graph."""
        g = Graph()
        # Component 1
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Component 2
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("C", "D")

        # Walk from A should only visit A and B
        path = g.random_walk("A", steps=20, seed=42, return_list=True)
        vertices_visited = set(path)
        a_id, b_id = g.get_id_from_label("A"), g.get_id_from_label("B")
        c_id, d_id = g.get_id_from_label("C"), g.get_id_from_label("D")
        assert vertices_visited.issubset({a_id, b_id})
        assert c_id not in vertices_visited
        assert d_id not in vertices_visited

    def test_self_loop(self):
        """Test walk with self-loops."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "A")  # self-loop
        g.add_edge("A", "B")

        # Walk can stay at A or move to B
        path = g.random_walk("A", steps=20, seed=42, return_list=True)
        assert len(path) == 21
        a_id, b_id = g.get_id_from_label("A"), g.get_id_from_label("B")
        assert all(v in [a_id, b_id] for v in path)

    def test_large_walk(self):
        """Test long walk on larger graph."""
        g = Graph()
        n = 100
        for i in range(n):
            g.add_node(label=str(i))
        # Create a cycle
        for i in range(n):
            g.add_edge(str(i), str((i + 1) % n))

        path = g.random_walk("0", steps=1000, seed=42, return_list=True)
        assert len(path) == 1001
        assert path[0] == g.get_id_from_label("0")

        # Should visit multiple vertices
        unique_vertices = len(set(path))
        assert unique_vertices > 10  # Should explore the graph
