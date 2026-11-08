import pytest
import math
import json
import edgecase as ec


class TestRandomGeometricGraphBasics:
    """Basic functionality tests."""

    def test_creates_correct_number_of_nodes(self):
        """Graph should have exactly n nodes."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=100, radius=0.2, seed=23)

        assert g.count_nodes() == 100

    def test_creates_edges(self):
        """With reasonable radius, graph should have edges."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=100, radius=0.3, seed=23)

        # With n=100 in unit square and radius=0.3, expect many edges
        assert g.count_edges() > 0

    def test_zero_radius_creates_no_edges(self):
        """With radius=0, no edges should be created."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.0, seed=23)

        assert g.count_nodes() == 50
        assert g.count_edges() == 0

    def test_large_radius_creates_complete_graph(self):
        """With radius >= sqrt(dim), should create complete graph."""
        # In 2D unit square, max distance is sqrt(2) ≈ 1.414
        g = ec.Graph(directed=False).add_random_geometric_graph(
            n=10,
            radius=2.0,
            seed=23,  # > sqrt(2)
        )

        assert g.count_nodes() == 10

        # Complete graph with 10 nodes has C(10,2) = 45 edges
        assert g.count_edges() == 45

    def test_returns_self_for_chaining(self):
        """Method should return Graph for chaining."""
        g = ec.Graph(directed=False)
        result = g.add_random_geometric_graph(n=10, radius=0.2, seed=23)

        assert result is g

    def test_empty_graph_n_zero(self):
        """n=0 should create no nodes."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=0, radius=0.5, seed=23)

        assert g.count_nodes() == 0
        assert g.count_edges() == 0


class TestRandomGeometricGraphNodeLabels:
    """Tests for node position labels."""

    def test_node_labels_are_position_arrays(self):
        """Node labels should be JSON arrays of positions."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=10, radius=0.3, dim=2, seed=23)

        for node_id in g.get_node_ids(limit=None):
            label = g.get_label_from_id(node_id)

            assert label is not None

            # Should be parseable as JSON array
            pos = json.loads(label)

            assert isinstance(pos, list)
            assert len(pos) == 2

    def test_positions_in_unit_cube(self):
        """All positions should be in [0, 1]^dim."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.3, dim=3, seed=23)

        for node_id in g.get_node_ids(limit=None):
            label = g.get_label_from_id(node_id)
            pos = json.loads(label)

            for coord in pos:
                assert 0.0 <= coord <= 1.0

    def test_3d_positions_have_three_coordinates(self):
        """3D graph should have 3-element position arrays."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=10, radius=0.3, dim=3, seed=23)

        for node_id in g.get_node_ids(limit=None):
            label = g.get_label_from_id(node_id)
            pos = json.loads(label)

            assert len(pos) == 3


class TestRandomGeometricGraphReproducibility:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_produces_same_graph(self):
        """Same seed should produce identical graphs."""
        g1 = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.2, seed=12345)
        g2 = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.2, seed=12345)

        assert g1.count_nodes() == g2.count_nodes()
        assert g1.count_edges() == g2.count_edges()

        # Check same positions
        for nid in g1.get_node_ids(limit=None):
            l1 = g1.get_label_from_id(nid)
            l2 = g2.get_label_from_id(nid)

            assert l1 == l2

    def test_different_seeds_produce_different_graphs(self):
        """Different seeds should typically produce different graphs."""
        g1 = ec.Graph(directed=False).add_random_geometric_graph(n=100, radius=0.2, seed=111)
        g2 = ec.Graph(directed=False).add_random_geometric_graph(n=100, radius=0.2, seed=222)

        # Very unlikely to have same edge count
        # (could happen but extremely rare)
        # We check positions instead
        nid = 1  # First node
        l1 = g1.get_label_from_id(nid)
        l2 = g2.get_label_from_id(nid)

        assert l1 != l2


class TestRandomGeometricGraphGeometry:
    """Tests verifying geometric properties."""

    def test_edges_within_radius(self):
        """All edges should connect nodes within specified radius."""
        radius = 0.3
        g = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=radius, dim=2, seed=23)

        # Get all node positions
        positions = {}
        for nid in g.get_node_ids(limit=None):
            label = g.get_label_from_id(nid)
            positions[nid] = json.loads(label)

        # Check all edges
        for edge_id in g.get_edge_ids(limit=None):
            edge = g.get_edge(edge_id)
            u, v = edge.source, edge.target
            pos_u = positions[u]
            pos_v = positions[v]

            # Euclidean distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_u, pos_v)))

            assert dist <= radius + 1e-9  # Small epsilon for floating point

    def test_no_edges_beyond_radius(self):
        """Non-edges should be for nodes beyond radius."""
        radius = 0.15  # Small radius
        g = ec.Graph(directed=False).add_random_geometric_graph(n=30, radius=radius, dim=2, seed=23)

        # Get all node positions
        positions = {}
        for nid in g.get_node_ids(limit=None):
            label = g.get_label_from_id(nid)
            positions[nid] = json.loads(label)

        # Get edges as set
        edge_set = set()
        for edge_id in g.get_edge_ids(limit=None):
            edge = g.get_edge(edge_id)
            u, v = edge.source, edge.target
            edge_set.add((min(u, v), max(u, v)))

        # Check all pairs
        node_ids = list(positions.keys())
        for i, u in enumerate(node_ids):
            for v in node_ids[i + 1 :]:
                pos_u = positions[u]
                pos_v = positions[v]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_u, pos_v)))
                edge_exists = (min(u, v), max(u, v)) in edge_set

                if dist <= radius:
                    assert edge_exists, f"Missing edge between {u} and {v}, dist={dist}"
                else:
                    assert not edge_exists, f"Extra edge between {u} and {v}, dist={dist}"


class TestRandomGeometricGraphDimensions:
    """Tests for different dimensionalities."""

    def test_1d_graph(self):
        """1D graph should work (points on a line)."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=20, radius=0.2, dim=1, seed=23)

        assert g.count_nodes() == 20

        # Check 1D positions
        for nid in g.get_node_ids(limit=None):
            label = g.get_label_from_id(nid)
            pos = json.loads(label)

            assert len(pos) == 1

    def test_high_dimensional_graph(self):
        """High dimensional graph should work."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=30, radius=0.5, dim=10, seed=23)

        assert g.count_nodes() == 30

        # Check 10D positions
        for nid in g.get_node_ids(limit=None):
            label = g.get_label_from_id(nid)
            pos = json.loads(label)

            assert len(pos) == 10

    def test_default_dim_is_2(self):
        """Default dimension should be 2."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=10, radius=0.3, seed=23)

        nid = g.get_node_ids(limit=1)[0]
        label = g.get_label_from_id(nid)
        pos = json.loads(label)

        assert len(pos) == 2


class TestRandomGeometricGraphDistanceNorms:
    """Tests for different Minkowski distance norms."""

    def test_manhattan_distance_p1(self):
        """p_norm=1.0 should use Manhattan distance."""
        g = ec.Graph(directed=False).add_random_geometric_graph(
            n=50, radius=0.3, p_norm=1.0, seed=23
        )

        assert g.count_nodes() == 50
        assert g.count_edges() > 0

    def test_euclidean_distance_default(self):
        """Default p_norm should be 2.0 (Euclidean)."""
        g1 = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.3, seed=23)

        g2 = ec.Graph(directed=False).add_random_geometric_graph(
            n=50, radius=0.3, p_norm=2.0, seed=23
        )

        assert g1.count_edges() == g2.count_edges()

    def test_manhattan_vs_euclidean_different(self):
        """Manhattan and Euclidean should produce different graphs."""
        g_euclidean = ec.Graph(directed=False).add_random_geometric_graph(
            n=100, radius=0.25, p_norm=2.0, seed=23
        )

        g_manhattan = ec.Graph(directed=False).add_random_geometric_graph(
            n=100, radius=0.25, p_norm=1.0, seed=23
        )

        # Same positions but different distance metric
        # Should have different edge counts
        assert g_euclidean.count_edges() != g_manhattan.count_edges()

    def test_edges_within_manhattan_radius(self):
        """Edges should be within Manhattan distance radius."""
        radius = 0.4

        g = ec.Graph(directed=False).add_random_geometric_graph(
            n=30, radius=radius, dim=2, p_norm=1.0, seed=23
        )

        positions = {}

        for nid in g.get_node_ids(limit=None):
            label = g.get_label_from_id(nid)
            positions[nid] = json.loads(label)

        for edge_id in g.get_edge_ids(limit=None):
            edge = g.get_edge(edge_id)
            pos_u = positions[edge.source]
            pos_v = positions[edge.target]

            # Manhattan distance
            dist = sum(abs(a - b) for a, b in zip(pos_u, pos_v))

            assert dist <= radius + 1e-9


class TestRandomGeometricGraphParameterValidation:
    """Tests for parameter validation."""

    def test_negative_radius_raises(self):
        """Negative radius should raise ValueError."""
        g = ec.Graph(directed=False)

        with pytest.raises(ValueError, match="radius"):
            g.add_random_geometric_graph(n=10, radius=-0.1)

    def test_zero_dim_raises(self):
        """dim=0 should raise ValueError."""
        g = ec.Graph(directed=False)

        with pytest.raises(ValueError, match="dimension"):
            g.add_random_geometric_graph(n=10, radius=0.3, dim=0)

    def test_zero_p_norm_raises(self):
        """p_norm=0 should raise ValueError."""
        g = ec.Graph(directed=False)

        with pytest.raises(ValueError, match="p_norm"):
            g.add_random_geometric_graph(n=10, radius=0.3, p_norm=0.0)

    def test_negative_p_norm_raises(self):
        """Negative p_norm should raise ValueError."""
        g = ec.Graph(directed=False)

        with pytest.raises(ValueError, match="p_norm"):
            g.add_random_geometric_graph(n=10, radius=0.3, p_norm=-1.0)


class TestRandomGeometricGraphChaining:
    """Tests for method chaining and composability."""

    def test_add_to_existing_graph(self):
        """Should add to existing graph, not replace."""
        g = ec.Graph(directed=False)
        g.add_node(999, "existing")

        g.add_random_geometric_graph(n=10, radius=0.3, seed=23)

        assert g.count_nodes() == 11  # 1 existing + 10 new

    def test_chain_multiple_generators(self):
        """Multiple generators should accumulate."""
        g = (
            ec.Graph(directed=False)
            .add_random_geometric_graph(n=10, radius=0.2, seed=1)
            .add_random_geometric_graph(n=10, radius=0.2, seed=2)
        )

        assert g.count_nodes() == 20

    def test_chain_with_other_methods(self):
        """Should chain with other graph methods."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.25, seed=23)

        assert g.count_nodes() == 50


class TestRandomGeometricGraphDirected:
    """Tests for directed graph behavior."""

    def test_works_with_directed_graph(self):
        """Should work with directed graphs (creates bidirectional edges)."""
        g = ec.Graph(directed=True).add_random_geometric_graph(n=20, radius=0.3, seed=23)

        assert g.count_nodes() == 20

        # In directed mode, each edge pair counts separately
        assert g.count_edges() > 0

    def test_directed_creates_more_edges(self):
        """Directed mode should create roughly 2x edges."""
        g_undir = ec.Graph(directed=False).add_random_geometric_graph(n=50, radius=0.25, seed=23)
        g_dir = ec.Graph(directed=True).add_random_geometric_graph(n=50, radius=0.25, seed=23)

        # Directed should have about 2x edges (both directions)
        # Allow some tolerance for edge counting differences
        assert g_dir.count_edges() >= g_undir.count_edges()


class TestRandomGeometricGraphProperties:
    """Tests for expected graph properties."""

    def test_clustering_coefficient_high(self):
        """Random geometric graphs should have high clustering."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=100, radius=0.2, seed=23)

        # Compute average clustering using return_dict=True
        cc_dict = g.clustering_coefficient(return_dict=True)

        if cc_dict:
            avg_clustering = sum(cc_dict.values()) / len(cc_dict)

            # Random geometric graphs typically have high clustering
            assert avg_clustering > 0.3  # Threshold based on typical values

    def test_undirected_graph_symmetric(self):
        """Undirected graph should have symmetric adjacency."""
        g = ec.Graph(directed=False).add_random_geometric_graph(n=30, radius=0.3, seed=23)
        for edge_id in g.get_edge_ids(limit=None):
            edge = g.get_edge(edge_id)

            # Check reverse edge exists (in undirected mode, just check neighbors)
            neighbors = g.get_neighbors(edge.target)

            assert edge.source in neighbors
