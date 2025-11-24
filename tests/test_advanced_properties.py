import pytest
import edgecase as ec


class TestArticulationPoints:
    """Test get_articulation_points() method."""

    def test_ap_empty_graph(self):
        """Empty graph has no articulation points."""
        G = ec.Graph()
        ap = G.get_articulation_points()
        assert ap == []

    def test_ap_single_node(self):
        """Single node has no articulation points."""
        G = ec.Graph()
        G.add_node("A")
        ap = G.get_articulation_points()
        assert ap == []

    def test_ap_path_graph(self):
        """Path graph has middle nodes as articulation points."""
        G = ec.Graph()
        G.add_path_graph(5)
        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        assert len(ap) == 3  # Middle 3 nodes are APs
        # End nodes (IDs 1 and 5) are NOT articulation points
        assert 1 not in ap
        assert 5 not in ap

    def test_ap_cycle_graph(self):
        """Cycle graph has no articulation points."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        ap = G.get_articulation_points()
        assert ap == []

    def test_ap_star_graph(self):
        """Star graph has hub as articulation point."""
        G = ec.Graph()
        G.add_star_graph(5)
        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        assert len(ap) == 1
        # The hub (first node added, ID 1) is the AP
        assert 1 in ap

    def test_ap_bridge_graph(self):
        """Graph with bridge has bridge endpoints as APs."""
        G = ec.Graph()
        # Create two triangles connected by a bridge
        # Create nodes first
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)

        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        G.add_edge("C", "D")  # Bridge
        G.add_edge("D", "E")
        G.add_edge("E", "F")
        G.add_edge("F", "D")

        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        # C is node ID 3, D is node ID 4
        assert 3 in ap
        assert 4 in ap

    def test_ap_directed_raises_error(self):
        """Articulation points only defined for undirected graphs."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")

        with pytest.raises(ValueError):
            G.get_articulation_points()


class TestPeriphery:
    """Test get_periphery() method."""

    def test_periphery_empty_graph(self):
        """Empty graph has no periphery."""
        G = ec.Graph()
        periphery = G.get_periphery()
        assert periphery == []

    def test_periphery_single_node(self):
        """Single node is its own periphery."""
        G = ec.Graph()
        G.add_node("A")
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        # Returns node ID 1, not label "A"
        assert periphery == [1]

    def test_periphery_path_graph(self):
        """Path graph has end nodes as periphery."""
        G = ec.Graph()
        G.add_path_graph(5)
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        assert len(periphery) == 2
        # Path nodes have IDs 1-5, periphery is nodes 1 and 5
        assert 1 in periphery
        assert 5 in periphery

    def test_periphery_star_graph(self):
        """Star graph has leaves as periphery."""
        G = ec.Graph()
        G.add_star_graph(5)
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        assert len(periphery) == 4  # All leaves (hub is at center)

    def test_periphery_cycle_graph(self):
        """Even cycle has all nodes as periphery."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        # All nodes have same eccentricity in a cycle
        assert len(periphery) == 6

    def test_periphery_disconnected_graph(self):
        """Disconnected graph periphery now returns nodes with max ecc in each component."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        periphery = G.get_periphery()
        # With updated eccentricity computation, isolated nodes have ecc 0
        # Both nodes have same eccentricity, so both are in periphery
        assert len(periphery) == 2


class TestGraphDiameter:
    """Test get_graph_diameter() method."""

    def test_diameter_empty_graph(self):
        """Empty graph has no diameter."""
        G = ec.Graph()
        diameter = G.get_diameter()
        assert diameter is None

    def test_diameter_single_node(self):
        """Single node has diameter 0."""
        G = ec.Graph()
        G.add_node("A")
        diameter = G.get_diameter()
        assert diameter == 0

    def test_diameter_path_graph(self):
        """Path graph diameter is n-1."""
        G = ec.Graph()
        G.add_path_graph(5)
        diameter = G.get_diameter()
        assert diameter == 4

    def test_diameter_star_graph(self):
        """Star graph diameter is 2 (leaf to leaf via hub)."""
        G = ec.Graph()
        G.add_star_graph(5)
        diameter = G.get_diameter()
        assert diameter == 2

    def test_diameter_cycle_graph(self):
        """Cycle graph diameter is floor(n/2)."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        diameter = G.get_diameter()
        assert diameter == 3  # Opposite side of cycle

    def test_diameter_disconnected_graph(self):
        """Disconnected graph behavior with unconnected parameter."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_node("X")  # Isolated node

        # With unconnected=False (strict mode), should return None
        diameter = G.get_diameter(unconnected=False)
        assert diameter is None

        # With unconnected=True (default), returns diameter of largest component
        diameter = G.get_diameter()
        assert diameter == 2  # Path of 3 nodes has diameter 2


class TestGraphRadius:
    """Test get_graph_radius() method."""

    def test_radius_empty_graph(self):
        """Empty graph has no radius."""
        G = ec.Graph()
        radius = G.get_radius()
        assert radius is None

    def test_radius_single_node(self):
        """Single node has radius 0."""
        G = ec.Graph()
        G.add_node("A")
        radius = G.get_radius()
        assert radius == 0

    def test_radius_path_graph(self):
        """Path graph radius is ceil((n-1)/2)."""
        G = ec.Graph()
        G.add_path_graph(5)
        radius = G.get_radius()
        assert radius == 2  # Center node has eccentricity 2

    def test_radius_star_graph(self):
        """Star graph radius is 1 (hub has eccentricity 1)."""
        G = ec.Graph()
        G.add_star_graph(5)
        radius = G.get_radius()
        assert radius == 1

    def test_radius_cycle_graph(self):
        """Cycle graph radius is floor(n/2)."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        radius = G.get_radius()
        assert radius == 3

    def test_radius_disconnected_graph(self):
        """Disconnected graph behavior with unconnected parameter."""
        G = ec.Graph()
        # Create nodes first
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("C", "D")

        # With unconnected=False (strict mode), should return None
        radius = G.get_radius(unconnected=False)
        assert radius is None

        # With unconnected=True (default), returns min radius across components
        radius = G.get_radius()
        assert radius == 1  # Both edges have nodes with eccentricity 1


class TestGirth:
    """Test get_girth() method."""

    def test_girth_empty_graph(self):
        """Empty graph has no girth."""
        G = ec.Graph()
        girth = G.get_girth()
        assert girth is None

    def test_girth_tree_graph(self):
        """Tree (acyclic) has no girth."""
        G = ec.Graph()
        G.add_balanced_tree(2, 3)
        girth = G.get_girth()
        assert girth is None

    def test_girth_triangle(self):
        """Triangle has girth 3."""
        G = ec.Graph()
        G.add_cycle_graph(3)
        girth = G.get_girth()
        assert girth == 3

    def test_girth_square(self):
        """Square has girth 4."""
        G = ec.Graph()
        G.add_cycle_graph(4)
        girth = G.get_girth()
        assert girth == 4

    def test_girth_hexagon(self):
        """Hexagon has girth 6."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        girth = G.get_girth()
        assert girth == 6

    def test_girth_complex_graph(self):
        """Graph with multiple cycles has shortest cycle as girth."""
        G = ec.Graph()
        # Create nodes first
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)
        # Create a graph with a triangle and a square
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")  # Triangle (girth 3)
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        G.add_edge("E", "F")
        G.add_edge("F", "C")  # Larger cycle

        girth = G.get_girth()
        assert girth == 3  # Triangle is shortest cycle


class TestAdhesion:
    """Test get_adhesion() method."""

    def test_adhesion_empty_graph(self):
        """Empty graph has adhesion 0."""
        G = ec.Graph()
        adhesion = G.get_adhesion()
        assert adhesion == 0

    def test_adhesion_disconnected_graph(self):
        """Disconnected graph has adhesion 0."""
        G = ec.Graph()
        # Create nodes first
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("C", "D")
        adhesion = G.get_adhesion()
        assert adhesion == 0

    def test_adhesion_path_graph(self):
        """Path graph has adhesion 1 (single edge cut)."""
        G = ec.Graph()
        G.add_path_graph(5)
        adhesion = G.get_adhesion()
        assert adhesion == 1

    def test_adhesion_cycle_graph(self):
        """Cycle graph has adhesion 2 (need 2 edge cuts)."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        adhesion = G.get_adhesion()
        assert adhesion == 2

    def test_adhesion_complete_graph(self):
        """Complete graph has adhesion n-1."""
        G = ec.Graph()
        G.add_full_graph(5)
        adhesion = G.get_adhesion()
        assert adhesion == 4  # K5 has adhesion 4

    def test_adhesion_directed_raises_error(self):
        """Adhesion only defined for undirected graphs."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")

        with pytest.raises(ValueError):
            G.get_adhesion()


class TestCohesion:
    """Test get_cohesion() method."""

    def test_cohesion_empty_graph(self):
        """Empty graph has cohesion 0."""
        G = ec.Graph()
        cohesion = G.get_cohesion()
        assert cohesion == 0

    def test_cohesion_disconnected_graph(self):
        """Disconnected graph has cohesion 0."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        cohesion = G.get_cohesion()
        assert cohesion == 0

    def test_cohesion_path_graph(self):
        """Path graph has cohesion 1 (single node cut)."""
        G = ec.Graph()
        G.add_path_graph(5)
        cohesion = G.get_cohesion()
        assert cohesion == 1

    def test_cohesion_cycle_graph(self):
        """Cycle graph has cohesion 2 (need 2 node cuts)."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        cohesion = G.get_cohesion()
        assert cohesion == 2

    def test_cohesion_complete_graph(self):
        """Complete graph has cohesion n-1."""
        G = ec.Graph()
        G.add_full_graph(5)
        cohesion = G.get_cohesion()
        assert cohesion == 4  # K5 has cohesion 4

    def test_cohesion_star_graph(self):
        """Star graph has cohesion 1 (remove hub)."""
        G = ec.Graph()
        G.add_star_graph(5)
        cohesion = G.get_cohesion()
        assert cohesion == 1

    def test_cohesion_directed_raises_error(self):
        """Cohesion only defined for undirected graphs."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")

        with pytest.raises(ValueError):
            G.get_cohesion()


class TestAdvancedPropertiesIntegration:
    """Integration tests for advanced structural properties."""

    def test_properties_on_path_graph(self):
        """Test multiple properties on path graph."""
        G = ec.Graph()
        G.add_path_graph(7)

        assert G.get_diameter() == 6
        assert G.get_radius() == 3
        assert G.get_girth() is None  # Trees have no cycles
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        assert len(periphery) == 2
        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        assert len(ap) == 5  # Middle 5 nodes
        assert G.get_adhesion() == 1
        assert G.get_cohesion() == 1

    def test_properties_on_cycle_graph(self):
        """Test multiple properties on cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(8)

        assert G.get_diameter() == 4
        assert G.get_radius() == 4
        assert G.get_girth() == 8
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        assert len(periphery) == 8  # All nodes in even cycle
        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        assert ap == []  # No APs in cycle
        assert G.get_adhesion() == 2
        assert G.get_cohesion() == 2

    def test_properties_on_complete_graph(self):
        """Test multiple properties on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)

        assert G.get_diameter() == 1
        assert G.get_radius() == 1
        assert G.get_girth() == 3  # Triangles
        periphery = G.get_periphery()  # Returns list[int] (node IDs)
        assert len(periphery) == 5  # All nodes have same eccentricity
        ap = G.get_articulation_points()  # Returns list[int] (node IDs)
        assert ap == []  # Highly connected
        assert G.get_adhesion() == 4
        assert G.get_cohesion() == 4
