"""
Tests for advanced structural property methods.

Tests the following methods:
- get_degree_histogram()
- get_articulation_points()
- get_periphery()
- get_graph_diameter()
- get_graph_radius()
- get_girth()
- get_adhesion()
- get_cohesion()
"""

import pytest
import edgecase as ec


class TestDegreeHistogram:
    """Test get_degree_histogram() method."""
    
    def test_histogram_empty_graph(self):
        """Empty graph should return empty histogram."""
        G = ec.Graph()
        hist = G.get_degree_histogram()
        assert hist == {}
    
    def test_histogram_path_graph(self):
        """Path graph has specific degree distribution."""
        G = ec.Graph()
        G.add_path_graph(5)
        hist = G.get_degree_histogram()
        assert hist[1] == 2  # Two end nodes with degree 1
        assert hist[2] == 3  # Three middle nodes with degree 2
    
    def test_histogram_star_graph(self):
        """Star graph has hub and leaves."""
        G = ec.Graph()
        G.add_star_graph(6)  # 1 hub + 5 leaves
        hist = G.get_degree_histogram()
        assert hist[1] == 5  # 5 leaves
        assert hist[5] == 1  # 1 hub
    
    def test_histogram_complete_graph(self):
        """Complete graph has all nodes with same degree."""
        G = ec.Graph()
        G.add_full_graph(5)
        hist = G.get_degree_histogram()
        assert hist[4] == 5  # All 5 nodes have degree 4
    
    def test_histogram_cycle_graph(self):
        """Cycle graph has all nodes with degree 2."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        hist = G.get_degree_histogram()
        assert hist[2] == 6  # All nodes have degree 2


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
        G.add_node('A')
        ap = G.get_articulation_points()
        assert ap == []
    
    def test_ap_path_graph(self):
        """Path graph has middle nodes as articulation points."""
        G = ec.Graph()
        G.add_path_graph(5)
        ap = G.get_articulation_points()
        assert len(ap) == 3  # Middle 3 nodes are APs
        # End nodes are NOT articulation points
        assert '1' not in ap
        assert '5' not in ap
    
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
        ap = G.get_articulation_points()
        assert len(ap) == 1
        # The hub (first node added) is the AP
    
    def test_ap_bridge_graph(self):
        """Graph with bridge has bridge endpoints as APs."""
        G = ec.Graph()
        # Create two triangles connected by a bridge
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        G.add_edge('C', 'D')  # Bridge
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        ap = G.get_articulation_points()
        assert 'C' in ap
        assert 'D' in ap
    
    def test_ap_directed_raises_error(self):
        """Articulation points only defined for undirected graphs."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        
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
        G.add_node('A')
        periphery = G.get_periphery()
        assert periphery == ['A']
    
    def test_periphery_path_graph(self):
        """Path graph has end nodes as periphery."""
        G = ec.Graph()
        G.add_path_graph(5)
        periphery = G.get_periphery()
        assert len(periphery) == 2
        assert '1' in periphery
        assert '5' in periphery
    
    def test_periphery_star_graph(self):
        """Star graph has leaves as periphery."""
        G = ec.Graph()
        G.add_star_graph(5)
        periphery = G.get_periphery()
        assert len(periphery) == 4  # All leaves
    
    def test_periphery_cycle_graph(self):
        """Even cycle has all nodes as periphery."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        periphery = G.get_periphery()
        # All nodes have same eccentricity in a cycle
        assert len(periphery) == 6
    
    def test_periphery_disconnected_graph(self):
        """Disconnected graph returns empty periphery."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        periphery = G.get_periphery()
        assert periphery == []


class TestGraphDiameter:
    """Test get_graph_diameter() method."""
    
    def test_diameter_empty_graph(self):
        """Empty graph has no diameter."""
        G = ec.Graph()
        diameter = G.get_graph_diameter()
        assert diameter is None
    
    def test_diameter_single_node(self):
        """Single node has diameter 0."""
        G = ec.Graph()
        G.add_node('A')
        diameter = G.get_graph_diameter()
        assert diameter == 0
    
    def test_diameter_path_graph(self):
        """Path graph diameter is n-1."""
        G = ec.Graph()
        G.add_path_graph(5)
        diameter = G.get_graph_diameter()
        assert diameter == 4
    
    def test_diameter_star_graph(self):
        """Star graph diameter is 2 (leaf to leaf via hub)."""
        G = ec.Graph()
        G.add_star_graph(5)
        diameter = G.get_graph_diameter()
        assert diameter == 2
    
    def test_diameter_cycle_graph(self):
        """Cycle graph diameter is floor(n/2)."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        diameter = G.get_graph_diameter()
        assert diameter == 3  # Opposite side of cycle
    
    def test_diameter_disconnected_graph(self):
        """Disconnected graph has no diameter."""
        G = ec.Graph()
        G.add_path_graph(3)
        G.add_node('X')  # Isolated node
        diameter = G.get_graph_diameter()
        assert diameter is None


class TestGraphRadius:
    """Test get_graph_radius() method."""
    
    def test_radius_empty_graph(self):
        """Empty graph has no radius."""
        G = ec.Graph()
        radius = G.get_graph_radius()
        assert radius is None
    
    def test_radius_single_node(self):
        """Single node has radius 0."""
        G = ec.Graph()
        G.add_node('A')
        radius = G.get_graph_radius()
        assert radius == 0
    
    def test_radius_path_graph(self):
        """Path graph radius is ceil((n-1)/2)."""
        G = ec.Graph()
        G.add_path_graph(5)
        radius = G.get_graph_radius()
        assert radius == 2  # Center node has eccentricity 2
    
    def test_radius_star_graph(self):
        """Star graph radius is 1 (hub has eccentricity 1)."""
        G = ec.Graph()
        G.add_star_graph(5)
        radius = G.get_graph_radius()
        assert radius == 1
    
    def test_radius_cycle_graph(self):
        """Cycle graph radius is floor(n/2)."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        radius = G.get_graph_radius()
        assert radius == 3
    
    def test_radius_disconnected_graph(self):
        """Disconnected graph has no radius."""
        G = ec.Graph()
        G.add_edge('A', 'B')
        G.add_edge('C', 'D')
        radius = G.get_graph_radius()
        assert radius is None


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
        # Create a graph with a triangle and a square
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')  # Triangle (girth 3)
        G.add_edge('C', 'D')
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'C')  # Larger cycle
        
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
        G.add_edge('A', 'B')
        G.add_edge('C', 'D')
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
        G.add_edge('A', 'B')
        
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
        G.add_node('A')
        G.add_node('B')
        cohesion = G.get_cohesion()
        assert cohesion == 0
    
    def test_cohesion_path_graph(self):
        """Path graph has cohesion 1 (single vertex cut)."""
        G = ec.Graph()
        G.add_path_graph(5)
        cohesion = G.get_cohesion()
        assert cohesion == 1
    
    def test_cohesion_cycle_graph(self):
        """Cycle graph has cohesion 2 (need 2 vertex cuts)."""
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
        G.add_edge('A', 'B')
        
        with pytest.raises(ValueError):
            G.get_cohesion()


class TestAdvancedPropertiesIntegration:
    """Integration tests for advanced structural properties."""
    
    def test_properties_on_path_graph(self):
        """Test multiple properties on path graph."""
        G = ec.Graph()
        G.add_path_graph(7)
        
        assert G.get_graph_diameter() == 6
        assert G.get_graph_radius() == 3
        assert G.get_girth() is None  # Trees have no cycles
        assert len(G.get_periphery()) == 2
        assert len(G.get_articulation_points()) == 5  # Middle 5 nodes
        assert G.get_adhesion() == 1
        assert G.get_cohesion() == 1
    
    def test_properties_on_cycle_graph(self):
        """Test multiple properties on cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(8)
        
        assert G.get_graph_diameter() == 4
        assert G.get_graph_radius() == 4
        assert G.get_girth() == 8
        assert len(G.get_periphery()) == 8  # All nodes in even cycle
        assert G.get_articulation_points() == []  # No APs in cycle
        assert G.get_adhesion() == 2
        assert G.get_cohesion() == 2
    
    def test_properties_on_complete_graph(self):
        """Test multiple properties on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        
        assert G.get_graph_diameter() == 1
        assert G.get_graph_radius() == 1
        assert G.get_girth() == 3  # Triangles
        assert len(G.get_periphery()) == 5  # All nodes have same eccentricity
        assert G.get_articulation_points() == []  # Highly connected
        assert G.get_adhesion() == 4
        assert G.get_cohesion() == 4
    
    def test_histogram_matches_degree_distribution(self):
        """get_degree_histogram() should match degree_distribution()."""
        G = ec.Graph()
        G.add_star_graph(10)
        
        hist = G.get_degree_histogram()
        dist = G.degree_distribution()
        
        assert hist == dist
