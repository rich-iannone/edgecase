"""
Integration tests for validating EdgeCase algorithms against NetworkX.

These tests convert graphs between EdgeCase and NetworkX to verify that
our implementations produce consistent results with the well-established
NetworkX library.
"""

import pytest
import math


# Check if NetworkX is available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

pytestmark = pytest.mark.skipif(
    not HAS_NETWORKX,
    reason="NetworkX is not installed"
)

from edgecase import Graph, pagerank, shortest_path
from edgecase.networkx import to_networkx, from_networkx


def graphs_are_similar(G_ec, G_nx):
    """Helper to verify two graphs have the same structure."""
    # Convert EdgeCase to NetworkX for comparison
    if isinstance(G_ec, Graph):
        G_ec = to_networkx(G_ec)
    
    if G_ec.number_of_nodes() != G_nx.number_of_nodes():
        return False
    if G_ec.number_of_edges() != G_nx.number_of_edges():
        return False
    if G_ec.is_directed() != G_nx.is_directed():
        return False
    
    return True


def create_test_graphs():
    """Create a variety of test graphs for validation."""
    graphs = []
    
    # Simple directed graph
    G1 = Graph(directed=True)
    G1.add_edge("A", "B", weight=1.0)
    G1.add_edge("B", "C", weight=1.0)
    G1.add_edge("C", "A", weight=1.0)
    graphs.append(("directed_cycle", G1))
    
    # Simple undirected graph
    G2 = Graph(directed=False)
    G2.add_edge("A", "B", weight=1.0)
    G2.add_edge("B", "C", weight=1.0)
    G2.add_edge("C", "D", weight=1.0)
    G2.add_edge("D", "A", weight=1.0)
    graphs.append(("undirected_square", G2))
    
    # Star graph
    G3 = Graph(directed=False)
    G3.add_edge("center", "A", weight=1.0)
    G3.add_edge("center", "B", weight=1.0)
    G3.add_edge("center", "C", weight=1.0)
    G3.add_edge("center", "D", weight=1.0)
    graphs.append(("star", G3))
    
    # Weighted directed graph
    G4 = Graph(directed=True, weighted=True)
    G4.add_edge("A", "B", weight=1.0)
    G4.add_edge("A", "C", weight=2.0)
    G4.add_edge("B", "C", weight=1.0)
    G4.add_edge("C", "D", weight=3.0)
    G4.add_edge("B", "D", weight=5.0)
    graphs.append(("weighted_dag", G4))
    
    # Complete graph K5
    G5 = Graph(directed=False)
    nodes = ["A", "B", "C", "D", "E"]
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            G5.add_edge(u, v, weight=1.0)
    graphs.append(("complete_k5", G5))
    
    return graphs


class TestPageRankValidation:
    """Validate PageRank implementation against NetworkX."""
    
    def test_pagerank_simple_directed(self):
        """Compare PageRank on simple directed graph."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        
        # EdgeCase PageRank
        ec_ranks = pagerank(G, alpha=0.85, max_iterations=100, tolerance=1e-6)
        
        # NetworkX PageRank
        nx_graph = to_networkx(G)
        nx_ranks = nx.pagerank(nx_graph, alpha=0.85, max_iter=100, tol=1e-6)
        
        # Compare results (should be very close)
        for node in G.nodes():
            assert abs(ec_ranks[node] - nx_ranks[node]) < 0.01, \
                f"PageRank mismatch for node {node}: EC={ec_ranks[node]}, NX={nx_ranks[node]}"
    
    @pytest.mark.parametrize("name,graph", create_test_graphs())
    def test_pagerank_various_graphs(self, name, graph):
        """Compare PageRank on various graph types."""
        if not graph.is_directed:
            pytest.skip("PageRank requires directed graphs")
        
        # EdgeCase PageRank
        ec_ranks = pagerank(graph, alpha=0.85, max_iterations=100, tolerance=1e-6)
        
        # NetworkX PageRank
        nx_graph = to_networkx(graph)
        nx_ranks = nx.pagerank(nx_graph, alpha=0.85, max_iter=100, tol=1e-6)
        
        # Compare results
        for node in graph.nodes():
            assert abs(ec_ranks[node] - nx_ranks[node]) < 0.01, \
                f"{name}: PageRank mismatch for node {node}"
    
    def test_pagerank_normalization(self):
        """Verify PageRank scores sum to 1.0."""
        G = Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "A")
        
        ec_ranks = pagerank(G)
        nx_graph = to_networkx(G)
        nx_ranks = nx.pagerank(nx_graph)
        
        # Both should sum to ~1.0
        assert abs(sum(ec_ranks.values()) - 1.0) < 0.001
        assert abs(sum(nx_ranks.values()) - 1.0) < 0.001


class TestShortestPathValidation:
    """Validate shortest path implementation against NetworkX."""
    
    def test_shortest_path_simple(self):
        """Compare shortest path on simple graph."""
        G = Graph(directed=True)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("A", "C", weight=3.0)
        
        # EdgeCase shortest path
        ec_path = shortest_path(G, "A", "C")
        
        # NetworkX shortest path
        nx_graph = to_networkx(G)
        nx_path = nx.shortest_path(nx_graph, "A", "C", weight="weight")
        
        assert ec_path == nx_path
    
    def test_shortest_path_weighted(self):
        """Compare shortest path on weighted graph."""
        G = Graph(directed=True, weighted=True)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)
        G.add_edge("A", "D", weight=1.0)
        G.add_edge("D", "C", weight=1.0)
        
        # EdgeCase shortest path (A->D->C is shorter than A->B->C)
        ec_path = shortest_path(G, "A", "C")
        
        # NetworkX shortest path
        nx_graph = to_networkx(G)
        nx_path = nx.shortest_path(nx_graph, "A", "C", weight="weight")
        
        # Both should find the same path
        assert ec_path == nx_path
    
    @pytest.mark.parametrize("name,graph", create_test_graphs())
    def test_shortest_path_various_graphs(self, name, graph):
        """Compare shortest paths on various graph types."""
        nodes = graph.nodes()
        if len(nodes) < 2:
            pytest.skip("Need at least 2 nodes")
        
        source = nodes[0]
        target = nodes[-1]
        
        # EdgeCase shortest path
        try:
            ec_path = shortest_path(graph, source, target)
        except:
            ec_path = None
        
        # NetworkX shortest path
        nx_graph = to_networkx(graph)
        try:
            nx_path = nx.shortest_path(nx_graph, source, target, weight="weight")
        except nx.NetworkXNoPath:
            nx_path = None
        
        # Both should either find a path or fail
        if ec_path is None:
            assert nx_path is None, f"{name}: EdgeCase found no path but NetworkX did"
        elif nx_path is None:
            assert ec_path is None, f"{name}: NetworkX found no path but EdgeCase did"
        else:
            # If both found paths, they should be the same length
            # (there might be multiple shortest paths, so we just check length)
            assert len(ec_path) == len(nx_path), \
                f"{name}: Path length mismatch: EC={len(ec_path)}, NX={len(nx_path)}"


class TestGraphPropertiesValidation:
    """Validate graph properties against NetworkX."""
    
    @pytest.mark.parametrize("name,graph", create_test_graphs())
    def test_node_count(self, name, graph):
        """Verify node count matches NetworkX."""
        nx_graph = to_networkx(graph)
        assert graph.num_nodes() == nx_graph.number_of_nodes(), \
            f"{name}: Node count mismatch"
    
    @pytest.mark.parametrize("name,graph", create_test_graphs())
    def test_edge_count(self, name, graph):
        """Verify edge count matches NetworkX."""
        nx_graph = to_networkx(graph)
        assert graph.num_edges() == nx_graph.number_of_edges(), \
            f"{name}: Edge count mismatch"
    
    @pytest.mark.parametrize("name,graph", create_test_graphs())
    def test_directionality(self, name, graph):
        """Verify directionality matches NetworkX."""
        nx_graph = to_networkx(graph)
        assert graph.is_directed == nx_graph.is_directed(), \
            f"{name}: Directionality mismatch"
    
    def test_density(self):
        """Verify graph density calculation."""
        G = Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")  # Triangle
        
        nx_graph = to_networkx(G)
        
        ec_density = G.density()
        nx_density = nx.density(nx_graph)
        
        assert abs(ec_density - nx_density) < 0.001, \
            f"Density mismatch: EC={ec_density}, NX={nx_density}"
    
    def test_degree_sequence(self):
        """Verify degree sequences match."""
        G = Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "A")
        G.add_edge("A", "C")
        
        nx_graph = to_networkx(G)
        
        # Get degree for each node
        for node in G.nodes():
            ec_degree = G.degree(node)
            nx_degree = nx_graph.degree[node]
            assert ec_degree == nx_degree, \
                f"Degree mismatch for node {node}: EC={ec_degree}, NX={nx_degree}"


class TestConversionConsistency:
    """Test that conversions preserve graph semantics."""
    
    def test_karate_club_graph(self):
        """Test on Zachary's Karate Club graph from NetworkX."""
        # Create from NetworkX famous graph
        nx_graph = nx.karate_club_graph()
        
        # Convert to EdgeCase
        G = from_networkx(nx_graph)
        
        # Basic properties should match
        assert G.num_nodes() == nx_graph.number_of_nodes()
        assert G.num_edges() == nx_graph.number_of_edges()
        assert not G.is_directed
        
        # Convert back to NetworkX
        nx_graph2 = to_networkx(G)
        
        # Should have same structure
        assert nx_graph2.number_of_nodes() == nx_graph.number_of_nodes()
        assert nx_graph2.number_of_edges() == nx_graph.number_of_edges()
    
    def test_erdos_renyi_graph(self):
        """Test on random Erdős-Rényi graph."""
        # Generate random graph with NetworkX
        nx_graph = nx.erdos_renyi_graph(n=20, p=0.3, seed=42)
        
        # Convert to EdgeCase
        G = from_networkx(nx_graph)
        
        # Should preserve structure
        assert G.num_nodes() == 20
        assert not G.is_directed
        
        # Convert back
        nx_graph2 = to_networkx(G)
        assert nx_graph2.number_of_nodes() == nx_graph.number_of_nodes()
        assert nx_graph2.number_of_edges() == nx_graph.number_of_edges()
    
    def test_path_graph(self):
        """Test on simple path graph."""
        # Create path graph in NetworkX
        nx_graph = nx.path_graph(10)
        
        # Convert to EdgeCase
        G = from_networkx(nx_graph)
        
        assert G.num_nodes() == 10
        assert G.num_edges() == 9
        assert not G.is_directed
        
        # Verify it's still a path (each node has degree 1 or 2)
        for node in G.nodes():
            degree = G.degree(node)
            assert degree in [1, 2], f"Invalid degree {degree} for path graph"


class TestAlgorithmConsistency:
    """Test that algorithms produce consistent results across conversions."""
    
    def test_pagerank_after_conversion(self):
        """PageRank should be consistent after conversion."""
        # Start with NetworkX graph
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")])
        
        # Get NetworkX PageRank
        nx_ranks1 = nx.pagerank(nx_graph)
        
        # Convert to EdgeCase and back
        G = from_networkx(nx_graph)
        nx_graph2 = to_networkx(G)
        nx_ranks2 = nx.pagerank(nx_graph2)
        
        # Results should be identical
        for node in nx_ranks1:
            assert abs(nx_ranks1[node] - nx_ranks2[node]) < 0.0001
    
    def test_shortest_path_after_conversion(self):
        """Shortest path should be consistent after conversion."""
        # Start with NetworkX graph
        nx_graph = nx.DiGraph()
        nx_graph.add_weighted_edges_from([
            ("A", "B", 1.0),
            ("B", "C", 2.0),
            ("A", "C", 5.0),
        ])
        
        # Get NetworkX shortest path
        nx_path1 = nx.shortest_path(nx_graph, "A", "C", weight="weight")
        
        # Convert to EdgeCase and back
        G = from_networkx(nx_graph)
        nx_graph2 = to_networkx(G)
        nx_path2 = nx.shortest_path(nx_graph2, "A", "C", weight="weight")
        
        # Paths should be identical
        assert nx_path1 == nx_path2
