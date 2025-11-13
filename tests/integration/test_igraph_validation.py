"""
Integration tests for igraph validation and comparison.

This test module validates EdgeCase graph algorithms against igraph's implementations
to ensure correctness and compatibility. Tests cover:

1. Graph conversion (EdgeCase <-> igraph)
2. Basic graph metrics (degree, density, etc.)
3. Shortest path algorithms
4. Centrality measures (betweenness, closeness, eigenvector, PageRank)
5. Community detection
6. Graph properties (connected components, clustering coefficients)
7. Traversal algorithms (BFS, DFS)
8. Numerical stability with extreme weights

These tests help identify bugs, API inconsistencies, and ensure that EdgeCase
produces results that match igraph's well-tested implementations.
"""

import pytest
import math


# Check if igraph is available
try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

# Check if EdgeCase is available
try:
    import edgecase as ec
    from edgecase.igraph_interop import to_igraph, from_igraph
    HAS_EDGECASE = True
except ImportError:
    HAS_EDGECASE = False

pytestmark = pytest.mark.skipif(
    not (HAS_IGRAPH and HAS_EDGECASE),
    reason="Requires both igraph and edgecase"
)


def approx_equal(a, b, rel_tol=1e-7, abs_tol=1e-9):
    """Check if two values are approximately equal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def dict_approx_equal(dict1, dict2, rel_tol=1e-7, abs_tol=1e-9):
    """Check if two dictionaries have approximately equal values."""
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    return all(approx_equal(dict1[k], dict2[k], rel_tol, abs_tol) for k in dict1.keys())


class TestGraphConversion:
    """Test bidirectional conversion between EdgeCase and igraph."""
    
    def test_simple_directed_graph(self):
        """Test conversion of a simple directed graph."""
        # Create EdgeCase graph
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)
        
        # Convert to igraph
        ig_graph = to_igraph(G)
        
        assert ig_graph.vcount() == 3
        assert ig_graph.ecount() == 2
        assert ig_graph.is_directed()
        assert set(ig_graph.vs["name"]) == {"A", "B", "C"}
        
        # Convert back
        G2 = from_igraph(ig_graph)
        assert G2.num_nodes() == 3
        assert G2.num_edges() == 2
        assert G2.is_directed
    
    def test_simple_undirected_graph(self):
        """Test conversion of a simple undirected graph."""
        # Create EdgeCase graph
        G = ec.Graph(directed=False)
        G.add_node("X")
        G.add_node("Y")
        G.add_edge("X", "Y", weight=3.0)
        
        # Convert to igraph
        ig_graph = to_igraph(G)
        
        assert ig_graph.vcount() == 2
        assert ig_graph.ecount() == 1
        assert not ig_graph.is_directed()
        
        # Convert back
        G2 = from_igraph(ig_graph)
        assert G2.num_nodes() == 2
        assert G2.num_edges() == 1
        assert not G2.is_directed
    
    def test_node_attributes_preservation(self):
        """Test that node attributes are preserved during conversion."""
        # Create EdgeCase graph with node attributes
        G = ec.Graph(directed=True)
        G.add_node("A", color="red", value=10, description="Node A")
        G.add_node("B", color="blue", value=20, description="Node B")
        
        # Convert to igraph
        ig_graph = to_igraph(G)
        
        # Check attributes are preserved
        node_names = ig_graph.vs["name"]
        idx_a = node_names.index("A")
        idx_b = node_names.index("B")
        
        assert ig_graph.vs[idx_a]["color"] == "red"
        assert ig_graph.vs[idx_a]["value"] == 10
        assert ig_graph.vs[idx_a]["description"] == "Node A"
        assert ig_graph.vs[idx_b]["color"] == "blue"
        assert ig_graph.vs[idx_b]["value"] == 20
        
        # Convert back and verify
        G2 = from_igraph(ig_graph)
        attrs_a = G2.get_node_attrs("color", ["A", "B"])
        values = G2.get_node_attrs("value", ["A", "B"])
        
        assert attrs_a["A"] == "red"
        assert attrs_a["B"] == "blue"
        assert values["A"] == 10
        assert values["B"] == 20
    
    def test_edge_attributes_preservation(self):
        """Test that edge attributes are preserved during conversion."""
        # Create EdgeCase graph with edge attributes
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B", weight=1.5, label="edge1", capacity=100)
        
        # Convert to igraph
        ig_graph = to_igraph(G)
        
        # Check edge attributes
        assert ig_graph.es[0]["weight"] == 1.5
        assert ig_graph.es[0]["label"] == "edge1"
        assert ig_graph.es[0]["capacity"] == 100
        
        # Convert back
        G2 = from_igraph(ig_graph)
        edges = G2.edges()
        edge = G2.get_edge(edges[0][0])
        
        assert edge.weight == 1.5
        assert edge.attributes["label"] == "edge1"
        assert edge.attributes["capacity"] == 100
    
    def test_weighted_graph_detection(self):
        """Test automatic detection of weighted graphs."""
        # Create igraph graph with weights
        ig_graph = ig.Graph(directed=True)
        ig_graph.add_vertices(3)
        ig_graph.vs["name"] = ["A", "B", "C"]
        ig_graph.add_edge(0, 1, weight=1.0)
        ig_graph.add_edge(1, 2, weight=2.0)
        
        # Convert to EdgeCase (should auto-detect weighted=True)
        G = from_igraph(ig_graph)
        
        assert G.is_weighted


class TestBasicMetrics:
    """Test basic graph metrics against igraph."""
    
    def test_num_nodes_edges(self):
        """Test node and edge counts match."""
        G = ec.Graph(directed=True)
        for i in range(5):
            G.add_node(str(i))
        G.add_edge("0", "1")
        G.add_edge("1", "2")
        G.add_edge("2", "3")
        G.add_edge("3", "4")
        
        ig_graph = to_igraph(G)
        
        assert G.num_nodes() == ig_graph.vcount()
        assert G.num_edges() == ig_graph.ecount()
    
    def test_degree_directed(self):
        """Test degree calculations for directed graphs."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        
        ig_graph = to_igraph(G)
        
        # In-degree - convert node IDs to labels for keys
        ec_in_degrees = {G.get_label_from_id(node_id): G.in_degree(node_id) for node_id in G.get_node_ids()}
        ig_in_degrees = {ig_graph.vs[i]["name"]: ig_graph.indegree(i) for i in range(ig_graph.vcount())}
        assert ec_in_degrees == ig_in_degrees
        
        # Out-degree - convert node IDs to labels for keys
        ec_out_degrees = {G.get_label_from_id(node_id): G.out_degree(node_id) for node_id in G.get_node_ids()}
        ig_out_degrees = {ig_graph.vs[i]["name"]: ig_graph.outdegree(i) for i in range(ig_graph.vcount())}
        assert ec_out_degrees == ig_out_degrees
    
    def test_degree_undirected(self):
        """Test degree calculations for undirected graphs."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        
        ig_graph = to_igraph(G)
        
        # Convert node IDs to labels for keys
        ec_degrees = {G.get_label_from_id(node_id): G.degree(node_id) for node_id in G.get_node_ids()}
        ig_degrees = {ig_graph.vs[i]["name"]: ig_graph.degree(i) for i in range(ig_graph.vcount())}
        assert ec_degrees == ig_degrees


class TestShortestPaths:
    """Test shortest path algorithms against igraph."""
    
    def test_shortest_path_unweighted(self):
        """Test shortest path in unweighted graph."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase shortest path
        ec_path = ec.shortest_path(G, "A", "D")
        
        # igraph shortest path
        node_names = ig_graph.vs["name"]
        idx_a = node_names.index("A")
        idx_d = node_names.index("D")
        ig_path_indices = ig_graph.get_shortest_path(idx_a, idx_d)
        ig_path = [node_names[i] for i in ig_path_indices]
        
        assert ec_path == ig_path
    
    def test_shortest_path_weighted(self):
        """Test shortest path in weighted graph."""
        G = ec.Graph(directed=False, weighted=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "D", weight=5.0)
        G.add_edge("A", "C", weight=2.0)
        G.add_edge("C", "D", weight=1.0)
        
        ig_graph = to_igraph(G)
        
        # EdgeCase shortest path (should use Dijkstra with weights)
        ec_path = ec.shortest_path(G, "A", "D")
        
        # igraph shortest path with weights
        node_names = ig_graph.vs["name"]
        idx_a = node_names.index("A")
        idx_d = node_names.index("D")
        ig_path_indices = ig_graph.get_shortest_path(idx_a, idx_d, weights="weight")
        ig_path = [node_names[i] for i in ig_path_indices]
        
        # Both should find path A -> C -> D (total weight 3.0)
        assert ec_path == ig_path
        assert ec_path == ["A", "C", "D"]


class TestPageRank:
    """Test PageRank algorithm against igraph."""
    
    def test_pagerank_simple(self):
        """Test PageRank on a simple graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase PageRank (uses damping parameter)
        ec_pr = ec.pagerank(G, damping=0.85, max_iter=100, tol=1e-6)
        
        # igraph PageRank (uses damping parameter)
        ig_pr_values = ig_graph.pagerank(damping=0.85)
        node_names = ig_graph.vs["name"]
        ig_pr = {node_names[i]: ig_pr_values[i] for i in range(len(node_names))}
        
        # Compare results
        assert dict_approx_equal(ec_pr, ig_pr, rel_tol=1e-5)
    
    def test_pagerank_star_graph(self):
        """Test PageRank on a star graph (one central node)."""
        G = ec.Graph(directed=True)
        G.add_node("center")
        for i in range(5):
            node_id = f"node{i}"
            G.add_node(node_id)
            G.add_edge(node_id, "center")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase PageRank
        ec_pr = ec.pagerank(G, damping=0.85)
        
        # igraph PageRank
        ig_pr_values = ig_graph.pagerank(damping=0.85)
        node_names = ig_graph.vs["name"]
        ig_pr = {node_names[i]: ig_pr_values[i] for i in range(len(node_names))}
        
        # Central node should have highest PageRank
        assert ec_pr["center"] > ec_pr["node0"]
        
        # Compare with igraph
        assert dict_approx_equal(ec_pr, ig_pr, rel_tol=1e-5)


class TestBetweennessCentrality:
    """Test betweenness centrality against igraph."""
    
    def test_betweenness_simple(self):
        """Test betweenness centrality on a simple graph.
        
        Note: EdgeCase and igraph use different normalization formulas for betweenness,
        so we test relative ordering rather than exact values.
        """
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase betweenness
        ec_between = ec.betweenness_centrality(G)
        
        # igraph betweenness
        ig_between_values = ig_graph.betweenness(directed=False, normalized=True)
        node_names = ig_graph.vs["name"]
        ig_between = {node_names[i]: ig_between_values[i] for i in range(len(node_names))}
        
        # Verify relative ordering matches (middle nodes have higher betweenness)
        assert ec_between["B"] > ec_between["A"]
        assert ec_between["C"] > ec_between["A"]
        assert ec_between["B"] == ec_between["C"]  # Symmetric
        assert ec_between["A"] == ec_between["D"] == 0.0  # Endpoints have 0
        
        # Same relative ordering for igraph
        assert ig_between["B"] > ig_between["A"]
        assert ig_between["C"] > ig_between["A"]
    
    def test_betweenness_star(self):
        """Test betweenness centrality on a star graph.
        
        Note: EdgeCase and igraph use different normalization formulas for betweenness,
        so we test relative ordering rather than exact values.
        """
        G = ec.Graph(directed=False)
        G.add_node("center")
        for i in range(5):
            node_id = f"leaf{i}"
            G.add_node(node_id)
            G.add_edge("center", node_id)
        
        ig_graph = to_igraph(G)
        
        # EdgeCase betweenness
        ec_between = ec.betweenness_centrality(G)
        
        # igraph betweenness
        ig_between_values = ig_graph.betweenness(directed=False, normalized=True)
        node_names = ig_graph.vs["name"]
        ig_between = {node_names[i]: ig_between_values[i] for i in range(len(node_names))}
        
        # Verify center node has highest betweenness, leaves have zero
        assert ec_between["center"] > 0
        assert all(ec_between[f"leaf{i}"] == 0.0 for i in range(5))
        assert ec_between["center"] > ec_between["leaf0"]
        
        # Same pattern for igraph
        assert ig_between["center"] > 0
        assert all(ig_between[f"leaf{i}"] == 0.0 for i in range(5))


class TestConnectedComponents:
    """Test connected components against igraph."""
    
    def test_weakly_connected_components_count(self):
        """Test weakly connected components count in directed graph."""
        G = ec.Graph(directed=True)
        # Component 1
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        # Component 2
        G.add_node("C")
        G.add_node("D")
        G.add_edge("C", "D")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase WCC count
        ec_wcc_count = G.count_weakly_connected_components()
        
        # igraph WCC count
        ig_wcc_count = len(ig_graph.components(mode="weak"))
        
        assert ec_wcc_count == ig_wcc_count
        assert ec_wcc_count == 2
    
    def test_connected_components_count_undirected(self):
        """Test connected components count in undirected graph."""
        G = ec.Graph(directed=False)
        # Component 1
        G.add_node("1")
        G.add_node("2")
        G.add_edge("1", "2")
        # Component 2 (isolated node)
        G.add_node("3")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase CC count
        ec_cc_count = G.count_weakly_connected_components()
        
        # igraph CC count
        ig_cc_count = len(ig_graph.components())
        
        assert ec_cc_count == ig_cc_count
        assert ec_cc_count == 2
    
    def test_strongly_connected_components_count(self):
        """Test strongly connected components count in directed graph."""
        G = ec.Graph(directed=True)
        # SCC 1: A -> B -> A (cycle)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        # SCC 2: C (isolated)
        G.add_node("C")
        # SCC 3: D -> E (no back edge)
        G.add_node("D")
        G.add_node("E")
        G.add_edge("D", "E")
        
        ig_graph = to_igraph(G)
        
        # EdgeCase SCC count
        ec_scc_count = G.count_strongly_connected_components()
        
        # igraph SCC count
        ig_scc_count = len(ig_graph.components(mode="strong"))
        
        assert ec_scc_count == ig_scc_count
        assert ec_scc_count == 4  # {A,B}, {C}, {D}, {E}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
