import warnings

import networkx as nx
import pytest

import edgecase as ec

warnings.filterwarnings("ignore")

# Import igraph if available
try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


def create_test_graphs():
    """Create diverse graph types for girth validation."""
    graphs = []

    # 1. Self-loop (girth = 1)
    nx_g = nx.Graph()
    nx_g.add_node(0)
    nx_g.add_edge(0, 0)  # Self-loop
    graphs.append(("self_loop", nx_g, False, 1))

    # 2. Multi-edge (girth = 2)
    nx_g = nx.MultiGraph()
    nx_g.add_edge(0, 1)
    nx_g.add_edge(0, 1)  # Duplicate edge
    graphs.append(("multi_edge", nx_g, False, 2))

    # 3. Triangle (girth = 3)
    nx_g = nx.cycle_graph(3)
    graphs.append(("triangle", nx_g, False, 3))

    # 4. Square (girth = 4)
    nx_g = nx.cycle_graph(4)
    graphs.append(("square", nx_g, False, 4))

    # 5. Pentagon (girth = 5)
    nx_g = nx.cycle_graph(5)
    graphs.append(("pentagon", nx_g, False, 5))

    # 6. Complete graph K5 (many triangles, girth = 3)
    nx_g = nx.complete_graph(5)
    graphs.append(("complete_5", nx_g, False, 3))

    # 7. Star graph (no cycles, acyclic)
    nx_g = nx.star_graph(10)
    graphs.append(("star_11", nx_g, False, None))

    # 8. Tree (acyclic)
    nx_g = nx.balanced_tree(2, 3)
    graphs.append(("tree_2_3", nx_g, False, None))

    # 9. Path graph (acyclic)
    nx_g = nx.path_graph(10)
    graphs.append(("path_10", nx_g, False, None))

    # 10. Grid graph (girth = 4)
    nx_g = nx.grid_2d_graph(4, 4)
    # Relabel nodes to integers
    nx_g = nx.convert_node_labels_to_integers(nx_g)
    graphs.append(("grid_4x4", nx_g, False, 4))

    # 11. Peterson graph (famous graph with girth = 5)
    nx_g = nx.petersen_graph()
    graphs.append(("petersen", nx_g, False, 5))

    # 12. Karate club (real-world network with triangles)
    nx_g = nx.karate_club_graph()
    graphs.append(("karate_club", nx_g, False, 3))

    # 13. Random sparse graph (likely has triangles)
    nx_g = nx.erdos_renyi_graph(30, 0.1, seed=23, directed=False)
    graphs.append(("random_sparse_30", nx_g, False, None))  # Unknown girth

    # 14. Bipartite graph (girth = 4 if has cycle, else None)
    nx_g = nx.complete_bipartite_graph(3, 3)
    graphs.append(("bipartite_3_3", nx_g, False, 4))

    # 15. Ladder graph (girth = 4)
    nx_g = nx.ladder_graph(5)
    graphs.append(("ladder_5", nx_g, False, 4))

    # 16. Circular ladder (Prism graph, girth = 4)
    nx_g = nx.circular_ladder_graph(6)
    graphs.append(("circular_ladder_6", nx_g, False, 4))

    # 17. Dodecahedral graph (girth = 5)
    nx_g = nx.dodecahedral_graph()
    graphs.append(("dodecahedral", nx_g, False, 5))

    # 18. Cubical graph (girth = 4)
    nx_g = nx.cubical_graph()
    graphs.append(("cubical", nx_g, False, 4))

    # 19. Diamond graph (girth = 3)
    nx_g = nx.diamond_graph()
    graphs.append(("diamond", nx_g, False, 3))

    # 20. Bull graph (girth = 3)
    nx_g = nx.bull_graph()
    graphs.append(("bull", nx_g, False, 3))

    return graphs


def nx_to_edgecase(nx_graph, directed=False):
    """Convert NetworkX graph to EdgeCase."""
    ec_graph = ec.Graph(directed=directed)

    # Add nodes
    for node in nx_graph.nodes():
        ec_graph.add_node(str(node))

    # Add edges
    for u, v in nx_graph.edges():
        ec_graph.add_edge(str(u), str(v))

    return ec_graph


def nx_to_igraph(nx_graph, directed=False):
    """Convert NetworkX graph to igraph."""
    node_list = list(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=directed)

    return ig_graph, node_list


def compute_girth_networkx(nx_graph):
    """
    Compute girth using NetworkX.

    NetworkX doesn't have a built-in girth function, so we compute it manually.
    """
    # Check for self-loops (girth = 1)
    for node in nx_graph.nodes():
        if nx_graph.has_edge(node, node):
            return 1

    # Check for multi-edges (girth = 2) - only for MultiGraph
    if isinstance(nx_graph, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v in nx_graph.edges():
            if nx_graph.number_of_edges(u, v) > 1:
                return 2

    # Find shortest cycle using BFS
    min_cycle_length = float("inf")

    for start_node in nx_graph.nodes():
        # BFS to find shortest cycle containing start_node
        visited = {start_node: (None, 0)}  # node: (parent, distance)
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            current_dist = visited[node][1]

            for neighbor in nx_graph.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = (node, current_dist + 1)
                    queue.append(neighbor)
                elif visited[node][0] != neighbor:  # Not the parent
                    # Found a cycle
                    cycle_length = current_dist + visited[neighbor][1] + 1
                    min_cycle_length = min(min_cycle_length, cycle_length)

    return min_cycle_length if min_cycle_length != float("inf") else None


@pytest.mark.parametrize("name,nx_graph,directed,expected_girth", create_test_graphs())
def test_girth_vs_networkx(name, nx_graph, directed, expected_girth):
    """Test EdgeCase girth against NetworkX implementation."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"  Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")
    print(f"{'=' * 70}")

    # Convert to EdgeCase
    ec_graph = nx_to_edgecase(nx_graph, directed)

    # Compute girth
    ec_girth = ec_graph.get_girth()
    nx_girth = compute_girth_networkx(nx_graph)

    print(f"EdgeCase girth: {ec_girth}")
    print(f"NetworkX girth: {nx_girth}")
    if expected_girth is not None:
        print(f"Expected girth: {expected_girth}")

    # Validate
    assert ec_girth == nx_girth, f"Girth mismatch for {name}: EC={ec_girth}, NX={nx_girth}"

    if expected_girth is not None:
        assert ec_girth == expected_girth, (
            f"Girth mismatch with expected for {name}: EC={ec_girth}, Expected={expected_girth}"
        )

    print(f"✅ {name}: girth={ec_girth} (PASS)")


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not available")
@pytest.mark.parametrize("name,nx_graph,directed,expected_girth", create_test_graphs())
def test_girth_vs_igraph(name, nx_graph, directed, expected_girth):
    """Test EdgeCase girth against igraph implementation."""
    # Skip multi-edge test for igraph (different handling)
    if name == "multi_edge":
        pytest.skip("igraph handles multi-edges differently")

    # Skip self-loop test (igraph removes self-loops, returns inf)
    if name == "self_loop":
        pytest.skip("igraph removes self-loops during preprocessing")

    print(f"\n{'=' * 70}")
    print(f"Testing: {name} (vs igraph)")
    print(f"  Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")
    print(f"{'=' * 70}")

    # Convert to EdgeCase
    ec_graph = nx_to_edgecase(nx_graph, directed)

    # Convert to igraph
    ig_graph, node_list = nx_to_igraph(nx_graph, directed)

    # Compute girth
    ec_girth = ec_graph.get_girth()
    try:
        ig_girth = ig_graph.girth()
    except Exception as e:
        print(f"⚠️  igraph error: {e}")
        pytest.skip(f"igraph error: {e}")

    print(f"EdgeCase girth: {ec_girth}")
    print(f"igraph girth:   {ig_girth}")
    if expected_girth is not None:
        print(f"Expected girth: {expected_girth}")

    # Handle different conventions for acyclic graphs
    # igraph returns inf, EdgeCase returns None
    if ec_girth is None and ig_girth == float("inf"):
        print(f"✅ {name}: Both agree graph is acyclic (EC=None, IG=inf)")
        return

    # Validate
    assert ec_girth == ig_girth, f"Girth mismatch for {name}: EC={ec_girth}, IG={ig_girth}"

    if expected_girth is not None:
        assert ec_girth == expected_girth, (
            f"Girth mismatch with expected for {name}: EC={ec_girth}, Expected={expected_girth}"
        )

    print(f"✅ {name}: girth={ec_girth} (PASS)")


def _build(G, directed):
    g = ec.Graph(directed=directed).add_nodes([str(n) for n in G])
    g.add_edges([(str(u), str(v)) for u, v in G.edges()])
    return g


def test_girth_after_an_acyclic_bfs():
    # A BFS that finds no cycle used to mark its nodes as done, so later searches skipped
    # the triangle 0, 1, 3 and reported the 7-cycle walk instead
    g = _build(nx.Graph([(0, 1), (0, 3), (1, 3), (2, 4), (3, 4), (4, 5)]), False)
    assert g.get_girth() == 3


def test_directed_girth():
    # A diamond has no directed cycle; a <-> b is a 2-cycle
    assert _build(nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)]), True).get_girth() is None
    assert _build(nx.DiGraph([(0, 1), (1, 0)]), True).get_girth() == 2
    assert _build(nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)]), True).get_girth() == 3


@pytest.mark.parametrize("directed", [False, True])
def test_random_girth_matches_networkx(directed):
    for seed in range(300):
        G = nx.gnp_random_graph(3 + seed % 8, (0.1, 0.2, 0.3, 0.5)[seed % 4], seed, directed)
        if directed:
            lengths = [len(c) for c in nx.simple_cycles(G)]
            expected = min(lengths) if lengths else None
        else:
            expected = nx.girth(G)
            expected = None if expected == float("inf") else expected
        assert _build(G, directed).get_girth() == expected, list(G.edges())
