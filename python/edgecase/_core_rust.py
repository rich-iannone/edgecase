"""
Python wrapper for Rust core to provide a consistent API.

This module wraps the raw Rust bindings to provide the same API
as the pure Python implementation.
"""

from typing import Optional, Callable, Any
from edgecase import _core as rust


class Graph:
    """Python wrapper around Rust Graph for consistent API."""
    
    def __init__(self, directed: bool = False, weighted: bool = False):
        self._graph = rust.Graph(directed=directed, weighted=weighted)
    
    def add_node(self, node_id, **attributes):
        """Add a node to the graph (convenience method).
        
        Args:
            node_id: Either a string node ID or a Node object
            **attributes: Optional node attributes (ignored if node_id is a Node)
        """
        # Handle both raw Node objects and string IDs
        if isinstance(node_id, rust.Node):
            # Direct Rust Node object
            self._graph.add_node(node_id)
        else:
            # String ID with attributes
            if not self._graph.has_node(node_id):
                node = rust.Node(id=node_id, attributes=attributes if attributes else None)
                self._graph.add_node(node)
    
    def add_edge(self, source, target=None, weight: float | None = None, edge_id: int | None = None, **attributes):
        """Add an edge to the graph (convenience method).
        
        Args:
            source: Either an Edge object or a source node ID
            target: Target node ID (required if source is a string)
            weight: Optional edge weight
            edge_id: Optional edge ID (auto-assigned if None)
            **attributes: Optional edge attributes
        """
        # Handle both raw Edge objects and (source, target) pairs
        if isinstance(source, rust.Edge):
            # Direct Rust Edge object
            self._graph.add_edge(source)
        else:
            # String IDs with attributes
            # Ensure nodes exist first (direct Rust hash lookup - O(1))
            if not self._graph.has_node(source):
                self.add_node(source)
            if not self._graph.has_node(target):
                self.add_node(target)
            
            edge = rust.Edge(
                id=edge_id if edge_id is not None else 0,  # 0 means auto-assign
                source=source,
                target=target,
                weight=weight,
                attributes=attributes if attributes else None
            )
            self._graph.add_edge(edge)
    
    def nodes(self) -> list[str]:
        """Return list of node IDs."""
        return self._graph.node_ids()
    
    def node_ids(self) -> list[str]:
        """Return list of node IDs (alias for nodes)."""
        return self._graph.node_ids()
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return self._graph.has_node(node_id)
    
    def edges(self) -> list[tuple[int, str, str]]:
        """Return list of edges as (id, source, target) tuples."""
        return self._graph.edges()
    
    def edge_ids(self) -> list[int]:
        """Return list of edge IDs."""
        return self._graph.edge_ids()
    
    def get_edge_by_id(self, edge_id: int) -> tuple[str, str] | None:
        """Get edge (source, target) by ID."""
        return self._graph.get_edge_by_id(edge_id)
    
    def has_edge_id(self, edge_id: int) -> bool:
        """Check if an edge with given ID exists."""
        return self._graph.has_edge_id(edge_id)
    
    def delete_edge_by_id(self, edge_id: int):
        """Delete edge by ID."""
        return self._graph.delete_edge_by_id(edge_id)
    
    def neighbors(self, node_id: str) -> list[str]:
        """Get neighbors of a node."""
        return self._graph.neighbors(node_id)
    
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return self._graph.num_nodes()
    
    def num_edges(self) -> int:
        """Return number of edges."""
        return self._graph.num_edges()
    
    @property
    def directed(self) -> bool:
        """Check if graph is directed."""
        return self._graph.is_directed
    
    @property
    def is_directed(self) -> bool:
        """Check if graph is directed (alias for directed)."""
        return self._graph.is_directed
    
    @property
    def weighted(self) -> bool:
        """Check if graph is weighted."""
        return self._graph.is_weighted
    
    @property
    def is_weighted(self) -> bool:
        """Check if graph is weighted (alias for weighted)."""
        return self._graph.is_weighted
    
    def degree(self, node_id: str) -> int:
        """Get the degree of a node.
        
        For directed graphs, returns total degree (in + out).
        For undirected graphs, returns the number of neighbors.
        
        Args:
            node_id: Node ID
            
        Returns:
            Degree of the node
        """
        neighbors = self._graph.neighbors(node_id)
        if self._graph.is_directed:
            # For directed, count both in and out
            # edges() returns (edge_id, source, target) tuples
            in_count = sum(1 for edge in self._graph.edges() if edge[2] == node_id)
            out_count = len(neighbors)
            return in_count + out_count
        else:
            return len(neighbors)
    
    def in_degree(self, node_id: str) -> int:
        """Get the in-degree of a node (directed graphs only).
        
        Args:
            node_id: Node ID
            
        Returns:
            Number of incoming edges
            
        Raises:
            ValueError: If graph is undirected
        """
        if not self._graph.is_directed:
            raise ValueError("in_degree only available for directed graphs")
        # edges() returns (edge_id, source, target) tuples
        return sum(1 for edge in self._graph.edges() if edge[2] == node_id)
    
    def out_degree(self, node_id: str) -> int:
        """Get the out-degree of a node (directed graphs only).
        
        Args:
            node_id: Node ID
            
        Returns:
            Number of outgoing edges
            
        Raises:
            ValueError: If graph is undirected
        """
        if not self._graph.is_directed:
            raise ValueError("out_degree only available for directed graphs")
        return len(self._graph.neighbors(node_id))
    
    # ========== Structural Properties ==========
    
    def density(self) -> float:
        """Calculate graph density (ratio of actual to possible edges).
        
        For directed graphs: density = E / (N * (N-1))
        For undirected graphs: density = 2E / (N * (N-1))
        
        Returns:
            Graph density between 0.0 and 1.0
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'C')
            >>> G.density()  # 2 edges, 3 nodes, undirected: 2/3 = 0.667
            0.6666666666666666
        """
        return self._graph.density()
    
    def is_connected(self) -> bool:
        """Check if graph is connected.
        
        For undirected graphs: checks if all nodes are reachable from any node.
        For directed graphs: checks weak connectivity (ignoring edge direction).
        
        Returns:
            True if graph is connected, False otherwise
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'C')
            >>> G.is_connected()
            True
            >>> G.add_node('D')  # Isolated node
            >>> G.is_connected()
            False
        """
        return self._graph.is_connected()
    
    def is_dag(self) -> bool:
        """Check if directed graph is a DAG (Directed Acyclic Graph).
        
        Returns:
            True if graph is a DAG, False if it contains cycles
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'C')
            >>> G.is_dag()
            True
            >>> G.add_edge('C', 'A')  # Creates cycle
            >>> G.is_dag()
            False
        """
        return self._graph.is_dag()
    
    def is_simple(self) -> bool:
        """Check if graph is simple (no self-loops or multi-edges).
        
        Returns:
            True if graph has no self-loops and no parallel edges
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.is_simple()
            True
            >>> G.add_edge('A', 'A')  # Self-loop
            >>> G.is_simple()
            False
        """
        return self._graph.is_simple()
    
    def is_bipartite(self) -> bool:
        """Check if graph is bipartite (2-colorable).
        
        A graph is bipartite if its nodes can be divided into two sets
        such that edges only connect nodes from different sets.
        
        Returns:
            True if graph is bipartite, False otherwise
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'C')
            >>> G.add_edge('C', 'A')  # Triangle - not bipartite
            >>> G.is_bipartite()
            False
        """
        return self._graph.is_bipartite()
    
    def degree_distribution(self) -> dict[int, int]:
        """Get degree distribution as a mapping from degree to count.
        
        Returns:
            Dictionary mapping degree values to the number of nodes with that degree
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(4)  # 1-2-2-1 degrees
            >>> G.degree_distribution()
            {1: 2, 2: 2}  # 2 nodes with degree 1, 2 nodes with degree 2
        """
        return self._graph.degree_distribution()
    
    def count_selfloops(self) -> int:
        """Count number of self-loops in the graph.
        
        Returns:
            Number of edges where source equals target
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('A', 'A')  # Self-loop
            >>> G.count_selfloops()
            1
        """
        return self._graph.count_selfloops()
    
    def average_degree(self) -> float:
        """Calculate average degree across all nodes.
        
        Returns:
            Mean degree value
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(4)  # All nodes have degree 2
            >>> G.average_degree()
            2.0
        """
        return self._graph.average_degree()
    
    def get_degree_histogram(self) -> dict[int, int]:
        """Get degree histogram for plotting.
        
        Returns a dictionary mapping degree values to their frequencies,
        suitable for creating degree distribution plots.
        
        Returns:
            Dictionary where keys are degree values and values are counts
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # 1 hub (degree 4) + 4 leaves (degree 1)
            >>> hist = G.get_degree_histogram()
            >>> hist
            {1: 4, 4: 1}
        """
        return self._graph.get_degree_histogram()
    
    def get_articulation_points(self) -> list[str]:
        """Find articulation points (cut vertices) in the graph.
        
        An articulation point is a node whose removal would disconnect
        the graph. Uses Tarjan's DFS-based algorithm.
        
        Only defined for undirected graphs.
        
        Returns:
            List of node IDs that are articulation points
            
        Raises:
            ValueError: If graph is directed
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear chain
            >>> ap = G.get_articulation_points()
            >>> len(ap)  # Middle 3 nodes are articulation points
            3
        """
        return self._graph.get_articulation_points()
    
    def get_periphery(self) -> list[str]:
        """Get periphery nodes (nodes with maximum eccentricity).
        
        Periphery nodes are the farthest from the graph's center.
        Eccentricity is the maximum distance to any other node.
        
        Returns:
            List of node IDs in the periphery, empty if graph is disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear: 0-1-2-3-4
            >>> periphery = G.get_periphery()
            >>> sorted(periphery)  # End nodes are periphery
            ['0', '4']
        """
        return self._graph.get_periphery()
    
    def get_graph_diameter(self) -> int | None:
        """Get graph diameter (maximum eccentricity).
        
        The diameter is the longest shortest path between any two nodes.
        Returns None if the graph is disconnected.
        
        Returns:
            Maximum distance between any two nodes, or None if disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear chain
            >>> G.get_graph_diameter()
            4
            >>> G.add_cycle_graph(6)  # Disconnected components
            >>> G.get_graph_diameter() is None
            True
        """
        return self._graph.get_graph_diameter()
    
    def get_graph_radius(self) -> int | None:
        """Get graph radius (minimum eccentricity).
        
        The radius is the minimum of all node eccentricities.
        Nodes with eccentricity equal to the radius form the center.
        Returns None if the graph is disconnected.
        
        Returns:
            Minimum eccentricity, or None if disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear chain
            >>> G.get_graph_radius()
            2
            >>> G.add_star_graph(5)  # Hub at center
            >>> G.get_graph_radius()
            1
        """
        return self._graph.get_graph_radius()
    
    def get_girth(self) -> int | None:
        """Get girth (length of shortest cycle).
        
        The girth is the length of the shortest cycle in the graph.
        Returns None if the graph is acyclic (a forest/tree).
        
        Uses BFS from each node to detect shortest cycles.
        
        Returns:
            Length of shortest cycle, or None if acyclic
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(6)  # Single 6-cycle
            >>> G.get_girth()
            6
            >>> G_tree = ec.Graph()
            >>> G_tree.add_balanced_tree(2, 3)  # Tree has no cycles
            >>> G_tree.get_girth() is None
            True
        """
        return self._graph.get_girth()
    
    def get_adhesion(self) -> int:
        """Get adhesion (minimum edge cut size).
        
        The adhesion is the minimum number of edges whose removal
        would disconnect the graph. Also called edge connectivity.
        
        Only defined for undirected graphs. Returns 0 if already disconnected.
        
        Note: Current implementation provides an upper bound (minimum degree)
        for efficiency. Exact computation requires max-flow algorithms.
        
        Returns:
            Minimum edge cut size
            
        Raises:
            ValueError: If graph is directed
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(5)  # Need to remove 2 edges to disconnect
            >>> G.get_adhesion()
            2
        """
        return self._graph.get_adhesion()
    
    def get_cohesion(self) -> int:
        """Get cohesion (minimum vertex cut size).
        
        The cohesion is the minimum number of nodes whose removal
        would disconnect the graph. Also called vertex connectivity.
        
        Only defined for undirected graphs. Returns 0 if already disconnected.
        
        Note: Current implementation provides an upper bound (minimum degree)
        for efficiency. Exact computation requires sophisticated algorithms.
        
        Returns:
            Minimum vertex cut size
            
        Raises:
            ValueError: If graph is directed
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(5)  # Need to remove 2 nodes to disconnect
            >>> G.get_cohesion()
            2
        """
        return self._graph.get_cohesion()
    
    def get_eccentricity(self) -> dict[str, int | None]:
        """Get eccentricity for each node.
        
        Eccentricity of a node is the maximum distance from that node to
        any other node in the graph. Returns None for a node if the graph
        is disconnected.
        
        Returns:
            Dictionary mapping node ID to eccentricity (None if disconnected)
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear chain
            >>> ecc = G.get_eccentricity()
            >>> ecc['3']  # Middle node has eccentricity 2
            2
        """
        return self._graph.get_eccentricity()
    
    def get_radiality(self) -> dict[str, float | None]:
        """Get radiality centrality for each node.
        
        Radiality measures how central a node is based on its eccentricity.
        Formula: radiality = (diameter + 1 - eccentricity) / (n - 1)
        
        Higher radiality means more central. Returns None if graph is disconnected.
        
        Returns:
            Dictionary mapping node ID to radiality score (None if disconnected)
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # Hub at center
            >>> rad = G.get_radiality()
            >>> # Hub has highest radiality
        """
        return self._graph.get_radiality()
    
    def get_closeness_vitality(self) -> dict[str, float]:
        """Get closeness vitality for each node.
        
        Closeness vitality measures a node's importance by computing the
        change in total graph distances when the node is removed.
        
        vitality(v) = sum_distances(G) - sum_distances(G - v)
        
        Higher values indicate more important/central nodes.
        
        Performance:
            - Uses parallel BFS (Rayon) for ~linear speedup with CPU cores
            - Preallocated data structures reduce memory allocations
            - Complexity: O(V²E / num_cores) with parallelization
            - Benchmarks (Apple M1, 8 cores):
                * 50 nodes: ~2ms (23K nodes/sec)
                * 200 nodes: ~120ms (1.6K nodes/sec)
                * 500 nodes: ~1.8s (273 nodes/sec)
        
        Note: Computationally expensive for large graphs. Consider using
        alternative centrality measures (degree, PageRank) for graphs
        with >1000 nodes unless high accuracy is required.
        
        Returns:
            Dictionary mapping node ID to vitality score
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # Hub should have high vitality
            >>> vitality = G.get_closeness_vitality()
            >>> # Hub removal greatly increases total distances
        """
        return self._graph.get_closeness_vitality()
    
    def get_coreness(self) -> dict[str, int]:
        """Get k-core decomposition (coreness for each node).
        
        The k-core is the maximal subgraph where every node has degree >= k.
        A node's coreness is the highest k for which it belongs to the k-core.
        
        Uses iterative pruning algorithm. Higher coreness indicates nodes
        in denser, more cohesive parts of the graph.
        
        Returns:
            Dictionary mapping node ID to coreness value
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_full_graph(5)  # Complete graph
            >>> coreness = G.get_coreness()
            >>> # All nodes have coreness 4 (K5 is 4-core)
        """
        return self._graph.get_coreness()
    
    def get_authority_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get HITS authority scores for directed graph.
        
        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores.
        Authority score = sum of hub scores of nodes pointing to this node.
        
        Good authorities are pointed to by good hubs. Requires directed graph.
        Uses power iteration until convergence.
        
        Args:
            max_iter: Maximum iterations (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dictionary mapping node ID to authority score
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('A', 'C')
            >>> auth = G.get_authority_centrality()
            >>> # B and C have high authority (pointed to by A)
        """
        return self._graph.get_authority_centrality(max_iter, tol)
    
    def get_hub_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get HITS hub scores for directed graph.
        
        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores.
        Hub score = sum of authority scores of nodes this node points to.
        
        Good hubs point to good authorities. Requires directed graph.
        Uses power iteration until convergence.
        
        Args:
            max_iter: Maximum iterations (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dictionary mapping node ID to hub score
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('A', 'C')
            >>> hub = G.get_hub_centrality()
            >>> # A has high hub score (points to many nodes)
        """
        return self._graph.get_hub_centrality(max_iter, tol)
    
    def get_alpha_centrality(self, alpha: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get alpha centrality for each node.
        
        Alpha centrality is a generalization of eigenvector centrality:
        x = alpha * A * x + e
        
        where alpha is attenuation factor, A is adjacency matrix, 
        e is exogenous factor (set to 1). Lower alpha emphasizes local
        neighborhood, higher alpha considers global structure.
        
        Args:
            alpha: Attenuation factor (default: 0.1, range: 0-1)
            max_iter: Maximum iterations (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dictionary mapping node ID to alpha centrality score
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)
            >>> alpha_cent = G.get_alpha_centrality(alpha=0.1)
            >>> # Hub has highest alpha centrality
        """
        return self._graph.get_alpha_centrality(alpha, max_iter, tol)
    
    def get_leverage_centrality(self) -> dict[str, float]:
        """Get leverage centrality for each node.
        
        Leverage centrality measures the degree connectivity of a node
        relative to its neighbors:
        
        leverage(v) = (1/k_v) * sum((k_v - k_n) / (k_v + k_n))
        
        where k_v is degree of v, k_n is degree of neighbor n.
        
        Positive values: node has more connections than its neighbors (influential)
        Negative values: node has fewer connections than its neighbors (peripheral)
        
        Returns:
            Dictionary mapping node ID to leverage score
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)
            >>> leverage = G.get_leverage_centrality()
            >>> # Hub has positive leverage (high degree)
            >>> # Leaves have negative leverage (low degree)
        """
        return self._graph.get_leverage_centrality()
    
    # ========== Selection Methods ==========
    
    def select_nodes_by_id(self, node_ids: list[str]) -> "Graph":
        """Select nodes by their IDs.
        
        Args:
            node_ids: List of node IDs to select
            
        Returns:
            Self for method chaining
            
        Raises:
            KeyError: If any node ID doesn't exist
        """
        self._graph.select_nodes_by_id(node_ids)
        return self
    
    def select_edges_by_id(self, edge_ids: list[int]) -> "Graph":
        """Select edges by their IDs.
        
        Args:
            edge_ids: List of edge IDs to select
            
        Returns:
            Self for method chaining
            
        Raises:
            KeyError: If any edge ID doesn't exist
        """
        self._graph.select_edges_by_id(edge_ids)
        return self
    
    def select_edges_by_node_id(self, node_ids: list[str]) -> "Graph":
        """Select all edges connected to the specified nodes.
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            Self for method chaining
        """
        self._graph.select_edges_by_node_id(node_ids)
        return self
    
    def select_nodes_by_degree(
        self,
        min_degree: int | None = None,
        max_degree: int | None = None,
        degree_type: str = "total"
    ) -> "Graph":
        """Select nodes by degree criteria.
        
        Args:
            min_degree: Minimum degree (inclusive)
            max_degree: Maximum degree (inclusive)
            degree_type: Type of degree - "in", "out", or "total" (default: "total")
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G.select_nodes_by_degree(min_degree=5, degree_type="out")
            >>> G.select_nodes_by_degree(max_degree=10)
        """
        self._graph.select_nodes_by_degree(
            min_degree=min_degree,
            max_degree=max_degree,
            degree_type=degree_type
        )
        return self
    
    def deselect_nodes(self, node_ids: list[str]) -> "Graph":
        """Remove nodes from selection.
        
        Args:
            node_ids: List of node IDs to deselect
            
        Returns:
            Self for method chaining
        """
        self._graph.deselect_nodes(node_ids)
        return self
    
    def deselect_edges(self, edge_ids: list[int]) -> "Graph":
        """Remove edges from selection.
        
        Args:
            edge_ids: List of edge IDs to deselect
            
        Returns:
            Self for method chaining
        """
        self._graph.deselect_edges(edge_ids)
        return self
    
    def clear_selection(self) -> "Graph":
        """Clear all node and edge selections.
        
        Returns:
            Self for method chaining
        """
        self._graph.clear_selection()
        return self
    
    def invert_node_selection(self) -> "Graph":
        """Invert node selection (select unselected, unselect selected).
        
        Returns:
            Self for method chaining
        """
        self._graph.invert_node_selection()
        return self
    
    def invert_edge_selection(self) -> "Graph":
        """Invert edge selection (select unselected, unselect selected).
        
        Returns:
            Self for method chaining
        """
        self._graph.invert_edge_selection()
        return self
    
    def get_selected_nodes(self) -> list[str]:
        """Get currently selected node IDs.
        
        Returns:
            List of selected node IDs
        """
        return self._graph.get_selected_nodes()
    
    def get_selected_edges(self) -> list[int]:
        """Get currently selected edge IDs.
        
        Returns:
            List of selected edge IDs
        """
        return self._graph.get_selected_edges()
    
    def count_selected_nodes(self) -> int:
        """Get count of currently selected nodes.
        
        Returns:
            Number of selected nodes
        """
        return self._graph.count_selected_nodes()
    
    def count_selected_edges(self) -> int:
        """Get count of currently selected edges.
        
        Returns:
            Number of selected edges
        """
        return self._graph.count_selected_edges()
    
    def select_nodes(self, predicate) -> "Graph":
        """Select nodes using a predicate function.
        
        The predicate function receives a Node object and should return True
        for nodes to be selected.
        
        Args:
            predicate: Callable that takes a Node and returns bool
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Select nodes with specific attribute
            >>> G.select_nodes(lambda n: n.attributes and n.attributes.get('type') == 'hub')
            >>> 
            >>> # Select nodes by ID pattern
            >>> G.select_nodes(lambda n: n.id.startswith('user_'))
        """
        self._graph.select_nodes(predicate)
        return self
    
    def select_edges(self, predicate) -> "Graph":
        """Select edges using a predicate function.
        
        The predicate function receives an Edge object and should return True
        for edges to be selected.
        
        Args:
            predicate: Callable that takes an Edge and returns bool
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Select edges with weight > 0.5
            >>> G.select_edges(lambda e: e.weight and e.weight > 0.5)
            >>> 
            >>> # Select edges with specific relationship type
            >>> G.select_edges(lambda e: e.attributes and e.attributes.get('rel') == 'friend')
        """
        self._graph.select_edges(predicate)
        return self
    
    def get_node(self, node_id: str):
        """Get a node object by ID.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Node object or None if not found
        """
        return self._graph.get_node(node_id)
    
    def get_edge(self, edge_id: int):
        """Get an edge object by ID.
        
        Args:
            edge_id: Edge ID to retrieve
            
        Returns:
            Edge object or None if not found
        """
        return self._graph.get_edge(edge_id)
    
    def select_last_nodes_created(self, n: int) -> "Graph":
        """Select the last N nodes that were created.
        
        Useful for incremental graph building workflows where you want to
        work with recently added nodes.
        
        Args:
            n: Number of most recently created nodes to select
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_node("A")
            >>> G.add_node("B")
            >>> G.add_node("C")
            >>> G.select_last_nodes_created(2)  # Selects B and C
            >>> G.get_selected_nodes()
            ['B', 'C']
        """
        self._graph.select_last_nodes_created(n)
        return self
    
    def select_last_edges_created(self, n: int) -> "Graph":
        """Select the last N edges that were created.
        
        Useful for incremental graph building workflows where you want to
        work with recently added edges.
        
        Args:
            n: Number of most recently created edges to select
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.add_edge("C", "D")
            >>> G.select_last_edges_created(2)  # Selects last 2 edges
            >>> len(G.get_selected_edges())
            2
        """
        self._graph.select_last_edges_created(n)
        return self
    
    def select_nodes_in_neighborhood(
        self,
        node_ids: list[str] | str,
        k: int = 1,
        mode: str = "out"
    ) -> "Graph":
        """Select all nodes within K hops of the specified seed nodes.
        
        Uses breadth-first search to find neighbors at distance up to k.
        Does not include the seed nodes themselves in the selection.
        
        Args:
            node_ids: Seed node ID(s) to start from
            k: Maximum number of hops/distance (default 1)
            mode: Direction to traverse - "out", "in", or "all" (default "out")
                  For undirected graphs, always uses "all" mode
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_path(10)
            >>> G.select_nodes_in_neighborhood("1", k=2, mode="out")
            >>> G.get_selected_nodes()  # Gets nodes 1, 2, 3
        """
        # Convert single string to list
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        
        self._graph.select_nodes_in_neighborhood(node_ids, k, mode)
        return self
    
    def select_nodes_by_centrality(
        self,
        measure: str = "degree",
        k: int = 10,
        include_ties: bool = False,
        **kwargs
    ) -> "Graph":
        """Select top-K nodes by centrality measure.
        
        Computes centrality scores and selects the highest-scoring nodes.
        
        Args:
            measure: Centrality measure - "degree", "in_degree", "out_degree",
                    "pagerank", or "betweenness" (default "degree")
            k: Number of top nodes to select (default 10)
            include_ties: If True, include all nodes tied with k-th score (default False)
            **kwargs: Additional arguments for centrality algorithms (e.g., damping for pagerank)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_star(20, "hub")
            >>> G.select_nodes_by_centrality("degree", k=5)
            >>> "hub" in G.get_selected_nodes()  # Hub has highest degree
            True
        """
        # Compute scores based on measure
        if measure == "degree":
            scores = {node_id: self.degree(node_id) for node_id in self.node_ids()}
        elif measure == "in_degree":
            if not self._graph.is_directed:
                raise ValueError("in_degree only available for directed graphs")
            scores = {node_id: self.in_degree(node_id) for node_id in self.node_ids()}
        elif measure == "out_degree":
            if not self._graph.is_directed:
                raise ValueError("out_degree only available for directed graphs")
            scores = {node_id: self.out_degree(node_id) for node_id in self.node_ids()}
        elif measure == "pagerank":
            scores = pagerank(self, **kwargs)
        elif measure == "betweenness":
            scores = betweenness_centrality(self)
        else:
            raise ValueError(
                f"Unknown centrality measure '{measure}'. "
                "Choose from: degree, in_degree, out_degree, pagerank, betweenness"
            )
        
        # Use the Rust helper to select top-k
        self._graph.select_nodes_by_scores(scores, k, include_ties)
        return self
    
    # ========== Query Operations ==========
    
    def get_node_attrs(self, attr: str, nodes: list[str] | None = None) -> dict[str, any]:
        """Get attribute values for nodes.
        
        Args:
            attr: Attribute name to extract
            nodes: Optional list of node IDs (default: all nodes)
            
        Returns:
            Dictionary mapping node IDs to attribute values
            
        Example:
            >>> G = Graph()
            >>> G.add_node("A", type="user", score=95)
            >>> G.add_node("B", type="admin", score=88)
            >>> G.get_node_attrs("score")
            {'A': 95, 'B': 88}
        """
        import polars as pl
        
        if nodes is None:
            nodes = self.node_ids()
        
        result = {}
        for node_id in nodes:
            node = self.get_node(node_id)
            if node and node.attributes:
                if attr in node.attributes:
                    result[node_id] = node.attributes[attr]
        
        return result
    
    def get_edge_attrs(self, attr: str, edges: list[int] | None = None) -> dict[int, any]:
        """Get attribute values for edges.
        
        Args:
            attr: Attribute name to extract
            edges: Optional list of edge IDs (default: all edges)
            
        Returns:
            Dictionary mapping edge IDs to attribute values
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B", weight=1.5, type="friend")
            >>> G.get_edge_attrs("weight")
            {1: 1.5}
        """
        if edges is None:
            edges = self.edge_ids()
        
        result = {}
        for edge_id in edges:
            edge = self.get_edge(edge_id)
            if edge:
                # Check special weight attribute first
                if attr == "weight" and edge.weight is not None:
                    result[edge_id] = edge.weight
                # Then check attributes dict
                elif edge.attributes and attr in edge.attributes:
                    result[edge_id] = edge.attributes[attr]
        
        return result
    
    def get_node_df(self, include_degree: bool = True) -> "pl.DataFrame":
        """Get nodes as a Polars DataFrame.
        
        Args:
            include_degree: If True, include degree column (default True)
            
        Returns:
            DataFrame with node IDs and attributes
            
        Example:
            >>> G = Graph()
            >>> G.add_node("A", type="user", score=95)
            >>> G.add_node("B", type="admin", score=88)
            >>> df = G.get_node_df()
            >>> df.columns
            ['id', 'type', 'score', 'degree']
        """
        import polars as pl
        
        node_ids = self.node_ids()
        if not node_ids:
            return pl.DataFrame()
        
        # Collect all attributes
        data = {'id': node_ids}
        
        # Get all unique attribute keys
        attr_keys = set()
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node and node.attributes:
                attr_keys.update(node.attributes.keys())
        
        # Extract attributes
        for key in sorted(attr_keys):
            data[key] = [
                self.get_node(nid).attributes.get(key) if self.get_node(nid) and self.get_node(nid).attributes else None
                for nid in node_ids
            ]
        
        # Add degree if requested
        if include_degree:
            data['degree'] = [self.degree(nid) for nid in node_ids]
        
        return pl.DataFrame(data)
    
    def get_edge_df(self, include_endpoints: bool = True) -> "pl.DataFrame":
        """Get edges as a Polars DataFrame.
        
        Args:
            include_endpoints: If True, include source and target columns (default True)
            
        Returns:
            DataFrame with edge IDs, endpoints, and attributes
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B", weight=1.5)
            >>> df = G.get_edge_df()
            >>> df.columns
            ['id', 'source', 'target', 'weight']
        """
        import polars as pl
        
        edge_ids = self.edge_ids()
        if not edge_ids:
            return pl.DataFrame()
        
        # Collect basic edge info
        data = {'id': edge_ids}
        
        if include_endpoints:
            edges = [self.get_edge_by_id(eid) for eid in edge_ids]
            data['source'] = [e[0] if e else None for e in edges]
            data['target'] = [e[1] if e else None for e in edges]
        
        # Get all unique attribute keys (including weight)
        attr_keys = set()
        edges_list = [self.get_edge(eid) for eid in edge_ids]
        
        # Check for weight attribute
        if any(e and e.weight is not None for e in edges_list):
            attr_keys.add("weight")
        
        # Add other attributes from dict
        for edge in edges_list:
            if edge and edge.attributes:
                attr_keys.update(edge.attributes.keys())
        
        # Extract attributes
        for key in sorted(attr_keys):
            if key == "weight":
                data[key] = [e.weight if e else None for e in edges_list]
            else:
                data[key] = [
                    e.attributes.get(key) if e and e.attributes else None
                    for e in edges_list
                ]
        
        return pl.DataFrame(data)
    
    def get_node_df_ws(self, include_degree: bool = True) -> "pl.DataFrame":
        """Get selected nodes as a Polars DataFrame.
        
        Only includes nodes in the current selection.
        
        Args:
            include_degree: If True, include degree column (default True)
            
        Returns:
            DataFrame with selected node IDs and attributes
            
        Example:
            >>> G = Graph()
            >>> G.add_path(5)
            >>> G.select_nodes_by_degree(min_degree=2)
            >>> df = G.get_node_df_ws()
            >>> len(df)  # Only selected nodes
        """
        import polars as pl
        
        node_ids = self.get_selected_nodes()
        if not node_ids:
            return pl.DataFrame()
        
        # Use same logic as get_node_df but with selected nodes
        data = {'id': node_ids}
        
        attr_keys = set()
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node and node.attributes:
                attr_keys.update(node.attributes.keys())
        
        for key in sorted(attr_keys):
            data[key] = [
                self.get_node(nid).attributes.get(key) if self.get_node(nid) and self.get_node(nid).attributes else None
                for nid in node_ids
            ]
        
        if include_degree:
            data['degree'] = [self.degree(nid) for nid in node_ids]
        
        return pl.DataFrame(data)
    
    def get_edge_df_ws(self, include_endpoints: bool = True) -> "pl.DataFrame":
        """Get selected edges as a Polars DataFrame.
        
        Only includes edges in the current selection.
        
        Args:
            include_endpoints: If True, include source and target columns (default True)
            
        Returns:
            DataFrame with selected edge IDs, endpoints, and attributes
            
        Example:
            >>> G = Graph()
            >>> G.add_path(5)
            >>> G.select_edges_by_id([1, 2])
            >>> df = G.get_edge_df_ws()
            >>> len(df)
            2
        """
        import polars as pl
        
        edge_ids = self.get_selected_edges()
        if not edge_ids:
            return pl.DataFrame()
        
        data = {'id': edge_ids}
        
        if include_endpoints:
            edges = [self.get_edge_by_id(eid) for eid in edge_ids]
            data['source'] = [e[0] if e else None for e in edges]
            data['target'] = [e[1] if e else None for e in edges]
        
        # Get all unique attribute keys (including weight)
        attr_keys = set()
        edges_list = [self.get_edge(eid) for eid in edge_ids]
        
        # Check for weight attribute
        if any(e and e.weight is not None for e in edges_list):
            attr_keys.add("weight")
        
        # Add other attributes from dict
        for edge in edges_list:
            if edge and edge.attributes:
                attr_keys.update(edge.attributes.keys())
        
        # Extract attributes
        for key in sorted(attr_keys):
            if key == "weight":
                data[key] = [e.weight if e else None for e in edges_list]
            else:
                data[key] = [
                    e.attributes.get(key) if e and e.attributes else None
                    for e in edges_list
                ]
        
        return pl.DataFrame(data)
    
    def get_neighbors(
        self,
        node_id: str,
        mode: str = "all",
        predicate: Optional[Callable] = None
    ) -> list[str]:
        """Get neighbors of a node with optional filtering.
        
        Args:
            node_id: Node ID
            mode: Direction - "out" (successors), "in" (predecessors), or "all" (default)
            predicate: Optional filter function that takes a node ID and returns bool
            
        Returns:
            List of neighbor node IDs
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B", weight=1.0)
            >>> G.add_edge("C", "A", weight=2.0)
            >>> G.get_neighbors("A", mode="out")
            ['B']
            >>> G.get_neighbors("A", mode="in")
            ['C']
        """
        if mode not in ["out", "in", "all"]:
            raise ValueError("mode must be 'out', 'in', or 'all'")
        
        neighbors = []
        
        if mode in ["out", "all"]:
            # Outgoing neighbors from adjacency list
            neighbors.extend(self._graph.neighbors(node_id))
        
        if mode in ["in", "all"]:
            # Incoming neighbors - find edges where this node is target
            for edge_tuple in self._graph.edges():
                # edge_tuple is (edge_id, source, target)
                if edge_tuple[2] == node_id and edge_tuple[1] not in neighbors:
                    neighbors.append(edge_tuple[1])
        
        # For undirected graphs, mode doesn't matter
        if not self._graph.is_directed and mode != "all":
            # Already got all neighbors via adjacency list
            pass
        
        # Apply predicate filter if provided
        if predicate:
            neighbors = [n for n in neighbors if predicate(n)]
        
        return neighbors
    
    def get_common_neighbors(self, node_id1: str, node_id2: str) -> list[str]:
        """Get nodes that are neighbors of both specified nodes.
        
        Args:
            node_id1: First node ID
            node_id2: Second node ID
            
        Returns:
            List of common neighbor node IDs
            
        Example:
            >>> G = Graph(directed=False)
            >>> G.add_edge("A", "C")
            >>> G.add_edge("B", "C")
            >>> G.add_edge("A", "D")
            >>> G.add_edge("B", "D")
            >>> G.get_common_neighbors("A", "B")
            ['C', 'D']
        """
        neighbors1 = set(self.get_neighbors(node_id1, mode="all"))
        neighbors2 = set(self.get_neighbors(node_id2, mode="all"))
        return list(neighbors1 & neighbors2)
    
    def get_predecessors(self, node_id: str) -> list[str]:
        """Get direct predecessors of a node (incoming neighbors).
        
        Only for directed graphs.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of predecessor node IDs
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "C")
            >>> G.add_edge("B", "C")
            >>> G.get_predecessors("C")
            ['A', 'B']
        """
        if not self._graph.is_directed:
            raise ValueError("get_predecessors only available for directed graphs")
        return self.get_neighbors(node_id, mode="in")
    
    def get_successors(self, node_id: str) -> list[str]:
        """Get direct successors of a node (outgoing neighbors).
        
        Only for directed graphs.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of successor node IDs
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("A", "C")
            >>> G.get_successors("A")
            ['B', 'C']
        """
        if not self._graph.is_directed:
            raise ValueError("get_successors only available for directed graphs")
        return self.get_neighbors(node_id, mode="out")
    
    def get_node_creation_order(self) -> list[str]:
        """Get the creation order of all nodes.
        
        Returns:
            List of node IDs in the order they were created
        """
        return self._graph.get_node_creation_order()
    
    def get_edge_creation_order(self) -> list[int]:
        """Get the creation order of all edges.
        
        Returns:
            List of edge IDs in the order they were created
        """
        return self._graph.get_edge_creation_order()
    
    # ===== Bulk Modification Operations (_ws = with selection) =====
    
    def delete_nodes_ws(self) -> "Graph":
        """Delete all selected nodes and their incident edges.
        
        Removes nodes in the current selection from the graph, along with
        all edges connected to those nodes. Updates selection state.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> from edgecase.builders import add_path
            >>> add_path(G, 5)
            >>> G.select_nodes_by_id(['2', '3'])
            >>> G.delete_nodes_ws()
            >>> G.num_nodes()
            3
        """
        selected = self.get_selected_nodes()
        
        # Delete each selected node (Rust handles incident edges)
        for node_id in selected:
            try:
                self._graph.delete_node(node_id)
            except Exception:
                pass  # Node might already be deleted
        
        return self
    
    def delete_edges_ws(self) -> "Graph":
        """Delete all selected edges.
        
        Removes edges in the current selection from the graph.
        Updates selection state.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_path(5)
            >>> G.select_edges_by_id([1, 2])
            >>> G.delete_edges_ws()
            >>> G.num_edges()
            2
        """
        selected = self.get_selected_edges()
        
        for edge_id in selected:
            self.delete_edge_by_id(edge_id)
        
        # Edge selection is automatically cleared since edges no longer exist
        
        return self
    
    def set_node_attrs_ws(self, **attributes) -> "Graph":
        """Set attributes on all selected nodes.
        
        Updates or adds attributes to nodes in the current selection.
        Existing attributes are preserved unless explicitly overwritten.
        
        Args:
            **attributes: Attribute name-value pairs to set
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_node('A', value=10)
            >>> G.select_nodes_by_id(['A'])
            >>> G.set_node_attrs_ws(color='red', importance=0.9)
            >>> G.get_node('A').attributes['color']
            'red'
        """
        selected = self.get_selected_nodes()
        
        for node_id in selected:
            for key, value in attributes.items():
                # Use Rust set_node_attr for persistence
                self._graph.set_node_attr(node_id, key, value)
        
        return self
    
    def set_edge_attrs_ws(self, **attributes) -> "Graph":
        """Set attributes on all selected edges.
        
        Updates or adds attributes to edges in the current selection.
        Special handling for 'weight' attribute.
        
        Args:
            **attributes: Attribute name-value pairs to set
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> from edgecase.builders import add_path
            >>> add_path(G, 3)
            >>> G.select_edges_by_id([1])
            >>> G.set_edge_attrs_ws(weight=5.0, type='strong')
            >>> G.get_edge(1).weight
            5.0
        """
        selected = self.get_selected_edges()
        
        for edge_id in selected:
            for key, value in attributes.items():
                # Use Rust set_edge_attr for persistence (handles weight specially)
                self._graph.set_edge_attr(edge_id, key, value)
        
        return self
    
    def mutate_node_attrs_ws(self, attr: str, func: Callable) -> "Graph":
        """Transform a node attribute using a function on selected nodes.
        
        Applies a transformation function to the specified attribute
        for all selected nodes. The function receives the current
        attribute value and returns the new value.
        
        Args:
            attr: Attribute name to transform
            func: Function that takes current value and returns new value
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_node('A', score=10)
            >>> G.add_node('B', score=20)
            >>> G.select_nodes_by_id(['A', 'B'])
            >>> G.mutate_node_attrs_ws('score', lambda x: x * 2)
            >>> G.get_node('A').attributes['score']
            20
        """
        selected = self.get_selected_nodes()
        
        for node_id in selected:
            node = self.get_node(node_id)
            if node and node.attributes and attr in node.attributes:
                old_value = node.attributes[attr]
                new_value = func(old_value)
                # Use Rust method to persist the change
                self._graph.set_node_attr(node_id, attr, new_value)
        
        return self
    
    def mutate_edge_attrs_ws(self, attr: str, func: Callable) -> "Graph":
        """Transform an edge attribute using a function on selected edges.
        
        Applies a transformation function to the specified attribute
        for all selected edges. Handles 'weight' field specially.
        
        Args:
            attr: Attribute name to transform (can be 'weight')
            func: Function that takes current value and returns new value
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_edge('A', 'B', weight=1.5)
            >>> G.select_edges_by_id([1])
            >>> G.mutate_edge_attrs_ws('weight', lambda x: x * 2)
            >>> G.get_edge(1).weight
            3.0
        """
        selected = self.get_selected_edges()
        
        for edge_id in selected:
            edge = self.get_edge(edge_id)
            if edge:
                if attr == 'weight' and edge.weight is not None:
                    new_value = func(edge.weight)
                    self._graph.set_edge_attr(edge_id, 'weight', new_value)
                elif edge.attributes and attr in edge.attributes:
                    old_value = edge.attributes[attr]
                    new_value = func(old_value)
                    self._graph.set_edge_attr(edge_id, attr, new_value)
        
        return self
    
    def fully_connect_nodes_ws(self, **edge_attributes) -> "Graph":
        """Create edges between all pairs of selected nodes (clique).
        
        Adds edges to form a complete graph among selected nodes.
        For directed graphs, creates edges in both directions.
        
        Args:
            **edge_attributes: Optional attributes for new edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_node('A')
            >>> G.add_node('B')
            >>> G.add_node('C')
            >>> G.select_nodes_by_id(['A', 'B', 'C'])
            >>> G.fully_connect_nodes_ws(type='clique')
            >>> G.num_edges()
            3  # or 6 if directed
        """
        selected = list(self.get_selected_nodes())
        
        # Create edges between all pairs
        for i, source in enumerate(selected):
            for target in selected[i+1:]:
                # Check if edge already exists
                has_edge = any(
                    (s == source and t == target) or
                    (not self.directed and s == target and t == source)
                    for _, s, t in self.edges()
                )
                
                if not has_edge:
                    self.add_edge(source, target, **edge_attributes)
                
                # For directed graphs, add reverse edge
                if self.directed:
                    has_reverse = any(
                        s == target and t == source
                        for _, s, t in self.edges()
                    )
                    if not has_reverse:
                        self.add_edge(target, source, **edge_attributes)
        
        return self
    
    def add_reverse_edges_ws(self, **edge_attributes) -> "Graph":
        """Add reverse edges for all selected edges (directed graphs only).
        
        For each selected edge (A → B), adds the reverse edge (B → A)
        if it doesn't already exist.
        
        Args:
            **edge_attributes: Optional attributes for new reverse edges
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.select_edges_by_id([1])
            >>> G.add_reverse_edges_ws()
            >>> G.num_edges()
            2
        """
        if not self.directed:
            raise ValueError("add_reverse_edges_ws() only works on directed graphs")
        
        selected = self.get_selected_edges()
        edges_to_add = []
        
        for edge_id in selected:
            edge_tuple = self.get_edge_by_id(edge_id)
            if edge_tuple:
                source, target = edge_tuple
                # Check if reverse edge exists
                has_reverse = any(
                    s == target and t == source
                    for _, s, t in self.edges()
                )
                if not has_reverse:
                    edges_to_add.append((target, source))
        
        # Add reverse edges
        for source, target in edges_to_add:
            self.add_edge(source, target, **edge_attributes)
        
        return self
    
    def rev_edge_dir_ws(self) -> "Graph":
        """Reverse the direction of all selected edges (directed graphs only).
        
        For each selected edge (A → B), changes it to (B → A).
        Preserves edge ID and attributes.
        
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge('A', 'B', weight=1.5)
            >>> G.select_edges_by_id([1])
            >>> G.rev_edge_dir_ws()
            >>> G.get_edge_by_id(1)
            ('B', 'A')
        """
        if not self.directed:
            raise ValueError("rev_edge_dir_ws() only works on directed graphs")
        
        selected = self.get_selected_edges()
        
        for edge_id in selected:
            # Use Rust method to reverse edge direction
            self._graph.reverse_edge_direction(edge_id)
        
        return self
    
    # ====================================================================
    # Graph Generators
    # ====================================================================
    
    def add_path_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a path graph (linear chain) to the current graph.
        
        Creates a sequence of n nodes connected linearly: 1 → 2 → 3 → ... → n
        
        Args:
            n: Number of nodes in the path
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)
            >>> G.num_nodes()
            5
            >>> G.num_edges()
            4
        """
        from edgecase.builders import add_path
        add_path(self, n, type=type, label=label, rel=rel)
        return self
    
    def add_cycle_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a cycle graph (closed loop) to the current graph.
        
        Creates n nodes in a circular pattern where the last connects back to first.
        
        Args:
            n: Number of nodes in the cycle (minimum 3)
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(4)
            >>> G.num_nodes()
            4
            >>> G.num_edges()
            4
        """
        from edgecase.builders import add_cycle
        add_cycle(self, n, type=type, label=label, rel=rel)
        return self
    
    def add_star_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a star graph (hub-and-spoke) to the current graph.
        
        Creates a central hub node connected to n-1 peripheral nodes.
        
        Args:
            n: Total number of nodes (minimum 2)
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # 1 hub + 4 spokes
            >>> G.num_nodes()
            5
            >>> G.num_edges()
            4
        """
        from edgecase.builders import add_star
        add_star(self, n, type=type, label=label, rel=rel)
        return self
    
    def add_full_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
        keep_loops: bool = False,
    ) -> "Graph":
        """Add a complete graph (clique) to the current graph.
        
        Creates n nodes where every node is connected to every other node.
        
        Args:
            n: Number of nodes
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            keep_loops: If True, include self-loops
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_full_graph(4)
            >>> G.num_nodes()
            4
            >>> G.num_edges()
            6
        """
        from edgecase.builders import add_full_graph
        add_full_graph(self, n, type=type, label=label, rel=rel, keep_loops=keep_loops)
        return self
    
    def add_balanced_tree(
        self,
        k: int,
        h: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a balanced k-ary tree to the current graph.
        
        Creates a tree where each internal node has exactly k children, with height h.
        
        Args:
            k: Number of children per node (branching factor)
            h: Height of the tree (root has height 0)
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_balanced_tree(k=2, h=3)  # Binary tree of height 3
            >>> G.num_nodes()
            15
        """
        from edgecase.builders import add_balanced_tree
        add_balanced_tree(self, k, h, type=type, label=label, rel=rel)
        return self
    
    def add_grid_2d(
        self,
        rows: int,
        cols: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a 2D grid/lattice graph to the current graph.
        
        Creates a rectangular grid where each node is connected to its
        neighbors (up, down, left, right).
        
        Args:
            rows: Number of rows
            cols: Number of columns
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_grid_2d(3, 4)  # 3x4 grid
            >>> G.num_nodes()
            12
        """
        from edgecase.builders import add_grid_2d
        add_grid_2d(self, rows, cols, type=type, label=label, rel=rel)
        return self
    
    def add_prism_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a prism graph to the current graph.
        
        Creates two parallel n-cycles connected by n vertical edges.
        
        Args:
            n: Number of nodes in each cycle (minimum 3)
            type: Optional node type/category for all nodes
            label: If True, label nodes with their IDs
            rel: Optional relationship label for all edges
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_prism_graph(5)
            >>> G.num_nodes()
            10
        """
        from edgecase.builders import add_prism
        add_prism(self, n, type=type, label=label, rel=rel)
        return self
    
    def add_gnp_graph(
        self,
        n: int,
        p: float,
        directed: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "Graph":
        """Add an Erdős-Rényi G(n,p) random graph to the current graph.
        
        Each possible edge is included independently with probability p.
        Uses fast Rust implementation for optimal performance.
        
        Args:
            n: Number of nodes to add
            p: Probability of edge creation (0.0 to 1.0)
            directed: If provided and differs from graph's directed setting,
                     creates temporary graph with specified directedness
            seed: Random seed for reproducibility
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_gnp_graph(100, 0.05, seed=42)
            >>> G.num_nodes()
            100
        """
        if directed is not None and directed != self.directed:
            # Create temporary graph with desired directedness
            temp = Graph(directed=directed)
            temp._graph.generate_gnp_graph(n, p, seed)
            # Copy nodes and edges to this graph
            for node_id in temp.nodes():
                self.add_node(node_id)
            for source, target in temp.edges():
                self.add_edge(source, target)
        else:
            self._graph.generate_gnp_graph(n, p, seed)

        return self
    
    def add_gnm_graph(
        self,
        n: int,
        m: int,
        directed: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "Graph":
        """Add an Erdős-Rényi G(n,m) random graph to the current graph.
        
        Graph with n nodes and exactly m randomly placed edges.
        Uses fast Rust implementation for optimal performance.
        
        Args:
            n: Number of nodes to add
            m: Number of edges to create
            directed: If provided and differs from graph's directed setting,
                     creates temporary graph with specified directedness
            seed: Random seed for reproducibility
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_gnm_graph(50, 100, seed=42)
            >>> G.num_edges()
            100
        """
        if directed is not None and directed != self.directed:
            # Create temporary graph with desired directedness
            temp = Graph(directed=directed)
            temp._graph.generate_gnm_graph(n, m, seed)
            # Copy nodes and edges to this graph
            for node_id in temp.nodes():
                self.add_node(node_id)
            for source, target in temp.edges():
                self.add_edge(source, target)
        else:
            self._graph.generate_gnm_graph(n, m, seed)
        
        return self
    
    def add_pa_graph(
        self,
        n: int,
        m: int = 2,
        seed: Optional[int] = None,
    ) -> "Graph":
        """Add a Barabási-Albert preferential attachment graph.
        
        Grows a graph by adding nodes one at a time, with each new node
        connecting to m existing nodes with probability proportional to degree.
        Uses fast Rust implementation for optimal performance.
        
        Args:
            n: Number of nodes to add
            m: Number of edges to attach from new node (default 2)
            seed: Random seed for reproducibility
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_pa_graph(100, m=3, seed=42)
            >>> G.num_nodes()
            100
        """
        self._graph.generate_pa_graph(n, m, seed)
        return self
    
    def add_smallworld_graph(
        self,
        n: int,
        k: int = 6,
        p: float = 0.1,
        seed: Optional[int] = None,
    ) -> "Graph":
        """Add a Watts-Strogatz small-world graph.
        
        Creates a ring lattice with rewiring probability p, producing
        graphs with high clustering and short path lengths.
        Uses fast Rust implementation for optimal performance.
        
        Args:
            n: Number of nodes to add
            k: Each node connected to k nearest neighbors (must be even)
            p: Probability of rewiring each edge (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_smallworld_graph(100, k=6, p=0.1, seed=42)
            >>> G.num_nodes()
            100
        """
        self._graph.generate_smallworld_graph(n, k, p, seed)
        return self
    
    def __repr__(self) -> str:
        return self._graph.__repr__()


# Re-export Node and Edge from Rust for direct access if needed
Node = rust.Node
Edge = rust.Edge


# Algorithm functions - wrap Rust functions to accept our Graph wrapper
def shortest_path(graph: Graph, source: str, target: str) -> list[str] | None:
    """Find shortest path between two nodes."""
    return rust.shortest_path(graph._graph, source, target)


def pagerank(
    graph: Graph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> dict[str, float]:
    """Calculate PageRank for all nodes."""
    return rust.pagerank(graph._graph, damping=damping, max_iter=max_iter, tol=tol)


def betweenness_centrality(graph: Graph) -> dict[str, float]:
    """Calculate betweenness centrality for all nodes."""
    return rust.betweenness_centrality(graph._graph)


def detect_communities(graph: Graph) -> dict[str, int]:
    """Detect communities in the graph."""
    return rust.detect_communities(graph._graph)


def find_cycles(graph: Graph) -> list[list[str]]:
    """Find all cycles in the graph."""
    return rust.find_cycles(graph._graph)
