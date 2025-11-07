"""
Python wrapper for Rust core to provide a consistent API.

This module wraps the raw Rust bindings to provide the same API
as the pure Python implementation.
"""

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
