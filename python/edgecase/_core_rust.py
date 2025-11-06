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
    
    def add_node(self, node_id: str, **attributes):
        """Add a node to the graph (convenience method)."""
        if not self._graph.has_node(node_id):
            node = rust.Node(id=node_id, attributes=attributes if attributes else None)
            self._graph.add_node(node)
    
    def add_edge(self, source: str, target: str, weight: float | None = None, **attributes):
        """Add an edge to the graph (convenience method)."""
        # Ensure nodes exist first (direct Rust hash lookup - O(1))
        if not self._graph.has_node(source):
            self.add_node(source)
        if not self._graph.has_node(target):
            self.add_node(target)
        
        edge = rust.Edge(
            source=source,
            target=target,
            weight=weight,
            attributes=attributes if attributes else None
        )
        self._graph.add_edge(edge)
    
    def nodes(self) -> list[str]:
        """Return list of node IDs."""
        return self._graph.node_ids()
    
    def edges(self) -> list[tuple[str, str]]:
        """Return list of edges as (source, target) tuples."""
        # TODO: Implement edges() method in Rust
        return []
    
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
    
    def __repr__(self) -> str:
        return self._graph.__repr__()


# Re-export Node and Edge from Rust for direct access if needed
Node = rust.Node
Edge = rust.Edge


# Algorithm functions - wrap Rust functions to accept our Graph wrapper
def shortest_path(graph: Graph, source: str, target: str) -> list[str]:
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
