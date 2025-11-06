"""
Simple Python implementation of core graph functionality.
This is a fallback when the Rust extension isn't available.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl


class Graph:
    """Simple Python-based graph implementation."""

    def __init__(self, directed: bool = False, weighted: bool = True):
        self.directed = directed
        self.weighted = weighted
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str], dict[str, Any]] = {}
        self._adjacency: dict[str, list[str]] = defaultdict(list)
    
    @property
    def is_directed(self) -> bool:
        """Check if graph is directed (alias for directed)."""
        return self.directed
    
    @property
    def is_weighted(self) -> bool:
        """Check if graph is weighted (alias for weighted)."""
        return self.weighted
        
    def add_node(self, node_id: str, **attributes):
        """Add a node to the graph."""
        if node_id not in self._nodes:
            self._nodes[node_id] = attributes
        else:
            self._nodes[node_id].update(attributes)

    def add_edge(self, source: str, target: str, weight: float | None = None, **attributes):
        """Add an edge to the graph."""
        # Ensure nodes exist
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
            
        edge_key = (source, target)
        edge_attrs = attributes.copy()
        if weight is not None:
            edge_attrs['weight'] = weight
            
        self._edges[edge_key] = edge_attrs
        self._adjacency[source].append(target)
        
        if not self.directed:
            reverse_key = (target, source)
            self._edges[reverse_key] = edge_attrs
            self._adjacency[target].append(source)
    
    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)
    
    def num_edges(self) -> int:
        """Return the number of edges."""
        return len(self._edges) if self.directed else len(self._edges) // 2

    def neighbors(self, node_id: str) -> list[str]:
        """Get neighbors of a node."""
        return self._adjacency.get(node_id, [])

    def nodes(self) -> list[str]:
        """Get all node IDs."""
        return list(self._nodes.keys())

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all edges with attributes."""
        if self.directed:
            return [(s, t, attrs) for (s, t), attrs in self._edges.items()]
        else:
            # Return only one direction for undirected graphs
            seen = set()
            result = []
            for (s, t), attrs in self._edges.items():
                if (s, t) not in seen and (t, s) not in seen:
                    result.append((s, t, attrs))
                    seen.add((s, t))
                    seen.add((t, s))
            return result


def shortest_path(graph: Graph, source: str, target: str) -> list[str] | None:
    """Find shortest path between two nodes using BFS."""
    if source not in graph._nodes or target not in graph._nodes:
        return None
    
    if source == target:
        return [source]
    
    visited = {source}
    queue = deque([(source, [source])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph.neighbors(node):
            if neighbor == target:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def pagerank(graph: Graph, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    """Calculate PageRank for all nodes."""
    nodes = graph.nodes()
    n = len(nodes)
    
    if n == 0:
        return {}
    
    # Initialize ranks
    ranks = {node: 1.0 / n for node in nodes}
    
    for _ in range(max_iter):
        new_ranks = {}
        diff = 0.0
        
        for node in nodes:
            rank_sum = 0.0
            
            # Sum ranks from incoming edges
            for other_node in nodes:
                if node in graph.neighbors(other_node):
                    out_degree = len(graph.neighbors(other_node))
                    if out_degree > 0:
                        rank_sum += ranks[other_node] / out_degree
            
            new_rank = (1 - damping) / n + damping * rank_sum
            new_ranks[node] = new_rank
            diff += abs(new_rank - ranks[node])
        
        ranks = new_ranks
        
        if diff < tol:
            break
    
    return ranks


def from_polars(df: pl.DataFrame, source_col: str, target_col: str,
                weight_col: str | None = None, directed: bool = False) -> Graph:
    """Create a graph from a Polars DataFrame."""
    graph = Graph(directed=directed)
    
    for row in df.iter_rows(named=True):
        source = str(row[source_col])
        target = str(row[target_col])
        
        weight = None
        if weight_col and weight_col in row:
            weight = row[weight_col]
        
        graph.add_edge(source, target, weight=weight)
    
    return graph
