"""
Python wrapper for Rust core to provide a consistent API.

This module wraps the raw Rust bindings to provide automatic label resolution
and backwards compatibility. After migration to u64 integer node IDs,
this wrapper provides automatic label resolution for convenience.
"""

from typing import Optional, Callable, Any, Union
from edgecase import _core as rust


class Graph:
    """Python wrapper around Rust Graph for consistent API.
    
    After the u64 migration, nodes are identified internally by integer IDs,
    but this wrapper allows using string labels for convenience. The mapping
    between labels and IDs is managed automatically.
    """
    
    def __init__(self, directed: bool = False, weighted: bool = False):
        self._graph = rust.Graph(directed=directed, weighted=weighted)
    
    def add_node(self, node_id: Optional[Union[int, str]] = None, label: Optional[str] = None, 
                 from_node: Optional[Union[int, str]] = None, to_node: Optional[Union[int, str]] = None,
                 **attributes) -> int:
        """Add a node to the graph (optionally connected to existing nodes).
        
        Add isolated nodes or nodes with immediate connections.
        
        Args:
            node_id: Optional node identifier. Can be:
                - None: Auto-generate a u64 ID (default)
                - int: Use this as the node ID
                - str: Use as label, auto-generate ID
            label: Optional string label for the node
            from_node: Optional source node (int ID or str label) to create edge FROM this new node TO
            to_node: Optional target node (int ID or str label) to create edge FROM TO this new node
            **attributes: Optional node attributes
            
        Returns:
            The integer ID of the added node (u64)
            
        Examples:
            >>> G = Graph()
            >>> # Add isolated node with auto-generated ID
            >>> id1 = G.add_node()  # Returns u64 like 1
            >>> # Add node with label
            >>> id2 = G.add_node(label="Alice")  # Returns u64, stores label
            >>> # Add node connected from existing node
            >>> id3 = G.add_node(label="Bob", from_node=id2)  # Creates edge id2->id3
            >>> # Add node connected to existing node
            >>> id4 = G.add_node(label="Charlie", to_node=id2)  # Creates edge id4->id2
            >>> # Use string labels for connections
            >>> id5 = G.add_node(label="Dave", from_node="Alice")  # Edge Alice->Dave
        """
        # Handle different input patterns for node_id
        if isinstance(node_id, rust.Node):
            # Direct Rust Node object - already has u64 ID
            new_id = self._graph.add_node(node_id)
        else:
            # Determine label and ID
            final_label = None
            final_id = None
            
            if isinstance(node_id, str):
                # String node_id is treated as a label
                final_label = node_id
            elif isinstance(node_id, int):
                # Integer node_id is used as the ID
                final_id = node_id
                final_label = label
            elif node_id is None:
                # Auto-generate ID
                final_label = label
            else:
                raise TypeError(f"node_id must be int, str, or None, got {type(node_id)}")
            
            # Create the node with u64 ID and optional label
            if final_id is not None:
                node = rust.Node(id=final_id, label=final_label, attributes=attributes if attributes else None)
            else:
                # Let Rust auto-assign the ID (pass 0 to trigger auto-assignment)
                node = rust.Node(id=0, label=final_label, attributes=attributes if attributes else None)
            
            new_id = self._graph.add_node(node)
        
        # Add edges if from_node or to_node specified
        if from_node is not None:
            # Create edge: new_node -> from_node
            self.add_edge(new_id, from_node)
        
        if to_node is not None:
            # Create edge: to_node -> new_node
            self.add_edge(to_node, new_id)
        
        return new_id
    
    def add_edge(self, source: Union[int, str, rust.Edge], target: Optional[Union[int, str]] = None, 
                 weight: Optional[float] = None, edge_id: Optional[int] = None, **attributes) -> int:
        """Add an edge to the graph between existing nodes.
        
        Requires nodes to exist. Can use integer IDs or
        string labels (if all labels are unique and non-empty).
        
        Args:
            source: Either an Edge object, or a source node ID/label
            target: Target node ID/label (required if source is not an Edge)
            weight: Optional edge weight
            edge_id: Optional edge ID (auto-assigned if None)
            **attributes: Optional edge attributes
            
        Returns:
            The ID of the added edge
            
        Raises:
            ValueError: If a label is not found or nodes don't exist
            
        Examples:
            >>> G = Graph()
            >>> id1 = G.add_node(label="Alice")
            >>> id2 = G.add_node(label="Bob")
            >>> # By ID
            >>> G.add_edge(id1, id2)
            >>> # By label (only if all labels are unique)
            >>> G.add_edge("Alice", "Bob")
            >>> # Mixed
            >>> G.add_edge(id1, "Bob")
        """
        # Handle direct Edge objects
        if isinstance(source, rust.Edge):
            return self._graph.add_edge(source)
        
        # Resolve source and target to u64 IDs (DO NOT auto-create)
        source_id = self._resolve_node_id(source, auto_create=False)
        target_id = self._resolve_node_id(target, auto_create=False)
        
        # Verify nodes exist
        if not self._graph.has_node_id(source_id):
            raise ValueError(f"Source node {source} (ID {source_id}) does not exist. Add it with add_node() first.")
        
        if not self._graph.has_node_id(target_id):
            raise ValueError(f"Target node {target} (ID {target_id}) does not exist. Add it with add_node() first.")
        
        # Create edge
        edge = rust.Edge(
            id=edge_id if edge_id is not None else 0,  # 0 means auto-assign
            source=source_id,
            target=target_id,
            weight=weight,
            attributes=attributes if attributes else None
        )
        return self._graph.add_edge(edge)
    
    def _resolve_node_id(self, node: Union[int, str], auto_create: bool = False) -> int:
        """Resolve a node identifier (int or str) to a u64 ID.
        
        Args:
            node: Either an integer ID or a string label
            auto_create: If True and node is a string not in graph, create it
            
        Returns:
            The u64 node ID
            
        Raises:
            ValueError: If label is not found and auto_create is False
            TypeError: If node is neither int nor str
        """
        if isinstance(node, int):
            return node
        elif isinstance(node, str):
            # Try to resolve label to ID
            try:
                node_id = self._graph.id_from_label(node)
                return node_id
            except (ValueError, Exception):
                # Label not found
                if auto_create:
                    # Create a new node with this label
                    return self.add_node(label=node)
                else:
                    raise ValueError(f"Node with label '{node}' not found in graph")
        else:
            raise TypeError(f"Node identifier must be int or str, got {type(node)}")
    
    def id_from_label(self, label: str) -> Optional[int]:
        """Get the integer ID for a node by its label.
        
        Args:
            label: The string label to look up
            
        Returns:
            The u64 node ID, or None if not found
            
        Example:
            >>> G = Graph()
            >>> G.add_node(label="Alice")
            1
            >>> G.id_from_label("Alice")
            1
            >>> G.id_from_label("Unknown")
            None
        """
        try:
            return self._graph.id_from_label(label)
        except (ValueError, Exception):
            return None
    
    def label_from_id(self, node_id: int) -> Optional[str]:
        """Get the label for a node by its ID.
        
        Args:
            node_id: The u64 node ID
            
        Returns:
            The string label, or None if node has no label or doesn't exist
            
        Example:
            >>> G = Graph()
            >>> id1 = G.add_node(label="Alice")
            >>> G.label_from_id(id1)
            'Alice'
            >>> G.label_from_id(999)
            None
        """
        try:
            return self._graph.label_from_id(node_id)
        except (ValueError, Exception):
            return None
    
    def get_node_ids(self) -> list[int]:
        """Get all node IDs in the graph.
        
        Returns:
            List of u64 node IDs
            
        Example:
            >>> G = Graph()
            >>> G.add_node(label="A")
            1
            >>> G.add_node(label="B")
            2
            >>> G.get_node_ids()
            [1, 2]
        """
        return self._graph.get_node_ids()
    
    def nodes(self) -> list[int]:
        """Return list of node IDs (alias for get_node_ids)."""
        return self._graph.get_node_ids()
    
    def node_ids(self) -> list[int]:
        """Return list of node IDs (alias for get_node_ids)."""
        return self._graph.get_node_ids()
    
    def is_graph_simple(self) -> bool:
        """Check if the graph is simple (no self-loops or multi-edges).
        
        A simple graph is one that does not contain any loops nor any multiple edges.
        
        Returns:
            True if the graph is simple, False otherwise
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B")
            >>> G.is_graph_simple()
            True
            >>> G.add_edge("A", "B")  # Add multi-edge
            >>> G.is_graph_simple()
            False
        """
        return self._graph.is_graph_simple()
    
    def has_cycles(self) -> bool:
        """Check if the graph contains any cycles.
        
        Uses depth-first search to detect cycles in the graph.
        
        Returns:
            True if the graph contains at least one cycle, False otherwise
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.has_cycles()
            False
            >>> G.add_edge("C", "A")  # Creates a cycle
            >>> G.has_cycles()
            True
        """
        return self._graph.has_cycles()
    
    def has_node(self, node: Union[int, str]) -> bool:
        """Check if a node exists in the graph.
        
        Args:
            node: Node ID (int) or label (str)
            
        Returns:
            True if the node exists
        """
        try:
            node_id = self._resolve_node_id(node, auto_create=False)
            return self._graph.has_node_id(node_id)
        except (ValueError, TypeError):
            return False
    
    def edges(self) -> list[tuple[int, str, str]]:
        """Return list of edges as (id, source, target) tuples."""
        return self._graph.edges()
    
    def get_edges(self) -> list[str]:
        """Get edges as formatted strings.
        
        Returns edge definitions:
        - Directed graphs: "A->B"
        - Undirected graphs: "A--B"
        
        Returns:
            List of edge strings
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.get_edges()
            ['A->B', 'B->C']
        """
        return self._graph.get_edges()
    
    def edge_ids(self) -> list[int]:
        """Return list of edge IDs."""
        return self._graph.edge_ids()
    
    def get_edge_by_id(self, edge_id: int) -> tuple[str, str] | None:
        """Get edge (source, target) by ID."""
        return self._graph.get_edge_by_id(edge_id)
    
    def has_edge_id(self, edge_id: int) -> bool:
        """Check if an edge with given ID exists."""
        return self._graph.has_edge_id(edge_id)
    
    def has_edge(self, source: Union[int, str], target: Union[int, str]) -> bool:
        """Check if an edge exists between two nodes.
        
        Args:
            source: Source node ID or label
            target: Target node ID or label (for undirected graphs, order doesn't matter)
            
        Returns:
            True if at least one edge exists between the nodes
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.has_edge("A", "B")
            True
            >>> G.has_edge("B", "A")
            False
        """
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)
        edge_ids = self._graph.find_edge_ids(source_id, target_id)
        return len(edge_ids) > 0
    
    def delete_edge_by_id(self, edge_id: int):
        """Delete edge by ID."""
        return self._graph.delete_edge_by_id(edge_id)
    
    def delete_edge(
        self, 
        source: Union[int, str, None] = None, 
        target: Union[int, str, None] = None, 
        id: int | None = None
    ) -> "Graph":
        """Delete an edge from the graph.
                
        Specify either (source, target) pair OR edge id. For multigraphs with source/target,
        only the first matching edge is deleted.
        
        Args:
            source: Source node ID or label
            target: Target node ID or label
            id: Edge ID to delete (if provided, source and target are ignored)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.delete_edge(source="A", target="B")
            
            >>> # Or delete by edge ID
            >>> G.delete_edge(id=1)
        """
        if id is not None:
            # Delete by edge ID
            self._graph.delete_edge_by_id(id)
        elif source is not None and target is not None:
            # Resolve labels to IDs
            source_id = self._resolve_node_id(source)
            target_id = self._resolve_node_id(target)
            # Find and delete first matching edge
            edge_ids = self._graph.find_edge_ids(source_id, target_id)
            if len(edge_ids) > 0:
                self._graph.delete_edge_by_id(edge_ids[0])
            else:
                raise KeyError(f"No edge found from '{source}' to '{target}'")
        else:
            raise ValueError("Must specify either (source, target) or id")
        
        return self
    
    def neighbors(self, node: Union[int, str]) -> list[int]:
        """Get neighbors of a node.
        
        Args:
            node: Node ID or label
            
        Returns:
            List of neighbor node IDs
        """
        node_id = self._resolve_node_id(node)
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
    
    def degree(self, node: Union[int, str]) -> int:
        """Get the degree of a node.
        
        For directed graphs, returns total degree (in + out).
        For undirected graphs, returns the number of neighbors.
        
        Args:
            node: Node ID or label
            
        Returns:
            Degree of the node
        """
        node_id = self._resolve_node_id(node)
        neighbors = self._graph.neighbors(node_id)
        if self._graph.is_directed:
            # For directed, count both in and out
            # edges() returns (edge_id, source, target) tuples
            in_count = sum(1 for edge in self._graph.edges() if edge[2] == node_id)
            out_count = len(neighbors)
            return in_count + out_count
        else:
            return len(neighbors)
    
    def in_degree(self, node: Union[int, str]) -> int:
        """Get the in-degree of a node (directed graphs only).
        
        Args:
            node: Node ID or label
            
        Returns:
            Number of incoming edges
            
        Raises:
            ValueError: If graph is undirected
        """
        if not self._graph.is_directed:
            raise ValueError("in_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
        # edges() returns (edge_id, source, target) tuples
        return sum(1 for edge in self._graph.edges() if edge[2] == node_id)
    
    def out_degree(self, node: Union[int, str]) -> int:
        """Get the out-degree of a node (directed graphs only).
        
        Args:
            node: Node ID or label
            
        Returns:
            Number of outgoing edges
            
        Raises:
            ValueError: If graph is undirected
        """
        if not self._graph.is_directed:
            raise ValueError("out_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
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
    
    def count_weakly_connected_components(self) -> int:
        """Count weakly connected components.
        
        For undirected graphs, this counts regular connected components.
        For directed graphs, treats edges as undirected (ignores direction).
        
        A weakly connected component is a maximal set of nodes where every
        node is reachable from every other node when ignoring edge direction.
        
        Returns:
            Number of weakly connected components
            
        Example:
            >>> G = ec.Graph(directed=False)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("X", "Y")
            >>> G.count_weakly_connected_components()
            2
        """
        return self._graph.count_weakly_connected_components()
    
    def count_strongly_connected_components(self) -> int:
        """Count strongly connected components (directed graphs only).
        
        A strongly connected component is a maximal set of nodes where every
        node is reachable from every other node following directed edges.
        
        Uses Kosaraju's algorithm.
        
        Returns:
            Number of strongly connected components
            
        Raises:
            ValueError: If called on an undirected graph
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.add_edge("C", "A")  # Cycle: all in same component
            >>> G.add_edge("X", "Y")  # Separate component
            >>> G.count_strongly_connected_components()
            2
        """
        return self._graph.count_strongly_connected_components()
    
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
    
    def count_loop_edges(self) -> int:
        """Count number of self-loops in the graph.
        
        Alias for count_selfloops().
        
        Returns:
            Number of edges where source equals target
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('A', 'A')  # Self-loop
            >>> G.count_loop_edges()
            1
        """
        return self._graph.count_selfloops()
    
    def count_multi_edges(self) -> int:
        """Count multi-edges (parallel edges between same node pair).
        
        Only counts edges beyond the first one for each node pair.
        For example, if there are 3 edges from A to B, this counts 2 of them.
        
        Returns:
            Number of duplicate edges (beyond first edge per pair)
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B', id=1)
            >>> G.add_edge('A', 'B', id=2)  # Duplicate
            >>> G.add_edge('A', 'B', id=3)  # Another duplicate
            >>> G.add_edge('B', 'C', id=4)
            >>> G.count_multi_edges()
            2
        """
        return self._graph.count_multi_edges()
    
    def count_mutual_node_pairs(self) -> int:
        """Count mutual node pairs (bidirectional edges) in directed graphs.
        
        Returns the number of node pairs (a,b) where both edges a->b and b->a exist.
        Only works for directed graphs. Self-loops are excluded.
        
        Returns:
            Number of node pairs with edges in both directions
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'A')  # Mutual pair with A
            >>> G.add_edge('A', 'C')  # One-way only
            >>> G.count_mutual_node_pairs()
            1
        """
        return self._graph.count_mutual_node_pairs()
    
    def count_asymmetric_node_pairs(self) -> int:
        """Count asymmetric node pairs (one-way edges) in directed graphs.
        
        Returns the number of node pairs (a,b) where edge a->b exists but not b->a.
        Only works for directed graphs. Self-loops are excluded.
        
        Returns:
            Number of node pairs with edges in only one direction
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'A')  # Mutual, not counted
            >>> G.add_edge('A', 'C')  # Asymmetric
            >>> G.add_edge('D', 'E')  # Asymmetric
            >>> G.count_asymmetric_node_pairs()
            2
        """
        return self._graph.count_asymmetric_node_pairs()
    
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
    
    def get_agg_degree_in(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """Get aggregate value from indegree of nodes.
        
        Computes a single aggregate statistic from the indegree values of all nodes
        (or a filtered subset) in a directed graph.
        
        Args:
            agg: Aggregation function - one of 'sum', 'min', 'max', 'mean', 'median'
            conditions: Optional callable to filter nodes. Takes a node object and
                       returns True to include it in the aggregation.
        
        Returns:
            Aggregated indegree value
            
        Raises:
            ValueError: If graph is undirected or agg function is invalid
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_gnm_graph(20, 35, seed=23)
            >>> G.get_agg_degree_in('mean')  # Mean indegree
            1.75
            >>> G.get_agg_degree_in('median')  # Median indegree
            1.5
        """
        if not self._graph.is_directed:
            raise ValueError("get_agg_degree_in() only works with directed graphs")
        
        valid_agg = {'sum', 'min', 'max', 'mean', 'median'}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")
        
        # Get indegrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.node_ids():
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.in_degree(node_id))
        
        if not degrees:
            return 0.0
        
        # Apply aggregation
        if agg == 'sum':
            return float(sum(degrees))
        elif agg == 'min':
            return float(min(degrees))
        elif agg == 'max':
            return float(max(degrees))
        elif agg == 'mean':
            return sum(degrees) / len(degrees)
        elif agg == 'median':
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n//2 - 1] + sorted_degrees[n//2]) / 2.0
            else:
                return float(sorted_degrees[n//2])
    
    def get_agg_degree_out(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """Get aggregate value from outdegree of nodes.
        
        Computes a single aggregate statistic from the outdegree values of all nodes
        (or a filtered subset) in a directed graph.
        
        Args:
            agg: Aggregation function - one of 'sum', 'min', 'max', 'mean', 'median'
            conditions: Optional callable to filter nodes. Takes a node object and
                       returns True to include it in the aggregation.
        
        Returns:
            Aggregated outdegree value
            
        Raises:
            ValueError: If graph is undirected or agg function is invalid
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_gnm_graph(20, 35, seed=23)
            >>> G.get_agg_degree_out('mean')  # Mean outdegree
            1.75
            >>> G.get_agg_degree_out('median')  # Median outdegree
            1.0
        """
        if not self._graph.is_directed:
            raise ValueError("get_agg_degree_out() only works with directed graphs")
        
        valid_agg = {'sum', 'min', 'max', 'mean', 'median'}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")
        
        # Get outdegrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.node_ids():
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.out_degree(node_id))
        
        if not degrees:
            return 0.0
        
        # Apply aggregation
        if agg == 'sum':
            return float(sum(degrees))
        elif agg == 'min':
            return float(min(degrees))
        elif agg == 'max':
            return float(max(degrees))
        elif agg == 'mean':
            return sum(degrees) / len(degrees)
        elif agg == 'median':
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n//2 - 1] + sorted_degrees[n//2]) / 2.0
            else:
                return float(sorted_degrees[n//2])
    
    def get_agg_degree_total(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """Get aggregate value from total degree of nodes.
        
        Computes a single aggregate statistic from the total degree values of all nodes
        (or a filtered subset). Works with both directed and undirected graphs.
        
        Args:
            agg: Aggregation function - one of 'sum', 'min', 'max', 'mean', 'median'
            conditions: Optional callable to filter nodes. Takes a node object and
                       returns True to include it in the aggregation.
        
        Returns:
            Aggregated total degree value
            
        Raises:
            ValueError: If agg function is invalid
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_gnm_graph(20, 35, seed=23)
            >>> G.get_agg_degree_total('mean')  # Mean total degree
            3.5
            >>> G.get_agg_degree_total('median')  # Median total degree
            3.0
        """
        valid_agg = {'sum', 'min', 'max', 'mean', 'median'}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")
        
        # Get total degrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.node_ids():
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.degree(node_id))
        
        if not degrees:
            return 0.0
        
        # Apply aggregation
        if agg == 'sum':
            return float(sum(degrees))
        elif agg == 'min':
            return float(min(degrees))
        elif agg == 'max':
            return float(max(degrees))
        elif agg == 'mean':
            return sum(degrees) / len(degrees)
        elif agg == 'median':
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n//2 - 1] + sorted_degrees[n//2]) / 2.0
            else:
                return float(sorted_degrees[n//2])
    
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
    
    def get_mean_distance(self) -> float | None:
        """Get mean distance (average shortest path length).
        
        Computes the average of all shortest path lengths between all pairs
        of reachable nodes in the graph. This is also known as the average
        path length or characteristic path length.
        
        Returns None if the graph is disconnected (since infinite distances exist).
        For directed graphs, only considers paths that exist.
        
        Returns:
            Average shortest path length, or None if graph is disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(4)  # Linear: 0-1-2-3
            >>> G.get_mean_distance()  # (1+2+3+1+2+1)/6 = 2.0
            2.0
            >>> G.add_cycle_graph(3)  # Disconnected
            >>> G.get_mean_distance() is None
            True
        """
        return self._graph.get_mean_distance()
    
    def get_min_eccentricity(self) -> int | None:
        """Get minimum eccentricity (graph radius).
        
        This is an alias for get_graph_radius(). The minimum eccentricity
        is the smallest maximum distance from any node to all other nodes.
        
        Returns None if the graph is disconnected.
        
        Returns:
            Minimum eccentricity (radius), or None if disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # Hub at center
            >>> G.get_min_eccentricity()  # Hub has eccentricity 1
            1
            
        See Also:
            get_graph_radius(): Equivalent method
        """
        return self.get_graph_radius()
    
    def get_max_eccentricity(self) -> int | None:
        """Get maximum eccentricity (graph diameter).
        
        This is an alias for get_graph_diameter(). The maximum eccentricity
        is the longest shortest path between any two nodes in the graph.
        
        Returns None if the graph is disconnected.
        
        Returns:
            Maximum eccentricity (diameter), or None if disconnected
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(5)  # Linear chain
            >>> G.get_max_eccentricity()  # Distance from end to end
            4
            
        See Also:
            get_graph_diameter(): Equivalent method
        """
        return self.get_graph_diameter()
    
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
    
    def get_reciprocity(self) -> float:
        """Get reciprocity (fraction of reciprocal edges).
        
        Reciprocity measures the proportion of edges that have a corresponding
        reverse edge. Only defined for directed graphs.
        
        Returns a value between 0.0 (no reciprocal edges) and 1.0 (all edges
        are reciprocal). Self-loops are excluded from the calculation.
        
        Returns:
            Fraction of reciprocal edges (0.0 to 1.0)
            
        Raises:
            ValueError: If graph is undirected
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "A")  # Reciprocal
            >>> G.add_edge("A", "C")  # Not reciprocal
            >>> G.get_reciprocity()  # 2/3 ≈ 0.667
            0.6666666666666666
        """
        return self._graph.get_reciprocity()
    
    def get_min_cut_between(self, source: str, sink: str) -> int:
        """Get minimum cut between source and sink nodes.
        
        Computes the minimum number of edges that must be removed to
        disconnect the source from the sink. Uses the Ford-Fulkerson
        algorithm with BFS (Edmonds-Karp) to find maximum flow, which
        equals the minimum cut by the max-flow min-cut theorem.
        
        For unweighted graphs, each edge has capacity 1. For undirected
        graphs, edges are treated as bidirectional.
        
        Args:
            source: Source node ID
            sink: Sink node ID
            
        Returns:
            Size of minimum cut (number of edges to remove)
            
        Raises:
            ValueError: If source or sink node doesn't exist, or if they're the same
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_path_graph(4)  # 0-1-2-3
            >>> G.get_min_cut_between("0", "3")  # Must cut at least 1 edge
            1
            >>> G.add_edge("1", "2")  # Add parallel edge
            >>> G.get_min_cut_between("0", "3")  # Now need to cut 2 edges
            2
        """
        return self._graph.get_min_cut_between(source, sink)
    
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
    
    # ========== Community Detection Methods ==========
    
    def get_cmty_fast_greedy(self) -> dict[str, int]:
        """
        Detect communities using Fast Greedy (Clauset-Newman-Moore) algorithm.
        
        This is a greedy modularity optimization algorithm that merges communities
        to maximize modularity. It's efficient for medium to large graphs and provides
        good quality communities.
        
        Algorithm:
            1. Start with each node in its own community
            2. Build a heap of merge candidates ordered by modularity gain (ΔQ)
            3. Repeatedly merge the pair with highest ΔQ until no improvement
            4. Return the partition with maximum modularity
        
        Returns:
            dict[str, int]: Mapping from node ID to community ID (0-indexed)
            
        Raises:
            ValueError: If graph is directed (requires undirected graph)
            
        Complexity:
            Time: O(n log²n) on sparse graphs
            Space: O(n + m)
            
        Examples:
            >>> G = ec.Graph(directed=False)
            >>> # Two disconnected cliques
            >>> G.add_edges([('A','B'), ('B','C'), ('C','A')])
            >>> G.add_edges([('D','E'), ('E','F'), ('F','D')])
            >>> communities = G.get_cmty_fast_greedy()
            >>> # Should find 2 communities: {A,B,C} and {D,E,F}
            >>> len(set(communities.values()))
            2
            
            >>> # Karate club example
            >>> G = ec.Graph(directed=False)
            >>> G.add_edges([...])  # Add karate club edges
            >>> communities = G.get_cmty_fast_greedy()
            >>> # Modularity typically > 0.40 for karate club
            
        References:
            Clauset, A., Newman, M. E., & Moore, C. (2004).
            "Finding community structure in very large networks."
            Physical review E, 70(6), 066111.
            
        See Also:
            - get_cmty_louvain(): Faster for very large graphs
            - get_cmty_edge_btwns(): More accurate but slower
        """
        return self._graph.get_cmty_fast_greedy()
    
    def get_cmty_edge_btwns(self) -> dict[str, int]:
        """Get community membership using Edge Betweenness (Girvan-Newman) algorithm.
        
        Iteratively removes edges with highest betweenness centrality to reveal
        community structure. Uses Brandes' algorithm for efficient betweenness
        calculation. More accurate than Fast Greedy but significantly slower.
        
        Algorithm:
            1. Compute edge betweenness for all edges using Brandes' algorithm
            2. Remove edge with highest betweenness
            3. Recompute betweenness and check if modularity improved
            4. Repeat until modularity stops improving
            5. Return partition with maximum modularity
        
        Best for:
            - Small to medium graphs (< 5K nodes)
            - Hierarchical community structure
            - When accuracy is more important than speed
            - Graphs with clear "bridge" edges between communities
        
        Complexity: O(m²n) where m = edges, n = nodes
        
        Returns:
            Dictionary mapping node IDs to community IDs (0-indexed).
            Community IDs are contiguous integers starting from 0.
        
        Raises:
            ValueError: If graph is directed (requires undirected graph)
        
        Examples:
            >>> import edgecase as ec
            >>> G = ec.Graph(directed=False)
            >>> 
            >>> # Two triangles connected by a bridge
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('B', 'C')
            >>> G.add_edge('C', 'A')
            >>> G.add_edge('C', 'D')  # Bridge edge
            >>> G.add_edge('D', 'E')
            >>> G.add_edge('E', 'F')
            >>> G.add_edge('F', 'D')
            >>> 
            >>> communities = G.get_cmty_edge_btwns()
            >>> # Returns: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
            >>> # The bridge edge C-D has highest betweenness and is removed first
        
        References:
            Girvan, M., & Newman, M. E. (2002). Community structure in social
            and biological networks. PNAS, 99(12), 7821-7826.
            
            Brandes, U. (2001). A faster algorithm for betweenness centrality.
            Journal of Mathematical Sociology, 25(2), 163-177.
        
        Note:
            This algorithm is O(m²n) and can be slow for large graphs.
            For graphs with > 10K nodes, consider get_cmty_fast_greedy() or
            get_cmty_louvain() instead.
            
        See Also:
            - get_cmty_fast_greedy(): Faster greedy modularity optimization
            - get_cmty_louvain(): Best for very large graphs
            - get_cmty_walktrap(): Alternative hierarchical method
        """
        return self._graph.get_cmty_edge_btwns()
    
    def get_cmty_l_eigenvec(self) -> dict[str, int]:
        """Get community membership using Leading Eigenvector (Newman) algorithm.
        
        Spectral optimization method using recursive bisection based on the
        leading eigenvector of the modularity matrix. Splits communities
        hierarchically when modularity improves.
        
        Algorithm:
            1. Start with all nodes in one community
            2. Compute modularity matrix B_ij = A_ij - (k_i * k_j) / 2m
            3. Find leading eigenvector using power iteration
            4. Split by sign: positive → community 1, negative → community 2
            5. Accept split only if modularity improves
            6. Recursively split each community
        
        Best for:
            - Medium-sized graphs (< 20K nodes)
            - Hierarchical community structure
            - When you want theoretically-grounded splits
            - Graphs with clear 2-way divisions
        
        Complexity: O(n²) per split, recursive
        
        Returns:
            Dictionary mapping node IDs to community IDs (0-indexed).
            Community IDs are contiguous integers starting from 0.
        
        Raises:
            ValueError: If graph is directed (requires undirected graph)
        
        Examples:
            >>> import edgecase as ec
            >>> G = ec.Graph(directed=False)
            >>> 
            >>> # Two cliques connected by a bridge
            >>> for i in ['A', 'B', 'C']:
            ...     for j in ['A', 'B', 'C']:
            ...         if i < j:
            ...             G.add_edge(i, j)
            >>> 
            >>> G.add_edge('C', 'D')  # Bridge
            >>> 
            >>> for i in ['D', 'E', 'F']:
            ...     for j in ['D', 'E', 'F']:
            ...         if i < j:
            ...             G.add_edge(i, j)
            >>> 
            >>> communities = G.get_cmty_l_eigenvec()
            >>> # Returns: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
            >>> # Eigenvector method finds the natural split at the bridge
        
        References:
            Newman, M. E. J. (2006). Finding community structure in networks
            using the eigenvectors of matrices. Physical Review E, 74(3), 036104.
            
            Newman, M. E. J. (2006). Modularity and community structure in
            networks. PNAS, 103(23), 8577-8582.
        
        Note:
            This algorithm uses power iteration to find the leading eigenvector,
            which is O(n²) per split. For very large graphs (> 50K nodes),
            consider get_cmty_fast_greedy() or get_cmty_louvain() instead.
            
            The algorithm only splits when modularity improves, so it may
            return fewer communities than other methods if the structure is
            not hierarchically organized.
            
        See Also:
            - get_cmty_fast_greedy(): Faster but less principled
            - get_cmty_edge_btwns(): More accurate but slower
            - get_cmty_louvain(): Best for very large graphs
        """
        return self._graph.get_cmty_l_eigenvec()
    
    def get_cmty_louvain(self, resolution: float = 1.0) -> dict[str, int]:
        """Detect communities using the Louvain multi-level optimization algorithm.
        
        The Louvain method is a hierarchical algorithm that maximizes modularity through
        iterative optimization. It's the most widely-used community detection algorithm
        in practice due to its speed and quality on large networks.
        
        Algorithm:
            1. Phase 1 (Local optimization): Each node tries joining neighboring 
               communities to maximize modularity gain ΔQ.
            2. Phase 2 (Aggregation): Communities are coarsened into super-nodes,
               creating a new network where each community becomes a single node.
            3. Repeat until modularity converges (no further improvement).
        
        The method produces a dendrogram hierarchy, but returns the partition with
        maximum modularity. Time complexity is O(n log n) for typical networks.
        
        Requires an undirected graph.
        
        Args:
            resolution: Resolution parameter (default: 1.0). Higher values favor 
                       smaller communities. Standard modularity uses resolution=1.0.
        
        Returns:
            Dictionary mapping node ID to community ID (0-indexed, contiguous).
            Nodes in the same community have the same ID.
        
        Raises:
            RuntimeError: If called on a directed graph.
        
        Examples:
            >>> G = ec.Graph(directed=False)
            >>> # Two triangles connected by bridge
            >>> for i in ['A', 'B', 'C']:
            ...     for j in ['A', 'B', 'C']:
            ...         if i < j:
            ...             G.add_edge(i, j)
            >>> for i in ['D', 'E', 'F']:
            ...     for j in ['D', 'E', 'F']:
            ...         if i < j:
            ...             G.add_edge(i, j)
            >>> G.add_edge('C', 'D')
            >>> communities = G.get_cmty_louvain()
            >>> communities['A'] == communities['B'] == communities['C']
            True
            >>> communities['D'] == communities['E'] == communities['F']
            True
            >>> communities['A'] != communities['D']  # Different communities
            True
        
        Notes:
            - The resolution parameter allows tuning community size:
              * resolution < 1.0: Larger communities
              * resolution = 1.0: Standard modularity optimization
              * resolution > 1.0: Smaller communities
            - The algorithm is deterministic but may produce different (equivalent)
              partitions depending on node ordering.
            - Handles disconnected components naturally.
            - Self-loops and multi-edges are currently not supported.
        
        References:
            - Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
              "Fast unfolding of communities in large networks." Journal of Statistical
              Mechanics: Theory and Experiment, 2008(10), P10008.
            - Lambiotte, R., Delvenne, J. C., & Barahona, M. (2014). "Random walks,
              Markov processes and the multiscale modular organization of complex
              networks." IEEE Transactions on Network Science and Engineering, 1(2), 76-90.
        
        See Also:
            - get_cmty_fast_greedy(): Simpler greedy optimization
            - get_cmty_edge_btwns(): Divisive hierarchical method
            - get_cmty_l_eigenvec(): Spectral bisection method
        """
        return self._graph.get_cmty_louvain(resolution)
    
    def get_cmty_walktrap(self, steps: int = 4) -> dict[str, int]:
        """Detect communities using the Walktrap random walk algorithm.
        
        The Walktrap algorithm exploits the property that random walks tend to stay
        trapped within densely connected communities. It computes distances between
        nodes based on random walk probabilities, then uses hierarchical clustering
        to identify community structure.
        
        Algorithm:
            1. Compute transition probability matrix P (P[i][j] = 1/deg(i) if edge exists)
            2. Compute t-step transition probabilities P^t by matrix exponentiation
            3. Calculate random walk distance between each pair of nodes:
               d(i,j) = sqrt(Σ_k [(P^t_ik - P^t_jk)^2 / deg(k)])
            4. Apply Ward's hierarchical clustering on the distance matrix
            5. Cut the dendrogram at the level that maximizes modularity
        
        The distance metric captures structural similarity: nodes in the same community
        have similar random walk probabilities to all other nodes. Time complexity is
        O(n²log n) for typical networks with small step parameter.
        
        Requires an undirected graph.
        
        Args:
            steps: Number of random walk steps (default: 4). Higher values capture
                  longer-range structural patterns. Typical values: 2-8.
                  - steps=2: Local structure only
                  - steps=4: Good balance (default, recommended)
                  - steps=6+: Longer-range patterns, more computation
        
        Returns:
            Dictionary mapping node ID to community ID (0-indexed, contiguous).
            Nodes in the same community have the same ID.
        
        Raises:
            ValueError: If called on a directed graph.
        
        Examples:
            >>> G = ec.Graph(directed=False)
            >>> # Two cliques connected by a bridge
            >>> G.add_edges([('A', 'B'), ('B', 'C'), ('C', 'A')])  # Clique 1
            >>> G.add_edges([('D', 'E'), ('E', 'F'), ('F', 'D')])  # Clique 2
            >>> G.add_edge('C', 'D')  # Bridge
            >>> communities = G.get_cmty_walktrap()
            >>> communities['A'] == communities['B'] == communities['C']
            True
            >>> communities['D'] == communities['E'] == communities['F']
            True
            >>> communities['A'] != communities['D']  # Different communities
            True
            
            >>> # Try different step lengths
            >>> G.add_edges([('X', 'Y'), ('Y', 'Z')])  # Another component
            >>> communities_2 = G.get_cmty_walktrap(steps=2)  # More local
            >>> communities_6 = G.get_cmty_walktrap(steps=6)  # More global
        
        Notes:
            - The step parameter controls the scale of community detection:
              * Lower steps (2-3): Detect finer, more local communities
              * Higher steps (5-8): Detect coarser, more global communities
            - Algorithm is deterministic (no randomness despite the name)
            - Works well on hierarchical community structures
            - Handles disconnected components automatically
            - Performance degrades on very large graphs (>10,000 nodes) due to
              O(n²) distance matrix computation
            - Self-loops and multi-edges are currently not supported
        
        References:
            - Pons, P., & Latapy, M. (2005). "Computing communities in large networks
              using random walks." In Computer and Information Sciences-ISCIS 2005
              (pp. 284-293). Springer, Berlin, Heidelberg.
            - Pons, P., & Latapy, M. (2006). "Computing communities in large networks
              using random walks (long version)." arXiv preprint physics/0512106.
        
        Performance:
            - Small graphs (<1000 nodes): <1 second
            - Medium graphs (1000-5000 nodes): 1-30 seconds
            - Large graphs (>5000 nodes): May be slow, consider Louvain instead
        
        See Also:
            - get_cmty_louvain(): Faster for large graphs
            - get_cmty_fast_greedy(): Alternative greedy approach
            - get_cmty_edge_btwns(): Divisive hierarchical method
            - get_cmty_l_eigenvec(): Spectral method
        """
        return self._graph.get_cmty_walktrap(steps)


    def colorize_node_attrs(
        self,
        node_attr_from: str = None,
        values: dict[str, Any] = None,
        node_attr_to: str = "color",
        palette: str | list[str] = "default",
        alpha: Optional[int] = None,
        nodes: Optional[list[str]] = None,
    ) -> "Graph":
        """Map node attribute values to colors.
        
        See edgecase.aesthetics.colorize_node_attrs() for full documentation.
        """
        from edgecase.aesthetics import colorize_node_attrs
        return colorize_node_attrs(
            self, node_attr_from=node_attr_from, values=values,
            node_attr_to=node_attr_to, palette=palette, alpha=alpha, nodes=nodes
        )
    
    def rescale_node_attrs(
        self,
        node_attr_from: str = None,
        values: dict[str, int | float] = None,
        node_attr_to: str = "size",
        to_lower_bound: float = 0.5,
        to_upper_bound: float = 1.0,
        nodes: Optional[list[str]] = None,
    ) -> "Graph":
        """Rescale node attribute values to a range.
        
        See edgecase.aesthetics.rescale_node_attrs() for full documentation.
        """
        from edgecase.aesthetics import rescale_node_attrs
        return rescale_node_attrs(
            self, node_attr_from=node_attr_from, values=values,
            node_attr_to=node_attr_to, to_lower_bound=to_lower_bound,
            to_upper_bound=to_upper_bound, nodes=nodes
        )
    
    def set_node_attr_to_display(
        self,
        attr: Optional[str] = None,
    ) -> "Graph":
        """Set which node attribute to display in tooltips/labels.
        
        See edgecase.aesthetics.set_node_attr_to_display() for full documentation.
        """
        from edgecase.aesthetics import set_node_attr_to_display
        return set_node_attr_to_display(self, attr=attr)
    
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
    
    def get_eigen_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get eigenvector centrality for each node.
        
        Eigenvector centrality measures node importance based on connections
        to other important nodes. It's the principal eigenvector of the
        adjacency matrix, computed via power iteration: x = A * x
        
        A node has high eigenvector centrality if it's connected to other
        nodes that themselves have high centrality. Unlike degree centrality,
        it accounts for the importance of neighbors.
        
        Args:
            max_iter: Maximum iterations (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            
        Returns:
            Dictionary mapping node ID to eigenvector centrality score (normalized)
            
        Example:
            >>> G = ec.Graph(directed=False)
            >>> G.add_star_graph(5)
            >>> eigen = G.get_eigen_centrality()
            >>> # Hub has highest eigenvector centrality
            >>> # All connected nodes have non-zero centrality
        """
        return self._graph.get_eigen_centrality(max_iter, tol)
    
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
    
    def get_dice_similarity(self) -> dict[str, dict[str, float]]:
        """Get Dice similarity coefficient between all node pairs.
        
        The Dice coefficient measures the similarity between two nodes based on
        their neighborhood overlap:
        
        Dice(A, B) = 2 * |neighbors(A) ∩ neighbors(B)| / (|neighbors(A)| + |neighbors(B)|)
        
        Range: [0, 1]
        - 0: No common neighbors
        - 1: Identical neighborhoods (or same node)
        
        Useful for:
        - Link prediction (similar nodes likely to connect)
        - Community detection (nodes with similar neighborhoods)
        - Recommendation systems (find similar users/items)
        
        Returns:
            Nested dictionary: {node1: {node2: similarity_score}}
            Self-similarity (node to itself) is always 1.0
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_edge('A', 'B')
            >>> G.add_edge('A', 'C')
            >>> G.add_edge('D', 'B')
            >>> G.add_edge('D', 'C')
            >>> sim = G.get_dice_similarity()
            >>> sim['A']['D']  # A and D share neighbors B and C
            1.0
        """
        result_ids = self._graph.get_dice_similarity()
        # Convert node IDs to labels in the result
        result = {}
        for node1_id, inner_dict in result_ids.items():
            node1_label = self.label_from_id(node1_id)
            result[node1_label] = {}
            for node2_id, score in inner_dict.items():
                node2_label = self.label_from_id(node2_id)
                result[node1_label][node2_label] = score
        return result
    
    def get_jaccard_similarity(self) -> dict[str, dict[str, float]]:
        """Get Jaccard similarity index between all node pairs.
        
        The Jaccard index is a classic set similarity measure:
        
        Jaccard(A, B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|
        
        Range: [0, 1]
        - 0: No common neighbors
        - 1: Identical neighborhoods (or same node)
        
        More conservative than Dice coefficient (always ≤ Dice score).
        Widely used in:
        - Document similarity (bag-of-words)
        - User similarity (common items/friends)
        - Biological networks (protein interactions)
        
        Returns:
            Nested dictionary: {node1: {node2: similarity_score}}
            Self-similarity (node to itself) is always 1.0
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_star_graph(5)  # Hub connected to 4 leaves
            >>> sim = G.get_jaccard_similarity()
            >>> # Leaves have zero Jaccard (no common neighbors)
            >>> sim['2']['3']
            0.0
        """
        result_ids = self._graph.get_jaccard_similarity()
        # Convert node IDs to labels in the result
        result = {}
        for node1_id, inner_dict in result_ids.items():
            node1_label = self.label_from_id(node1_id)
            result[node1_label] = {}
            for node2_id, score in inner_dict.items():
                node2_label = self.label_from_id(node2_id)
                result[node1_label][node2_label] = score
        return result
    
    def get_cosine_similarity(self) -> dict[str, dict[str, float]]:
        """Get Cosine similarity between all node pairs.
        
        Cosine similarity treats nodes as vectors in the graph adjacency space:
        
        Cosine(A, B) = |A ∩ B| / (√|A| * √|B|)
        
        Range: [0, 1]
        - 0: No common neighbors
        - 1: Identical neighborhoods (or same node)
        
        Geometric interpretation: angle between adjacency vectors.
        Useful for:
        - High-dimensional graph embeddings
        - Collaborative filtering (user-item matrices)
        - Text analysis (TF-IDF vectors)
        
        Returns:
            Nested dictionary: {node1: {node2: similarity_score}}
            Self-similarity (node to itself) is always 1.0
            
        Example:
            >>> G = ec.Graph()
            >>> G.add_cycle_graph(5)
            >>> sim = G.get_cosine_similarity()
            >>> # Adjacent nodes in cycle share one neighbor
            >>> sim['1']['2'] > 0
            True
        """
        result_ids = self._graph.get_cosine_similarity()
        # Convert node IDs to labels in the result
        result = {}
        for node1_id, inner_dict in result_ids.items():
            node1_label = self.label_from_id(node1_id)
            result[node1_label] = {}
            for node2_id, score in inner_dict.items():
                node2_label = self.label_from_id(node2_id)
                result[node1_label][node2_label] = score
        return result
    
    def get_overlap_similarity(self) -> dict[str, dict[str, float]]:
        """Get Overlap coefficient between all node pairs.
        
        The Overlap coefficient emphasizes subset relationships:
        
        Overlap(A, B) = |neighbors(A) ∩ neighbors(B)| / min(|neighbors(A)|, |neighbors(B)|)
        
        Range: [0, 1]
        - 0: No common neighbors
        - 1: One neighborhood is a subset of the other (or same node)
        
        Useful for:
        - Detecting hierarchical relationships (parent-child)
        - Finding hubs (nodes whose neighbors are subsets of others)
        - Asymmetric similarity (specialization vs generalization)
        
        Returns:
            Nested dictionary: {node1: {node2: similarity_score}}
            Self-similarity (node to itself) is always 1.0
            
        Example:
            >>> G = ec.Graph()
            >>> # Create hub-spoke with one spoke having extra connection
            >>> G.add_edge('hub', 'A')
            >>> G.add_edge('hub', 'B')
            >>> G.add_edge('A', 'X')
            >>> sim = G.get_overlap_similarity()
            >>> # hub's neighborhood is subset of A's neighborhood
            >>> sim['hub']['A']
            1.0
        """
        result_ids = self._graph.get_overlap_similarity()
        # Convert node IDs to labels in the result
        result = {}
        for node1_id, inner_dict in result_ids.items():
            node1_label = self.label_from_id(node1_id)
            result[node1_label] = {}
            for node2_id, score in inner_dict.items():
                node2_label = self.label_from_id(node2_id)
                result[node1_label][node2_label] = score
        return result
    
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
    
    def transform_to_subgraph_ws(self) -> "Graph":
        """Create a subgraph from current selection.
                
        Creates a new graph containing:
        - If nodes are selected: those nodes and all edges between them
        - If edges are selected: those edges and their endpoint nodes
        - If both selected: uses node selection
        
        Returns:
            New Graph object containing the subgraph
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_path_graph(6)  # 1 → 2 → 3 → 4 → 5 → 6
            >>> G.select_nodes(lambda n: int(n.id) > 2)
            >>> subgraph = G.transform_to_subgraph_ws()
            >>> subgraph.num_nodes()  # 4 nodes (3, 4, 5, 6)
            4
        """
        selected_nodes = self.get_selected_nodes()
        selected_edges = self.get_selected_edges()
        
        # Create new graph with same directedness
        subgraph = Graph(directed=self.directed)
        
        if len(selected_nodes) > 0:
            # Node-based subgraph: include selected nodes and edges between them
            node_set = set(selected_nodes)
            
            # Add nodes with their attributes
            for node_id in selected_nodes:
                node = self.get_node(node_id)
                if node and node.attributes:
                    subgraph.add_node(node_id, **node.attributes)
                else:
                    subgraph.add_node(node_id)
            
            # Add edges that connect selected nodes
            for edge_id in self.edge_ids():
                edge = self.get_edge(edge_id)
                if edge and edge.source in node_set and edge.target in node_set:
                    # Prepare edge attributes
                    attrs = {}
                    if edge.weight is not None:
                        attrs['weight'] = edge.weight
                    if edge.attributes:
                        attrs.update(edge.attributes)
                    subgraph.add_edge(edge.source, edge.target, **attrs)
                    
        elif len(selected_edges) > 0:
            # Edge-based subgraph: include selected edges and their endpoints
            nodes_to_add = set()
            edges_to_add = []
            
            # Collect nodes and edges
            for edge_id in selected_edges:
                edge = self.get_edge(edge_id)
                if edge:
                    nodes_to_add.add(edge.source)
                    nodes_to_add.add(edge.target)
                    edges_to_add.append(edge)
            
            # Add nodes
            for node_id in nodes_to_add:
                node = self.get_node(node_id)
                if node and node.attributes:
                    subgraph.add_node(node_id, **node.attributes)
                else:
                    subgraph.add_node(node_id)
            
            # Add edges
            for edge in edges_to_add:
                attrs = {}
                if edge.weight is not None:
                    attrs['weight'] = edge.weight
                if edge.attributes:
                    attrs.update(edge.attributes)
                subgraph.add_edge(edge.source, edge.target, **attrs)
        
        return subgraph
    
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
    
    def get_node(self, node: Union[int, str]):
        """Get a node object by ID or label.
        
        Args:
            node: Node ID (int) or label (str) to retrieve
            
        Returns:
            Node object or None if not found
        """
        node_id = self._resolve_node_id(node, auto_create=False)
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
    
    def set_node_attrs(self, node_attr: str, values, nodes: list[str] | None = None) -> "Graph":
        """Set attribute values for one or more nodes.
        
        Args:
            node_attr: The name of the attribute to set
            values: The value(s) to set. Can be a single value (applied to all nodes)
                   or a list/dict of values matching the nodes
            nodes: Optional list of node IDs to set attributes for (default: all nodes)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_node("A")
            >>> G.add_node("B")
            >>> G.set_node_attrs("color", "green", nodes=["A"])
            >>> G.get_node_attrs("color")
            {'A': 'green'}
            
            >>> # Set for all nodes
            >>> G.set_node_attrs("status", "active")
            >>> G.get_node_attrs("status")
            {'A': 'active', 'B': 'active'}
        """
        if nodes is None:
            nodes = self.node_ids()
        
        # Handle different value types
        if isinstance(values, dict):
            # Dict mapping node IDs to values
            for node_id in nodes:
                if node_id in values:
                    self._graph.set_node_attr(node_id, node_attr, values[node_id])
        elif isinstance(values, list):
            # List of values (must match length of nodes)
            if len(values) != len(nodes):
                raise ValueError(f"Length of values ({len(values)}) must match length of nodes ({len(nodes)})")
            for node_id, value in zip(nodes, values):
                self._graph.set_node_attr(node_id, node_attr, value)
        else:
            # Single value applied to all nodes
            for node_id in nodes:
                self._graph.set_node_attr(node_id, node_attr, values)
        
        return self
    
    def get_edge_attrs(
        self, 
        edge_attr: str, 
        sources: list[Union[int, str]] | None = None, 
        targets: list[Union[int, str]] | None = None
    ) -> dict[str, any]:
        """Get edge attribute values.
                
        Args:
            edge_attr: The name of the attribute to get
            sources: Optional list of source node IDs/labels to filter edges
            targets: Optional list of target node IDs/labels to filter edges
            
        Returns:
            Dictionary mapping edge strings (e.g., "A->B") to attribute values
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_edge("A", "B", weight=1.5, type="friend")
            >>> G.add_edge("B", "C", weight=2.0, type="friend")
            >>> G.get_edge_attrs("weight")
            {'A->B': 1.5, 'B->C': 2.0}
            
            >>> # Filter by source/target nodes
            >>> G.get_edge_attrs("weight", sources=["A"], targets=["B"])
            {'A->B': 1.5}
        """
        result = {}
        separator = "->" if self.directed else "--"
        
        # Determine which edges to check
        if sources is not None and targets is not None:
            # Filter by specific source/target pairs
            for source in sources:
                source_id = self._resolve_node_id(source)
                for target in targets:
                    target_id = self._resolve_node_id(target)
                    edge_ids = self._graph.find_edge_ids(source_id, target_id)
                    for edge_id in edge_ids:
                        edge = self.get_edge(edge_id)
                        if edge:
                            # Convert IDs to labels
                            source_label = self.label_from_id(edge.source)
                            target_label = self.label_from_id(edge.target)
                            edge_key = f"{source_label}{separator}{target_label}"
                            # Check special weight attribute first
                            if edge_attr == "weight" and edge.weight is not None:
                                result[edge_key] = edge.weight
                            # Then check attributes dict
                            elif edge.attributes and edge_attr in edge.attributes:
                                result[edge_key] = edge.attributes[edge_attr]
        else:
            # Get all edges
            for edge_id in self.edge_ids():
                edge = self.get_edge(edge_id)
                if edge:
                    # Convert IDs to labels
                    source_label = self.label_from_id(edge.source)
                    target_label = self.label_from_id(edge.target)
                    edge_key = f"{source_label}{separator}{target_label}"
                    # Check special weight attribute first
                    if edge_attr == "weight" and edge.weight is not None:
                        result[edge_key] = edge.weight
                    # Then check attributes dict
                    elif edge.attributes and edge_attr in edge.attributes:
                        result[edge_key] = edge.attributes[edge_attr]
        
        return result
    
    def get_edge_attrs_by_id(self, attr: str, edges: list[int] | None = None) -> dict[int, any]:
        """Get attribute values for edges by edge ID.
        
        Args:
            attr: Attribute name to extract
            edges: Optional list of edge IDs (default: all edges)
            
        Returns:
            Dictionary mapping edge IDs to attribute values
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B", weight=1.5, type="friend")
            >>> G.get_edge_attrs_by_id("weight")
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
    
    def set_edge_attrs(
        self, 
        edge_attr: str, 
        values, 
        edges: list[tuple[Union[int, str], Union[int, str]]] | None = None,
        sources: list[Union[int, str]] | None = None,
        targets: list[Union[int, str]] | None = None
    ) -> "Graph":
        """Set attribute values for one or more edges.
        
        Args:
            edge_attr: The name of the attribute to set
            values: The value(s) to set. Can be a single value (applied to all edges),
                   a list of values matching the edges, or a dict mapping edge IDs to values
            edges: Optional list of (source, target) tuples specifying edges to set attributes for
            sources: Optional list of source node IDs/labels to filter edges (used with targets)
            targets: Optional list of target node IDs/labels to filter edges (used with sources)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph()
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> # Set attribute for specific edges
            >>> G.set_edge_attrs("weight", 2.5, edges=[("A", "B")])
            >>> G.get_edge_attrs("weight")
            {'A->B': 2.5}
            
            >>> # Set for all edges
            >>> G.set_edge_attrs("status", "active")
            >>> G.get_edge_attrs("status")
            {'A->B': 'active', 'B->C': 'active'}
            
            >>> # Filter by source/target nodes
            >>> G.set_edge_attrs("type", "friend", sources=["A"], targets=["B"])
            >>> G.get_edge_attrs("type")
            {'A->B': 'friend'}
        """
        # Determine which edges to update
        target_edge_ids = []
        
        if edges is not None:
            # Use specific (source, target) tuples
            for source, target in edges:
                try:
                    source_id = self._resolve_node_id(source)
                    target_id = self._resolve_node_id(target)
                    edge_ids = self._graph.find_edge_ids(source_id, target_id)
                    target_edge_ids.extend(edge_ids)
                except ValueError:
                    # Node doesn't exist - silently skip
                    pass
        elif sources is not None and targets is not None:
            # Filter by source/target node lists
            for source in sources:
                try:
                    source_id = self._resolve_node_id(source)
                except ValueError:
                    continue
                for target in targets:
                    try:
                        target_id = self._resolve_node_id(target)
                        edge_ids = self._graph.find_edge_ids(source_id, target_id)
                        target_edge_ids.extend(edge_ids)
                    except ValueError:
                        # Node doesn't exist - silently skip
                        pass
        else:
            # All edges
            target_edge_ids = self.edge_ids()
        
        # Handle different value types
        if isinstance(values, dict):
            # Dict mapping edge IDs to values
            for edge_id in target_edge_ids:
                if edge_id in values:
                    self._graph.set_edge_attr(edge_id, edge_attr, values[edge_id])
        elif isinstance(values, list):
            # List of values (must match length of target edges)
            if len(values) != len(target_edge_ids):
                raise ValueError(
                    f"Length of values ({len(values)}) must match length of target edges ({len(target_edge_ids)})"
                )
            for edge_id, value in zip(target_edge_ids, values):
                self._graph.set_edge_attr(edge_id, edge_attr, value)
        else:
            # Single value applied to all target edges
            for edge_id in target_edge_ids:
                self._graph.set_edge_attr(edge_id, edge_attr, values)
        
        return self
    
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
    
    def get_non_neighbors(self, node_id: str) -> list[str]:
        """Get nodes that are NOT neighbors of the specified node.
        
        Returns all nodes in the graph except:
        - The specified node itself
        - Neighbors of the specified node
        
        Args:
            node_id: Node ID to find non-neighbors for
            
        Returns:
            List of non-neighbor node IDs
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_path_graph(5)  # 1 → 2 → 3 → 4 → 5
            >>> G.get_non_neighbors("2")  # Not connected to 4, 5
            ['4', '5']
        """
        all_nodes = set(self.node_ids())
        neighbors = set(self.get_neighbors(node_id, mode="all"))
        
        # Remove the node itself and its neighbors
        non_neighbors = all_nodes - neighbors - {node_id}
        
        return sorted(list(non_neighbors))
    
    def get_similar_neighbors(
        self,
        node_id: str,
        node_attr: str,
        tol_abs: tuple[float, float] | None = None,
        tol_pct: tuple[float, float] | None = None
    ) -> list[str]:
        """Get neighbors with similar attribute values.
        
        Returns neighbors whose attribute values are within tolerance of the
        starting node's value. For numeric attributes only.
        
        Args:
            node_id: Starting node ID
            node_attr: Attribute name to compare
            tol_abs: Absolute tolerance as (lower_bound, upper_bound) from starting value
            tol_pct: Percentage tolerance as (lower_pct, upper_pct) from starting value
            
        Returns:
            List of neighbor node IDs with similar attribute values
            
        Example:
            >>> G = Graph()
            >>> G.add_node("A", value=5.0)
            >>> G.add_node("B", value=5.5)
            >>> G.add_node("C", value=10.0)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("A", "C")
            >>> G.get_similar_neighbors("A", "value", tol_abs=(1.0, 1.0))
            ['B']  # C's value (10) is outside tolerance
        """
        # Get starting node's attribute value
        start_node = self.get_node(node_id)
        if not start_node or not start_node.attributes:
            return []
        
        if node_attr not in start_node.attributes:
            return []
        
        start_value = start_node.attributes[node_attr]
        
        # Must be numeric
        if not isinstance(start_value, (int, float)):
            raise ValueError(f"Attribute '{node_attr}' must be numeric for similarity comparison")
        
        # Determine bounds
        if tol_abs is not None:
            lower_bound = start_value - tol_abs[0]
            upper_bound = start_value + tol_abs[1]
        elif tol_pct is not None:
            lower_bound = start_value * (1 - tol_pct[0] / 100.0)
            upper_bound = start_value * (1 + tol_pct[1] / 100.0)
        else:
            # Exact match if no tolerance specified
            lower_bound = start_value
            upper_bound = start_value
        
        # Get all neighbors
        neighbors = self.get_neighbors(node_id, mode="all")
        
        # Filter by attribute similarity
        similar = []
        for neighbor_id in neighbors:
            neighbor = self.get_node(neighbor_id)
            if neighbor and neighbor.attributes and node_attr in neighbor.attributes:
                neighbor_value = neighbor.attributes[node_attr]
                if isinstance(neighbor_value, (int, float)):
                    if lower_bound <= neighbor_value <= upper_bound:
                        similar.append(neighbor_id)
        
        return similar
    
    def get_all_connected_nodes(self, node_id: str) -> list[str]:
        """Get all nodes reachable from the specified node.
        
        Uses BFS to find all nodes in the same connected component.
        For directed graphs, considers the graph as undirected.
        
        Args:
            node_id: Starting node ID
            
        Returns:
            List of all reachable node IDs (excluding the starting node)
            
        Example:
            >>> G = Graph()
            >>> G.add_path_graph(3)  # 1-2-3
            >>> G.add_path_graph(2)  # 4-5 (separate component)
            >>> G.get_all_connected_nodes("1")
            ['2', '3']
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' does not exist")
        
        visited = set()
        queue = [node_id]
        visited.add(node_id)
        
        while queue:
            current = queue.pop(0)
            
            # Get all neighbors (treating directed as undirected)
            neighbors = self.get_neighbors(current, mode="all")
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Remove the starting node from results
        visited.discard(node_id)
        
        return sorted(list(visited))
    
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
    
    # ========================================
    # TRAVERSAL OPERATIONS
    # ========================================
    
    def trav_out(self) -> 'Graph':
        """Traverse from selected nodes to outward neighbors.
        
        From the currently selected nodes, moves to all nodes that can be
        reached by following outward edges (where selected nodes are sources).
        The current node selection is replaced with the traversed nodes.
        
        For directed graphs, follows edges in the forward direction.
        For undirected graphs, follows edges in all directions (equivalent to trav_both).
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.select_nodes_by_id(["A"]).trav_out().get_selected_nodes()
            ['B']
            >>> G.clear_selection().select_nodes_by_id(["A"]).trav_out().trav_out().get_selected_nodes()
            ['C']
        """
        self._graph.trav_out()
        return self
    
    def trav_in(self) -> 'Graph':
        """Traverse from selected nodes to inward neighbors.
        
        From the currently selected nodes, moves to all nodes that have
        outward edges leading to the selected nodes (where selected nodes are targets).
        The current node selection is replaced with the traversed nodes.
        
        For directed graphs, follows edges in the reverse direction.
        For undirected graphs, follows edges in all directions (equivalent to trav_both).
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "C")
            >>> G.select_nodes_by_id(["C"]).trav_in().get_selected_nodes()
            ['B']
            >>> G.clear_selection().select_nodes_by_id(["C"]).trav_in().trav_in().get_selected_nodes()
            ['A']
        """
        self._graph.trav_in()
        return self
    
    def trav_both(self) -> 'Graph':
        """Traverse from selected nodes to all neighbors.
        
        From the currently selected nodes, moves to all connected neighbors
        in both directions (both outward and inward edges).
        The current node selection is replaced with the traversed nodes.
        
        Works the same for directed and undirected graphs.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("C", "B")
            >>> G.select_nodes_by_id(["B"]).trav_both().get_selected_nodes()
            ['A', 'C']
        """
        self._graph.trav_both()
        return self
    
    def trav_out_edge(self) -> 'Graph':
        """Traverse from selected nodes to outbound edges.
        
        From the currently selected nodes, moves to all edges where the
        selected nodes are sources (outbound edges). The current node
        selection is cleared and replaced with an edge selection.
        
        For directed graphs, selects edges where from=selected_node.
        For undirected graphs, selects all edges incident to selected nodes.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("A", "C")
            >>> G.select_nodes_by_id(["A"]).trav_out_edge()
            >>> len(G.get_selected_edges())
            2
        """
        self._graph.trav_out_edge()
        return self
    
    def trav_in_edge(self) -> 'Graph':
        """Traverse from selected nodes to inbound edges.
        
        From the currently selected nodes, moves to all edges where the
        selected nodes are targets (inbound edges). The current node
        selection is cleared and replaced with an edge selection.
        
        For directed graphs, selects edges where to=selected_node.
        For undirected graphs, selects all edges incident to selected nodes.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "C")
            >>> G.add_edge("B", "C")
            >>> G.select_nodes_by_id(["C"]).trav_in_edge()
            >>> len(G.get_selected_edges())
            2
        """
        self._graph.trav_in_edge()
        return self
    
    def trav_reverse_edge(self, add_to_selection: bool = False) -> 'Graph':
        """Traverse to reverse edges.
        
        From the currently selected edges, finds the reverse edges.
        For each selected edge A→B, looks for the edge B→A.
        
        Args:
            add_to_selection: If True, adds reverse edges to the current selection.
                If False (default), replaces the selection with reverse edges.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = ec.Graph(directed=True)
            >>> G.add_edge("A", "B")
            >>> G.add_edge("B", "A")  # Reverse edge
            >>> G.select_edges_by_node_id("A", "B")
            >>> G.trav_reverse_edge()
            >>> edges = G.get_selected_edges()
            >>> # Now selects the B→A edge
        """
        self._graph.trav_reverse_edge(add_to_selection)
        return self
    
    def trav_out_until(
        self,
        conditions: callable,
        max_steps: int = 30,
        exclude_unmatched: bool = True,
        add_to_selection: bool = False
    ) -> 'Graph':
        """Traverse outward node-by-node until stopping conditions are met.
        
        From the currently selected nodes, moves along outward edges to connected
        nodes, continuing until reaching nodes that satisfy the condition or until
        max_steps is reached. The traversal proceeds in a breadth-first manner.
        
        Args:
            conditions: A callable that takes a node ID (str) and returns True if
                the stopping condition is met for that node. The callable can access
                node attributes via graph.get_node(node_id).attributes
            max_steps: Maximum number of traversal steps (node-to-node hops) before
                stopping. Default is 30.
            exclude_unmatched: If True (default), only nodes satisfying the conditions
                are kept in the final selection. If False, all reached nodes are kept.
            add_to_selection: If True, all traversed nodes are included in the final
                selection. If False (default), only the final nodes are selected.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_path_graph(10)
            >>> for i in range(1, 11):
            ...     G.set_node_attrs("value", {str(i): i})
            >>> G.select_nodes_by_id(["1"])
            >>> G.trav_out_until(lambda nid: G.get_node(nid).attributes.get("value") == 8)
            >>> G.get_selected_nodes()
            ['8']
            
            >>> # Example with multiple starting points
            >>> G.clear_selection().select_nodes_by_id(["1", "2"])
            >>> G.trav_out_until(
            ...     lambda nid: G.get_node(nid).attributes.get("value", 0) >= 5,
            ...     max_steps=10
            ... )
            >>> # Returns first nodes >= 5 from each starting point
        """
        selected = self.get_selected_nodes()
        if not selected:
            return self
        
        starting_nodes = set(selected)
        all_visited = set()  # For add_to_selection=True
        final_nodes = set()  # Nodes that satisfy conditions or reach max_steps
        
        # BFS from each starting node
        for start_node in starting_nodes:
            visited = {start_node}
            current_level = {start_node}
            
            if add_to_selection:
                all_visited.add(start_node)
            
            # Check if start node already satisfies condition
            if conditions(start_node):
                final_nodes.add(start_node)
                continue
            
            # Multi-hop BFS traversal
            for step in range(max_steps):
                next_level = set()
                
                for node_id in current_level:
                    # Get outward neighbors
                    try:
                        neighbors = self.get_neighbors(node_id, mode="out")
                    except (ValueError, KeyError):
                        neighbors = []
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            
                            if add_to_selection:
                                all_visited.add(neighbor)
                            
                            # Check if condition is satisfied
                            if conditions(neighbor):
                                final_nodes.add(neighbor)
                            else:
                                # Continue traversal from this node
                                next_level.add(neighbor)
                
                # If no more nodes to explore, stop
                if not next_level:
                    break
                
                current_level = next_level
            
            # If we reached max_steps without satisfying condition
            if not add_to_selection and current_level:
                # Add the final frontier nodes
                for node_id in current_level:
                    if not exclude_unmatched or conditions(node_id):
                        final_nodes.add(node_id)
        
        # Update selection
        self.clear_selection()
        
        if add_to_selection:
            # Include all visited nodes
            nodes_to_select = all_visited
            if exclude_unmatched:
                # But only keep those that satisfy conditions
                nodes_to_select = {n for n in all_visited if conditions(n)}
        else:
            # Only final nodes
            nodes_to_select = final_nodes
        
        if nodes_to_select:
            self.select_nodes_by_id(list(nodes_to_select))
        
        return self
    
    def trav_in_until(
        self,
        conditions: callable,
        max_steps: int = 30,
        exclude_unmatched: bool = True,
        add_to_selection: bool = False
    ) -> 'Graph':
        """Traverse inward node-by-node until stopping conditions are met.
        
        From the currently selected nodes, moves along inward edges to connected
        nodes, continuing until reaching nodes that satisfy the condition or until
        max_steps is reached. The traversal proceeds in a breadth-first manner.
        
        Args:
            conditions: A callable that takes a node ID (str) and returns True if
                the stopping condition is met for that node. The callable can access
                node attributes via graph.get_node(node_id).attributes
            max_steps: Maximum number of traversal steps (node-to-node hops) before
                stopping. Default is 30.
            exclude_unmatched: If True (default), only nodes satisfying the conditions
                are kept in the final selection. If False, all reached nodes are kept.
            add_to_selection: If True, all traversed nodes are included in the final
                selection. If False (default), only the final nodes are selected.
        
        Returns:
            Self for method chaining
            
        Example:
            >>> G = Graph(directed=True)
            >>> G.add_path_graph(10)
            >>> for i in range(1, 11):
            ...     G.set_node_attrs("value", {str(i): i})
            >>> G.select_nodes_by_id(["10"])
            >>> G.trav_in_until(lambda nid: G.get_node(nid).attributes.get("value") == 1)
            >>> G.get_selected_nodes()
            ['1']
            
            >>> # Example with condition checking multiple values
            >>> G.clear_selection().select_nodes_by_id(["6", "12"])
            >>> G.trav_in_until(
            ...     lambda nid: G.get_node(nid).attributes.get("value", 0) in [1, 2, 10]
            ... )
            >>> # Returns nodes with value in [1, 2, 10]
        """
        selected = self.get_selected_nodes()
        if not selected:
            return self
        
        starting_nodes = set(selected)
        all_visited = set()  # For add_to_selection=True
        final_nodes = set()  # Nodes that satisfy conditions or reach max_steps
        
        # BFS from each starting node
        for start_node in starting_nodes:
            visited = {start_node}
            current_level = {start_node}
            
            if add_to_selection:
                all_visited.add(start_node)
            
            # Check if start node already satisfies condition
            if conditions(start_node):
                final_nodes.add(start_node)
                continue
            
            # Multi-hop BFS traversal
            for step in range(max_steps):
                next_level = set()
                
                for node_id in current_level:
                    # Get inward neighbors
                    try:
                        neighbors = self.get_neighbors(node_id, mode="in")
                    except (ValueError, KeyError):
                        neighbors = []
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            
                            if add_to_selection:
                                all_visited.add(neighbor)
                            
                            # Check if condition is satisfied
                            if conditions(neighbor):
                                final_nodes.add(neighbor)
                            else:
                                # Continue traversal from this node
                                next_level.add(neighbor)
                
                # If no more nodes to explore, stop
                if not next_level:
                    break
                
                current_level = next_level
            
            # If we reached max_steps without satisfying condition
            if not add_to_selection and current_level:
                # Add the final frontier nodes
                for node_id in current_level:
                    if not exclude_unmatched or conditions(node_id):
                        final_nodes.add(node_id)
        
        # Update selection
        self.clear_selection()
        
        if add_to_selection:
            # Include all visited nodes
            nodes_to_select = all_visited
            if exclude_unmatched:
                # But only keep those that satisfy conditions
                nodes_to_select = {n for n in all_visited if conditions(n)}
        else:
            # Only final nodes
            nodes_to_select = final_nodes
        
        if nodes_to_select:
            self.select_nodes_by_id(list(nodes_to_select))
        
        return self
    
    def __repr__(self) -> str:
        return self._graph.__repr__()


# Re-export Node and Edge from Rust for direct access if needed
Node = rust.Node
Edge = rust.Edge


# Algorithm functions - wrap Rust functions to accept our Graph wrapper
def shortest_path(graph: Graph, source: Union[int, str], target: Union[int, str]) -> list[str] | None:
    """Find shortest path between two nodes.
    
    Args:
        graph: The graph to search
        source: Source node (integer ID or string label)
        target: Target node (integer ID or string label)
        
    Returns:
        List of node labels in the path, or None if no path exists
    """
    source_id = graph._resolve_node_id(source, auto_create=False)
    target_id = graph._resolve_node_id(target, auto_create=False)
    path_ids = rust.shortest_path(graph._graph, source_id, target_id)
    if path_ids is None:
        return None
    # Convert node IDs to labels
    return [graph.label_from_id(node_id) for node_id in path_ids]


def pagerank(
    graph: Graph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> dict[str, float]:
    """Calculate PageRank for all nodes.
    
    Returns:
        Dictionary mapping node labels to PageRank scores
    """
    result_ids = rust.pagerank(graph._graph, damping=damping, max_iter=max_iter, tol=tol)
    # Convert node IDs to labels
    return {graph.label_from_id(node_id): score for node_id, score in result_ids.items()}


def betweenness_centrality(graph: Graph) -> dict[str, float]:
    """Calculate betweenness centrality for all nodes.
    
    Returns:
        Dictionary mapping node labels to betweenness centrality scores
    """
    result_ids = rust.betweenness_centrality(graph._graph)
    # Convert node IDs to labels
    return {graph.label_from_id(node_id): score for node_id, score in result_ids.items()}


def detect_communities(graph: Graph) -> dict[str, int]:
    """Detect communities in the graph."""
    return rust.detect_communities(graph._graph)


def find_cycles(graph: Graph) -> list[list[str]]:
    """Find all cycles in the graph."""
    return rust.find_cycles(graph._graph)


# Community detection algorithms
def get_cmty_fast_greedy(graph: Graph) -> dict[str, int]:
    """
    Detect communities using Fast Greedy (Clauset-Newman-Moore) algorithm.
    
    This is a greedy modularity optimization algorithm that merges communities
    to maximize modularity. It's efficient for medium to large graphs and provides
    good quality communities.
    
    Algorithm:
        1. Start with each node in its own community
        2. Build a heap of merge candidates ordered by modularity gain (ΔQ)
        3. Repeatedly merge the pair with highest ΔQ until no improvement
        4. Return the partition with maximum modularity
    
    Args:
        graph: An undirected Graph instance
        
    Returns:
        dict[str, int]: Mapping from node ID to community ID (0-indexed)
        
    Raises:
        ValueError: If graph is directed (requires undirected graph)
        
    Complexity:
        Time: O(n log²n) on sparse graphs
        Space: O(n + m)
        
    Examples:
        >>> G = ec.Graph(directed=False)
        >>> # Two disconnected cliques
        >>> G.add_edges([('A','B'), ('B','C'), ('C','A')])
        >>> G.add_edges([('D','E'), ('E','F'), ('F','D')])
        >>> communities = G.get_cmty_fast_greedy()
        >>> # Should find 2 communities: {A,B,C} and {D,E,F}
        >>> len(set(communities.values()))
        2
        
        >>> # Karate club example
        >>> G = ec.Graph(directed=False)
        >>> G.add_edges([...])  # Add karate club edges
        >>> communities = G.get_cmty_fast_greedy()
        >>> # Modularity typically > 0.40 for karate club
        
    References:
        Clauset, A., Newman, M. E., & Moore, C. (2004).
        "Finding community structure in very large networks."
        Physical review E, 70(6), 066111.
        
    See Also:
        - get_cmty_louvain(): Faster for very large graphs
        - get_cmty_edge_btwns(): More accurate but slower
    """
    return graph._graph.get_cmty_fast_greedy()
