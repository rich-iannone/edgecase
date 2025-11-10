from typing import Optional, Callable, Any, Union
from edgecase import _core as rust
import polars as pl


class Graph:
    """A high-performance graph data structure with optional direction and weights.
    
    The `Graph` class provides a Python interface to a Rust-based graph implementation,
    offering both performance and ease of use. Nodes are identified by 64-bit unsigned
    integers (u64 IDs) and can optionally have string labels for human-readable access.
    
    Parameters
    ----------
    directed : bool, default False
        Whether the graph should be directed. If `True`, edges have direction from
        source to target. If `False`, edges are bidirectional.
    weighted : bool, default False
        Whether edges can have weights. If `True`, edge weights are stored and
        considered in algorithms. If `False`, all edges are treated equally.
    
    Examples
    --------
    Create an undirected, unweighted graph:
    
    ```python
    from edgecase import Graph
    
    graph = (
        Graph()
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_edge("Alice", "Bob")
    )
    ```
    
    Create a directed, weighted graph for modeling networks:
    
    ```python
    graph = (
        Graph(directed=True, weighted=True)
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_edge("Alice", "Bob", weight=0.8)
    )
    ```
    
    See Also
    --------
    add_node : Add nodes to the graph
    add_edge : Add edges between nodes
    """
    
    def __init__(self, directed: bool = False, weighted: bool = False):
        self._graph = rust.Graph(directed=directed, weighted=weighted)
    
    def add_node(self, node_id: Optional[Union[int, str]] = None, label: Optional[str] = None, 
                 from_node: Optional[Union[int, str]] = None, to_node: Optional[Union[int, str]] = None,
                 **attributes) -> "Graph":
        """Add a node to the graph with optional connections and attributes.
        
        This method provides flexible node creation, allowing you to add isolated nodes,
        nodes with labels, or nodes that are immediately connected to existing nodes.
        Node IDs are 64-bit unsigned integers managed internally, while labels provide
        human-readable identifiers.
        
        Parameters
        ----------
        node_id : int | str | None, default None
            The node identifier, which can be:
            - `None`: Auto-generate a unique u64 ID (recommended)
            - `int`: Use this specific integer as the node ID
            - `str`: Treat as a label and auto-generate an ID
        label : str | None, default None
            Optional string label for human-readable node access. Labels should be
            unique if you plan to use them for node lookup in operations like
            `add_edge()`.
        from_node : int | str | None, default None
            Optional node (ID or label) to connect FROM the new node. Creates a
            directed edge: `new_node -> from_node`.
        to_node : int | str | None, default None
            Optional node (ID or label) to connect TO the new node. Creates a
            directed edge: `to_node -> new_node`.
        **attributes : dict
            Arbitrary key-value pairs to store as node attributes.
        
        Returns
        -------
        Graph
            Returns self to enable method chaining.
        
        Examples
        --------
        Add an isolated node with auto-generated ID:
        
        ```python
        graph = Graph()
        graph.add_node(label="Node1")
        node_id = graph.id_from_label("Node1")
        print(f"Created node with ID: {node_id}")
        ```
        
        Add nodes with labels for easier reference:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
        )
        ```
        
        Create a node immediately connected to existing nodes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob", from_node="Alice")     # Edge: Alice -> Bob
            .add_node(label="Charlie", to_node="Alice")   # Edge: Charlie -> Alice
        )
        ```
        
        Add nodes with custom attributes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice", age=30, role="engineer")
            .add_node(label="Bob", age=25, role="designer")
        )
        ```
        
        See Also
        --------
        add_edge : Add edges between existing nodes
        id_from_label : Look up node ID by label
        label_from_id : Look up node label by ID
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
        
        return self
    
    def add_edge(self, source: Union[int, str, rust.Edge], target: Optional[Union[int, str]] = None, 
                 weight: Optional[float] = None, edge_id: Optional[int] = None, **attributes) -> "Graph":
        """Add an edge between two existing nodes in the graph.
        
        Edges connect existing nodes and can optionally carry weights and attributes.
        Both source and target nodes must already exist in the graph. You can reference
        nodes by their integer IDs or string labels (if labels are unique).
        
        Parameters
        ----------
        source : int | str | Edge
            The source node identifier (ID or label), or a pre-constructed Edge object.
            If providing an Edge object, other parameters are ignored.
        target : int | str | None, default None
            The target node identifier (ID or label). Required unless `source` is an
            Edge object.
        weight : float | None, default None
            Optional edge weight for weighted graphs. Ignored if the graph is unweighted.
        edge_id : int | None, default None
            Optional edge ID. If `None`, an ID is auto-assigned.
        **attributes : dict
            Arbitrary key-value pairs to store as edge attributes.
        
        Returns
        -------
        Graph
            Returns self to enable method chaining.
        
        Raises
        ------
        ValueError
            If source or target node doesn't exist in the graph, or if a label
            cannot be resolved to a node ID.
        
        Examples
        --------
        Add edges using node IDs:
        
        ```python
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        alice_id = graph.id_from_label("Alice")
        bob_id = graph.id_from_label("Bob")
        graph.add_edge(alice_id, bob_id)
        ```
        
        Add edges using labels (only if labels are unique):
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_edge("Alice", "Bob")
        )
        ```
        
        Mix IDs and labels:
        
        ```python
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        alice_id = graph.id_from_label("Alice")
        graph.add_edge(alice_id, "Bob")
        ```
        
        Add weighted edges:
        
        ```python
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B", weight=0.75)
        )
        ```
        
        Add edges with custom attributes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_edge("Alice", "Bob", relationship="friend", since=2020)
        )
        ```
        
        See Also
        --------
        add_node : Add nodes to the graph
        get_edge : Retrieve edge information
        """
        # Handle direct Edge objects
        if isinstance(source, rust.Edge):
            self._graph.add_edge(source)
            return self
        
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
        self._graph.add_edge(edge)
        return self
    
    def _resolve_node_id(self, node: Union[int, str], auto_create: bool = False) -> int:
        """Resolve a node identifier (int or str) to a u64 ID.
        
        Internal helper method that converts either integer IDs or string labels
        to the underlying u64 node ID used by the Rust core.
        
        Parameters
        ---
        node : int | str
            Either an integer ID or a string label to resolve.
        auto_create : bool, default False
            If `True` and `node` is a string not in the graph, create a new node
            with that label. If `False`, raise an error for unknown labels.
            
        Returns
        ---
        int
            The u64 node ID corresponding to the provided identifier.
            
        Raises
        ---
        ValueError
            If `node` is a label not found in the graph and `auto_create` is `False`.
        TypeError
            If `node` is neither an integer nor a string.
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
        
        Translates a human-readable string label to the internal u64 node ID.
        Useful when you need the ID for operations that require integer identifiers.
        
        Parameters
        ----------
        label : str
            The string label to look up.
        
        Returns
        -------
        int | None
            The u64 node ID if found, or `None` if the label doesn't exist.
        
        Examples
        --------
        Look up node IDs by label:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
        )
        
        alice_id = graph.id_from_label("Alice")
        print(alice_id)  # 1
        
        unknown_id = graph.id_from_label("Unknown")
        print(unknown_id)  # None
        ```
        
        See Also
        --------
        label_from_id : Get label from node ID
        has_node : Check if node exists
        """
        try:
            return self._graph.id_from_label(label)
        except (ValueError, Exception):
            return None
    
    def label_from_id(self, node_id: int) -> Optional[str]:
        """Get the label for a node by its ID.
        
        Translates an internal u64 node ID to its human-readable string label.
        Returns `None` if the node doesn't exist or has no label assigned.
        
        Parameters
        ----------
        node_id : int
            The u64 node ID to look up.
        
        Returns
        -------
        str | None
            The string label if it exists, or `None` if the node has no label
            or doesn't exist in the graph.
        
        Examples
        --------
        Retrieve labels for node IDs:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
        )
        
        label = graph.label_from_id(1)
        print(label)  # "Alice"
        
        no_label = graph.label_from_id(999)
        print(no_label)  # None
        ```
        
        See Also
        --------
        id_from_label : Get node ID from label
        get_node_ids : Get all node IDs in the graph
        """
        try:
            return self._graph.label_from_id(node_id)
        except (ValueError, Exception):
            return None
    
    def get_node_ids(self) -> list[int]:
        """Get all node IDs in the graph.
        
        Returns the complete list of u64 node IDs currently in the graph.
        Useful for iterating over all nodes or checking graph size.
        
        Returns
        -------
        list[int]
            List of u64 node IDs in the graph.
        
        Examples
        --------
        Get all node IDs:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        
        ids = graph.get_node_ids()
        print(ids)  # [1, 2, 3]
        ```
        
        See Also
        --------
        nodes : Alias for this method
        num_nodes : Get the count of nodes
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
        
        A simple graph contains no self-loops (edges from a node to itself) and
        no multi-edges (multiple edges between the same pair of nodes). This is
        a fundamental graph property used in many theoretical results.
        
        Returns
        -------
        bool
            `True` if the graph is simple, `False` if it has self-loops or multi-edges.
        
        Examples
        --------
        Check if a graph is simple:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
        )
        
        print(graph.is_graph_simple())  # True
        
        # Add a self-loop
        graph.add_edge("A", "A")
        print(graph.is_graph_simple())  # False
        ```
        
        Multi-edges also make a graph non-simple:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B", edge_id=1)
            .add_edge("A", "B", edge_id=2)  # Duplicate edge
        )
        
        print(graph.is_graph_simple())  # False
        ```
        
        See Also
        --------
        count_selfloops : Count self-loops in the graph
        count_multi_edges : Count parallel edges
        """
        return self._graph.is_graph_simple()
    
    def has_cycles(self) -> bool:
        """Check if the graph contains any cycles.
        
        Uses depth-first search to detect cycles. For directed graphs, looks for
        back edges. For undirected graphs, checks for cycles excluding trivial
        two-node back-and-forth paths.
        
        Returns
        -------
        bool
            `True` if the graph contains at least one cycle, `False` if acyclic.
        
        Examples
        --------
        Check for cycles in a directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        
        print(graph.has_cycles())  # False
        
        # Add edge to create a cycle
        graph.add_edge("C", "A")
        print(graph.has_cycles())  # True
        ```
        
        See Also
        --------
        is_dag : Check if directed graph is acyclic
        get_girth : Get length of shortest cycle
        """
        return self._graph.has_cycles()
    
    def has_node(self, node: Union[int, str]) -> bool:
        """Check if a node exists in the graph.
        
        Verifies whether a node (specified by ID or label) exists in the graph.
        Returns `False` for invalid inputs or missing nodes.
        
        Parameters
        ----------
        node : int | str
            Node ID (int) or label (str) to check.
        
        Returns
        -------
        bool
            `True` if the node exists, `False` otherwise.
        
        Examples
        --------
        Check node existence:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
        )
        
        print(graph.has_node("Alice"))   # True
        print(graph.has_node(1))         # True  
        print(graph.has_node("Charlie")) # False
        print(graph.has_node(999))       # False
        ```
        
        See Also
        --------
        has_edge : Check if an edge exists
        num_nodes : Get total node count
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
            >>> for node in ["A", "B", "C"]:
            ...     G.add_node(node)
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
        
        For undirected graphs, the order of source and target doesn't matter.
        For directed graphs, checks specifically for an edge from source to target.
        
        Parameters
        ----------
        source : int | str
            Source node ID or label.
        target : int | str
            Target node ID or label.
        
        Returns
        -------
        bool
            `True` if at least one edge exists between the nodes, `False` otherwise.
        
        Examples
        --------
        Check edge existence in directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
        )
        
        print(graph.has_edge("A", "B"))  # True
        print(graph.has_edge("B", "A"))  # False (directed)
        ```
        
        For undirected graphs, order doesn't matter:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
        )
        
        print(graph.has_edge("A", "B"))  # True
        print(graph.has_edge("B", "A"))  # True (undirected)
        ```
        
        See Also
        --------
        add_edge : Add edges to the graph
        delete_edge : Remove edges from the graph
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
        
        Remove edges by specifying either the source and target nodes, or the
        edge ID directly. For multigraphs with source/target, only the first
        matching edge is deleted. Returns self for method chaining.
        
        Parameters
        ----------
        source : int | str | None, default None
            Source node ID or label. Required if `id` is not provided.
        target : int | str | None, default None
            Target node ID or label. Required if `id` is not provided.
        id : int | None, default None
            Edge ID to delete. If provided, `source` and `target` are ignored.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Raises
        ------
        KeyError
            If no edge exists matching the criteria.
        ValueError
            If neither (`source`, `target`) nor `id` is provided.
        
        Examples
        --------
        Delete edge by source and target:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .delete_edge(source="A", target="B")
        )
        
        print(graph.has_edge("A", "B"))  # False
        ```
        
        Delete edge by ID:
        
        ```python
        graph = Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")
        edges = graph.get_edges()
        edge_id = edges[0].id
        
        graph.delete_edge(id=edge_id)
        print(graph.has_edge("A", "B"))  # False
        ```
        
        See Also
        --------
        add_edge : Add edges to the graph
        has_edge : Check if edge exists
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
        
        Returns the node IDs of all nodes directly connected to the specified node.
        For directed graphs, returns only outgoing neighbors (nodes this node points to).
        
        Parameters
        ----------
        node : int | str
            Node ID or label to get neighbors for.
        
        Returns
        -------
        list[int]
            List of neighbor node IDs.
        
        Examples
        --------
        Get neighbors in an undirected graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("A", "C")
        )
        
        neighbors = graph.neighbors("A")
        print(len(neighbors))  # 2
        ```
        
        In directed graphs, only outgoing edges count:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")  # A -> B
            .add_edge("C", "A")  # C -> A
        )
        
        # A points to B, so B is a neighbor
        print(len(graph.neighbors("A")))  # 1
        ```
        
        See Also
        --------
        degree : Get node degree
        has_edge : Check if edge exists
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
        
        For directed graphs, returns total degree (in-degree + out-degree).
        For undirected graphs, returns the number of neighbors.
        
        Parameters
        ----------
        node : int | str
            Node ID or label.
        
        Returns
        -------
        int
            Degree of the node.
        
        Examples
        --------
        Get node degree in undirected graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("A", "C")
        )
        
        print(graph.degree("A"))  # 2
        print(graph.degree("B"))  # 1
        ```
        
        In directed graphs, counts both incoming and outgoing:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")  # A -> B (out for A, in for B)
            .add_edge("C", "A")  # C -> A (in for A)
        )
        
        print(graph.degree("A"))  # 2 (1 in + 1 out)
        ```
        
        See Also
        --------
        in_degree : Get in-degree for directed graphs
        out_degree : Get out-degree for directed graphs
        average_degree : Get average degree across all nodes
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
        
        In-degree is the number of edges pointing into the node. Only meaningful
        for directed graphs.
        
        Parameters
        ----------
        node : int | str
            Node ID or label.
        
        Returns
        -------
        int
            Number of incoming edges.
        
        Raises
        ------
        ValueError
            If the graph is undirected.
        
        Examples
        --------
        Get in-degrees in a directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")  # B has 1 incoming
            .add_edge("C", "B")  # B has 2 incoming
        )
        
        print(graph.in_degree("B"))  # 2
        print(graph.in_degree("A"))  # 0
        ```
        
        See Also
        --------
        out_degree : Get out-degree for directed graphs
        degree : Get total degree
        """
        if not self._graph.is_directed:
            raise ValueError("in_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
        # edges() returns (edge_id, source, target) tuples
        return sum(1 for edge in self._graph.edges() if edge[2] == node_id)
    
    def out_degree(self, node: Union[int, str]) -> int:
        """Get the out-degree of a node (directed graphs only).
        
        Out-degree is the number of edges pointing out from the node. Only meaningful
        for directed graphs.
        
        Parameters
        ----------
        node : int | str
            Node ID or label.
        
        Returns
        -------
        int
            Number of outgoing edges.
        
        Raises
        ------
        ValueError
            If the graph is undirected.
        
        Examples
        --------
        Get out-degrees in a directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")  # A has 1 outgoing
            .add_edge("A", "C")  # A has 2 outgoing
        )
        
        print(graph.out_degree("A"))  # 2
        print(graph.out_degree("B"))  # 0
        ```
        
        See Also
        --------
        in_degree : Get in-degree for directed graphs
        degree : Get total degree
        """
        if not self._graph.is_directed:
            raise ValueError("out_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
        return len(self._graph.neighbors(node_id))
    
    # ========== Structural Properties ==========
    
    def density(self) -> float:
        """Calculate graph density (ratio of actual to possible edges).
        
        Density measures how complete a graph is. A complete graph has density 1.0,
        while a graph with no edges has density 0.0.
        
        For directed graphs: density = E / (N × (N-1))
        For undirected graphs: density = 2E / (N × (N-1))
        
        Returns
        -------
        float
            Graph density between 0.0 and 1.0.
        
        Examples
        --------
        Calculate density of a sparse graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        
        # 2 edges, 3 nodes: 2 / (3 × 2 / 2) = 0.667
        print(f"{graph.density():.3f}")  # 0.667
        ```
        
        Compare sparse vs. dense graphs:
        
        ```python
        # Complete graph (all possible edges)
        complete = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("A", "C")
        )
        
        print(f"{complete.density():.3f}")  # 1.000
        ```
        
        See Also
        --------
        num_nodes : Get number of nodes
        num_edges : Get number of edges
        """
        return self._graph.density()
    
    def is_connected(self) -> bool:
        """Check if graph is connected.
        
        A graph is connected if there is a path between every pair of nodes.
        For directed graphs, checks weak connectivity (treating edges as undirected).
        
        Returns
        -------
        bool
            `True` if all nodes are reachable from any node, `False` if the graph
            has disconnected components.
        
        Examples
        --------
        Check connectivity of a simple path:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        
        print(graph.is_connected())  # True
        
        # Adding isolated nodes breaks connectivity
        graph.add_node(label="D")  # Isolated node
        print(graph.is_connected())  # False
        ```
        
        See Also
        --------
        count_weakly_connected_components : Count connected components
        count_strongly_connected_components : For directed graphs
        """
        return self._graph.is_connected()
    
    def is_dag(self) -> bool:
        """Check if directed graph is a DAG (Directed Acyclic Graph).
        
        A DAG is a directed graph with no cycles - you cannot follow directed edges
        and return to a starting node. DAGs are fundamental for dependency resolution,
        scheduling, and topological sorting.
        
        Returns
        -------
        bool
            `True` if the graph is acyclic, `False` if it contains cycles.
        
        Raises
        ------
        ValueError
            If called on an undirected graph.
        
        Examples
        --------
        Check if a directed graph is acyclic:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        
        print(graph.is_dag())  # True
        
        # Adding a cycle makes it not a DAG
        graph.add_edge("C", "A")  # Creates cycle A->B->C->A
        print(graph.is_dag())  # False
        ```
        
        See Also
        --------
        has_cycles : Check for cycles in any graph
        """
        return self._graph.is_dag()
    
    def is_simple(self) -> bool:
        """Check if graph is simple (no self-loops or multi-edges).
        
        A simple graph has no self-loops (edges from a node to itself) and no
        parallel edges (multiple edges between the same pair of nodes). This is
        a fundamental property in graph theory.
        
        Returns
        -------
        bool
            `True` if the graph has no self-loops and no parallel edges.
        
        Examples
        --------
        Check if a graph is simple:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
        )
        
        print(graph.is_simple())  # True
        
        # Self-loops make a graph non-simple
        graph.add_edge("A", "A")  # Self-loop
        print(graph.is_simple())  # False
        ```
        
        See Also
        --------
        count_selfloops : Count self-loops
        count_multi_edges : Count parallel edges
        is_graph_simple : Alias for this method
        """
        return self._graph.is_simple()
    
    def is_bipartite(self) -> bool:
        """Check if graph is bipartite (2-colorable).
        
        A graph is bipartite if its nodes can be divided into two independent sets
        such that every edge connects nodes from different sets. Equivalently, the
        graph can be colored with 2 colors such that no adjacent nodes share a color.
        
        All trees and forests are bipartite. Graphs with odd-length cycles are not.
        
        Returns
        -------
        bool
            `True` if the graph is bipartite, `False` otherwise.
        
        Examples
        --------
        A simple path is bipartite:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        
        print(graph.is_bipartite())  # True
        ```
        
        Triangles are not bipartite (odd cycle):
        
        ```python
        triangle = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Creates odd-length cycle
        )
        
        print(triangle.is_bipartite())  # False
        ```
        
        See Also
        --------
        get_girth : Get length of shortest cycle
        """
        return self._graph.is_bipartite()
    
    def count_weakly_connected_components(self) -> int:
        """Count weakly connected components in the graph.
        
        For undirected graphs, counts regular connected components. For directed graphs,
        treats all edges as undirected (ignoring direction) when finding components.
        
        A weakly connected component is a maximal set of nodes where every node is
        reachable from every other node when edge directions are ignored.
        
        Returns
        -------
        int
            Number of weakly connected components.
        
        Examples
        --------
        Count components in a disconnected graph:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge("A", "B")
            .add_edge("X", "Y")
        )
        
        print(graph.count_weakly_connected_components())  # 2
        ```
        
        See Also
        --------
        is_connected : Check if graph is connected
        count_strongly_connected_components : For directed graphs
        """
        return self._graph.count_weakly_connected_components()
    
    def count_strongly_connected_components(self) -> int:
        """Count strongly connected components (directed graphs only).
        
        A strongly connected component is a maximal set of nodes where every node
        is reachable from every other node following directed edges. Uses Kosaraju's
        algorithm for efficient computation.
        
        Returns
        -------
        int
            Number of strongly connected components.
        
        Raises
        ------
        ValueError
            If called on an undirected graph.
        
        Examples
        --------
        Count strongly connected components:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Cycle: all in same component
            .add_edge("X", "Y")  # Separate component
        )
        
        print(graph.count_strongly_connected_components())  # 2
        ```
        
        See Also
        --------
        count_weakly_connected_components : For weak connectivity
        is_connected : Check overall connectivity
        """
        return self._graph.count_strongly_connected_components()
    
    def degree_distribution(self) -> dict[int, int]:
        """Get degree distribution as a mapping from degree to count.
        
        Returns a histogram of node degrees, showing how many nodes have each
        degree value. Useful for analyzing network topology and identifying
        hub nodes or scale-free properties.
        
        Returns
        -------
        dict[int, int]
            Dictionary mapping degree values to the number of nodes with that degree.
        
        Examples
        --------
        Get degree distribution:
        
        ```python
        graph = (
            Graph()
            .add_node(label="Hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("Hub", "A")
            .add_edge("Hub", "B")
            .add_edge("Hub", "C")
            .add_edge("Hub", "D")
        )
        
        dist = graph.degree_distribution()
        print(dist)  # {1: 4, 4: 1} - 4 nodes with degree 1, 1 node with degree 4
        ```
        
        See Also
        --------
        degree : Get degree of a single node
        average_degree : Get mean degree
        get_degree_histogram : Alias for this method
        """
        return self._graph.degree_distribution()
    
    def count_selfloops(self) -> int:
        """Count number of self-loops in the graph.
        
        A self-loop is an edge where the source and target are the same node.
        Self-loops are common in certain applications like state machines or
        reflexive relationships.
        
        Returns
        -------
        int
            Number of edges where source equals target.
        
        Examples
        --------
        Count self-loops:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
            .add_edge("A", "A")  # Self-loop
            .add_edge("B", "B")  # Self-loop
        )
        
        print(graph.count_selfloops())  # 2
        ```
        
        See Also
        --------
        is_simple : Check if graph has no self-loops or multi-edges
        count_loop_edges : Alias for this method
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
        Multi-edges are useful for modeling scenarios like multiple flights
        between cities or repeated transactions.
        
        Returns
        -------
        int
            Number of duplicate edges (beyond first edge per pair).
        
        Examples
        --------
        Count multi-edges in a directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B", id=1)
            .add_edge("A", "B", id=2)  # Multi-edge
            .add_edge("A", "B", id=3)  # Another multi-edge
            .add_edge("B", "A", id=4)  # Different direction
        )
        
        print(graph.count_multi_edges())  # 2
        ```
        
        See Also
        --------
        is_simple : Check if graph has no self-loops or multi-edges
        count_selfloops : Count self-loop edges
        """
        return self._graph.count_multi_edges()
    
    def count_mutual_node_pairs(self) -> int:
        """Count mutual node pairs (bidirectional edges) in directed graphs.
        
        Returns the number of node pairs (a,b) where both edges a->b and b->a exist.
        Only works for directed graphs. Self-loops are excluded. This metric is
        useful for analyzing reciprocity in directed networks like social graphs.
        
        Returns
        -------
        int
            Number of node pairs with edges in both directions.
            
        Raises
        ------
        ValueError
            If graph is undirected.
        
        Examples
        --------
        Count mutual relationships:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "A")  # Mutual with A
            .add_edge("A", "C")  # One-way only
        )
        
        print(graph.count_mutual_node_pairs())  # 1
        ```
        
        See Also
        --------
        count_asymmetric_node_pairs : Count one-way relationships
        reciprocity : Calculate reciprocity ratio
        """
        return self._graph.count_mutual_node_pairs()
    
    def count_asymmetric_node_pairs(self) -> int:
        """Count asymmetric node pairs (one-way edges) in directed graphs.
        
        Returns the number of node pairs (a,b) where edge a->b exists but not b->a.
        Only works for directed graphs. Self-loops are excluded. This metric helps
        analyze directional flow or hierarchy in directed networks.
        
        Returns
        -------
        int
            Number of node pairs with edges in only one direction.
            
        Raises
        ------
        ValueError
            If graph is undirected.
        
        Examples
        --------
        Count one-way relationships:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "A")  # Mutual, not counted
            .add_edge("A", "C")  # Asymmetric
        )
        
        print(graph.count_asymmetric_node_pairs())  # 1
        ```
        
        See Also
        --------
        count_mutual_node_pairs : Count bidirectional relationships
        reciprocity : Calculate reciprocity ratio
        """
        return self._graph.count_asymmetric_node_pairs()
    
    def average_degree(self) -> float:
        """Calculate average degree across all nodes.
        
        For directed graphs, this returns the average of (in-degree + out-degree).
        For undirected graphs, each edge contributes to the degree of both endpoints.
        
        Returns
        -------
        float
            Mean degree value across all nodes.
        
        Examples
        --------
        Calculate average degree:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )
        
        print(graph.average_degree())  # 2.0 (cycle graph)
        ```
        
        See Also
        --------
        degree_distribution : Get full degree distribution
        degree : Get degree of a specific node
        """
        return self._graph.average_degree()
    
    def get_agg_degree_in(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """Get aggregate value from indegree of nodes.
        
        Computes a single aggregate statistic from the indegree values of all nodes
        (or a filtered subset) in a directed graph. Useful for summarizing the
        indegree distribution or analyzing filtered node subsets.
        
        Parameters
        ---
        agg : str
            Aggregation function to apply. Must be one of:
            - `'sum'` : Sum of all indegrees
            - `'min'` : Minimum indegree
            - `'max'` : Maximum indegree
            - `'mean'` : Mean indegree
            - `'median'` : Median indegree
        conditions : callable, optional
            Filter function that takes a node object and returns `True` to include
            the node in the aggregation. If `None`, all nodes are included.
        
        Returns
        ---
        float
            The computed aggregate indegree value. Returns `0.0` if no nodes match
            the conditions.
            
        Raises
        ---
        ValueError
            If the graph is undirected or if `agg` is not a valid aggregation function.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        # Create directed graph
        graph = ec.Graph(directed=True)
        graph.add_gnm_graph(20, 35, seed=23)
        
        # Mean indegree across all nodes
        mean_in = graph.get_agg_degree_in('mean')
        print(mean_in)  # 1.75
        
        # Median indegree
        median_in = graph.get_agg_degree_in('median')
        print(median_in)  # 1.5
        
        # Filtered aggregation: mean indegree of high-degree nodes
        high_degree_mean = graph.get_agg_degree_in(
            'mean',
            conditions=lambda n: graph.degree(n) > 3
        )
        ```
        
        See Also
        --------
        get_agg_degree_out : Aggregate outdegree values
        get_agg_degree_total : Aggregate total degree values
        in_degree : Get indegree of a single node
        degree_distribution : Get full degree distribution
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
        (or a filtered subset) in a directed graph. Useful for summarizing the
        outdegree distribution or analyzing filtered node subsets.
        
        Parameters
        ---
        agg : str
            Aggregation function to apply. Must be one of:
            - `'sum'` : Sum of all outdegrees
            - `'min'` : Minimum outdegree
            - `'max'` : Maximum outdegree
            - `'mean'` : Mean outdegree
            - `'median'` : Median outdegree
        conditions : callable, optional
            Filter function that takes a node object and returns `True` to include
            the node in the aggregation. If `None`, all nodes are included.
        
        Returns
        ---
        float
            The computed aggregate outdegree value. Returns `0.0` if no nodes match
            the conditions.
            
        Raises
        ---
        ValueError
            If the graph is undirected or if `agg` is not a valid aggregation function.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        # Create directed graph
        graph = ec.Graph(directed=True)
        graph.add_gnm_graph(20, 35, seed=23)
        
        # Mean outdegree across all nodes
        mean_out = graph.get_agg_degree_out('mean')
        print(mean_out)  # 1.75
        
        # Median outdegree
        median_out = graph.get_agg_degree_out('median')
        print(median_out)  # 1.0
        
        # Filtered aggregation: max outdegree of nodes with attributes
        max_out = graph.get_agg_degree_out(
            'max',
            conditions=lambda n: 'type' in n.attributes
        )
        ```
        
        See Also
        --------
        get_agg_degree_in : Aggregate indegree values
        get_agg_degree_total : Aggregate total degree values
        out_degree : Get outdegree of a single node
        degree_distribution : Get full degree distribution
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
        (or a filtered subset). Works with both directed and undirected graphs. For
        directed graphs, total degree = indegree + outdegree.
        
        Parameters
        ---
        agg : str
            Aggregation function to apply. Must be one of:
            - `'sum'` : Sum of all total degrees
            - `'min'` : Minimum total degree
            - `'max'` : Maximum total degree
            - `'mean'` : Mean total degree
            - `'median'` : Median total degree
        conditions : callable, optional
            Filter function that takes a node object and returns `True` to include
            the node in the aggregation. If `None`, all nodes are included.
        
        Returns
        ---
        float
            The computed aggregate total degree value. Returns `0.0` if no nodes
            match the conditions.
            
        Raises
        ---
        ValueError
            If `agg` is not a valid aggregation function.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        # Create directed graph
        graph = ec.Graph(directed=True)
        graph.add_gnm_graph(20, 35, seed=23)
        
        # Mean total degree across all nodes
        mean_total = graph.get_agg_degree_total('mean')
        print(mean_total)  # 3.5
        
        # Median total degree
        median_total = graph.get_agg_degree_total('median')
        print(median_total)  # 3.0
        
        # Works with undirected graphs too
        graph_undirected = ec.Graph(directed=False)
        graph_undirected.add_cycle_graph(10)
        mean_deg = graph_undirected.get_agg_degree_total('mean')
        print(mean_deg)  # 2.0
        ```
        
        See Also
        --------
        get_agg_degree_in : Aggregate indegree values (directed only)
        get_agg_degree_out : Aggregate outdegree values (directed only)
        degree : Get total degree of a single node
        average_degree : Get mean degree across all nodes
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
        
        Computes average shortest path length between all reachable node pairs.
        Also known as average path length or characteristic path length. Returns
        None if graph is disconnected (infinite distances exist).
        
        Returns
        -------
        float | None
            Average shortest path length, or None if graph is disconnected.
        
        Examples
        --------
        Calculate mean distance in a path graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )
        
        mean_dist = graph.get_mean_distance()
        # Paths: A-B(1), A-C(2), A-D(3), B-C(1), B-D(2), C-D(1)
        # Average: (1+2+3+1+2+1)/6 = 1.67
        print(mean_dist)
        ```
        
        Notes
        -----
        Important network metric:
        - Small-world networks: Low mean distance despite high clustering
        - Scale-free networks: Mean distance grows logarithmically with size
        - Used in epidemiology: Average infection spread time
        
        See Also
        --------
        get_graph_diameter : Maximum shortest path length
        get_graph_radius : Minimum eccentricity
        """
        return self._graph.get_mean_distance()
    
    def get_min_eccentricity(self) -> int | None:
        """Get minimum eccentricity (graph radius).
        
        Alias for `get_graph_radius()`. The minimum eccentricity is the smallest
        maximum distance from any node to all other nodes. Returns None if graph
        is disconnected.
        
        Returns
        -------
        int | None
            Minimum eccentricity (radius), or None if disconnected.
        
        Examples
        --------
        Find the radius of a star graph:
        
        ```python
        graph = Graph().add_star_graph(5, center_label="hub")
        
        radius = graph.get_min_eccentricity()
        print(radius)  # 1 (hub reaches all nodes in 1 step)
        ```
        
        See Also
        --------
        get_graph_radius : Equivalent method
        get_max_eccentricity : Maximum eccentricity (diameter)
        get_eccentricity : Eccentricity of each node
        """
        return self.get_graph_radius()
    
    def get_max_eccentricity(self) -> int | None:
        """Get maximum eccentricity (graph diameter).
        
        Alias for `get_graph_diameter()`. The maximum eccentricity is the longest
        shortest path between any two nodes. Returns None if graph is disconnected.
        
        Returns
        -------
        int | None
            Maximum eccentricity (diameter), or None if disconnected.
        
        Examples
        --------
        Find the diameter of a path graph:
        
        ```python
        graph = Graph().add_path_graph(5)
        
        diameter = graph.get_max_eccentricity()
        print(diameter)  # 4 (end-to-end distance)
        ```
        
        See Also
        --------
        get_graph_diameter : Equivalent method
        get_min_eccentricity : Minimum eccentricity (radius)
        get_eccentricity : Eccentricity of each node
        """
        return self.get_graph_diameter()
    
    def get_articulation_points(self) -> list[str]:
        """Find articulation points (cut vertices) in the graph.
        
        An articulation point is a node whose removal disconnects the graph or
        increases the number of connected components. Uses Tarjan's DFS-based
        algorithm. Only defined for undirected graphs.
        
        Returns
        -------
        list[str]
            List of node IDs that are articulation points.
        
        Raises
        ------
        ValueError
            If graph is directed.
        
        Examples
        --------
        Find articulation points in a path graph:
        
        ```python
        graph = Graph().add_path_graph(5)
        
        # In a path: A-B-C-D-E
        # Middle nodes B, C, D are articulation points
        cut_vertices = graph.get_articulation_points()
        print(len(cut_vertices))  # 3
        ```
        
        Notes
        -----
        Applications:
        - Network vulnerability analysis: Critical infrastructure nodes
        - Social networks: Key influencers whose removal fragments communities
        - Communication networks: Single points of failure
        
        See Also
        --------
        get_min_cut_between : Edge-based connectivity measure
        is_connected : Check if graph is connected
        """
        return self._graph.get_articulation_points()
    
    def get_periphery(self) -> list[str]:
        """Get periphery nodes (nodes with maximum eccentricity).
        
        Periphery nodes are farthest from the graph's center. These nodes have
        the maximum eccentricity (longest shortest path to any other node).
        Returns empty list if graph is disconnected.
        
        Returns
        -------
        list[str]
            List of node IDs in the periphery, empty if disconnected.
        
        Examples
        --------
        Find periphery nodes in a path graph:
        
        ```python
        graph = Graph().add_path_graph(5)
        
        # In path A-B-C-D-E, ends A and E are periphery
        periphery = graph.get_periphery()
        print(sorted(periphery))  # ['0', '4']
        ```
        
        Notes
        -----
        Characteristics:
        - Opposite of center: Peripheral vs central nodes
        - In trees: Always leaf nodes
        - Multiple nodes can have maximum eccentricity
        
        See Also
        --------
        get_max_eccentricity : Maximum eccentricity value (diameter)
        get_eccentricity : Eccentricity of all nodes
        """
        return self._graph.get_periphery()
    
    def get_graph_diameter(self) -> int | None:
        """Get graph diameter (maximum eccentricity).
        
        The diameter is the longest shortest path between any two nodes.
        Returns None if the graph is disconnected. This metric is useful
        for understanding the "width" of a network.
        
        Returns
        -------
        int | None
            Maximum distance between any two nodes, or None if disconnected.
        
        Examples
        --------
        Calculate diameter of a path:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )
        
        print(graph.get_graph_diameter())  # 3 (A to D)
        ```
        
        See Also
        --------
        get_graph_radius : Get minimum eccentricity
        eccentricity : Get eccentricity of a specific node
        """
        return self._graph.get_graph_diameter()
    
    def get_graph_radius(self) -> int | None:
        """Get graph radius (minimum eccentricity).
        
        The radius is the minimum of all node eccentricities. Nodes with
        eccentricity equal to the radius form the center of the graph.
        Returns None if the graph is disconnected.
        
        Returns
        -------
        int | None
            Minimum eccentricity, or None if disconnected.
        
        Examples
        --------
        Calculate radius of a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        print(graph.get_graph_radius())  # 1 (hub is center)
        ```
        
        See Also
        --------
        get_graph_diameter : Get maximum eccentricity
        eccentricity : Get eccentricity of a specific node
        """
        return self._graph.get_graph_radius()
    
    def get_girth(self) -> int | None:
        """Get girth (length of shortest cycle).
        
        The girth is the length of the shortest cycle in the graph.
        Returns None if the graph is acyclic (a forest/tree). Uses BFS
        from each node to detect shortest cycles efficiently.
        
        Returns
        -------
        int | None
            Length of shortest cycle, or None if acyclic.
        
        Examples
        --------
        Find girth of a cycle graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )
        
        print(graph.get_girth())  # 4 (cycle length)
        ```
        
        Trees have no cycles:
        
        ```python
        tree = Graph(directed=True).add_balanced_tree(k=2, h=3)
        print(tree.get_girth())  # None
        ```
        
        See Also
        --------
        has_cycles : Check if graph contains any cycles
        is_dag : Check if directed graph is acyclic
        """
        return self._graph.get_girth()
    
    def get_adhesion(self) -> int:
        """Get adhesion (minimum edge cut size).
        
        The adhesion is the minimum number of edges whose removal would disconnect
        the graph. Also called edge connectivity. Only defined for undirected graphs.
        Returns 0 if already disconnected.
        
        Note: Current implementation provides an upper bound (minimum degree) for
        efficiency. Exact computation requires max-flow algorithms.
        
        Returns
        -------
        int
            Minimum edge cut size.
            
        Raises
        ------
        ValueError
            If graph is directed.
        
        Examples
        --------
        Calculate adhesion:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )
        
        print(graph.get_adhesion())  # 2
        ```
        
        See Also
        --------
        get_cohesion : Get vertex connectivity
        is_connected : Check if graph is connected
        """
        return self._graph.get_adhesion()
    
    def get_cohesion(self) -> int:
        """Get cohesion (minimum vertex cut size).
        
        The cohesion is the minimum number of nodes whose removal would disconnect
        the graph. Also called vertex connectivity. Only defined for undirected graphs.
        Returns 0 if already disconnected.
        
        Note: Current implementation provides an upper bound (minimum degree) for
        efficiency. Exact computation requires sophisticated algorithms.
        
        Returns
        -------
        int
            Minimum vertex cut size.
            
        Raises
        ------
        ValueError
            If graph is directed.
        
        Examples
        --------
        Calculate cohesion:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )
        
        print(graph.get_cohesion())  # 2
        ```
        
        See Also
        --------
        get_adhesion : Get edge connectivity
        is_connected : Check if graph is connected
        """
        return self._graph.get_cohesion()
    
    def get_reciprocity(self) -> float:
        """Get reciprocity (fraction of reciprocal edges).
        
        Reciprocity measures the proportion of edges that have a corresponding
        reverse edge. Only defined for directed graphs. Returns a value between
        0.0 (no reciprocal edges) and 1.0 (all edges are reciprocal). Self-loops
        are excluded from the calculation.
        
        Returns
        -------
        float
            Fraction of reciprocal edges (0.0 to 1.0).
            
        Raises
        ------
        ValueError
            If graph is undirected.
        
        Examples
        --------
        Calculate reciprocity:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "A")  # Reciprocal
            .add_edge("A", "C")  # Not reciprocal
        )
        
        print(f"{graph.get_reciprocity():.3f}")  # 0.667 (2/3 edges)
        ```
        
        See Also
        --------
        count_mutual_node_pairs : Count bidirectional relationships
        count_asymmetric_node_pairs : Count one-way relationships
        """
        return self._graph.get_reciprocity()
    
    def get_min_cut_between(self, source: str, sink: str) -> int:
        """Get minimum cut between source and sink nodes.
        
        Computes the minimum number of edges that must be removed to disconnect
        the source from the sink. Uses the Ford-Fulkerson algorithm with BFS
        (Edmonds-Karp) to find maximum flow, which equals the minimum cut by
        the max-flow min-cut theorem.
        
        For unweighted graphs, each edge has capacity 1. For undirected graphs,
        edges are treated as bidirectional.
        
        Parameters
        ----------
        source : str
            Source node ID.
        sink : str
            Sink node ID.
            
        Returns
        -------
        int
            Size of minimum cut (number of edges to remove).
            
        Raises
        ------
        ValueError
            If source or sink node doesn't exist, or if they're the same.
        
        Examples
        --------
        Find minimum cut:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )
        
        print(graph.get_min_cut_between("A", "D"))  # 1 (bottleneck)
        ```
        
        See Also
        --------
        get_adhesion : Get minimum edge cut for entire graph
        is_connected : Check if graph is connected
        """
        return self._graph.get_min_cut_between(source, sink)
    
    def get_eccentricity(self) -> dict[str, int | None]:
        """Get eccentricity for each node in the graph.
        
        Eccentricity of a node is the maximum distance from that node to any
        other reachable node in the graph. It measures how far a node is from
        the most distant node it can reach. Returns None for nodes in disconnected
        graphs where some nodes are unreachable.
        
        Returns
        -------
        dict[str, int | None]
            Dictionary mapping node IDs to eccentricity values. Values are None
            for nodes in disconnected components.
        
        Examples
        --------
        Calculate eccentricities in a path graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "E")
        )
        
        ecc = graph.get_eccentricity()
        # End nodes have max eccentricity (4)
        # Middle node C has min eccentricity (2)
        ```
        
        See Also
        --------
        get_graph_diameter : Get maximum eccentricity
        get_graph_radius : Get minimum eccentricity
        get_periphery : Get nodes with maximum eccentricity
        """
        return self._graph.get_eccentricity()
    
    def get_radiality(self) -> dict[str, float | None]:
        """Get radiality centrality for each node in the graph.
        
        Radiality measures how central a node is based on its eccentricity
        relative to the graph diameter. The formula is:
        
        radiality(v) = (diameter + 1 - eccentricity(v)) / (n - 1)
        
        where n is the number of nodes. Higher radiality indicates a more
        central position in the network. Returns None for nodes in disconnected
        graphs.
        
        Returns
        -------
        dict[str, float | None]
            Dictionary mapping node IDs to radiality scores (0.0 to 1.0).
            Values are None for nodes in disconnected components.
        
        Examples
        --------
        Calculate radiality in a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        rad = graph.get_radiality()
        # Hub has highest radiality (most central)
        # Leaves have lower radiality (peripheral)
        ```
        
        See Also
        --------
        get_eccentricity : Get eccentricity values
        get_graph_diameter : Get graph diameter
        get_closeness_vitality : Alternative centrality measure
        """
        return self._graph.get_radiality()
    
    def get_closeness_vitality(self) -> dict[str, float]:
        """Get closeness vitality centrality for each node.
        
        Closeness vitality measures a node's importance by computing the change
        in total graph distances when the node is removed:
        
        vitality(v) = sum_distances(G) - sum_distances(G - v)
        
        Higher values indicate more important/central nodes whose removal would
        significantly increase average path lengths. Uses parallel BFS with Rayon
        for efficient computation.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to vitality scores. Higher scores indicate
            more structurally important nodes.
        
        Examples
        --------
        Calculate closeness vitality in a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        vitality = graph.get_closeness_vitality()
        # Hub has highest vitality (removal disconnects graph)
        # Leaves have low vitality (removal has minimal impact)
        ```
        
        Notes
        -----
        Computationally expensive for large graphs (O(V²E / num_cores) complexity).
        Consider alternative centrality measures (degree, PageRank) for graphs
        with >1000 nodes unless high accuracy is required.
        
        Performance benchmarks (Apple M1, 8 cores):
        - 50 nodes: ~2ms (23K nodes/sec)
        - 200 nodes: ~120ms (1.6K nodes/sec)
        - 500 nodes: ~1.8s (273 nodes/sec)
        
        See Also
        --------
        pagerank : Faster alternative for large graphs
        betweenness_centrality : Measures flow control
        get_radiality : Eccentricity-based centrality
        """
        return self._graph.get_closeness_vitality()
    
    def get_coreness(self) -> dict[str, int]:
        """Get k-core decomposition (coreness for each node).
        
        The k-core is the maximal subgraph where every node has degree >= k.
        A node's coreness is the highest k-value for which it belongs to the k-core.
        Uses iterative pruning: repeatedly removes nodes with degree < k until
        all remaining nodes have degree >= k.
        
        Higher coreness indicates nodes in denser, more cohesive parts of the graph.
        The k-core decomposition is useful for finding the "nucleus" of a network.
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping node IDs to coreness values (non-negative integers).
        
        Examples
        --------
        Calculate coreness in a complete graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )
        
        coreness = graph.get_coreness()
        # All nodes in K4 have coreness 3
        ```
        
        See Also
        --------
        degree : Get node degree
        density : Get overall graph density
        """
        return self._graph.get_coreness()
    
    # ========== Community Detection Methods ==========
    
    def get_cmty_fast_greedy(self) -> dict[str, int]:
        """Detect communities using Fast Greedy (Clauset-Newman-Moore) algorithm.
        
        This is a greedy modularity optimization algorithm that merges communities
        to maximize modularity. It's efficient for medium to large graphs and provides
        good quality communities. Only works with undirected graphs.
        
        Algorithm:
        1. Start with each node in its own community
        2. Build a heap of merge candidates ordered by modularity gain (ΔQ)
        3. Repeatedly merge the pair with highest ΔQ until no improvement
        4. Return the partition with maximum modularity
        
        Returns
        -------
        dict[str, int]
            Mapping from node ID to community ID (0-indexed).
            
        Raises
        ------
        ValueError
            If graph is directed (requires undirected graph).
        
        Examples
        --------
        Detect communities in disconnected cliques:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Clique 1
            .add_edge("D", "E")  # Clique 2
        )
        
        communities = graph.get_cmty_fast_greedy()
        print(len(set(communities.values())))  # 2 communities
        ```
        
        Notes
        -----
        Complexity: O(n log²n) on sparse graphs, O(n + m) space.
        
        References
        ----------
        Clauset, A., Newman, M. E., & Moore, C. (2004).
        "Finding community structure in very large networks."
        Physical review E, 70(6), 066111.
        
        See Also
        --------
        get_cmty_louvain : Faster for very large graphs
        get_cmty_edge_btwns : More accurate but slower
        """
        return self._graph.get_cmty_fast_greedy()
    
    def get_cmty_edge_btwns(self) -> dict[str, int]:
        """Detect communities using Edge Betweenness (Girvan-Newman) algorithm.
        
        Iteratively removes edges with highest betweenness centrality to reveal
        community structure. Uses Brandes' algorithm for efficient betweenness
        calculation. More accurate than Fast Greedy but significantly slower
        (O(m²n) complexity).
        
        Best for small to medium graphs (< 5K nodes) with hierarchical community
        structure and clear "bridge" edges between communities. When accuracy is
        more important than speed.
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping node IDs to community IDs (0-indexed, contiguous).
            
        Raises
        ------
        ValueError
            If graph is directed (requires undirected graph).
        
        Examples
        --------
        Detect communities in bridged triangles:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Triangle 1
            .add_edge("C", "D")  # Bridge edge
            .add_edge("D", "E")
            .add_edge("E", "F")
            .add_edge("F", "D")  # Triangle 2
        )
        
        communities = graph.get_cmty_edge_btwns()
        # Bridge edge C-D has highest betweenness and is removed first
        # Result: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
        ```
        
        Notes
        -----
        Algorithm steps:
        1. Compute edge betweenness for all edges using Brandes' algorithm
        2. Remove edge with highest betweenness
        3. Recompute betweenness and check if modularity improved
        4. Repeat until modularity stops improving
        5. Return partition with maximum modularity
        
        Complexity: O(m²n) where m = edges, n = nodes. For graphs with > 10K nodes,
        consider `get_cmty_fast_greedy()` or `get_cmty_louvain()` instead.
        
        References
        ----------
        Girvan, M., & Newman, M. E. (2002). "Community structure in social
        and biological networks." PNAS, 99(12), 7821-7826.
        
        Brandes, U. (2001). "A faster algorithm for betweenness centrality."
        Journal of Mathematical Sociology, 25(2), 163-177.
        
        See Also
        --------
        get_cmty_fast_greedy : Faster greedy modularity optimization
        get_cmty_louvain : Best for very large graphs
        get_cmty_walktrap : Alternative hierarchical method
        """
        return self._graph.get_cmty_edge_btwns()
    
    def get_cmty_l_eigenvec(self) -> dict[str, int]:
        """Detect communities using Leading Eigenvector (Newman) algorithm.
        
        Spectral optimization method using recursive bisection based on the leading
        eigenvector of the modularity matrix. Splits communities hierarchically when
        modularity improves. Best for medium-sized graphs (< 20K nodes) with clear
        2-way divisions and hierarchical structure.
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping node IDs to community IDs (0-indexed, contiguous).
            
        Raises
        ------
        ValueError
            If graph is directed (requires undirected graph).
        
        Examples
        --------
        Detect communities in bridged cliques:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
        )
        
        # Build two complete triangles
        for i, j in [("A", "B"), ("B", "C"), ("C", "A")]:
            graph.add_edge(i, j)
        
        graph.add_edge("C", "D")  # Bridge
        
        for i, j in [("D", "E"), ("E", "F"), ("F", "D")]:
            graph.add_edge(i, j)
        
        communities = graph.get_cmty_l_eigenvec()
        # Eigenvector method finds natural split at bridge
        # Result: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
        ```
        
        Notes
        -----
        Algorithm steps:
        1. Start with all nodes in one community
        2. Compute modularity matrix: B_ij = A_ij - (k_i × k_j) / 2m
        3. Find leading eigenvector using power iteration
        4. Split by sign: positive → community 1, negative → community 2
        5. Accept split only if modularity improves
        6. Recursively split each community
        
        Complexity: O(n²) per split (recursive). For very large graphs (> 50K nodes),
        consider `get_cmty_fast_greedy()` or `get_cmty_louvain()` instead.
        
        The algorithm only splits when modularity improves, so it may return fewer
        communities than other methods if the structure is not hierarchically organized.
        
        References
        ----------
        Newman, M. E. J. (2006). "Finding community structure in networks using
        the eigenvectors of matrices." Physical Review E, 74(3), 036104.
        
        Newman, M. E. J. (2006). "Modularity and community structure in networks."
        PNAS, 103(23), 8577-8582.
        
        See Also
        --------
        get_cmty_fast_greedy : Faster but less principled
        get_cmty_edge_btwns : More accurate but slower
        get_cmty_louvain : Best for very large graphs
        """
        return self._graph.get_cmty_l_eigenvec()
    
    def get_cmty_louvain(self, resolution: float = 1.0) -> dict[str, int]:
        """Detect communities using Louvain multi-level optimization algorithm.
        
        The Louvain method is a hierarchical algorithm that maximizes modularity
        through iterative optimization. It's the most widely-used community detection
        algorithm in practice due to its speed (O(n log n)) and quality on large networks.
        
        The method produces a dendrogram hierarchy but returns the partition with
        maximum modularity. Works only with undirected graphs.
        
        Parameters
        ----------
        resolution : float, default 1.0
            Resolution parameter controlling community size. Higher values favor
            smaller communities. Standard modularity uses resolution=1.0.
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping node IDs to community IDs (0-indexed, contiguous).
            
        Raises
        ------
        RuntimeError
            If called on a directed graph.
        
        Examples
        --------
        Detect communities in bridged triangles:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
        )
        
        # Build two complete triangles connected by bridge
        for i, j in [("A", "B"), ("B", "C"), ("C", "A")]:
            graph.add_edge(i, j)
        
        graph.add_edge("C", "D")  # Bridge
        
        for i, j in [("D", "E"), ("E", "F"), ("F", "D")]:
            graph.add_edge(i, j)
        
        communities = graph.get_cmty_louvain()
        # Nodes in same triangle have same community ID
        assert communities["A"] == communities["B"] == communities["C"]
        assert communities["D"] == communities["E"] == communities["F"]
        assert communities["A"] != communities["D"]
        ```
        
        Control community granularity with resolution:
        
        ```python
        # Smaller communities (higher resolution)
        fine = graph.get_cmty_louvain(resolution=1.5)
        
        # Larger communities (lower resolution)
        coarse = graph.get_cmty_louvain(resolution=0.5)
        ```
        
        Notes
        -----
        Algorithm steps:
        1. Phase 1 (Local optimization): Each node tries joining neighboring
           communities to maximize modularity gain ΔQ
        2. Phase 2 (Aggregation): Communities are coarsened into super-nodes,
           creating a new network where each community becomes a single node
        3. Repeat until modularity converges (no further improvement)
        
        The resolution parameter tunes community size:
        - resolution < 1.0: Larger communities
        - resolution = 1.0: Standard modularity optimization
        - resolution > 1.0: Smaller communities
        
        The algorithm is deterministic but may produce different (equivalent)
        partitions depending on node ordering. Handles disconnected components
        naturally. Self-loops and multi-edges are currently not supported.
        
        References
        ----------
        Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
        "Fast unfolding of communities in large networks." Journal of Statistical
        Mechanics: Theory and Experiment, 2008(10), P10008.
        
        Lambiotte, R., Delvenne, J. C., & Barahona, M. (2014). "Random walks,
        Markov processes and the multiscale modular organization of complex
        networks." IEEE Transactions on Network Science and Engineering, 1(2), 76-90.
        
        See Also
        --------
        get_cmty_fast_greedy : Simpler greedy optimization
        get_cmty_edge_btwns : Divisive hierarchical method
        get_cmty_l_eigenvec : Spectral bisection method
        """
        return self._graph.get_cmty_louvain(resolution)
    
    def get_cmty_walktrap(self, steps: int = 4) -> dict[str, int]:
        """Detect communities using Walktrap random walk algorithm.
        
        The Walktrap algorithm exploits the property that random walks tend to stay
        trapped within densely connected communities. It computes distances between
        nodes based on random walk probabilities, then uses hierarchical clustering
        to identify community structure.
        
        Works only with undirected graphs. Time complexity is O(n²log n) for typical
        networks with small step parameter.
        
        Parameters
        ----------
        steps : int, default 4
            Number of random walk steps. Higher values capture longer-range
            structural patterns. Typical values: 2-8.
            - steps=2: Local structure only
            - steps=4: Good balance (default, recommended)
            - steps=6+: Longer-range patterns, more computation
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping node IDs to community IDs (0-indexed, contiguous).
            
        Raises
        ------
        ValueError
            If called on a directed graph.
        
        Examples
        --------
        Detect communities with default parameters:
        
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
        )
        
        # Two cliques connected by a bridge
        for i, j in [("A", "B"), ("B", "C"), ("C", "A")]:
            graph.add_edge(i, j)
        
        for i, j in [("D", "E"), ("E", "F"), ("F", "D")]:
            graph.add_edge(i, j)
        
        graph.add_edge("C", "D")  # Bridge
        
        communities = graph.get_cmty_walktrap()
        assert communities["A"] == communities["B"] == communities["C"]
        assert communities["D"] == communities["E"] == communities["F"]
        assert communities["A"] != communities["D"]
        ```
        
        Try different step lengths for different scales:
        
        ```python
        # More local communities
        local_cmty = graph.get_cmty_walktrap(steps=2)
        
        # More global communities
        global_cmty = graph.get_cmty_walktrap(steps=6)
        ```
        
        Notes
        -----
        Algorithm steps:
        1. Compute transition probability matrix P (P[i][j] = 1/deg(i) if edge exists)
        2. Compute t-step transition probabilities P^t by matrix exponentiation
        3. Calculate random walk distance between each pair of nodes:
           d(i,j) = sqrt(Σ_k [(P^t_ik - P^t_jk)^2 / deg(k)])
        4. Apply Ward's hierarchical clustering on the distance matrix
        5. Cut the dendrogram at the level that maximizes modularity
        
        The step parameter controls the scale of community detection:
        - Lower steps (2-3): Detect finer, more local communities
        - Higher steps (5-8): Detect coarser, more global communities
        
        The algorithm is deterministic (no randomness despite the name). Works well
        on hierarchical community structures and handles disconnected components
        automatically. Performance degrades on very large graphs (>10,000 nodes)
        due to O(n²) distance matrix computation. Self-loops and multi-edges are
        currently not supported.
        
        Performance:
        - Small graphs (<1000 nodes): <1 second
        - Medium graphs (1000-5000 nodes): 1-30 seconds
        - Large graphs (>5000 nodes): May be slow, consider Louvain instead
        
        References
        ----------
        Pons, P., & Latapy, M. (2005). "Computing communities in large networks
        using random walks." In Computer and Information Sciences-ISCIS 2005
        (pp. 284-293). Springer, Berlin, Heidelberg.
        
        Pons, P., & Latapy, M. (2006). "Computing communities in large networks
        using random walks (long version)." arXiv preprint physics/0512106.
        
        See Also
        --------
        get_cmty_louvain : Faster for large graphs
        get_cmty_fast_greedy : Alternative greedy approach
        get_cmty_edge_btwns : Divisive hierarchical method
        get_cmty_l_eigenvec : Spectral method
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
        """Get HITS authority scores for each node (directed graphs only).
        
        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores
        for nodes in a directed graph. Authority score measures how much a node
        is pointed to by good hubs:
        
        authority(v) = sum of hub scores of nodes pointing to v
        
        Uses power iteration until convergence. Good authorities are nodes that
        are pointed to by many high-quality hubs. Requires a directed graph.
        
        Parameters
        ----------
        max_iter : int, default 100
            Maximum number of iterations for convergence.
        tol : float, default 1e-6
            Convergence tolerance threshold.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to normalized authority scores.
            
        Raises
        ------
        ValueError
            If graph is undirected.
        
        Examples
        --------
        Calculate authority scores in a citation network:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="Paper1")
            .add_node(label="Paper2")
            .add_node(label="Paper3")
            .add_edge("Paper1", "Paper2")  # Paper1 cites Paper2
            .add_edge("Paper1", "Paper3")  # Paper1 cites Paper3
        )
        
        auth = graph.get_authority_centrality()
        # Paper2 and Paper3 have high authority (cited)
        # Paper1 has low authority (not cited)
        ```
        
        See Also
        --------
        get_hub_centrality : Get hub scores
        pagerank : Alternative prestige measure
        """
        return self._graph.get_authority_centrality(max_iter, tol)
    
    def get_hub_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get HITS hub scores for each node (directed graphs only).
        
        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores
        for nodes in a directed graph. Hub score measures how much a node points
        to good authorities:
        
        hub(v) = sum of authority scores of nodes v points to
        
        Uses power iteration until convergence. Good hubs are nodes that point
        to many high-quality authorities. Requires a directed graph.
        
        Parameters
        ----------
        max_iter : int, default 100
            Maximum number of iterations for convergence.
        tol : float, default 1e-6
            Convergence tolerance threshold.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to normalized hub scores.
            
        Raises
        ------
        ValueError
            If graph is undirected.
        
        Examples
        --------
        Calculate hub scores in a citation network:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="Paper1")
            .add_node(label="Paper2")
            .add_node(label="Paper3")
            .add_edge("Paper1", "Paper2")  # Paper1 cites Paper2
            .add_edge("Paper1", "Paper3")  # Paper1 cites Paper3
        )
        
        hub = graph.get_hub_centrality()
        # Paper1 has high hub score (cites many papers)
        # Paper2 and Paper3 have low hub scores
        ```
        
        See Also
        --------
        get_authority_centrality : Get authority scores
        pagerank : Alternative prestige measure
        """
        return self._graph.get_hub_centrality(max_iter, tol)
    
    def get_eigen_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get eigenvector centrality for each node.
        
        Eigenvector centrality measures node importance based on connections
        to other important nodes. It's the principal eigenvector of the adjacency
        matrix, computed via power iteration:
        
        x = λ⁻¹ · A · x
        
        A node has high eigenvector centrality if it's connected to other nodes
        that themselves have high centrality. Unlike degree centrality, it accounts
        for the quality (not just quantity) of connections.
        
        Parameters
        ----------
        max_iter : int, default 100
            Maximum number of iterations for convergence.
        tol : float, default 1e-6
            Convergence tolerance threshold.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to normalized eigenvector centrality scores.
        
        Examples
        --------
        Calculate eigenvector centrality in a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        eigen = graph.get_eigen_centrality()
        # Hub has highest eigenvector centrality
        # All connected nodes have non-zero centrality
        ```
        
        See Also
        --------
        pagerank : Alternative prestige measure
        get_alpha_centrality : Generalization of eigenvector centrality
        degree : Simple connectivity measure
        """
        return self._graph.get_eigen_centrality(max_iter, tol)
    
    def get_alpha_centrality(self, alpha: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Get alpha centrality for each node.
        
        Alpha centrality is a generalization of eigenvector centrality that adds
        an exogenous factor:
        
        x = α · A · x + e
        
        where α is the attenuation factor, A is the adjacency matrix, and e is
        the exogenous factor (set to 1 for all nodes). Lower α emphasizes local
        neighborhood structure, while higher α considers more global structure.
        
        Parameters
        ----------
        alpha : float, default 0.1
            Attenuation factor (range: 0 to 1). Controls the balance between
            local and global structure.
        max_iter : int, default 100
            Maximum number of iterations for convergence.
        tol : float, default 1e-6
            Convergence tolerance threshold.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to alpha centrality scores.
        
        Examples
        --------
        Calculate alpha centrality with different α values:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        alpha_local = graph.get_alpha_centrality(alpha=0.05)  # Local emphasis
        alpha_global = graph.get_alpha_centrality(alpha=0.5)  # Global emphasis
        ```
        
        See Also
        --------
        get_eigen_centrality : Special case with α → 1
        pagerank : Alternative prestige measure
        """
        return self._graph.get_alpha_centrality(alpha, max_iter, tol)
    
    def get_leverage_centrality(self) -> dict[str, float]:
        """Get leverage centrality for each node.
        
        Leverage centrality measures the degree connectivity of a node relative
        to its neighbors:
        
        leverage(v) = (1/k_v) · Σ[(k_v - k_n) / (k_v + k_n)]
        
        where k_v is the degree of node v, and k_n is the degree of each neighbor n.
        
        Interpretation:
        - Positive values: Node has more connections than its neighbors (influential)
        - Negative values: Node has fewer connections than its neighbors (peripheral)
        - Zero: Node has similar connectivity to its neighbors
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping node IDs to leverage centrality scores.
        
        Examples
        --------
        Calculate leverage centrality in a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        leverage = graph.get_leverage_centrality()
        # Hub has positive leverage (high degree, low-degree neighbors)
        # Leaves have negative leverage (low degree, high-degree neighbor)
        ```
        
        See Also
        --------
        degree : Get node degree
        get_eigen_centrality : Alternative importance measure
        """
        return self._graph.get_leverage_centrality()
    
    # ========== Selection Methods ==========
    
    def get_dice_similarity(self) -> dict[str, dict[str, float]]:
        """Get Dice similarity coefficient between all node pairs.
        
        The Dice coefficient measures similarity between two nodes based on their
        neighborhood overlap:
        
        Dice(A, B) = 2 × |neighbors(A) ∩ neighbors(B)| / (|neighbors(A)| + |neighbors(B)|)
        
        Range: [0, 1] where 0 means no common neighbors and 1 means identical
        neighborhoods (or same node). Self-similarity is always 1.0.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores.
            Format: {node1_id: {node2_id: similarity_score}}
        
        Examples
        --------
        Calculate Dice similarity for link prediction:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("D", "B")
            .add_edge("D", "C")
        )
        
        sim = graph.get_dice_similarity()
        # A and D share neighbors B and C
        print(sim["A"]["D"])  # 1.0 (perfect overlap)
        ```
        
        Notes
        -----
        Useful for:

        - Link prediction (similar nodes likely to connect)
        - Community detection (nodes with similar neighborhoods)
        - Recommendation systems (find similar users/items)
        
        See Also
        --------
        get_jaccard_similarity : More conservative similarity measure
        get_cosine_similarity : Geometric similarity measure
        get_overlap_similarity : Emphasizes subset relationships
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
        
        Range: [0, 1] where 0 means no common neighbors and 1 means identical
        neighborhoods (or same node). More conservative than Dice coefficient
        (always ≤ Dice score). Self-similarity is always 1.0.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores.
            Format: {node1_id: {node2_id: similarity_score}}
        
        Examples
        --------
        Calculate Jaccard similarity in a star graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A")
            .add_edge("hub", "B")
            .add_edge("hub", "C")
        )
        
        sim = graph.get_jaccard_similarity()
        # Leaves have no common neighbors
        print(sim["A"]["B"])  # 0.0
        ```
        
        Notes
        -----
        Widely used in:

        - Document similarity (bag-of-words)
        - User similarity (common items/friends)
        - Biological networks (protein interactions)
        
        See Also
        --------
        get_dice_similarity : Less conservative alternative
        get_cosine_similarity : Geometric similarity measure
        get_overlap_similarity : Emphasizes subset relationships
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
        """Get cosine similarity between all node pairs.
        
        Cosine similarity treats neighbors as vector components:
        
        Cosine(A, B) = |neighbors(A) ∩ neighbors(B)| / √(|neighbors(A)| × |neighbors(B)|)
        
        Range: [0, 1] where 0 means no common neighbors and 1 means identical
        neighborhoods. Geometric interpretation: cosine of angle between
        neighbor vectors. Self-similarity is always 1.0.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores.
            Format: {node1_id: {node2_id: similarity_score}}
        
        Examples
        --------
        Calculate cosine similarity in a triangle graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        
        sim = graph.get_cosine_similarity()
        # Each node shares 1 neighbor with the others
        print(sim["A"]["B"])  # 0.5 (1 common / sqrt(2 * 2))
        ```
        
        Notes
        -----
        Advantages:

        - Scales well with high-degree nodes
        - Balances degree differences
        - Standard in text mining and recommendation systems
        
        See Also
        --------
        get_dice_similarity : Harmonic mean of degrees
        get_jaccard_similarity : Set-theoretic similarity
        get_overlap_similarity : Min-based similarity
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
        """Get overlap coefficient between all node pairs.
        
        The overlap coefficient emphasizes subset relationships:
        
        Overlap(A, B) = |neighbors(A) ∩ neighbors(B)| / min(|neighbors(A)|, |neighbors(B)|)
        
        Range: [0, 1] where 0 means no common neighbors and 1 means one
        neighborhood is a subset of the other. Asymmetric measure useful for
        hierarchical patterns. Self-similarity is always 1.0.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores.
            Format: {node1_id: {node2_id: similarity_score}}
        
        Examples
        --------
        Detect hierarchical relationships in a hub-spoke graph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="spoke")
            .add_node(label="extended")
            .add_edge("hub", "spoke")
            .add_edge("hub", "extended")
            .add_edge("spoke", "extra")
        )
        
        sim = graph.get_overlap_similarity()
        # Hub's neighborhood is subset of spoke's
        print(sim["hub"]["spoke"])  # 1.0
        ```
        
        Notes
        -----
        Use cases:

        - Detecting specialization/generalization
        - Finding hubs whose neighbors are subsets of others
        - Hierarchical community structure
        
        See Also
        --------
        get_jaccard_similarity : Symmetric set similarity
        get_dice_similarity : Harmonic mean alternative
        get_cosine_similarity : Geometric interpretation
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
        
        Marks specified nodes as selected for subsequent operations like building
        subgraphs or filtering. Selection is cumulative unless cleared first.
        
        Parameters
        ----------
        node_ids : list[str]
            List of node IDs to select.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Raises
        ------
        KeyError
            If any node ID doesn't exist in the graph.
        
        Examples
        --------
        Select specific nodes and build a subgraph:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .select_nodes_by_id(["A", "B"])
        )
        
        selected = graph.get_selected_nodes()
        print(selected)  # ['A', 'B']
        ```
        
        See Also
        --------
        get_selected_nodes : Get list of currently selected nodes
        clear_selection : Clear all selections
        select_nodes : Select using predicate function
        """
        self._graph.select_nodes_by_id(node_ids)
        return self
    
    def select_edges_by_id(self, edge_ids: list[int]) -> "Graph":
        """Select edges by their IDs.
        
        Marks specified edges as selected for subsequent operations. Edge IDs are
        internal integer identifiers assigned when edges are created. Selection is
        cumulative unless cleared first.
        
        Parameters
        ----------
        edge_ids : list[int]
            List of edge IDs to select.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Raises
        ------
        KeyError
            If any edge ID doesn't exist in the graph.
        
        Examples
        --------
        Select specific edges:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        
        edge_ids = graph.edge_ids()
        graph.select_edges_by_id([edge_ids[0], edge_ids[1]])
        
        selected_count = graph.count_selected_edges()
        print(selected_count)  # 2
        ```
        
        See Also
        --------
        get_selected_edges : Get list of currently selected edges
        edge_ids : Get all edge IDs
        select_edges : Select using predicate function
        """
        self._graph.select_edges_by_id(edge_ids)
        return self
    
    def select_edges_by_node_id(self, node_ids: list[str]) -> "Graph":
        """Select all edges connected to specified nodes.
        
        Selects all edges where at least one endpoint is in the provided node list.
        For directed graphs, includes both incoming and outgoing edges. Useful for
        selecting the local connectivity around specific nodes.
        
        Parameters
        ----------
        node_ids : list[str]
            List of node IDs whose incident edges should be selected.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select edges incident to a specific node:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .select_edges_by_node_id(["B"])
        )
        
        # Selects A-B and B-C edges
        selected = graph.count_selected_edges()
        print(selected)  # 2
        ```
        
        See Also
        --------
        select_nodes_by_id : Select specific nodes
        neighbors : Get neighbors of a node
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
        
        Filters nodes based on their degree (number of connections). Useful for
        finding hubs (high degree) or peripheral nodes (low degree). Can filter
        by in-degree, out-degree, or total degree.
        
        Parameters
        ----------
        min_degree : int | None, default None
            Minimum degree threshold (inclusive). None means no lower bound.
        max_degree : int | None, default None
            Maximum degree threshold (inclusive). None means no upper bound.
        degree_type : str, default "total"
            Type of degree to use: "in", "out", or "total".
            For undirected graphs, only "total" is valid.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select high-degree hub nodes:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(10, center_label="hub")
            .select_nodes_by_degree(min_degree=5)
        )
        
        # Hub node has degree 10, all others have degree 1
        hubs = graph.get_selected_nodes()
        print(hubs)  # ['hub']
        ```
        
        Filter by degree range:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_path_graph(10)
            .select_nodes_by_degree(min_degree=1, max_degree=2, degree_type="out")
        )
        
        # Selects nodes with 1-2 outgoing edges
        ```
        
        See Also
        --------
        degree : Get degree of a node
        in_degree : Get in-degree of a node
        out_degree : Get out-degree of a node
        """
        self._graph.select_nodes_by_degree(
            min_degree=min_degree,
            max_degree=max_degree,
            degree_type=degree_type
        )
        return self
    
    def deselect_nodes(self, node_ids: list[str]) -> "Graph":
        """Remove nodes from selection.
        
        Unmarks specified nodes, removing them from the current selection set.
        Has no effect on nodes that aren't currently selected.
        
        Parameters
        ----------
        node_ids : list[str]
            List of node IDs to deselect.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Refine a selection by removing specific nodes:
        
        ```python
        graph = (
            Graph()
            .add_path_graph(5)
            .select_nodes_by_id(["0", "1", "2", "3", "4"])
            .deselect_nodes(["0", "4"])
        )
        
        # Only middle nodes remain selected
        selected = graph.get_selected_nodes()
        print(selected)  # ['1', '2', '3']
        ```
        
        See Also
        --------
        select_nodes_by_id : Select specific nodes
        clear_selection : Clear all selections
        """
        self._graph.deselect_nodes(node_ids)
        return self
    
    def deselect_edges(self, edge_ids: list[int]) -> "Graph":
        """Remove edges from selection.
        
        Unmarks specified edges, removing them from the current selection set.
        Has no effect on edges that aren't currently selected.
        
        Parameters
        ----------
        edge_ids : list[int]
            List of edge IDs to deselect.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Refine edge selection:
        
        ```python
        graph = Graph().add_cycle_graph(5)
        
        all_edges = graph.edge_ids()
        graph.select_edges_by_id(all_edges)
        
        # Remove first edge from selection
        graph.deselect_edges([all_edges[0]])
        
        count = graph.count_selected_edges()
        print(count)  # 4
        ```
        
        See Also
        --------
        select_edges_by_id : Select specific edges
        clear_selection : Clear all selections
        """
        self._graph.deselect_edges(edge_ids)
        return self
    
    def clear_selection(self) -> "Graph":
        """Clear all node and edge selections.
        
        Resets the selection state, unmarking all previously selected nodes and
        edges. Useful for starting fresh with a new selection workflow.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Reset selection state:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(5)
            .select_nodes_by_degree(min_degree=3)
            .clear_selection()
        )
        
        # No nodes selected after clearing
        count = graph.count_selected_nodes()
        print(count)  # 0
        ```
        
        See Also
        --------
        get_selected_nodes : Check current node selection
        get_selected_edges : Check current edge selection
        """
        self._graph.clear_selection()
        return self
    
    def invert_node_selection(self) -> "Graph":
        """Invert node selection (select unselected, unselect selected).
        
        Toggles the selection state of all nodes: currently selected nodes become
        unselected, and unselected nodes become selected. Useful for getting the
        complement of a selection.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Get complement of a selection:
        
        ```python
        graph = (
            Graph()
            .add_path_graph(10)
            .select_nodes_by_degree(max_degree=1)  # Select endpoints
            .invert_node_selection()  # Select middle nodes
        )
        
        # Middle 8 nodes now selected
        count = graph.count_selected_nodes()
        print(count)  # 8
        ```
        
        See Also
        --------
        invert_edge_selection : Invert edge selection
        """
        self._graph.invert_node_selection()
        return self
    
    def invert_edge_selection(self) -> "Graph":
        """Invert edge selection (select unselected, unselect selected).
        
        Toggles the selection state of all edges: currently selected edges become
        unselected, and unselected edges become selected. Useful for getting the
        complement of an edge selection.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select all edges except specific ones:
        
        ```python
        graph = Graph().add_cycle_graph(5)
        
        # Select one edge, then invert to select all others
        first_edge = graph.edge_ids()[0]
        graph.select_edges_by_id([first_edge]).invert_edge_selection()
        
        count = graph.count_selected_edges()
        print(count)  # 4 (all except first)
        ```
        
        See Also
        --------
        invert_node_selection : Invert node selection
        """
        self._graph.invert_edge_selection()
        return self
    
    def get_selected_nodes(self) -> list[str]:
        """Get currently selected node IDs.
        
        Returns the list of node IDs that are currently marked as selected.
        Order is not guaranteed.
        
        Returns
        -------
        list[str]
            List of selected node IDs, empty if no nodes selected.
        
        Examples
        --------
        Query current selection:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(5, center_label="hub")
            .select_nodes_by_degree(min_degree=3)
        )
        
        selected = graph.get_selected_nodes()
        print(selected)  # ['hub']
        ```
        
        See Also
        --------
        count_selected_nodes : Get count of selected nodes
        select_nodes_by_id : Select specific nodes
        """
        return self._graph.get_selected_nodes()
    
    def get_selected_edges(self) -> list[int]:
        """Get currently selected edge IDs.
        
        Returns the list of edge IDs that are currently marked as selected.
        Order is not guaranteed.
        
        Returns
        -------
        list[int]
            List of selected edge IDs, empty if no edges selected.
        
        Examples
        --------
        Query edge selection:
        
        ```python
        graph = (
            Graph()
            .add_cycle_graph(4)
            .select_edges(lambda e: e.weight and e.weight > 0.5)
        )
        
        selected = graph.get_selected_edges()
        print(len(selected))
        ```
        
        See Also
        --------
        count_selected_edges : Get count of selected edges
        select_edges_by_id : Select specific edges
        """
        return self._graph.get_selected_edges()
    
    def count_selected_nodes(self) -> int:
        """Get count of currently selected nodes.
        
        Returns the number of nodes currently marked as selected. More efficient
        than len(get_selected_nodes()) when you only need the count.
        
        Returns
        -------
        int
            Number of selected nodes.
        
        Examples
        --------
        Count selected nodes:
        
        ```python
        graph = (
            Graph()
            .add_full_graph(10)
            .select_nodes_by_degree(min_degree=5)
        )
        
        count = graph.count_selected_nodes()
        print(f"{count} high-degree nodes")
        ```
        
        See Also
        --------
        get_selected_nodes : Get list of selected node IDs
        count_selected_edges : Count selected edges
        """
        return self._graph.count_selected_nodes()
    
    def count_selected_edges(self) -> int:
        """Get count of currently selected edges.
        
        Returns the number of edges currently marked as selected. More efficient
        than len(get_selected_edges()) when you only need the count.
        
        Returns
        -------
        int
            Number of selected edges.
        
        Examples
        --------
        Count selected edges:
        
        ```python
        graph = (
            Graph()
            .add_cycle_graph(10)
            .select_edges(lambda e: e.weight and e.weight > 0.7)
        )
        
        count = graph.count_selected_edges()
        print(f"{count} high-weight edges")
        ```
        
        See Also
        --------
        get_selected_edges : Get list of selected edge IDs
        count_selected_nodes : Count selected nodes
        """
        return self._graph.count_selected_edges()
    
    def transform_to_subgraph_ws(self) -> "Graph":
        """Create a subgraph from current selection.
        
        Extracts the currently selected nodes and/or edges into a new Graph object.
        The behavior depends on what's selected:
        - Nodes selected: Include those nodes plus all edges between them
        - Edges selected: Include those edges plus their endpoint nodes
        - Both selected: Uses node selection (ignores edge selection)
        
        All node and edge attributes are preserved in the subgraph.
        
        Returns
        -------
        Graph
            New Graph object containing the subgraph. Has same directedness as
            original graph.
        
        Examples
        --------
        Extract a node-induced subgraph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_path_graph(6)
            .select_nodes_by_id(["2", "3", "4"])
        )
        
        subgraph = graph.transform_to_subgraph_ws()
        print(subgraph.num_nodes())  # 3
        print(subgraph.num_edges())  # 2 (edges 2→3 and 3→4)
        ```
        
        Extract an edge-induced subgraph:
        
        ```python
        graph = Graph().add_cycle_graph(5)
        
        edges = graph.edge_ids()[:2]
        graph.select_edges_by_id(edges)
        
        subgraph = graph.transform_to_subgraph_ws()
        # Contains only selected edges and their endpoints
        ```
        
        See Also
        --------
        select_nodes_by_id : Select specific nodes
        select_edges_by_id : Select specific edges
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
        
        Applies a custom filter function to all nodes, selecting those for which
        the predicate returns True. The predicate receives a Node object with
        properties: id, label, and attributes.
        
        Parameters
        ----------
        predicate : callable
            Function that takes a Node object and returns bool. Return True to
            select the node.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select nodes by attribute value:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A", type="hub", score=10)
            .add_node(label="B", type="leaf", score=5)
            .add_node(label="C", type="hub", score=8)
            .select_nodes(lambda n: n.attributes.get("type") == "hub")
        )
        
        selected = graph.get_selected_nodes()
        print(selected)  # ['A', 'C']
        ```
        
        Select nodes by ID pattern:
        
        ```python
        graph.select_nodes(lambda n: n.id.startswith("user_"))
        ```
        
        See Also
        --------
        select_nodes_by_id : Select by explicit ID list
        select_edges : Select edges with predicate
        """
        self._graph.select_nodes(predicate)
        return self
    
    def select_edges(self, predicate) -> "Graph":
        """Select edges using a predicate function.
        
        Applies a custom filter function to all edges, selecting those for which
        the predicate returns True. The predicate receives an Edge object with
        properties: id, source, target, weight, and attributes.
        
        Parameters
        ----------
        predicate : callable
            Function that takes an Edge object and returns bool. Return True to
            select the edge.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select edges by weight threshold:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B", weight=0.3)
            .add_edge("B", "C", weight=0.8)
            .add_edge("C", "D", weight=0.6)
            .select_edges(lambda e: e.weight and e.weight > 0.5)
        )
        
        count = graph.count_selected_edges()
        print(count)  # 2
        ```
        
        Select edges by relationship type:
        
        ```python
        graph.select_edges(
            lambda e: e.attributes and e.attributes.get("rel") == "friend"
        )
        ```
        
        See Also
        --------
        select_edges_by_id : Select by explicit ID list
        select_nodes : Select nodes with predicate
        """
        self._graph.select_edges(predicate)
        return self
    
    def get_node(self, node: Union[int, str]):
        """Get a node object by ID or label.
        
        Retrieves the Node object for the specified node. The Node object contains
        id, label, and attributes properties. Returns None if node doesn't exist.
        
        Parameters
        ----------
        node : int | str
            Node ID (str) or internal ID (int) to retrieve.
        
        Returns
        -------
        Node | None
            Node object or None if not found.
        
        Examples
        --------
        Access node properties:
        
        ```python
        graph = Graph().add_node(label="A", color="red", size=10)
        
        node = graph.get_node("A")
        print(node.id)  # 'A'
        print(node.attributes)  # {'color': 'red', 'size': 10}
        ```
        
        See Also
        --------
        get_edge : Get edge object by ID
        has_node : Check if node exists
        """
        node_id = self._resolve_node_id(node, auto_create=False)
        return self._graph.get_node(node_id)
    
    def get_edge(self, edge_id: int):
        """Get an edge object by ID.
        
        Retrieves the Edge object for the specified edge ID. The Edge object
        contains id, source, target, weight, and attributes properties. Returns
        None if edge doesn't exist.
        
        Parameters
        ----------
        edge_id : int
            Edge ID to retrieve.
        
        Returns
        -------
        Edge | None
            Edge object or None if not found.
        
        Examples
        --------
        Access edge properties:
        
        ```python
        graph = Graph().add_node("A").add_node("B").add_edge("A", "B", weight=1.5, type="friend")
        
        edge_ids = graph.edge_ids()
        edge = graph.get_edge(edge_ids[0])
        
        print(edge.source)  # Node ID for 'A'
        print(edge.target)  # Node ID for 'B'
        print(edge.weight)  # 1.5
        print(edge.attributes)  # {'type': 'friend'}
        ```
        
        See Also
        --------
        get_node : Get node object by ID
        edge_ids : Get all edge IDs
        """
        return self._graph.get_edge(edge_id)
    
    def select_last_nodes_created(self, n: int) -> "Graph":
        """Select the last N nodes that were created.
        
        Selects nodes in reverse chronological order based on creation time.
        Useful for incremental graph building workflows where you want to work
        with recently added nodes.
        
        Parameters
        ----------
        n : int
            Number of most recently created nodes to select.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select recently added nodes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .select_last_nodes_created(2)
        )
        
        # C and D are selected (most recent)
        selected = graph.get_selected_nodes()
        print(selected)  # ['C', 'D']
        ```
        
        See Also
        --------
        select_last_edges_created : Select recently created edges
        get_selected_nodes : Get currently selected nodes
        """
        self._graph.select_last_nodes_created(n)
        return self
    
    def select_last_edges_created(self, n: int) -> "Graph":
        """Select the last N edges that were created.
        
        Selects edges in reverse chronological order based on creation time.
        Useful for incremental graph building workflows where you want to work
        with recently added edges.
        
        Parameters
        ----------
        n : int
            Number of most recently created edges to select.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select recently added edges:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "E")
            .select_last_edges_created(2)
        )
        
        # Last 2 edges selected
        count = graph.count_selected_edges()
        print(count)  # 2
        ```
        
        See Also
        --------
        select_last_nodes_created : Select recently created nodes
        get_selected_edges : Get currently selected edges
        """
        self._graph.select_last_edges_created(n)
        return self
    
    def select_nodes_in_neighborhood(
        self,
        node_ids: list[str] | str,
        k: int = 1,
        mode: str = "out"
    ) -> "Graph":
        """Select all nodes within K hops of specified seed nodes.
        
        Uses breadth-first search to find neighbors at distance up to k from
        the seed nodes. Does not include the seed nodes themselves in the
        selection. Useful for extracting local neighborhoods.
        
        Parameters
        ----------
        node_ids : list[str] | str
            Seed node ID(s) to start from. Can be a single ID string or list.
        k : int, default 1
            Maximum number of hops/distance to traverse.
        mode : str, default "out"
            Direction to traverse: "out", "in", or "all".
            For undirected graphs, always uses "all" mode.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select 1-hop neighborhood:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(5, center_label="hub")
            .select_nodes_in_neighborhood("hub", k=1, mode="all")
        )
        
        # Selects all neighbors of hub (not hub itself)
        count = graph.count_selected_nodes()
        print(count)  # 5
        ```
        
        Select 2-hop neighborhood in directed graph:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_path_graph(10)
            .select_nodes_in_neighborhood("0", k=2, mode="out")
        )
        
        # Selects nodes 1 and 2 (reachable within 2 hops)
        ```
        
        See Also
        --------
        neighbors : Get direct neighbors of a node
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
        
        Computes centrality scores for all nodes and selects the k highest-scoring
        nodes. Useful for identifying the most important or influential nodes by
        various centrality metrics.
        
        Parameters
        ----------
        measure : str, default "degree"
            Centrality measure to use: "degree", "in_degree", "out_degree",
            "pagerank", or "betweenness".
        k : int, default 10
            Number of top nodes to select.
        include_ties : bool, default False
            If True, include all nodes tied with the k-th highest score.
        **kwargs
            Additional arguments for centrality algorithms (e.g., damping for pagerank).
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Select high-degree hub nodes:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(10, center_label="hub")
            .add_path_graph(10)
            .select_nodes_by_centrality("degree", k=3)
        )
        
        # Selects top 3 nodes by degree
        hubs = graph.get_selected_nodes()
        ```
        
        Select by PageRank:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_path_graph(20)
            .select_nodes_by_centrality("pagerank", k=5, damping=0.85)
        )
        ```
        
        See Also
        --------
        degree : Get node degree
        pagerank : Compute PageRank scores
        betweenness_centrality : Compute betweenness centrality
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
        
        Extracts a specific attribute from one or more nodes. Only returns entries
        for nodes that have the specified attribute set. Useful for collecting
        properties across the graph.
        
        Parameters
        ----------
        attr : str
            Attribute name to extract.
        nodes : list[str] | None, default None
            Optional list of node IDs to query. If None, queries all nodes.
        
        Returns
        -------
        dict[str, any]
            Dictionary mapping node IDs to attribute values. Only includes nodes
            that have the attribute set.
        
        Examples
        --------
        Get specific attribute from all nodes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A", type="user", score=95)
            .add_node(label="B", type="admin", score=88)
            .add_node(label="C", type="user", score=92)
        )
        
        scores = graph.get_node_attrs("score")
        print(scores)  # {'A': 95, 'B': 88, 'C': 92}
        
        types = graph.get_node_attrs("type")
        print(types)  # {'A': 'user', 'B': 'admin', 'C': 'user'}
        ```
        
        Get attribute from specific nodes:
        
        ```python
        user_scores = graph.get_node_attrs("score", nodes=["A", "C"])
        print(user_scores)  # {'A': 95, 'C': 92}
        ```
        
        See Also
        --------
        set_node_attrs : Set attribute values for nodes
        get_node : Get full node object with all attributes
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
        
        Updates a specific attribute on nodes. Can set the same value for all nodes,
        or provide different values via a list or dictionary. Creates the attribute
        if it doesn't exist, overwrites if it does.
        
        Parameters
        ----------
        node_attr : str
            The name of the attribute to set.
        values : any | list | dict
            The value(s) to set. Can be:
            - Single value: Applied to all specified nodes
            - List: Must match length of nodes (paired by position)
            - Dict: Maps node IDs to values (only sets specified nodes)
        nodes : list[str] | None, default None
            Optional list of node IDs to update. If None, updates all nodes.
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Raises
        ------
        ValueError
            If values is a list and length doesn't match nodes length.
        
        Examples
        --------
        Set same value for specific nodes:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .set_node_attrs("color", "blue", nodes=["A", "B"])
        )
        
        colors = graph.get_node_attrs("color")
        print(colors)  # {'A': 'blue', 'B': 'blue'}
        ```
        
        Set same value for all nodes:
        
        ```python
        graph.set_node_attrs("status", "active")
        
        statuses = graph.get_node_attrs("status")
        # All nodes have status='active'
        ```
        
        Set different values using a list:
        
        ```python
        graph.set_node_attrs(
            "score",
            [100, 85, 92],
            nodes=["A", "B", "C"]
        )
        ```
        
        Set different values using a dictionary:
        
        ```python
        graph.set_node_attrs(
            "category",
            {"A": "primary", "B": "secondary", "C": "primary"}
        )
        ```
        
        See Also
        --------
        get_node_attrs : Get attribute values from nodes
        add_node : Add node with initial attributes
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
        
        Extracts a specific attribute from edges. Can optionally filter by source
        and/or target nodes. Edge keys in the result use the format "A->B" for
        directed graphs or "A--B" for undirected graphs.
        
        Parameters
        ----------
        edge_attr : str
            The name of the attribute to get.
        sources : list[int | str] | None, default None
            Optional list of source node IDs/labels to filter edges.
        targets : list[int | str] | None, default None
            Optional list of target node IDs/labels to filter edges.
        
        Returns
        -------
        dict[str, any]
            Dictionary mapping edge strings (e.g., "A->B") to attribute values.
            Only includes edges that have the attribute set.
        
        Examples
        --------
        Get attribute from all edges:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_edge("A", "B", weight=1.5, type="friend")
            .add_edge("B", "C", weight=2.0, type="colleague")
            .add_edge("C", "D", weight=1.0, type="friend")
        )
        
        weights = graph.get_edge_attrs("weight")
        print(weights)  # {'A->B': 1.5, 'B->C': 2.0, 'C->D': 1.0}
        
        types = graph.get_edge_attrs("type")
        print(types)  # {'A->B': 'friend', 'B->C': 'colleague', ...}
        ```
        
        Filter by source and target nodes:
        
        ```python
        filtered = graph.get_edge_attrs(
            "weight",
            sources=["A", "B"],
            targets=["B", "C"]
        )
        print(filtered)  # {'A->B': 1.5, 'B->C': 2.0}
        ```
        
        See Also
        --------
        get_edge_attrs_by_id : Get attributes using edge IDs
        set_edge_attrs : Set edge attribute values
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
        
        Similar to get_edge_attrs but uses internal edge IDs instead of node
        pairs. Returns a dictionary keyed by integer edge IDs. Useful when
        working directly with edge ID references.
        
        Parameters
        ----------
        attr : str
            Attribute name to extract.
        edges : list[int] | None, default None
            Optional list of edge IDs to query. If None, queries all edges.
        
        Returns
        -------
        dict[int, any]
            Dictionary mapping edge IDs to attribute values. Only includes edges
            that have the attribute set.
        
        Examples
        --------
        Get attributes by edge ID:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B", weight=1.5, type="friend")
            .add_edge("B", "C", weight=2.0, type="colleague")
        )
        
        edge_ids = graph.edge_ids()
        weights_by_id = graph.get_edge_attrs_by_id("weight", edges=edge_ids[:1])
        print(weights_by_id)  # {0: 1.5} or similar (edge ID varies)
        ```
        
        Get all edge types:
        
        ```python
        all_types = graph.get_edge_attrs_by_id("type")
        # Maps each edge ID to its type value
        ```
        
        See Also
        --------
        get_edge_attrs : Get attributes using node pairs
        edge_ids : Get all edge IDs
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
        
        Updates a specific attribute on edges. Can target edges by explicit
        (source, target) pairs, by filtering via source/target node lists, or
        apply to all edges. Supports various value input formats.
        
        Parameters
        ----------
        edge_attr : str
            The name of the attribute to set.
        values : any | list | dict
            The value(s) to set. Can be:
            - Single value: Applied to all targeted edges
            - List: Must match length of targeted edges (paired by position)
            - Dict: Maps edge IDs to values
        edges : list[tuple[int | str, int | str]] | None, default None
            Optional list of (source, target) tuples specifying edges to update.
        sources : list[int | str] | None, default None
            Optional list of source node IDs/labels to filter edges (use with targets).
        targets : list[int | str] | None, default None
            Optional list of target node IDs/labels to filter edges (use with sources).
        
        Returns
        -------
        Graph
            Self for method chaining.
        
        Raises
        ------
        ValueError
            If values is a list and length doesn't match targeted edges length.
        
        Examples
        --------
        Set attribute for specific edges:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .set_edge_attrs("weight", 2.5, edges=[("A", "B"), ("B", "C")])
        )
        
        weights = graph.get_edge_attrs("weight")
        print(weights)  # {'A--B': 2.5, 'B--C': 2.5}
        ```
        
        Set same value for all edges:
        
        ```python
        graph.set_edge_attrs("status", "active")
        
        statuses = graph.get_edge_attrs("status")
        # All edges have status='active'
        ```
        
        Filter by source and target nodes:
        
        ```python
        graph.set_edge_attrs(
            "type",
            "connection",
            sources=["A", "B"],
            targets=["B", "C"]
        )
        ```
        
        Set different values using a list:
        
        ```python
        edge_list = [("A", "B"), ("B", "C")]
        graph.set_edge_attrs("priority", [1, 2], edges=edge_list)
        ```
        
        See Also
        --------
        get_edge_attrs : Get edge attribute values
        add_edge : Add edge with initial attributes
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
        
        Exports all nodes and their attributes to a Polars DataFrame for tabular
        analysis. Each row represents a node, with columns for the node ID and
        any attributes set on nodes. Requires the `polars` package.
        
        Parameters
        ----------
        include_degree : bool, default True
            If True, adds a 'degree' column with the degree of each node.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - 'id': Node ID (str)
            - Attribute columns: One per unique attribute across all nodes
            - 'degree': Node degree (int, if include_degree=True)
            
            Empty DataFrame if graph has no nodes.
        
        Examples
        --------
        Export nodes to DataFrame:
        
        ```python
        graph = (
            Graph()
            .add_node(label="A", type="user", score=95)
            .add_node(label="B", type="admin", score=88)
            .add_node(label="C", type="user", score=92)
        )
        
        df = graph.get_node_df()
        print(df.columns)  # ['id', 'score', 'type', 'degree']
        print(df.shape)    # (3, 4)
        ```
        
        Analyze node attributes:
        
        ```python
        # Calculate statistics on node attributes
        avg_score = df["score"].mean()
        user_count = df.filter(pl.col("type") == "user").height
        
        # Find high-degree nodes
        hubs = df.filter(pl.col("degree") > 5)
        ```
        
        Exclude degree column:
        
        ```python
        df_no_degree = graph.get_node_df(include_degree=False)
        ```
        
        Notes
        -----
        - Attributes not present on a node will have `None` values
        - Columns are sorted alphabetically (except 'id' and 'degree')
        - Requires `polars` package: `pip install polars`
        
        See Also
        --------
        get_edge_df : Get edges as DataFrame
        get_node_df_ws : Get selected nodes as DataFrame
        get_node_attrs : Get specific attribute across nodes
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
        
        Exports all edges and their attributes to a Polars DataFrame for tabular
        analysis. Each row represents an edge, with columns for the edge ID,
        endpoints, and any attributes. Requires the `polars` package.
        
        Parameters
        ----------
        include_endpoints : bool, default True
            If True, includes 'source' and 'target' columns with node IDs.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - 'id': Edge ID (int)
            - 'source': Source node ID (str, if include_endpoints=True)
            - 'target': Target node ID (str, if include_endpoints=True)
            - Attribute columns: One per unique attribute across all edges
            - 'weight': Edge weight (float, if any edge has weight set)
            
            Empty DataFrame if graph has no edges.
        
        Examples
        --------
        Export edges to DataFrame:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_edge("A", "B", weight=1.5, type="friend")
            .add_edge("B", "C", weight=2.0, type="colleague")
            .add_edge("C", "D", weight=1.0, type="friend")
        )
        
        df = graph.get_edge_df()
        print(df.columns)  # ['id', 'source', 'target', 'type', 'weight']
        print(df.shape)    # (3, 5)
        ```
        
        Analyze edge properties:
        
        ```python
        # Filter edges by weight
        strong_edges = df.filter(pl.col("weight") > 1.5)
        
        # Count edge types
        type_counts = df.group_by("type").count()
        
        # Average weight per source node
        avg_by_source = df.group_by("source").agg(pl.col("weight").mean())
        ```
        
        Exclude endpoint columns:
        
        ```python
        df_no_endpoints = graph.get_edge_df(include_endpoints=False)
        ```
        
        Notes
        -----
        - Attributes not present on an edge will have `None` values
        - Weight is treated as a special attribute column
        - Columns are sorted alphabetically (except 'id', 'source', 'target')
        - Requires `polars` package: `pip install polars`
        
        See Also
        --------
        get_node_df : Get nodes as DataFrame
        get_edge_df_ws : Get selected edges as DataFrame
        get_edge_attrs : Get specific attribute across edges
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
        
        Similar to get_node_df but only exports currently selected nodes. Useful
        for analyzing subsets of nodes after filtering with selection methods.
        Requires the `polars` package.
        
        Parameters
        ----------
        include_degree : bool, default True
            If True, adds a 'degree' column with the degree of each node.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - 'id': Node ID (str)
            - Attribute columns: One per unique attribute across selected nodes
            - 'degree': Node degree (int, if include_degree=True)
            
            Empty DataFrame if no nodes are selected.
        
        Examples
        --------
        Export filtered nodes:
        
        ```python
        graph = (
            Graph()
            .add_star_graph(10, center_label="hub")
            .add_path_graph(5)
            .select_nodes_by_degree(min_degree=3)
        )
        
        df = graph.get_node_df_ws()
        # Only high-degree nodes in DataFrame
        print(df.shape[0])  # Number of selected nodes
        ```
        
        Combine selection with DataFrame analysis:
        
        ```python
        graph.select_nodes_by_id(["A", "B", "C"])
        selected_df = graph.get_node_df_ws()
        
        # Analyze just the selected nodes
        avg_degree = selected_df["degree"].mean()
        ```
        
        Notes
        -----
        - Returns empty DataFrame if no nodes are selected
        - Use selection methods before calling this
        - Same behavior as get_node_df but filtered to selection
        
        See Also
        --------
        get_node_df : Get all nodes as DataFrame
        get_edge_df_ws : Get selected edges as DataFrame
        get_selected_nodes : Get list of selected node IDs
        select_nodes_by_id : Select specific nodes
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
        
        Similar to get_edge_df but only exports currently selected edges. Useful
        for analyzing subsets of edges after filtering with selection methods.
        Requires the `polars` package.
        
        Parameters
        ----------
        include_endpoints : bool, default True
            If True, includes 'source' and 'target' columns with node IDs.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - 'id': Edge ID (int)
            - 'source': Source node ID (str, if include_endpoints=True)
            - 'target': Target node ID (str, if include_endpoints=True)
            - Attribute columns: One per unique attribute across selected edges
            - 'weight': Edge weight (float, if any selected edge has weight)
            
            Empty DataFrame if no edges are selected.
        
        Examples
        --------
        Export filtered edges:
        
        ```python
        graph = (
            Graph()
            .add_edge("A", "B", weight=1.5)
            .add_edge("B", "C", weight=0.5)
            .add_edge("C", "D", weight=2.0)
            .select_edges(lambda e: e.weight and e.weight > 1.0)
        )
        
        df = graph.get_edge_df_ws()
        # Only high-weight edges in DataFrame
        print(df["weight"].mean())
        ```
        
        Combine selection with DataFrame analysis:
        
        ```python
        edge_ids = graph.edge_ids()[:5]
        graph.select_edges_by_id(edge_ids)
        
        selected_df = graph.get_edge_df_ws()
        # Analyze just the first 5 edges
        ```
        
        Notes
        -----
        - Returns empty DataFrame if no edges are selected
        - Use selection methods before calling this
        - Same behavior as get_edge_df but filtered to selection
        
        See Also
        --------
        get_edge_df : Get all edges as DataFrame
        get_node_df_ws : Get selected nodes as DataFrame
        get_selected_edges : Get list of selected edge IDs
        select_edges_by_id : Select specific edges
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
        
        Returns the neighboring nodes of a specified node, with control over
        directionality (for directed graphs) and optional predicate filtering.
        
        Parameters
        ---
        node_id : str
            The node ID to find neighbors for.
        mode : str, default 'all'
            Direction mode for neighbor lookup:
            - `'out'` : Only successors (outgoing edges)
            - `'in'` : Only predecessors (incoming edges)
            - `'all'` : Both successors and predecessors
        predicate : callable, optional
            Filter function that takes a node ID and returns `True` to include it
            in the results. If `None`, all neighbors are returned.
            
        Returns
        ---
        list of str
            List of neighbor node IDs matching the specified criteria.
            
        Raises
        ---
        ValueError
            If `mode` is not one of `'out'`, `'in'`, or `'all'`.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        # Directed graph example
        graph = ec.Graph(directed=True)
        graph.add_edge("A", "B", weight=1.0)
        graph.add_edge("C", "A", weight=2.0)
        
        # Get successors (outgoing neighbors)
        out_neighbors = graph.get_neighbors("A", mode="out")
        print(out_neighbors)  # ['B']
        
        # Get predecessors (incoming neighbors)
        in_neighbors = graph.get_neighbors("A", mode="in")
        print(in_neighbors)  # ['C']
        
        # Get all neighbors
        all_neighbors = graph.get_neighbors("A", mode="all")
        print(all_neighbors)  # ['B', 'C']
        ```
        
        Filtered neighbor lookup with predicate:
        
        ```python
        # Find neighbors with high degree
        high_degree_neighbors = graph.get_neighbors(
            "A",
            predicate=lambda n: graph.degree(n) > 2
        )
        ```
        
        See Also
        --------
        get_predecessors : Get incoming neighbors (directed only)
        get_successors : Get outgoing neighbors (directed only)
        get_common_neighbors : Get shared neighbors between two nodes
        neighbors : Get neighbors of a node (simpler interface)
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
        
        Returns the intersection of neighbor sets for two nodes, useful for finding
        shared connections or potential triangles in the graph.
        
        Parameters
        ---
        node_id1 : str
            First node ID.
        node_id2 : str
            Second node ID.
            
        Returns
        ---
        list of str
            List of node IDs that are neighbors to both `node_id1` and `node_id2`.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=False)
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")
        graph.add_edge("A", "D")
        graph.add_edge("B", "D")
        
        # Find nodes connected to both A and B
        common = graph.get_common_neighbors("A", "B")
        print(common)  # ['C', 'D']
        ```
        
        Triangle detection using common neighbors:
        
        ```python
        # Find all triangles involving node A
        for neighbor in graph.neighbors("A"):
            common = graph.get_common_neighbors("A", neighbor)
            for third_node in common:
                print(f"Triangle: A - {neighbor} - {third_node}")
        ```
        
        See Also
        --------
        get_neighbors : Get neighbors of a single node
        get_similarity : Measure neighbor overlap between nodes
        """
        neighbors1 = set(self.get_neighbors(node_id1, mode="all"))
        neighbors2 = set(self.get_neighbors(node_id2, mode="all"))
        return list(neighbors1 & neighbors2)
    
    def get_predecessors(self, node_id: str) -> list[str]:
        """Get direct predecessors of a node (incoming neighbors).
        
        Returns nodes that have outgoing edges to the specified node. Only available
        for directed graphs.
        
        Parameters
        ---
        node_id : str
            Node ID to find predecessors for.
            
        Returns
        ---
        list of str
            List of predecessor node IDs (nodes with edges pointing to `node_id`).
            
        Raises
        ---
        ValueError
            If the graph is undirected.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")
        
        # Find nodes with edges pointing to C
        predecessors = graph.get_predecessors("C")
        print(predecessors)  # ['A', 'B']
        ```
        
        See Also
        --------
        get_successors : Get outgoing neighbors
        get_neighbors : Get neighbors with direction control
        in_degree : Get count of incoming edges
        """
        if not self._graph.is_directed:
            raise ValueError("get_predecessors only available for directed graphs")
        return self.get_neighbors(node_id, mode="in")
    
    def get_successors(self, node_id: str) -> list[str]:
        """Get direct successors of a node (outgoing neighbors).
        
        Returns nodes that are targets of outgoing edges from the specified node.
        Only available for directed graphs.
        
        Parameters
        ---
        node_id : str
            Node ID to find successors for.
            
        Returns
        ---
        list of str
            List of successor node IDs (nodes that `node_id` points to).
            
        Raises
        ---
        ValueError
            If the graph is undirected.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        
        # Find nodes that A points to
        successors = graph.get_successors("A")
        print(successors)  # ['B', 'C']
        ```
        
        See Also
        --------
        get_predecessors : Get incoming neighbors
        get_neighbors : Get neighbors with direction control
        out_degree : Get count of outgoing edges
        """
        if not self._graph.is_directed:
            raise ValueError("get_successors only available for directed graphs")
        return self.get_neighbors(node_id, mode="out")
    
    def get_non_neighbors(self, node_id: str) -> list[str]:
        """Get nodes that are NOT neighbors of the specified node.
        
        Returns all nodes in the graph except the specified node itself and its
        neighbors. Useful for finding potential new connections or disconnected nodes.
        
        Parameters
        ---
        node_id : str
            Node ID to find non-neighbors for.
            
        Returns
        ---
        list of str
            Sorted list of node IDs that are not neighbors of `node_id` and are not
            `node_id` itself.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_path_graph(5)  # 1 → 2 → 3 → 4 → 5
        
        # Node 2 is connected to 1 and 3
        # Non-neighbors are 4 and 5
        non_neighbors = graph.get_non_neighbors("2")
        print(non_neighbors)  # ['4', '5']
        ```
        
        Find potential new connections:
        
        ```python
        # Recommend connections for a node
        node = "A"
        candidates = graph.get_non_neighbors(node)
        
        # Score candidates by common neighbors
        scores = [
            (candidate, len(graph.get_common_neighbors(node, candidate)))
            for candidate in candidates
        ]
        ```
        
        See Also
        --------
        get_neighbors : Get neighbors of a node
        neighbors : Get neighbor list
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
        
        Returns neighbors whose numeric attribute values fall within a specified
        tolerance range of the starting node's value. Useful for finding connected
        nodes with comparable properties.
        
        Parameters
        ---
        node_id : str
            Starting node ID to compare against.
        node_attr : str
            Name of the numeric attribute to compare.
        tol_abs : tuple of (float, float), optional
            Absolute tolerance as `(lower_bound, upper_bound)` from the starting
            value. For example, `(1.0, 2.0)` means values from `start_value - 1.0`
            to `start_value + 2.0` are considered similar.
        tol_pct : tuple of (float, float), optional
            Percentage tolerance as `(lower_pct, upper_pct)` from the starting value.
            For example, `(10, 20)` means values from `start_value × 0.9` to
            `start_value × 1.2` are considered similar.
            
        Returns
        ---
        list of str
            List of neighbor node IDs with similar attribute values. Empty list if
            the starting node lacks the attribute or has no neighbors in range.
            
        Raises
        ---
        ValueError
            If the attribute exists but is not numeric (int or float).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_node("A", value=5.0)
        graph.add_node("B", value=5.5)
        graph.add_node("C", value=10.0)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        
        # Find neighbors within ±1.0 of A's value (5.0)
        similar = graph.get_similar_neighbors("A", "value", tol_abs=(1.0, 1.0))
        print(similar)  # ['B'] (5.5 is within range, 10.0 is not)
        ```
        
        Using percentage tolerance:
        
        ```python
        # Find neighbors within ±20% of A's value
        similar_pct = graph.get_similar_neighbors(
            "A",
            "value",
            tol_pct=(20, 20)
        )
        # Range: 4.0 to 6.0 (20% of 5.0)
        ```
        
        Notes
        -----
        - If neither `tol_abs` nor `tol_pct` is specified, only exact matches are returned.
        - Neighbors without the specified attribute are excluded from results.
        - Non-numeric attribute values raise a `ValueError`.
        
        See Also
        --------
        get_neighbors : Get all neighbors without filtering
        get_node_attrs : Get attributes for multiple nodes
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
        
        Uses breadth-first search (BFS) to find all nodes in the same connected
        component. For directed graphs, treats edges as undirected during traversal.
        
        Parameters
        ---
        node_id : str
            Starting node ID for the search.
            
        Returns
        ---
        list of str
            List of all reachable node IDs, excluding the starting node itself.
            Empty list if the node is isolated.
            
        Raises
        ---
        KeyError
            If `node_id` does not exist in the graph.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        
        # Create two separate components
        graph.add_path_graph(3)  # Component 1: 1-2-3
        graph.add_path_graph(2)  # Component 2: 4-5
        
        # Find all nodes reachable from node 1
        connected = graph.get_all_connected_nodes("1")
        print(connected)  # ['2', '3']
        
        # Node 4 is in a separate component
        connected_4 = graph.get_all_connected_nodes("4")
        print(connected_4)  # ['5']
        ```
        
        Component size analysis:
        
        ```python
        # Find size of each component
        visited = set()
        components = []
        
        for node in graph.node_ids():
            if node not in visited:
                component = graph.get_all_connected_nodes(node)
                component.append(node)  # Include starting node
                components.append(component)
                visited.update(component)
        
        print(f"Found {len(components)} components")
        ```
        
        See Also
        --------
        is_connected : Check if graph is fully connected
        count_weakly_connected_components : Count connected components
        get_neighbors : Get immediate neighbors only
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
            >>> G.add_path( 5)
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
        
        Updates or adds attributes to nodes in the current selection. Existing
        attributes are preserved unless explicitly overwritten. Requires an active
        node selection.
        
        Parameters
        ---
        **attributes : keyword arguments
            Attribute name-value pairs to set on selected nodes. Each keyword
            becomes an attribute name, and its value becomes the attribute value.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_node('A', value=10)
        graph.add_node('B', value=20)
        
        # Select nodes and set attributes
        (graph
            .select_nodes_by_id(['A', 'B'])
            .set_node_attrs_ws(color='red', importance=0.9)
        )
        
        # Check updated attributes
        print(graph.get_node('A').attributes['color'])  # 'red'
        print(graph.get_node('A').attributes['value'])  # 10 (preserved)
        ```
        
        Bulk attribute updates with filtering:
        
        ```python
        # Set category on high-value nodes
        (graph
            .select_nodes(lambda n: n.attributes.get('value', 0) > 15)
            .set_node_attrs_ws(category='high')
        )
        ```
        
        See Also
        --------
        set_node_attrs : Set attributes on specific nodes by ID
        select_nodes_by_id : Select nodes for bulk operations
        get_node_attrs : Get attributes from multiple nodes
        mutate_node_attrs_ws : Transform attributes with a function
        """
        selected = self.get_selected_nodes()
        
        for node_id in selected:
            for key, value in attributes.items():
                # Use Rust set_node_attr for persistence
                self._graph.set_node_attr(node_id, key, value)
        
        return self
    
    def set_edge_attrs_ws(self, **attributes) -> "Graph":
        """Set attributes on all selected edges.
        
        Updates or adds attributes to edges in the current selection. The `weight`
        attribute receives special handling as it's a first-class edge property.
        Requires an active edge selection.
        
        Parameters
        ---
        **attributes : keyword arguments
            Attribute name-value pairs to set on selected edges. Use `weight` to
            update edge weights directly.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_path( 3)
        
        # Select and update edge attributes
        (graph
            .select_edges_by_id([1])
            .set_edge_attrs_ws(weight=5.0, type='strong')
        )
        
        # Check updated attributes
        edge = graph.get_edge(1)
        print(edge.weight)  # 5.0
        print(edge.attributes['type'])  # 'strong'
        ```
        
        Bulk weight updates:
        
        ```python
        # Set all edge weights to 1.0
        (graph
            .select_edges(lambda e: True)  # Select all edges
            .set_edge_attrs_ws(weight=1.0)
        )
        ```
        
        See Also
        --------
        set_edge_attrs : Set attributes on specific edges by ID
        select_edges_by_id : Select edges for bulk operations
        get_edge_attrs : Get attributes from multiple edges
        mutate_edge_attrs_ws : Transform edge attributes with a function
        """
        selected = self.get_selected_edges()
        
        for edge_id in selected:
            for key, value in attributes.items():
                # Use Rust set_edge_attr for persistence (handles weight specially)
                self._graph.set_edge_attr(edge_id, key, value)
        
        return self
    
    def mutate_node_attrs_ws(self, attr: str, func: Callable) -> "Graph":
        """Transform a node attribute using a function on selected nodes.
        
        Applies a transformation function to the specified attribute for all selected
        nodes. The function receives each node's current attribute value and returns
        the new value. Requires an active node selection.
        
        Parameters
        ---
        attr : str
            Name of the attribute to transform.
        func : callable
            Transformation function that takes the current attribute value and
            returns the new value. Called once per selected node.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_node('A', score=10)
        graph.add_node('B', score=20)
        graph.add_node('C', score=30)
        
        # Double the score for selected nodes
        (graph
            .select_nodes_by_id(['A', 'B'])
            .mutate_node_attrs_ws('score', lambda x: x * 2)
        )
        
        print(graph.get_node('A').attributes['score'])  # 20
        print(graph.get_node('B').attributes['score'])  # 40
        print(graph.get_node('C').attributes['score'])  # 30 (unchanged)
        ```
        
        Conditional transformations:
        
        ```python
        # Apply discount to high-value nodes
        (graph
            .select_nodes(lambda n: n.attributes.get('price', 0) > 100)
            .mutate_node_attrs_ws('price', lambda p: p * 0.9)
        )
        ```
        
        Notes
        -----
        Nodes without the specified attribute are skipped.
        
        See Also
        --------
        set_node_attrs_ws : Set attributes directly without transformation
        mutate_edge_attrs_ws : Transform edge attributes
        get_node_attrs : Get attributes for inspection before transformation
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
        
        Applies a transformation function to the specified attribute for all selected
        edges. Supports transforming the `weight` field directly. Requires an active
        edge selection.
        
        Parameters
        ---
        attr : str
            Name of the attribute to transform. Can be `'weight'` to transform
            edge weights directly.
        func : callable
            Transformation function that takes the current attribute value and
            returns the new value. Called once per selected edge.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_edge('A', 'B', weight=1.5)
        graph.add_edge('B', 'C', weight=2.0)
        
        # Double the weight of selected edges
        (graph
            .select_edges_by_id([1])
            .mutate_edge_attrs_ws('weight', lambda x: x * 2)
        )
        
        print(graph.get_edge(1).weight)  # 3.0
        print(graph.get_edge(2).weight)  # 2.0 (unchanged)
        ```
        
        Normalize edge weights:
        
        ```python
        # Get max weight for normalization
        max_weight = max(e.weight for e in [graph.get_edge(eid) for eid in graph.edge_ids()] if e.weight)
        
        # Normalize all weights to [0, 1]
        (graph
            .select_edges(lambda e: e.weight is not None)
            .mutate_edge_attrs_ws('weight', lambda w: w / max_weight)
        )
        ```
        
        Notes
        -----
        Edges without the specified attribute are skipped.
        
        See Also
        --------
        set_edge_attrs_ws : Set attributes directly without transformation
        mutate_node_attrs_ws : Transform node attributes
        get_edge_attrs : Get edge attributes for inspection
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
        
        Adds edges to form a complete graph (clique) among all selected nodes.
        For directed graphs, creates edges in both directions. Skips edges that
        already exist.
        
        Parameters
        ---
        **edge_attributes : keyword arguments
            Optional attributes to set on newly created edges.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_node('C')
        
        # Create clique among selected nodes
        (graph
            .select_nodes_by_id(['A', 'B', 'C'])
            .fully_connect_nodes_ws(type='clique', weight=1.0)
        )
        
        # For undirected: 3 edges (A-B, A-C, B-C)
        print(graph.num_edges())  # 3
        ```
        
        Directed clique (bidirectional edges):
        
        ```python
        graph_directed = ec.Graph(directed=True)
        graph_directed.add_node('X')
        graph_directed.add_node('Y')
        
        (graph_directed
            .select_nodes_by_id(['X', 'Y'])
            .fully_connect_nodes_ws()
        )
        
        # Creates X→Y and Y→X
        print(graph_directed.num_edges())  # 2
        ```
        
        Notes
        -----
        - For `n` selected nodes in an undirected graph, creates up to `n(n-1)/2` edges.
        - For `n` selected nodes in a directed graph, creates up to `n(n-1)` edges.
        - Existing edges are not modified or duplicated.
        
        See Also
        --------
        add_full_graph : Create a complete graph from scratch
        select_nodes_by_id : Select nodes for clique formation
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
        
        For each selected edge (A → B), adds the reverse edge (B → A) if it
        doesn't already exist. Useful for making directed graphs bidirectional
        or adding reciprocal relationships.
        
        Parameters
        ---
        **edge_attributes : keyword arguments
            Optional attributes to set on the newly created reverse edges.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Raises
        ---
        ValueError
            If the graph is undirected.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_edge('A', 'B', weight=1.0)
        
        # Add reverse edge
        (graph
            .select_edges_by_id([1])
            .add_reverse_edges_ws(weight=1.0)
        )
        
        print(graph.num_edges())  # 2 (A→B and B→A)
        ```
        
        Make selected edges bidirectional:
        
        ```python
        # Add reverse edges for all edges with high weight
        (graph
            .select_edges(lambda e: e.weight and e.weight > 5.0)
            .add_reverse_edges_ws()
        )
        ```
        
        See Also
        --------
        rev_edge_dir_ws : Reverse direction of selected edges in place
        trav_reverse_edge : Traverse to reverse edges
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
            >>> G.add_node('A')
            >>> G.add_node('B')
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
        
        Creates a sequence of n nodes connected linearly: 1 → 2 → 3 → ... → n.
        End nodes have degree 1, internal nodes have degree 2.
        
        Parameters
        ----------
        n : int
            Number of nodes in the path.
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default=True
            If True, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Create a linear path:
        
        ```python
        graph = Graph().add_path_graph(n=5)
        
        print(graph.num_nodes())  # 5
        print(graph.num_edges())  # 4
        ```
        
        See Also
        --------
        add_cycle_graph : Add a cycle (path with loop back)
        add_star_graph : Add a hub-and-spoke graph
        """
        self.add_path( n, type=type, label=label, rel=rel)
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
        Each node has exactly degree 2.
        
        Parameters
        ----------
        n : int
            Number of nodes in the cycle (minimum 3).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default=True
            If True, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Create a cycle:
        
        ```python
        graph = Graph().add_cycle_graph(n=4)
        
        print(graph.num_nodes())  # 4
        print(graph.num_edges())  # 4
        ```
        
        See Also
        --------
        add_path_graph : Add a linear path (no cycle)
        add_full_graph : Add a complete graph
        """
        self.add_cycle( n, type=type, label=label, rel=rel)
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
        All edges connect the hub to spokes; no edges between spokes.
        
        Parameters
        ----------
        n : int
            Total number of nodes (minimum 2).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default=True
            If True, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Create a star graph:
        
        ```python
        graph = Graph().add_star_graph(n=5)  # 1 hub + 4 spokes
        
        print(graph.num_nodes())  # 5
        print(graph.num_edges())  # 4
        ```
        
        See Also
        --------
        add_full_graph : Add a complete graph (clique)
        add_path_graph : Add a linear path
        """
        self.add_star( n, type=type, label=label, rel=rel)
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
        In a complete graph with n nodes, there are n(n-1)/2 edges (without self-loops).
        
        Parameters
        ----------
        n : int
            Number of nodes.
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default=True
            If True, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
        keep_loops : bool, default=False
            If True, include self-loops (adding n more edges).
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Create a complete graph:
        
        ```python
        graph = Graph().add_full_graph(n=4)
        
        print(graph.num_nodes())  # 4
        print(graph.num_edges())  # 6
        ```
        
        See Also
        --------
        add_star_graph : Add a hub-and-spoke graph
        add_cycle_graph : Add a cycle
        """
        self.add_full_graph( n, type=type, label=label, rel=rel, keep_loops=keep_loops)
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
        Total nodes = (k^(h+1) - 1) / (k - 1).
        
        Parameters
        ----------
        k : int
            Number of children per node (branching factor).
        h : int
            Height of the tree (root has height 0).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default=True
            If True, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Create a binary tree:
        
        ```python
        graph = (
            Graph(directed=True)
            .add_balanced_tree(k=2, h=3)
        )
        
        print(graph.num_nodes())  # 15 (1 + 2 + 4 + 8)
        ```
        
        See Also
        --------
        add_star_graph : Add a hub-and-spoke graph
        add_path_graph : Add a linear path
        """
        self.add_balanced_tree( k, h, type=type, label=label, rel=rel)
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
        
        Creates a rectangular grid where each interior node is connected to its
        four neighbors (up, down, left, right). Edge nodes have fewer connections.
        
        Parameters
        ---
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_grid_2d(3, 4)  # 3×4 grid
        
        print(graph.num_nodes())  # 12
        print(graph.num_edges())  # 17 (edges: 3*3 + 2*4)
        ```
        
        See Also
        --------
        add_balanced_tree : Add a tree structure
        add_cycle_graph : Add a cycle
        """
        self.add_grid_2d( rows, cols, type=type, label=label, rel=rel)
        return self
    
    def add_prism_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a prism graph to the current graph.
        
        Creates two parallel n-cycles connected by n vertical edges, forming
        a 3D prism structure when visualized.
        
        Parameters
        ---
        n : int
            Number of nodes in each cycle (minimum 3).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_prism_graph(5)
        
        print(graph.num_nodes())  # 10 (2 cycles of 5)
        print(graph.num_edges())  # 15 (5+5 cycle edges + 5 vertical)
        ```
        
        See Also
        --------
        add_cycle_graph : Add a single cycle
        add_grid_2d : Add a 2D grid
        """
        self.add_prism( n, type=type, label=label, rel=rel)
        return self
    
    def add_gnp_graph(
        self,
        n: int,
        p: float,
        directed: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "Graph":
        """Add an Erdős-Rényi G(n,p) random graph to the current graph.
        
        Each possible edge is included independently with probability p. Uses fast
        Rust implementation for optimal performance. This model is useful for
        generating random graphs with a specific edge density.
        
        Parameters
        ---
        n : int
            Number of nodes to add.
        p : float
            Probability of edge creation, in range [0.0, 1.0].
        directed : bool, optional
            If provided and differs from graph's directed setting, creates a
            temporary graph with the specified directedness.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph()
        graph.add_gnp_graph(n=50, p=0.1, seed=42)
        
        print(graph.num_nodes())  # 50
        # Expected ~245 edges (50*49/2 * 0.1)
        ```
        
        See Also
        --------
        add_gnm_graph : Add random graph with fixed edge count
        add_pa_graph : Add preferential attachment graph
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
        
        Creates a graph with n nodes and exactly m randomly placed edges.
        Uses fast Rust implementation for optimal performance. This model
        is useful for generating random graphs with a fixed number of edges.
        
        Parameters
        ----------
        n : int
            Number of nodes to add.
        m : int
            Number of edges to create.
        directed : bool, optional
            If provided and differs from graph's directed setting, creates
            temporary graph with specified directedness.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Generate a random graph:
        
        ```python
        graph = (
            Graph()
            .add_gnm_graph(n=50, m=100, seed=42)
        )
        
        print(graph.num_nodes())  # 50
        print(graph.num_edges())  # 100
        ```
        
        See Also
        --------
        add_pa_graph : Add preferential attachment graph
        add_ws_graph : Add Watts-Strogatz small-world graph
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
        
        Grows a graph by adding nodes one at a time, with each new node connecting
        to m existing nodes with probability proportional to degree. This creates
        scale-free networks with power-law degree distributions. Uses fast Rust
        implementation for optimal performance.
        
        Parameters
        ----------
        n : int
            Number of nodes to add.
        m : int, default=2
            Number of edges to attach from each new node.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Generate a scale-free network:
        
        ```python
        graph = (
            Graph()
            .add_pa_graph(n=100, m=3, seed=42)
        )
        
        print(graph.num_nodes())  # 100
        print(graph.num_edges())  # (100-m)*m = 294
        ```
        
        See Also
        --------
        add_gnm_graph : Add Erdős-Rényi random graph
        add_ws_graph : Add Watts-Strogatz small-world graph
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
        
        Creates a ring lattice with rewiring probability p, producing graphs
        with high clustering and short path lengths. Small-world networks model
        many real-world phenomena. Uses fast Rust implementation for optimal
        performance.
        
        Parameters
        ----------
        n : int
            Number of nodes to add.
        k : int, default=6
            Each node connected to k nearest neighbors (must be even).
        p : float, default=0.1
            Probability of rewiring each edge (0.0 to 1.0).
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        Graph
            Self for method chaining.
        
        Examples
        --------
        Generate a small-world network:
        
        ```python
        graph = (
            Graph()
            .add_smallworld_graph(n=100, k=6, p=0.1, seed=42)
        )
        
        print(graph.num_nodes())  # 100
        ```
        
        See Also
        --------
        add_gnm_graph : Add Erdős-Rényi random graph
        add_pa_graph : Add preferential attachment graph
        add_ws_graph : Alias for this method
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
        
        From the currently selected edges, finds the reverse edges. For each
        selected edge A→B, looks for the corresponding edge B→A. Only works
        with directed graphs.
        
        Parameters
        ---
        add_to_selection : bool, default False
            If `True`, adds reverse edges to the current selection. If `False`,
            replaces the selection with reverse edges only.
        
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")  # Reverse edge
        
        # Select A→B edge
        graph.select_edges_by_node_id("A", "B")
        
        # Traverse to reverse edge (B→A)
        graph.trav_reverse_edge()
        edges = graph.get_selected_edges()
        # Now selects only the B→A edge
        ```
        
        Add reverse edges to selection:
        
        ```python
        # Select both an edge and its reverse
        (graph
            .select_edges_by_id([1])
            .trav_reverse_edge(add_to_selection=True)
        )
        # Selection now includes both A→B and B→A
        ```
        
        See Also
        --------
        add_reverse_edges_ws : Add new reverse edges
        rev_edge_dir_ws : Reverse edge directions in place
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
        `max_steps` is reached. The traversal proceeds in a breadth-first manner.
        
        Parameters
        ---
        conditions : callable
            A function that takes a node ID (str) and returns `True` if the stopping
            condition is met for that node. The callable can access node attributes
            via `graph.get_node(node_id).attributes`.
        max_steps : int, default 30
            Maximum number of traversal steps (node-to-node hops) before stopping.
        exclude_unmatched : bool, default True
            If `True`, only nodes satisfying the conditions are kept in the final
            selection. If `False`, all reached nodes are kept.
        add_to_selection : bool, default False
            If `True`, all traversed nodes are included in the final selection.
            If `False`, only the final nodes are selected.
        
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_path_graph(10)
        
        # Set value attributes
        for i in range(1, 11):
            graph.set_node_attrs("value", {str(i): i})
        
        # Traverse from node 1 until reaching node with value 8
        (graph
            .select_nodes_by_id(["1"])
            .trav_out_until(lambda nid: graph.get_node(nid).attributes.get("value") == 8)
        )
        
        print(graph.get_selected_nodes())  # ['8']
        ```
        
        Multiple starting points:
        
        ```python
        # Traverse from multiple nodes
        (graph
            .clear_selection()
            .select_nodes_by_id(["1", "2"])
            .trav_out_until(
                lambda nid: graph.get_node(nid).attributes.get("value", 0) >= 5,
                max_steps=10
            )
        )
        # Returns first nodes ≥ 5 from each starting point
        ```
        
        See Also
        --------
        trav_in_until : Traverse inward with conditions
        trav_out : Simple outward traversal (one step)
        select_nodes : Select nodes with conditions
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
        `max_steps` is reached. The traversal proceeds in a breadth-first manner.
        
        Parameters
        ---
        conditions : callable
            A function that takes a node ID (str) and returns `True` if the stopping
            condition is met for that node. The callable can access node attributes
            via `graph.get_node(node_id).attributes`.
        max_steps : int, default 30
            Maximum number of traversal steps (node-to-node hops) before stopping.
        exclude_unmatched : bool, default True
            If `True`, only nodes satisfying the conditions are kept in the final
            selection. If `False`, all reached nodes are kept.
        add_to_selection : bool, default False
            If `True`, all traversed nodes are included in the final selection.
            If `False`, only the final nodes are selected.
        
        Returns
        ---
        Graph
            Self for method chaining.
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = ec.Graph(directed=True)
        graph.add_path_graph(10)
        
        # Set value attributes
        for i in range(1, 11):
            graph.set_node_attrs("value", {str(i): i})
        
        # Traverse from node 10 inward until reaching node with value 1
        (graph
            .select_nodes_by_id(["10"])
            .trav_in_until(lambda nid: graph.get_node(nid).attributes.get("value") == 1)
        )
        
        print(graph.get_selected_nodes())  # ['1']
        ```
        
        Multiple stopping conditions:
        
        ```python
        # Find nodes with value in specific set
        (graph
            .clear_selection()
            .select_nodes_by_id(["6", "12"])
            .trav_in_until(
                lambda nid: graph.get_node(nid).attributes.get("value", 0) in [1, 2, 10]
            )
        )
        # Returns nodes with value in [1, 2, 10]
        ```
        
        See Also
        --------
        trav_out_until : Traverse outward with conditions
        trav_in : Simple inward traversal (one step)
        select_nodes : Select nodes with conditions
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
    
    # Graph Builder Methods
    
    def add_path(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a path graph to this graph.
        
        A path graph is a sequence of nodes connected linearly: 1 → 2 → 3 → ... → n.
        End nodes have degree 1, internal nodes have degree 2.
        
        Parameters
        ---
        n : int
            Number of nodes in the path.
        type : str, optional
            Optional node type/category for all nodes in path.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges in path.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=True)
            .add_path(n=5, type="step", rel="next")
        )
        
        print(graph.num_nodes())  # 5
        print(graph.num_edges())  # 4
        
        # Chain multiple paths
        graph = (
            ec.Graph()
            .add_path(n=3)
            .add_path(n=4)
        )
        print(graph.num_nodes())  # 7
        ```
        
        See Also
        --------
        add_cycle : Add a cycle graph (closed loop)
        add_star : Add a star graph
        """
        if n < 1:
            raise ValueError("Path must have at least 1 node")
        
        # Get starting node ID (continue from existing nodes)
        start_id = self.num_nodes() + 1
        
        # Add nodes
        for i in range(n):
            node_id = str(start_id + i)
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = node_id
            self.add_node(node_id, **attrs)
        
        # Add edges (path has n-1 edges)
        for i in range(n - 1):
            source = str(start_id + i)
            target = str(start_id + i + 1)
            edge_attrs = {}
            if rel:
                edge_attrs['rel'] = rel
            self.add_edge(source, target, **edge_attrs)
        
        return self
    
    def add_cycle(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a cycle graph to this graph.
        
        A cycle graph is a closed loop of nodes: 1 → 2 → 3 → ... → n → 1.
        Each node has exactly degree 2.
        
        Parameters
        ---
        n : int
            Number of nodes in the cycle (minimum 3).
        type : str, optional
            Optional node type/category for all nodes in cycle.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges in cycle.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=True)
            .add_cycle(n=4, type="state", rel="transitions_to")
        )
        
        print(graph.num_nodes())  # 4
        print(graph.num_edges())  # 4
        
        # Combine structures
        graph = (
            ec.Graph()
            .add_cycle(n=5)
            .add_path(n=3)
        )
        ```
        
        See Also
        --------
        add_path : Add a path graph (linear, no cycle)
        add_full_graph : Add a complete graph
        """
        if n < 3:
            raise ValueError("Cycle must have at least 3 nodes")
        
        # Add path first
        self.add_path(n, type=type, label=label, rel=rel)
        
        # Close the cycle with final edge
        start_id = self.num_nodes() - n + 1
        end_id = self.num_nodes()
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        self.add_edge(str(end_id), str(start_id), **edge_attrs)
        
        return self
    
    def add_star(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a star graph to this graph.
        
        A star graph has one central node connected to n-1 peripheral nodes.
        The center node has degree n-1, all peripheral nodes have degree 1.
        
        Parameters
        ---
        n : int
            Total number of nodes (including center, minimum 2).
        type : str, optional
            Optional node type/category for all nodes in star.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges from center.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=False)
            .add_star(n=6, type="network", rel="connected_to")
        )
        
        print(graph.num_nodes())  # 6
        print(graph.num_edges())  # 5
        ```
        
        See Also
        --------
        add_full_graph : Add a complete graph (all-to-all connections)
        add_path : Add a linear path
        """
        if n < 2:
            raise ValueError("Star must have at least 2 nodes")
        
        start_id = self.num_nodes() + 1
        center_id = str(start_id)
        
        # Add all nodes
        for i in range(n):
            node_id = str(start_id + i)
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = node_id
            self.add_node(node_id, **attrs)
        
        # Add edges from center to all other nodes
        for i in range(1, n):
            peripheral_id = str(start_id + i)
            edge_attrs = {}
            if rel:
                edge_attrs['rel'] = rel
            self.add_edge(center_id, peripheral_id, **edge_attrs)
        
        return self
    
    def add_full_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
        keep_loops: bool = False,
    ) -> "Graph":
        """Add a complete graph (clique) to this graph.
        
        A complete graph has edges between every pair of nodes. In a complete graph
        with n nodes, there are n(n-1)/2 edges for undirected or n(n-1) for directed.
        
        Parameters
        ---
        n : int
            Number of nodes in the complete graph.
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
        keep_loops : bool, default False
            If `True`, include self-loops (edges from node to itself).
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=False)
            .add_full_graph(n=4, type="complete")
        )
        
        print(graph.num_nodes())  # 4
        print(graph.num_edges())  # 6 (for undirected: n*(n-1)/2)
        ```
        
        See Also
        --------
        add_star : Add a hub-and-spoke graph
        add_cycle : Add a cycle
        """
        if n < 1:
            raise ValueError("Complete graph must have at least 1 node")
        
        start_id = self.num_nodes() + 1
        
        # Add all nodes
        for i in range(n):
            node_id = str(start_id + i)
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = node_id
            self.add_node(node_id, **attrs)
        
        # Add edges between all pairs
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        
        for i in range(n):
            for j in range(n):
                source_id = str(start_id + i)
                target_id = str(start_id + j)
                
                # Skip self-loops unless explicitly requested
                if i == j and not keep_loops:
                    continue
                
                # For undirected graphs, avoid duplicate edges
                if not self.is_directed and i > j:
                    continue
                
                self.add_edge(source_id, target_id, **edge_attrs)
        
        return self
    
    def add_balanced_tree(
        self,
        k: int,
        h: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a balanced tree to this graph.
        
        A balanced k-ary tree of height h where each non-leaf node has exactly k children.
        Total nodes = (k^(h+1) - 1) / (k - 1) for k > 1.
        
        Parameters
        ---
        k : int
            Branching factor (number of children per node, minimum 1).
        h : int
            Height of the tree (root is at height 0).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=True)
            .add_balanced_tree(k=2, h=3, type="tree_node", rel="parent_of")
        )
        
        print(graph.num_nodes())  # 15 (1 + 2 + 4 + 8)
        ```
        
        See Also
        --------
        add_star : Add a hub-and-spoke graph
        add_path : Add a linear path
        """
        if k < 1:
            raise ValueError("Branching factor k must be at least 1")
        if h < 0:
            raise ValueError("Height h must be non-negative")
        
        start_id = self.num_nodes() + 1
        
        # First, add all nodes
        total_nodes = (k**(h + 1) - 1) // (k - 1) if k > 1 else h + 1
        for i in range(total_nodes):
            node_id = start_id + i
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = str(node_id)
            self.add_node(str(node_id), **attrs)
        
        # Then, add edges level by level using BFS
        nodes_at_level = [start_id]  # Start with root
        node_counter = 1  # Next node to assign as child
        
        for level in range(h):
            next_level = []
            
            for node_id in nodes_at_level:
                # Add children
                for _ in range(k):
                    child_id = start_id + node_counter
                    node_counter += 1
                    next_level.append(child_id)
                    
                    # Add edge from parent to child
                    edge_attrs = {}
                    if rel:
                        edge_attrs['rel'] = rel
                    self.add_edge(str(node_id), str(child_id), **edge_attrs)
            
            nodes_at_level = next_level
        
        return self
    
    def add_prism(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a prism graph to this graph.
        
        A prism graph consists of two parallel n-cycles connected by matching edges.
        It's like taking two cycle graphs and connecting corresponding nodes. Total
        edges = 3n (n in each cycle + n connecting edges).
        
        Parameters
        ---
        n : int
            Number of nodes in each cycle (minimum 3).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their IDs.
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=False)
            .add_prism(n=4, type="prism_node")
        )
        
        print(graph.num_nodes())  # 8 (2 cycles of 4 nodes each)
        print(graph.num_edges())  # 12 (4 + 4 + 4 edges)
        ```
        
        See Also
        --------
        add_cycle : Add a single cycle
        add_grid_2d : Add a 2D grid
        """
        if n < 3:
            raise ValueError("Prism must have at least 3 nodes per cycle")
        
        start_id = self.num_nodes() + 1
        
        # Add first cycle
        cycle1_start = start_id
        for i in range(n):
            node_id = str(cycle1_start + i)
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = f"{node_id}_top"
            self.add_node(node_id, **attrs)
        
        # Connect first cycle
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        for i in range(n):
            source = str(cycle1_start + i)
            target = str(cycle1_start + ((i + 1) % n))
            self.add_edge(source, target, **edge_attrs)
        
        # Add second cycle
        cycle2_start = cycle1_start + n
        for i in range(n):
            node_id = str(cycle2_start + i)
            attrs = {}
            if type:
                attrs['type'] = type
            if label:
                attrs['label'] = f"{node_id}_bottom"
            self.add_node(node_id, **attrs)
        
        # Connect second cycle
        for i in range(n):
            source = str(cycle2_start + i)
            target = str(cycle2_start + ((i + 1) % n))
            self.add_edge(source, target, **edge_attrs)
        
        # Connect corresponding nodes between cycles
        for i in range(n):
            top_node = str(cycle1_start + i)
            bottom_node = str(cycle2_start + i)
            self.add_edge(top_node, bottom_node, **edge_attrs)
        
        return self
    
    def add_grid_2d(
        self,
        x: int,
        y: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """Add a 2D grid graph to this graph.
        
        Creates a rectangular lattice where each node is connected to its orthogonal
        neighbors (up, down, left, right). Total edges = (x-1)·y + x·(y-1).
        
        Parameters
        ---
        x : int
            Width of the grid (minimum 1).
        y : int
            Height of the grid (minimum 1).
        type : str, optional
            Optional node type/category for all nodes.
        label : bool, default True
            If `True`, label nodes with their grid coordinates "(i,j)".
        rel : str, optional
            Optional relationship label for all edges.
            
        Returns
        ---
        Graph
            The graph object (for method chaining).
            
        Examples
        --------
        ```python
        import edgecase as ec
        
        graph = (
            ec.Graph(directed=False)
            .add_grid_2d(x=3, y=3, type="cell")
        )
        
        print(graph.num_nodes())  # 9 (3 × 3 grid)
        print(graph.num_edges())  # 12 ((3-1)·3 + 3·(3-1))
        
        # Access node at position (1,2)
        nodes = graph.node_df()
        node = nodes.filter((nodes["x"] == 1) & (nodes["y"] == 2))
        ```
        
        See Also
        --------
        add_path : Add a linear path
        add_cycle : Add a cycle
        add_prism : Add a prism structure
        """
        if x < 1 or y < 1:
            raise ValueError("Grid dimensions must be at least 1x1")
        
        start_id = self.num_nodes() + 1
        
        # Create mapping from (i, j) to node_id
        node_map = {}
        node_counter = 0
        
        # Add all nodes
        for j in range(y):
            for i in range(x):
                node_id = str(start_id + node_counter)
                node_map[(i, j)] = node_id
                
                attrs = {}
                if type:
                    attrs['type'] = type
                if label:
                    attrs['label'] = f"({i},{j})"
                attrs['x'] = i
                attrs['y'] = j
                
                self.add_node(node_id, **attrs)
                node_counter += 1
        
        # Add edges to neighbors
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        
        for j in range(y):
            for i in range(x):
                current = node_map[(i, j)]
                
                # Connect to right neighbor
                if i + 1 < x:
                    right = node_map[(i + 1, j)]
                    self.add_edge(current, right, **edge_attrs)
                
                # Connect to bottom neighbor
                if j + 1 < y:
                    bottom = node_map[(i, j + 1)]
                    self.add_edge(current, bottom, **edge_attrs)
        
        return self
    
    @classmethod
    def from_dataframes(
        cls,
        nodes_df: Optional[pl.DataFrame] = None,
        edges_df: Optional[pl.DataFrame] = None,
        directed: bool = False,
    ) -> "Graph":
        """Create a graph from Polars DataFrames following an NDF/EDF pattern.
        
        Constructs a graph from structured tabular data. Node DataFrame (NDF) must
        have 'id' column; Edge DataFrame (EDF) must have 'from'/'to' columns. Additional
        columns become node/edge attributes.
        
        Parameters
        ---
        nodes_df : polars.DataFrame, optional
            Node data with required 'id' column plus optional 'type', 'label', and
            custom attribute columns.
        edges_df : polars.DataFrame, optional
            Edge data with required 'from'/'to' columns plus optional 'id', 'rel',
            and custom attribute columns. If 'id' omitted, auto-assigned.
        directed : bool, default False
            Whether to create a directed graph.
            
        Returns
        ---
        Graph
            A new Graph object with nodes/edges from the DataFrames.
            
        Examples
        --------
        ```python
        import edgecase as ec
        import polars as pl
        
        nodes = pl.DataFrame({
            'id': ['A', 'B', 'C'],
            'type': ['user', 'user', 'admin'],
            'label': ['Alice', 'Bob', 'Charlie']
        })
        edges = pl.DataFrame({
            'id': [1, 2],
            'from': ['A', 'B'],
            'to': ['B', 'C'],
            'rel': ['follows', 'follows']
        })
        
        graph = ec.Graph.from_dataframes(nodes_df=nodes, edges_df=edges, directed=True)
        print(graph.num_nodes())  # 3
        print(graph.num_edges())  # 2
        ```
        
        See Also
        --------
        add_node : Add individual nodes
        add_edge : Add individual edges
        node_df : Extract node data as DataFrame
        edge_df : Extract edge data as DataFrame
        """
        g = cls(directed=directed)
        
        # Add nodes if provided
        if nodes_df is not None:
            if 'id' not in nodes_df.columns:
                raise ValueError("nodes_df must have an 'id' column")
            
            for row in nodes_df.iter_rows(named=True):
                node_id = str(row['id'])
                attrs = {k: v for k, v in row.items() if k != 'id' and v is not None}
                g.add_node(node_id, **attrs)
        
        # Add edges if provided
        if edges_df is not None:
            if 'from' not in edges_df.columns or 'to' not in edges_df.columns:
                raise ValueError("edges_df must have 'from' and 'to' columns")
            
            for row in edges_df.iter_rows(named=True):
                source = str(row['from'])
                target = str(row['to'])
                
                # Extract edge_id if provided
                edge_id = row.get('id', None)
                
                # Separate edge attributes from special columns
                attrs = {k: v for k, v in row.items() 
                        if k not in ['id', 'from', 'to'] and v is not None}
                
                # Ensure nodes exist
                if not g.has_node(source):
                    g.add_node(source)
                if not g.has_node(target):
                    g.add_node(target)
                
                g.add_edge(source, target, edge_id=edge_id, **attrs)
        
        return g
    
    def __repr__(self) -> str:
        return self._graph.__repr__()


# Re-export Node and Edge from Rust for direct access if needed
Node = rust.Node
Edge = rust.Edge


# Algorithm functions - wrap Rust functions to accept our Graph wrapper
def shortest_path(graph: Graph, source: Union[int, str], target: Union[int, str]) -> list[str] | None:
    """Find shortest path between two nodes.
    
    Uses breadth-first search (BFS) to find the shortest path by number of hops.
    For weighted shortest paths, use Dijkstra's algorithm instead.
    
    Parameters
    ---
    graph : Graph
        The graph to search. Can be directed or undirected.
    source : int | str
        Source node (integer ID or string label).
    target : int | str
        Target node (integer ID or string label).
        
    Returns
    ---
    list of str | None
        List of node labels in the path from source to target (inclusive), or
        `None` if no path exists.
        
    Examples
    --------
    ```python
    import edgecase as ec
    from edgecase import shortest_path
    
    graph = ec.Graph()
    graph.add_path_graph(5)  # 1-2-3-4-5
    
    # Find shortest path
    path = shortest_path(graph, "1", "5")
    print(path)  # ['1', '2', '3', '4', '5']
    
    # No path between disconnected components
    graph.add_node("isolated")
    path = shortest_path(graph, "1", "isolated")
    print(path)  # None
    ```
    
    See Also
    --------
    get_graph_diameter : Get longest shortest path in the graph
    get_mean_distance : Get average shortest path length
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
    """Calculate PageRank scores for all nodes in the graph.
    
    PageRank is a link analysis algorithm that assigns importance scores to nodes
    based on their incoming connections. Originally developed by Google to rank web
    pages, it's widely used in social network analysis, recommendation systems, and
    influence detection.
    
    The algorithm iteratively distributes "rank" from each node to its neighbors,
    with a damping factor to model random jumps. Nodes with many incoming edges from
    high-ranking nodes receive higher scores.
    
    Parameters
    ----------
    graph : Graph
        The graph to analyze. Can be directed or undirected.
    damping : float, default 0.85
        The probability of following an edge (vs. random jump). Higher values (closer
        to 1.0) give more weight to the graph structure. Typical range: 0.8-0.95.
    max_iter : int, default 100
        Maximum number of iterations to run. The algorithm stops early if convergence
        is reached.
    tol : float, default 1e-6
        Convergence tolerance. The algorithm stops when the sum of absolute differences
        between iterations is less than this value.
    
    Returns
    -------
    dict[str, float]
        Dictionary mapping node labels to their PageRank scores. Scores sum to 1.0
        across all nodes.
    
    Examples
    --------
    Calculate PageRank on a simple social network:
    
    ```python
    from edgecase import Graph, pagerank
    
    graph = (
        Graph(directed=True)
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_node(label="Charlie")
        .add_edge("Alice", "Bob")
        .add_edge("Alice", "Charlie")
        .add_edge("Bob", "Charlie")
    )
    
    scores = pagerank(graph)
    print(scores)
    # {'Charlie': 0.475, 'Bob': 0.262, 'Alice': 0.262}
    ```
    
    Use custom damping factor for more exploration:
    
    ```python
    scores = pagerank(graph, damping=0.5)  # More random jumping
    ```
    
    Find the most influential nodes:
    
    ```python
    scores = pagerank(graph)
    top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_nodes:
        print(f"{node}: {score:.4f}")
    ```
    
    See Also
    --------
    betweenness_centrality : Measure nodes that bridge communities
    """
    result_ids = rust.pagerank(graph._graph, damping=damping, max_iter=max_iter, tol=tol)
    # Convert node IDs to labels
    return {graph.label_from_id(node_id): score for node_id, score in result_ids.items()}


def betweenness_centrality(graph: Graph) -> dict[str, float]:
    """Calculate betweenness centrality scores for all nodes.
    
    Betweenness centrality measures how often a node appears on the shortest paths
    between other nodes in the graph. Nodes with high betweenness act as bridges or
    bottlenecks, controlling the flow of information or resources through the network.
    
    This metric is particularly useful for:
    - Identifying key connectors in social networks
    - Finding critical infrastructure nodes in transportation/communication networks
    - Detecting potential bottlenecks or points of failure
    
    The algorithm computes shortest paths between all pairs of nodes and counts how
    many times each node appears on these paths.
    
    Parameters
    ----------
    graph : Graph
        The graph to analyze. Works with both directed and undirected graphs.
    
    Returns
    -------
    dict[str, float]
        Dictionary mapping node labels to their betweenness centrality scores.
        Scores are normalized to the range [0, 1].
    
    Examples
    --------
    Find bridge nodes in a network:
    
    ```python
    from edgecase import Graph, betweenness_centrality
    
    # Create two clusters connected by a bridge node
    graph = (
        Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="Bridge")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge("A", "B")
        .add_edge("B", "Bridge")
        .add_edge("Bridge", "C")
        .add_edge("C", "D")
    )
    
    scores = betweenness_centrality(graph)
    print(scores["Bridge"])  # High score - bridges two clusters
    ```
    
    Identify the most important connectors:
    
    ```python
    scores = betweenness_centrality(graph)
    connectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for node, score in connectors:
        print(f"{node}: {score:.4f}")
    ```
    
    See Also
    --------
    pagerank : Measure node importance by incoming connections
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
    """Detect communities using Fast Greedy (Clauset-Newman-Moore) algorithm.
    
    This is a greedy modularity optimization algorithm that merges communities
    to maximize modularity. It's efficient for medium to large graphs and provides
    good quality communities.
    
    Algorithm
    ---
    1. Start with each node in its own community
    2. Build a heap of merge candidates ordered by modularity gain (ΔQ)
    3. Repeatedly merge the pair with highest ΔQ until no improvement
    4. Return the partition with maximum modularity
    
    Parameters
    ---
    graph : Graph
        An undirected Graph instance.
        
    Returns
    ---
    dict of str to int
        Mapping from node ID to community ID (0-indexed).
        
    Raises
    ---
    ValueError
        If the graph is directed (requires undirected graph).
        
    Complexity
    ---
    - Time: O(n log²n) on sparse graphs
    - Space: O(n + m)
        
    Examples
    --------
    ```python
    import edgecase as ec
    
    # Two disconnected cliques
    graph = ec.Graph(directed=False)
    for node in ['A', 'B', 'C', 'D', 'E', 'F']:
        graph.add_node(node)
    for src, tgt in [('A','B'), ('B','C'), ('C','A'), ('D','E'), ('E','F'), ('F','D')]:
        graph.add_edge(src, tgt)
    
    communities = graph.get_cmty_fast_greedy()
    
    # Should find 2 communities: {A,B,C} and {D,E,F}
    print(len(set(communities.values())))  # 2
    ```
    
    Karate club example:
    
    ```python
    graph = ec.Graph(directed=False)
    # Add karate club edges...
    communities = graph.get_cmty_fast_greedy()
    # Modularity typically > 0.40 for karate club
    ```
        
    References
    ---
    Clauset, A., Newman, M. E., & Moore, C. (2004).
    "Finding community structure in very large networks."
    Physical review E, 70(6), 066111.
        
    See Also
    --------
    get_cmty_louvain : Faster for very large graphs
    get_cmty_edge_btwns : More accurate but slower
    """
    return graph._graph.get_cmty_fast_greedy()
