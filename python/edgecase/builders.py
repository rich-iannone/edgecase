"""
Graph building functions.

These functions create common graph structures and patterns, making it easy
to construct graphs for analysis and visualization.
"""

from typing import Optional, Dict, Any
import polars as pl


def add_path(
    graph,
    n: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a path graph to an existing graph.
    
    A path graph is a sequence of nodes connected linearly: 1 → 2 → 3 → ... → n.
    End nodes have degree 1, internal nodes have degree 2.
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=True)
    graph = ec.add_path(graph, n=5, type="step", rel="next")
    
    print(graph.num_nodes())  # 5
    print(graph.num_edges())  # 4
    ```
    
    See Also
    --------
    add_cycle : Add a cycle graph (closed loop)
    add_star : Add a star graph
    """
    if n < 1:
        raise ValueError("Path must have at least 1 node")
    
    # Get starting node ID (continue from existing nodes)
    start_id = graph.num_nodes() + 1
    
    # Add nodes
    for i in range(n):
        node_id = str(start_id + i)
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = node_id
        graph.add_node(node_id, **attrs)
    
    # Add edges (path has n-1 edges)
    for i in range(n - 1):
        source = str(start_id + i)
        target = str(start_id + i + 1)
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        graph.add_edge(source, target, **edge_attrs)
    
    return graph


def add_cycle(
    graph,
    n: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a cycle graph to an existing graph.
    
    A cycle graph is a closed loop of nodes: 1 → 2 → 3 → ... → n → 1.
    Each node has exactly degree 2.
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=True)
    graph = ec.add_cycle(graph, n=4, type="state", rel="transitions_to")
    
    print(graph.num_nodes())  # 4
    print(graph.num_edges())  # 4
    ```
    
    See Also
    --------
    add_path : Add a path graph (linear, no cycle)
    add_full_graph : Add a complete graph
    """
    if n < 3:
        raise ValueError("Cycle must have at least 3 nodes")
    
    # Add path first
    graph = add_path(graph, n, type=type, label=label, rel=rel)
    
    # Close the cycle with final edge
    start_id = graph.num_nodes() - n + 1
    end_id = graph.num_nodes()
    edge_attrs = {}
    if rel:
        edge_attrs['rel'] = rel
    graph.add_edge(str(end_id), str(start_id), **edge_attrs)
    
    return graph


def add_star(
    graph,
    n: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a star graph to an existing graph.
    
    A star graph has one central node connected to n-1 peripheral nodes.
    The center node has degree n-1, all peripheral nodes have degree 1.
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=False)
    graph = ec.add_star(graph, n=6, type="network", rel="connected_to")
    
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
    
    start_id = graph.num_nodes() + 1
    center_id = str(start_id)
    
    # Add all nodes
    for i in range(n):
        node_id = str(start_id + i)
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = node_id
        graph.add_node(node_id, **attrs)
    
    # Add edges from center to all other nodes
    for i in range(1, n):
        peripheral_id = str(start_id + i)
        edge_attrs = {}
        if rel:
            edge_attrs['rel'] = rel
        graph.add_edge(center_id, peripheral_id, **edge_attrs)
    
    return graph


def add_full_graph(
    graph,
    n: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
    keep_loops: bool = False,
) -> "Graph":
    """Add a complete graph (clique) to an existing graph.
    
    A complete graph has edges between every pair of nodes. In a complete graph
    with n nodes, there are n(n-1)/2 edges for undirected or n(n-1) for directed.
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=False)
    graph = ec.add_full_graph(graph, n=4, type="complete")
    
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
    
    start_id = graph.num_nodes() + 1
    
    # Add all nodes
    for i in range(n):
        node_id = str(start_id + i)
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = node_id
        graph.add_node(node_id, **attrs)
    
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
            if not graph.is_directed and i > j:
                continue
            
            graph.add_edge(source_id, target_id, **edge_attrs)
    
    return graph


def add_balanced_tree(
    graph,
    k: int,
    h: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a balanced tree to an existing graph.
    
    A balanced k-ary tree of height h where each non-leaf node has exactly k children.
    Total nodes = (k^(h+1) - 1) / (k - 1) for k > 1.
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=True)
    graph = ec.add_balanced_tree(graph, k=2, h=3, type="tree_node", rel="parent_of")
    
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
    
    start_id = graph.num_nodes() + 1
    
    # First, add all nodes
    total_nodes = (k**(h + 1) - 1) // (k - 1) if k > 1 else h + 1
    for i in range(total_nodes):
        node_id = start_id + i
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = str(node_id)
        graph.add_node(str(node_id), **attrs)
    
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
                graph.add_edge(str(node_id), str(child_id), **edge_attrs)
        
        nodes_at_level = next_level
    
    return graph


def add_prism(
    graph,
    n: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a prism graph to an existing graph.
    
    A prism graph consists of two parallel n-cycles connected by matching edges.
    It's like taking two cycle graphs and connecting corresponding nodes. Total
    edges = 3n (n in each cycle + n connecting edges).
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=False)
    graph = ec.add_prism(graph, n=4, type="prism_node")
    
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
    
    start_id = graph.num_nodes() + 1
    
    # Add first cycle
    cycle1_start = start_id
    for i in range(n):
        node_id = str(cycle1_start + i)
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = f"{node_id}_top"
        graph.add_node(node_id, **attrs)
    
    # Connect first cycle
    edge_attrs = {}
    if rel:
        edge_attrs['rel'] = rel
    for i in range(n):
        source = str(cycle1_start + i)
        target = str(cycle1_start + ((i + 1) % n))
        graph.add_edge(source, target, **edge_attrs)
    
    # Add second cycle
    cycle2_start = cycle1_start + n
    for i in range(n):
        node_id = str(cycle2_start + i)
        attrs = {}
        if type:
            attrs['type'] = type
        if label:
            attrs['label'] = f"{node_id}_bottom"
        graph.add_node(node_id, **attrs)
    
    # Connect second cycle
    for i in range(n):
        source = str(cycle2_start + i)
        target = str(cycle2_start + ((i + 1) % n))
        graph.add_edge(source, target, **edge_attrs)
    
    # Connect corresponding nodes between cycles
    for i in range(n):
        top_node = str(cycle1_start + i)
        bottom_node = str(cycle2_start + i)
        graph.add_edge(top_node, bottom_node, **edge_attrs)
    
    return graph


def add_grid_2d(
    graph,
    x: int,
    y: int,
    type: Optional[str] = None,
    label: bool = True,
    rel: Optional[str] = None,
) -> "Graph":
    """Add a 2D grid graph to an existing graph.
    
    Creates a rectangular lattice where each node is connected to its orthogonal
    neighbors (up, down, left, right). Total edges = (x-1)·y + x·(y-1).
    
    Parameters
    ---
    graph : Graph
        The graph object to add nodes/edges to.
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
    
    graph = ec.Graph(directed=False)
    graph = ec.add_grid_2d(graph, x=3, y=3, type="cell")
    
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
    
    start_id = graph.num_nodes() + 1
    
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
            
            graph.add_node(node_id, **attrs)
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
                graph.add_edge(current, right, **edge_attrs)
            
            # Connect to bottom neighbor
            if j + 1 < y:
                bottom = node_map[(i, j + 1)]
                graph.add_edge(current, bottom, **edge_attrs)
    
    return graph


def create_graph(
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
    
    graph = ec.create_graph(nodes_df=nodes, edges_df=edges, directed=True)
    print(graph.num_nodes())  # 3
    print(graph.num_edges())  # 2
    ```
    
    See Also
    --------
    Graph.add_node : Add individual nodes
    Graph.add_edge : Add individual edges
    Graph.node_df : Extract node data as DataFrame
    Graph.edge_df : Extract edge data as DataFrame
    """
    # Import here to avoid circular dependency
    from edgecase import Graph
    
    g = Graph(directed=directed)
    
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
