from typing import Any, Callable, Optional, Union

import polars as pl

from edgecase import _core as rust

# ==============================================================================
# Core Data Structures
# ==============================================================================

# These are the Node and Edge classes from Rust backend for direct instantiation.
# These are typically used in advanced scenarios or testing where you need
# fine-grained control over graph element creation.

Node = rust.Node
"""
Node data structure.

Represents a graph node with an ID, optional label, and optional attributes. Typically created via
`Graph.add_node()`, but can be instantiated directly for testing or advanced use cases.

Attributes
----------
id
    Internal node ID (64-bit unsigned integer).
label
    Optional human-readable label.
attributes
    Optional dictionary of node attributes.

Examples
--------
Create a basic node:

```python
from edgecase import Node

# Node with ID and label only
node = Node(1, "A")
```

Create a node with attributes:

```python
from edgecase import Node

# Node with ID, label, and attributes
node = Node(2, "B", {"type": "user", "score": 0.9})
```

Use in graph construction:

```python
from edgecase import Graph, Node

# Create graph and add nodes directly
graph = Graph()
graph.add_node(Node(1, "Alice"))
graph.add_node(Node(2, "Bob", {"role": "admin"}))
```
"""

Edge = rust.Edge
"""
Edge data structure.

Represents a graph edge connecting two nodes, with optional weight and attributes. Typically created
via `Graph.add_edge()`, but can be instantiated directly for testing or advanced use cases.

Attributes
----------
id
    Internal edge ID (64-bit unsigned integer).
source
    Source node ID.
target
    Target node ID.
weight
    Optional edge weight (for weighted graphs).
attributes
    Optional dictionary of edge attributes.

Examples
--------
Create a basic edge:

```python
from edgecase import Edge

# Auto-assign ID, connect nodes 1 -> 2
edge = Edge(0, 1, 2, None, None)
```

Create a weighted edge with attributes:

```python
from edgecase import Edge

# Edge with weight and attributes
edge = Edge(0, 1, 2, 1.5, {"type": "friend", "since": 2020})
```

Use in graph construction:

```python
from edgecase import Graph, Node, Edge

# Create graph with direct node and edge objects
graph = Graph(directed=True, weighted=True)
graph.add_node(Node(1, "Alice"))
graph.add_node(Node(2, "Bob"))
graph.add_edge(Edge(0, 1, 2, 0.8, {"relationship": "colleague"}))
```
"""


class Graph:
    """
    A high-performance graph data structure with optional direction and weights.

    The `Graph` class provides a Python interface to a Rust-based graph implementation, offering
    both performance and ease of use. Nodes are identified by 64-bit unsigned integers (u64 IDs) and
    can optionally have string labels for human-readable access.

    Parameters
    ----------
    directed
        Whether the graph should be directed. If `True`, edges have direction from source to target.
        If `False`, edges are bidirectional.
    weighted
        Whether edges can have weights. If `True`, edge weights are stored and considered in
        algorithms. If `False`, all edges are treated equally.
    nodes_df
        Optional Polars DataFrame to initialize nodes. Should contain at minimum an `id` column with
        node identifiers. The values in the `id` column are used as node *labels* (not internal node
        IDs, which are auto-managed). Additional columns are treated as node attributes.
    edges_df
        Optional Polars DataFrame to initialize edges. Should contain at minimum `from` and `to`
        columns with node identifiers that reference the `id` values from `nodes_df`. Additional
        columns are treated as edge attributes (including optional `weight` column).
    graph_name
        Optional name for the graph (stored as a graph attribute).

    Examples
    --------
    Create an undirected, unweighted graph:

    ```python
    from edgecase import Graph

    graph = (
        Graph()
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_edge(source="Alice", target="Bob")
    )
    ```

    Create a graph from DataFrames.

    ```python
    import polars as pl
    from edgecase import Graph

    # Create node data frame (NDF)
    # Note: 'id' column values become node labels, not internal node IDs
    nodes_df = pl.DataFrame({
        "id": ["A", "B", "C"],
        "type": ["person", "person", "person"],
        "value": [10, 20, 15]
    })

    # Create edge data frame (EDF)
    edges_df = pl.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"],
        "weight": [1.5, 2.0],
        "relation": ["friend", "colleague"]
    })

    # Initialize graph with both DataFrames
    graph = Graph(nodes_df=nodes_df, edges_df=edges_df, directed=True, weighted=True)
    ```

    Create a directed, weighted graph for modeling networks:

    ```python
    graph = (
        Graph(directed=True, weighted=True)
        .add_node(label="Alice")
        .add_node(label="Bob")
        .add_edge(source="Alice", target="Bob", weight=0.8)
    )
    ```
    """

    def __init__(
        self,
        directed: bool = False,
        weighted: bool = False,
        nodes_df: Optional[pl.DataFrame] = None,
        edges_df: Optional[pl.DataFrame] = None,
        graph_name: Optional[str] = None,
    ):
        self._graph = rust.Graph(directed=directed, weighted=weighted)
        self._graph_name = graph_name
        self._nodes_df_provided = nodes_df is not None

        # Track computations and aesthetics for describe()
        self._computations = []  # List of (method_name, params, attr_name) tuples
        self._aesthetics = []  # List of (attr_name, source_attr, operation) tuples

        # Initialize from DataFrames if provided
        id_to_label_map = {}
        if nodes_df is not None:
            id_to_label_map = self._init_from_nodes_df(nodes_df)

        if edges_df is not None:
            self._init_from_edges_df(edges_df, id_to_label_map)

    def _init_from_nodes_df(self, nodes_df: pl.DataFrame) -> dict[str, str]:
        """
        Initialize nodes from a Polars DataFrame.

        The 'id' column values are used as node labels, not internal node IDs. Internal node IDs are
        auto-assigned by EdgeCase. If a separate 'label' column exists, it takes precedence as the
        label, and the 'id' column values are used only for edge resolution.

        Parameters
        ----------
        nodes_df
            DataFrame with at minimum an 'id' column. Values in 'id' become node labels (unless a
            separate 'label' column exists). Other columns become node attributes.

        Returns
        -------
        dict[str, str]
            Mapping from id values to label values (for edge resolution)
        """
        if "id" not in nodes_df.columns:
            raise ValueError("nodes_df must contain an 'id' column")

        # Get attribute columns (everything except 'id')
        attr_cols = [col for col in nodes_df.columns if col != "id"]
        id_to_label = {}

        # Add each node
        for row in nodes_df.iter_rows(named=True):
            node_id_value = row["id"]

            # Extract label (use 'label' column if present, otherwise use id)
            if "label" in attr_cols:
                label = str(row["label"])
                # Track id -> label mapping for edge resolution
                id_to_label[str(node_id_value)] = label
            else:
                # Use id as the label
                label = str(node_id_value)

            # Extract type if present
            type_value = row.get("type") if "type" in attr_cols else None

            # Build attributes dict (excluding 'label' and 'type' since they're first-class)
            attributes = {col: row[col] for col in attr_cols if col not in ["label", "type"]}

            # Add node with first-class parameters
            self.add_node(label=label, type=type_value, **attributes)

        return id_to_label

    def _init_from_edges_df(
        self, edges_df: pl.DataFrame, id_to_label_map: dict[str, str] = None
    ) -> None:
        """
        Initialize edges from a Polars DataFrame.

        Automatically creates nodes if they don't exist yet (only if nodes_df was not provided).
        When nodes_df was provided with id/label columns, edge 'from' and 'to' values are mapped
        through `id_to_label_map` to find the actual node labels.

        Parameters
        ----------
        edges_df
            DataFrame with at minimum 'from' and 'to' columns. Other columns become edge attributes.
        id_to_label_map
            Mapping from node ids to labels (from nodes_df processing).
        """
        if "from" not in edges_df.columns or "to" not in edges_df.columns:
            raise ValueError("edges_df must contain 'from' and 'to' columns")

        if id_to_label_map is None:
            id_to_label_map = {}

        # If no nodes_df was provided, auto-create nodes from edges (as labels)
        if not getattr(self, "_nodes_df_provided", False):
            # Collect all unique nodes mentioned in edges
            unique_nodes = set()
            for row in edges_df.iter_rows(named=True):
                unique_nodes.add(row["from"])
                unique_nodes.add(row["to"])

            # Create nodes with these values as labels
            for node_value in sorted(unique_nodes):
                label = str(node_value)
                self.add_node(label=label)

        # Get attribute columns (everything except 'from', 'to', and 'id')
        attr_cols = [col for col in edges_df.columns if col not in ["from", "to", "id"]]

        # Add each edge with its attributes
        for row in edges_df.iter_rows(named=True):
            source_id = str(row["from"])
            target_id = str(row["to"])

            # Map through id_to_label if provided (from nodes_df with separate id/label)
            source = id_to_label_map.get(source_id, source_id)
            target = id_to_label_map.get(target_id, target_id)

            # Extract weight, rel, and other attributes
            weight = None
            rel_value = None
            other_attrs = {}

            for col in attr_cols:
                value = row[col]
                if value is not None:  # Skip None values
                    if col == "weight":
                        if self.is_weighted():
                            weight = value
                        else:
                            # If graph not weighted, treat weight as regular attribute
                            other_attrs[col] = value
                    elif col == "rel":
                        rel_value = value
                    else:
                        other_attrs[col] = value

            # Only pass weight parameter if graph is weighted
            if self.is_weighted():
                self.add_edge(source, target, weight=weight, rel=rel_value, **other_attrs)
            else:
                self.add_edge(source, target, rel=rel_value, **other_attrs)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the graph.

        Uses the `describe()` method to show comprehensive graph information including structure,
        computations, and aesthetics.
        """
        import io
        import sys

        # Capture describe() output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            self.describe()
            result = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        return result.rstrip()  # Remove trailing newline

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # FOUNDATION
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Graph Construction
    # ==============================================================================

    def add_node(
        self,
        label: Optional[str] = None,
        type: Optional[str] = None,
        from_node: Optional[Union[int, str]] = None,
        to_node: Optional[Union[int, str]] = None,
        **attributes,
    ) -> "Graph":
        """
        Add a node to the graph with optional connections and attributes.

        This method provides flexible node creation, allowing you to add isolated nodes, nodes with
        labels, or nodes that are immediately connected to existing nodes. Node IDs are 64-bit
        unsigned integers managed automatically by EdgeCase, while labels provide human-readable
        identifiers.

        Parameters
        ----------
        label
            Optional string label for human-readable node access. Labels should be unique if you
            plan to use them for node lookup in operations like `add_edge()`.
        type
            Optional string type classification for the node (e.g., 'user', 'admin',
            'organization'). This is a first-class attribute that always appears as the third column
            in node DataFrames.
        from_node
            Optional node (ID or label) to connect FROM the new node. Creates a directed edge:
            `new_node -> from_node`.
        to_node
            Optional node (ID or label) to connect TO the new node. Creates a directed edge:
            `to_node -> new_node`.
        **attributes
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
        node_id = graph.get_id_from_label("Node1")
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
            Graph(directed=True)
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

        %family Graph Construction
        """
        # Handle direct Rust Node object (for backward compatibility)
        if isinstance(label, rust.Node):
            new_id = self._graph.add_node(label)
        else:
            # Convert label to string if it's not None
            str_label = str(label) if label is not None else None

            # Handle type as a first-class attribute
            if type is not None:
                attributes = attributes.copy() if attributes else {}
                attributes["type"] = type

            # Let Rust auto-assign the ID (pass 0 to trigger auto-assignment)
            node = rust.Node(id=0, label=str_label, attributes=attributes if attributes else None)
            new_id = self._graph.add_node(node)

        # Add edges if from_node or to_node specified
        if from_node is not None:
            # Create edge: new_node -> from_node
            self.add_edge(new_id, from_node)

        if to_node is not None:
            # Create edge: to_node -> new_node
            self.add_edge(to_node, new_id)

        return self

    def add_edge(
        self,
        source: Union[int, str, rust.Edge],
        target: Optional[Union[int, str]] = None,
        rel: Optional[str] = None,
        weight: Optional[float] = None,
        auto_create_nodes: bool = False,
        **attributes,
    ) -> "Graph":
        """
        Add an edge between two existing nodes in the graph.

        Edges connect existing nodes and can optionally carry weights and attributes. Both source
        and target nodes must already exist in the graph. You can reference nodes by their integer
        IDs or string labels (if labels are unique).

        Parameters
        ----------
        source
            The source node identifier (ID or label).
        target
            The target node identifier (ID or label).
        weight
            Optional edge weight for weighted graphs. Ignored if the graph is unweighted.
        rel
            Optional string relationship type for the edge (e.g., 'friend', 'follows', 'manages').
            This is a first-class attribute that always appears as the fourth column in edge
            DataFrames.
        auto_create_nodes
            If `True`, automatically create nodes (with labels) if they don't exist. If `False`,
            raises an error if source or target nodes don't exist.
        **attributes
            Arbitrary key-value pairs to store as edge attributes.

        Returns
        -------
        Graph
            Returns self to enable method chaining.

        Raises
        ------
        ValueError
            If source or target node doesn't exist in the graph, or if a label cannot be resolved to
            a node ID.

        Examples
        --------
        Add edges using node IDs:

        ```python
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        alice_id = graph.get_id_from_label(label="Alice")
        bob_id = graph.get_id_from_label(label="Bob")
        graph.add_edge(source=alice_id, target=bob_id)
        ```

        Add edges using labels (only if labels are unique):

        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_edge(source="Alice", target="Bob")
        )
        ```

        Mix IDs and labels:

        ```python
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        alice_id = graph.get_id_from_label(label="Alice")
        graph.add_edge(source=alice_id, target="Bob")
        ```

        Add weighted edges:

        ```python
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B", weight=0.75)
        )
        ```

        Add edges with relationship type:

        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_edge(source="Alice", target="Bob", rel="friend")
        )
        ```

        Add edges with custom attributes:

        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_edge(source="Alice", target="Bob", rel="friend", since=2020)
        )
        ```

        %family Graph Construction
        """
        # Handle direct Rust Edge objects (internal/advanced use only)
        if isinstance(source, rust.Edge):
            self._graph.add_edge(source)
            return self

        # Resolve source and target to u64 IDs
        source_id = self._resolve_node_id(source, auto_create=auto_create_nodes)
        target_id = self._resolve_node_id(target, auto_create=auto_create_nodes)

        # Verify nodes exist if not auto-creating
        if not auto_create_nodes:
            if not self._graph.has_node_id(source_id):
                raise ValueError(
                    f"Source node {source} (ID {source_id}) does not exist. "
                    f"Add it with add_node() first or use auto_create_nodes=True."
                )
            if not self._graph.has_node_id(target_id):
                raise ValueError(
                    f"Target node {target} (ID {target_id}) does not exist. "
                    f"Add it with add_node() first or use auto_create_nodes=True."
                )

        # Handle rel as a first-class attribute
        if rel is not None:
            attributes = attributes.copy() if attributes else {}
            attributes["rel"] = rel

        # Create edge with auto-assigned ID (0 triggers auto-assignment)
        edge = rust.Edge(
            id=0,
            source=source_id,
            target=target_id,
            weight=weight,
            attributes=attributes if attributes else None,
        )
        self._graph.add_edge(edge)
        return self

    def _resolve_node_id(self, node: Union[int, str], auto_create: bool = False) -> int:
        """
        Resolve a node identifier (int or str) to a u64 ID.

        Internal helper method that converts either integer IDs or string labels to the underlying
        u64 node ID used by the Rust core.

        Parameters
        ----------
        node
            Either an integer ID or a string label to resolve.
        auto_create
            If `True` and `node=` is a string not in the graph, create a new node with that label.
            If `False`, raise an error for unknown labels.

        Returns
        -------
        int
            The u64 node ID corresponding to the provided identifier.

        Raises
        ------
        ValueError
            If `node` is a label not found in the graph and `auto_create=` is `False`.
        TypeError
            If `node=` is neither an integer nor a string.
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
                    # Create a new node with this label and return its ID
                    self.add_node(label=node)
                    return self._graph.id_from_label(node)
                else:
                    raise ValueError(f"Node with label '{node}' not found in graph")
        else:
            raise TypeError(f"Node identifier must be int or str, got {type(node)}")

    def delete_node(self, node: Union[int, str]) -> "Graph":
        """
        Delete a node from the graph.

        Removes a node and all its incident edges from the graph. The node can be specified by its
        ID or label.

        Parameters
        ----------
        node
            Node ID (int) or label (str) to delete.

        Returns
        -------
        Graph
            Returns self to enable method chaining.

        Raises
        ------
        KeyError
            If the node doesn't exist in the graph.

        Examples
        --------
        Delete node by label:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )

        graph.delete_node(node="B")
        print(graph.has_node(node="B"))  # False
        ```

        Delete node by ID:

        ```python
        graph = Graph().add_node(label="Alice")
        node_id = graph.get_id_from_label(label="Alice")

        graph.delete_node(node=node_id)
        print(graph.has_node(node=node_id))  # False
        ```

        %family Graph Construction
        """
        try:
            node_id = self._resolve_node_id(node, auto_create=False)
        except ValueError as e:
            # Convert ValueError to KeyError for consistency
            raise KeyError(str(e))

        # Rust delete_node raises PyKeyError if node doesn't exist
        self._graph.delete_node(node_id)
        return self

    def delete_edge(
        self,
        source: Union[int, str, None] = None,
        target: Union[int, str, None] = None,
        id: int | None = None,
    ) -> "Graph":
        """
        Delete an edge from the graph.

        Remove edges by specifying either the source and target nodes, or the edge ID directly. For
        multigraphs with source/target, only the first matching edge is deleted.

        Parameters
        ----------
        source
            Source node ID or label. Required if `id=` is not provided.
        target
            Target node ID or label. Required if `id=` is not provided.
        id
            Edge ID to delete. If provided, `source=` and `target=` are ignored.

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
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .delete_edge(source="A", target="B")
        )

        print(graph.has_edge(source="A", target="B"))  # False
        ```

        Delete edge by ID:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )
        edge_id = graph.get_edge_ids()[0]

        graph.delete_edge(id=edge_id)
        print(graph.has_edge(source="A", target="B"))  # False
        ```

        %family Graph Construction
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

    def delete_edge_by_id(self, edge_id: int) -> "Graph":
        """
        Delete an edge from the graph by its ID.

        Removes an edge directly using its unique edge identifier. This is more efficient than
        deleting by source and target nodes when the edge ID is known, and is particularly useful
        for multigraphs where multiple edges may exist between the same pair of nodes.

        Parameters
        ----------
        edge_id
            The unique identifier of the edge to remove.

        Returns
        -------
        Graph
            Returns self to enable method chaining.

        Raises
        ------
        KeyError
            If no edge exists with the specified ID.

        Examples
        --------
        Delete an edge by its ID:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )

        # Get the edge ID
        edge_id = graph.get_edge_ids()[0]

        # Delete the edge
        graph.delete_edge_by_id(edge_id=edge_id)
        print(graph.has_edge(source="A", target="B"))  # False
        ```

        Delete specific edges in a multigraph:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="A", target="B", weight=2.0)
        )

        # Get edge IDs and delete the first one
        edge_ids = graph.get_edge_ids()
        graph.delete_edge_by_id(edge_id=edge_ids[0])
        print(graph.has_edge(source="A", target="B"))  # True (second edge remains)
        ```

        %family Graph Construction
        """
        self._graph.delete_edge_by_id(edge_id)

    def delete_nodes_ws(self) -> "Graph":
        """
        Delete all selected nodes and their incident edges.

        Removes nodes in the current selection from the graph, along with all edges connected to
        those nodes. Updates selection state and removes the selected nodes from the graph
        structure.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Delete specific nodes from a path:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        graph.add_path_graph(5)  # Creates path: 0-1-2-3-4

        # Select and delete middle nodes
        node_ids = list(graph.get_node_ids())
        graph.select_nodes_by_id([node_ids[2], node_ids[3]])
        graph.delete_nodes_ws()

        print(graph.count_nodes())  # 3 (nodes 0, 1, 4 remain)
        print(graph.count_edges())  # 1 (only edge 0-1 remains)
        ```

        Delete nodes by filtering:

        ```python
        graph = ec.Graph()
        graph.add_node("A", value=10)
        graph.add_node("B", value=25)
        graph.add_node("C", value=30)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # Delete high-value nodes
        graph.select_nodes(lambda n: n.attributes.get("value", 0) > 20)
        graph.delete_nodes_ws()

        print(graph.count_nodes())  # 1 (only A remains)
        ```

        %family Graph Construction
        """
        selected = self.get_selected_node_ids()

        # Delete each selected node (Rust handles incident edges)
        for node_id in selected:
            try:
                self._graph.delete_node(node_id)
            except Exception:
                pass  # Node might already be deleted

        return self

    def delete_edges_ws(self) -> "Graph":
        """
        Delete all selected edges.

        Removes edges in the current selection from the graph. Node connectivity may be affected,
        but nodes themselves remain in the graph. Updates selection state automatically.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Delete specific edges from a path:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        graph.add_path_graph(5)  # Creates path: 0-1-2-3-4

        # Select and delete middle edges
        edge_ids = graph.get_edge_ids()[1:3]  # Get edges 1 and 2
        graph.select_edges_by_id(edge_ids)
        graph.delete_edges_ws()

        print(graph.count_nodes())  # 5 (all nodes remain)
        print(graph.count_edges())  # 2 (only first and last edges remain)
        ```

        Delete edges by filtering:

        ```python
        graph = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="C", weight=5.0)
            .add_edge(source="C", target="D", weight=2.0)
        )

        # Delete low-weight edges
        graph.select_edges(lambda e: e.weight and e.weight < 2.0)
        graph.delete_edges_ws()

        print(graph.count_edges())  # 2 (edges with weight >= 2.0 remain)
        ```

        %family Graph Construction
        """
        selected = self.get_selected_edge_ids()

        for edge_id in selected:
            self.delete_edge_by_id(edge_id)

        # Edge selection is automatically cleared since edges no longer exist

        return self

    def remove_selfloops(self) -> "Graph":
        """
        Remove all self-loops from the graph.

        A self-loop is an edge where the source and target nodes are the same. This method removes
        all such edges from the graph, which is a common preprocessing step for many graph
        algorithms that assume simple graphs.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Remove self-loops from a graph:

        ```python
        from edgecase import Graph

        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B")
        g.add_edge(source="A", target="A")  # Self-loop
        g.add_edge(source="B", target="B")  # Self-loop

        print(g.count_edges())  # 3
        g.remove_selfloops()
        print(g.count_edges())  # 1
        ```

        Use in preprocessing pipeline:

        ```python
        g = (
            Graph()
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge(source="X", target="Y")
            .add_edge(source="Y", target="Z")
            .add_edge(source="X", target="X")  # Self-loop
            .remove_selfloops()
        )

        print(g.is_simple())  # True (if no multi-edges)
        ```

        Notes
        -----
        - works for both directed and undirected graphs.
        - does not affect multi-edges (use `remove_duplicate_edges()` or `simplify()` for those).
        - use `which_loops()` to identify self-loops before removal.
        - use `count_loops()` to count self-loops per node before removal.

        %family Graph Construction
        """
        self._graph.remove_selfloops()
        return self

    def remove_duplicate_edges(self, keep_distinct_attributes: bool = False) -> "Graph":
        """
        Remove duplicate edges, keeping only one edge per node pair.

        For each pair of nodes with multiple edges between them, keeps only the first edge (by edge
        ID order) and removes all others. This converts a multigraph into a simple graph with
        respect to multi-edges, but does not remove self-loops.

        Parameters
        ----------
        keep_distinct_attributes
            If `False` (the default), edges are considered duplicates based solely on their source
            and target nodes, regardless of weight, `rel`, or other attributes. If `True`, edges
            with different attribute values (weight, `rel`, or custom attributes) are considered
            distinct and will not be removed.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Remove duplicate edges from a multigraph (default behavior):

        ```python
        from edgecase import Graph

        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="A", target="B", weight=2.0)  # Duplicate
        g.add_edge(source="A", target="B", weight=3.0)  # Duplicate

        print(g.count_edges())  # 3
        g.remove_duplicate_edges()
        print(g.count_edges())  # 1
        ```

        Keep edges with different attributes:

        ```python
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B", rel="friend")
        g.add_edge(source="A", target="B", rel="colleague")
        g.add_edge(source="A", target="B", rel="friend")  # True duplicate

        print(g.count_edges())  # 3
        g.remove_duplicate_edges(keep_distinct_attributes=True)
        print(g.count_edges())  # 2 (friend and colleague kept)
        ```

        Use in preprocessing pipeline:

        ```python
        g = (
            Graph()
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge(source="X", target="Y", type="email")
            .add_edge(source="X", target="Y", type="message")
            .add_edge(source="X", target="Y", type="call")
            .remove_duplicate_edges()
        )

        print(g.count_edges())  # 1 (first edge kept)
        ```

        Notes
        -----
        - by default, **edge attributes are ignored**: edges are considered duplicates based
        solely on their source and target nodes
        - use `keep_distinct_attributes=True` to preserve edges that have different weights,
        `rel` values, or other attributes
        - the first edge (lowest edge ID) in each group of true duplicates is retained
        - for directed graphs, edges `(u, v)` and `(v, u)` are considered different pairs
        - for undirected graphs, edges `(u, v)` and `(v, u)` are considered the same pair
        - does not remove self-loops (use `remove_selfloops()` or `simplify()` for those)
        - use `which_multiple()` to identify duplicate edges before removal
        - use `count_multiple()` to count duplicates per node pair before removal

        %family Graph Construction
        """
        self._graph.remove_duplicate_edges(keep_distinct_attributes)
        return self

    def simplify(self) -> "Graph":
        """
        Remove all self-loops and duplicate edges, making the graph simple.

        A simple graph has no self-loops (edges from a node to itself) and no multi-edges (multiple
        edges between the same pair of nodes). This method removes both types, which is a common
        preprocessing step for algorithms that require simple graphs.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Simplify a graph with both self-loops and multi-edges:

        ```python
        from edgecase import Graph

        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="A", target="B", weight=2.0)  # Duplicate
        g.add_edge(source="A", target="A")  # Self-loop
        g.add_edge(source="B", target="B")  # Self-loop

        print(g.count_edges())  # 4
        print(g.is_simple())  # False

        g.simplify()
        print(g.count_edges())  # 1
        print(g.is_simple())  # True
        ```

        Use in preprocessing pipeline:

        ```python
        g = (
            Graph()
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge(source="X", target="Y")
            .add_edge(source="X", target="Y")  # Duplicate
            .add_edge(source="Y", target="Z")
            .add_edge(source="Z", target="Z")  # Self-loop
            .simplify()
        )

        print(g.is_simple())  # True
        ```

        Notes
        -----
        - equivalent to calling `remove_selfloops()` followed by `remove_duplicate_edges()`
        - for duplicate edges, the first edge (lowest edge ID) in each group is retained
        - for directed graphs, edges `(u, v)` and `(v, u)` are considered different pairs
        - for undirected graphs, edges `(u, v)` and `(v, u)` are considered the same pair
        - after simplification, `is_simple()` will return `True`

        %family Graph Construction
        """
        self._graph.simplify()
        return self

    # ====================================================================
    # Graph Information
    # ====================================================================

    def describe(self, describe_hook: Optional[callable] = None) -> None:
        """
        Print a comprehensive description of the graph.

        Displays key graph properties including node/edge counts, connectivity, degree statistics,
        and optional custom properties via a hook function.

        Parameters
        ----------
        describe_hook
            A function that takes a Graph as input and returns a dictionary of additional properties
            to include. Keys are property names, values are the corresponding values to display.

        Examples
        --------
        Basic usage:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        graph.describe()
        # Number of nodes                : 3
        # Number of edges                : 4
        # Directed                       : False
        # Density                        : 0.67
        # Average degree (min, max)      : 1.33 (1, 2)
        # Connected                      : True
        # Number of connected components : 1
        ```

        With custom hook for additional properties:

        ```python
        from edgecase import karate_club
        graph = karate_club()

        def custom_stats(g):
            return {
                "Clustering": g.get_avg_clustering_coefficient(),
                "Diameter": g.get_diameter() or "N/A"
            }

        graph.describe(describe_hook=custom_stats)
        # ... (basic stats)
        # Clustering                     : 0.571
        # Diameter                       : 5
        ```

        Karate club example:

        ```python
        from edgecase import karate_club
        G = karate_club()
        G.describe()
        # Number of nodes                : 34
        # Number of edges                : 78
        # Directed                       : False
        # Density                        : 0.14
        # Average degree (min, max)      : 4.59 (1, 17)
        # Connected                      : True
        # Number of connected components : 1
        ```

        Notes
        -----
        The `describe_hook=` parameter allows you to compute expensive metrics only when needed. For
        large graphs, computing properties like diameter or clustering can be slow.

        %family Graph Information
        """
        info_dict = self._create_describe_info_dict()

        # Add custom properties if hook provided
        if describe_hook is not None:
            additional_info = describe_hook(self)
            info_dict.update(additional_info)

        # Calculate column width for alignment
        max_key_len = max(len(k) for k in info_dict) if info_dict else 0

        # Print formatted output
        for key, val in info_dict.items():
            print(f"{key:<{max_key_len}} : {val}")

        # Print computations section if any
        if hasattr(self, "_computations") and self._computations:
            print()
            print("Computations on nodes:")
            for method_name, params, attr_name in self._computations:
                # Format parameters
                if params:
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    param_str = f"({param_str})"
                else:
                    param_str = "()"
                # Get count of non-null values
                try:
                    ndf = self.get_node_df()
                    if attr_name in ndf.columns:
                        n_values = ndf.select(pl.col(attr_name).is_not_null().sum()).item()
                        n_total = ndf.height
                        coverage = f"{n_values} of {n_total} nodes"
                    else:
                        coverage = "attr not found"
                except Exception:
                    coverage = "unknown"
                print(f"  - {method_name}{param_str} -> attr: '{attr_name}' ({coverage})")

        # Print aesthetics section if any
        if hasattr(self, "_aesthetics") and self._aesthetics:
            print()
            print("Node aesthetics:")
            for item in self._aesthetics:
                # Handle both old format (3-tuple) and new format (4-tuple)
                if len(item) == 3:
                    attr_name, source_attr, operation = item
                    params = {}
                else:
                    attr_name, source_attr, operation, params = item

                # Format operation with parameters
                if operation == "rescale" and "range" in params:
                    range_vals = params["range"]
                    op_str = f"rescale (range: [{range_vals[0]}, {range_vals[1]}])"
                elif operation == "colorize" and "palette" in params:
                    palette = params["palette"]
                    op_str = f'colorize (palette: "{palette}")'
                else:
                    op_str = operation

                # Get summary stats for the aesthetic attribute
                try:
                    ndf = self.get_node_df()
                    if attr_name in ndf.columns:
                        col = ndf.select(attr_name).to_series()
                        n_values = col.drop_nulls().len()
                        n_total = ndf.height
                        # Try to get value summary
                        unique_vals = col.drop_nulls().unique()
                        if unique_vals.len() <= 3:
                            # Show all unique values for very small sets
                            vals = sorted([str(v) for v in unique_vals.to_list()])
                            val_str = f"values: {', '.join(vals)}"
                        elif unique_vals.len() <= 5:
                            # Show first 3 + "and N more" for small sets
                            vals = sorted([str(v) for v in unique_vals.to_list()])
                            shown = ", ".join(vals[:3])
                            remaining = len(vals) - 3
                            val_str = f"values: {shown} and {remaining} more"
                        else:
                            # Show count for larger sets
                            val_str = f"{unique_vals.len()} unique values"
                        summary = f"{val_str}, {n_values} of {n_total} nodes"
                    else:
                        summary = "attr not found"
                except Exception:
                    summary = "unknown"
                print(f"  - {attr_name} <- {source_attr} via {op_str} ({summary})")

    def _create_describe_info_dict(self) -> dict:
        """
        Create dictionary of graph properties for describe().

        Internal method that computes standard graph properties.
        """
        info = {}

        # Add graph name if it exists
        name = self.get_graph_name()
        if name:
            info["Name of Graph"] = name

        # Basic properties
        n_nodes = self.count_nodes()
        n_edges = self.count_edges()

        info.update(
            {
                "Number of nodes": n_nodes,
                "Number of edges": n_edges,
                "Directed": self.is_directed(),
            }
        )

        # Only compute additional stats if graph has nodes
        if n_nodes == 0:
            return info

        # Density
        density = self.get_density()
        info["Density"] = f"{density:.2f}"

        # Degree statistics: compute for all nodes
        node_ids = self.get_node_ids(limit=None)
        degrees = [self.get_degree(self.get_label_from_id(nid) or str(nid)) for nid in node_ids]

        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            min_degree = min(degrees)
            max_degree = max(degrees)
            info["Average degree (min, max)"] = f"{avg_degree:.2f} ({min_degree}, {max_degree})"

        # Connectivity
        if self.is_directed():
            # For directed graphs, check strong and weak connectivity via component counts
            strong_components = self.count_strongly_connected_components()
            weak_components = self.count_weakly_connected_components()

            info["Strongly connected"] = strong_components == 1
            info["Weakly connected"] = weak_components == 1
            info["Number of strongly connected components"] = strong_components
            info["Number of weakly connected components"] = weak_components
        else:
            # For undirected graphs, just check connectivity
            is_connected = self.is_connected()
            info["Connected"] = is_connected

            # Always show component count (useful even if connected)
            components = self.count_weakly_connected_components()
            info["Number of connected components"] = components

        # Tree check (only for undirected graphs, quick check)
        if not self.is_directed() and n_nodes > 0:
            # A tree has exactly n-1 edges and is connected
            is_tree = (n_edges == n_nodes - 1) and self.is_connected()
            info["Tree"] = is_tree

        # Bipartite check (only for undirected graphs)
        if not self.is_directed():
            info["Bipartite"] = self.is_bipartite()

        return info

    # ====================================================================
    # Graph Rendering
    # ====================================================================

    def show(
        self,
        target: str = "auto",
        width: int = 800,
        height: int = 600,
        node_size: float = 20.0,
        edge_width: float = 2.0,
        layout: str = "force",
    ) -> None:
        """
        Display the graph visualization in a notebook or web browser.

        Parameters
        ----------
        target
            Where to show the graph: `"auto"`, `"notebook"`, or `"browser"`. If `"auto"`, infers
            whether to display inline (notebook) or open browser.
        width
            Canvas width in pixels
        height
            Canvas height in pixels
        node_size
            Node radius in pixels
        edge_width
            Edge line width in pixels
        layout
            Layout algorithm: "force" (vis-network physics), "neato" (Graphviz spring),
            "dot" (hierarchical), "fdp" (force-directed), "circo" (circular), "twopi" (radial)

        Examples
        --------
        ```python
        from edgecase import Graph
        G = Graph()
        G.add_node(label="A")
        G.add_node(label="B")
        G.add_node(label="C")
        G.add_edge(source="A", target="B")
        G.add_edge(source="B", target="C")
        # G.show()  # Auto-detects notebook vs browser
        # G.show(target="browser")  # Force browser
        # G.show(target="notebook")  # Force notebook display
        ```

        %family Graph Rendering
        """
        from edgecase.viz import plot_graph

        if target == "auto":
            target = self._infer_render_target()

        if target == "notebook":
            # Display inline in notebook - plot_graph handles this now
            plot_graph(
                self,
                output_path=None,
                open_browser=False,
                width=width,
                height=height,
                node_size=node_size,
                edge_width=edge_width,
                layout=layout,
            )
        elif target == "browser":
            # Open in browser
            plot_graph(
                self,
                output_path=None,
                open_browser=True,
                width=width,
                height=height,
                node_size=node_size,
                edge_width=edge_width,
                layout=layout,
            )
        else:
            raise ValueError(f"Unknown target display: {target}")

    def _infer_render_target(self) -> str:
        """Infer whether to render in notebook or browser."""
        try:
            # Check if we're in an IPython/Jupyter environment
            from IPython import get_ipython

            if get_ipython() is not None:
                return "notebook"
        except ImportError:
            pass

        return "browser"

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
        """
        Add a path graph (linear chain) to the current graph.

        Creates a sequence of `n` nodes connected linearly: `1 -> 2 -> 3 -> ... -> n`. End nodes
        have degree `1`, internal nodes have degree 2.

        Parameters
        ----------
        n
            Number of nodes in the path.
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
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

        print(graph.count_nodes())  # 5
        print(graph.count_edges())  # 4
        ```

        %family Graph Generators
        """
        if n < 1:
            raise ValueError("Path must have at least 1 node")

        # Get starting node ID (continue from existing nodes)
        start_id = self.count_nodes() + 1

        # Add nodes
        for i in range(n):
            node_label = str(start_id + i) if label else None
            attrs = {}
            if type:
                attrs["type"] = type
            self.add_node(label=node_label, **attrs)

        # Add edges (path has n-1 edges)
        for i in range(n - 1):
            source = str(start_id + i)
            target = str(start_id + i + 1)
            edge_attrs = {}
            if rel:
                edge_attrs["rel"] = rel
            self.add_edge(source, target, **edge_attrs)

        return self

    def add_cycle_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """
        Add a cycle graph (closed loop) to the current graph.

        Creates n nodes in a circular pattern where the last connects back to first. Each node has
        exactly degree `2`.

        Parameters
        ----------
        n
            Number of nodes in the cycle (minimum `3`).
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
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

        print(graph.count_nodes())  # 4
        print(graph.count_edges())  # 4
        ```

        %family Graph Generators
        """
        if n < 3:
            raise ValueError("Cycle must have at least 3 nodes")

        # Add path first
        self.add_path_graph(n, type=type, label=label, rel=rel)

        # Close the cycle with final edge
        start_id = self.count_nodes() - n + 1
        end_id = self.count_nodes()
        edge_attrs = {}
        if rel:
            edge_attrs["rel"] = rel
        self.add_edge(str(end_id), str(start_id), **edge_attrs)

        return self

    def add_star_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """
        Add a star graph (hub-and-spoke) to the current graph.

        Creates a central hub node connected to `n - 1` peripheral nodes. All edges connect the hub
        to spokes; no edges between spokes.

        Parameters
        ----------
        n
            Total number of nodes (minimum `2`).
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
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

        print(graph.count_nodes())  # 5
        print(graph.count_edges())  # 4
        ```

        %family Graph Generators
        """
        if n < 2:
            raise ValueError("Star must have at least 2 nodes")

        start_id = self.count_nodes() + 1
        center_id = str(start_id)

        # Add all nodes
        for i in range(n):
            node_label = str(start_id + i) if label else None
            attrs = {}
            if type:
                attrs["type"] = type
            self.add_node(label=node_label, **attrs)

        # Add edges from center to all other nodes
        for i in range(1, n):
            peripheral_id = str(start_id + i)
            edge_attrs = {}
            if rel:
                edge_attrs["rel"] = rel
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
        """
        Add a complete graph (clique) to the current graph.

        Creates `n` nodes where every node is connected to every other node. In a complete graph
        with `n` nodes, there are `n(n - 1) / 2` edges (without self-loops).

        Parameters
        ----------
        n
            Number of nodes.
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
            Optional relationship label for all edges.
        keep_loops
            If `True`, include self-loops (adding n more edges).

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Create a complete graph:

        ```python
        graph = Graph().add_full_graph(n=4)

        print(graph.count_nodes())  # 4
        print(graph.count_edges())  # 6
        ```

        %family Graph Generators
        """
        if n < 1:
            raise ValueError("Complete graph must have at least 1 node")

        start_id = self.count_nodes() + 1

        # Add nodes
        for i in range(n):
            node_label = str(start_id + i) if label else None
            attrs = {}
            if type:
                attrs["type"] = type
            self.add_node(label=node_label, **attrs)

        # Add edges between all pairs
        edge_attrs = {}
        if rel:
            edge_attrs["rel"] = rel

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-loop
                    if keep_loops:
                        source = str(start_id + i)
                        self.add_edge(source, source, **edge_attrs)
                elif i < j or self.is_directed():
                    # For undirected, only add each edge once (i < j)
                    # For directed, add both directions
                    source = str(start_id + i)
                    target = str(start_id + j)
                    self.add_edge(source, target, **edge_attrs)

        return self

    def add_balanced_tree(
        self,
        k: int,
        h: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """
        Add a balanced k-ary tree to the current graph.

        Creates a tree where each internal node has exactly `k` children, with height `h`. The total
        number of nodes will be equal to `(k^(h+1) - 1) / (k - 1)`.

        Parameters
        ----------
        k
            Number of children per node (branching factor).
        h
            Height of the tree (root has height 0).
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
            Optional relationship (`rel`) label for all edges.

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

        print(graph.count_nodes())  # 15 (1 + 2 + 4 + 8)
        ```

        %family Graph Generators
        """
        if k < 1:
            raise ValueError("Branching factor k must be at least 1")
        if h < 0:
            raise ValueError("Height h cannot be negative")

        start_id = self.count_nodes() + 1
        node_count = 0

        # Calculate total nodes: sum of k^i for i from 0 to h
        # For k=1: total = h+1 (linear chain)
        # For k>1: total = (k^(h+1) - 1) / (k - 1)

        # Add nodes level by level
        edge_attrs = {}
        if rel:
            edge_attrs["rel"] = rel

        # Use a queue to track parent nodes and their children
        from collections import deque

        queue = deque()

        # Add root
        root_label = str(start_id + node_count) if label else None
        attrs = {}
        if type:
            attrs["type"] = type
        self.add_node(label=root_label, **attrs)
        # Get the ID of the newly added node
        if root_label:
            root_id = self.get_id_from_label(root_label)
        else:
            root_id = self.count_nodes()  # Last added node ID
        queue.append((root_id, 0))  # (node_id, level)
        node_count += 1

        # Build tree level by level
        while queue:
            parent_id, level = queue.popleft()

            if level < h:
                # Add k children for this parent
                for _ in range(k):
                    child_label = str(start_id + node_count) if label else None
                    attrs = {}
                    if type:
                        attrs["type"] = type
                    self.add_node(label=child_label, **attrs)
                    # Get the ID of newly added node
                    if child_label:
                        child_id = self.get_id_from_label(child_label)
                    else:
                        child_id = self.count_nodes()  # Last added node
                    self.add_edge(parent_id, child_id, **edge_attrs)
                    queue.append((child_id, level + 1))
                    node_count += 1

        return self

    def add_grid_2d(
        self,
        rows: int,
        cols: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """
        Add a 2D grid/lattice graph to the current graph.

        Creates a rectangular grid where each interior node is connected to its four neighbors
        (up, down, left, right). Edge nodes have fewer connections.

        Parameters
        ----------
        rows
            Number of rows in the grid.
        cols
            Number of columns in the grid.
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
            Optional relationship (`rel`) label for all edges.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_grid_2d(3, 4)  # 3×4 grid

        print(graph.count_nodes())  # 12
        print(graph.count_edges())  # 17 (edges: 3*3 + 2*4)
        ```

        %family Graph Generators
        """
        if rows < 1 or cols < 1:
            raise ValueError("Grid must have at least 1 row and 1 column")

        start_id = self.count_nodes() + 1

        # Add nodes in grid layout
        for i in range(rows):
            for j in range(cols):
                if label:
                    node_label = f"({i},{j})"
                else:
                    node_label = None
                attrs = {}
                if type:
                    attrs["type"] = type
                self.add_node(label=node_label, **attrs)

        # Add edges
        edge_attrs = {}
        if rel:
            edge_attrs["rel"] = rel

        for i in range(rows):
            for j in range(cols):
                if label:
                    current = f"({i},{j})"
                else:
                    # Need to use node IDs so get from position in creation order
                    current_idx = start_id + i * cols + j

                    # Since we can't reliably get IDs without labels, we need to track them
                    # For now, use position-based ID lookup
                    current = start_id + i * cols + j

                # Connect to right neighbor (if exists)
                if j < cols - 1:
                    if label:
                        right = f"({i},{j + 1})"
                    else:
                        right = start_id + i * cols + (j + 1)
                    self.add_edge(current, right, **edge_attrs)

                # Connect to bottom neighbor (if exists)
                if i < rows - 1:
                    if label:
                        bottom = f"({i + 1},{j})"
                    else:
                        bottom = start_id + (i + 1) * cols + j
                    self.add_edge(current, bottom, **edge_attrs)

        return self

    def add_prism_graph(
        self,
        n: int,
        type: Optional[str] = None,
        label: bool = True,
        rel: Optional[str] = None,
    ) -> "Graph":
        """
        Add a prism graph to the current graph.

        Creates two parallel n-cycles connected by n vertical edges, forming a 3D prism structure
        when visualized.

        Parameters
        ----------
        n
            Number of nodes in each cycle (minimum `3`).
        type
            Optional node type/category for all nodes.
        label
            If `True`, label nodes with their IDs.
        rel
            Optional relationship label for all edges.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_prism_graph(5)

        print(graph.count_nodes())  # 10 (2 cycles of 5)
        print(graph.count_edges())  # 15 (5+5 cycle edges + 5 vertical)
        ```

        %family Graph Generators
        """
        if n < 3:
            raise ValueError("Prism must have at least 3 nodes per cycle")

        start_id = self.count_nodes() + 1

        # Add first cycle
        cycle1_start = start_id
        for i in range(n):
            if label:
                node_label = f"{cycle1_start + i}_top"
            else:
                node_label = None
            attrs = {}
            if type:
                attrs["type"] = type
            self.add_node(label=node_label, **attrs)

        # Connect first cycle
        edge_attrs = {}
        if rel:
            edge_attrs["rel"] = rel
        for i in range(n):
            if label:
                current = f"{cycle1_start + i}_top"
                next_node = f"{cycle1_start + ((i + 1) % n)}_top"
            else:
                current = cycle1_start + i
                next_node = cycle1_start + ((i + 1) % n)
            self.add_edge(current, next_node, **edge_attrs)

        # Add second cycle
        cycle2_start = cycle1_start + n
        for i in range(n):
            if label:
                node_label = f"{cycle2_start + i}_bottom"
            else:
                node_label = None
            attrs = {}
            if type:
                attrs["type"] = type
            self.add_node(label=node_label, **attrs)

        # Connect second cycle
        for i in range(n):
            if label:
                current = f"{cycle2_start + i}_bottom"
                next_node = f"{cycle2_start + ((i + 1) % n)}_bottom"
            else:
                current = cycle2_start + i
                next_node = cycle2_start + ((i + 1) % n)
            self.add_edge(current, next_node, **edge_attrs)

        # Connect corresponding nodes between cycles
        for i in range(n):
            if label:
                top_node = f"{cycle1_start + i}_top"
                bottom_node = f"{cycle2_start + i}_bottom"
            else:
                top_node = cycle1_start + i
                bottom_node = cycle2_start + i
            self.add_edge(top_node, bottom_node, **edge_attrs)

        return self

    def add_gnp_graph(
        self,
        n: int,
        p: float,
        directed: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "Graph":
        """
        Add an Erdős-Rényi G(n,p) random graph to the current graph.

        Each possible edge is included independently with probability `p`. This model is useful for
        generating random graphs with a specific edge density.

        Parameters
        ----------
        n
            Number of nodes to add.
        p
            Probability of edge creation, in range of `[0.0, 1.0]`.
        directed
            If provided and differs from graph's directed setting, creates a temporary graph with
            the specified directedness.
        seed
            Random seed for reproducibility.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = Graph()
        graph.add_gnp_graph(n=50, p=0.1, seed=23)

        print(graph.count_nodes())  # 50
        # Expected ~245 edges (50*49/2 * 0.1)
        ```

        %family Graph Generators
        """
        if directed is not None and directed != self.is_directed():
            # Create temporary graph with desired directedness
            temp = Graph(directed=directed)
            temp._graph.generate_gnp_graph(n, p, seed)
            # Copy nodes and edges to this graph
            for node_id in temp.get_node_ids(limit=None):
                node_label = temp.get_label_from_id(node_id)
                self.add_node(label=node_label)
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
        """
        Add an Erdős-Rényi G(n,m) random graph to the current graph.

        Creates a graph with `n` nodes and exactly m randomly placed edges. This model is useful for
        generating random graphs with a fixed number of edges.

        Parameters
        ----------
        n
            Number of nodes to add.
        m
            Number of edges to create.
        directed
            If provided and differs from graph's directed setting, creates temporary graph with
            specified directedness.
        seed
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
            .add_gnm_graph(n=50, m=100, seed=23)
        )

        print(graph.count_nodes())  # 50
        print(graph.count_edges())  # 100
        ```

        %family Graph Generators
        """
        if directed is not None and directed != self.is_directed():
            # Create temporary graph with desired directedness
            temp = Graph(directed=directed)
            temp._graph.generate_gnm_graph(n, m, seed)
            # Copy nodes and edges to this graph
            for node_id in temp.get_node_ids(limit=None):
                node_label = temp.get_label_from_id(node_id)
                self.add_node(label=node_label)
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
        """
        Add a Barabási-Albert preferential attachment graph.

        Grows a graph by adding nodes one at a time, with each new node connecting to `m` existing
        nodes with probability proportional to degree. This creates scale-free networks with
        power-law degree distributions.

        Parameters
        ----------
        n
            Number of nodes to add.
        m
            Number of edges to attach from each new node.
        seed
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
            .add_pa_graph(n=100, m=3, seed=23)
        )

        print(graph.count_nodes())  # 100
        print(graph.count_edges())  # (100-m)*m = 294
        ```

        %family Graph Generators
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
        """
        Add a Watts-Strogatz small-world graph.

        Creates a ring lattice with rewiring probability `p`, producing graphs with high clustering
        and short path lengths. Small-world networks model many real-world phenomena.

        Parameters
        ----------
        n
            Number of nodes to add.
        k
            Each node connected to k nearest neighbors (must be even).
        p
            Probability of rewiring each edge (`0.0` to `1.0`).
        seed
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
            .add_smallworld_graph(n=100, k=6, p=0.1, seed=23)
        )

        print(graph.count_nodes())  # 100
        ```

        %family Graph Generators
        """
        self._graph.generate_smallworld_graph(n, k, p, seed)
        return self

    def add_johnson_graph(self, n: int, k: int) -> "Graph":
        """
        Add a Johnson graph J(n, k) to this graph.

        Creates a graph where nodes represent `k`-subsets of `{0, 1, ..., n-1}`, and two nodes are
        adjacent if their subsets intersect in exactly `k-1` elements (i.e., they differ by exactly
        one element).

        Johnson graphs are distance-regular and `k(n-k)`-regular. They generalize to the family
        `J(n,k,s)` where `s` is the intersection size; this method implements the standard case
        where `s = k - 1`. When `k = 0` or `k = n`, the graph is trivial (single node, no edges).

        Parameters
        ----------
        n
            Size of the base set. Must be non-negative.
        k
            Size of each subset. Must satisfy `0 <= k <= n`.

        Returns
        -------
        Graph
            Self for method chaining.

        Raises
        ------
        ValueError
            If `k=` > `n=`.

        Properties
        ----------
        - **Nodes**: `C(n,k)` nodes where `C` is the binomial coefficient "n choose k"
        - **Degree**: Each node has degree `k(n-k)` (regular graph)
        - **Distance-regular**: Distances between nodes follow regular pattern
        - **Undirected**: All edges are bidirectional

        Relationship to Other Graphs
        -----------------------------
        - **Kneser graphs K(n,k)**: Nodes adjacent if subsets are disjoint.
          Johnson J(n,k) and Kneser K(n,k) are complementary for certain parameters.
        - **Hamming graphs**: Johnson graphs are a special case of Hamming graphs.
        - **Petersen graph**: J(5,2) is the Petersen graph (10 nodes, degree 3).

        Complexity
        ----------
        - Time: O(C(n,k)² × k) where C(n,k) is "n choose k"
        - Space: O(C(n,k) + edges)

        Examples
        --------
        Create J(4,2) - the graph of 2-subsets of {0,1,2,3}:

        ```python
        import edgecase as ec

        graph = ec.Graph(directed=False).add_johnson_graph(n=4, k=2)
        print(graph.count_nodes())  # 6 (C(4,2) = 6)
        print(graph.count_edges())  # 12

        # Each node is a 2-subset
        nodes = [graph.get_label_from_id(nid) for nid in graph.get_node_ids(limit=None)]
        print(nodes)  # ['0,1', '0,2', '0,3', '1,2', '1,3', '2,3']

        # Verify regularity: each node has degree k(n-k) = 2*2 = 4
        degrees = [graph.get_degree(nid) for nid in graph.get_node_ids(limit=None)]
        print(set(degrees))  # {4}
        ```

        Create the Petersen graph J(5,2):

        ```python
        johnson_5_2 = ec.Graph(directed=False).add_johnson_graph(n=5, k=2)
        print(johnson_5_2.count_nodes())  # 10 (C(5,2) = 10)
        print(johnson_5_2.count_edges())  # 30

        # Verify regularity: degree = k(n-k) = 2*(5-2) = 6
        degrees = [johnson_5_2.get_degree(nid) for nid in johnson_5_2.get_node_ids(limit=None)]
        print(f"Degree: {degrees[0]}")  # 6

        # Note: The Petersen graph is actually the Kneser graph K(5,2),
        # not Johnson J(5,2). J(5,2) has 10 nodes and is 6-regular,
        # while Petersen has 10 nodes but is 3-regular.
        ```

        %family Graph Generators
        """
        self._graph.generate_johnson_graph(n, k)
        return self

    def add_random_geometric_graph(
        self,
        n: int,
        radius: float,
        dim: int = 2,
        p_norm: float = 2.0,
        seed: Optional[int] = None,
    ) -> "Graph":
        """
        Add a random geometric graph.

        Places `n` nodes uniformly at random in the unit hypercube and connects nodes whose
        distance is at most the value provided in `radius=`. This creates spatial networks where
        nearby points are more likely to be connected.

        Random geometric graphs are useful for modeling:

        - wireless networks (nodes in radio range)
        - social networks (geographically close individuals)
        - biological networks (spatially proximate cells)
        - ad-hoc networks and sensor networks

        Parameters
        ----------
        n
            Number of nodes to add.
        radius
            Maximum distance for edge creation. Nodes within this distance are connected. Must be
            non-negative.
        dim
            Number of dimensions for the unit hypercube (default `2`). Use `2` for planar graphs,
            `3` for spatial networks.
        p_norm
            Minkowski distance parameter (default `2.0` for Euclidean). Common values are `1.0`
            (Manhattan), `2.0` (Euclidean), `inf` (Chebyshev).
        seed
            Random seed for reproducibility.

        Returns
        -------
        Graph
            Self for method chaining.

        Raises
        ------
        ValueError
            If `radius=` is negative, `dim=` < `1`, or `p_norm=` <= `0`.

        Properties
        ----------
        - **Undirected**: All edges are bidirectional.
        - **Spatial embedding**: Nodes have positions in `[0, 1]^dim`.
        - **Node labels**: Position stored as JSON array, e.g., `"[0.234, 0.567]"`.
        - **Local clustering**: High clustering due to geometric proximity.
        - **Threshold behavior**: Graph connectivity exhibits phase transition as radius increases
        past a critical threshold.

        Complexity
        ----------
        - Time: `O(n²)` for distance comparisons (naive algorithm)
        - Space: `O(n × dim)` for positions plus O(edges) for graph

        Examples
        --------
        Create a 2D random geometric graph:

        ```python
        import edgecase as ec

        graph = ec.Graph(directed=False).add_random_geometric_graph(
            n=100, radius=0.2, seed=23
        )
        print(graph.count_nodes())  # 100

        # Nodes have position labels
        node_id = graph.get_node_ids(limit=1)[0]
        label = graph.get_label_from_id(node_id)
        print(label)  # e.g., '[0.374540, 0.950714]'
        ```

        Create a 3D spatial network:

        ```python
        spatial_graph = ec.Graph(directed=False).add_random_geometric_graph(
            n=50, radius=0.3, dim=3, seed=123
        )
        print(f"Nodes: {spatial_graph.count_nodes()}")  # 50
        print(f"Edges: {spatial_graph.count_edges()}")  # varies by random seed
        ```

        Use Manhattan distance (p_norm=1.0):

        ```python
        manhattan_graph = ec.Graph(directed=False).add_random_geometric_graph(
            n=100, radius=0.25, p_norm=1.0, seed=23
        )
        # With Manhattan distance, the "ball" of radius r is a rotated square
        print(f"Edges: {manhattan_graph.count_edges()}")
        ```

        %family Graph Generators
        """
        self._graph.generate_random_geometric_graph(n, radius, dim, p_norm, seed)
        return self

    # ==============================================================================
    # Node and Edge Accessors
    # ==============================================================================

    def get_node(self, node: Union[int, str]) -> Optional[rust.Node]:
        """
        Get a node object by ID or label.

        Retrieves the `Node` object for the specified node. The `Node` object contains `id`,
        `label`, and attributes properties. Returns `None` if node doesn't exist.

        Parameters
        ----------
        node
            Node ID (str) or internal ID (int) to retrieve.

        Returns
        -------
        Node | None
            `Node` object or `None` if not found.

        Examples
        --------
        Access node properties:

        ```python
        graph = Graph().add_node(label="A", color="red", size=10)

        node = graph.get_node("A")
        print(node.id)  # 'A'
        print(node.attributes)  # {'color': 'red', 'size': 10}
        ```

        %family Node and Edge Accessors
        """
        node_id = self._resolve_node_id(node, auto_create=False)
        return self._graph.get_node(node_id)

    def get_edge(self, edge_id: int) -> Optional[rust.Edge]:
        """
        Get an edge object by an edge ID.

        Retrieves the `Edge` object for the specified edge ID. The `Edge` object contains id, source,
        target, weight, and attributes properties. Returns `None` if edge doesn't exist.

        Parameters
        ----------
        edge_id
            Edge ID to retrieve.

        Returns
        -------
        Edge | None
            `Edge` object or `None` if not found.

        Examples
        --------
        Access edge properties:

        ```python
        graph = Graph().add_node("A").add_node("B").add_edge("A", "B", weight=1.5, type="friend")

        edge_ids = graph.get_edge_ids()
        edge = graph.get_edge(edge_ids[0])

        print(edge.source)  # Node ID for 'A'
        print(edge.target)  # Node ID for 'B'
        print(edge.weight)  # 1.5
        print(edge.attributes)  # {'type': 'friend'}
        ```

        %family Node and Edge Accessors
        """
        return self._graph.get_edge(edge_id)

    # ==============================================================================
    # Graph Inspection
    # ==============================================================================

    def get_node_ids(self, limit: Optional[int] = 1000) -> list[int]:
        """
        Return list of node IDs.

        Returns node IDs in the graph up to the specified limit. Node IDs are unique integers
        (`u64`) assigned to each node, useful for iterating over nodes or performing operations on
        specific nodes.

        Parameters
        ----------
        limit
            Maximum number of node IDs to return. Pass `None` to return all node IDs (warning: may
            be slow for large graphs). Default limit of `1000` prevents performance issues when
            working with large graphs.

        Returns
        -------
        list[int]
            List of node IDs in the graph, limited to `limit`.

        Examples
        --------
        Get first 1000 node IDs (default):

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )

        ids = graph.get_node_ids()  # Returns up to 1000 node IDs
        print(f"Showing {len(ids)} node IDs: {ids}")
        ```

        Get a smaller sample:

        ```python
        graph = Graph()
        for i in range(100):
            graph.add_node(label=f"Node_{i}")

        # Get only first 10 node IDs for preview
        sample_ids = graph.get_node_ids(limit=10)
        print(f"Sample: {sample_ids}")
        ```

        Get all node IDs regardless of size:

        ```python
        graph = Graph()
        for i in range(5):
            graph.add_node(label=f"Node_{i}")

        # Use None to get all nodes (may be slow for large graphs!)
        all_ids = graph.get_node_ids(limit=None)
        print(f"Total nodes: {len(all_ids)}")
        ```

        %family Graph Inspection
        """
        ids = self._graph.get_node_ids()
        return ids if limit is None else ids[:limit]

    def get_edge_ids(self, limit: Optional[int] = 1000) -> list[int]:
        """
        Return list of edge IDs.

        Returns edge IDs in the graph up to the specified limit. Edge IDs are unique integers
        assigned to each edge, useful for direct edge access and manipulation.

        Parameters
        ----------
        limit
            Maximum number of edge IDs to return. Pass `None` to return all edge IDs (warning: may
            be slow for large graphs). Default limit of `1000` prevents performance issues when
            working with large graphs.

        Returns
        -------
        list[int]
            List of edge IDs in the graph, limited to `limit`.

        Examples
        --------
        Get first 1000 edge IDs (default):

        ```python
        import edgecase as ec

        graph = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )

        ids = graph.get_edge_ids()  # Returns up to 1000 edge IDs
        print(f"Showing {len(ids)} edge IDs: {ids}")
        ```

        Get a smaller sample:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        for i in range(20):
            graph.add_node(label=f"Node_{i}")
        for i in range(19):
            graph.add_edge(source=f"Node_{i}", target=f"Node_{i+1}")

        # Get only first 10 edge IDs for preview
        sample_ids = graph.get_edge_ids(limit=10)
        print(f"Sample: {sample_ids}")
        ```

        Get all edge IDs regardless of size:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        for i in range(5):
            graph.add_node(label=f"Node_{i}")
        for i in range(4):
            graph.add_edge(source=f"Node_{i}", target=f"Node_{i+1}")

        # Use None to get all edges (may be slow for large graphs!)
        all_ids = graph.get_edge_ids(limit=None)
        print(f"Total edges: {len(all_ids)}")
        ```

        %family Graph Inspection
        """
        ids = self._graph.edge_ids()
        return ids if limit is None else ids[:limit]

    def get_node_ids_by_edge_id(self, edge_id: int) -> tuple[int, int] | None:
        """
        Get edge (source, target) node IDs by edge ID.

        Retrieves the source and target node IDs for a specific edge using its edge ID. Returns
        `None` if the edge ID doesn't exist.

        Parameters
        ----------
        edge_id
            The edge ID to look up.

        Returns
        -------
        tuple[int, int] | None
            A tuple of (`source_node_id`, `target_node_id`) if the edge exists, `None` otherwise.

        Examples
        --------
        Look up an edge by its ID:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )

        # Get first edge ID
        edge_id = graph.get_edge_ids()[0]

        # Look up the edge
        source, target = graph.get_node_ids_by_edge_id(edge_id=edge_id)
        print(f"Edge {edge_id} goes from {source} to {target}")

        # Non-existent edge returns None
        result = graph.get_node_ids_by_edge_id(edge_id=9999)
        print(result)  # None
        ```

        %family Graph Inspection
        """
        return self._graph.get_edge_by_id(edge_id)

    def get_id_from_label(self, label: str) -> Optional[int]:
        """
        Get the integer ID for a node by its label.

        Translates a human-readable string label to the internal `u64` node ID. Useful when you need
        the ID for operations that require integer identifiers.

        Parameters
        ----------
        label
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

        alice_id = graph.get_id_from_label("Alice")
        print(alice_id)  # 1

        unknown_id = graph.get_id_from_label("Unknown")
        print(unknown_id)  # None
        ```

        %family Graph Inspection
        """
        try:
            return self._graph.id_from_label(label)
        except (ValueError, Exception):
            return None

    def get_label_from_id(self, node_id: int) -> Optional[str]:
        """
        Get the label for a node by its ID.

        Translates an internal `u64` node ID to its human-readable string label. Returns `None` if
        the node doesn't exist or has no label assigned.

        Parameters
        ----------
        node_id
            The `u64` node ID to look up.

        Returns
        -------
        str | None
            The string label if it exists, or `None` if the node has no label or doesn't exist in
            the graph.

        Examples
        --------
        Retrieve labels for node IDs:

        ```python
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
        )

        label = graph.get_label_from_id(1)
        print(label)  # "Alice"

        no_label = graph.get_label_from_id(999)
        print(no_label)  # None
        ```

        %family Graph Inspection
        """
        try:
            return self._graph.label_from_id(node_id)
        except (ValueError, Exception):
            return None

    def get_node_creation_order(self) -> list[int]:
        """
        Get nodes in their creation order.

        Returns node IDs ordered by when they were added to the graph. The first node created
        appears first, the most recently created node appears last. Useful for temporal analysis or
        recreating graph construction history.

        Returns
        -------
        list[int]
            Node IDs in chronological creation order.

        Examples
        --------
        Track node creation order:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        creation_order = graph.get_node_creation_order()
        print(creation_order)  # [<id_A>, <id_B>, <id_C>]

        # Get labels in creation order
        labels = [graph.get_node(nid).label for nid in creation_order]
        print(labels)  # ['A', 'B', 'C']
        ```

        Select recently created nodes:

        ```python
        graph = ec.Graph().add_path_graph(10)
        # Select the last 5 nodes created
        recent = graph.get_node_creation_order()[-5:]
        graph.select_nodes_by_id(recent)
        ```

        %family Graph Inspection
        """
        return self._graph.get_node_creation_order()

    def get_edge_creation_order(self) -> list[int]:
        """
        Get edges in their creation order.

        Returns edge IDs ordered by when they were added to the graph. The first edge created
        appears first, the most recently created edge appears last. Useful for temporal analysis or
        understanding edge addition patterns.

        Returns
        -------
        list[int]
            Edge IDs in chronological creation order.

        Examples
        --------
        Track edge creation order:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        creation_order = graph.get_edge_creation_order()
        print(len(creation_order))  # 3

        # Get endpoints in creation order
        for eid in creation_order:
            source, target = graph.get_node_ids_by_edge_id(eid)
            print(f"Edge {eid}: {source} -> {target}")
        ```

        Analyze edge addition patterns:

        ```python
        graph = ec.Graph().add_path_graph(20)
        # Select first 10 edges added
        early_edges = graph.get_edge_creation_order()[:10]
        graph.select_edges_by_id(early_edges)

        # Check if edges were added chronologically by weight
        order = graph.get_edge_creation_order()
        weights = [graph.get_edge(eid).weight for eid in order]
        ```

        %family Graph Inspection
        """
        return self._graph.get_edge_creation_order()

    def get_edges_as_tuples(self) -> list[tuple[int, int, int]]:
        """
        Return list of edges as (id, source, target) tuples.

        Returns all edges in the graph as tuples containing the edge ID and the source and target
        node IDs. Useful for iterating over all edges or analyzing edge structure.

        Returns
        -------
        list[tuple[int, int, int]]
            List of tuples where each tuple is (edge_id, source_id, target_id).

        Examples
        --------
        Get all edges in a graph:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        for edge_id, source, target in graph.get_edges_as_tuples():
            print(f"Edge {edge_id}: {source} -> {target}")
        ```

        %family Graph Inspection
        """
        return self._graph.edges()

    def get_edges_str_repr(
        self, limit: Optional[int] = 1000, use_labels: bool = False
    ) -> list[str]:
        """
        Get edges as formatted string representations.

        Returns edge representations as strings showing node identifiers with arrow notation. The
        format depends on whether the graph is directed or undirected:

        - directed graphs: "source->target"
        - undirected graphs: "source--target"

        Parameters
        ----------
        limit
            Maximum number of edge strings to return. Pass `None` to return all edges (warning: may
            be slow for large graphs). Default limit of `1000` prevents performance issues when
            working with large graphs.
        use_labels
            If `True`, preferentially use node labels instead of IDs. Falls back to node IDs when
            labels are not available for a node. If `False`, always use node IDs.

        Returns
        -------
        list[str]
            List of edge strings with arrow notation indicating direction. Uses node IDs by default,
            or labels when `use_labels=True` and available.

        Examples
        --------
        Get formatted edges with node IDs:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(graph.get_edges_str_repr())  # ['1->2', '2->3']
        ```

        Get formatted edges with labels:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(graph.get_edges_str_repr(use_labels=True))  # ['A->B', 'B->C']
        ```

        Undirected graphs use double dashes:

        ```python
        graph = (
            ec.Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge(source="X", target="Y")
        )

        print(graph.get_edges_str_repr())  # ['1--2']
        print(graph.get_edges_str_repr(use_labels=True))  # ['X--Y']
        ```

        Mixed labels (some nodes have labels, others don't):

        ```python
        graph = (
            ec.Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")  # Add node with label
        )

        # Get the node IDs to create an edge
        alice_id = graph.get_id_from_label(label="Alice")
        bob_id = graph.get_id_from_label(label="Bob")
        graph.add_edge(source=alice_id, target=bob_id)

        # Use labels in string representation
        print(graph.get_edges_str_repr(use_labels=True))  # ['Alice--Bob']
        ```

        Get all edges regardless of size:

        ```python
        import edgecase as ec

        graph = ec.Graph()
        for i in range(5):
            graph.add_node(label=f"Node_{i}")
        for i in range(4):
            graph.add_edge(source=f"Node_{i}", target=f"Node_{i+1}")

        # Use None to get all edges (may be slow for large graphs!)
        all_edges = graph.get_edges_str_repr(limit=None, use_labels=True)
        print(f"Total edges: {len(all_edges)}")
        ```

        %family Graph Inspection
        """
        if use_labels:
            # Build edge strings using labels when available
            separator = "->" if self.is_directed() else "--"
            edge_list = self.get_edges_as_tuples()
            edge_strs = []

            for edge_id, source_id, target_id in edge_list:
                # Try to get labels, fall back to IDs
                source_label = self.get_label_from_id(source_id)
                target_label = self.get_label_from_id(target_id)

                source_str = source_label if source_label is not None else str(source_id)
                target_str = target_label if target_label is not None else str(target_id)

                edge_strs.append(f"{source_str}{separator}{target_str}")

            return edge_strs if limit is None else edge_strs[:limit]
        else:
            # Use node IDs (default behavior)
            edge_strs = self._graph.get_edges()
            return edge_strs if limit is None else edge_strs[:limit]

    def has_node(self, node: Union[int, str]) -> bool:
        """
        Check if a node exists in the graph.

        Verifies whether a node (specified by ID or label) exists in the graph. Returns `False` for
        invalid inputs or missing nodes.

        Parameters
        ----------
        node
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

        %family Graph Inspection
        """
        try:
            node_id = self._resolve_node_id(node, auto_create=False)
            return self._graph.has_node_id(node_id)
        except (ValueError, TypeError):
            return False

    def has_edge(self, source: Union[int, str], target: Union[int, str]) -> bool:
        """
        Check if an edge exists between two nodes.

        For undirected graphs, the order of source and target doesn't matter. For directed graphs,
        checks specifically for an edge from source to target.

        Parameters
        ----------
        source
            Source node ID or label.
        target
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
            .add_edge(source="A", target="B")
        )

        print(graph.has_edge(source="A", target="B"))  # True
        print(graph.has_edge(source="B", target="A"))  # False (directed)
        ```

        For undirected graphs, order doesn't matter:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )

        print(graph.has_edge(source="A", target="B"))  # True
        print(graph.has_edge(source="B", target="A"))  # True (undirected)
        ```

        %family Graph Inspection
        """
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)
        edge_ids = self._graph.find_edge_ids(source_id, target_id)
        return len(edge_ids) > 0

    def has_edge_id(self, edge_id: int) -> bool:
        """
        Check if an edge with given ID exists.

        Verifies whether a specific edge ID is present in the graph. Useful for validating edge IDs
        before performing operations.

        Parameters
        ----------
        edge_id
            The edge ID to check.

        Returns
        -------
        bool
            `True` if an edge with this ID exists, `False` otherwise.

        Examples
        --------
        Check if edge IDs exist:

        ```python
        import edgecase as ec

        graph = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )

        edge_id = graph.get_edge_ids()[0]
        print(graph.has_edge_id(edge_id=edge_id))  # True
        print(graph.has_edge_id(edge_id=9999))     # False
        ```

        %family Graph Inspection
        """
        return self._graph.has_edge_id(edge_id)

    def find_edge_ids(self, source: Union[int, str], target: Union[int, str]) -> list[int]:
        """
        Find edge IDs between two nodes.

        Returns a list of all edge IDs connecting the source and target nodes. For undirected
        graphs, edges are found regardless of direction. Supports multi-edges (multiple edges
        between the same pair of nodes).

        Parameters
        ----------
        source
            Source node ID or label.
        target
            Target node ID or label.

        Returns
        -------
        list[int]
            List of edge IDs. Empty list if no edges exist.

        Examples
        --------
        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge(source="A", target="B")
        graph.add_edge(source="A", target="B")  # Multi-edge

        edge_ids = graph.find_edge_ids(source="A", target="B")
        print(len(edge_ids))  # 2
        ```

        %family Graph Inspection
        """
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)
        return self._graph.find_edge_ids(source_id, target_id)

    def count_nodes(self) -> int:
        """
        Count the total number of nodes in the graph.

        Returns the current number of nodes (vertices) in the graph. This count includes all nodes
        that have been added and not deleted.

        Returns
        -------
        int
            The total number of nodes in the graph.

        Examples
        --------
        Count nodes in a simple graph:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        print(graph.count_nodes())  # 3
        ```

        Track nodes after deletion:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .delete_node("B")
        )
        print(graph.count_nodes())  # 2
        ```

        %family Graph Inspection
        """
        return self._graph.num_nodes()

    def count_edges(self) -> int:
        """
        Count the total number of edges in the graph.

        Returns the current number of edges (links/connections) in the graph. For undirected graphs,
        each edge is counted once. For directed graphs, each directed edge is counted separately
        (`A->B` is different from `B->A`).

        Returns
        -------
        int
            The total number of edges in the graph.

        Examples
        --------
        Count edges in a simple graph:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )
        print(graph.count_edges())  # 2
        ```

        Directed vs undirected edge counting:

        ```python
        # Directed graph
        directed = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")
        )
        print(directed.count_edges())  # 2 (A->B and B->A are separate)

        # Undirected graph
        undirected = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )
        print(undirected.count_edges())  # 1
        ```

        Track edges after deletion:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .delete_edge(source="A", target="B")
        )
        print(graph.count_edges())  # 1
        ```

        %family Graph Inspection
        """
        return self._graph.num_edges()

    def get_graph_name(self) -> Optional[str]:
        """
        Get the name of the graph.

        Returns the human-readable name assigned to this graph, if any. The name is an optional
        identifier that can be set when creating the graph or updated later. It's useful for
        distinguishing between multiple graphs in analysis or visualization contexts.

        Returns
        -------
        str | None
            The graph's name if set, or None if no name was assigned.

        Examples
        --------
        Get graph name:

        ```python
        graph = Graph(graph_name="Social Network")
        print(graph.get_graph_name())  # "Social Network"
        ```

        Handle graphs without names:

        ```python
        graph = Graph()
        name = graph.get_graph_name()
        print(name)  # None

        if name:
            print(f"Analyzing {name}")
        else:
            print("Analyzing unnamed graph")
        ```

        %family Graph Inspection
        """
        return self._graph_name

    # ==============================================================================
    # Graph Properties
    # ==============================================================================

    def is_directed(self) -> bool:
        """
        Check if the graph is directed.

        Returns whether this graph treats edges as directed (with distinct source and target) or
        undirected (bidirectional connections). This property is set when the graph is created and
        cannot be changed afterward.

        In directed graphs:

        - edge `A->B` is different from `B->A`
        - each direction must be explicitly added
        - useful for modeling asymmetric relationships (follows, citations, etc.)

        In undirected graphs:

        - edge `A--B` is the same as `B--A`
        - adding `A--B` automatically adds `B--A`
        - useful for modeling symmetric relationships (friendship, proximity, etc.)

        Returns
        -------
        bool
            `True` if the graph is directed, `False` if undirected.

        Examples
        --------
        Check graph directionality:

        ```python
        directed_graph = Graph(directed=True)
        print(directed_graph.is_directed())  # True

        undirected_graph = Graph(directed=False)
        print(undirected_graph.is_directed())  # False
        ```

        Use directionality to customize behavior:

        ```python
        graph = Graph(directed=True).add_node(label="A").add_node(label="B")

        if graph.is_directed():
            # Add both directions for bidirectional relationship
            graph.add_edge(source="A", target="B")
            graph.add_edge(source="B", target="A")
        else:
            # Single edge suffices for undirected
            graph.add_edge(source="A", target="B")
        ```

        %family Graph Properties
        """
        return self._graph.is_directed

    def is_weighted(self) -> bool:
        """
        Check if the graph is weighted.

        Returns whether this graph supports edge weights. This property is set when the graph is
        created and cannot be changed afterward. Weighted graphs can store numeric weights on edges,
        which are used in algorithms like shortest path, minimum spanning tree, etc.

        In weighted graphs:

        - each edge can have an associated numeric weight
        - weights represent costs, distances, capacities, etc.
        - algorithms consider weights when computing paths and metrics

        In unweighted graphs:

        - all edges are treated equally (implicit weight of `1`)
        - weights are ignored or unavailable
        - algorithms use hop count instead of weight sum

        Returns
        -------
        bool
            `True` if the graph supports weights, `False` otherwise.

        Examples
        --------
        Check if graph is weighted:

        ```python
        weighted_graph = Graph(weighted=True)
        print(weighted_graph.is_weighted())  # True

        unweighted_graph = Graph()
        print(unweighted_graph.is_weighted())  # False
        ```

        Add weighted edges only if supported:

        ```python
        graph = Graph(weighted=True).add_node(label="A").add_node(label="B")

        if graph.is_weighted():
            graph.add_edge(source="A", target="B", weight=2.5)
        else:
            graph.add_edge(source="A", target="B")
        ```

        %family Graph Properties
        """
        return self._graph.is_weighted

    def is_simple(self) -> bool:
        """
        Check if the graph is simple (no self-loops or multi-edges).

        A simple graph contains no self-loops (edges from a node to itself) and no multi-edges
        (multiple edges between the same pair of nodes). This is a fundamental graph property used
        in many theoretical results.

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

        print(graph.is_simple())  # True

        # Add a self-loop
        graph.add_edge("A", "A")
        print(graph.is_simple())  # False
        ```

        Multi-edges also make a graph non-simple:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
            .add_edge("A", "B")  # Duplicate edge (auto-assigned different IDs)
        )

        print(graph.is_simple())  # False
        ```

        %family Graph Properties
        """
        return self._graph.is_simple()

    def is_dag(self) -> bool:
        """
        Check if directed graph is a DAG (Directed Acyclic Graph).

        A DAG is a directed graph with no cycles. You cannot follow directed edges and return to a
        starting node. DAGs are fundamental for dependency resolution, scheduling, and topological
        sorting.

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

        %family Graph Properties
        """
        return self._graph.is_dag()

    def is_connected(self) -> bool:
        """
        Check if graph is connected.

        A graph is connected if there is a path between every pair of nodes. For directed graphs,
        checks weak connectivity (treating edges as undirected).

        Returns
        -------
        bool
            `True` if all nodes are reachable from any node, `False` if the graph has disconnected
            components.

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

        %family Graph Properties
        %seealso is_weakly_connected, is_strongly_connected, count_weakly_connected_components
        """
        return self._graph.is_connected()

    def is_bipartite(self) -> bool:
        """
        Check if graph is bipartite (2-colorable).

        A graph is bipartite if its nodes can be divided into two independent sets such that every
        edge connects nodes from different sets. Equivalently, the graph can be colored with two
        colors such that no adjacent nodes share a color.

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

        %family Graph Properties
        """
        return self._graph.is_bipartite()

    def is_tree(self) -> bool:
        """
        Check if the graph is a tree.

        A tree is a connected, acyclic, undirected graph. For a graph with *n* nodes to be a tree
        it:

        - must be undirected
        - must be connected (single component)
        - must have exactly *n - 1* edges
        - must be acyclic

        Returns
        -------
        bool
            `True` if graph is a tree, `False` otherwise.

        Notes
        -----
        For directed graphs, use `is_arborescence()` instead. For disconnected forests, use
        `is_forest()`.

        Examples
        --------
        Path graph is a tree:

        ```python
        tree = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(tree.is_tree())  # True
        ```

        Graph with cycle is not a tree:

        ```python
        cycle = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        print(cycle.is_tree())  # False (has cycle)
        ```

        Disconnected graph is not a tree:

        ```python
        disconnected = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            # C is isolated
        )

        print(disconnected.is_tree())  # False (not connected)
        ```

        Star graph is a tree:

        ```python
        star = Graph(directed=False)
        star.add_star_graph(n=5)
        print(star.is_tree())  # True
        ```

        %family Graph Properties
        """
        return self._graph.is_tree()

    def is_forest(self) -> bool:
        """
        Check if the graph is a forest.

        A forest is an acyclic, undirected graph (disjoint union of trees). For a graph with *n*
        nodes and *k* connected components to be a forest it:

        - must be undirected
        - must have exactly *n - k* edges
        - must be acyclic

        Returns
        -------
        bool
            `True` if graph is a forest, `False` otherwise.

        Notes
        -----
        A tree is a special case of a forest with k=1 (single component).

        Examples
        --------
        Single tree is a forest:

        ```python
        tree = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(tree.is_forest())  # True
        ```

        Multiple disjoint trees form a forest:

        ```python
        forest = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")  # Tree 1
            .add_edge("C", "D")  # Tree 2
        )

        print(forest.is_forest())  # True
        print(forest.is_tree())    # False (not connected)
        ```

        Graph with cycle is not a forest:

        ```python
        cycle = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        print(cycle.is_forest())  # False (has cycle)
        ```

        Empty graph is a forest:

        ```python
        empty = Graph(directed=False)
        print(empty.is_forest())  # True
        ```

        %family Graph Properties
        """
        return self._graph.is_forest()

    def is_arborescence(self) -> bool:
        """
        Check if the graph is an arborescence (directed tree).

        An arborescence is a directed graph which:

        - must be directed
        - has exactly one root (a node with in-degree `0`)
        - has other nodes with in-degree exactly `1`
        - must be weakly connected
        - has exactly `n - 1` edges

        This represents a tree structure with a clear parent-child hierarchy flowing from a single
        root.

        Returns
        -------
        bool
            `True` if graph is an arborescence, `False` otherwise.

        Notes
        -----
        For undirected graphs, use `is_tree()` instead. An arborescence is also called a 'directed
        rooted tree' or 'out-tree'.

        Examples
        --------
        Directed tree with single root:

        ```python
        arb = (
            Graph(directed=True)
            .add_node(label="root")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("root", "A")
            .add_edge("root", "B")
            .add_edge("A", "C")
        )

        print(arb.is_arborescence())  # True
        ```

        Multiple roots is not an arborescence:

        ```python
        multi_root = (
            Graph(directed=True)
            .add_node(label="root1")
            .add_node(label="root2")
            .add_node(label="A")
            .add_edge("root1", "A")
            .add_edge("root2", "A")
        )

        print(multi_root.is_arborescence())  # False (A has in-degree 2)
        ```

        Directed cycle is not an arborescence:

        ```python
        cycle = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        print(cycle.is_arborescence())  # False (no root, all in-degree 1)
        ```

        Balanced tree is an arborescence:

        ```python
        tree = Graph(directed=True)
        tree.add_balanced_tree(k=2, h=3)
        print(tree.is_arborescence())  # True
        ```

        %family Graph Properties
        """
        return self._graph.is_arborescence()

    def is_planar(self) -> bool:
        """
        Check if the graph is planar.

        A planar graph can be embedded in a plane without any edge crossings. This method uses
        Euler's formula and necessary conditions to test planarity. For simple planar graphs with
        `n ≥ 3` nodes, the number of edges *e* must satisfy `e ≤ 3n - 6`. For graphs without
        triangles, the stricter bound *e* ≤ *2n - 4* applies.

        Returns
        -------
        bool
            `True` if the graph satisfies planarity conditions, `False` otherwise.

        Notes
        -----
        This is a *necessary but not sufficient* test. Some non-planar graphs may pass this test.
        For complete planarity testing, specialized algorithms implementing Kuratowski's theorem
        would be required. However, this test will correctly identify many non-planar graphs
        efficiently.

        Examples
        --------
        Small graphs are typically planar:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(graph.is_planar())  # True
        ```

        The complete graph K_4 is planar:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(n=4)
        print(graph.is_planar())  # True (4 nodes, 6 edges: 6 ≤ 3 * 4 - 6 = 6)
        ```

        Complete graph K_5 is not planar:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(n=5)
        print(graph.is_planar())  # False (5 nodes, 10 edges: 10 > 3 * 5 - 6 = 9)
        ```

        Trees are always planar:

        ```python
        tree = Graph(directed=False)
        tree.add_balanced_tree(k=3, h=4)
        print(tree.is_planar())  # True (trees have `e = n - 1`)
        ```

        Grid graphs are planar:

        ```python
        grid = Graph(directed=False)
        grid.add_grid_2d(rows=5, cols=5)
        print(grid.is_planar())  # True
        ```

        %family Graph Properties
        """
        return self._graph.is_planar()

    def is_isomorphic(self, other: "Graph") -> bool:
        """
        Check if this graph is isomorphic to another graph.

        Two graphs are isomorphic if there exists a one-to-one correspondence (bijection) between
        their nodes that preserves the adjacency relationship. In simpler terms, they have the same
        structure but possibly different node labels.

        This implementation checks the following necessary conditions:

        - same number of nodes
        - same number of edges
        - same degree sequence
        - same directedness

        Parameters
        ----------
        other
            Another `Graph` object to compare with.

        Returns
        -------
        bool
            `True` if graphs satisfy isomorphism conditions, `False` otherwise.

        Notes
        -----
        This is a heuristic check using necessary but not sufficient conditions. It will correctly
        identify *non-isomorphic* graphs (returning `False`) but may give false positives. A
        complete isomorphism test requires algorithms like VF2. This method is useful as a fast
        filter before more expensive isomorphism tests.

        Examples
        --------
        Same graph structure with different labels:

        ```python
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        g2 = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        print(g1.is_isomorphic(g2))  # True (both are paths of length 2)
        ```

        Different structures are not isomorphic:

        ```python
        path = Graph(directed=False).add_path_graph(n=4)
        star = Graph(directed=False).add_star_graph(n=4)

        print(path.is_isomorphic(star))  # False (different degree sequences)
        ```

        Directed vs undirected are not isomorphic:

        ```python
        directed = Graph(directed=True).add_path_graph(n=3)
        undirected = Graph(directed=False).add_path_graph(n=3)

        print(directed.is_isomorphic(undirected))  # False (different directedness)
        ```

        Complete graphs of same size:

        ```python
        k4_a = Graph(directed=False).add_full_graph(n=4)
        k4_b = Graph(directed=False).add_full_graph(n=4)

        print(k4_a.is_isomorphic(k4_b))  # True (same structure)
        ```

        Different sized graphs:

        ```python
        g3 = Graph(directed=False).add_cycle_graph(n=3)
        g4 = Graph(directed=False).add_cycle_graph(n=4)

        print(g3.is_isomorphic(g4))  # False (different sizes)
        ```

        %family Graph Properties
        """
        return self._graph.is_isomorphic(other._graph)

    def is_regular(self) -> bool:
        """
        Check if the graph is regular (all nodes have the same degree).

        A graph is regular if every node has the same degree. For directed graphs, this means all
        nodes have the same in-degree and the same out-degree (though in-degree and out-degree don't
        need to equal each other).

        Returns
        -------
        bool
            `True` if the graph is regular, `False` otherwise.

        Examples
        --------
        Check if a cycle is regular:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(5)

        print(graph.is_regular())  # True (all nodes have degree 2)
        ```

        Check if a star graph is regular:

        ```python
        graph = Graph()
        graph.add_star_graph(4)

        print(graph.is_regular())  # False (hub has different degree than leaves)
        ```

        Check a complete graph:

        ```python
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        print(graph.is_regular())  # True (all nodes have degree n-1)
        ```

        Notes
        -----
        - empty graphs and single-node graphs are considered regular
        - cycles are 2-regular
        - complete graphs `K_n` are `(n-1)`-regular
        - for directed graphs, checks both in-degree and out-degree regularity

        %family Graph Properties
        """
        from edgecase._core import is_regular

        return is_regular(self._graph)

    def is_k_regular(self, k: int) -> bool:
        """
        Check if the graph is k-regular (all nodes have degree `k`).

        A graph is k-regular if every node has degree exactly `k`. For directed graphs, this means
        all nodes have in-degree `k` and out-degree `k`.

        Parameters
        ----------
        k
            The degree value to check for.

        Returns
        -------
        bool
            `True` if the graph is k-regular, `False` otherwise.

        Examples
        --------
        Check if a cycle is 2-regular:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(6)

        print(graph.is_k_regular(2))  # True (cycles are 2-regular)
        print(graph.is_k_regular(3))  # False
        ```

        Check complete graph regularity:

        ```python
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        # Complete graph K_n is (n-1)-regular
        print(graph.is_k_regular(n - 1))  # True
        print(graph.is_k_regular(n))      # False
        ```

        Check isolated nodes:

        ```python
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        # All isolated nodes have degree 0
        print(graph.is_k_regular(0))  # True
        print(graph.is_k_regular(1))  # False
        ```

        Notes
        -----
        - empty graphs are considered 0-regular
        - cycles are 2-regular
        - complete graphs `K_n` are *(n-1)*-regular
        - path graphs are *not* regular (end nodes have degree `1`, middle nodes have degree `2`)
        - for directed graphs, checks both in-degree and out-degree equal `k`

        %family Graph Properties
        """
        from edgecase._core import is_k_regular

        return is_k_regular(self._graph, k)

    # ==============================================================================
    # Simple Counts
    # ==============================================================================

    def count_all_loops(self) -> int:
        """
        Count total number of self-loop edges in the graph.

        A self-loop is an edge where the source and target are the same node. This counts all
        self-loop edges in the entire graph, regardless of which nodes they belong to. Use
        `count_loops()` to get per-node counts.

        Self-loops are common in state machines, reflexive relationships, and chemical reaction
        networks.

        Returns
        -------
        int
            Total number of edges where source equals target.

        Examples
        --------
        Count all self-loops in graph:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
            .add_edge("A", "A")  # Self-loop
            .add_edge("B", "B")  # Self-loop
        )

        print(graph.count_all_loops())  # 2
        ```

        Multiple self-loops on same node:

        ```python
        graph = Graph()
        graph.add_node(label="A")
        graph.add_edge("A", "A")  # First self-loop
        graph.add_edge("A", "A")  # Second self-loop

        print(graph.count_all_loops())  # 2
        ```

        %family Simple Counts
        """
        return self._graph.count_all_loops()

    def count_multi_edges(self) -> int:
        """
        Count multi-edges (parallel edges between same node pair).

        Only counts edges beyond the first one for each node pair. For example, if there are 3 edges
        from `A` to `B`, this counts 2 of them. Multi-edges are useful for modeling scenarios like
        multiple flights between cities or repeated transactions.

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

        %family Simple Counts
        """
        return self._graph.count_multi_edges()

    def count_mutual_node_pairs(self) -> int:
        """
        Count mutual node pairs (bidirectional edges) in directed graphs.

        Returns the number of node pairs `(a,b)` where both edges `a->b` and `b->a` exist. Only
        works for directed graphs. Self-loops are excluded. This metric is useful for analyzing
        reciprocity in directed networks like social graphs.

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

        %family Simple Counts
        """
        return self._graph.count_mutual_node_pairs()

    def count_asymmetric_node_pairs(self) -> int:
        """
        Count asymmetric node pairs (one-way edges) in directed graphs.

        Returns the number of node pairs `(a,b)` where edge `a->b` exists but not `b->a`. Only works
        for directed graphs. Self-loops are excluded. This metric helps analyze directional flow or
        hierarchy in directed networks.

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

        %family Simple Counts
        """
        return self._graph.count_asymmetric_node_pairs()

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # RELATIONSHIPS & ACCESS
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Neighbors and Connectivity
    # ==============================================================================

    def neighbors(self, node: Union[int, str], mode: str = "out") -> list[int]:
        """
        Get neighbors of a node.

        Returns the neighboring nodes of a specified node, with control over directionality (for
        directed graphs).

        For directed graphs, the default mode is `"out"` (outgoing neighbors only). For undirected
        graphs, mode doesn't affect the result since all edges are bidirectional.

        Parameters
        ----------
        node
            The node ID or label to find neighbors for.
        mode
            Direction mode for neighbor lookup: (1) `"out"`, only successors (outgoing edges) which
            is the default for directed graphs, (2) `"in"`, only predecessors (incoming edges), and
            (3) `"all"`, both successors and predecessors.

        Returns
        -------
        list[int]
            List of neighbor node IDs.

        Raises
        ------
        ValueError
            If `mode=` is not one of `"out"`, `"in"`, or `"all"`.

        Examples
        --------
        ```python
        import edgecase as ec

        # Directed graph example
        graph = ec.Graph(directed=True)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B", weight=1.0)
        graph.add_edge("C", "A", weight=2.0)

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")

        # Get successors (outgoing neighbors); returns node IDs
        out_neighbors = graph.neighbors(a_id, mode="out")
        print(out_neighbors)  # [<b_id>]: node IDs, not labels

        # Get predecessors (incoming neighbors)
        in_neighbors = graph.neighbors(a_id, mode="in")
        print(in_neighbors)  # [<c_id>]

        # Get all neighbors
        all_neighbors = graph.neighbors(a_id, mode="all")
        print(all_neighbors)  # [<b_id>, <c_id>]

        # Convert IDs to labels if needed
        neighbor_labels = [graph.get_label_from_id(nid) for nid in out_neighbors]
        print(neighbor_labels)  # ['B']
        ```

        %family Neighbors and Connectivity
        """
        # Resolve node to integer ID
        node_id = self._resolve_node_id(node)

        if mode not in ["out", "in", "all"]:
            raise ValueError("mode must be 'out', 'in', or 'all'")

        neighbors = []

        if mode in ["out", "all"]:
            # Outgoing neighbors from adjacency list
            neighbors.extend(self._graph.neighbors(node_id))

        if mode in ["in", "all"]:
            # Incoming neighbors; find edges where this node is target
            for edge_tuple in self._graph.edges():
                # edge_tuple is (edge_id, source, target)
                if edge_tuple[2] == node_id and edge_tuple[1] not in neighbors:
                    neighbors.append(edge_tuple[1])

        # For undirected graphs, mode doesn't matter
        if not self._graph.is_directed and mode != "all":
            # Already got all neighbors via adjacency list
            pass

        return neighbors

    def get_predecessors(self, node: Union[int, str]) -> list[int]:
        """
        Get direct predecessors of a node (incoming neighbors).

        Returns nodes that have outgoing edges to the specified node. Only available for directed
        graphs.

        Parameters
        ----------
        node
            Node ID or label to find predecessors for.

        Returns
        -------
        list[int]
            List of predecessor node IDs (nodes with edges pointing to `node=`).

        Raises
        ------
        ValueError
            If the graph is undirected.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")

        c_id = graph.get_id_from_label("C")

        # Find nodes with edges pointing to C; returns node IDs
        predecessors = graph.get_predecessors(c_id)
        # Convert to labels: ['A', 'B']
        labels = [graph.get_label_from_id(nid) for nid in predecessors]
        print(labels)  # ['A', 'B']
        ```

        %family Neighbors and Connectivity
        """
        if not self._graph.is_directed:
            raise ValueError("get_predecessors only available for directed graphs")
        return self.neighbors(node, mode="in")

    def get_successors(self, node: Union[int, str]) -> list[int]:
        """
        Get direct successors of a node (outgoing neighbors).

        Returns nodes that are targets of outgoing edges from the specified node. Only available for
        directed graphs.

        Parameters
        ----------
        node
            Node ID or label to find successors for.

        Returns
        -------
        list[int]
            List of successor node IDs (nodes that `node=` points to).

        Raises
        ------
        ValueError
            If the graph is undirected.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")

        a_id = graph.get_id_from_label("A")

        # Find nodes that A points to; returns node IDs
        successors = graph.get_successors(a_id)

        # Convert to labels: ['B', 'C']
        labels = [graph.get_label_from_id(nid) for nid in successors]

        print(labels)  # ['B', 'C']
        ```

        %family Neighbors and Connectivity
        """
        if not self._graph.is_directed:
            raise ValueError("get_successors only available for directed graphs")
        return self.neighbors(node, mode="out")

    def get_non_neighbors(self, node: Union[int, str]) -> list[int]:
        """
        Get nodes that are NOT neighbors of the specified node.

        Returns all nodes in the graph except the specified node itself and its neighbors. Useful
        for finding potential new connections or disconnected nodes.

        Parameters
        ----------
        node
            Node ID or label to find non-neighbors for.

        Returns
        -------
        list[int]
            Sorted list of node IDs that are not neighbors of `node=` and are not `node=` itself.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_path_graph(5)  # 1 -> 2 -> 3 -> 4 -> 5

        # Node 2 is connected to 1 and 3
        # Non-neighbors are 4 and 5 (returns node IDs)
        non_neighbors = graph.get_non_neighbors(2)
        print(non_neighbors)  # [4, 5]
        ```

        Find potential new connections:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )
        # Recommend connections for a node (using node ID)
        node_id = graph.get_id_from_label("A")
        candidates = graph.get_non_neighbors(node_id)

        # Score candidates by common neighbors
        scores = [
            (candidate, len(graph.get_common_neighbors(node_id, candidate)))
            for candidate in candidates
        ]
        ```

        %family Neighbors and Connectivity
        """
        # Resolve node to integer ID
        node_id = self._resolve_node_id(node)

        all_nodes = set(self.get_node_ids(limit=None))
        neighbors = set(self.neighbors(node_id, mode="all"))

        # Remove the node itself and its neighbors
        non_neighbors = all_nodes - neighbors - {node_id}

        return sorted(list(non_neighbors))

    def get_common_neighbors(self, node_1: Union[int, str], node_2: Union[int, str]) -> list[int]:
        """
        Get nodes that are neighbors of both specified nodes.

        Returns the intersection of neighbor sets for two nodes, useful for finding shared
        connections or potential triangles in the graph.

        Parameters
        ----------
        node_1
            First node ID (int) or label (str).
        node_2
            Second node ID (int) or label (str).

        Returns
        -------
        list[int]
            List of node IDs that are neighbors to both `node_1=` and `node_2=`.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=False)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        graph.add_node("D")
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")
        graph.add_edge("A", "D")
        graph.add_edge("B", "D")

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # Find nodes connected to both A and B; returns node IDs
        common = graph.get_common_neighbors(a_id, b_id)

        # Convert to labels: ['C', 'D']
        common_labels = [graph.get_label_from_id(nid) for nid in common]

        print(common_labels)  # ['C', 'D']
        ```

        Triangle detection using common neighbors:

        ```python
        # Find all triangles involving node A
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
            .add_edge(source="B", target="C")
            .add_edge(source="A", target="D")
        )

        a_id = graph.get_id_from_label(label="A")
        for neighbor_id in graph.neighbors(node=a_id):
            common = graph.get_common_neighbors(a_id, neighbor_id)
            for third_id in common:
                neighbor_label = graph.get_label_from_id(node_id=neighbor_id)
                third_label = graph.get_label_from_id(node_id=third_id)
                print(f"Triangle: A - {neighbor_label} - {third_label}")
        ```

        %family Neighbors and Connectivity
        """
        neighbors1 = set(self.neighbors(node_1, mode="all"))
        neighbors2 = set(self.neighbors(node_2, mode="all"))
        return list(neighbors1 & neighbors2)

    def get_similar_neighbors(
        self,
        node: Union[int, str],
        node_attr: str,
        tol_abs: tuple[float, float] | None = None,
        tol_pct: tuple[float, float] | None = None,
    ) -> list[int]:
        """
        Get neighbors with similar attribute values.

        Returns neighbors whose numeric attribute values fall within a specified tolerance range of
        the starting node's value. Useful for finding connected nodes with comparable properties.

        Parameters
        ----------
        node
            Starting node ID or label to compare against.
        node_attr
            Name of the numeric attribute to compare.
        tol_abs
            Absolute tolerance as `(lower_bound, upper_bound)` from the starting value. For example,
            `(1.0, 2.0)` means values from `start_value - 1.0` to `start_value + 2.0` are considered
            similar.
        tol_pct
            Percentage tolerance as `(lower_pct, upper_pct)` from the starting value. For example,
            `(10, 20)` means values from `start_value × 0.9` to `start_value × 1.2` are considered
            similar.

        Returns
        -------
        list[int]
            List of neighbor node IDs with similar attribute values. Empty list if the starting node
            lacks the attribute or has no neighbors in range.

        Raises
        ------
        ValueError
            If the attribute exists but is not numeric (int or float).

        Examples
        --------
        ```python
        graph = Graph()
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
        graph = (
            Graph()
            .add_node(label="A", value=5.0)
            .add_node(label="B", value=4.5)
            .add_node(label="C", value=10.0)
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
        )
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
        - if neither `tol_abs` nor `tol_pct` is specified, only exact matches are returned.
        - neighbors without the specified attribute are excluded from results.
        - non-numeric attribute values raise a `ValueError`.

        %family Neighbors and Connectivity
        """
        # Get starting node's attribute value
        start_node = self.get_node(node)
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
        neighbors = self.neighbors(node, mode="all")

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

    # ==============================================================================
    # Graph Selection System
    # ==============================================================================

    def select_nodes(self, predicate) -> "Graph":
        """
        Select nodes using a predicate function.

        Applies a custom filter function to all nodes, selecting those for which the predicate
        returns `True`. The predicate receives a `Node` object with properties: `id`, `label`, and
        attributes.

        Parameters
        ----------
        predicate
            Function that takes a `Node` object and returns `bool`. Should return `True` to select
            the nodes.

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

        selected = graph.get_selected_node_ids()
        print(selected)  # Node IDs for 'A' and 'C'

        # Select nodes by ID pattern (if using integer IDs)
        # graph.select_nodes(lambda n: n.id > 100)
        ```

        %family Graph Selection
        """
        self._graph.select_nodes(predicate)
        return self

    def select_edges(self, predicate) -> "Graph":
        """
        Select edges using a predicate function.

        Applies a custom filter function to all edges, selecting those for which the predicate
        returns `True`. The predicate receives an Edge object with properties: `id`, `source`,
        `target`, `weight`, and other attributes.

        Parameters
        ----------
        predicate
            Function that takes an `Edge` object and returns `bool`. Should return `True` to select
            the edge.

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
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_edge("A", "B", weight=0.3)
            .add_edge("B", "C", weight=0.8)
            .add_edge("C", "D", weight=0.6)
            .select_edges(lambda e: e.weight and e.weight > 0.5)
        )

        count = graph.count_selected_edges()
        print(count)  # 2

        # Select edges by relationship type
        # graph.select_edges(
        #     lambda e: e.attributes and e.attributes.get("rel") == "friend"
        # )
        ```

        %family Graph Selection
        """
        self._graph.select_edges(predicate)
        return self

    def select_nodes_by_id(self, node_ids: list[int]) -> "Graph":
        """
        Select nodes by their IDs.

        Marks specified nodes as selected for subsequent operations like building subgraphs or
        filtering. Selection is cumulative unless cleared first.

        Parameters
        ----------
        node_ids
            List of node IDs (integers) to select.

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
        )

        # Get IDs for nodes A and B
        id_a = graph.get_id_from_label("A")
        id_b = graph.get_id_from_label("B")
        graph.select_nodes_by_id([id_a, id_b])

        selected = graph.get_selected_node_ids()
        print(selected)  # ['A', 'B']
        ```

        %family Graph Selection
        """
        self._graph.select_nodes_by_id(node_ids)
        return self

    def select_edges_by_id(self, edge_ids: list[int]) -> "Graph":
        """
        Select edges by their IDs.

        Marks specified edges as selected for subsequent operations. Edge IDs are internal integer
        identifiers assigned when edges are created. Selection is cumulative unless cleared first.

        Parameters
        ----------
        edge_ids
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
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        edge_ids = graph.get_edge_ids()
        graph.select_edges_by_id(edge_ids=[edge_ids[0], edge_ids[1]])

        selected_count = graph.count_selected_edges()
        print(selected_count)  # 2
        ```

        %family Graph Selection
        """
        self._graph.select_edges_by_id(edge_ids)
        return self

    def select_edges_by_node(self, nodes: list[int | str]) -> "Graph":
        """
        Select all edges connected to specified nodes.

        Selects all edges where at least one endpoint is in the provided node list. For directed
        graphs, includes both incoming and outgoing edges. Useful for selecting the local
        connectivity around specific nodes.

        Parameters
        ----------
        nodes
            List of node IDs (int) or labels (str) whose incident edges should be selected.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Select edges incident to a specific node using node ID:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
        )

        b_id = graph.get_id_from_label("B")
        graph.select_edges_by_node([b_id])

        # Selects A-B and B-C edges
        selected = graph.count_selected_edges()
        print(selected)  # 2
        ```

        Can also use labels directly:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
        )

        graph.select_edges_by_node(["B"])
        ```

        %family Graph Selection
        """
        # Resolve labels to IDs
        resolved_ids = [self._resolve_node_id(nid) for nid in nodes]
        self._graph.select_edges_by_node_id(resolved_ids)
        return self

    def select_last_nodes_created(self, n: int) -> "Graph":
        """
        Select the last N nodes that were created.

        Selects nodes in reverse chronological order based on creation time. Useful for incremental
        graph building workflows where you want to work with recently added nodes.

        Parameters
        ----------
        n
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
        selected = graph.get_selected_node_ids()
        print(selected)  # ['C', 'D']
        ```

        %family Graph Selection
        """
        self._graph.select_last_nodes_created(n)
        return self

    def select_last_edges_created(self, n: int) -> "Graph":
        """
        Select the last N edges that were created.

        Selects edges in reverse chronological order based on creation time. Useful for incremental
        graph building workflows where you want to work with recently added edges.

        Parameters
        ----------
        n
            Number of most recently created edges to select.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Select recently added edges:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="E")
            .select_last_edges_created(n=2)
        )

        # Last 2 edges selected
        count = graph.count_selected_edges()
        print(count)  # 2
        ```

        %family Graph Selection
        """
        self._graph.select_last_edges_created(n)
        return self

    def select_nodes_in_neighborhood(
        self, nodes: list[Union[int, str]] | int | str, k: int = 1, mode: str = "out"
    ) -> "Graph":
        """
        Select all nodes within K hops of specified seed nodes.

        Uses breadth-first search to find neighbors at distance up to k from the seed nodes. Does
        not include the seed nodes themselves in the selection. Useful for extracting local
        neighborhoods.

        Parameters
        ----------
        nodes
            Seed node(s) to start from. Can be a single node ID (int), a single label (str), or a
            list of node IDs/labels.
        k
            Maximum number of hops/distance to traverse.
        mode
            Direction to traverse: `"out"`, `"in"`, or `"all"`. For undirected graphs, always uses
            the `"all"` mode.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Select 1-hop neighborhood using node ID:

        ```python
        from edgecase import Graph

        graph = Graph().add_star_graph(n=5)
        # Get the center node ID (first node created)
        all_node_ids = list(graph.get_node_ids(limit=None))
        center_id = all_node_ids[0]
        graph.select_nodes_in_neighborhood(nodes=center_id, k=1, mode="all")

        # Selects all neighbors of center (not center itself)
        count = graph.count_selected_nodes()
        print(count)  # Expected: neighbors of center
        ```

        Select 2-hop neighborhood using label:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        graph.select_nodes_in_neighborhood(nodes="A", k=2, mode="out")
        # Selects B and C (reachable within 2 hops from A)
        ```

        Select neighborhood from multiple seed nodes:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "E")
        )

        # Select using multiple seed nodes
        graph.select_nodes_in_neighborhood(nodes=["A", "B"], k=1, mode="all")
        # Selects neighbors of both A and B (not A and B themselves)
        ```

        %family Graph Selection
        """
        # Convert single value to list
        if isinstance(nodes, (str, int)):
            nodes = [nodes]

        # Resolve labels to IDs
        resolved_ids = [self._resolve_node_id(nid) for nid in nodes]

        self._graph.select_nodes_in_neighborhood(resolved_ids, k, mode)
        return self

    def select_nodes_by_degree(
        self,
        min_degree: int | None = None,
        max_degree: int | None = None,
        degree_type: str = "total",
    ) -> "Graph":
        """
        Select nodes by degree criteria.

        Filters nodes based on their degree (number of connections). Useful for finding hubs (high
        degree) or peripheral nodes (low degree). Can filter by in-degree, out-degree, or total
        degree.

        Parameters
        ----------
        min_degree
            Minimum degree threshold (inclusive). `None` means no lower bound.
        max_degree
            Maximum degree threshold (inclusive). `None` means no upper bound.
        degree_type
            Type of degree to use: `"in"`, `"out"`, or `"total"`. For undirected graphs, only
            `"total"` is valid.

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
            .add_star_graph(10)
            .select_nodes_by_degree(min_degree=5)
        )

        # Hub node has degree 10, all others have degree 1
        hubs = graph.get_selected_node_ids()
        print(len(hubs))  # 1 (the center node)
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

        %family Graph Selection
        """
        self._graph.select_nodes_by_degree(
            min_degree=min_degree, max_degree=max_degree, degree_type=degree_type
        )
        return self

    def select_nodes_by_centrality(
        self, measure: str = "degree", k: int = 10, include_ties: bool = False, **kwargs
    ) -> "Graph":
        """
        Select top-K nodes by centrality measure.

        Computes centrality scores for all nodes and selects the k highest-scoring nodes. Useful for
        identifying the most important or influential nodes by various centrality metrics.

        Parameters
        ----------
        measure
            Centrality measure to use: `"degree"`, `"in_degree"`, `"out_degree"`, `"pagerank"`, or
            "`betweenness"`.
        k
            Number of top nodes to select.
        include_ties
            If `True`, include all nodes tied with the k-th highest score.
        **kwargs
            Additional arguments for centrality algorithms (e.g., damping for Pagerank).

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
            .add_star_graph(10)
            .add_path_graph(10)
            .select_nodes_by_centrality("degree", k=3)
        )

        # Selects top 3 nodes by degree
        hubs = graph.get_selected_node_ids()
        print(len(hubs))  # 3
        ```

        Select by PageRank:

        ```python
        graph = (
            Graph(directed=True)
            .add_path_graph(20)
            .select_nodes_by_centrality("pagerank", k=5, damping=0.85)
        )
        ```

        %family Graph Selection
        """
        # Compute scores based on measure
        if measure == "degree":
            scores = {
                node_id: self.get_degree(node_id) for node_id in self.get_node_ids(limit=None)
            }
        elif measure == "in_degree":
            if not self._graph.is_directed:
                raise ValueError("in_degree only available for directed graphs")
            scores = {
                node_id: self.get_in_degree(node_id) for node_id in self.get_node_ids(limit=None)
            }
        elif measure == "out_degree":
            if not self._graph.is_directed:
                raise ValueError("out_degree only available for directed graphs")
            scores = {
                node_id: self.get_out_degree(node_id) for node_id in self.get_node_ids(limit=None)
            }
        elif measure == "pagerank":
            scores = self.pagerank(return_dict=True, **kwargs)
        elif measure == "betweenness":
            scores = self.betweenness_centrality(return_dict=True)
        else:
            raise ValueError(
                f"Unknown centrality measure '{measure}'. "
                "Choose from: degree, in_degree, out_degree, pagerank, betweenness"
            )

        # Use the Rust helper to select top-k
        self._graph.select_nodes_by_scores(scores, k, include_ties)
        return self

    def deselect_nodes(self, node_ids: list[int | str]) -> "Graph":
        """
        Remove nodes from selection.

        Unmarks specified nodes, removing them from the current selection set. Has no effect on
        nodes that aren't currently selected.

        Parameters
        ----------
        node_ids
            List of node IDs (int) or labels (str) to deselect.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        Refine a selection by removing specific nodes using node IDs:

        ```python
        graph = (
            Graph()
            .add_path_graph(5)
        )

        # Select all nodes first
        all_ids = list(graph.get_node_ids(limit=None))
        graph.select_nodes_by_id(all_ids)

        # Deselect first and last nodes
        node_list = list(graph.get_node_ids(limit=None))
        first_id = node_list[0]
        last_id = node_list[4]
        graph.deselect_nodes([first_id, last_id])

        # Only middle nodes remain selected
        selected = graph.get_selected_node_ids()
        print(len(selected))  # 3
        ```

        Can also use labels directly:

        ```python
        graph = (
            Graph()
            .add_node(label="0")
            .add_node(label="1")
            .add_node(label="2")
            .add_node(label="3")
            .add_node(label="4")
            .add_edge(source="0", target="1")
            .add_edge(source="1", target="2")
            .add_edge(source="2", target="3")
            .add_edge(source="3", target="4")
        )

        # Select all nodes first
        all_ids = list(graph.get_node_ids(limit=None))
        graph.select_nodes_by_id(all_ids)

        # Deselect first and last nodes using labels
        graph.deselect_nodes(["0", "4"])
        ```

        %family Graph Selection
        """
        # Resolve labels to IDs
        resolved_ids = [self._resolve_node_id(nid) for nid in node_ids]
        self._graph.deselect_nodes(resolved_ids)
        return self

    def deselect_edges(self, edge_ids: list[int]) -> "Graph":
        """
        Remove edges from selection.

        Unmarks specified edges, removing them from the current selection set. Has no effect on
        edges that aren't currently selected.

        Parameters
        ----------
        edge_ids
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

        all_edges = graph.get_edge_ids()
        graph.select_edges_by_id(all_edges)

        # Remove first edge from selection
        graph.deselect_edges([all_edges[0]])

        count = graph.count_selected_edges()
        print(count)  # 4
        ```

        %family Graph Selection
        """
        self._graph.deselect_edges(edge_ids)
        return self

    def clear_selection(self) -> "Graph":
        """
        Clear all node and edge selections.

        Resets the selection state, unmarking all previously selected nodes and edges. Useful for
        starting fresh with a new selection workflow.

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

        %family Graph Selection
        """
        self._graph.clear_selection()
        return self

    def invert_node_selection(self) -> "Graph":
        """
        Invert node selection (select unselected, unselect selected).

        Toggles the selection state of all nodes: currently selected nodes become unselected, and
        unselected nodes become selected. Useful for getting the complement of a selection.

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

        %family Graph Selection
        """
        self._graph.invert_node_selection()
        return self

    def invert_edge_selection(self) -> "Graph":
        """
        Invert edge selection (select unselected, unselect selected).

        Toggles the selection state of all edges: currently selected edges become unselected, and
        unselected edges become selected. Useful for getting the complement of an edge selection.

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
        first_edge = graph.get_edge_ids()[0]
        graph.select_edges_by_id([first_edge]).invert_edge_selection()

        count = graph.count_selected_edges()
        print(count)  # 4 (all except first)
        ```

        %family Graph Selection
        """
        self._graph.invert_edge_selection()
        return self

    def get_selected_node_ids(self) -> list[int]:
        """
        Get currently selected node IDs.

        Returns the list of node IDs that are currently marked as selected. Order is not guaranteed.

        Returns
        -------
        list[int]
            List of selected node IDs, empty if no nodes selected.

        Examples
        --------
        Query current selection:

        ```python
        graph = (
            Graph()
            .add_star_graph(5)
            .select_nodes_by_degree(min_degree=3)
        )

        selected = graph.get_selected_node_ids()
        print(len(selected))  # 1 (the center node)
        ```

        %family Graph Selection
        """
        return self._graph.get_selected_node_ids()

    def get_selected_edge_ids(self) -> list[int]:
        """
        Get currently selected edge IDs.

        Returns the list of edge IDs that are currently marked as selected. Order is not guaranteed.

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
            .select_edges(lambda e: e.weight is not None and e.weight > 0.5)
        )

        selected = graph.get_selected_edge_ids()
        print(len(selected))
        ```

        %family Graph Selection
        """
        return self._graph.get_selected_edge_ids()

    def count_selected_nodes(self) -> int:
        """
        Get count of currently selected nodes.

        Returns the number of nodes currently marked as selected. More efficient than
        `len(get_selected_nodes())` when you only need the count.

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

        %family Graph Selection
        """
        return self._graph.count_selected_nodes()

    def count_selected_edges(self) -> int:
        """
        Get count of currently selected edges.

        Returns the number of edges currently marked as selected. More efficient than
        `len(get_selected_edges())` when you only need the count.

        Returns
        -------
        int
            Number of selected edges.

        Examples
        --------
        Count selected edges:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_cycle_graph(n=10)
            .select_edges(lambda e: e.weight is not None and e.weight > 0.7)
        )

        count = graph.count_selected_edges()
        print(f"{count} high-weight edges")
        ```

        %family Graph Selection
        """
        return self._graph.count_selected_edges()

    # ========================================
    # Graph Traversal System
    # ========================================

    def trav_out(self) -> "Graph":
        """
        Traverse from selected nodes to outward neighbors.

        From the currently selected nodes, moves to all nodes that can be reached by following
        outward edges (where selected nodes are sources). The current node selection is replaced
        with the traversed nodes.

        For directed graphs, follows edges in the forward direction. For undirected graphs, follows
        edges in all directions (equivalent to `trav_both()`).

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node("A").add_node("B").add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        node_a_id = graph.get_id_from_label("A")
        graph.select_nodes_by_id([node_a_id]).trav_out()
        selected = graph.get_selected_node_ids()
        # Returns [node_id_for_B]
        ```

        %family Graph Traversal
        """
        self._graph.trav_out()
        return self

    def trav_in(self) -> "Graph":
        """
        Traverse from selected nodes to inward neighbors.

        From the currently selected nodes, moves to all nodes that have outward edges leading to the
        selected nodes (where selected nodes are targets). The current node selection is replaced
        with the traversed nodes.

        For directed graphs, follows edges in the reverse direction. For undirected graphs, follows
        edges in all directions (equivalent to `trav_both()`).

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node("A").add_node("B").add_node("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        node_c_id = graph.get_id_from_label("C")
        graph.select_nodes_by_id([node_c_id]).trav_in()
        selected = graph.get_selected_node_ids()
        # Returns [node_id_for_B]
        ```

        %family Graph Traversal
        """
        self._graph.trav_in()
        return self

    def trav_both(self) -> "Graph":
        """
        Traverse from selected nodes to all neighbors.

        From the currently selected nodes, moves to all connected neighbors in both directions (both
        outward and inward edges). The current node selection is replaced with the traversed nodes.

        Works the same for directed and undirected graphs.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge(source="A", target="B")
        graph.add_edge(source="C", target="B")

        node_b_id = graph.get_id_from_label("B")
        graph.select_nodes_by_id([node_b_id]).trav_both()

        # Get labels of selected nodes
        labels = [graph.get_label_from_id(nid) for nid in graph.get_selected_node_ids()]
        # Returns ['A', 'C'] (or ['C', 'A'] where the order may vary)
        ```

        %family Graph Traversal
        """
        self._graph.trav_both()
        return self

    def trav_out_edge(self) -> "Graph":
        """
        Traverse from selected nodes to outbound edges.

        From the currently selected nodes, moves to all edges where the selected nodes are sources
        (outbound edges). The current node selection is cleared and replaced with an edge selection.

        For directed graphs, selects edges where from=selected_node. For undirected graphs, selects
        all edges incident to selected nodes.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge(source="A", target="B")
        graph.add_edge(source="A", target="C")

        node_a_id = graph.get_id_from_label("A")
        graph.select_nodes_by_id([node_a_id]).trav_out_edge()

        print(len(graph.get_selected_edge_ids()))  # 2
        ```

        %family Graph Traversal
        """
        self._graph.trav_out_edge()
        return self

    def trav_in_edge(self) -> "Graph":
        """
        Traverse from selected nodes to inbound edges.

        From the currently selected nodes, moves to all edges where the selected nodes are targets
        (inbound edges). The current node selection is cleared and replaced with an edge selection.

        For directed graphs, selects edges where to=selected_node. For undirected graphs, selects
        all edges incident to selected nodes.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge(source="A", target="C")
        graph.add_edge(source="B", target="C")

        node_c_id = graph.get_id_from_label("C")
        graph.select_nodes_by_id([node_c_id]).trav_in_edge()

        print(len(graph.get_selected_edge_ids()))  # 2
        ```

        %family Graph Traversal
        """
        self._graph.trav_in_edge()
        return self

    def trav_reverse_edge(self, add_to_selection: bool = False) -> "Graph":
        """
        Traverse to reverse edges.

        From the currently selected edges, finds the reverse edges. For each selected edge `A->B`,
        looks for the corresponding edge `B->A`. Only works with directed graphs.

        Parameters
        ----------
        add_to_selection
            If `True`, adds reverse edges to the current selection. If `False`, replaces the
            selection with reverse edges only.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")  # Reverse edge
        )

        # Select A->B edge
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        graph.select_edges_by_node([a_id, b_id])

        # Traverse to reverse edge (B->A)
        graph.trav_reverse_edge()
        edges = graph.get_selected_edge_ids()
        # Now selects only the B->A edge
        ```

        Add reverse edges to selection:

        ```python
        graph = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")
        )
        edge_ids = list(graph.get_edge_ids())
        # Select both an edge and its reverse
        (graph
            .select_edges_by_id([edge_ids[0]])
            .trav_reverse_edge(add_to_selection=True)
        )
        # Selection now includes both A->B and B->A
        ```

        %family Graph Traversal
        """
        self._graph.trav_reverse_edge(add_to_selection)
        return self

    def trav_out_until(
        self,
        conditions: callable,
        max_steps: int = 30,
        exclude_unmatched: bool = True,
        add_to_selection: bool = False,
    ) -> "Graph":
        """
        Traverse outward node-by-node until stopping conditions are met.

        From the currently selected nodes, moves along outward edges to connected nodes, continuing
        until reaching nodes that satisfy the condition or until `max_steps=` is reached. The
        traversal proceeds in a breadth-first manner.

        Parameters
        ----------
        conditions
            A function that takes a node ID (str) and returns `True` if the stopping condition is
            met for that node. The callable can access node attributes via
            `graph.get_node(node_id).attributes`.
        max_steps
            Maximum number of traversal steps (node-to-node hops) before stopping.
        exclude_unmatched
            If `True`, only nodes satisfying the conditions are kept in the final selection. If
            `False`, all reached nodes are kept.
        add_to_selection
            If `True`, all traversed nodes are included in the final selection. If `False`, only the
            final nodes are selected.

        Returns
        -------
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
        node_ids = list(graph.get_node_ids())
        (graph
            .select_nodes_by_id([node_ids[0]])
            .trav_out_until(lambda nid: graph.get_node(nid).attributes.get("value") == 8)
        )

        print(graph.get_selected_node_ids())  # ['8']
        ```

        Multiple starting points:

        ```python
        graph = ec.Graph(directed=True).add_path_graph(10)
        # Set value attributes
        for i, nid in enumerate(graph.get_node_ids(), start=1):
            graph.set_node_attrs("value", {nid: i})

        node_ids = list(graph.get_node_ids())
        # Traverse from multiple nodes
        (graph
            .clear_selection()
            .select_nodes_by_id([node_ids[0], node_ids[1]])
            .trav_out_until(
                lambda nid: graph.get_node(nid).attributes.get("value", 0) >= 5,
                max_steps=10
            )
        )
        # Returns first nodes ≥ 5 from each starting point
        ```

        %family Graph Traversal
        """
        selected = self.get_selected_node_ids()
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
                        neighbors = self.neighbors(node_id, mode="out")
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
        add_to_selection: bool = False,
    ) -> "Graph":
        """
        Traverse inward node-by-node until stopping conditions are met.

        From the currently selected nodes, moves along inward edges to connected nodes, continuing
        until reaching nodes that satisfy the condition or until `max_steps=` is reached. The
        traversal proceeds in a breadth-first manner.

        Parameters
        ----------
        conditions
            A function that takes a node ID (str) and returns `True` if the stopping condition is
            met for that node. The callable can access node attributes via
            `graph.get_node(node_id).attributes`.
        max_steps
            Maximum number of traversal steps (node-to-node hops) before stopping.
        exclude_unmatched
            If `True`, only nodes satisfying the conditions are kept in the final selection. If
            `False`, all reached nodes are kept.
        add_to_selection
            If `True`, all traversed nodes are included in the final selection. If `False`, only the
            final nodes are selected.

        Returns
        -------
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
        node_ids = list(graph.get_node_ids())
        (graph
            .select_nodes_by_id([node_ids[-1]])
            .trav_in_until(lambda nid: graph.get_node(nid).attributes.get("value") == 1)
        )

        print(graph.get_selected_node_ids())  # ['1']
        ```

        Multiple stopping conditions:

        ```python
        graph = ec.Graph(directed=True).add_path_graph(12)
        # Set value attributes
        for i, nid in enumerate(graph.get_node_ids(), start=1):
            graph.set_node_attrs("value", {nid: i})

        node_ids = list(graph.get_node_ids())
        # Find nodes with value in specific set
        (graph
            .clear_selection()
            .select_nodes_by_id([node_ids[5], node_ids[11]])
            .trav_in_until(
                lambda nid: graph.get_node(nid).attributes.get("value", 0) in [1, 2, 10]
            )
        )
        # Returns nodes with value in [1, 2, 10]
        ```

        %family Graph Traversal
        """
        selected = self.get_selected_node_ids()
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
                        neighbors = self.neighbors(node_id, mode="in")
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

    # ==============================================================================
    # NDF/EDF Setters and Getters
    # ==============================================================================

    def set_node_attrs(
        self,
        node_attr: str,
        values: Any | list[Any] | dict[int, Any],
        nodes: list[int | str] | None = None,
    ) -> "Graph":
        """
        Set attribute values for one or more nodes.

        Updates a specific attribute on nodes. Can set the same value for all nodes, or provide
        different values via a list or dictionary. Creates the attribute if it doesn't exist,
        overwrites if it does.

        Parameters
        ----------
        node_attr
            The name of the attribute to set.
        values
            The value(s) to set. Can be a (1) single value, applied to all specified nodes, (2) a
            list, must match length of nodes (paired by position), and (3) a dict, maps node IDs to
            values (only sets specified nodes).
        nodes
            Optional list of node IDs (int) or labels (str) to update. If `None`, updates all nodes.

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
        Set same value for specific nodes using node IDs:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )

        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        graph.set_node_attrs("color", "blue", nodes=[a_id, b_id])

        colors = graph.get_node_attrs("color")
        # Returns dict mapping node IDs to 'blue': {1: 'blue', 2: 'blue'}
        ```

        Can also use labels:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        graph.set_node_attrs("color", "red", nodes=["A", "B"])
        ```

        Set same value for all nodes:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        graph.set_node_attrs("status", "active")
        ```

        Set different values using a list:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        node_ids = [graph.get_id_from_label(l) for l in ["A", "B", "C"]]
        graph.set_node_attrs("score", [100, 85, 92], nodes=node_ids)
        ```

        Set different values using a dictionary:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
        )
        id_map = {graph.get_id_from_label("A"): "primary",
                  graph.get_id_from_label("B"): "secondary",
                  graph.get_id_from_label("C"): "primary"}
        graph.set_node_attrs("category", id_map)
        ```

        %family Node/Edge Data Frames
        """
        if nodes is None:
            nodes = self.get_node_ids(limit=None)
        else:
            # Resolve labels to IDs
            nodes = [self._resolve_node_id(nid) for nid in nodes]

        # Handle different value types
        if isinstance(values, dict):
            # Dict mapping node IDs to values: resolve keys if they're labels
            resolved_values = {}
            for key, value in values.items():
                resolved_key = self._resolve_node_id(key) if isinstance(key, (int, str)) else key
                resolved_values[resolved_key] = value

            for node_id in nodes:
                if node_id in resolved_values:
                    self._graph.set_node_attr(node_id, node_attr, resolved_values[node_id])
        elif isinstance(values, list):
            # List of values (must match length of nodes)
            if len(values) != len(nodes):
                raise ValueError(
                    f"Length of values ({len(values)}) must match length of nodes ({len(nodes)})"
                )
            for node_id, value in zip(nodes, values):
                self._graph.set_node_attr(node_id, node_attr, value)
        else:
            # Single value applied to all nodes
            for node_id in nodes:
                self._graph.set_node_attr(node_id, node_attr, values)

        return self

    def set_node_attrs_ws(self, **attributes) -> "Graph":
        """
        Set attributes on all selected nodes.

        Updates or adds attributes to nodes in the current selection. Existing attributes are
        preserved unless explicitly overwritten. Requires an active node selection.

        Parameters
        ----------
        **attributes
            Attribute name-value pairs to set on selected nodes. Each keyword becomes an attribute
            name, and its value becomes the attribute value.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_node('A', value=10)
        graph.add_node('B', value=20)

        # Select nodes and set attributes
        id_a = graph.get_id_from_label('A')
        id_b = graph.get_id_from_label('B')
        (graph
            .select_nodes_by_id([id_a, id_b])
            .set_node_attrs_ws(color='red', importance=0.9)
        )

        # Check updated attributes
        print(graph.get_node(id_a).attributes['color'])  # 'red'
        print(graph.get_node(id_a).attributes['value'])  # 10 (preserved)
        ```

        Bulk attribute updates with filtering:

        ```python
        graph = (
            Graph()
            .add_node(label="A", value=10)
            .add_node(label="B", value=20)
        )
        # Set category on high-value nodes
        (graph
            .select_nodes(lambda n: n.attributes.get('value', 0) > 15)
            .set_node_attrs_ws(category='high')
        )
        ```

        %family Node/Edge Data Frames
        """
        selected = self.get_selected_node_ids()

        for node_id in selected:
            for key, value in attributes.items():
                # Use Rust set_node_attr for persistence
                self._graph.set_node_attr(node_id, key, value)

        return self

    def set_edge_attrs(
        self,
        edge_attr: str,
        values: Any | list[Any] | dict[int, Any],
        edge_ids: list[int] | None = None,
        edges: list[tuple[Union[int, str], Union[int, str]]] | None = None,
        sources: list[Union[int, str]] | None = None,
        targets: list[Union[int, str]] | None = None,
    ) -> "Graph":
        """
        Set attribute values for one or more edges.

        Updates a specific attribute on edges. Can target edges by edge IDs, explicit (source,
        target) pairs, by filtering via source/target node lists, or apply to all edges. Supports
        various value input formats.

        Parameters
        ----------
        edge_attr
            The name of the attribute to set.
        values
            The value(s) to set. Can be a (1) single value, applied to all targeted edges, (2) a
            list, must match length of targeted edges, and (3) a dict, maps edge IDs to
            values.
        edge_ids
            Optional list of edge IDs specifying edges to update.
        edges
            Optional list of (source, target) tuples specifying edges to update.
        sources
            Optional list of source node IDs/labels to filter edges (use with targets).
        targets
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
        Set attribute for specific edges using edge IDs:

        ```python
        graph = (
            Graph()
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        # Set edge attributes using edge IDs (simpler!)
        edge_ids = list(graph.get_edge_ids())[:2]
        graph.set_edge_attrs("weight", 2.5, edge_ids=edge_ids)

        # Set same value for all edges
        graph.set_edge_attrs("status", "active")

        # Set different values using a list
        all_edge_ids = list(graph.get_edge_ids())
        graph.set_edge_attrs("priority", [1, 2, 3], edge_ids=all_edge_ids)
        ```

        Can also use (source, target) tuples:

        ```python
        graph = (
            Graph()
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Set using (source, target) tuples
        graph.set_edge_attrs("weight", 2.5, edges=[("A", "B")])
        ```

        %family Node/Edge Data Frames
        """
        # Determine which edges to update
        target_edge_ids = []

        if edge_ids is not None:
            # Use specific edge IDs directly
            target_edge_ids = edge_ids
        elif edges is not None:
            # Use specific (source, target) tuples
            for source, target in edges:
                try:
                    source_id = self._resolve_node_id(source)
                    target_id = self._resolve_node_id(target)
                    found_edge_ids = self._graph.find_edge_ids(source_id, target_id)
                    target_edge_ids.extend(found_edge_ids)
                except ValueError:
                    # Node doesn't exist so silently skip
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
                        # Node doesn't exist so silently skip
                        pass
        else:
            # All edges
            target_edge_ids = self.get_edge_ids()

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

    def set_edge_attrs_ws(self, **attributes) -> "Graph":
        """Set attributes on all selected edges.

        Updates or adds attributes to edges in the current selection. The `weight` attribute
        receives special handling as it's a first-class edge property. Requires an active edge
        selection.

        Parameters
        ----------
        **attributes
            Attribute name-value pairs to set on selected edges. Use `weight` to update edge weights
            directly.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_path_graph(3)

        # Select and update edge attributes
        edge_ids = graph.get_edge_ids()
        (graph
            .select_edges_by_id([edge_ids[0]])
            .set_edge_attrs_ws(weight=5.0, type='strong')
        )

        # Check updated attributes
        edge = graph.get_edge(edge_ids[0])
        print(edge.weight)  # 5.0
        print(edge.attributes['type'])  # 'strong'
        ```

        Bulk weight updates:

        ```python
        graph = Graph().add_path_graph(3)
        # Set all edge weights to 1.0
        (graph
            .select_edges(lambda e: True)  # Select all edges
            .set_edge_attrs_ws(weight=1.0)
        )
        ```

        %family Node/Edge Data Frames
        """
        selected = self.get_selected_edge_ids()

        for edge_id in selected:
            for key, value in attributes.items():
                # Use Rust set_edge_attr for persistence (handles weight specially)
                self._graph.set_edge_attr(edge_id, key, value)

        return self

    def get_node_attrs(
        self, node_attr: str, nodes: list[int | str] | None = None
    ) -> dict[int, any]:
        """
        Get attribute values for nodes.

        Extracts a specific attribute from one or more nodes. Only returns entries for nodes that
        have the specified attribute set. Useful for collecting properties across the graph.

        Parameters
        ----------
        node_attr
            Attribute name to extract.
        nodes
            Optional list of node IDs (int) or labels (str) to query. If `None`, queries all nodes.

        Returns
        -------
        dict[int, any]
            Dictionary mapping node IDs to attribute values. Only includes nodes that have the
            attribute set.

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
        # Returns dict with node IDs as keys: {1: 95, 2: 88, 3: 92}
        print(scores)

        types = graph.get_node_attrs("type")
        # Returns: {1: 'user', 2: 'admin', 3: 'user'}
        print(types)
        ```

        Get attribute from specific nodes using node IDs:

        ```python
        graph = (
            Graph()
            .add_node(label="A", type="user", score=95)
            .add_node(label="B", type="admin", score=88)
            .add_node(label="C", type="user", score=92)
        )
        node_ids = list(graph.get_node_ids())
        user_scores = graph.get_node_attrs("score", nodes=[node_ids[0], node_ids[2]])
        print(user_scores)  # {1: 95, 3: 92}
        ```

        Can also use labels:

        ```python
        graph = (
            Graph()
            .add_node(label="A", type="user", score=95)
            .add_node(label="B", type="admin", score=88)
            .add_node(label="C", type="user", score=92)
        )
        user_scores = graph.get_node_attrs("score", nodes=["A", "C"])
        print(user_scores)  # {1: 95, 3: 92}
        ```

        %family Node/Edge Data Frames
        """

        if nodes is None:
            nodes = self.get_node_ids(limit=None)
        else:
            # Resolve labels to IDs
            nodes = [self._resolve_node_id(nid) for nid in nodes]

        result = {}
        for node_id in nodes:
            node = self.get_node(node_id)
            if node and node.attributes:
                if node_attr in node.attributes:
                    result[node_id] = node.attributes[node_attr]

        return result

    def get_edge_attrs(
        self,
        edge_attr: str,
        sources: list[Union[int, str]] | None = None,
        targets: list[Union[int, str]] | None = None,
    ) -> dict[int, any]:
        """
        Get edge attribute values.

        Extracts a specific attribute from edges. Can optionally filter by source and/or target
        nodes. Returns a dictionary keyed by edge IDs.

        Parameters
        ----------
        edge_attr
            The name of the attribute to get.
        sources
            Optional list of source node IDs/labels to filter edges.
        targets
            Optional list of target node IDs/labels to filter edges.

        Returns
        -------
        dict[int, any]
            Dictionary mapping edge IDs (integers) to attribute values. Only includes edges that
            have the attribute set.

        Examples
        --------
        Get attribute from all edges:

        ```python
        graph = (
            Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_edge("A", "B", weight=1.5, type="friend")
            .add_edge("B", "C", weight=2.0, type="colleague")
            .add_edge("C", "D", weight=1.0, type="friend")
        )

        weights = graph.get_edge_attrs("weight")
        # Returns dict mapping edge IDs to weights

        types = graph.get_edge_attrs("type")
        # Returns dict mapping edge IDs to types
        ```

        %family Node/Edge Data Frames
        """
        result = {}

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
                            # Check special weight attribute first
                            if edge_attr == "weight" and edge.weight is not None:
                                result[edge_id] = edge.weight
                            # Then check attributes dict
                            elif edge.attributes and edge_attr in edge.attributes:
                                result[edge_id] = edge.attributes[edge_attr]
        else:
            # Get all edges
            for edge_id in self.get_edge_ids():
                edge = self.get_edge(edge_id)
                if edge:
                    # Check special weight attribute first
                    if edge_attr == "weight" and edge.weight is not None:
                        result[edge_id] = edge.weight
                    # Then check attributes dict
                    elif edge.attributes and edge_attr in edge.attributes:
                        result[edge_id] = edge.attributes[edge_attr]

        return result

    def get_edge_attrs_by_id(
        self, edge_attr: str, edges: list[int] | None = None
    ) -> dict[int, any]:
        """
        Get attribute values for edges by edge ID.

        Similar to `get_edge_attrs()` but uses internal edge IDs instead of node pairs. Returns a
        dictionary keyed by integer edge IDs. Useful when working directly with edge ID references.

        Parameters
        ----------
        edge_attr
            Attribute name to extract.
        edges
            Optional list of edge IDs to query. If `None`, queries all edges.

        Returns
        -------
        dict[int, any]
            Dictionary mapping edge IDs to attribute values. Only includes edges that have the
            attribute set.

        Examples
        --------
        Get attributes by edge ID:

        ```python
        graph = (
            Graph()
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B", weight=1.5, type="friend")
            .add_edge("B", "C", weight=2.0, type="colleague")
        )

        edge_ids = graph.get_edge_ids()
        weights_by_id = graph.get_edge_attrs_by_id("weight", edges=edge_ids[:1])
        # Returns {edge_id: 1.5} for first edge

        # Get all edge types
        all_types = graph.get_edge_attrs_by_id("type")
        # Maps each edge ID to its type value
        ```

        %family Node/Edge Data Frames
        """
        if edges is None:
            edges = self.get_edge_ids()

        result = {}
        for edge_id in edges:
            edge = self.get_edge(edge_id)
            if edge:
                # Check special weight attribute first
                if edge_attr == "weight" and edge.weight is not None:
                    result[edge_id] = edge.weight
                # Then check attributes dict
                elif edge.attributes and edge_attr in edge.attributes:
                    result[edge_id] = edge.attributes[edge_attr]

        return result

    def get_node_df(self) -> "pl.DataFrame":
        """
        Get nodes as a Polars DataFrame.

        Exports all nodes and their attributes to a Polars DataFrame for tabular analysis. Each row
        represents a node, with columns for the node ID and any attributes set on nodes. The
        DataFrame always includes 'id', 'label', and 'type' as the first three columns, sorted by
        `id` in ascending order.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame with mandatory columns 'id', 'label', 'type' (first three), followed
            by other attributes and 'degree'. Empty DataFrame if graph has no nodes.

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
        print(df.columns)  # ['id', 'label', 'type', 'score', 'degree']
        print(df.shape)    # (3, 5)

        # Calculate statistics on node attributes
        import polars as pl
        avg_score = df["score"].mean()
        user_count = df.filter(pl.col("type") == "user").height

        # Find high-degree nodes
        hubs = df.filter(pl.col("degree") > 5)
        ```

        Notes
        -----
        - the 'id', 'label', and 'type' columns are always present (first three columns)
        - if a node has no label, the label column will contain `None`
        - if a node has no type attribute, the type column will contain `None`
        - rows are sorted by id in ascending order
        - other attributes not present on a node will have `None` values
        - the 'degree' column is always last

        %family Node/Edge Data Frames
        """
        import polars as pl

        node_ids = self.get_node_ids(limit=None)
        if not node_ids:
            return pl.DataFrame()

        # Build data dictionary with mandatory columns first
        nodes = [self.get_node(nid) for nid in node_ids]

        data = {
            "id": node_ids,
            "label": [node.label if node else None for node in nodes],
            "type": [
                node.attributes.get("type") if node and node.attributes else None for node in nodes
            ],
        }

        # Get all unique attribute keys (excluding 'type' since it's already handled)
        attr_keys = set()
        for node in nodes:
            if node and node.attributes:
                attr_keys.update(node.attributes.keys())

        # Remove 'type' from attr_keys since it's a first-class column
        attr_keys.discard("type")

        # Extract other attributes (sorted alphabetically)
        for key in sorted(attr_keys):
            data[key] = [
                node.attributes.get(key) if node and node.attributes else None for node in nodes
            ]

        # Add degree column last
        data["degree"] = [self.get_degree(nid) for nid in node_ids]

        # Create DataFrame and sort by id
        df = pl.DataFrame(data)
        return df.sort("id")

    def get_node_df_ws(self) -> "pl.DataFrame":
        """
        Get selected nodes as a Polars DataFrame.

        Similar to get_node_df but only exports currently selected nodes. Useful for analyzing
        subsets of nodes after filtering with selection methods. The DataFrame always includes 'id',
        'label', and 'type' as the first three columns, sorted by 'id' in ascending order.

        Returns
        -------
        pl.DataFrame
            DataFrame with mandatory columns 'id', 'label', 'type' (first three), followed by other
            attributes and 'degree'. Empty DataFrame if no nodes are selected.

        Examples
        --------
        Export filtered nodes:

        ```python
        graph = (
            Graph()
            .add_star_graph(10)
            .add_path_graph(5)
            .select_nodes_by_degree(min_degree=3)
        )

        df = graph.get_node_df_ws()
        # Only high-degree nodes in DataFrame
        print(df.shape[0])  # Number of selected nodes
        ```

        Combine selection with DataFrame analysis:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")
        graph.select_nodes_by_id([a_id, b_id, c_id])
        selected_df = graph.get_node_df_ws()

        # Analyze just the selected nodes
        avg_degree = selected_df["degree"].mean()
        ```

        Notes
        -----
        - returns empty DataFrame if no nodes are selected
        - the 'id', 'label', and 'type' columns are always present (first three)
        - if a node has no label, the label column will contain None
        - if a node has no type attribute, the type column will contain None
        - rows are sorted by id in ascending order
        - use selection methods before calling this
        - same structure as `get_node_df()` but filtered to selection

        %family Node/Edge Data Frames
        """
        import polars as pl

        node_ids = self.get_selected_node_ids()
        if not node_ids:
            return pl.DataFrame()

        # Build data dictionary with mandatory columns first
        nodes = [self.get_node(nid) for nid in node_ids]

        data = {
            "id": node_ids,
            "label": [node.label if node else None for node in nodes],
            "type": [
                node.attributes.get("type") if node and node.attributes else None for node in nodes
            ],
        }

        # Get all unique attribute keys (excluding 'type' since it's already handled)
        attr_keys = set()
        for node in nodes:
            if node and node.attributes:
                attr_keys.update(node.attributes.keys())

        # Remove 'type' from attr_keys since it's a first-class column
        attr_keys.discard("type")

        # Extract other attributes (sorted alphabetically)
        for key in sorted(attr_keys):
            data[key] = [
                node.attributes.get(key) if node and node.attributes else None for node in nodes
            ]

        # Add degree column last
        data["degree"] = [self.get_degree(nid) for nid in node_ids]

        # Create DataFrame and sort by id
        df = pl.DataFrame(data)
        return df.sort("id")
        data["degree"] = [self.get_degree(nid) for nid in node_ids]

        return pl.DataFrame(data)

    def get_edge_df(self) -> "pl.DataFrame":
        """
        Get edges as a Polars DataFrame with standardized structure.

        Returns all edges with a consistent column structure:

        - column 1: `id` (i64, mandatory, sorted ascending)
        - column 2: `source` (i64, mandatory)
        - column 3: `target` (i64, mandatory)
        - column 4: `rel` (str, mandatory, null if not set)
        - columns 5-N: Other attributes (alphabetically sorted)
        - weight is included as a special attribute if present

        The DataFrame is always sorted by edge ID in ascending order.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame with edges sorted by `id`. Returns an empty DataFrame if the graph
            has no edges.

        Examples
        --------
        Export edges to DataFrame:

        ```python
        graph = (
            Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_edge("A", "B", weight=1.5, rel="friend")
            .add_edge("B", "C", weight=2.0, rel="colleague")
            .add_edge("C", "D", weight=1.0, rel="friend")
        )

        df = graph.get_edge_df()
        print(df.columns)  # ['id', 'source', 'target', 'rel', 'weight']
        print(df.shape)    # (3, 5)

        # Filter edges by relationship type
        import polars as pl
        friends = df.filter(pl.col("rel") == "friend")

        # Count relationship types
        rel_counts = df.group_by("rel").count()

        # Average weight per relationship type
        avg_by_rel = df.group_by("rel").agg(pl.col("weight").mean())
        ```

        Notes
        -----
        - first 4 columns always: `id`, `source`, `target`, `rel` (in that order)
        - missing `rel` values appear as `None`
        - other attributes are sorted alphabetically
        - DataFrame sorted by `id` ascending
        - `weight` treated as a special attribute column

        %family Node/Edge Data Frames
        """
        import polars as pl

        edge_ids = self.get_edge_ids(limit=None)
        if not edge_ids:
            return pl.DataFrame()

        # Get edge objects (avoid repeated get_edge calls)
        edges_list = [self.get_edge(eid) for eid in edge_ids]

        # Get source and target for each edge
        endpoints = [self.get_node_ids_by_edge_id(eid) for eid in edge_ids]

        # Build ordered dictionary for consistent column order
        data = {}
        data["id"] = edge_ids
        data["source"] = [e[0] if e else None for e in endpoints]
        data["target"] = [e[1] if e else None for e in endpoints]

        # Always include rel column (even if all null)
        data["rel"] = [e.attributes.get("rel") if e and e.attributes else None for e in edges_list]

        # Collect all attribute keys (excluding rel since it's first-class)
        attr_keys = set()

        # Check for weight
        if any(e and e.weight is not None for e in edges_list):
            attr_keys.add("weight")

        # Add other attributes from dict
        for edge in edges_list:
            if edge and edge.attributes:
                attr_keys.update(edge.attributes.keys())

        # Remove rel from generic attributes (already handled as first-class)
        attr_keys.discard("rel")

        # Sort remaining attributes alphabetically
        for key in sorted(attr_keys):
            if key == "weight":
                data[key] = [e.weight if e else None for e in edges_list]
            else:
                data[key] = [
                    e.attributes.get(key) if e and e.attributes else None for e in edges_list
                ]

        # Create DataFrame and sort by id
        df = pl.DataFrame(data)
        return df.sort("id")

    def get_edge_df_ws(self) -> "pl.DataFrame":
        """
        Get selected edges as a Polars DataFrame with standardized structure.

        Similar to `get_edge_df()` but only exports currently selected edges. Uses the same
        consistent column structure with `id`, `source`, `target`, and `rel` as first 4 columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with edges sorted by id, containing:

            - column 1: `id` (i64, mandatory)
            - column 2: `source` (i64, mandatory)
            - column 3: `target` (i64, mandatory)
            - column 4: `rel` (str, mandatory, `None` if not set)
            - columns 5-N: Other attributes (alphabetically sorted)

            Returns empty DataFrame if no edges are selected.

        Examples
        --------
        Export filtered edges:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.5, rel="friend")
            .add_edge(source="B", target="C", weight=0.5, rel="colleague")
            .add_edge(source="C", target="D", weight=2.0, rel="friend")
            .select_edges(lambda e: e.attributes and e.attributes.get("rel") == "friend")
        )

        df = graph.get_edge_df_ws()
        # Only friend edges in DataFrame
        print(df["rel"].value_counts())
        ```

        Combine selection with DataFrame analysis:

        ```python
        graph = Graph().add_path_graph(10)
        edge_ids = list(graph.get_edge_ids())[:5]
        graph.select_edges_by_id(edge_ids)

        selected_df = graph.get_edge_df_ws()
        # Analyze just the first 5 edges
        ```

        Notes
        -----
        - first 4 columns always: `id`, `source`, `target`, `rel` (in that order)
        - missing rel values appear as `None`
        - DataFrame sorted by `id` ascending
        - same structure as `get_edge_df()` but filtered to selection

        %family Node/Edge Data Frames
        """
        import polars as pl

        edge_ids = self.get_selected_edge_ids()
        if not edge_ids:
            return pl.DataFrame()

        # Get edge objects (avoid repeated get_edge calls)
        edges_list = [self.get_edge(eid) for eid in edge_ids]

        # Get source and target for each edge
        endpoints = [self.get_node_ids_by_edge_id(eid) for eid in edge_ids]

        # Build ordered dictionary for consistent column order
        data = {}
        data["id"] = edge_ids
        data["source"] = [e[0] if e else None for e in endpoints]
        data["target"] = [e[1] if e else None for e in endpoints]

        # Always include rel column (even if all null)
        data["rel"] = [e.attributes.get("rel") if e and e.attributes else None for e in edges_list]

        # Collect all attribute keys (excluding rel since it's first-class)
        attr_keys = set()

        # Check for weight
        if any(e and e.weight is not None for e in edges_list):
            attr_keys.add("weight")

        # Add other attributes from dict
        for edge in edges_list:
            if edge and edge.attributes:
                attr_keys.update(edge.attributes.keys())

        # Remove rel from generic attributes (already handled as first-class)
        attr_keys.discard("rel")

        # Sort remaining attributes alphabetically
        for key in sorted(attr_keys):
            if key == "weight":
                data[key] = [e.weight if e else None for e in edges_list]
            else:
                data[key] = [
                    e.attributes.get(key) if e and e.attributes else None for e in edges_list
                ]

        # Create DataFrame and sort by id
        df = pl.DataFrame(data)
        return df.sort("id")

    def mutate_node_attrs_ws(self, node_attr: str, func: Callable) -> "Graph":
        """
        Transform a node attribute using a function on selected nodes.

        Applies a transformation function to the specified attribute for all selected nodes. The
        function receives each node's current attribute value and returns the new value. Requires an
        active node selection.

        Parameters
        ----------
        node_attr
            Name of the attribute to transform.
        func
            Transformation function that takes the current attribute value and returns the new
            value. Called once per selected node.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_node('A', score=10)
        graph.add_node('B', score=20)
        graph.add_node('C', score=30)

        # Double the score for selected nodes
        id_a = graph.get_id_from_label('A')
        id_b = graph.get_id_from_label('B')
        id_c = graph.get_id_from_label('C')
        (graph
            .select_nodes_by_id([id_a, id_b])
            .mutate_node_attrs_ws('score', lambda x: x * 2)
        )

        print(graph.get_node(id_a).attributes['score'])  # 20
        print(graph.get_node(id_b).attributes['score'])  # 40
        print(graph.get_node(id_c).attributes['score'])  # 30 (unchanged)
        ```

        Conditional transformations:

        ```python
        graph = (
            Graph()
            .add_node(label="A", price=80)
            .add_node(label="B", price=120)
            .add_node(label="C", price=150)
        )
        # Apply discount to high-value nodes
        (graph
            .select_nodes(lambda n: n.attributes.get('price', 0) > 100)
            .mutate_node_attrs_ws('price', lambda p: p * 0.9)
        )
        ```

        Notes
        -----
        Nodes without the specified attribute are skipped.

        %family Node/Edge Data Frames
        """
        selected = self.get_selected_node_ids()

        for node_id in selected:
            node = self.get_node(node_id)
            if node and node.attributes and node_attr in node.attributes:
                old_value = node.attributes[node_attr]
                new_value = func(old_value)
                # Use Rust method to persist the change
                self._graph.set_node_attr(node_id, node_attr, new_value)

        return self

    def mutate_edge_attrs_ws(self, edge_attr: str, func: Callable) -> "Graph":
        """
        Transform an edge attribute using a function on selected edges.

        Applies a transformation function to the specified attribute for all selected edges.
        Supports transforming the `weight` field directly. Requires an active edge selection.

        Parameters
        ----------
        edge_attr
            Name of the attribute to transform. Can be `"weight"` to transform edge weights
            directly.
        func
            Transformation function that takes the current attribute value and returns the new
            value. Called once per selected edge.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B", weight=1.5)
            .add_edge(source="B", target="C", weight=2.0)
        )

        # Double the weight of selected edges
        edge_ids = list(graph.get_edge_ids())
        (graph
            .select_edges_by_id([edge_ids[0]])
            .mutate_edge_attrs_ws('weight', lambda x: x * 2)
        )

        print(graph.get_edge(edge_ids[0]).weight)  # 3.0
        print(graph.get_edge(edge_ids[1]).weight)  # 2.0 (unchanged)
        ```

        Normalize edge weights:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B", weight=1.5)
            .add_edge(source="B", target="C", weight=3.0)
        )
        # Get max weight for normalization
        max_weight = max(e.weight for e in [graph.get_edge(eid) for eid in graph.get_edge_ids()] if e.weight)

        # Normalize all weights to [0, 1]
        (
            graph
            .select_edges(lambda e: e.weight is not None)
            .mutate_edge_attrs_ws('weight', lambda w: w / max_weight)
        )
        ```

        Notes
        -----
        Edges without the specified attribute are skipped.

        %family Node/Edge Data Frames
        """
        selected = self.get_selected_edge_ids()

        for edge_id in selected:
            edge = self.get_edge(edge_id)
            if edge:
                if edge_attr == "weight" and edge.weight is not None:
                    new_value = func(edge.weight)
                    self._graph.set_edge_attr(edge_id, "weight", new_value)
                elif edge.attributes and edge_attr in edge.attributes:
                    old_value = edge.attributes[edge_attr]
                    new_value = func(old_value)
                    self._graph.set_edge_attr(edge_id, edge_attr, new_value)

        return self

    def colorize_node_attrs(
        self,
        node_attr_from: str = None,
        values: dict[str, Any] = None,
        node_attr_to: str = "color",
        palette: str | list[str] = "default",
        alpha: Optional[int] = None,
        nodes: Optional[list[str]] = None,
    ) -> "Graph":
        """
        Map node attribute values to colors.

        Sets a color attribute on nodes based on categorical or continuous values. For categorical
        data, assigns distinct colors from a palette. For continuous data, interpolates between
        colors.

        Parameters
        ----------
        node_attr_from
            Attribute name to read values from (e.g., 'community', 'type'). If `None`, must provide
            values dict.
        values
            Dictionary mapping node IDs to values. If `None`, reads from `node_attr_from=`
            attribute.
        node_attr_to
            Attribute name to write colors to (e.g., 'color', 'fillcolor', etc.).
        palette
            Color palette name or list of hex color strings. Available palettes: ColorBrewer (e.g.,
            'Greens', 'RdBu', 'Set1'), Viridis ('viridis', 'magma', 'plasma'), or 'default'.
        alpha
            Alpha transparency (0-100), not yet implemented.
        nodes
            Subset of nodes to colorize (default: all nodes).

        Returns
        -------
        Graph
            Self for method chaining

        Examples
        --------
        Color by community:

        ```python
        # Create a simple graph for demonstration
        G = Graph(directed=False)
        for i in range(6):
            G.add_node(label=str(i))

        # Create two communities
        G.add_edge(source="0", target="1")
        G.add_edge(source="1", target="2")
        G.add_edge(source="0", target="2")
        G.add_edge(source="3", target="4")
        G.add_edge(source="4", target="5")
        G.add_edge(source="3", target="5")

        # Detect communities and colorize
        G.cmty_louvain(attr_name="community")
        G.colorize_node_attrs(node_attr_from="community", palette="vibrant")
        ```

        Color by node type:

        ```python
        # Create graph with typed nodes
        G = Graph()
        G.add_node(label="A", type="person")
        G.add_node(label="B", type="person")
        G.add_node(label="C", type="organization")
        G.add_edge(source="A", target="B")
        G.add_edge(source="B", target="C")

        G.colorize_node_attrs(node_attr_from="type", palette="tableau")
        ```

        Custom color list:

        ```python
        G = Graph()
        G.add_node(label="A", status="active")
        G.add_node(label="B", status="inactive")
        G.add_edge(source="A", target="B")

        G.colorize_node_attrs(
            node_attr_from="status",
            palette=["#ff0000", "#00ff00"]
        )
        ```

        %family Node/Edge Data Frames
        """
        from edgecase.colors import COLOR_PALETTES

        if nodes is None:
            nodes = self.get_node_ids(limit=None)

        # Get values to colorize
        if values is None:
            if node_attr_from is None:
                raise ValueError("Must provide either node_attr_from or values")
            values = self.get_node_attrs(node_attr_from, nodes=nodes)

        if not values:
            return self

        # Get unique values and assign colors
        unique_values = sorted(list(set(values.values())))

        # Get color palette
        if isinstance(palette, str):
            colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])
        else:
            colors = palette

        # Create color mapping
        n_values = len(unique_values)
        if n_values <= len(colors):
            color_list = colors[:n_values]
        else:
            # Cycle through colors if we have more values than colors
            color_list = (colors * (n_values // len(colors) + 1))[:n_values]

        color_map = dict(zip(unique_values, color_list))

        # Set colors on nodes
        for node_label, value in values.items():
            color = color_map.get(value, "#cccccc")
            # Convert string label to internal node ID if needed
            node_id = (
                node_label if isinstance(node_label, int) else self.get_id_from_label(node_label)
            )
            self._graph.set_node_attr(node_id, node_attr_to, color)

        # Track aesthetic
        source = node_attr_from if node_attr_from else "values"
        palette_name = palette if isinstance(palette, str) else "custom"
        self._aesthetics.append((node_attr_to, source, "colorize", {"palette": palette_name}))

        return self

    def rescale_node_attrs(
        self,
        node_attr_from: str = None,
        values: dict[str, int | float] = None,
        node_attr_to: str = "size",
        range: list[float] = None,
        nodes: Optional[list[str]] = None,
    ) -> "Graph":
        """
        Rescale node attribute values to a range.

        Normalizes numeric values to a specified range, useful for sizing nodes by centrality,
        degree, or other continuous metrics.

        Parameters
        ----------
        node_attr_from
            Attribute name to read values from (e.g., 'betweenness', 'degree'). If `None`, must
            provide values dict.
        values
            Dictionary mapping node IDs to numeric values. If `None`, reads from the
            `node_attr_from=` attribute.
        node_attr_to
            Attribute name to write rescaled values to.
        range
            Output range as [min, max]. Values will be rescaled to this range.
        nodes
            Subset of nodes to rescale (default: all nodes).

        Returns
        -------
        Graph
            Self for method chaining

        Examples
        --------
        Size by betweenness centrality:

        ```python
        # Create a simple graph
        G = Graph()
        for i in range(5):
            G.add_node(label=str(i))
        G.add_edge(source="0", target="1")
        G.add_edge(source="1", target="2")
        G.add_edge(source="2", target="3")
        G.add_edge(source="3", target="4")

        # Calculate betweenness and rescale to node sizes
        betweenness = G.betweenness_centrality(return_dict=True)
        G.rescale_node_attrs(
            values=betweenness,
            node_attr_to="size",
            range=[5, 20]
        )
        ```

        Scale by degree:

        ```python
        # Create a simple graph
        G = Graph()
        for i in range(4):
            G.add_node(label=str(i))
        G.add_edge(source="0", target="1")
        G.add_edge(source="0", target="2")
        G.add_edge(source="0", target="3")

        # Calculate degrees and rescale
        degrees = {n: len(G.neighbors(node=n)) for n in G.get_node_ids(limit=None)}
        G.rescale_node_attrs(
            values=degrees,
            node_attr_to="width",
            range=[1, 10]
        )
        ```

        %family Node/Edge Data Frames
        """
        # Default range
        if range is None:
            range = [0.5, 1.0]

        if len(range) != 2:
            raise ValueError("range must be a list of two values [min, max]")

        to_lower_bound, to_upper_bound = range[0], range[1]

        if nodes is None:
            nodes = self.get_node_ids(limit=None)

        # Get values to rescale
        if values is None:
            if node_attr_from is None:
                raise ValueError("Must provide either node_attr_from or values")
            values = self.get_node_attrs(node_attr_from, nodes=nodes)

        if not values:
            return self

        # Get min/max for normalization
        value_list = list(values.values())
        min_val = min(value_list)
        max_val = max(value_list)

        # Avoid division by zero
        value_range = max_val - min_val if max_val > min_val else 1.0
        output_range = to_upper_bound - to_lower_bound

        # Rescale and set on nodes
        for node_label, value in values.items():
            normalized = (value - min_val) / value_range
            rescaled = to_lower_bound + (normalized * output_range)
            # Convert string label to internal node ID if needed
            node_id = (
                node_label if isinstance(node_label, int) else self.get_id_from_label(node_label)
            )
            self._graph.set_node_attr(node_id, node_attr_to, rescaled)

        # Track aesthetic
        source = node_attr_from if node_attr_from else "values"
        self._aesthetics.append(
            (node_attr_to, source, "rescale", {"range": [to_lower_bound, to_upper_bound]})
        )

        return self

    def set_node_attr_to_display(
        self,
        attr: Optional[str] = None,
    ) -> "Graph":
        """
        Set which node attribute to display in tooltips/labels.

        Parameters
        ----------
        attr
            Attribute name to display. If `None`, shows node ID.

        Returns
        -------
        Graph
            Self for method chaining

        Examples
        --------
        ```python
        G = Graph()
        G.add_node(label="Alice", age=30)
        G.add_node(label="Bob", age=25)
        G.add_edge(source="Alice", target="Bob")

        G.set_node_attr_to_display(attr="label")
        G.set_node_attr_to_display(attr=None)  # Reset to node IDs
        ```

        %family Node/Edge Data Frames
        """
        self._display_attr = attr
        return self

    # ==============================================================================
    # Link Analysis
    # ==============================================================================

    def dispersion(self, u: Union[int, str], v: Union[int, str], normalized: bool = True) -> float:
        """
        Calculate dispersion between two nodes.

        Dispersion measures the embeddedness of a link in the network. A link between two nodes
        (`u` and `v`) has high dispersion when their mutual neighbors (friends in common) are not
        well connected to each other. This can indicate structural diversity in the network.

        Parameters
        ----------
        u
            Source node (node ID or label).
        v
            Target node (node ID or label).
        normalized
            If `True` (default), normalize by the number of common neighbors (embeddedness).

        Returns
        -------
        float
            The dispersion score between `u` and `v`.

        Examples
        --------
        Basic dispersion calculation:

        ```python
        from edgecase import Graph

        # Create a simple social network
        graph = Graph()
        for i in range(1, 6):
            graph.add_node()
        for s, t in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]:
            graph.add_edge(s, t)

        # Calculate dispersion between nodes 1 and 2
        disp = graph.dispersion(1, 2)
        print(f"Dispersion: {disp:.2f}")
        ```

        Using node labels:

        ```python
        graph = Graph()
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        graph.add_node(label="Charlie")
        graph.add_node(label="David")

        alice = graph.get_id_from_label("Alice")
        bob = graph.get_id_from_label("Bob")
        charlie = graph.get_id_from_label("Charlie")
        david = graph.get_id_from_label("David")

        graph.add_edge(alice, bob)
        graph.add_edge(alice, charlie)
        graph.add_edge(bob, charlie)
        graph.add_edge(bob, david)

        # Check dispersion
        disp = graph.dispersion(alice, bob, normalized=True)
        print(f"Dispersion: {disp:.2f}")
        ```

        Normalized vs unnormalized:

        ```python
        graph = Graph()
        graph.add_cycle_graph(6)

        nodes = sorted(graph.get_node_ids())

        # Normalized divides by embeddedness
        norm = graph.dispersion(nodes[0], nodes[1], normalized=True)
        # Unnormalized gives raw count
        unnorm = graph.dispersion(nodes[0], nodes[1], normalized=False)

        print(f"Normalized: {norm:.2f}, Unnormalized: {unnorm}")
        ```

        Notes
        -----
        - higher dispersion indicates the link spans different network regions
        - low dispersion means the link is embedded in a dense cluster
        - useful for identifying bridges between communities

        %family Link Analysis
        """
        from edgecase._core import dispersion

        u_id = self._resolve_node_id(u)
        v_id = self._resolve_node_id(v)

        return dispersion(self._graph, u_id, v_id, normalized)

    # ==============================================================================
    # Link Prediction
    # ==============================================================================

    def common_neighbors(
        self, ebunch: list[tuple[Union[int, str], Union[int, str]]] = None
    ) -> list[tuple[int, int, int]]:
        """
        Compute the number of common neighbors for node pairs.

        For each pair of nodes `(u, v)`, counts how many nodes are neighbors of both `u` and `v`.
        Common neighbors is a simple but effective link prediction measure.

        Parameters
        ----------
        ebunch
            List of node pairs to compute predictions for. Each pair can use node IDs (integers)
            or labels (strings). If `None`, computes for all pairs of nodes that are not already
            connected.

        Returns
        -------
        list[tuple[int, int, int]]
            List of `(u, v, count)` tuples where count is the number of common neighbors. Results
            are returned with node IDs.

        Examples
        --------
        Predict missing links in a social network:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        for name in ["Alice", "Bob", "Carol", "Dave"]:
            graph.add_node(label=name)

        # Alice and Carol share Bob as a friend
        graph.add_edge(source="Alice", target="Bob")
        graph.add_edge(source="Carol", target="Bob")
        graph.add_edge(source="Carol", target="Dave")

        # Check common neighbors for unconnected pairs
        predictions = graph.common_neighbors()
        for u, v, count in predictions:
            print(f"{u} - {v}: {count} common neighbors")
        ```

        Check specific pairs:

        ```python
        graph = Graph(directed=False)
        for name in ["Alice", "Bob", "Carol", "Dave"]:
            graph.add_node(label=name)
        graph.add_edge(source="Alice", target="Bob")
        graph.add_edge(source="Carol", target="Bob")

        pairs = [("Alice", "Carol"), ("Alice", "Dave")]
        predictions = graph.common_neighbors(pairs)
        # Alice-Carol have 1 common neighbor (Bob)
        # Alice-Dave have 0 common neighbors
        ```

        Notes
        -----
        - time complexity: O(k^2) per pair where k is the average degree
        - works best for social networks with high clustering
        - simple baseline for link prediction tasks
        - higher count suggests higher likelihood of future connection

        %family Link Prediction
        """
        from edgecase._core import common_neighbors

        resolved_ebunch = None
        if ebunch is not None:
            resolved_ebunch = [
                (self._resolve_node_id(u), self._resolve_node_id(v)) for u, v in ebunch
            ]

        return common_neighbors(self._graph, resolved_ebunch)

    def jaccard_coefficient(
        self, ebunch: list[tuple[Union[int, str], Union[int, str]]] = None
    ) -> list[tuple[int, int, float]]:
        """
        Compute Jaccard coefficient for node pairs.

        For each pair of nodes `(u, v)`, computes `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`, which is the
        ratio of common neighbors to total distinct neighbors.

        Parameters
        ----------
        ebunch
            List of node pairs to compute predictions for. Each pair can use node IDs (integers) or
            labels (strings). If `None`, computes for all pairs of nodes that are not already
            connected.

        Returns
        -------
        list[tuple[int, int, float]]
            List of `(u, v, coefficient)` tuples where coefficient is the Jaccard similarity.
            Results are returned with node IDs. Coefficient is `0.0` if both nodes have no
            neighbors.

        Examples
        --------
        Find similar nodes for link prediction:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        graph.add_path_graph(5)  # 1-2-3-4-5

        # Check similarity between non-adjacent nodes
        predictions = graph.jaccard_coefficient([(1, 3), (1, 4), (2, 4)])
        for u, v, coef in predictions:
            print(f"{u} - {v}: Jaccard = {coef:.3f}")
        ```

        Notes
        -----
        - time complexity: `O(k)` per pair where k is the average degree
        - returns value in `[0, 1]` where `1` means identical neighborhoods
        - more robust than common neighbors for nodes with different degrees
        - commonly used as similarity metric in network analysis

        %family Link Prediction
        """
        from edgecase._core import jaccard_coefficient

        resolved_ebunch = None
        if ebunch is not None:
            resolved_ebunch = [
                (self._resolve_node_id(u), self._resolve_node_id(v)) for u, v in ebunch
            ]

        return jaccard_coefficient(self._graph, resolved_ebunch)

    def adamic_adar_index(
        self, ebunch: list[tuple[Union[int, str], Union[int, str]]] = None
    ) -> list[tuple[int, int, float]]:
        """
        Compute Adamic-Adar index for node pairs.

        For each pair of nodes `(u, v)`, computes `Σ(1 / log(|N(w)|))` for all common neighbors `w`.
        This gives more weight to common neighbors that themselves have fewer connections (rarer
        connections are more significant).

        Parameters
        ----------
        ebunch
            List of node pairs to compute predictions for. Each pair can use node IDs (integers)
            or labels (strings). If `None`, computes for all pairs of nodes that are not already
            connected.

        Returns
        -------
        list[tuple[int, int, float]]
            List of `(u, v, index)` tuples where index is the Adamic-Adar score. Results are
            returned with node IDs.

        Examples
        --------
        Predict links giving more weight to rare common neighbors:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        # Create a network with a hub (node 1) and peripheral nodes
        graph.add_node()  # Hub node 1
        for i in range(2, 10):
            graph.add_node()
            graph.add_edge(1, i)  # Hub connected to all

        # Nodes 2 and 3 share hub as common neighbor
        # But hub has high degree, so contributes less
        predictions = graph.adamic_adar_index([(2, 3)])
        print(f"Adamic-Adar: {predictions[0][2]:.3f}")
        ```

        Compare with common neighbors:

        ```python
        graph = Graph(directed=False)
        graph.add_cycle_graph(6)

        pairs = [(1, 3), (1, 4)]
        cn = graph.common_neighbors(pairs)
        aa = graph.adamic_adar_index(pairs)

        for (_, _, count), (u, v, score) in zip(cn, aa):
            print(f"{u}-{v}: CN={count}, AA={score:.3f}")
        ```

        Notes
        -----
        - time complexity: `O(k)` per pair where `k` is the average degree
        - more accurate than common neighbors for many real networks
        - handles hub nodes better by downweighting their contribution
        - score is unbounded (higher is better)

        %family Link Prediction
        """
        from edgecase._core import adamic_adar_index

        resolved_ebunch = None
        if ebunch is not None:
            resolved_ebunch = [
                (self._resolve_node_id(u), self._resolve_node_id(v)) for u, v in ebunch
            ]

        return adamic_adar_index(self._graph, resolved_ebunch)

    def resource_allocation_index(
        self, ebunch: list[tuple[Union[int, str], Union[int, str]]] = None
    ) -> list[tuple[int, int, float]]:
        """
        Compute resource allocation index for node pairs.

        For each pair of nodes `(u, v)`, computes `Σ(1 / |N(w)|)` for all common neighbors `w`.
        Similar to Adamic-Adar but uses `1/degree` instead of `1/log(degree)`, penalizing
        high-degree common neighbors more strongly.

        Parameters
        ----------
        ebunch
            List of node pairs to compute predictions for. Each pair can use node IDs (integers) or
            labels (strings). If `None`, computes for all pairs of nodes that are not already
            connected.

        Returns
        -------
        list[tuple[int, int, float]]
            List of `(u, v, index)` tuples where index is the resource allocation score. Results are
            returned with node IDs.

        Examples
        --------
        Compare with Adamic-Adar:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        # Star graph: node 1 is hub connected to 2-10
        graph.add_node()  # Hub node 1
        for i in range(2, 11):
            graph.add_node()
            graph.add_edge(1, i)

        pairs = [(2, 3), (2, 5)]
        aa = graph.adamic_adar_index(pairs)
        ra = graph.resource_allocation_index(pairs)

        for (_, _, aa_score), (u, v, ra_score) in zip(aa, ra):
            print(f"{u}-{v}: AA={aa_score:.3f}, RA={ra_score:.3f}")
        ```

        Notes
        -----
        - time complexity: O(k) per pair where k is the average degree
        - penalizes high-degree common neighbors more than Adamic-Adar
        - inspired by resource allocation dynamics on networks
        - often outperforms Adamic-Adar on social networks

        %family Link Prediction
        """
        from edgecase._core import resource_allocation_index

        resolved_ebunch = None
        if ebunch is not None:
            resolved_ebunch = [
                (self._resolve_node_id(u), self._resolve_node_id(v)) for u, v in ebunch
            ]

        return resource_allocation_index(self._graph, resolved_ebunch)

    def preferential_attachment(
        self, ebunch: list[tuple[Union[int, str], Union[int, str]]] = None
    ) -> list[tuple[int, int, int]]:
        """
        Compute preferential attachment score for node pairs.

        For each pair of nodes `(u, v)`, computes `|N(u)| × |N(v)|`, the product of their degrees.
        Based on the "rich get richer" principle where high-degree nodes are more likely to acquire
        new connections.

        Parameters
        ----------
        ebunch
            List of node pairs to compute predictions for. Each pair can use node IDs (integers) or
            labels (strings). If `None`, computes for all pairs of nodes that are not already
            connected.

        Returns
        -------
        list[tuple[int, int, int]]
            List of `(u, v, score)` tuples where `score` is `degree(u) × degree(v)`. Results are
            returned with node IDs.

        Examples
        --------
        Predict links based on node popularity:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        # Create network with varying degrees
        graph.add_star_graph(5)  # Hub node 1 with degree 4
        graph.add_node()  # Node 6, isolated

        pairs = [(2, 3), (2, 6)]
        predictions = graph.preferential_attachment(pairs)

        for u, v, score in predictions:
            print(f"{u}-{v}: PA score = {score}")
        # 2-3 will have higher score (both connected to hub)
        ```

        Compare different metrics:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(5)  # Complete graph K5
        graph.delete_edge(source=1, target=2)  # Remove one edge

        pairs = [(1, 2)]
        cn = graph.common_neighbors(pairs)
        pa = graph.preferential_attachment(pairs)

        print(f"Common neighbors: {cn[0][2]}")
        print(f"Preferential attachment: {pa[0][2]}")
        ```

        Notes
        -----
        - time complexity: O(1) per pair (just degree lookup)
        - does not use local neighborhood structure
        - captures "rich get richer" dynamics in scale-free networks
        - good baseline for networks with power-law degree distribution

        %family Link Prediction
        """
        from edgecase._core import preferential_attachment

        resolved_ebunch = None
        if ebunch is not None:
            resolved_ebunch = [
                (self._resolve_node_id(u), self._resolve_node_id(v)) for u, v in ebunch
            ]

        return preferential_attachment(self._graph, resolved_ebunch)

    # ==============================================================================
    # Similarity
    # ==============================================================================

    def get_pairwise_similarity(
        self, nodes: list[Union[int, str]] = None, method: str = "jaccard"
    ) -> list[list[float]]:
        """
        Calculate pairwise similarity between nodes based on neighborhoods.

        Computes structural similarity using various metrics that compare the neighborhoods
        (adjacent nodes) of node pairs. Returns a similarity matrix where higher values indicate
        more similar neighborhood structures.

        Parameters
        ----------
        nodes
            List of node IDs (integers) or labels (strings) to compute similarities for. Can mix IDs
            and labels in the same list. If `None` or empty, computes for all nodes in the graph.
        method
            Similarity metric to use. Options are: (1) `"jaccard"` for the Jaccard coefficient
            (intersection over union), (2) `"dice"` for the Dice coefficient (2× intersection over
            sum of sizes), (3) `"invlogweighted"` as inverse log-weighted (Adamic-Adar index), and
            (4) `"cosine"` which uses Cosine similarity.

        Returns
        -------
        list[list[float]]
            Similarity matrix where entry (i,j) is the similarity between nodes[i] and nodes[j].
            Diagonal entries are always `1.0`.

        Raises
        ------
        ValueError
            If a node ID or label is not found, or if method is invalid.

        Examples
        --------
        **Link Prediction in Social Network**

        ```python
        # Find similar users for recommendation
        g = Graph(directed=False)
        for name in ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]:
            g.add_node(label=name)
        g.add_edge(source="Alice", target="Bob")
        g.add_edge(source="Alice", target="Carol")
        g.add_edge(source="Bob", target="Carol")
        g.add_edge(source="Dave", target="Eve")
        g.add_edge(source="Dave", target="Frank")
        g.add_edge(source="Eve", target="Frank")

        # Alice and Dave have similar patterns (both connect to 2 people)
        sim = g.get_pairwise_similarity(["Alice", "Dave"], method="jaccard")
        # sim[0][1] = similarity between Alice and Dave

        # Can also use node IDs (or mix IDs and labels)
        sim = g.get_pairwise_similarity([1, 4], method="jaccard")
        ```

        **Comparing Similarity Metrics**

        ```python
        # Bipartite-like structure
        g = Graph(directed=False)
        for name in ["A", "B", "X", "Y"]:
            g.add_node(label=name)
        g.add_edge(source="A", target="X")
        g.add_edge(source="A", target="Y")
        g.add_edge(source="B", target="X")
        g.add_edge(source="B", target="Y")

        # A and B have identical neighborhoods
        jaccard = g.get_pairwise_similarity(["A", "B"], method="jaccard")
        dice = g.get_pairwise_similarity(["A", "B"], method="dice")
        cosine = g.get_pairwise_similarity(["A", "B"], method="cosine")

        # All should show high similarity (≈ 1.0)
        assert jaccard[0][1] == 1.0
        assert dice[0][1] == 1.0
        assert cosine[0][1] == 1.0
        ```

        **Link Prediction with Inverse Log-Weighted**

        ```python
        # Protein interaction network
        g = Graph(directed=False)
        for name in ["P1", "P2", "Hub", "P3", "P4"]:
            g.add_node(label=name)
        g.add_edge(source="P1", target="Hub")
        g.add_edge(source="P2", target="Hub")
        g.add_edge(source="Hub", target="P3")
        g.add_edge(source="Hub", target="P4")

        # P1 and P2 connected through high-degree hub
        # Inverse log-weighted gives lower weight to high-degree nodes
        sim = g.get_pairwise_similarity(["P1", "P2"], method="invlogweighted")
        # Lower similarity due to common high-degree neighbor
        ```

        **Community Detection Helper**

        ```python
        # Identify similar nodes within network
        g = Graph(directed=False)
        # Create nodes first
        for i in range(5):
            g.add_node(label=f"C1_{i}")
        # Add edges for a community structure
        for i in range(5):
            for j in range(i+1, 5):
                g.add_edge(source=f"C1_{i}", target=f"C1_{j}")

        # Get all pairwise similarities
        all_sim = g.get_pairwise_similarity(method="jaccard")
        # Nodes in same dense community have high similarity
        ```

        Notes
        -----
        **Jaccard Similarity**:

        - the most common metric
        - range: [0, 1]
        - sim(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

        **Dice Similarity**:

        - gives more weight to common neighbors than Jaccard
        - range: [0, 1]
        - sim(u,v) = 2|N(u) ∩ N(v)| / (|N(u)| + |N(v)|)

        **Inverse Log-Weighted (Adamic-Adar)**:

        - weights common neighbors by inverse log of their degree
        - rare neighbors contribute more
        - sim(u,v) = Σ_{w ∈ N(u) ∩ N(v)} 1/log(|N(w)|)
        - popular for link prediction

        **Cosine Similarity**:

        - normalized dot product
        - range: [0, 1]
        - sim(u,v) = |N(u) ∩ N(v)| / √(|N(u)| × |N(v)|)

        **Use Cases**:

        - link prediction: which edges are likely to form?
        - recommendations: find similar users/items
        - community detection: group similar nodes
        - network embedding: similarity-based features

        %family Similarity
        """
        if nodes is None:
            nodes = []
        return self._graph.get_pairwise_similarity(nodes, method)

    def get_dice_similarity(self) -> dict[int, dict[int, float]]:
        """
        Get Dice similarity coefficient between all node pairs.

        The Dice coefficient measures similarity between two nodes based on their neighborhood
        overlap:

        `Dice(A, B) = 2 × |neighbors(A) ∩ neighbors(B)| / (|neighbors(A)| + |neighbors(B)|)`

        Range: `[0, 1]` where `0` means no common neighbors and `1` means identical neighborhoods
        (or same node). Self-similarity is always `1.0`.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores. Format:
            `{node1_id: {node2_id: similarity_score}}`.

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
        # Returns dict with node IDs as keys
        a_id = graph.get_id_from_label("A")
        d_id = graph.get_id_from_label("D")
        print(sim[a_id][d_id])  # 1.0 (perfect overlap)
        ```

        Notes
        -----
        Useful for:

        - link prediction (similar nodes likely to connect)
        - community detection (nodes with similar neighborhoods)
        - recommendation systems (find similar users/items)

        %family Similarity
        """
        return self._graph.get_dice_similarity()

    def get_jaccard_similarity(self) -> dict[int, dict[int, float]]:
        """
        Get Jaccard similarity index between all node pairs.

        The Jaccard index is a classic set similarity measure:

        `Jaccard(A, B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|`

        Range: `[0, 1]` where `0` means no common neighbors and `1` means identical neighborhoods
        (or same node). More conservative than Dice coefficient (always ≤ Dice score).
        Self-similarity is always `1.0`.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores. Format:
            `{node1_id: {node2_id: similarity_score}}`.

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

        # Leaves have no common neighbors and returns dict with node IDs
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        print(sim[a_id][b_id])  # 0.0
        ```

        Notes
        -----
        Widely used in:

        - document similarity (bag-of-words)
        - user similarity (common items/friends)
        - biological networks (protein interactions)

        %family Similarity
        """
        return self._graph.get_jaccard_similarity()

    def get_cosine_similarity(self) -> dict[int, dict[int, float]]:
        """
        Get cosine similarity between all node pairs.

        Cosine similarity treats neighbors as vector components:

        `Cosine(A, B) = |neighbors(A) ∩ neighbors(B)| / √(|neighbors(A)| × |neighbors(B)|)`

        Range: `[0, 1]` where `0` means no common neighbors and `1` means identical neighborhoods.
        Geometric interpretation: cosine of angle between neighbor vectors. Self-similarity is
        always `1.0`.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores. Format:
            `{node1_id: {node2_id: similarity_score}}`.

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

        # Each node shares 1 neighbor with the others and returns dict with node IDs
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        print(sim[a_id][b_id])  # 0.5 (1 common / sqrt(2 * 2))
        ```

        Notes
        -----
        Advantages:

        - scales well with high-degree nodes
        - balances degree differences
        - standard in text mining and recommendation systems

        %family Similarity
        """
        return self._graph.get_cosine_similarity()

    def get_overlap_similarity(self) -> dict[int, dict[int, float]]:
        """
        Get overlap coefficient between all node pairs.

        The overlap coefficient emphasizes subset relationships:

        `Overlap(A, B) = |neighbors(A) ∩ neighbors(B)| / min(|neighbors(A)|, |neighbors(B)|)`

        Range: `[0, 1]` where `0` means no common neighbors and `1` means one neighborhood is a
        subset of the other. Asymmetric measure useful for hierarchical patterns. Self-similarity is
        always `1.0`.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary mapping node IDs to dictionaries of similarity scores. Format:
            `{node1_id: {node2_id: similarity_score}}`.

        Examples
        --------
        Detect hierarchical relationships in a hub-spoke graph:

        ```python
        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="spoke")
            .add_node(label="extended")
            .add_node(label="extra")
            .add_edge(source="hub", target="spoke")
            .add_edge(source="hub", target="extended")
            .add_edge(source="spoke", target="extra")
        )

        sim = graph.get_overlap_similarity()

        # Hub's neighborhood is subset of spoke's and returns dict with node IDs
        hub_id = graph.get_id_from_label(label="hub")
        spoke_id = graph.get_id_from_label(label="spoke")

        print(sim[hub_id][spoke_id])  # 1.0
        ```

        Notes
        -----
        Use cases:

        - detecting specialization/generalization
        - finding hubs whose neighbors are subsets of others
        - hierarchical community structure

        %family Similarity
        """
        return self._graph.get_overlap_similarity()

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # STRUCTURAL PATTERNS
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Cycle Detection
    # ==============================================================================

    def has_cycles(self) -> bool:
        """
        Check if the graph contains any cycles.

        Uses depth-first search to detect cycles. For directed graphs, looks for back edges. For
        undirected graphs, checks for cycles excluding trivial two-node back-and-forth paths.

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

        %family Cycle Detection
        """
        return self._graph.has_cycles()

    def get_cycles(self, min_length: int = 2) -> list[list[int]]:
        """
        Get all elementary cycles in the graph using Johnson's algorithm.

        Implements Donald B. Johnson's 1975 algorithm for finding all elementary (simple) cycles in
        a directed graph. Each cycle is found exactly once and contains no repeated nodes except the
        first/last.

        This method only works on directed graphs. For undirected graphs, convert to directed first
        or consider that every edge forms a 2-cycle.

        Parameters
        ----------
        min_length
            Minimum cycle length to report. Cycles shorter than this are ignored. The default is `2`
            (includes self-loops if present).

        Returns
        -------
        list[list[int]]
            A list of cycles, where each cycle is a list of node IDs forming the cycle. The start
            node is not repeated at the end.

        Raises
        ------
        ValueError
            If the graph is undirected (cycle enumeration requires directed graphs).

        Examples
        --------
        Find all cycles in a directed graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Creates cycle: 1 -> 2 -> 3 -> 1
            .add_edge(source="B", target="A")  # Creates cycle: 1 -> 2 -> 1
        )

        cycles = graph.get_cycles()
        print(cycles)  # [[1, 2], [1, 2, 3]] (node IDs)
        ```

        Find only longer cycles:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        cycles = graph.get_cycles(min_length=3)
        print(cycles)  # [[1, 2, 3]]
        ```

        Notes
        -----
        - time complexity: O(V + E)(C + 1) where C is the number of cycles
        - space complexity: O(V + E)
        - the algorithm uses backtracking with a blocking mechanism to avoid redundant search
        - for large dense graphs, the number of cycles can be exponential

        %family Cycle Detection
        """
        if not self.is_directed():
            raise ValueError("get_cycles() requires a directed graph")

        # Get cycles as node IDs from Rust
        return self._graph.get_cycles(float(min_length))

    def simple_cycles(self, length_bound: int | None = None) -> list[list[int]]:
        """
        Find all simple cycles in the graph.

        A simple cycle is a closed path where no node appears more than once (except that the first
        and last node are the same). This method works for both directed and undirected graphs.

        For directed graphs, uses Johnson's algorithm which is efficient and finds each cycle
        exactly once.

        For undirected graphs, uses a backtracking DFS approach that finds all elementary cycles by
        starting from each node and exploring all paths that return to the starting node.

        Parameters
        ----------
        length_bound
            Optional maximum cycle length. If specified, only cycles with at most this many nodes
            are returned. If `None` (default), all cycles are returned regardless of length.

        Returns
        -------
        list[list[int]]
            A list of cycles, where each cycle is a list of node IDs forming the cycle. The start
            node is not repeated at the end.

        Examples
        --------
        Find all cycles in a directed graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Creates cycle A -> B -> C -> A
            .add_edge(source="B", target="A")  # Creates cycle A -> B -> A
        )

        cycles = graph.simple_cycles()
        print(len(cycles))  # 2 cycles found
        ```

        Find cycles in an undirected graph (triangle):

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        cycles = graph.simple_cycles()
        print(len(cycles))  # 1 cycle (the triangle)
        ```

        Limit cycle length:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")  # 2-cycle
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="A")  # 4-cycle
        )

        short_cycles = graph.simple_cycles(length_bound=3)
        # Only returns cycles of length <= 3
        ```

        Notes
        -----
        - for directed graphs: `O(V + E)(C + 1)` where `C` is the number of cycles
        - for undirected graphs: Can be exponential in worst case
        - unlike `get_cycles()`, this method works for both directed and undirected graphs
        - cycles are represented as sequences of node IDs without repeating the start node
        - for undirected graphs, each cycle is returned only once (canonicalized)

        %family Cycle Detection
        """
        from edgecase._core import simple_cycles as rust_simple_cycles

        return rust_simple_cycles(self._graph, length_bound)

    def cycle_basis(self, root: int | None = None) -> list[list[int]]:
        """
        Find a fundamental cycle basis of the graph.

        A cycle basis is a minimal set of cycles such that any cycle in the graph can be expressed
        as a symmetric difference (XOR) of cycles in the basis. For a connected graph with `V`
        vertices and `E` edges, the cycle basis has exactly `E - V + 1` cycles (the cyclomatic
        complexity).

        The algorithm builds a spanning tree using BFS from a root node, then for each non-tree edge
        (back edge), finds the fundamental cycle it creates with the tree.

        Parameters
        ----------
        root
            Optional root node ID to start BFS from. If `None`, uses an arbitrary node. Different
            root nodes may produce different (but equally valid) cycle bases.

        Returns
        -------
        list[list[int]]
            A list of cycles forming the cycle basis. Each cycle is a list of node IDs. The cycles
            are fundamental cycles (each contains exactly one back edge).

        Raises
        ------
        ValueError
            If called on a directed graph (cycle basis only applies to undirected graphs).
        ValueError
            If the specified root node does not exist in the graph.

        Examples
        --------
        Find the cycle basis of a simple graph:

        ```python
        # Square with diagonal
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
            .add_edge("A", "C")  # diagonal
        )

        basis = graph.cycle_basis()
        print(len(basis))  # 2 fundamental cycles
        # The two triangles formed by the diagonal
        ```

        Specify a root node:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        node_ids = graph.get_node_ids()
        basis = graph.cycle_basis(root=node_ids[0])
        print(len(basis))  # 1 (the triangle)
        ```

        Notes
        -----
        - Time complexity: `O(V + E)` for building spanning tree and finding cycles.
        - The cycle basis is not unique, different spanning trees yield different bases.
        - Works on disconnected graphs (finds cycles in each component).
        - For weighted graphs needing minimum total weight, use `minimum_cycle_basis()`.

        %family Cycle Detection
        """
        from edgecase._core import cycle_basis as rust_cycle_basis

        return rust_cycle_basis(self._graph, root)

    def minimum_cycle_basis(self, weight: str | None = None) -> list[list[int]]:
        """
        Find a minimum weight cycle basis for the graph.

        A minimum cycle basis is a cycle basis where the sum of weights of all cycles is minimized.
        For unweighted graphs, this minimizes the sum of cycle lengths.

        This implements a variant of Horton's algorithm:

        1. for each edge `(u,v)`, find the shortest path from `u` to `v` not using `(u,v)`
        2. this gives a candidate cycle for each edge
        3. sort candidates by weight and greedily select linearly independent cycles

        Parameters
        ----------
        weight
            Edge attribute name to use as weight. If `None`, uses the default edge weight (or `1.0`
            if edges don't have weights). The algorithm finds the basis that minimizes the sum of
            cycle weights.

        Returns
        -------
        list[list[int]]
            A list of cycles forming a minimum weight cycle basis. Each cycle is a list of node IDs.

        Raises
        ------
        ValueError
            If called on a directed graph.

        Examples
        --------
        Find minimum cycle basis of an unweighted graph:

        ```python
        # Two triangles sharing an edge
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            .add_edge("B", "D")
            .add_edge("D", "C")
        )

        basis = graph.minimum_cycle_basis()
        print(len(basis))  # 2 cycles (the two triangles)
        # Sum of cycle lengths is minimized (3 + 3 = 6)
        ```

        Find minimum cycle basis with edge weights:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
            .add_edge("C", "A", weight=3.0)
        )

        basis = graph.minimum_cycle_basis(weight="weight")
        # Returns the triangle with total weight 6.0
        ```

        Notes
        -----
        - time complexity: O(E * V^2) for Horton's algorithm variant.
        - the minimum cycle basis is unique in terms of total weight (though there may be multiple
        bases with the same minimum weight).
        - works on disconnected graphs (finds minimum basis for each component).
        - for simple (unweighted) cycle basis, use `cycle_basis()` which is faster.

        %family Cycle Detection
        """
        from edgecase._core import minimum_cycle_basis as rust_minimum_cycle_basis

        return rust_minimum_cycle_basis(self._graph, weight)

    # ==============================================================================
    # Clique Operations
    # ==============================================================================

    def find_cliques(self) -> list[list[int]]:
        """
        Find all maximal cliques in the graph.

        A clique is a subset of nodes where every pair of nodes is connected by an edge. A maximal
        clique is a clique that cannot be extended by adding another node. Uses the Bron-Kerbosch
        algorithm for efficient enumeration.

        Note that this finds *maximal* cliques (cannot be extended), not necessarily *maximum*
        cliques (largest possible). Use `max_clique()` to find a maximum clique.

        Returns
        -------
        list[list[int]]
            List of cliques, where each clique is a sorted list of node IDs.

        Raises
        ------
        ValueError
            If called on a directed graph.

        Notes
        -----
        - time complexity: `O(3^(n/3))` (worst case), but typically much faster in practice
        - only works on undirected graphs
        - returns cliques in arbitrary order
        - each node ID list within a clique is sorted

        Examples
        --------
        Find all maximal cliques in a triangle:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        cliques = graph.find_cliques()
        print(len(cliques))  # 1 (the triangle itself)
        print(len(cliques[0]))  # 3 (all three nodes)
        ```

        Multiple cliques in a graph:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        cliques = graph.find_cliques()
        # Returns pairs of connected nodes: [[A,B], [B,C], [C,D]]
        print(len(cliques))  # 3 maximal cliques
        ```

        A complete graph has one clique:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(n=5)

        cliques = graph.find_cliques()
        print(len(cliques))  # 1 (entire graph is a clique)
        print(len(cliques[0]))  # 5 (all nodes)
        ```

        %family Clique Operations
        %seealso max_clique, clique_number, maximum_independent_set
        """
        return self._graph.find_cliques()

    def max_clique(self) -> list[int]:
        """
        Find a maximum clique in the graph.

        A maximum clique is a largest possible clique in the graph. If multiple maximum cliques
        exist with the same size, returns one of them (arbitrary choice). Uses the Bron-Kerbosch
        algorithm to enumerate all maximal cliques and selects the largest.

        Returns
        -------
        list[int]
            Sorted list of node IDs forming a maximum clique. Empty list if graph is empty.

        Raises
        ------
        ValueError
            If called on a directed graph.

        Notes
        -----
        - time complexity: `O(3^(n/3))` in the worst case
        - only works on undirected graphs
        - finding the maximum clique is NP-hard
        - returns one maximum clique if multiple exist

        Examples
        --------
        Maximum clique in a triangle:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        max_clique = graph.max_clique()

        print(len(max_clique))  # 3 (all nodes form a clique)
        ```

        Maximum clique in a path graph:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        max_clique = graph.max_clique()

        print(len(max_clique))  # 2 (any connected pair)
        ```

        Empty graph:

        ```python
        graph = Graph(directed=False)
        max_clique = graph.max_clique()
        print(max_clique)  # []
        ```

        Complete graph:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(n=6)

        max_clique = graph.max_clique()
        print(len(max_clique))  # 6 (all nodes)
        ```

        %family Clique Operations
        %seealso find_cliques, clique_number, maximum_independent_set
        """
        return self._graph.max_clique()

    def clique_number(self) -> int:
        """
        Get the clique number (size of the maximum clique).

        The clique number `ω(G)` is the size of the largest clique in the graph. This is a
        fundamental graph invariant used in graph theory and combinatorial optimization.

        Returns
        -------
        int
            Size of the maximum clique. Returns `0` for empty graphs.

        Raises
        ------
        ValueError
            If called on a directed graph.

        Notes
        -----
        - time complexity: `O(3^(n/3))` in the worst case
        - only works on undirected graphs
        - determining the clique number is NP-hard
        - for a complete graph `K_n`, the clique number is `n`
        - for a bipartite graph, the clique number is at most `2`

        Examples
        --------
        Clique number of a triangle:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        print(graph.clique_number())  # 3
        ```

        Clique number of a path:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        print(graph.clique_number())  # 2 (any edge)
        ```

        Empty graph has clique number 0:

        ```python
        graph = Graph(directed=False)
        print(graph.clique_number())  # 0
        ```

        Complete graph K_n has clique number n:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(n=7)
        print(graph.clique_number())  # 7
        ```

        Bipartite graphs have clique number at most 2:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "D")
        )

        print(graph.clique_number())  # 2 (bipartite K_2,2)
        ```

        %family Clique Operations
        %seealso find_cliques, max_clique, chromatic_number_upper_bound
        """
        return self._graph.clique_number()

    # ==============================================================================
    # Triangle and Transitivity Analysis
    # ==============================================================================

    def count_triangles(self) -> dict[int, int]:
        """
        Count triangles containing each node.

        Returns the number of triangles that include each node in the graph. For undirected graphs,
        counts each triangle once per node. For directed graphs, counts directed triangles
        (closed triplets).

        A triangle is a set of three nodes where all three pairs are connected. This is the smallest
        non-trivial cycle and a fundamental measure of local clustering in networks.

        Time complexity: `O(m * d_max)` where `m` is edges and `d_max` is maximum degree.

        Returns
        -------
        dict[int, int]
            Dictionary mapping node IDs to triangle counts.

        Examples
        --------
        Count triangles in a simple triangle:

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

        counts = graph.count_triangles()
        # Returns node IDs: {node_id_a: 1, node_id_b: 1, node_id_c: 1}

        # Convert to labels if needed
        counts_by_label = {graph.get_label_from_id(nid): count for nid, count in counts.items()}
        print(counts_by_label)  # {"A": 1, "B": 1, "C": 1}
        ```

        Count triangles in a more complex graph:

        ```python
        # Graph with 2 triangles sharing an edge
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Triangle 1
            .add_edge(source="B", target="D")
            .add_edge(source="D", target="C")  # Triangle 2 with B-C
        )

        counts = graph.count_triangles()
        # Returns node IDs: B and C each participate in 2 triangles

        # Convert to labels for display
        for node_id, count in counts.items():
            label = graph.get_label_from_id(node_id)
            print(f"{label}: {count} triangles")
        ```

        %family Triangle and Transitivity
        """
        return self._graph.count_triangles()

    def list_triangles(self) -> list[tuple[int, int, int]]:
        """
        List all triangles in the graph.

        Returns a list of all triangles (3-cycles) in the graph. Each triangle is represented as a
        sorted tuple of three node IDs. For undirected graphs, each triangle appears once. For
        directed graphs, lists directed triangles.

        Time complexity: `O(m * d_max)` where `m` is edges and `d_max` is maximum degree.

        Returns
        -------
        list[tuple[int, int, int]]
            List of triangles, where each triangle is a tuple of three node IDs.

        Examples
        --------
        List triangles in a graph:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # One triangle
            .add_edge(source="A", target="D")
            .add_edge(source="D", target="B")  # Another triangle
        )

        triangles = graph.list_triangles()
        # Returns node IDs: [(a_id, b_id, c_id), (a_id, b_id, d_id)]

        # Convert to labels if needed
        triangles_labels = [
            tuple(graph.get_label_from_id(nid) for nid in tri)
            for tri in triangles
        ]
        print(triangles_labels)  # [("A", "B", "C"), ("A", "B", "D")]
        ```

        For larger graphs, this can identify all triadic closures:

        ```python
        # Social network
        graph = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
            .add_node(label="David")
            .add_edge(source="Alice", target="Bob")
            .add_edge(source="Bob", target="Charlie")
            .add_edge(source="Charlie", target="Alice")
            .add_edge(source="Alice", target="David")
            .add_edge(source="David", target="Bob")
        )

        # Returns node IDs
        for tri in graph.list_triangles():
            # Convert to labels for display
            labels = tuple(graph.get_label_from_id(nid) for nid in tri)
            print(f"Triangle: {labels}")
        ```

        %family Triangle and Transitivity
        """
        return self._graph.list_triangles()

    def get_transitivity(self) -> Optional[float]:
        """
        Compute global transitivity (clustering coefficient).

        Transitivity measures the probability that two neighbors of a node are also connected. It's
        computed as:

        `transitivity = 3 * (number of triangles) / (number of connected triples)`

        A connected triple is a path of length 2 (three nodes where at least two edges connect
        them). Higher transitivity indicates more clustering.

        Returns `None` if there are no connected triples in the graph.

        Time complexity: `O(m * d_max)` where `m` is edges and `d_max` is maximum degree.


        %family Triangle and Transitivity

        Returns
        -------
        Optional[float]
            Global clustering coefficient between `0.0` and `1.0`, or `None` if undefined.

        Examples
        --------
        Perfect triangle has transitivity 1.0:

        ```python
        from edgecase import Graph

        triangle = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        print(triangle.get_transitivity())  # 1.0 (all triples are closed)
        ```

        Chain graph has transitivity 0.0:

        ```python
        from edgecase import Graph

        chain = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
        )

        print(chain.get_transitivity())  # 0.0 (no triangles, only open triples)
        ```

        Real-world networks typically have values between 0.1 and 0.5:

        ```python
        from edgecase import Graph

        social_network = (
            Graph()
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
            .add_node(label="David")
            .add_edge(source="Alice", target="Bob")
            .add_edge(source="Bob", target="Charlie")
            .add_edge(source="Charlie", target="David")
            .add_edge(source="David", target="Alice")
            .add_edge(source="Alice", target="Charlie")  # Creates some triangles
        )

        trans = social_network.get_transitivity()
        print(f"Clustering: {trans:.3f}")
        ```

        %family Triangle and Transitivity
        """
        return self._graph.transitivity()

    # ==============================================================================
    # Clustering
    # ==============================================================================

    def local_clustering_coefficient(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, Optional[float]], "Graph"]:
        """
        Compute local clustering coefficient for each node.

        The local clustering coefficient measures how close a node's neighbors are to being a
        complete subgraph (clique). For a node with k neighbors, it's the fraction of possible edges
        between neighbors that actually exist.

        For undirected graphs: `C(v) = 2 * triangles(v) / (k * (k-1))`

        For directed graphs: `C(v) = triangles(v) / (k_out * (k_out-1))`

        Nodes with fewer than 2 neighbors return `None` (clustering undefined).

        Time complexity: `O(m * d_max)` where `m` is edges and `d_max` is maximum degree.

        Parameters
        ----------
        attr_name
            Name of the node attribute to store clustering coefficient values. If `None`, uses
            `local_clustering_coefficient` as the attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to clustering coefficients. If `False`,
            stores values as node attributes and returns the Graph for chaining.

        Returns
        -------
        Union[dict[int, Optional[float]], Graph]
            If `return_dict=` is `True` we receive a dictionary mapping node IDs to clustering
            coefficients (`0.0` to `1.0`), or `None` for nodes with degree < `2`. If `return_dict=`
            is `False`, this returns self for method chaining.

        Examples
        --------
        Measure clustering in a network:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Triangle: A,B,C
            .add_edge(source="A", target="D")
            .add_edge(source="A", target="E")  # A has additional neighbors
        )

        clustering = graph.local_clustering_coefficient(return_dict=True)
        node_a_id = graph.get_id_from_label(label="A")
        node_b_id = graph.get_id_from_label(label="B")
        node_d_id = graph.get_id_from_label(label="D")
        if clustering[node_a_id] is not None:
            print(f"Node A: {clustering[node_a_id]:.3f}")  # Lower (not all neighbors connected)
        if clustering[node_b_id] is not None:
            print(f"Node B: {clustering[node_b_id]:.3f}")  # 1.0 (all neighbors connected)
        print(f"Node D: {clustering[node_d_id]}")      # None (degree 1)

        # Convert to labels if needed
        clustering_by_label = {graph.get_label_from_id(node_id=nid): coeff for nid, coeff in clustering.items()}
        if clustering_by_label['A'] is not None:
            print(f"Node A: {clustering_by_label['A']:.3f}")
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
            .local_clustering_coefficient()  # Stores as 'local_clustering_coefficient' attribute
        )

        # Access stored values
        node = graph.get_node(graph.get_id_from_label("A"))
        print(f"Node A clustering: {node.attributes['local_clustering_coefficient']}")
        ```

        Find highly clustered nodes (hubs):

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        clustering = graph.local_clustering_coefficient(return_dict=True)
        hubs = {
            node_id: coeff
            for node_id, coeff in clustering.items()
            if coeff and coeff > 0.5
        }
        print(f"Highly clustered node IDs: {hubs}")
        ```

        %family Clustering
        """
        if attr_name is None:
            attr_name = "local_clustering_coefficient"

        clustering_values = self._graph.local_clustering_coefficient()

        if return_dict:
            return clustering_values

        # Store as node attributes
        for node_id, value in clustering_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        self._computations.append(("local_clustering_coefficient", {}, attr_name))

        return self

    def clustering_coefficient(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float], "Graph"]:
        """
        Calculate local clustering coefficient for each node.

        The clustering coefficient measures the degree to which a node's neighbors are connected to
        each other. For a node `v`:

        `C(v) = (number of edges between neighbors) / (number of possible edges between neighbors)
             = (2 × triangles through v) / (degree(v) × (degree(v) - 1))`

        Returns `0.0` for nodes with degree < `2` (cannot form triangles).

        Range: `[0, 1]` where:

        - `0` = neighbors are not connected at all
        - `1` = neighbors form a complete graph (clique)

        Parameters
        ----------
        attr_name
            Name of the node attribute to store clustering coefficient values. If `None`, uses
            `clustering_coefficient` as the attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to clustering coefficients. If `False`,
            stores values as node attributes and returns the Graph for chaining.

        Returns
        -------
        dict[int, float] or Graph
            If `return_dict=` is `True`: dictionary mapping node IDs to clustering coefficients
            (`0.0` to `1.0`). If `return_dict=` is `False`: returns self for method chaining.

        Examples
        --------
        Calculate and store clustering coefficients:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            .clustering_coefficient()  # Stores as 'clustering_coefficient' attribute
        )
        # All nodes have clustering coefficient 1.0 (complete triangle)
        ```

        Calculate clustering in a star graph:

        ```python
        graph = (
            Graph()
            .add_node(label="center")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("center", "A")
            .add_edge("center", "B")
            .add_edge("center", "C")
        )

        clustering = graph.clustering_coefficient(return_dict=True)
        center_id = graph.get_id_from_label("center")
        print(f"Center clustering: {clustering[center_id]}")  # 0.0 (no edges between spokes)
        ```

        Notes
        -----
        High clustering coefficients indicate nodes in tightly-knit groups. The clustering
        coefficient is a key measure of network transitivity and is used to identify small-world
        networks.

        %family Clustering
        """
        if attr_name is None:
            attr_name = "clustering_coefficient"

        clustering_values = self._graph.get_clustering_coefficient()

        if return_dict:
            return clustering_values

        # Store as node attributes
        for node_id, value in clustering_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        self._computations.append(("clustering_coefficient", {}, attr_name))

        return self

    def get_edge_clustering_coefficient(
        self,
        edge_ids: list[int] | None = None,
        k: int = 3,
        normalize: bool = True,
        offset: bool = True,
    ) -> dict[int, float]:
        """
        Get edge clustering coefficient for specified edges.

        The edge clustering coefficient measures how "clustered" the neighborhood around an edge is,
        based on k-cycles (cycles of length k) the edge participates in.

        For `k=3` (triangles):

        - counts triangles containing the edge
        - maximum possible: `min(d_i - 1, d_j - 1)` where `d_i`, `d_j` are endpoint degrees
        - value of `1.0` means edge is in a complete subgraph

        For `k=4` (4-cycles):

        - counts 4-cycles containing the edge
        - maximum possible: `(d_i - 1)(d_j - 1)`
        - measures "squareness" of edge neighborhood

        Parameters
        ----------
        edge_ids
            List of edge IDs (integers) to compute coefficient for. If `None`, computes for all
            edges.
        k
            Cycle size to use. Must be `3` (triangles) or `4` (4-cycles).
        normalize
            Whether to normalize by maximum possible cycles given endpoint degrees. If `False`,
            returns raw cycle count.
        offset
            Whether to add `1` to cycle count (original Radicchi et al. definition). When
            `offset=True`: `C(e) = (z + 1) / s`. When `offset=False`: `C(e) = z / s`.

        Returns
        -------
        dict[int, float]
            Mapping from edge ID (integer) to clustering coefficient. With `normalize=True`: range
            `[0, 1+]` (can exceed `1` with offset). With `normalize=False`: raw cycle count.

        Raises
        ------
        ValueError
            If k is not 3 or 4.

        Examples
        --------
        Triangle graph (complete graph K3):

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # All edges are in 1 triangle
        ecc = graph.get_edge_clustering_coefficient(k=3, normalize=True, offset=True)
        # Each edge: (1 triangle + 1) / min(2-1, 2-1) = 2/1 = 2.0

        # Without offset
        ecc = graph.get_edge_clustering_coefficient(k=3, normalize=True, offset=False)
        # Each edge: 1 / 1 = 1.0 (maximum clustering)
        ```

        Square graph (4-cycle):

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )

        # k=3: No triangles
        ecc3 = graph.get_edge_clustering_coefficient(k=3, offset=False)
        # All edges: 0.0 (no triangles)

        # k=4: Each edge participates in 1 four-cycle
        ecc4 = graph.get_edge_clustering_coefficient(k=4, offset=False)
        # Edge A-B: 1 / ((2-1)*(2-1)) = 1.0
        ```

        Karate club network:

        ```python
        from edgecase import karate_club
        G = karate_club()

        # Get k=3 edge clustering for all edges
        ecc = G.get_edge_clustering_coefficient(k=3, normalize=True, offset=False)

        # Find edges with highest clustering (most embedded in triangles)
        sorted_edges = sorted(ecc.items(), key=lambda x: x[1], reverse=True)
        print(f"Top 5 edges by clustering: {sorted_edges[:5]}")
        ```

        Notes
        -----
        Complexity:

        - k=3: O(|V| d log d + |E| d) where d is max degree
        - k=4: O(|V| d log d + |E| d²)

        The original Radicchi et al. definition uses `normalize=True` and `offset=True`. Setting
        `offset=False` allows the normalized metric to reach exactly `1.0` for edges in complete
        subgraphs.

        %family Clustering
        """
        # Use edge_ids directly (they are already integers)
        edge_id_list = edge_ids if edge_ids is not None else []

        # Call Rust implementation
        result = self._graph.get_edge_clustering_coefficient(edge_id_list, k, normalize, offset)

        # Return result directly (keys are already integers)
        return result

    def get_avg_clustering_coefficient(self) -> float:
        """
        Get average clustering coefficient for the graph.

        Returns the mean of all node clustering coefficients. This is a global measure of how
        clustered the graph is overall.

        Returns
        -------
        float
            Average clustering coefficient (`0.0` to `1.0`).

        Examples
        --------
        Compare clustering in different graph types:

        ```python
        # Triangle graph: highly clustered
        triangle = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        print(f"Triangle: {triangle.get_avg_clustering_coefficient():.2f}")
        # Triangle: 1.00

        # Star graph: not clustered
        star = (
            Graph()
            .add_node(label="center")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("center", "A")
            .add_edge("center", "B")
            .add_edge("center", "C")
        )
        print(f"Star: {star.get_avg_clustering_coefficient():.2f}")
        # Star: 0.00
        ```

        Notes
        -----
        Values close to `1.0` indicate a highly clustered network (many triangles). Values close to
        `0.0` indicate a tree-like or sparse network structure.

        Small-world networks typically have high clustering coefficients combined with low average
        path lengths.

        %family Clustering
        """
        return self._graph.get_avg_clustering_coefficient()

    # ==============================================================================
    # Community Detection
    # ==============================================================================

    def cmty_fast_greedy(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Detect communities using Fast Greedy (Clauset-Newman-Moore) algorithm.

        This is a greedy modularity optimization algorithm that merges communities to maximize
        modularity. It's efficient for medium to large graphs and provides good quality communities.
        Only works with undirected graphs.

        Algorithm:

        1. start with each node in its own community
        2. build a heap of merge candidates ordered by modularity gain (ΔQ)
        3. repeatedly merge the pair with highest ΔQ until no improvement
        4. return the partition with maximum modularity

        Parameters
        ----------
        attr_name
            column name to store community assignments in the node DataFrame. If `None` (default),
            automatically uses `cmty_fast_greedy`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to community IDs (0-indexed).
            If `return_dict=` is `False` (default): returns self for method chaining.

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

        communities = graph.cmty_fast_greedy(return_dict=True)
        print(len(set(communities.values())))  # 2 communities
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Clique 1
            .add_edge(source="D", target="E")  # Clique 2
        )

        # (graph
        #     .cmty_fast_greedy(attr_name="community")
        #     .colorize_node_attrs(node_attr_from="community", palette="Set1")
        #     .show())
        ```

        Notes
        -----
        Complexity: O(n log²n) on sparse graphs, O(n + m) space.

        %family Community Detection
        %seealso cmty_louvain, cmty_edge_btwns, cmty_l_eigenvec, cmty_walktrap
        """
        communities = self._graph.get_cmty_fast_greedy()

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "cmty_fast_greedy"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, communities)

        # Track computation
        if not return_dict:
            self._computations.append(("cmty_fast_greedy", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return communities
        return self

    def cmty_edge_btwns(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Detect communities using Edge Betweenness (Girvan-Newman) algorithm.

        Iteratively removes edges with highest betweenness centrality to reveal community structure.
        Uses Brandes' algorithm for efficient betweenness calculation. More accurate than Fast
        Greedy but significantly slower (`O(m²n)` complexity).

        Best for small to medium graphs (< 5K nodes) with hierarchical community structure and clear
        "bridge" edges between communities. When accuracy is more important than speed.

        Parameters
        ----------
        attr_name
            Column name to store community assignments in the node DataFrame. If `None` (default),
            automatically uses `cmty_edge_btwns`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to community IDs (0-indexed,
            contiguous). If `return_dict=` is `False` (default): returns self for method chaining.

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

        communities = graph.cmty_edge_btwns(return_dict=True)
        # Bridge edge C-D has highest betweenness and is removed first
        # Result: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")  # Triangle 1
            .add_edge(source="C", target="D")  # Bridge edge
            .add_edge(source="D", target="E")
            .add_edge(source="E", target="F")
            .add_edge(source="F", target="D")  # Triangle 2
        )

        # (graph
        #     .cmty_edge_btwns(attr_name="community")
        #     .colorize_node_attrs(node_attr_from="community", palette="Pastel1")
        #     .show())
        ```

        Notes
        -----
        Algorithm steps:

        1. compute edge betweenness for all edges using Brandes' algorithm
        2. remove edge with highest betweenness
        3. recompute betweenness and check if modularity improved
        4. repeat until modularity stops improving
        5. return partition with maximum modularity

        Complexity: `O(m²n)` where `m` is edges, `n` is nodes. For graphs with > 10K nodes, consider
        `cmty_fast_greedy()` or `cmty_louvain()` instead.

        %family Community Detection
        %seealso cmty_louvain, cmty_fast_greedy, betweenness_centrality, bridges
        """
        communities = self._graph.get_cmty_edge_btwns()

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "cmty_edge_btwns"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, communities)

        # Track computation
        if not return_dict:
            self._computations.append(("cmty_edge_btwns", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return communities
        return self

    def cmty_l_eigenvec(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Detect communities using Leading Eigenvector (Newman) algorithm.

        Spectral optimization method using recursive bisection based on the leading eigenvector of
        the modularity matrix. Splits communities hierarchically when modularity improves. Best for
        medium-sized graphs (< 20K nodes) with clear 2-way divisions and hierarchical structure.

        Parameters
        ----------
        attr_name
            Column name to store community assignments in the node DataFrame. If `None` (default),
            automatically uses `cmty_l_eigenvec`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to community IDs (0-indexed,
            contiguous). If `return_dict=` is `False` (default): returns self for method chaining.

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

        communities = graph.cmty_l_eigenvec(return_dict=True)
        # Eigenvector method finds natural split at bridge
        # Result: {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1}
        ```

        Store as node attributes and chain:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="E")
            .add_edge(source="E", target="F")
            .add_edge(source="F", target="D")
        )

        # (graph
        #     .cmty_l_eigenvec(attr_name="community")
        #     .colorize_node_attrs(node_attr_from="community", palette="Spectral")
        #     .show())
        ```

        Notes
        -----
        Algorithm steps:

        1. start with all nodes in one community
        2. compute modularity matrix: B_ij = A_ij - (k_i × k_j) / 2m
        3. find leading eigenvector using power iteration
        4. split by sign: positive -> community 1, negative -> community 2
        5. accept split only if modularity improves
        6. recursively split each community

        Complexity: `O(n²)` per split (recursive). For very large graphs (> 50K nodes), consider
        using `get_cmty_fast_greedy()` or `cmty_louvain()` instead.

        The algorithm only splits when modularity improves, so it may return fewer communities than
        other methods if the structure is not hierarchically organized.

        %family Community Detection
        %seealso cmty_louvain, cmty_fast_greedy, eigen_centrality
        """
        communities = self._graph.get_cmty_l_eigenvec()

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "cmty_l_eigenvec"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, communities)

        # Track computation
        if not return_dict:
            self._computations.append(("cmty_l_eigenvec", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return communities
        return self

    def cmty_louvain(
        self, resolution: float = 1.0, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Detect communities using Louvain multi-level optimization algorithm.

        The Louvain method is a hierarchical algorithm that maximizes modularity through iterative
        optimization. It's the most widely-used community detection algorithm in practice due to its
        speed (`O(n log n)`) and quality on large networks.

        The method produces a dendrogram hierarchy but returns the partition with maximum
        modularity. Works only with undirected graphs.

        Parameters
        ----------
        resolution
            Resolution parameter controlling community size. Higher values favor smaller
            communities. Standard modularity uses resolution=1.0.
        attr_name
            Column name to store community assignments in the node DataFrame. If `None` (default),
            automatically uses `cmty_louvain`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to community IDs (0-indexed,
            contiguous). If `return_dict=` is `False` (default): returns self for method chaining.

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

        communities = graph.cmty_louvain(return_dict=True)
        # Nodes in same triangle have same community ID
        node_a = graph.get_id_from_label("A")
        node_b = graph.get_id_from_label("B")
        node_c = graph.get_id_from_label("C")
        node_d = graph.get_id_from_label("D")
        assert communities[node_a] == communities[node_b] == communities[node_c]
        assert communities[node_d] == communities[graph.get_id_from_label("E")]

        # Control community granularity with resolution
        fine = graph.cmty_louvain(resolution=1.5, return_dict=True)  # Smaller communities
        coarse = graph.cmty_louvain(resolution=0.5, return_dict=True)  # Larger communities
        ```

        Store as node attributes and chain:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="E")
            .add_edge(source="E", target="F")
            .add_edge(source="F", target="D")
        )

        # (graph
        #     .cmty_louvain(attr_name="community")
        #     .colorize_node_attrs(node_attr_from="community", palette="viridis")
        #     .show())
        ```

        Notes
        -----
        Algorithm steps:

        1. phase 1 (Local optimization): each node tries joining neighboring communities to maximize
        modularity gain `ΔQ`
        2. phase 2 (Aggregation): communities are coarsened into super-nodes, creating a new network
        where each community becomes a single node
        3. repeat until modularity converges (no further improvement)

        The resolution parameter tunes community size:

        - `resolution < 1.0`: Larger communities
        - `resolution = 1.0`: Standard modularity optimization
        - `resolution > 1.0`: Smaller communities

        The algorithm is deterministic but may produce different (equivalent) partitions depending
        on node ordering. Handles disconnected components naturally. Self-loops and multi-edges are
        currently not supported.

        %family Community Detection
        %seealso cmty_fast_greedy, cmty_walktrap, cmty_edge_btwns, cmty_l_eigenvec
        """
        communities = self._graph.get_cmty_louvain(resolution)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "cmty_louvain"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, communities)

        # Track computation
        if not return_dict:
            params = {"resolution": resolution} if resolution != 1.0 else {}
            self._computations.append(("cmty_louvain", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return communities
        return self

    def cmty_walktrap(
        self, steps: int = 4, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Detect communities using Walktrap random walk algorithm.

        The Walktrap algorithm exploits the property that random walks tend to stay trapped within
        densely connected communities. It computes distances between nodes based on random walk
        probabilities, then uses hierarchical clustering to identify community structure.

        Works only with undirected graphs. Time complexity is `O(n²log n)` for typical networks with
        small step parameter.

        Parameters
        ----------
        steps
            Number of random walk steps. Higher values capture longer-range
            structural patterns. Typical values: 2-8.

            - steps=2: Local structure only
            - steps=4: Good balance (default, recommended)
            - steps=6+: Longer-range patterns, more computation
        attr_name
            Column name to store community assignments in the node DataFrame. If `None` (default),
            automatically uses `cmty_walktrap`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to community IDs (0-indexed,
            contiguous). If `return_dict=` is `False` (default): returns self for method chaining.

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

        communities = graph.cmty_walktrap(return_dict=True)
        node_a = graph.get_id_from_label("A")
        node_b = graph.get_id_from_label("B")
        node_c = graph.get_id_from_label("C")
        node_d = graph.get_id_from_label("D")
        assert communities[node_a] == communities[node_b] == communities[node_c]
        assert communities[node_d] == communities[graph.get_id_from_label("E")]

        # Try different step lengths for different scales
        local_cmty = graph.cmty_walktrap(return_dict=True, steps=2)   # More local communities
        global_cmty = graph.cmty_walktrap(return_dict=True, steps=6)  # More global communities
        ```

        Store as node attributes and chain:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="E")
            .add_edge(source="E", target="F")
            .add_edge(source="F", target="D")
        )

        # (graph
        #     .cmty_walktrap(attr_name="community")
        #     .colorize_node_attrs(node_attr_from="community", palette="tab10")
        #     .show())
        ```

        Notes
        -----
        Algorithm steps:

        1. compute transition probability matrix `P` (`P[i][j] = 1/deg(i)` if edge exists)
        2. compute t-step transition probabilities `P^t` by matrix exponentiation
        3. calculate random walk distance between each pair of nodes:
           `d(i,j) = sqrt(Σ_k [(P^t_ik - P^t_jk)^2 / deg(k)])`
        4. apply Ward's hierarchical clustering on the distance matrix
        5. cut the dendrogram at the level that maximizes modularity

        The step parameter controls the scale of community detection:

        - lower steps (2-3): Detect finer, more local communities
        - higher steps (5-8): Detect coarser, more global communities

        The algorithm is deterministic (no randomness despite the name). Works well on hierarchical
        community structures and handles disconnected components automatically. Performance degrades
        on very large graphs (>10,000 nodes) due to `O(n²)` distance matrix computation. Self-loops
        and multi-edges are currently not supported.

        %family Community Detection
        %seealso cmty_louvain, cmty_fast_greedy, random_walk
        """
        communities = self._graph.get_cmty_walktrap(steps)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "cmty_walktrap"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, communities)

        # Track computation
        params = {}
        if steps != 4:
            params["steps"] = steps
        if not return_dict:
            self._computations.append(("cmty_walktrap", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return communities
        return self

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # TRANSFORMATIONS
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Graph Type Conversions
    # ==============================================================================

    def to_undirected(self) -> "Graph":
        """
        Convert the graph to an undirected graph.

        Creates a new graph with the same nodes and edges, but without edge direction. If the graph
        is already undirected, returns a new copy. For directed graphs, converts directed edges to
        undirected edges, removing duplicate pairs.

        Returns
        -------
        Graph
            A new undirected Graph with the same nodes and edges.

        Examples
        --------
        Convert a directed graph to undirected:

        ```python
        directed = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        undirected = directed.to_undirected()
        print(undirected.is_directed())  # False
        print(undirected.count_edges())  # 3 (one per unique pair)
        ```

        Use case for symmetrizing a social network:

        ```python
        # Directed: A follows B, B follows C
        followers = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
            .add_edge("Alice", "Bob")
            .add_edge("Bob", "Charlie")
        )

        # Undirected: mutual connections
        connections = followers.to_undirected()
        ```

        %family Graph Type Conversions
        """
        result = Graph.__new__(Graph)
        result._graph = self._graph.to_undirected()
        return result

    def to_directed(self) -> "Graph":
        """
        Convert the graph to a directed graph.

        Creates a new graph with the same nodes but directed edges. If the graph is already
        directed, returns a new copy. For undirected graphs, creates both directions (u->v and v->u)
        for each edge.

        Returns
        -------
        Graph
            A new directed Graph with the same nodes and bidirectional edges.

        Examples
        --------
        Convert an undirected graph to directed:

        ```python
        undirected = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        directed = undirected.to_directed()
        print(directed.is_directed())  # True
        print(directed.count_edges())  # 4 (A->B, B->A, B->C, C->B)
        ```

        Create directed graph for flow analysis:

        ```python
        # Undirected road network
        roads = Graph(directed=False)
        roads.add_node(label="A")
        roads.add_node(label="B")
        roads.add_node(label="C")
        roads.add_edge("A", "B", weight=5)
        roads.add_edge("B", "C", weight=3)

        # Directed for traffic flow (both directions)
        traffic = roads.to_directed()
        # Now can analyze: traffic from A to C vs C to A
        ```

        Use with algorithms requiring directed graphs:

        ```python
        g = Graph(directed=False)
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_node(label="Z")
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")

        # Convert for topological sort or cycle detection
        g_dir = g.to_directed()
        cycles = g_dir.get_cycles()

        %family Graph Type Conversions
        ```
        """
        result = Graph.__new__(Graph)
        result._graph = self._graph.to_directed()
        return result

    def reverse(self) -> "Graph":
        """
        Reverse the direction of all edges in the graph.

        Creates a new graph with the same nodes but all edge directions reversed (`u->v` becomes
        `v->u`). For undirected graphs, returns a new copy (no structural change). Preserves all
        node and edge attributes.

        Returns
        -------
        Graph
            A new Graph with reversed edge directions.

        Examples
        --------
        Reverse a directed graph:

        ```python
        original = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        reversed_graph = original.reverse()

        # Original: A->B->C
        # Reversed: C->B->A
        print(original.has_edge("A", "B"))    # True
        print(reversed_graph.has_edge("B", "A"))  # True
        ```

        Analyze reverse dependencies:

        ```python
        # Dependencies: B depends on A, C depends on B
        deps = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", label="provides")
            .add_edge("B", "C", label="provides")
        )

        # Reverse to find what depends on each package
        dependents = deps.reverse()
        # Now: B->A (B dependent on A), C->B (C dependent on B)
        ```

        Use for transpose graph algorithms:

        ```python
        g = Graph(directed=True)
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_node(label="Z")
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")

        # Find strongly connected components
        # (some algorithms use both G and G^T)
        g_transpose = g.reverse()
        ```

        %family Graph Type Conversions
        """
        result = Graph.__new__(Graph)
        result._graph = self._graph.reverse()
        return result

    def complement(self) -> "Graph":
        """
        Return the complement of the graph.

        Creates a new graph with the same nodes but complementary edges. An edge exists in the
        complement if and only if it does *not* exist in the original graph. Self-loops are never
        included. New edges have no attributes and default weight (`1.0` for weighted graphs).

        For directed graphs, complements directed edges independently in each direction. For
        undirected graphs, complements undirected edges.

        Returns
        -------
        Graph
            A new Graph with complementary edge set.

        Examples
        --------
        Complement of a simple graph:

        ```python
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
        )

        comp = g.complement()
        print(comp.count_edges())  # 2 (A-C and B-C)
        print(comp.has_edge("A", "C"))  # True
        print(comp.has_edge("A", "B"))  # False (was in original)
        ```

        Finding missing connections in a network:

        ```python
        # Social network with some friendships
        friends = (
            Graph(directed=False)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
            .add_node(label="Diana")
            .add_edge("Alice", "Bob")
            .add_edge("Bob", "Charlie")
        )

        # Find all non-friend pairs (potential connections)
        non_friends = friends.complement()
        # Contains: Alice-Charlie, Alice-Diana, Bob-Diana, Charlie-Diana
        ```

        Use in graph theory analysis:

        ```python
        # Check if graph is self-complementary
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        comp = g.complement()
        # comp has edges: A-C, A-D, B-C, B-D
        # Can test if `g` and `comp` are isomorphic
        ```

        Directed graph complement:

        ```python
        # Directed graph
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
        )

        comp = g.complement()
        # `comp` has directed edges: A->C, B->A, B->C, C->A, C->B
        # (all edges except A->B)

        %family Graph Type Conversions
        ```
        """
        result = Graph.__new__(Graph)
        result._graph = self._graph.complement()
        return result

    def line_graph(self, create_using_directed: bool | None = None) -> "Graph":
        """
        Return the line graph of the graph.

        The line graph `L(G)` of a graph `G` transforms edges into nodes: each edge in `G` becomes a
        node in `L(G)`, and two nodes in `L(G)` are connected if and only if their corresponding
        edges in `G` share an endpoint.

        Parameters
        ----------
        create_using_directed
            Whether the resulting line graph should be directed.

            - If `None` (default): inherits directedness from source graph
            - If `True`: creates a directed line graph
            - If `False`: creates an undirected line graph

            For directed graphs, edge `(u,v)` connects to edge `(v,w)` if they share the
            intermediate node `v` (i.e., head of first = tail of second).

        Returns
        -------
        Graph
            A new Graph representing the line graph. Each node in the line graph corresponds to an
            edge in the original graph, with attributes:

            - `source`: source node label of the original edge
            - `target`: target node label of the original edge
            - `edge_id`: ID of the original edge
            - `weight`: weight of the original edge (if weighted)

        Notes
        -----
        **Definition**: For a graph `G = (V, E)`, the line graph `L(G) = (E, E')`, where nodes in
        `L(G)` are edges of `G`, and edges in `L(G)` connect adjacent edges in `G`.

        **Adjacency Rules**:

        - **Undirected graphs**: edges are adjacent if they share any vertex; edge `{a, b}` is
        adjacent to edge `{b, c}` because they share node `b`
        - **Directed graphs**: edge `(u, v)` is adjacent to edge `(v, w)` only; the head of the
        first edge must equal the tail of the second

        **Properties**:

        - `|V(L(G))| = |E(G)|` (line graph has as many nodes as `G` has edges)
        - For undirected `G`: `|E(L(G))| = (1/2) * Σ deg(v)² - |E(G)|`
        - The line graph of a complete graph `Kₙ` is a triangular graph
        - The line graph of a cycle is another cycle of the same length

        **Use Cases**:

        - edge clustering: cluster edges by treating them as nodes
        - network flow analysis: study edge relationships
        - social networks: analyze "edge communities"
        - transportation: model transfer points between routes
        - dual representations: some graph problems are easier on L(G)

        Examples
        --------
        Line graph of a simple path:

        ```python
        from edgecase import Graph

        # Create path: A -- B -- C
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        lg = g.line_graph()
        print(lg.count_nodes())  # 2 (one per edge)
        print(lg.count_edges())  # 1 (edges A-B and B-C share B)
        ```

        Line graph of a triangle:

        ```python
        from edgecase import Graph

        # Triangle: A -- B -- C -- A
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        lg = g.line_graph()
        print(lg.count_nodes())  # 3
        print(lg.count_edges())  # 3 (also a triangle)
        ```

        Line graph of a directed graph:

        ```python
        from edgecase import Graph

        # Chain: A -> B -> C
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        lg = g.line_graph()
        print(lg.count_nodes())  # 2
        print(lg.count_edges())  # 1 (A->B connects to B->C)
        ```

        Accessing original edge information:

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_node(label="Z")
        g.add_edge("X", "Y", weight=2.0)
        g.add_edge("Y", "Z", weight=3.0)

        lg = g.line_graph()

        # Each line graph node stores info about original edge
        for node_id in lg.get_node_ids():
            node = lg.get_node(node_id)
            attrs = node.attributes
            print(f"LG node {node.label}:")
            print(f"  Original edge: {attrs['original_source']} -> {attrs['original_target']}")
            print(f"  Edge ID: {attrs['original_edge_id']}")
            print(f"  Weight: {attrs['original_weight']}")
        ```

        Line graph of a star graph:

        ```python
        from edgecase import Graph

        # Star with center C
        g = Graph(directed=False)
        g.add_node(label="C")
        for i in range(4):
            g.add_node(label=f"L{i}")
            g.add_edge("C", f"L{i}")

        lg = g.line_graph()
        print(lg.count_nodes())  # 4 (one per spoke)
        print(lg.count_edges())  # 6 (all spokes share center)
        # The line graph of a star is a complete graph!
        ```

        %family Graph Type Conversions
        """
        result = Graph.__new__(Graph)
        result._graph = rust.line_graph(self._graph, create_using_directed)
        return result

    # ==============================================================================
    # Node Contraction
    # ==============================================================================

    def contract_nodes(self, u: int | str, v: int | str, self_loops: bool = True) -> "Graph":
        """
        Contract two nodes by merging v into u.

        Creates a new graph where node `v` is removed and all its edges are redirected to node `u`.
        If there was an edge between `u` and `v`, it becomes a self-loop on `u` (unless
        `self_loops=False`). This operation does not modify the original graph.

        Parameters
        ----------
        u
            The node to keep (node ID or label). All edges from `v` will be redirected here.
        v
            The node to merge into `u` (node ID or label). This node will be removed.
        self_loops
            If `True` (default), edges between `u` and `v` become self-loops on `u`. If `False`,
            such edges are removed.

        Returns
        -------
        Graph
            A new graph with `v` contracted into `u`.

        Raises
        ------
        KeyError
            If either node does not exist.
        ValueError
            If `u` and `v` are the same node.

        Examples
        --------
        Basic node contraction:

        ```python
        from edgecase import Graph

        # Create a path: A -- B -- C -- D
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        # Contract B into A (merge B into A)
        result = g.contract_nodes("A", "B")

        print(result.count_nodes())  # 3 (A, C, D)
        print(result.has_node("B"))  # False
        # Now A is connected directly to C
        ```

        Contraction creates self-loops:

        ```python
        # Triangle: A -- B -- C -- A
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # Contract B into A
        result = g.contract_nodes("A", "B", self_loops=True)
        # Edge A-B becomes a self-loop on A
        # Edge B-C becomes A-C

        # Suppress self-loops
        result_no_loops = g.contract_nodes("A", "B", self_loops=False)
        # Edge A-B is removed
        ```

        Graph simplification:

        ```python
        # Use contraction to simplify a graph by merging similar nodes
        g = Graph(directed=False)
        g.add_path_graph(5)  # 1 -- 2 -- 3 -- 4 -- 5

        # Merge 2 and 3 into a single node
        simplified = g.contract_nodes(2, 3)
        print(simplified.count_nodes())  # 4
        ```

        Notes
        -----
        - time complexity: `O(E)` where `E` is the number of edges
        - creates a new graph; original is unchanged
        - node `u` retains its label and attributes
        - edge weights are preserved
        - for multigraphs, duplicate edges may be removed

        %family Node Contraction
        """
        from edgecase._core import contract_nodes as _contract_nodes

        u_id = self._resolve_node_id(u)
        v_id = self._resolve_node_id(v)

        result = Graph.__new__(Graph)
        result._graph = _contract_nodes(self._graph, u_id, v_id, self_loops)
        return result

    def contracted_edge(self, u: int | str, v: int | str, self_loops: bool = True) -> "Graph":
        """
        Contract an edge by merging its endpoints.

        Given an edge between `u` and `v`, creates a new graph where `v` is merged into `u`. This is
        equivalent to `contract_nodes(u, v)` but explicitly requires that an edge exists between the
        nodes.

        Parameters
        ----------
        u
            Source endpoint of the edge (node ID or label).
        v
            Target endpoint of the edge (node ID or label). This node will be removed.
        self_loops
            If `True` (default), the contracted edge becomes a self-loop on `u`. If `False`, the
            contracted edge is removed.

        Returns
        -------
        Graph
            A new graph with the edge contracted.

        Raises
        ------
        KeyError
            If either node does not exist.
        ValueError
            If no edge exists between `u` and `v`.

        Examples
        --------
        Contract an edge:

        ```python
        from edgecase import Graph

        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Contract edge A-B
        result = g.contracted_edge("A", "B")
        print(result.count_nodes())  # 2 (A and C)
        print(result.has_edge("A", "C"))  # True (was B-C)
        ```

        Edge contraction in directed graphs:

        ```python
        g = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Contract directed edge A->B
        result = g.contracted_edge("A", "B")
        # Now A->C exists (was B->C)
        ```

        Notes
        -----
        - equivalent to `contract_nodes(u, v)` when edge `(u, v)` exists
        - raises ValueError if the edge doesn't exist
        - for undirected graphs, edge direction doesn't matter

        %family Node Contraction
        """
        from edgecase._core import contracted_edge as _contracted_edge

        u_id = self._resolve_node_id(u)
        v_id = self._resolve_node_id(v)

        result = Graph.__new__(Graph)
        result._graph = _contracted_edge(self._graph, u_id, v_id, self_loops)
        return result

    def contract_nodes_multi(self, nodes: list[int | str], self_loops: bool = True) -> "Graph":
        """
        Contract multiple nodes into a single node.

        Merges all specified nodes into the first node in the list. All edges incident to any of the
        contracted nodes are redirected to the first node. Edges between contracted nodes become
        self-loops (if `self_loops=True`).

        Parameters
        ----------
        nodes
            List of node IDs or labels to contract. The first node is the target; all other nodes
            are merged into it.
        self_loops
            If `True` (default), edges between contracted nodes become self-loops. If `False`, such
            edges are removed.

        Returns
        -------
        Graph
            A new graph with all specified nodes contracted into one.

        Raises
        ------
        KeyError
            If any node does not exist.
        ValueError
            If the node list is empty.

        Examples
        --------
        Contract a community into a single node:

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        g.add_path_graph(6)  # 1 -- 2 -- 3 -- 4 -- 5 -- 6

        # Contract nodes 2, 3, 4 into node 2
        result = g.contract_nodes_multi([2, 3, 4])

        print(result.count_nodes())  # 3 (nodes 1, 2, 5/6... wait let me check)
        # Result: 1 -- 2 -- 5 -- 6 (nodes 3,4 merged into 2)
        ```

        Contract detected communities:

        ```python
        g = Graph(directed=False)
        g.add_path_graph(10)  # Build a simple graph

        # Detect communities (use return_dict=True to get dict)
        membership = g.cmty_louvain(return_dict=True)

        # Get nodes in community 0
        community_0 = [n for n, c in membership.items() if c == 0]

        # Contract that community
        if len(community_0) > 1:
            simplified = g.contract_nodes_multi(community_0)
        ```

        Notes
        -----
        - time complexity: `O(k * E)` where `k` is number of nodes to contract
        - the first node in the list retains its ID and attributes
        - all other nodes are removed from the graph

        %family Node Contraction
        """
        from edgecase._core import contract_nodes_multi as _contract_nodes_multi

        resolved_nodes = [self._resolve_node_id(n) for n in nodes]

        result = Graph.__new__(Graph)
        result._graph = _contract_nodes_multi(self._graph, resolved_nodes, self_loops)
        return result

    def quotient_graph(self, partition: list[list[int | str]]) -> "Graph":
        """
        Create a quotient graph from a node partition.

        Each block (group of nodes) in the partition becomes a single node in the quotient graph.
        Edges between different blocks become edges in the quotient. Edges within the same block are
        removed.

        Parameters
        ----------
        partition
            List of node groups. Each inner list contains node IDs or labels belonging to the same
            block. Every node must appear exactly once.

        Returns
        -------
        Graph
            A new quotient graph where each partition block is a node. Nodes are labeled
            `"block_0"`, `"block_1"`, etc.

        Raises
        ------
        KeyError
            If any node in the partition does not exist.
        ValueError
            If a node appears in multiple blocks or if some nodes are missing.

        Examples
        --------
        Create quotient from community partition:

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        g.add_path_graph(6)  # 1 -- 2 -- 3 -- 4 -- 5 -- 6

        # Partition: {1,2,3} and {4,5,6}
        partition = [[1, 2, 3], [4, 5, 6]]
        quotient = g.quotient_graph(partition)

        print(quotient.count_nodes())  # 2
        print(quotient.count_edges())  # 1 (from edge 3-4)
        ```

        Quotient from community detection:

        ```python
        g = Graph(directed=False)
        g.add_path_graph(20)  # Build a simple graph

        # Detect communities (use return_dict=True to get dict)
        membership = g.cmty_louvain(return_dict=True)

        # Build partition from membership
        from collections import defaultdict
        communities = defaultdict(list)
        for node, comm in membership.items():
            communities[comm].append(node)
        partition = list(communities.values())

        # Create quotient graph showing inter-community edges
        quotient = g.quotient_graph(partition)
        print(f"Communities: {quotient.count_nodes()}")
        print(f"Inter-community edges: {quotient.count_edges()}")
        ```

        Using labels:

        ```python
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )

        # Partition by labels
        quotient = g.quotient_graph([["A", "B"], ["C", "D"]])
        # Results in block_0 -- block_1
        ```

        Notes
        -----
        - every node must appear in exactly one block
        - intra-block edges are discarded (no self-loops in quotient)
        - edge weights from the first inter-block edge are preserved
        - quotient nodes are numbered `1, 2, 3, ...` with labels `"block_0"`, etc.

        %family Node Contraction
        """
        from edgecase._core import quotient_graph as _quotient_graph

        resolved_partition = [[self._resolve_node_id(n) for n in block] for block in partition]

        result = Graph.__new__(Graph)
        result._graph = _quotient_graph(self._graph, resolved_partition)
        return result

    # ==============================================================================
    # Graph Algebra Operations
    # ==============================================================================

    def union(self, other: "Graph") -> "Graph":
        """
        Compute the union of this graph with another graph.

        Returns a new graph containing all nodes and edges from both graphs. Nodes with the same ID
        are merged (this graph's attributes take precedence). Duplicate edges are not added.

        Parameters
        ----------
        other
            The graph to union with.

        Returns
        -------
        Graph
            A new graph containing all nodes and edges from both graphs.

        Raises
        ------
        ValueError
            If the graphs have different directedness (one directed, one undirected).

        Examples
        --------
        Merge two graphs:

        ```python
        from edgecase import Graph

        # Create two graphs
        g1 = Graph(directed=False)
        g1.add_node(label="A")
        g1.add_node(label="B")
        g1.add_edge("A", "B")

        g2 = Graph(directed=False)
        g2.add_node(label="B")
        g2.add_node(label="C")
        g2.add_edge("B", "C")

        # Union combines both
        result = g1.union(g2)
        print(f"Nodes: {result.count_nodes()}")  # 3 (A, B, C)
        print(f"Edges: {result.count_edges()}")  # 2 (A-B, B-C)
        ```

        Combine network snapshots:

        ```python
        # Network at time t1
        t1 = Graph(directed=False)
        t1.add_path_graph(3)  # 1-2-3

        # Network at time t2 (new connection)
        t2 = Graph(directed=False)
        for i in range(1, 4):
            t2.add_node()
        t2.add_edge(1, 3)  # Direct connection

        # Combined network
        combined = t1.union(t2)
        print(f"Edges: {combined.count_edges()}")  # 3
        ```

        Notes
        -----
        - Both graphs must have the same directedness
        - Node IDs are preserved; overlapping IDs are merged
        - This graph's node labels/attributes take precedence
        - Weighted if either graph is weighted

        %family Graph Algebra
        """
        from edgecase._core import graph_union

        result = Graph.__new__(Graph)
        result._graph = graph_union(self._graph, other._graph)
        return result

    def intersection(self, other: "Graph") -> "Graph":
        """
        Compute the intersection of this graph with another graph.

        Returns a new graph containing only edges that exist in both graphs. Nodes are included only
        if they are endpoints of common edges.

        Parameters
        ----------
        other
            The graph to intersect with.

        Returns
        -------
        Graph
            A new graph containing edges present in both graphs.

        Raises
        ------
        ValueError
            If the graphs have different directedness.

        Examples
        --------
        Find common structure:

        ```python
        from edgecase import Graph

        # Triangle
        g1 = Graph(directed=False)
        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        # Square with diagonal
        g2 = Graph(directed=False)
        for i in range(1, 5):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(3, 4)
        g2.add_edge(4, 1)

        # Common edges: 1-2, 2-3
        common = g1.intersection(g2)
        print(f"Common edges: {common.count_edges()}")  # 2
        ```

        This is useful for finding stable connections that exist at multiple
        time points or in multiple network versions.

        Notes
        -----
        - Both graphs must have the same directedness
        - Only edges present in both graphs are included
        - Nodes without edges are excluded
        - Uses this graph's edge weights/attributes

        %family Graph Algebra
        """
        from edgecase._core import graph_intersection

        result = Graph.__new__(Graph)
        result._graph = graph_intersection(self._graph, other._graph)
        return result

    def difference(self, other: "Graph") -> "Graph":
        """
        Compute the difference of this graph and another graph.

        Returns a new graph containing edges that are in this graph but not in the other graph.
        Nodes are included if they are endpoints of remaining edges.

        Parameters
        ----------
        other
            The graph to subtract.

        Returns
        -------
        Graph
            A new graph containing edges in this graph but not in other.

        Raises
        ------
        ValueError
            If the graphs have different directedness.

        Examples
        --------
        Find edges unique to one graph:

        ```python
        from edgecase import Graph

        # Original network
        g1 = Graph(directed=False)
        for i in range(1, 5):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 4)

        # Subset network
        g2 = Graph(directed=False)
        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)

        # Edges in g1 but not g2
        diff = g1.difference(g2)
        print(f"Unique edges: {diff.count_edges()}")  # 1 (3-4)
        ```

        This is useful for detecting network changes: `g_new.difference(g_old)`
        gives new edges, while `g_old.difference(g_new)` gives removed edges.

        Notes
        -----
        - Both graphs must have the same directedness
        - This is NOT symmetric: `g1.difference(g2) != g2.difference(g1)`
        - Nodes without edges are excluded

        %family Graph Algebra
        """
        from edgecase._core import graph_difference

        result = Graph.__new__(Graph)
        result._graph = graph_difference(self._graph, other._graph)
        return result

    def symmetric_difference(self, other: "Graph") -> "Graph":
        """
        Compute the symmetric difference of this graph and another graph.

        Returns a new graph containing edges that are in exactly one of the two graphs. Equivalent
        to `(G1 - G2) ∪ (G2 - G1)`.

        Parameters
        ----------
        other
            The graph to compare with.

        Returns
        -------
        Graph
            A new graph containing edges in exactly one of the graphs.

        Raises
        ------
        ValueError
            If the graphs have different directedness.

        Examples
        --------
        Find edges unique to either graph:

        ```python
        from edgecase import Graph

        # Triangle
        g1 = Graph(directed=False)
        for i in range(1, 4):
            g1.add_node()
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g1.add_edge(3, 1)

        # Path
        g2 = Graph(directed=False)
        for i in range(1, 4):
            g2.add_node()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)

        # XOR: edges in one but not both
        sym_diff = g1.symmetric_difference(g2)
        print(f"Unique edges: {sym_diff.count_edges()}")  # 1 (3-1)
        ```

        This is useful for comparing network versions to find all edges that
        changed (either added or removed) between snapshots.

        Notes
        -----
        - Both graphs must have the same directedness
        - Symmetric: `g1.symmetric_difference(g2) == g2.symmetric_difference(g1)`
        - Nodes without edges are excluded

        %family Graph Algebra
        """
        from edgecase._core import graph_symmetric_difference

        result = Graph.__new__(Graph)
        result._graph = graph_symmetric_difference(self._graph, other._graph)
        return result

    def compose(self, other: "Graph") -> "Graph":
        """
        Compose this graph with another graph.

        Similar to union, but the other graph's node attributes/labels take precedence over this
        graph's. Useful when you want to merge updates from another graph.

        Parameters
        ----------
        other
            The graph to compose with (its attributes take precedence).

        Returns
        -------
        Graph
            A new composed graph.

        Raises
        ------
        ValueError
            If the graphs have different directedness.

        Examples
        --------
        Merge with attribute updates:

        ```python
        from edgecase import Graph

        # Original graph
        g1 = Graph(directed=False)
        g1.add_node(label="Alice")
        g1.add_node(label="Bob")
        g1.add_edge("Alice", "Bob")

        # Updates (new label for node 2)
        g2 = Graph(directed=False)
        g2.add_node(label="Robert")  # Node 1 with new label

        # Compose: g2's labels override g1's
        result = g1.compose(g2)
        # Node 1 now has label "Robert" instead of "Alice"
        ```

        Notes
        -----
        - Similar to union but other graph's attributes take precedence
        - Useful for applying updates to a base graph

        %family Graph Algebra
        """
        from edgecase._core import graph_compose

        result = Graph.__new__(Graph)
        result._graph = graph_compose(self._graph, other._graph)
        return result

    def disjoint_union(self, other: "Graph") -> "Graph":
        """
        Compute the disjoint union of this graph with another graph.

        Combines graphs by renumbering the other graph's nodes to avoid conflicts. The resulting
        graph has all nodes and edges from both, with no overlap.

        Parameters
        ----------
        other
            The graph to combine with (its nodes will be renumbered).

        Returns
        -------
        Graph
            A new graph with both graphs combined disjointly.

        Raises
        ------
        ValueError
            If the graphs have different directedness.

        Examples
        --------
        Combine independent networks:

        ```python
        from edgecase import Graph

        # Two separate triangles
        t1 = Graph(directed=False)
        for i in range(1, 4):
            t1.add_node()
        t1.add_edge(1, 2)
        t1.add_edge(2, 3)
        t1.add_edge(3, 1)

        t2 = Graph(directed=False)
        for i in range(1, 4):
            t2.add_node()
        t2.add_edge(1, 2)
        t2.add_edge(2, 3)
        t2.add_edge(3, 1)

        # Disjoint union: two separate triangles
        combined = t1.disjoint_union(t2)
        print(f"Nodes: {combined.count_nodes()}")  # 6 (3 + 3)
        print(f"Edges: {combined.count_edges()}")  # 6 (3 + 3)
        ```

        Build multi-component graph:

        ```python
        # Create graph with multiple disconnected components
        component1 = Graph().add_star_graph(5)
        component2 = Graph().add_cycle_graph(6)
        component3 = Graph().add_path_graph(4)

        multi = component1.disjoint_union(component2).disjoint_union(component3)
        print(f"Components: 3")
        ```

        Notes
        -----
        - The other graph's nodes are renumbered starting after this graph's max ID
        - Labels are preserved but IDs change
        - Use regular `union()` if you want to merge overlapping node IDs

        %family Graph Algebra
        """
        from edgecase._core import graph_disjoint_union

        result = Graph.__new__(Graph)
        result._graph = graph_disjoint_union(self._graph, other._graph)
        return result

    # ==============================================================================
    # Whole-Graph Transformations
    # ==============================================================================

    def transform_with_spanner(
        self, stretch: float, weight: str | None = None, seed: int | None = None
    ) -> "Graph":
        """
        Transform graph to a spanner with given stretch.

        A spanner is a sparse subgraph that approximates distances in the original graph. For any
        pair of nodes `u` and `v`, the distance in the spanner is at most `stretch=` times the
        distance in the original graph:

        `distance_spanner(u, v) ≤ stretch × distance_original(u, v)`

        The resulting spanner preserves approximate distances while using fewer edges, making it
        useful for graph compression, network backbone extraction, and approximate distance queries.

        Parameters
        ----------
        stretch
            Maximum distance approximation factor. Must be ≥ `1.0`. Lower `stretch=` values give
            better approximations but more edges. Higher `stretch=` values create sparser spanners.
        weight
            Edge attribute to use as distance/weight. If `None`, all edges have weight `1`
            (unweighted graph).
        seed
            Random seed for reproducibility. The algorithm has randomized components, so different
            seeds may produce different spanners with the same stretch guarantee.

        Returns
        -------
        Graph
            A new graph with the same nodes but a subset of edges that forms a spanner with the
            specified stretch.

        Raises
        ------
        ValueError
            If stretch < `1.0` or if graph is directed.

        Notes
        -----
        **Algorithm**: This implements a greedy spanner algorithm that iteratively
        adds edges if they improve the stretch bound. The Baswana-Sen randomized
        algorithm [1]_ would provide better theoretical guarantees but is more complex.

        **Complexity**: O(mn) where m = number of edges, n = number of nodes.
        For sparse graphs this is O(n²), for dense graphs O(n³).

        **Sparsity**: The number of edges in the spanner depends on the stretch:

        - unweighted graphs: `O(n^(1+1/k))` edges where `k = ⌊(stretch+1)/2⌋`
        - weighted graphs: May have more edges but still significantly fewer than the original graph

        **Properties**:

        - preserves graph connectivity
        - guarantees stretch bound for all node pairs
        - returns undirected graph (same as input)
        - preserves node attributes
        - preserves edge weights (when `weight` parameter is provided)

        **Use Cases**:

        - network backbones: extract core network structure
        - graph compression: reduce storage while preserving distances
        - approximate queries: fast distance estimation
        - visualization: simplify dense networks for clearer plots
        - preprocessing: reduce graph size before expensive algorithms

        Examples
        --------
        Create a simple 2-spanner (distances at most 2× original):

        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))

        # Create a cycle
        for i in range(5):
            g.add_edge(source=str(i), target=str((i+1) % 5))

        spanner = g.transform_with_spanner(stretch=2.0)
        print(spanner.count_edges() < g.count_edges())  # True: Fewer edges
        ```

        Create a 3-spanner of a weighted graph:

        ```python
        from edgecase import Graph

        g = Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="B", target="C", weight=1.0)
        g.add_edge(source="A", target="C", weight=5.0)

        spanner = g.transform_with_spanner(stretch=3.0, weight="weight")
        # The direct A-C edge might be omitted since going A->B->C
        # gives distance 2, which is < 3 × (distance 5)
        print(spanner.count_edges() <= 2)  # True
        ```

        Compare sparsity for different stretch values:

        ```python
        import networkx as nx
        from edgecase import Graph

        # Create a random graph
        nx_g = nx.erdos_renyi_graph(50, 0.1, seed=23)
        g = Graph()
        for node in nx_g.nodes():
            g.add_node(label=str(node))
        for u, v in nx_g.edges():
            g.add_edge(source=str(u), target=str(v))

        original_edges = g.count_edges()
        spanner2 = g.transform_with_spanner(stretch=2.0, seed=23)
        spanner3 = g.transform_with_spanner(stretch=3.0, seed=23)

        # Higher stretch -> fewer edges
        result = spanner3.count_edges() < spanner2.count_edges() < original_edges
        print(result)  # True
        ```

        Extract network backbone for visualization:

        ```python
        g = Graph()
        # ... build large social network ...
        # Extract 2-spanner for cleaner visualization
        backbone = g.transform_with_spanner(stretch=2.0)
        # Now visualize backbone instead of full graph
        ```

        %family Graph Algebra
        """
        if stretch < 1.0:
            raise ValueError("stretch must be at least 1.0")

        if self.is_directed():
            raise ValueError("spanner only works on undirected graphs")

        # Create result graph and wrap the Rust graph
        result = Graph()
        result._graph = self._graph.spanner(stretch, weight, seed)
        return result

    def transform_to_subgraph_ws(self) -> "Graph":
        """
        Create a subgraph from current selection.

        Extracts the currently selected nodes and/or edges into a new Graph object. The behavior
        depends on what's selected:

        - nodes selected: include those nodes plus all edges between them
        - edges selected: include those edges plus their endpoint nodes
        - both selected: uses node selection (ignores edge selection)

        All node and edge attributes are preserved in the subgraph.

        Returns
        -------
        Graph
            New Graph object containing the subgraph. Has same directedness as original graph.

        Examples
        --------
        Extract a node-induced subgraph:

        ```python
        from edgecase import Graph

        graph = Graph(directed=True).add_path_graph(n=6)

        # Get node IDs for nodes at positions 2, 3, 4
        all_node_ids = list(graph.get_node_ids(limit=None))
        node_ids_to_select = [all_node_ids[2], all_node_ids[3], all_node_ids[4]]
        graph.select_nodes_by_id(node_ids=node_ids_to_select)

        subgraph = graph.transform_to_subgraph_ws()
        print(subgraph.count_nodes())  # 3
        print(subgraph.count_edges())  # 2 (edges between selected nodes)
        ```

        Extract an edge-induced subgraph:

        ```python
        graph = Graph().add_cycle_graph(5)

        edges = graph.get_edge_ids()[:2]
        graph.select_edges_by_id(edges)

        subgraph = graph.transform_to_subgraph_ws()
        # Contains only selected edges and their endpoints
        ```

        %family Graph Algebra
        """
        selected_nodes = self.get_selected_node_ids()
        selected_edges = self.get_selected_edge_ids()

        # Create new graph with same directedness
        subgraph = Graph(directed=self.is_directed())

        if len(selected_nodes) > 0:
            # Node-based subgraph: include selected nodes and edges between them
            node_set = set(selected_nodes)

            # Add nodes with their attributes
            for node_id in selected_nodes:
                node = self.get_node(node_id)
                node_label = self.get_label_from_id(node_id)
                if node and node.attributes:
                    subgraph.add_node(label=node_label, **node.attributes)
                else:
                    subgraph.add_node(label=node_label)

            # Add edges that connect selected nodes
            for edge_id in self.get_edge_ids():
                edge = self.get_edge(edge_id)
                if edge and edge.source in node_set and edge.target in node_set:
                    # Get labels for source and target (use labels, not original IDs)
                    source_label = self.get_label_from_id(edge.source)
                    target_label = self.get_label_from_id(edge.target)

                    # Prepare edge attributes
                    attrs = {}
                    if edge.weight is not None:
                        attrs["weight"] = edge.weight
                    if edge.attributes:
                        attrs.update(edge.attributes)
                    subgraph.add_edge(source_label, target_label, **attrs)

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
                node_label = self.get_label_from_id(node_id)
                if node and node.attributes:
                    subgraph.add_node(label=node_label, **node.attributes)
                else:
                    subgraph.add_node(label=node_label)

            # Add edges
            for edge in edges_to_add:
                # Get labels for source and target (use labels, not original IDs)
                source_label = self.get_label_from_id(edge.source)
                target_label = self.get_label_from_id(edge.target)

                attrs = {}
                if edge.weight is not None:
                    attrs["weight"] = edge.weight
                if edge.attributes:
                    attrs.update(edge.attributes)
                subgraph.add_edge(source_label, target_label, **attrs)

        return subgraph

    def subgraph(self, nodes: list[int]) -> "Graph":
        """
        Extract an induced subgraph containing the specified nodes.

        Returns a new graph containing only the specified nodes and all edges between them (the
        induced subgraph). This is a direct method that doesn't require prior selection, unlike
        `transform_to_subgraph_ws()`.

        Parameters
        ----------
        nodes
            List of node IDs to include in the subgraph.

        Returns
        -------
        Graph
            New graph containing only the specified nodes and edges between them. Preserves
            directedness, weightedness, and node labels. Edge weights are preserved.

        Raises
        ------
        ValueError
            If any node ID is not found in the graph.

        Examples
        --------
        Extract a subgraph from a path graph:

        ```python
        from edgecase import Graph

        # Create a path: 1 -- 2 -- 3 -- 4 -- 5
        graph = Graph().add_path_graph(5)

        # Get node IDs
        node_ids = list(graph.get_node_ids())

        # Extract subgraph of nodes at positions 1, 2, 3 (middle nodes)
        middle_nodes = node_ids[1:4]
        sub = graph.subgraph(middle_nodes)
        print(f"Nodes: {sub.count_nodes()}")  # 3
        print(f"Edges: {sub.count_edges()}")  # 2
        ```

        Extract a clique from a larger graph:

        ```python
        from edgecase import Graph

        graph = Graph().add_full_graph(5)  # K5

        # Extract K3 subgraph
        node_ids = list(graph.get_node_ids())[:3]
        k3 = graph.subgraph(node_ids)
        print(f"Edges in K3: {k3.count_edges()}")  # 3
        ```

        %family Graph Algebra
        """
        from edgecase._core import subgraph

        result = Graph.__new__(Graph)
        result._graph = subgraph(self._graph, nodes)
        return result

    def edge_subgraph(self, edges: list[int]) -> "Graph":
        """
        Extract a subgraph containing the specified edges and their endpoint nodes.

        Returns a new graph containing only the specified edges and the nodes that are endpoints of
        those edges. This is useful when you want to focus on specific relationships rather than
        specific nodes.

        Parameters
        ----------
        edges
            List of edge IDs to include in the subgraph.

        Returns
        -------
        Graph
            New graph containing only the specified edges and their endpoint nodes. Preserves
            directedness, weightedness, and node labels. Edge weights are preserved.

        Raises
        ------
        ValueError
            If any edge ID is not found in the graph.

        Examples
        --------
        Extract specific edges from a graph:

        ```python
        from edgecase import Graph

        # Create a cycle: 1 -- 2 -- 3 -- 4 -- 5 -- 1
        graph = Graph().add_cycle_graph(5)

        # Get first two edge IDs
        edge_ids = list(graph.get_edge_ids())[:2]
        sub = graph.edge_subgraph(edge_ids)
        print(f"Nodes: {sub.count_nodes()}")  # 3 (endpoints of 2 edges)
        print(f"Edges: {sub.count_edges()}")  # 2
        ```

        Extract a path from a complete graph:

        ```python
        from edgecase import Graph

        graph = Graph().add_full_graph(4)  # K4 with 6 edges

        # Select specific edges to form a path
        edge_ids = list(graph.get_edge_ids())[:2]
        path = graph.edge_subgraph(edge_ids)
        ```

        %family Graph Algebra
        """
        from edgecase._core import edge_subgraph

        result = Graph.__new__(Graph)
        result._graph = edge_subgraph(self._graph, edges)
        return result

    def fully_connect_nodes_ws(self, **edge_attributes) -> "Graph":
        """
        Create edges between all pairs of selected nodes (clique).

        Adds edges to form a complete graph (clique) among all selected nodes. For directed graphs,
        creates edges in both directions. Skips edges that already exist.

        Parameters
        ----------
        **edge_attributes
            Optional attributes to set on newly created edges.

        Returns
        -------
        Graph
            Self for method chaining.

        Examples
        --------
        ```python
        graph = Graph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_node('C')

        # Create clique among selected nodes
        id_a = graph.get_id_from_label('A')
        id_b = graph.get_id_from_label('B')
        id_c = graph.get_id_from_label('C')
        (graph
            .select_nodes_by_id([id_a, id_b, id_c])
            .fully_connect_nodes_ws(type='clique', weight=1.0)
        )

        # For undirected: 3 edges (A-B, A-C, B-C)
        print(graph.count_edges())  # 3
        ```

        Directed clique (bidirectional edges):

        ```python
        graph_directed = ec.Graph(directed=True)
        graph_directed.add_node(label='X')
        graph_directed.add_node(label='Y')

        x_id = graph_directed.get_id_from_label('X')
        y_id = graph_directed.get_id_from_label('Y')
        (graph_directed
            .select_nodes_by_id([x_id, y_id])
            .fully_connect_nodes_ws()
        )

        # Creates X->Y and Y->X
        print(graph_directed.count_edges())  # 2
        ```

        Notes
        -----
        - for `n` selected nodes in an undirected graph, creates up to `n(n-1)/2` edges.
        - for `n` selected nodes in a directed graph, creates up to `n(n-1)` edges.
        - existing edges are not modified or duplicated.

        %family Graph Algebra
        """
        selected = list(self.get_selected_node_ids())

        # Create edges between all pairs
        for i, source in enumerate(selected):
            for target in selected[i + 1 :]:
                # Check if edge already exists
                has_edge = any(
                    (s == source and t == target)
                    or (not self.is_directed() and s == target and t == source)
                    for _, s, t in self.get_edges_as_tuples()
                )

                if not has_edge:
                    self.add_edge(source, target, **edge_attributes)

                # For directed graphs, add reverse edge
                if self.is_directed():
                    has_reverse = any(
                        s == target and t == source for _, s, t in self.get_edges_as_tuples()
                    )
                    if not has_reverse:
                        self.add_edge(target, source, **edge_attributes)

        return self

    def add_reverse_edges_ws(self, **edge_attributes) -> "Graph":
        """
        Add reverse edges for all selected edges (directed graphs only).

        For each selected edge (A->B), adds the reverse edge (B->A) if it doesn't already exist.
        Useful for making directed graphs bidirectional or adding reciprocal relationships.

        Parameters
        ----------
        **edge_attributes
            Optional attributes to set on the newly created reverse edges.

        Returns
        -------
        Graph
            Self for method chaining.

        Raises
        ------
        ValueError
            If the graph is undirected.

        Examples
        --------
        ```python
        import edgecase as ec

        graph = (
            ec.Graph(directed=True)
            .add_node(label='A')
            .add_node(label='B')
            .add_edge(source='A', target='B', weight=1.0)
        )

        # Add reverse edge
        edge_ids = list(graph.get_edge_ids())
        (graph
            .select_edges_by_id([edge_ids[0]])
            .add_reverse_edges_ws(weight=1.0)
        )

        print(graph.count_edges())  # 2 (A->B and B->A)
        ```

        Make selected edges bidirectional:

        ```python
        graph = (
            ec.Graph(directed=True)
            .add_node(label='A')
            .add_node(label='B')
            .add_node(label='C')
            .add_edge(source='A', target='B', weight=10.0)
            .add_edge(source='B', target='C', weight=3.0)
        )

        # Add reverse edges for all edges with high weight
        (graph
            .select_edges(lambda e: e.weight is not None and e.weight > 5.0)
            .add_reverse_edges_ws()
        )
        ```

        %family Graph Algebra
        """
        if not self.is_directed():
            raise ValueError("add_reverse_edges_ws() only works on directed graphs")

        selected = self.get_selected_edge_ids()
        edges_to_add = []

        for edge_id in selected:
            edge_tuple = self.get_node_ids_by_edge_id(edge_id)
            if edge_tuple:
                source, target = edge_tuple
                # Check if reverse edge exists
                has_reverse = any(
                    s == target and t == source for _, s, t in self.get_edges_as_tuples()
                )
                if not has_reverse:
                    edges_to_add.append((target, source))

        # Add reverse edges
        for source, target in edges_to_add:
            self.add_edge(source, target, **edge_attributes)

        return self

    def rev_edge_dir_ws(self) -> "Graph":
        """
        Reverse the direction of all selected edges (directed graphs only).

        For each selected edge (A->B), changes it to (B->A). Edge IDs and all attributes (including
        weights) are preserved during reversal. Only the direction is changed.

        Returns
        -------
        Graph
            Self for method chaining.

        Raises
        ------
        ValueError
            If the graph is undirected (undirected edges have no direction to reverse).

        Examples
        --------
        Reverse specific edges:

        ```python
        import edgecase as ec

        graph = ec.Graph(directed=True)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B", weight=1.5)

        # Reverse the edge direction
        edge_ids = graph.get_edge_ids()
        graph.select_edges_by_id([edge_ids[0]])
        graph.rev_edge_dir_ws()

        # Edge now goes B -> A
        source, target = graph.get_node_ids_by_edge_id(edge_ids[0])
        print(f"{source} -> {target}")  # B -> A

        # Weight is preserved
        edge = graph.get_edge(edge_ids[0])
        print(edge.weight)  # 1.5
        ```

        Reverse all edges in a subgraph:

        ```python
        graph = ec.Graph(directed=True)
        graph.add_path_graph(4)  # 0->1->2->3

        # Reverse middle edges
        edge_ids = graph.get_edge_ids()[1:3]
        graph.select_edges_by_id(edge_ids)
        graph.rev_edge_dir_ws()

        # Path is now: 0->1<-2<-3
        ```

        Notes
        -----
        - only works on directed graphs
        - edge attributes and IDs remain unchanged
        - selection state is preserved after reversal

        %family Graph Algebra
        """
        if not self.is_directed():
            raise ValueError("rev_edge_dir_ws() only works on directed graphs")

        selected = self.get_selected_edge_ids()

        for edge_id in selected:
            # Use Rust method to reverse edge direction
            self._graph.reverse_edge_direction(edge_id)

        return self

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # NODE METRICS
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Degree
    # ==============================================================================

    def degrees(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Get degree for all nodes in the graph.

        For directed graphs, returns total degree (in-degree + out-degree). For undirected graphs,
        returns the number of neighbors.

        This batch method is more efficient than calling get_degree() for each node individually and
        automatically stores results in the node DataFrame.

        Parameters
        ----------
        attr_name
            Column name to store degree values in the node DataFrame. If `None` (default),
            automatically uses `degree`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, int], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to degree values. If
            `return_dict=` is `False` (default): returns self for method chaining.

        Examples
        --------
        Compute degrees for all nodes:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .degrees(return_dict=False)  # Stores in 'degree' column, returns self
        )

        # Access via node DataFrame
        ndf = graph.get_node_df()
        print(ndf.select(["id", "degree"]))
        ```

        Use in method chain for visualization:

        ```python
        graph = (
            ec.karate_club()
            .degrees(return_dict=False)
            .rescale_node_attrs(
                node_attr_from="degree",
                node_attr_to="size",
                range=[0.5, 2.0],
            )
            .show()
        )
        ```

        %family Degree
        """
        # Compute degrees for all nodes
        node_ids = self.get_node_ids(limit=None)
        degrees = {node_id: self.get_degree(node_id) for node_id in node_ids}

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "degree"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, degrees)

        # Track computation
        if not return_dict:
            self._computations.append(("degrees", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return degrees
        return self

    def get_degree(self, node: Union[int, str]) -> int:
        """
        Get the degree of a node.

        For directed graphs, returns total degree (in-degree + out-degree). For undirected graphs,
        returns the number of neighbors.

        Parameters
        ----------
        node
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

        print(graph.get_degree("A"))  # 2
        print(graph.get_degree("B"))  # 1
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

        print(graph.get_degree("A"))  # 2 (1 in + 1 out)
        ```

        %family Degree
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

    def get_in_degree(self, node: Union[int, str]) -> int:
        """
        Get the in-degree of a node (directed graphs only).

        In-degree is the number of edges pointing into the node. Only meaningful for directed
        graphs.

        Parameters
        ----------
        node
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

        print(graph.get_in_degree("B"))  # 2
        print(graph.get_in_degree("A"))  # 0
        ```

        %family Degree
        """
        if not self._graph.is_directed:
            raise ValueError("in_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
        # edges() returns (edge_id, source, target) tuples
        return sum(1 for edge in self._graph.edges() if edge[2] == node_id)

    def get_out_degree(self, node: Union[int, str]) -> int:
        """
        Get the out-degree of a node (directed graphs only).

        Out-degree is the number of edges pointing out from the node. Only meaningful for directed
        graphs.

        Parameters
        ----------
        node
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

        print(graph.get_out_degree("A"))  # 2
        print(graph.get_out_degree("B"))  # 0
        ```

        %family Degree
        """
        if not self._graph.is_directed:
            raise ValueError("out_degree only available for directed graphs")
        node_id = self._resolve_node_id(node)
        return len(self._graph.neighbors(node_id))

    def get_degree_distribution(self) -> dict[int, int]:
        """
        Get degree distribution as a mapping from degree to count.

        Returns a histogram of node degrees, showing how many nodes have each degree value. Useful
        for analyzing network topology and identifying hub nodes or scale-free properties.

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

        dist = graph.get_degree_distribution()
        print(dist)  # {1: 4, 4: 1}; 4 nodes with degree 1, 1 node with degree 4
        ```

        %family Degree
        """
        return self._graph.degree_distribution()

    def get_average_degree(self) -> float:
        """
        Calculate average degree across all nodes.

        For directed graphs, this returns the average of (in-degree + out-degree). For undirected
        graphs, each edge contributes to the degree of both endpoints.

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

        print(graph.get_average_degree())  # 2.0 (cycle graph)
        ```

        %family Degree
        """
        return self._graph.average_degree()

    def get_agg_degree_in(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """
        Get aggregate value from indegree of nodes.

        Computes a single aggregate statistic from the indegree values of all nodes in a directed
        graph. Useful for summarizing the indegree distribution or analyzing filtered node subsets.

        Parameters
        ----------
        agg
            Aggregation function to apply. Must be one of:

            - `"sum"`: sum of all indegrees
            - `"min"`: minimum indegree
            - `"max"`: maximum indegree
            - `"mean"`: mean indegree
            - `"median"`: median indegree
        conditions
            Filter function that takes a node object and returns `True` to include the node in the
            aggregation. If `None`, all nodes are included.

        Returns
        -------
        float
            The computed aggregate indegree value. Returns `0.0` if no nodes match the conditions.

        Raises
        ------
        ValueError
            If the graph is undirected or if `agg=` is not a valid aggregation function.

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
            conditions=lambda n: graph.get_degree(node=n.id) > 3
        )
        ```

        %family Degree
        """
        if not self._graph.is_directed:
            raise ValueError("get_agg_degree_in() only works with directed graphs")

        valid_agg = {"sum", "min", "max", "mean", "median"}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")

        # Get indegrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.get_node_ids(limit=None):
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.get_in_degree(node_id))

        if not degrees:
            return 0.0

        # Apply aggregation
        if agg == "sum":
            return float(sum(degrees))
        elif agg == "min":
            return float(min(degrees))
        elif agg == "max":
            return float(max(degrees))
        elif agg == "mean":
            return sum(degrees) / len(degrees)
        elif agg == "median":
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n // 2 - 1] + sorted_degrees[n // 2]) / 2.0
            else:
                return float(sorted_degrees[n // 2])

    def get_agg_degree_out(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """
        Get aggregate value from outdegree of nodes.

        Computes a single aggregate statistic from the outdegree values of all nodes in a directed
        graph. Useful for summarizing the outdegree distribution or analyzing filtered node subsets.

        Parameters
        ----------
        agg
            Aggregation function to apply. Must be one of:

            - `"sum"`: sum of all outdegrees
            - `"min"`: minimum outdegree
            - `"max"`: maximum outdegree
            - `"mean"`: mean outdegree
            - `"median"`: median outdegree
        conditions
            Filter function that takes a node object and returns `True` to include the node in the
            aggregation. If `None`, all nodes are included.

        Returns
        -------
        float
            The computed aggregate outdegree value. Returns `0.0` if no nodes match the conditions.

        Raises
        ------
        ValueError
            If the graph is undirected or if `agg=` is not a valid aggregation function.

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
        # First set an attribute on some nodes
        node_list = list(graph.get_node_ids(limit=5))
        graph.set_node_attrs(node_attr='type', values='special', nodes=node_list)
        max_out = graph.get_agg_degree_out(
            'max',
            conditions=lambda n: n.attributes and 'type' in n.attributes
        )
        ```

        %family Degree
        """
        if not self._graph.is_directed:
            raise ValueError("get_agg_degree_out() only works with directed graphs")

        valid_agg = {"sum", "min", "max", "mean", "median"}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")

        # Get outdegrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.get_node_ids(limit=None):
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.get_out_degree(node_id))

        if not degrees:
            return 0.0

        # Apply aggregation
        if agg == "sum":
            return float(sum(degrees))
        elif agg == "min":
            return float(min(degrees))
        elif agg == "max":
            return float(max(degrees))
        elif agg == "mean":
            return sum(degrees) / len(degrees)
        elif agg == "median":
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n // 2 - 1] + sorted_degrees[n // 2]) / 2.0
            else:
                return float(sorted_degrees[n // 2])

    def get_agg_degree_total(self, agg: str, conditions: Optional[Callable] = None) -> float:
        """
        Get aggregate value from total degree of nodes.

        Computes a single aggregate statistic from the total degree values of all nodes. Works with
        both directed and undirected graphs. For directed graphs, total degree = indegree +
        outdegree.

        Parameters
        ----------
        agg
            Aggregation function to apply. Must be one of:

            - `"sum"`: Sum of all total degrees
            - `"min"`: Minimum total degree
            - `"max"`: Maximum total degree
            - `"mean"`: Mean total degree
            - `"median"`: Median total degree
        conditions
            Filter function that takes a node object and returns `True` to include the node in the
            aggregation. If `None`, all nodes are included.

        Returns
        -------
        float
            The computed aggregate total degree value. Returns `0.0` if no nodes match the
            conditions.

        Raises
        ------
        ValueError
            If `agg=` is not a valid aggregation function.

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

        %family Degree
        """
        valid_agg = {"sum", "min", "max", "mean", "median"}
        if agg not in valid_agg:
            raise ValueError(f"agg must be one of {valid_agg}, got '{agg}'")

        # Get total degrees for all nodes (or filtered nodes)
        degrees = []
        for node_id in self.get_node_ids(limit=None):
            if conditions is None or conditions(self.get_node(node_id)):
                degrees.append(self.get_degree(node_id))

        if not degrees:
            return 0.0

        # Apply aggregation
        if agg == "sum":
            return float(sum(degrees))
        elif agg == "min":
            return float(min(degrees))
        elif agg == "max":
            return float(max(degrees))
        elif agg == "mean":
            return sum(degrees) / len(degrees)
        elif agg == "median":
            sorted_degrees = sorted(degrees)
            n = len(sorted_degrees)
            if n % 2 == 0:
                return (sorted_degrees[n // 2 - 1] + sorted_degrees[n // 2]) / 2.0
            else:
                return float(sorted_degrees[n // 2])

    # ==============================================================================
    # Centrality
    # ==============================================================================

    def pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Calculate PageRank scores for all nodes in the graph.

        PageRank is a link analysis algorithm that assigns importance scores to nodes based on their
        incoming connections. Originally developed by Google to rank web pages, it's widely used in
        social network analysis, recommendation systems, and influence detection.

        The algorithm iteratively distributes "rank" from each node to its neighbors, with a damping
        factor to model random jumps. Nodes with many incoming edges from high-ranking nodes receive
        higher scores.

        Parameters
        ----------
        damping
            The probability of following an edge (vs. random jump). Higher values (closer to `1.0`)
            give more weight to the graph structure. Typical range: `0.8`-`0.95`.
        max_iter
            Maximum number of iterations to run. The algorithm stops early if convergence is
            reached.
        tol
            Convergence tolerance. The algorithm stops when the sum of absolute differences between
            iterations is less than this value.
        attr_name
            Column name to store PageRank scores in the node DataFrame. If `None` (default),
            automatically uses `pagerank`. Overwrites column if it already exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict` is `True`: dictionary mapping node IDs to PageRank scores (sum to
            `1.0`). If `return_dict` is `False` (default): returns self for method chaining.

        Examples
        --------
        Calculate PageRank on a simple social network:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
        )
        alice_id = graph.get_id_from_label("Alice")
        bob_id = graph.get_id_from_label("Bob")
        charlie_id = graph.get_id_from_label("Charlie")

        graph.add_edge(alice_id, bob_id)
        graph.add_edge(alice_id, charlie_id)
        graph.add_edge(bob_id, charlie_id)

        scores = graph.pagerank(return_dict=True)
        print(scores)
        # {1: 0.262, 2: 0.262, 3: 0.475}  # Node IDs to scores

        # Convert to labels if needed
        label_scores = {graph.get_label_from_id(nid): score for nid, score in scores.items()}
        print(label_scores)
        # {'Alice': 0.262, 'Bob': 0.262, 'Charlie': 0.475}
        ```

        Use custom damping factor for more exploration:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
        )
        graph.add_edge(source="Alice", target="Bob")
        graph.add_edge(source="Alice", target="Charlie")
        graph.add_edge(source="Bob", target="Charlie")

        scores = graph.pagerank(damping=0.5)  # More random jumping
        ```

        Find the most influential nodes:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Charlie")
        )
        graph.add_edge(source="Alice", target="Bob")
        graph.add_edge(source="Alice", target="Charlie")
        graph.add_edge(source="Bob", target="Charlie")

        scores = graph.pagerank(return_dict=True)
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for node_id, score in top_nodes:
            label = graph.get_label_from_id(node_id=node_id)
            print(f"{label} (ID {node_id}): {score:.4f}")
        ```

        %family Centrality
        %seealso betweenness_centrality, load_centrality, constraint
        """
        result_ids = rust.pagerank(self._graph, damping=damping, max_iter=max_iter, tol=tol)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "pagerank"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            params = {}
            if damping != 0.85:
                params["damping"] = damping
            if max_iter != 100:
                params["max_iter"] = max_iter
            if tol != 1e-6:
                params["tol"] = tol
            self._computations.append(("pagerank", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def constraint(
        self, weights: str | None = None, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float | None], "Graph"]:
        """
        Calculate Burt's constraint for each node.

        Burt's constraint measures the extent to which ego's contacts are redundant, that is,
        connected to each other. Higher constraint indicates fewer structural holes and less
        brokerage opportunity. Lower constraint indicates a network position with more non-redundant
        contacts, offering greater access to diverse information and control benefits.

        The constraint of node i's ego network is:

        `C_i = Σ_j (p_ij + Σ_q p_iq × p_qj)²`

        where `p_ij` are proportional tie strengths:

        `p_ij = (a_ij + a_ji) / Σ_k (a_ik + a_ki)`

        and `a_ij` are elements of the adjacency matrix (edge weights).

        Time complexity: `O(V × D²)` where `D` is average degree
        Space complexity: `O(V)`

        Parameters
        ----------
        weights
            Name of edge attribute to use as weights. If `None`, uses edge
            weight field. If weights don't exist, treats all edges equally.
        attr_name
            Name of the node attribute to store constraint values. If `None`, uses `constraint` as
            the attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to constraint values.
            If `False`, stores values as node attributes and returns the Graph for chaining.

        Returns
        -------
        dict[int, float | None] or Graph
            If `return_dict=` is `True`: Dictionary mapping node IDs to constraint values.
            Returns `None` for isolated nodes (undefined constraint).
            If `return_dict=` is `False`: Returns self for method chaining.

        Examples
        --------
        Calculate and store constraint values:

        ```python
        # Simple network with a broker (B)
        g = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")  # Broker
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("B", "D")
            # Note: C and D are not connected
            .constraint()  # Stores as 'constraint' attribute
        )

        # Access values by node ID
        b_id = g.get_id_from_label("B")
        a_id = g.get_id_from_label("A")
        constraints = g.constraint(return_dict=True)
        print(f"B (broker): {constraints[b_id]:.3f}")  # Low constraint
        print(f"A (connected): {constraints[a_id]:.3f}")  # High constraint
        ```

        Analyze structural holes:

        ```python
        from edgecase import Graph

        # Compare dense vs. sparse ego networks
        g = Graph()
        for i in range(6):
            g.add_node(label=str(i))

        # Node 0: connects to 1,2 which are connected (no structural hole)
        g.add_edge(source="0", target="1")
        g.add_edge(source="0", target="2")
        g.add_edge(source="1", target="2")

        # Node 3: connects to 4,5 which are NOT connected (structural hole)
        g.add_edge(source="3", target="4")
        g.add_edge(source="3", target="5")
        # Note: 4 and 5 are not connected

        constraints = g.constraint(return_dict=True)
        node_0_id = g.get_id_from_label("0")
        node_3_id = g.get_id_from_label("3")
        if constraints[node_0_id] is not None:
            print(f"Node 0 (dense): {constraints[node_0_id]:.3f}")  # Higher
        if constraints[node_3_id] is not None:
            print(f"Node 3 (sparse): {constraints[node_3_id]:.3f}")  # Lower
        ```

        Weighted networks:

        ```python
        g = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=5.0)
            .add_edge("A", "C", weight=1.0)
            .add_edge("B", "C", weight=3.0)
        )

        # Strong tie to B dominates A's network
        constraints = g.constraint(return_dict=True)
        print(constraints)
        ```

        Notes
        -----
        **Interpretation**:

        - **High constraint** (approaching 1.0): Ego's contacts are highly
          interconnected. Limited brokerage opportunities, high closure.
        - **Low constraint** (approaching 0.0): Ego's contacts are disconnected.
          Many structural holes, high brokerage potential.

        **Structural holes** occur when ego's contacts are not connected to each
        other, providing opportunities for:

        - information brokerage (access to diverse information)
        - control benefits (mediating between disconnected groups)
        - innovation (combining ideas from separate clusters)

        **Use cases**:

        - social capital analysis: who has advantageous network positions?
        - innovation research: individuals spanning structural holes
        - organizational networks: identifying key boundary spanners
        - entrepreneurship: access to diverse resources and information

        **Undefined values**: Isolated nodes have no ego network and return `None`.

        %family Centrality
        """
        if attr_name is None:
            attr_name = "constraint"

        constraint_values = self._graph.constraint(weights)

        if return_dict:
            return constraint_values

        # Store as node attributes
        for node_id, value in constraint_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        params = {}
        if weights is not None:
            params["weights"] = weights
        self._computations.append(("constraint", params, attr_name))

        return self

    def betweenness_centrality(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float], "Graph"]:
        """
        Calculate betweenness centrality scores for all nodes.

        Betweenness centrality measures how often a node appears on the shortest paths between other
        nodes in the graph. Nodes with high betweenness act as bridges or bottlenecks, controlling
        the flow of information or resources through the network.

        This metric is particularly useful for:

        - identifying key connectors in social networks
        - finding critical infrastructure nodes in transportation/communication networks
        - detecting potential bottlenecks or points of failure

        The algorithm computes shortest paths between all pairs of nodes and counts how many times
        each node appears on these paths.

        Parameters
        ----------
        attr_name
            Column name to store betweenness scores in the node DataFrame. If `None` (default),
            automatically uses `betweenness_centrality`. Overwrites column if it already exists.
        return_dict
            If `True`, returns a dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to betweenness scores (range of
            `[0, 1]`). If `return_dict=` is `False` (default): returns self for method chaining.

        Examples
        --------
        Find bridge nodes in a network:

        ```python
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

        scores = graph.betweenness_centrality(return_dict=True)
        bridge_id = graph.get_id_from_label("Bridge")
        print(scores[bridge_id])  # High score: bridges two clusters

        # Convert to labels if needed
        label_scores = {graph.get_label_from_id(nid): score for nid, score in scores.items()}
        print(label_scores["Bridge"])  # Same result
        ```

        Identify the most important connectors:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="E")
        )

        scores = graph.betweenness_centrality(return_dict=True)
        connectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        for node_id, score in connectors:
            label = graph.get_label_from_id(node_id)
            print(f"{label} (ID {node_id}): {score:.4f}")
        ```

        %family Centrality
        %seealso pagerank, load_centrality, constraint
        """
        result_ids = rust.betweenness_centrality(self._graph)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "betweenness_centrality"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            self._computations.append(("betweenness_centrality", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def load_centrality(
        self,
        normalized: bool = True,
        cutoff: int = None,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Calculate load centrality scores for all nodes.

        Load centrality measures the fraction of shortest paths that pass through each node. It is
        similar to betweenness centrality but uses a different accumulation formula that counts the
        'load' or traffic on nodes from shortest path routing.

        The key differences from betweenness centrality:

        - **Betweenness**: uses dependency accumulation with path count ratios
        - **Load**: uses simpler load counting based on predecessors

        Both metrics identify important bridge nodes, but load centrality can produce slightly
        different rankings that may be more intuitive for certain applications like network traffic
        analysis or identifying bottlenecks in routing systems.

        Parameters
        ----------
        normalized
            If `True` (default), normalizes scores by dividing by `(n-1)(n-2)` where `n` is the
            number of nodes. If `False`, returns raw load counts.
        cutoff
            Optional maximum path length to consider. Only paths of length ≤ cutoff are included.
            If `None` (default), considers all path lengths.
        attr_name
            Column name to store load centrality scores in the node DataFrame. If `None` (default),
            automatically uses 'load_centrality'. Overwrites column if it already exists.
        return_dict
            If `True`, returns a dictionary instead of self. If `False` (default), stores in NDF
            and returns self for chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to load centrality scores. If
            `return_dict=` is `False` (default): returns self for method chaining.

        Examples
        --------
        Identify bridge nodes using load centrality:

        ```python
        from edgecase import Graph

        # Create a network with a central bridge
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

        scores = graph.load_centrality(return_dict=True)
        bridge_id = graph.get_id_from_label("Bridge")
        print(f"Bridge load: {scores[bridge_id]:.4f}")  # High load score
        ```

        Compare load vs betweenness centrality:

        ```python
        from edgecase import Graph

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

        load = graph.load_centrality(return_dict=True)
        betweenness = graph.betweenness_centrality(return_dict=True)

        for node_id in sorted(load.keys()):
            label = graph.get_label_from_id(node_id)
            print(f"{label}: load={load[node_id]:.3f}, betweenness={betweenness[node_id]:.3f}")
        ```

        Store centrality as node attribute for later use:

        ```python
        from edgecase import Graph

        # Create a simple path graph
        graph = Graph()
        for i in range(5):
            graph.add_node(label=str(i))
        for i in range(4):
            graph.add_edge(str(i), str(i+1))

        # Compute and store load centrality
        graph.load_centrality(attr_name="load")

        # Verify it was stored
        any_comp = any("load" in str(c) for c in graph._computations)
        print(f"Load centrality computed: {any_comp}")
        ```

        %family Centrality
        %seealso betweenness_centrality, pagerank, constraint
        """
        # Call Rust implementation
        result_ids = rust.load_centrality(self._graph, normalized=normalized, cutoff=cutoff)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "load_centrality"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            self._computations.append(
                ("load_centrality", {"normalized": normalized, "cutoff": cutoff}, attr_name)
            )

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def second_order_centrality(
        self,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Compute the second order centrality for nodes in the graph.

        Second order centrality measures the standard deviation of return times to a node in a
        perpetual random walk on the graph. Lower values indicate higher centrality, as nodes with
        lower return time variance are more centrally positioned in the network structure.

        This metric is particularly useful for identifying nodes that are consistently reached in
        random walks, indicating their importance in network flow and connectivity patterns.

        Parameters
        ----------
        attr_name
            Name of node attribute to store centrality values. If `None`, uses
            `second_order_centrality`. Only used when `return_dict=False`.
        return_dict
            If `True`, returns dict mapping node IDs to centrality values. If `False`, stores
            values in graph and returns self for chaining.

        Returns
        -------
        dict or Graph
            If r`eturn_dict=True`: dictionary mapping node ID to centrality value. If
            `return_dict=False`: self (for method chaining).

        Raises
        ------
        ValueError
            - if graph is directed (algorithm only works for undirected graphs)
            - if graph is empty
            - if graph is not connected

        Notes
        -----
        **Algorithm Overview:**

        The algorithm implements the analytical solution from Kermarrec et al. (2011)
        based on Metropolis-Hastings random walks:

        1. Balance the graph by adding self-loops to equalize in-degrees
        2. Create transition probability matrix P
        3. For each node i, solve the linear system: (I - Q_i) * M[:, i] = 1
           where Q_i is P with column i zeroed
        4. Compute centrality as: sqrt(2 * sum(M[:, i]) - n * (n + 1))

        **Complexity:**

        - Time: O(n³) due to solving n linear systems of size n×n
        - Space: O(n²) for storing matrices
        - **Viable only for small to medium graphs (< 1,000 nodes)**

        **Interpretation:**

        - **Lower values** = **higher centrality** (more central nodes)
        - Values represent consistency of return times in random walks
        - Captures both direct and indirect influence through network structure
        - Complements first-order metrics like degree or betweenness

        **Current Limitation:**

        - Only supports unweighted graphs (all edges treated equally)

        Examples
        --------
        Identify central nodes in a path graph (lower values = higher centrality):

        ```python
        from edgecase import Graph

        # Create a simple path: 0 - 1 - 2 - 3 - 4
        graph = Graph()
        for i in range(5):
            graph.add_node(label=str(i))
        for i in range(4):
            graph.add_edge(str(i), str(i+1))

        soc = graph.second_order_centrality(return_dict=True)

        # Middle nodes have lower values (higher centrality)
        # Note: keys are node IDs, not labels
        most_central_id = min(soc, key=soc.get)
        most_central_label = graph.get_label_from_id(most_central_id)
        print(f"Most central node: {most_central_label}")  # Node '2'
        print(f"Centrality value: {soc[most_central_id]:.3f}")
        ```

        Compare centrality across different graph structures:

        ```python
        from edgecase import Graph

        # Star graph: center should be most central
        star = Graph()
        star.add_node(label='center')
        for i in range(5):
            star.add_node(label=f'leaf_{i}')
            star.add_edge('center', f'leaf_{i}')

        star_soc = star.second_order_centrality(return_dict=True)
        center_id = star.get_id_from_label('center')
        leaf_id = star.get_id_from_label('leaf_0')
        print(f"Center: {star_soc[center_id]:.3f}")
        print(f"Leaf: {star_soc[leaf_id]:.3f}")  # Higher value (less central)

        # Complete graph: all nodes equally central
        complete = Graph()
        for i in range(4):
            complete.add_node(label=str(i))
        for i in range(4):
            for j in range(i+1, 4):
                complete.add_edge(str(i), str(j))

        complete_soc = complete.second_order_centrality(return_dict=True)
        vals = list(complete_soc.values())
        print(f"All equal: {all(abs(v - vals[0]) < 0.01 for v in vals)}")
        ```

        Store centrality as node attribute for later analysis:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label='A')
            .add_node(label='B')
            .add_node(label='C')
            .add_node(label='D')
            .add_edge('A', 'B')
            .add_edge('B', 'C')
            .add_edge('C', 'D')
        )

        # Store with custom attribute name
        graph.second_order_centrality(attr_name='soc')

        # Verify it was computed
        has_soc = any('soc' in str(c) for c in graph._computations)
        print(f"Second order centrality computed: {has_soc}")
        ```

        %family Centrality
        """
        if self.is_directed():
            raise ValueError(
                "second_order_centrality() requires an undirected graph. "
                "The algorithm is not defined for directed graphs."
            )

        # Call Rust implementation
        result_ids = rust.second_order_centrality(self._graph)

        if not result_ids:
            return {} if return_dict else self

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "second_order_centrality"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            self._computations.append(("second_order_centrality", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def communicability(
        self,
        return_dict: bool = True,
    ) -> dict[int, dict[int, float]]:
        """
        Compute communicability between all pairs of nodes.

        Communicability measures the "amount of communication" between pairs of nodes based on all
        walks between them, with shorter walks weighted more heavily. It uses the matrix
        exponential of the adjacency matrix: exp(A).

        For nodes u and v, the communicability is:

        `C(u, v) = \\sum_{j=1}^{n} \\phi_j(u) \\phi_j(v) e^{\\lambda_j}`

        where :math:`\\phi_j` are eigenvectors and :math:`\\lambda_j` are eigenvalues of the
        adjacency matrix.

        Parameters
        ----------
        return_dict
            If `True`, return the communicability as a nested dictionary. Currently only dict
            return is supported.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary where `result[u][v]` is the communicability between nodes `u` and `v`.
            Values are always positive.

        Raises
        ------
        ValueError
            If the graph is directed or has self-loops.

        Notes
        -----
        - only works on undirected graphs without self-loops
        - time complexity: O(n³) for eigendecomposition + O(n²) for result
        - space complexity: O(n²) for matrices and results
        - communicability is symmetric: C(u, v) = C(v, u)
        - self-communicability C(u, u) > 0 (related to subgraph centrality)
        - higher values indicate more "paths" (walks of all lengths) between nodes

        Examples
        --------
        Basic usage on a simple graph:

        ```python
        from edgecase import Graph

        # Create a path graph: 1 -- 2 -- 3
        graph = Graph(directed=False)
        graph.add_node(label="1")
        graph.add_node(label="2")
        graph.add_node(label="3")
        graph.add_edge("1", "2")
        graph.add_edge("2", "3")

        comm = graph.communicability()
        id1 = graph.get_id_from_label("1")
        id2 = graph.get_id_from_label("2")
        id3 = graph.get_id_from_label("3")

        # Adjacent nodes have higher communicability
        print(f"C(1,2): {comm[id1][id2]:.3f}")  # Higher (adjacent)
        print(f"C(1,3): {comm[id1][id3]:.3f}")  # Lower (distance 2)
        ```

        Analyze information flow in a network:

        ```python
        from edgecase import Graph

        # Create a simple network
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("B", "D")

        comm = graph.communicability()

        # Node B is a hub - check its communicability with others
        b_id = graph.get_id_from_label("B")
        total_comm = sum(comm[b_id].values())
        print(f"Total communicability from B: {total_comm:.2f}")
        ```

        %family Centrality
        """
        from edgecase._core import communicability as rust_communicability

        result = rust_communicability(self._graph)
        return result

    def communicability_betweenness_centrality(
        self,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Compute communicability betweenness centrality for all nodes.

        This centrality measure quantifies the contribution of each node to the communicability
        between all other pairs of nodes. A node with high communicability betweenness is important
        for information flow in the network.

        The communicability betweenness of node r is:

        `\\omega_r = \\frac{1}{C} \\sum_{p \\neq q, q \\neq r} \\frac{G_{prq}}{G_{pq}}`

        where:
        - `G_{pq} = \\exp(A)_{pq}` is the communicability between p and q
        - `G_{prq}` is the decrease in communicability when node r is removed
        - `C = (n-1)^2 - (n-1)` is a normalization factor

        Parameters
        ----------
        attr_name : str, optional
            Name to store the centrality as a node attribute. If `None`, defaults to
            `communicability_betweenness`.
        return_dict : bool, default False
            If `True`, return centrality as a dictionary. If `False`, store as node attribute and
            return self for chaining.

        Returns
        -------
        dict[int, float] or Graph
            If `return_dict=` is `True`: Dictionary mapping node IDs to centrality values.
            If `return_dict=` is `False`: Self (for method chaining).

        Raises
        ------
        ValueError
            If the graph is directed or has self-loops.

        Notes
        -----
        - values are normalized to be in `[0, 1]`
        - time complexity: `O(n × n³) = O(n⁴)` due to eigendecomposition for each node
        - best suited for small to medium graphs (n < `500`)
        - unlike standard betweenness, considers all walks, not just shortest paths

        Examples
        --------
        Basic usage:

        ```python
        from edgecase import Graph

        # Create a simple network
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")

        cbc = graph.communicability_betweenness_centrality(return_dict=True)
        # Middle nodes B and C have higher betweenness
        for node_id, value in sorted(cbc.items(), key=lambda x: -x[1]):
            label = graph.get_label_from_id(node_id)
            print(f"{label}: {value:.3f}")
        ```

        Store as node attribute:

        ```python
        from edgecase import Graph

        graph = Graph(directed=False)
        graph.add_node(label="1")
        graph.add_node(label="2")
        graph.add_node(label="3")
        graph.add_edge("1", "2")
        graph.add_edge("2", "3")

        # Store with custom name and chain
        graph.communicability_betweenness_centrality(attr_name="cbc")
        # Result is stored in node attributes
        ```

        Compare with standard betweenness:

        ```python
        from edgecase import Graph

        # Star graph
        star = Graph(directed=False)
        star.add_node(label="center")
        for i in range(4):
            star.add_node(label=f"leaf_{i}")
            star.add_edge("center", f"leaf_{i}")

        cbc = star.communicability_betweenness_centrality(return_dict=True)

        # Center node should have highest communicability betweenness
        center_id = star.get_id_from_label("center")
        print(f"Center CBC: {cbc[center_id]:.3f}")  # Close to 1.0
        ```

        %family Centrality
        """
        from edgecase._core import communicability_betweenness_centrality as rust_cbc

        result = rust_cbc(self._graph)

        if not result:
            return {} if return_dict else self

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "communicability_betweenness"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result)

        # Track computation
        if not return_dict:
            self._computations.append(("communicability_betweenness_centrality", {}, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result
        return self

    # -------------------------------------------------------------------------
    # Percolation Methods
    # -------------------------------------------------------------------------

    def percolation_centrality(
        self,
        states: dict[int, float] | None = None,
        weight: str | None = None,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Compute percolation centrality for all nodes.

        Percolation centrality measures the importance of nodes based on the proportion of
        'percolated paths' that pass through them. This is particularly useful for analyzing
        information or disease spreading in networks.

        When all nodes have percolation state = `1.0`, this reduces to standard betweenness
        centrality.

        Parameters
        ----------
        states
            Dictionary mapping node labels to percolation states (values between `0.0` and `1.0`).
            States represent the probability of a node being in the "percolated" state (e.g.,
            infected, informed). If `None`, all nodes default to state `1.0`.
        weight
            Edge attribute name for weights. Currently not implemented.
        attr_name
            Name of node attribute to store results. If `None`, uses `percolation_centrality`. Only
            used when `return_dict=False`.
        return_dict
            If `True`, returns dict mapping node labels to percolation centrality. If `False`,
            stores results as node attributes and returns self.

        Returns
        -------
        dict or Graph
            If `return_dict=True`, returns `{node_label: centrality_value}`. If `return_dict=False`,
            returns self with node attributes set.

        Examples
        --------
        Basic usage with default states:

        ```python
        graph = Graph(directed=False)
        graph.add_path_graph(5)
        pc = graph.percolation_centrality(return_dict=True)
        ```

        With custom percolation states (e.g., infection probabilities):

        ```python
        graph = Graph(directed=False)
        graph.add_path_graph(5)
        states = {"1": 1.0, "2": 0.5, "3": 0.8, "4": 0.3, "5": 1.0}
        pc = graph.percolation_centrality(states=states, return_dict=True)
        ```

        %family Centrality
        """
        from edgecase._core import percolation_centrality as rust_pc

        # Convert states from labels to IDs if provided
        states_by_id = None
        if states is not None:
            states_by_id = {}
            for label, state in states.items():
                node_id = self._graph.id_from_label(str(label))
                states_by_id[node_id] = state

        result_by_id = rust_pc(self._graph, states_by_id, weight)

        # Convert result from IDs to labels
        result = {}
        for node_id, centrality in result_by_id.items():
            label = self._graph.label_from_id(node_id)
            if label is not None:
                result[label] = centrality
            else:
                result[node_id] = centrality

        # Store as node attributes if not returning dict
        if not return_dict:
            attr = attr_name if attr_name is not None else "percolation_centrality"
            # Use set_node_attrs with a dict mapping node IDs to values
            self.set_node_attrs(attr, result_by_id)

        # Track computation
        if not return_dict:
            self._computations.append(
                ("percolation_centrality", {"states": states, "weight": weight}, attr_name)
            )

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result
        return self

    def node_percolation(
        self,
        q: float,
        seed: int | None = None,
    ) -> "Graph":
        """
        Simulate node percolation by randomly removing nodes.

        Creates a new graph with a fraction of nodes randomly removed, along with all their incident
        edges. This is useful for analyzing network robustness under random node failures.

        Parameters
        ----------
        q
            Fraction of nodes to remove (must be between `0.0` and `1.0`). For example, `q=0.2`
            removes 20% of nodes.
        seed
            Random seed for reproducibility.

        Returns
        -------
        Graph
            A new Graph containing only the surviving nodes and edges.

        Examples
        --------
        Remove 30% of nodes randomly:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(10)

        # Remove 30% of nodes randomly
        percolated = graph.node_percolation(q=0.3, seed=23)
        print(percolated.count_nodes())  # 7 nodes remain
        ```

        %family Centrality
        """
        from edgecase._core import node_percolation as rust_np

        result_graph = rust_np(self._graph, q, seed)

        # Wrap the result in a new Graph object
        new_graph = Graph.__new__(Graph)
        new_graph._graph = result_graph
        new_graph._computations = []

        return new_graph

    def edge_percolation(
        self,
        q: float,
        seed: int | None = None,
    ) -> "Graph":
        """
        Simulate edge percolation by randomly removing edges.

        Creates a new graph with a fraction of edges randomly removed. All nodes are preserved.
        This is useful for analyzing network robustness under random edge failures or communication
        link failures.

        Parameters
        ----------
        q
            Fraction of edges to remove (must be between `0.0` and `1.0`). For example, `q=0.2`
            removes 20% of edges.
        seed
            Random seed for reproducibility.

        Returns
        -------
        Graph
            A new Graph containing all nodes but only surviving edges.

        Examples
        --------
        Remove 50% of edges randomly:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(5)  # K5 has 10 edges

        # Remove 50% of edges randomly
        percolated = graph.edge_percolation(q=0.5, seed=23)
        print(percolated.count_edges())  # 5 edges remain
        ```

        %family Centrality
        """
        from edgecase._core import edge_percolation as rust_ep

        result_graph = rust_ep(self._graph, q, seed)

        # Wrap the result in a new Graph object
        new_graph = Graph.__new__(Graph)
        new_graph._graph = result_graph
        new_graph._computations = []

        return new_graph

    def largest_component_size(self) -> int:
        """
        Get the size of the largest connected component.

        For undirected graphs, finds the largest connected component. For directed graphs, finds the
        largest weakly connected component.

        Returns
        -------
        int
            Number of nodes in the largest connected component.

        Examples
        --------
        Get the size of the largest component:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(5)
        print(graph.largest_component_size())  # 5
        ```

        After removing nodes, the component may fragment:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(5)
        percolated = graph.node_percolation(q=0.5, seed=23)
        print(percolated.largest_component_size())  # May be smaller
        ```

        %family Centrality
        """
        from edgecase._core import largest_component_size as rust_lcs

        return rust_lcs(self._graph)

    def percolation_threshold_analysis(
        self,
        steps: int = 20,
        trials: int = 10,
        seed: int | None = None,
    ) -> list[tuple[float, float]]:
        """
        Perform percolation threshold analysis.

        Systematically removes nodes at different fractions and measures the relative size of the
        giant connected component (GCC) at each step. This helps identify the critical percolation
        threshold where the network fragments.

        Parameters
        ----------
        steps
            Number of removal fractions to test. Results in (steps + 1) data points
            from q=0.0 to q=1.0.
        trials
            Number of random trials per step for averaging. More trials give
            smoother results but take longer.
        seed
            Random seed for reproducibility.

        Returns
        -------
        list of tuple
            List of `(q, relative_gcc_size)` tuples where:

            - `q` is the fraction of nodes removed
            - `relative_gcc_size` is the average relative size of the giant connected component
            (GCC size / surviving nodes)

        Notes
        -----
        The percolation threshold is typically identified where the relative GCC size drops sharply.
        For random Erdős-Rényi graphs, this occurs around `q_c = 1 - 1/<k>` where `<k>` is the
        average degree.

        Examples
        --------
        Perform percolation threshold analysis:

        ```python
        graph = Graph(directed=False)
        graph.add_full_graph(50)

        results = graph.percolation_threshold_analysis(steps=10, trials=5)

        for q, gcc_size in results:
            print(f"q={q:.1f}: GCC={gcc_size:.2f}")
        # q=0.0: GCC=1.00
        # q=0.1: GCC=1.00
        # ...
        ```

        %family Centrality
        """
        from edgecase._core import percolation_threshold_analysis as rust_pta

        return rust_pta(self._graph, steps, trials, seed)

    def average_neighbor_degree(
        self,
        source: str = "out",
        target: str = "out",
        weight: str | None = None,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Compute average neighbor degree for each node.

        For each node, calculates the average degree of its neighbors. This metric helps analyze
        degree correlations and mixing patterns in networks, indicating whether nodes tend to
        connect to similar-degree neighbors (assortative) or different-degree neighbors
        (disassortative).

        For undirected graphs, computes the average degree of all neighbors. For directed graphs,
        the `source` parameter controls which neighbors to consider, and the `target` parameter
        controls how to measure their degrees.

        Parameters
        ----------
        source
            For directed graphs only, specifies which neighbors to consider: (1) `"out"` for
            out-neighbors (successors), (2) `"in"` for in-neighbors (predecessors), and (3)
            `"in+out"` for both in-neighbors and out-neighbors. Raises error if used with undirected
            graphs.
        target
            For directed graphs only, specifies how to measure neighbor degrees: (1) `"out"` for
            out-neighbors (successors), (2) `"in"` for in-neighbors (predecessors), and (3)
            `"in+out"` for both in-neighbors and out-neighbors. Raises error if used with undirected
            graphs.
        weight
            Edge attribute name containing weights. If provided, computes weighted average using:
            `k_nn,i = (1 / s_i) * Σ_j w_ij * k_j`, where `s_i` is the weighted degree. Currently not
            implemented and will raise error if specified.
        attr_name
            Name of node attribute to store results. If `None`, uses `average_neighbor_degree`. Only
            used when `return_dict=False`.
        return_dict
            If `True`, returns dict mapping node labels to average neighbor degrees. If `False`,
            stores values in graph and returns self for chaining.

        Returns
        -------
        dict or Graph
            If `return_dict=True`: dictionary mapping node ID to average neighbor degree. If
            `return_dict=False`: self (for method chaining).

        Raises
        ------
        ValueError
            - if graph is empty
            - if `source` or `target` parameters are used with undirected graphs
            - if `source` or `target` values are invalid
            - if `weight` is specified (not yet implemented)

        Examples
        --------
        **Path Graph - Undirected**

        ```python
        from edgecase import Graph

        # In a path graph, end nodes have neighbors with degree 1,
        # middle nodes have neighbors with varying degrees
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i+1))

        and_values = g.average_neighbor_degree(return_dict=True)
        # Node '0': neighbor '1' has degree 2 -> average = 2.0
        # Node '1': neighbors '0','2' have degrees 1,2 -> average = 1.5
        # Node '2': neighbors '1','3' have degrees 2,2 -> average = 2.0
        ```

        **Directed Graph - Different source/target combinations**

        ```python
        g = Graph(directed=True)
        g.add_node(label='A')
        g.add_node(label='B')
        g.add_node(label='C')
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')

        # Average out-degree of out-neighbors
        and_out_out = g.average_neighbor_degree(source="out", target="out", return_dict=True)

        # Average in-degree of in-neighbors
        and_in_in = g.average_neighbor_degree(source="in", target="in", return_dict=True)
        ```

        **Store in Graph and Chain**

        ```python
        g = (
            Graph(directed=False)
            .add_node(label='A')
            .add_node(label='B')
            .add_node(label='C')
            .add_edge('A', 'B')
            .add_edge('B', 'C')
            .average_neighbor_degree(attr_name='knn')
        )

        # Values stored as node attribute 'knn'
        ```

        Notes
        -----
        - measures degree correlation patterns in networks
        - high values indicate nodes connect to high-degree neighbors (assortative mixing)
        - low values indicate nodes connect to low-degree neighbors (disassortative mixing)
        - for undirected graphs, ignores `source` and `target` parameters
        - nodes with no neighbors have average neighbor degree of `0`
        - weighted version not yet implemented

        %family Centrality
        """
        from edgecase import _core as rust

        # Call Rust implementation
        result_ids = rust.average_neighbor_degree(
            self._graph,
            source=source if self.is_directed() else None,
            target=target if self.is_directed() else None,
            weight=weight,
        )

        if not result_ids:
            return {} if return_dict else self

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "average_neighbor_degree"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            params = {}
            if self.is_directed():
                params["source"] = source
                params["target"] = target
            if weight is not None:
                params["weight"] = weight
            self._computations.append(("average_neighbor_degree", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def harmonic_centrality(
        self,
        mode: str = "out",
        weights: str = None,
        normalized: bool = False,
        cutoff: int = -1,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Calculate harmonic centrality scores for all nodes.

        Harmonic centrality is the sum of the reciprocal of shortest path distances from a node to
        all other reachable nodes. For unreachable nodes, the distance is considered infinite, and
        the reciprocal is `0`. This makes harmonic centrality more robust than closeness centrality
        for disconnected graphs.

        The formula is: `H(v) = Σ 1 / d(v,u)` for all `u ≠ v`, where `d(v,u)` is the shortest path
        distance from `v` to `u`.

        When `normalized=True`, the result is divided by `(n - 1)`, giving the mean inverse
        distance, which is bounded between `0` and `1`.

        Parameters
        ----------
        mode
            Direction for measuring distances in directed graphs where: (1) `"out"` follows edges in
            their direction (default), (2) `"in"` follows edges in reverse, and (3) `"all"` or
            `"total"` ignores edge directions (only for undirected graphs).
        weights
            Optional edge attribute name for weighted distances. Currently not implemented; uses
            unweighted shortest paths (hop count).
        normalized
            If `True`, divides by `(n - 1)` to normalize scores to the `[0, 1]` range. If `False`
            (the default), returns raw sum of inverse distances.
        cutoff
            Maximum path length to consider. Paths longer than cutoff are ignored. Use `-1`
            (default) for no limit.
        attr_name
            Column name to store harmonic centrality scores in the node DataFrame. If `None`
            (default), automatically uses 'harmonic_centrality'. Overwrites column if it already
            exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to harmonic centrality scores.
            If `return_dict=` is `False` (the default): returns self for method chaining.

        Examples
        --------
        Calculate harmonic centrality in a simple network:

        ```python
        from edgecase import Graph

        # Create a star network
        graph = (
            Graph()
            .add_node(label="Center")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("Center", "A")
            .add_edge("Center", "B")
            .add_edge("Center", "C")
        )

        scores = graph.harmonic_centrality(return_dict=True)
        center_id = graph.get_id_from_label("Center")
        print(scores[center_id])  # Center has highest harmonic centrality
        ```

        Compare with normalized version:

        ```python
        from edgecase import Graph

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

        # Raw harmonic centrality
        raw_scores = graph.harmonic_centrality(return_dict=True)

        # Normalized to [0, 1] range
        norm_scores = graph.harmonic_centrality(normalized=True, return_dict=True)
        ```

        Use with cutoff for local neighborhoods:

        ```python
        from edgecase import Graph

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

        # Only consider paths of length 2 or less
        local_scores = graph.harmonic_centrality(cutoff=2, return_dict=True)
        ```

        Chain with other operations:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .harmonic_centrality(attr_name="harmonic", normalized=True)
        )

        # Access the stored values
        ndf = graph.get_node_df()
        print(ndf[["label", "harmonic"]])
        ```

        For directed graphs, use mode to control direction:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Out-going paths (default)
        out_scores = graph.harmonic_centrality(mode="out", return_dict=True)

        # In-coming paths
        in_scores = graph.harmonic_centrality(mode="in", return_dict=True)
        ```

        %family Centrality
        %seealso closeness_vitality, betweenness_centrality, radiality
        """
        result_ids = rust.harmonic_centrality(
            self._graph,
            mode=mode,
            weights=weights,
            normalized=normalized,
            cutoff=cutoff,
        )

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "harmonic_centrality"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, result_ids)

        # Track computation
        if not return_dict:
            params = {}
            if mode != "out":
                params["mode"] = mode
            if weights is not None:
                params["weights"] = weights
            if normalized:
                params["normalized"] = normalized
            if cutoff != -1:
                params["cutoff"] = cutoff
            self._computations.append(("harmonic_centrality", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return result_ids
        return self

    def authority_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get HITS authority scores for each node (directed graphs only).

        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores for nodes in a
        directed graph. Authority score measures how much a node is pointed to by good hubs:

        `authority(v) = sum of hub scores of nodes pointing to v`

        Uses power iteration until convergence. Good authorities are nodes that are pointed to by
        many high-quality hubs. Requires a directed graph.

        Parameters
        ----------
        max_iter
            Maximum number of iterations for convergence.
        tol
            Convergence tolerance threshold.
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`authority_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to normalized authority scores.
            If `return_dict=` is `False`: returns self for method chaining.

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

        auth = graph.authority_centrality(return_dict=True, )
        # Paper2 and Paper3 have high authority (cited)
        # Paper1 has low authority (not cited)
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="Paper1")
            .add_node(label="Paper2")
            .add_node(label="Paper3")
            .add_edge(source="Paper1", target="Paper2")
            .add_edge(source="Paper1", target="Paper3")
        )

        # (graph
        #     .authority_centrality(attr_name="authority")
        #     .rescale_node_attrs(node_attr_from="authority", node_attr_to="size")
        #     .show())
        ```

        %family Centrality
        """
        centrality = self._graph.get_authority_centrality(max_iter, tol)

        if return_dict:
            return centrality

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "authority_centrality"

        self.set_node_attrs(attr_name, centrality)

        # Track computation
        params = {}
        if max_iter != 100:
            params["max_iter"] = max_iter
        if tol != 1e-6:
            params["tol"] = tol
        self._computations.append(("authority_centrality", params, attr_name))

        return self

    def hub_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get HITS hub scores for each node (directed graphs only).

        HITS (Hyperlink-Induced Topic Search) computes authority and hub scores for nodes in a
        directed graph. Hub score measures how much a node points to good authorities:

        `hub(v) = sum of authority scores of nodes v points to`

        Uses power iteration until convergence. Good hubs are nodes that point to many high-quality
        authorities. Requires a directed graph.

        Parameters
        ----------
        max_iter
            Maximum number of iterations for convergence.
        tol
            Convergence tolerance threshold.
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`hub_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to normalized hub scores. If
            `return_dict=` is `False`: returns self for method chaining.

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

        hub = graph.hub_centrality(return_dict=True, )
        # Paper1 has high hub score (cites many papers)
        # Paper2 and Paper3 have low hub scores
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="Paper1")
            .add_node(label="Paper2")
            .add_node(label="Paper3")
            .add_edge(source="Paper1", target="Paper2")
            .add_edge(source="Paper1", target="Paper3")
        )

        # (graph
        #     .hub_centrality(attr_name="hub")
        #     .rescale_node_attrs(node_attr_from="hub", node_attr_to="size")
        #     .show())
        ```

        %family Centrality
        """
        centrality = self._graph.get_hub_centrality(max_iter, tol)

        if return_dict:
            return centrality

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "hub_centrality"

        self.set_node_attrs(attr_name, centrality)

        # Track computation
        params = {}
        if max_iter != 100:
            params["max_iter"] = max_iter
        if tol != 1e-6:
            params["tol"] = tol
        self._computations.append(("hub_centrality", params, attr_name))

        return self

    def eigen_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get eigenvector centrality for each node.

        Eigenvector centrality measures node importance based on connections to other important
        nodes. It's the principal eigenvector of the adjacency matrix, computed via power iteration:

        `x = λ⁻¹ · A · x`

        A node has high eigenvector centrality if it's connected to other nodes that themselves have
        high centrality. Unlike degree centrality, it accounts for the quality (not just quantity)
        of connections.

        Parameters
        ----------
        max_iter
            Maximum number of iterations for convergence.
        tol
            Convergence tolerance threshold.
        attr_name
            Column name to store eigenvector centrality scores in the node DataFrame. If `None`
            (default), automatically uses `eigen_centrality`. Overwrites column if it already
            exists.
        return_dict
            If `True`, returns dictionary instead of self. If `False` (default), stores in NDF and
            returns self for chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to centrality scores. If
            `return_dict=` is `False` (default): returns self for method chaining.

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

        eigen = graph.eigen_centrality()
        # Hub has highest eigenvector centrality
        # All connected nodes have non-zero centrality
        ```

        Store as node attributes and chain other operations:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="hub", target="A")
            .add_edge(source="hub", target="B")
            .add_edge(source="hub", target="C")
        )

        # (graph
        #     .eigen_centrality(attr_name="centrality")
        #     .rescale_node_attrs(node_attr_from="centrality", node_attr_to="size", range=[0.5, 2.0])
        #     .show())
        ```

        %family Centrality
        %seealso katz_centrality, alpha_centrality, pagerank, hub_centrality, authority_centrality
        """
        centrality = self._graph.get_eigen_centrality(max_iter, tol)

        # Auto-generate attribute name if not provided
        if attr_name is None:
            attr_name = "eigen_centrality"

        # Store in node DataFrame
        self.set_node_attrs(attr_name, centrality)

        # Track computation
        if not return_dict:
            params = {}
            if max_iter != 100:
                params["max_iter"] = max_iter
            if tol != 1e-6:
                params["tol"] = tol
            self._computations.append(("eigen_centrality", params, attr_name))

        # Return dict if requested, otherwise return self for chaining
        if return_dict:
            return centrality
        return self

    def alpha_centrality(
        self,
        alpha: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get alpha centrality for each node.

        Alpha centrality is a generalization of eigenvector centrality that adds an exogenous
        factor:

        `x = α · A · x + e`

        where `α` is the attenuation factor, `A` is the adjacency matrix, and `e` is the exogenous
        factor (set to `1` for all nodes). Lower `α` emphasizes local neighborhood structure, while
        higher `α` considers more global structure.

        Parameters
        ----------
        alpha
            Attenuation factor (range: `0` to `1`). Controls the balance between local and global
            structure.
        max_iter
            Maximum number of iterations for convergence.
        tol
            Convergence tolerance threshold.
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`alpha_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to centrality scores. If
            `return_dict=` is `False`: returns self for method chaining.

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

        alpha_local = graph.alpha_centrality(return_dict=True, alpha=0.05)  # Local emphasis
        alpha_global = graph.alpha_centrality(return_dict=True, alpha=0.5)  # Global emphasis
        ```

        Store as node attributes and chain visualization:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="hub", target="A")
            .add_edge(source="hub", target="B")
            .add_edge(source="hub", target="C")
        )

        # (graph
        #     .alpha_centrality(attr_name="alpha_cent")
        #     .colorize_node_attrs(node_attr_from="alpha_cent", node_attr_to="fillcolor")
        #     .show()
        # )
        ```

        %family Centrality
        """
        centrality = self._graph.get_alpha_centrality(alpha, max_iter, tol)

        if return_dict:
            return centrality

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "alpha_centrality"

        self.set_node_attrs(attr_name, centrality)

        # Track computation
        params = {}
        if alpha != 0.1:
            params["alpha"] = alpha
        if max_iter != 100:
            params["max_iter"] = max_iter
        if tol != 1e-6:
            params["tol"] = tol
        self._computations.append(("alpha_centrality", params, attr_name))

        return self

    def bonacich_power_centrality(
        self,
        beta: float = 0.1,
        normalized: bool = False,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get Bonacich power centrality for each node.

        Bonacich power centrality generalizes eigenvector centrality to capture both cooperative and
        competitive network dynamics. The key innovation is the β parameter, which controls whether
        neighbors' power increases (+β) or decreases (-β) a node's own power.

        The centrality is computed as:

        `c(α, β) = α(I - βA)^(-1) A 1`

        where `A` is the adjacency matrix, `α` is a scaling parameter (set automatically), `β`
        controls the nature of relations, and `1` is a vector of ones.

        `β > 0`: cooperative relations: being connected to powerful nodes makes you powerful (e.g.,
        collaboration networks, social influence, information diffusion)

        `β < 0`: competitive relations: being connected to weak nodes makes you powerful (e.g.,
        exchange networks, competitive markets, bargaining power)

        `β = 1/λ_max`: equivalent to eigenvector centrality

        The magnitude of `|β|` controls how far influence propagates through the network. Larger
        `|β|` values mean distant nodes have more influence; a smaller `|β|` localizes influence.

        Parameters
        ----------
        beta
            The decay/attenuation parameter. Controls the nature and reach of influence. See the
            *Details on the beta Parameter* section for more information on settings.
        normalized
            The normalization method. With `False`, we get L2 normalization where `sum(c^2) = n`,
            making `c=1` the 'middle' value. Using `True` results in a sum normalization where
            `sum(c) = 1` (probability distribution).
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`bonacich_power_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to centrality scores. If
            `return_dict=` is `False`: returns self for method chaining. The interpretation depends
            on `β`: (1) when `β > 0` higher scores = more powerful due to powerful neighbors, and
            (2) when `β < 0` higher scores = more powerful due to weak neighbors.

        Raises
        ------
        RuntimeError
            If `(I - βA)` is singular. This happens when `β` is too close to `1/λ_max`. Try reducing
            `|β|` or choosing a different value.

        Details on the beta Parameter
        -----------------------------
        The `beta=` should usually be given a value between `-0.5` to `0.5`. Here's a breakdown on
        the effect of the `β` value:

        - `β > 0`: Cooperative (powerful neighbors -> more power)
        - `β < 0`: Competitive (weak neighbors -> more power)
        - `β = 0`: Reduces to simple degree centrality

        Examples
        --------
        Cooperative network (`β > 0`) star graph:

        ```python
        import edgecase as ec

        # Create star: center connected to 3 leaves
        g = (
            ec.Graph(directed=True)
            .add_node(label="center")
            .add_node(label="leaf1")
            .add_node(label="leaf2")
            .add_node(label="leaf3")
            .add_edge(source="center", target="leaf1")
            .add_edge(source="center", target="leaf2")
            .add_edge(source="center", target="leaf3")
        )

        # Cooperative: powerful neighbors increase your power
        power = g.bonacich_power_centrality(return_dict=True, beta=0.1)
        # center will have highest centrality (connected to many)
        ```

        Competitive network (`β < 0`) exchange network:

        ```python
        import edgecase as ec

        # In exchange/bargaining networks, connecting to weak partners
        # gives you more bargaining power
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="C")
        g.add_edge(source="C", target="D")

        # Negative beta: weak neighbors increase your power
        power = g.bonacich_power_centrality(return_dict=True, beta=-0.2)
        # End nodes (A, D) have higher power, so, fewer alternatives
        ```

        Compare with eigenvector centrality:

        ```python
        import edgecase as ec

        # β near 1/λ_max approximates eigenvector centrality
        # For small graphs, λ_max ≈ max_degree, so try β ≈ 1/max_degree
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="C")
        g.add_edge(source="B", target="D")

        max_deg = max(g.get_degree(node=nid) for nid in g.get_node_ids())
        bonacich = g.bonacich_power_centrality(return_dict=True, beta=0.9/max_deg)
        eigen = g.eigen_centrality()
        # Should be similar
        ```

        Notes
        -----
        - **Algorithm**: uses matrix inversion via LU decomposition to solve `(I - βA)x = A1`
        - **Time complexity**: `O(n³)` for dense graphs due to matrix operations
        - **Space complexity**: `O(n²)` to store the adjacency matrix
        - **Numerical stability**: the algorithm may fail if `β` is too close to `1/λ_max`, where
        `λ_max` is the largest eigenvalue of `A`. Choose `|β| < 0.9/λ_max` for safety.
        - **β selection**: there's no universal "correct" `β` value; it should be chosen based on
        theory and the specific application.
        - **Negative β instability**: centrality can be more sensitive to `β` changes when `β < 0`
        compared to `β > 0`. Test multiple values.
        - **Directed graphs**: works for both directed and undirected graphs. In directed graphs,
        the interpretation relates to in-degree influence patterns.

        %family Centrality
        """
        result = self._graph.get_bonacich_power_centrality(beta, normalized)
        # Empty result indicates singular matrix error
        if not result and self.count_nodes() > 0:
            raise RuntimeError(
                "Matrix (I - β·A) is singular. β is too close to 1/λ_max. "
                "Try a smaller |β| value (e.g., reduce by half)."
            )

        if return_dict:
            return result

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "bonacich_power_centrality"

        self.set_node_attrs(attr_name, result)

        # Track computation
        params = {}
        if beta != 0.1:
            params["beta"] = beta
        if normalized:
            params["normalized"] = normalized
        self._computations.append(("bonacich_power_centrality", params, attr_name))

        return self

    def katz_centrality(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get Katz centrality for each node.

        Katz centrality measures node influence based on the number of walks of all lengths reaching
        the node, with walks weighted by `α^length`:

        `x = α · Aᵀ · x + β`

        where α is the attenuation factor, `Aᵀ` is the transpose of the adjacency matrix, and `β` is
        a bias vector (constant for all nodes). For directed graphs, this counts incoming paths,
        making it suitable for measuring "prestige" or "influence received".

        Key differences from other centralities:

        - PageRank: uses teleportation and out-degree normalization
        - Eigenvector: only considers infinite-length walks (principal eigenvector)
        - Alpha centrality: uses `A` instead of `Aᵀ` (outgoing vs incoming)
        - Katz: weights walks by path length, considers all finite walks

        Parameters
        ----------
        alpha
            Attenuation factor for walk lengths. Must be smaller than the reciprocal of the largest
            eigenvalue (typically < `0.2` for safety). Higher values give more weight to longer
            paths.
        beta
            Bias/weight added to each node. Acts as a 'free' centrality that every node gets. Higher
            values reduce the relative importance of network position.
        max_iter
            Maximum number of iterations for convergence.
        tol
            Convergence tolerance threshold.
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`katz_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to normalized Katz centrality
            scores. If `return_dict=` is `False`: returns self for method chaining. Scores sum to
            `1.0`.

        Examples
        --------
        Calculate Katz centrality on a simple directed graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_node("D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("A", "D")
        )

        katz = graph.katz_centrality(return_dict=True, alpha=0.1)
        # D has highest Katz (receives edges from A and C via B)
        ```

        Compare different alpha values:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="A", target="D")
        )

        katz_local = graph.katz_centrality(return_dict=True, alpha=0.05)  # Emphasis on short paths
        katz_global = graph.katz_centrality(return_dict=True, alpha=0.15) # Include longer paths
        ```

        Notes
        -----
        - for undirected graphs, Aᵀ = A, so behavior is similar to alpha centrality
        - for directed graphs, high Katz = many incoming paths (prestige/authority)
        - alpha should be < 1/λ_max to ensure convergence (typically α < 0.2)
        - beta = 1.0 treats all nodes equally; adjust to bias certain nodes
        - normalizes final scores to sum to 1.0 for interpretability

        %family Centrality
        %seealso eigen_centrality, alpha_centrality, pagerank, bonacich_power_centrality
        """
        scores_by_id = self._graph.get_katz_centrality(alpha, beta, max_iter, tol)

        if return_dict:
            return scores_by_id

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "katz_centrality"

        self.set_node_attrs(attr_name, scores_by_id)

        # Track computation
        params = {}
        if alpha != 0.1:
            params["alpha"] = alpha
        if beta != 1.0:
            params["beta"] = beta
        if max_iter != 100:
            params["max_iter"] = max_iter
        if tol != 1e-6:
            params["tol"] = tol
        self._computations.append(("katz_centrality", params, attr_name))

        return self

    def leverage_centrality(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get leverage centrality for each node.

        Leverage centrality measures the degree connectivity of a node relative to its neighbors:

        `leverage(v) = (1/k_v) · Σ[(k_v - k_n) / (k_v + k_n)]`

        where `k_v` is the degree of node `v`, and `k_n` is the degree of each neighbor `n`.

        Interpretation:

        - positive values: node has more connections than its neighbors (influential)
        - negative values: node has fewer connections than its neighbors (peripheral)
        - zero: node has similar connectivity to its neighbors

        Parameters
        ----------
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`leverage_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to leverage centrality scores.
            If `return_dict=` is `False`: returns self for method chaining.

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

        leverage = graph.leverage_centrality(return_dict=True)
        # Hub has positive leverage (high degree, low-degree neighbors)
        # Leaves have negative leverage (low degree, high-degree neighbor)
        ```

        Store as node attributes and chain:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="hub", target="A")
            .add_edge(source="hub", target="B")
            .add_edge(source="hub", target="C")
        )

        # (graph
        #     .leverage_centrality(attr_name="leverage")
        #     .colorize_node_attrs(node_attr_from="leverage", node_attr_to="fillcolor", palette="RdBu")
        #     .show())
        ```

        %family Centrality
        """
        centrality = self._graph.get_leverage_centrality()

        if return_dict:
            return centrality

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "leverage_centrality"

        self.set_node_attrs(attr_name, centrality)

        # Track computation (no non-default params)
        self._computations.append(("leverage_centrality", {}, attr_name))

        return self

    def domirank_centrality(
        self,
        sigma: float | None = None,
        alpha: float = 0.95,
        method: str = "iterative",
        tolerance: float = 1e-5,
        max_iter: int = 1000,
        patience: int = 10,
        max_depth: int = 50,
        return_sigma: bool = False,
        attr_name: str | None = None,
        return_dict: bool = False,
    ) -> Union[dict[int, float], tuple[dict[int, float], float], "Graph", tuple["Graph", float]]:
        """
        Get DomiRank centrality for each node.

        DomiRank centrality quantifies node dominance in their neighborhoods. The competition
        parameter sigma (or alpha) provides a tunable trade-off between local and global topological
        information:

        `d𝚪ᵢ(t)/dt = σ(dᵢ - Σⱼ Aᵢⱼ 𝚪ⱼ(t)) - 𝚪ᵢ(t)`

        Where:

        - `𝚪ᵢ` is the dominance score for node `i`
        - `σ` is the competition parameter
        - `dᵢ` is the degree of node `i`
        - `A` is the adjacency matrix

        Unlike eigenvector-type centralities (where being connected to important nodes makes you
        important), DomiRank identifies nodes that dominate their local structure. Being close to a
        dominant node can suppress your own dominance in high-competition contexts.

        For `σ -> 0`: Converges to degree centrality (only local information)
        For large `σ`: Identifies globally dominant nodes

        Parameters
        ----------
        sigma
            Competition parameter. If `None` (default), automatically computed from smallest
            eigenvalue and alpha. Higher values increase global competition for dominance.
        alpha
            Alpha parameter for auto-computing sigma (`sigma = alpha / |λ_min|`). Only used if sigma
            is `None`. Must be positive.
        method
            Solution method. Currently only "iterative" is implemented. (Analytical method may be
            added in the future for small graphs.)
        tolerance
            Convergence tolerance for iterative solver.
        max_iter
            Maximum iterations for Newton's method.
        patience
            Number of steps between convergence checks. Must satisfy `max_iter > 3 * patience`.
        max_depth
            Maximum bisection steps for finding smallest eigenvalue (when auto-computing sigma).
        return_sigma
            If `True`, return (centrality_dict, sigma) tuple or (Graph, sigma) tuple instead of just
            dict or Graph.
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`domirank_centrality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        dict[int, float] or tuple[dict[int, float], float] or Graph or tuple[Graph, float]
            If `return_dict=` is `True` and `return_sigma` is `False`: dictionary mapping node IDs
            to dominance scores. If `return_dict` is `True` and `return_sigma` is `True`: tuple of
            (centrality dict, sigma value). If `return_dict` is `False` and `return_sigma` is
            `False`: returns self for method chaining. If `return_dict=` is `False` and
            `return_sigma` is `True`: tuple of (self, sigma value). Nodes with positive scores are
            dominant; negative scores indicate submissive nodes.

        Raises
        ------
        ValueError
            If alpha <= `0`, `dt` not in `(0, 1)`, tolerance <= `0`, or max_iter <= `3*patience`.

        Examples
        --------
        Auto-optimized DomiRank for a path graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node("A")
            .add_node("B")
            .add_node("C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Cycle
        )

        # Auto-optimize sigma based on graph structure
        domirank = graph.domirank_centrality(return_dict=True, )
        # Middle nodes often have high dominance scores

        # Get sigma value used
        domirank, sigma = graph.domirank_centrality(return_dict=True, return_sigma=True)
        print(f"Used sigma={sigma:.3f}")
        ```

        Manual sigma for specific competition level:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        # High competition (more global)
        dr_high = graph.domirank_centrality(return_dict=True, sigma=2.0)

        # Low competition (more local, like degree)
        dr_low = graph.domirank_centrality(return_dict=True, sigma=0.1)
        ```

        Notes
        -----
        - computational complexity: O(E × iterations) where E is edges
        - suitable for large graphs (tested on millions of nodes)
        - works for directed, undirected, weighted, and unweighted graphs
        - negative values indicate submissive nodes that yield to neighbors
        - joint dominance: Multiple nodes can co-dominate a neighborhood

        Algorithm notes:

        - uses Newton iteration with adaptive convergence detection
        - automatically finds optimal sigma via smallest eigenvalue
        - detects divergence via monotonic increase in updates

        %family Centrality
        """
        if method != "iterative":
            raise ValueError("Only method='iterative' is currently supported")

        # Import the Rust function
        from edgecase._core import domirank as rust_domirank

        # Call Rust implementation
        scores_by_id, final_sigma, converged = rust_domirank(
            self._graph,
            sigma,
            alpha,
            0.1,  # dt fixed at 0.1 for stability
            tolerance,
            max_iter,
            patience,
            max_depth,
        )

        if not converged:
            import warnings

            warnings.warn(
                "DomiRank did not converge within max_iter iterations. "
                "Consider increasing max_iter or adjusting tolerance.",
                RuntimeWarning,
            )

        if return_dict:
            if return_sigma:
                return scores_by_id, final_sigma
            return scores_by_id

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "domirank_centrality"

        self.set_node_attrs(attr_name, scores_by_id)

        # Track computation
        params = {}
        if sigma is not None:
            params["sigma"] = sigma
        if alpha != 0.95:
            params["alpha"] = alpha
        if tolerance != 1e-5:
            params["tolerance"] = tolerance
        if max_iter != 1000:
            params["max_iter"] = max_iter
        if patience != 10:
            params["patience"] = patience
        if max_depth != 50:
            params["max_depth"] = max_depth
        self._computations.append(("domirank_centrality", params, attr_name))

        if return_sigma:
            return self, final_sigma
        return self

    # ==============================================================================
    # Core and Structural Importance
    # ==============================================================================

    def closeness_vitality(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float], "Graph"]:
        """
        Get closeness vitality centrality for each node.

        Closeness vitality measures a node's importance by computing the change in total graph
        distances when the node is removed:

        `vitality(v) = sum_distances(G) - sum_distances(G - v)`

        Higher values indicate more important/central nodes whose removal would significantly
        increase average path lengths. Uses parallel BFS with Rayon for efficient computation.

        Parameters
        ----------
        attr_name
            Name for the node attribute to store results. If not provided, auto-generates name from
            method name (`closeness_vitality`).
        return_dict
            If `True`, returns dictionary of results. If `False`, stores results as node attributes
            and returns self for method chaining.

        Returns
        -------
        Union[dict[int, float], Graph]
            If `return_dict=` is `True`: dictionary mapping node IDs to vitality scores. If
            `return_dict=` is `False`: returns self for method chaining. Higher scores indicate more
            structurally important nodes.

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

        vitality = graph.closeness_vitality(return_dict=True)
        hub_id = graph.get_id_from_label("hub")
        a_id = graph.get_id_from_label("A")

        # Hub has highest vitality (removal disconnects graph)
        assert vitality[hub_id] > vitality[a_id]

        # Convert to labels if needed
        label_vitality = {graph.get_label_from_id(nid): score for nid, score in vitality.items()}
        ```

        Notes
        -----
        Computationally expensive for large graphs (O(V²E / num_cores) complexity). Consider
        alternative centrality measures (degree, PageRank) for graphs with >1000 nodes unless high
        accuracy is required.

        %family Core Decomposition
        %seealso harmonic_centrality, get_node_vitality, betweenness_centrality
        """
        vitality = self._graph.get_closeness_vitality()

        if return_dict:
            return vitality

        # Auto-generate attr_name if not provided
        if attr_name is None:
            attr_name = "closeness_vitality"

        self.set_node_attrs(attr_name, vitality)

        # Track computation (no non-default params)
        self._computations.append(("closeness_vitality", {}, attr_name))

        return self

    def coreness(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int], "Graph"]:
        """
        Calculate k-core decomposition (coreness for each node).

        The k-core is the maximal subgraph where every node has degree >= k. A node's coreness is
        the highest k-value for which it belongs to the k-core. Uses iterative pruning: repeatedly
        removes nodes with degree < k until all remaining nodes have degree >= k.

        Higher coreness indicates nodes in denser, more cohesive parts of the graph. The k-core
        decomposition is useful for finding the 'nucleus' of a network.

        Parameters
        ----------
        attr_name
            Name of the node attribute to store coreness values. If `None`, uses `coreness` as the
            attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to coreness values. If `False`, stores
            values as node attributes and returns the Graph for chaining.

        Returns
        -------
        dict[int, int] or Graph
            If `return_dict=` is `True`: Dictionary mapping node IDs to coreness values
            (non-negative integers). If `return_dict=` is `False`: Returns self for method chaining.

        Examples
        --------
        Calculate and store coreness:

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
            .coreness()  # Stores as 'coreness' attribute
        )
        # All nodes in K4 have coreness 3
        ```

        %family Core Decomposition
        %seealso k_core, k_shell, k_crust, k_truss, k_corona
        """
        if attr_name is None:
            attr_name = "coreness"

        coreness_values = self._graph.get_coreness()

        if return_dict:
            return coreness_values

        # Store as node attributes
        for node_id, value in coreness_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        self._computations.append(("coreness", {}, attr_name))

        return self

    def k_core(self, k: int) -> "Graph":
        """
        Extract the k-core subgraph.

        The k-core is the maximal subgraph where all nodes have degree >= k. Iteratively removes
        nodes with degree < k until all remaining nodes satisfy the condition.

        Parameters
        ----------
        k
            Minimum degree required for nodes to remain in the k-core.

        Returns
        -------
        Graph
            New graph containing only the k-core nodes and edges.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(6):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "4")
        g.add_edge("4", "5")

        # Get 2-core (nodes with degree >= 2)
        core_2 = g.k_core(2)
        print(f"Nodes in 2-core: {core_2.count_nodes()}")
        ```

        %family Core Decomposition
        %seealso coreness, k_shell, k_crust, k_corona
        """
        from edgecase._core import k_core

        result = Graph.__new__(Graph)
        result._graph = k_core(self._graph, k)
        return result

    def k_shell(self, k: int) -> "Graph":
        """
        Extract the k-shell subgraph.

        The k-shell contains nodes with coreness exactly equal to `k`. The coreness of a node is the
        highest `k` for which the node is part of the k-core.

        Parameters
        ----------
        k
            Coreness value to extract.

        Returns
        -------
        Graph
            New graph containing only the k-shell nodes and their connecting edges.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "4")

        # Get shell of nodes with coreness exactly 1
        shell_1 = g.k_shell(1)
        print(f"Nodes in 1-shell: {shell_1.count_nodes()}")

        %family Core Decomposition
        %seealso coreness, k_core, k_crust, k_corona
        ```
        """
        from edgecase._core import k_shell

        result = Graph.__new__(Graph)
        result._graph = k_shell(self._graph, k)
        return result

    def k_crust(self, k: int) -> "Graph":
        """
        Extract the k-crust subgraph.

        The k-crust contains all nodes with coreness <= `k`. This is the complement of nodes with
        high coreness values.

        Parameters
        ----------
        k
            Maximum coreness value to include.

        Returns
        -------
        Graph
            New graph containing only the k-crust nodes and their connecting edges.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "4")

        # Get nodes with coreness <= 2
        crust_2 = g.k_crust(2)
        print(f"Nodes in 2-crust: {crust_2.count_nodes()}")
        ```

        %family Core Decomposition
        %seealso coreness, k_core, k_shell, k_corona
        """
        from edgecase._core import k_crust

        result = Graph.__new__(Graph)
        result._graph = k_crust(self._graph, k)
        return result

    def k_truss(self, k: int) -> "Graph":
        """
        Extract the k-truss subgraph.

        The k-truss is the maximal subgraph where every edge is part of at least `(k-2)` triangles.
        Iteratively removes edges with insufficient triangle support.

        Parameters
        ----------
        k
            Minimum number of triangles + 2 for edges. Must be >= 3.

        Returns
        -------
        Graph
            New graph containing only the k-truss edges and their endpoints. Isolated nodes are
            removed.

        Raises
        ------
        ValueError
            If k < 3.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))
        # Create complete graph K4
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(str(i), str(j))

        # Get 4-truss (edges in >= 2 triangles)
        truss_4 = g.k_truss(4)
        print(f"Edges in 4-truss: {truss_4.count_edges()}")
        ```

        %family Core Decomposition
        %seealso coreness, k_core, count_triangles, list_triangles
        """
        from edgecase._core import k_truss

        result = Graph.__new__(Graph)
        result._graph = k_truss(self._graph, k)
        return result

    def k_corona(self, k: int) -> "Graph":
        """
        Extract the k-corona subgraph.

        The k-corona is the subgraph induced by nodes that are in the k-core and have exactly `k`
        neighbors in the k-core. These are the "boundary" nodes of the k-core that would be removed
        first when computing the `(k+1)`-core.

        The k-corona helps identify nodes at the periphery of dense substructures, which is useful
        for understanding hierarchical organization and vulnerability analysis.

        Parameters
        ----------
        k
            The core number. Nodes in the k-corona have coreness >= `k` and exactly `k` neighbors
            within the k-core.

        Returns
        -------
        Graph
            New graph containing only the k-corona nodes and their connecting edges. May be empty if
            no nodes satisfy the k-corona criteria.

        Examples
        --------
        Find the 2-corona in a simple graph:

        ```python
        from edgecase import Graph

        # Create a graph with a dense core and peripheral nodes
        g = Graph()
        # Dense core: nodes 1, 2, 3 form a triangle
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")

        # Add peripheral nodes
        g.add_node(label="4")
        g.add_node(label="5")
        g.add_edge("1", "4")
        g.add_edge("2", "5")

        # The 2-corona contains nodes in the 2-core with exactly 2 neighbors in 2-core
        corona_2 = g.k_corona(2)
        print(f"Nodes in 2-corona: {corona_2.count_nodes()}")
        ```

        Analyze hierarchical structure:

        ```python
        from edgecase import Graph

        # Build a graph and examine different corona levels
        g = Graph()
        g.add_full_graph(5)  # K5 complete graph

        # In K5, all nodes have degree 4, so they're in the 4-core
        # Each node has exactly 4 neighbors in the 4-core
        corona_4 = g.k_corona(4)
        print(f"4-corona nodes: {corona_4.count_nodes()}")  # All 5 nodes
        ```

        Notes
        -----
        - the k-corona is a subset of the k-shell (nodes with coreness exactly k)
        - nodes in the k-corona are those that would lose their k-core membership
          if any of their k-core neighbors were removed
        - the k-corona can be empty even when the k-core is non-empty
        - useful for identifying structurally vulnerable nodes in dense subgraphs

        %family Core Decomposition
        %seealso coreness, k_core, k_shell, k_crust
        """
        from edgecase._core import k_corona

        result = Graph.__new__(Graph)
        result._graph = k_corona(self._graph, k)
        return result

    def rich_club_coefficient(self) -> dict[int, float]:
        """
        Compute the rich-club coefficient for all degree values.

        The rich-club coefficient `φ(k)` measures the connectivity among high-degree nodes (the
        'rich club' of the network). For each degree `k`, it computes the ratio of actual edges to
        potential edges among nodes with degree greater than `k`:

        `\\phi(k) = \\frac{2E_k}{N_k(N_k - 1)}`

        where:
        - `N_k` is the number of nodes with degree > k
        - `E_k` is the number of edges among those nodes

        A rich-club coefficient close to `1` indicates that high-degree nodes tend to be connected
        to each other, forming a 'rich club'. Values less than expected in random graphs suggest the
        hubs avoid connecting to each other.

        Returns
        -------
        dict[int, float]
            Dictionary mapping degree `k` to rich-club coefficient `φ(k)`. Only degrees for
            which at least `2` nodes have degree > `k` are included (otherwise the coefficient
            is undefined).

        Examples
        --------
        Analyze hub connectivity in a network:

        ```python
        from edgecase import Graph

        # Create a network with clear hub structure
        graph = (
            Graph()
            .add_node(label="hub1")
            .add_node(label="hub2")
            .add_node(label="hub3")
            .add_node(label="spoke1")
            .add_node(label="spoke2")
            .add_node(label="spoke3")
            # Connect hubs to each other (rich club)
            .add_edge("hub1", "hub2")
            .add_edge("hub2", "hub3")
            .add_edge("hub3", "hub1")
            # Connect spokes to hubs
            .add_edge("hub1", "spoke1")
            .add_edge("hub2", "spoke2")
            .add_edge("hub3", "spoke3")
        )

        rc = graph.rich_club_coefficient()
        print(f"Rich club coefficients: {rc}")
        # Higher values at larger k indicate hubs prefer connecting to each other
        ```

        Compare with random network:

        ```python
        from edgecase import Graph

        # Scale-free networks often show rich-club phenomenon
        ba_graph = Graph().add_pa_graph(100, 3)
        rc = ba_graph.rich_club_coefficient()

        # Check if high-degree nodes are more interconnected than expected
        for k, phi in sorted(rc.items()):
            if phi > 0.5:
                print(f"Degree {k}: φ = {phi:.3f}")
        ```

        Notes
        -----
        - self-loops are ignored in the edge count
        - edge weights are ignored
        - for directed graphs, total degree (in + out) is used
        - returns unnormalized coefficients; for normalized values comparing to random networks,
        additional computation with degree-preserving random graphs would be needed

        %family Core Decomposition
        """
        from edgecase._core import rich_club_coefficient

        return rich_club_coefficient(self._graph)

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # PATHS & DISTANCE
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Distance and Path Metrics
    # ==============================================================================

    def get_mean_distance(self, unconnected: bool = True) -> float | None:
        """
        Get mean distance (average shortest path length).

        Computes average shortest path length between all reachable node pairs. Also known as
        average path length or characteristic path length.

        Parameters
        ----------
        unconnected
            If `True`, computes weighted average across all connected components. If `False`,
            returns `None` for disconnected graphs.

        Returns
        -------
        float
            Average shortest path length, or `None` if graph is disconnected (when
            `unconnected=False`).

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

        - small-world networks: low mean distance despite high clustering
        - scale-free networks: mean distance grows logarithmically with size
        - used in epidemiology: average infection spread time

        %family Distance and Paths
        """
        return self._graph.get_mean_distance_with_unconnected(unconnected)

    def get_min_eccentricity(self, unconnected: bool = True) -> int | None:
        """
        Get minimum eccentricity (graph radius).

        Alias for `get_radius()`. The minimum eccentricity is the smallest maximum distance from any
        node to all other nodes.

        Parameters
        ----------
        unconnected
            If `True`, computes minimum radius across all connected components. If `False`, returns
            `None` for disconnected graphs.

        Returns
        -------
        int | None
            Minimum eccentricity (radius), or `None` if disconnected (when `unconnected=False`).

        Examples
        --------
        Find the radius of a star graph:

        ```python
        graph = Graph().add_star_graph(5)

        radius = graph.get_min_eccentricity()
        print(radius)  # 1 (center reaches all nodes in 1 step)
        ```

        %family Distance and Paths
        """
        return self.get_radius(unconnected)

    def get_max_eccentricity(self, unconnected: bool = True) -> int | None:
        """
        Get maximum eccentricity (graph diameter).

        Alias for `get_diameter()`. The maximum eccentricity is the longest shortest path between
        any two nodes.

        Parameters
        ----------
        unconnected
            If `True`, computes maximum diameter across all connected components. If `False`,
            returns `None` for disconnected graphs.

        Returns
        -------
        int | None
            Maximum eccentricity (diameter), or `None` if disconnected (when `unconnected=False`).

        Examples
        --------
        Find the diameter of a path graph:

        ```python
        graph = Graph().add_path_graph(5)

        diameter = graph.get_max_eccentricity()
        print(diameter)  # 4 (end-to-end distance)
        ```

        %family Distance and Paths
        """
        return self.get_diameter(unconnected)

    def get_diameter(self, unconnected: bool = True) -> int | None:
        """
        Get graph diameter (maximum eccentricity).

        The diameter is the longest shortest path between any two nodes. This metric is useful for
        understanding the "width" of a network.

        Parameters
        ----------
        unconnected
            If `True`, computes maximum diameter across all connected components. If `False`,
            returns `None` for disconnected graphs.

        Returns
        -------
        int
            Maximum distance between any two nodes, or `None` if disconnected (when
            `unconnected=False`).

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

        print(graph.get_diameter())  # 3 (A to D)
        ```

        %family Distance and Paths
        """
        return self._graph.get_diameter_with_unconnected(unconnected)

    def get_radius(self, unconnected: bool = True) -> int | None:
        """
        Get graph radius (minimum eccentricity).

        The radius is the minimum of all node eccentricities. Nodes with eccentricity equal to the
        radius form the center of the graph.

        Parameters
        ----------
        unconnected
            If `True`, computes minimum radius across all connected components. If `False`, returns
            `None` for disconnected graphs.

        Returns
        -------
        int | None
            Minimum eccentricity, or `None` if disconnected (when `unconnected=False`).

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

        print(graph.get_radius())  # 1 (hub is center)
        ```

        %family Distance and Paths
        """
        return self._graph.get_radius_with_unconnected(unconnected)

    def get_girth(self) -> int | None:
        """
        Get girth (length of shortest cycle).

        The girth is the length of the shortest cycle in the graph. Returns `None` if the graph is
        acyclic (a forest/tree). Uses BFS from each node to detect shortest cycles efficiently.

        The implementation includes fast-path checks:

        - returns `1` for graphs with self-loops
        - returns `2` for undirected graphs with multi-edges
        - returns `3` or more for cycles found via BFS

        Returns
        -------
        int | None
            Length of shortest cycle, or `None` if acyclic.

            - 1: Graph contains a self-loop (edge from node to itself)
            - 2: Undirected graph contains multi-edges (duplicate edges)
            - 3+: Shortest cycle length found via BFS
            - `None`: Graph is acyclic (tree/forest)

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

        Self-loops have girth 1:

        ```python
        graph = Graph().add_node(label="A").add_edge("A", "A")
        print(graph.get_girth())  # 1
        ```

        Triangles have girth 3:

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
        print(graph.get_girth())  # 3
        ```

        Trees have no cycles:

        ```python
        tree = Graph(directed=True).add_balanced_tree(k=2, h=3)
        print(tree.get_girth())  # None
        ```

        %family Distance and Paths
        """
        return self._graph.get_girth()

    def eccentricity(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, int | None], "Graph"]:
        """
        Calculate eccentricity for each node in the graph.

        Eccentricity of a node is the maximum distance from that node to any other reachable node in
        the graph. It measures how far a node is from the most distant node it can reach. Returns
        `None` for nodes in disconnected graphs where some nodes are unreachable.

        Parameters
        ----------
        attr_name
            Name of the node attribute to store eccentricity values. If `None`, uses `eccentricity`
            as the attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to eccentricity values. If `False`,
            stores values as node attributes and returns the Graph for chaining.

        Returns
        -------
        dict[int, int | None] or Graph
            If `return_dict=` is `True`: Dictionary mapping node IDs to eccentricity values. Values
            are `None` for nodes in disconnected components. If `return_dict=` is `False`: Returns
            self for method chaining.

        Examples
        --------
        Calculate and store eccentricities:

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
            .eccentricity()  # Stores as 'eccentricity' attribute
        )
        # End nodes have max eccentricity (4)
        # Middle node C has min eccentricity (2)
        ```

        %family Distance and Paths
        """
        if attr_name is None:
            attr_name = "eccentricity"

        eccentricity_values = self._graph.get_eccentricity()

        if return_dict:
            return eccentricity_values

        # Store as node attributes
        for node_id, value in eccentricity_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        self._computations.append(("eccentricity", {}, attr_name))

        return self

    def radiality(
        self, attr_name: str | None = None, return_dict: bool = False
    ) -> Union[dict[int, float | None], "Graph"]:
        """
        Calculate radiality centrality for each node in the graph.

        Radiality measures how central a node is based on its eccentricity relative to the graph
        diameter. The formula is:

        `radiality(v) = (diameter + 1 - eccentricity(v)) / (n - 1)`

        where `n` is the number of nodes. Higher radiality indicates a more central position in the
        network. Returns `None` for nodes in disconnected graphs.

        Parameters
        ----------
        attr_name
            Name of the node attribute to store radiality values. If `None`, uses `radiality` as the
            attribute name.
        return_dict
            If `True`, returns a dictionary mapping node IDs to radiality scores. If `False`, stores
            values as node attributes and returns the Graph for chaining.

        Returns
        -------
        dict[int, float | None] or Graph
            If `return_dict=` is `True`: Dictionary mapping node IDs to radiality scores (`0.0` to
            `1.0`). Values are `None` for nodes in disconnected components. If `return_dict=` is
            `False` returns self for method chaining.

        Examples
        --------
        Calculate and store radiality:

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
            .radiality()  # Stores as 'radiality' attribute
        )
        # Hub has highest radiality (most central)
        # Leaves have lower radiality (peripheral)
        ```

        %family Distance and Paths
        """
        if attr_name is None:
            attr_name = "radiality"

        radiality_values = self._graph.get_radiality()

        if return_dict:
            return radiality_values

        # Store as node attributes
        for node_id, value in radiality_values.items():
            self._graph.set_node_attr(node_id, attr_name, value)

        # Track computation
        self._computations.append(("radiality", {}, attr_name))

        return self

    def bellman_ford_shortest_path(
        self,
        source: Union[int, str],
        target: Union[int, str],
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[int], "Graph", None]:
        """
        Find shortest path using Bellman-Ford algorithm.

        Computes the shortest weighted path from source to target using the Bellman-Ford algorithm.
        Unlike Dijkstra's algorithm, Bellman-Ford can handle negative edge weights and will detect
        negative weight cycles.

        Can return the path as a list, or store the position of each node along the path as a node
        attribute for visualization.

        The Bellman-Ford algorithm is slower than Dijkstra (O(VE) vs O(E log V)) but more versatile.
        Use it when:

        - the graph has negative edge weights
        - the graph is sparse, with few edges
        - you need to detect negative cycles

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        target
            Target node (integer ID or string label).
        attr_name
            Name of the node attribute to store path position. If `None`, uses
            `bellman_ford_path_order` as the attribute name. Position is 0-indexed (source = 0,
            next node = 1, ..., target = n-1). Only nodes on the path get this attribute set.
        return_list
            If `True`, returns list of node IDs in the path.
            If `False`, stores positions as node attributes and returns self for chaining.

        Returns
        -------
        Union[list[int], Graph, None]
            If `return_list=` is `True`: list of node IDs from source to target (inclusive),
            or `None` if no path exists.
            If `return_list=` is `False`: returns self for method chaining, or `None` if no path
            exists.

        Raises
        ------
        ValueError
            If the graph contains a negative weight cycle.

        Examples
        --------
        Find shortest path with negative weights:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=4.0)
            .add_edge("A", "C", weight=2.0)
            .add_edge("C", "B", weight=-3.0)  # Negative weight
            .add_edge("B", "D", weight=1.0)
        )

        path = graph.bellman_ford_shortest_path("A", "D", return_list=True)
        # Path: A -> C -> B -> D (takes negative weight edge)
        print(path)  # [1, 3, 2, 4] (node IDs)
        ```

        Store path as node attributes for visualization:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=4.0)
            .add_edge("A", "C", weight=2.0)
            .add_edge("C", "B", weight=-3.0)
            .add_edge("B", "D", weight=1.0)
            .bellman_ford_shortest_path("A", "D", return_list=False)
            .colorize_node_attrs(
                node_attr_from="bellman_ford_path_order",
                palette="plasma"
            )
        )
        ```

        Detect negative cycle:

        ```python
        bad_graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=-3.0)
            .add_edge("C", "A", weight=1.0)  # Creates negative cycle
        )

        try:
            path = bad_graph.bellman_ford_shortest_path("A", "B", return_list=True)
        except ValueError as e:
            print(e)  # "Graph contains a negative weight cycle"
        ```

        Handle unreachable target:

        ```python
        graph = Graph(directed=True)
        graph.add_node(label="A")
        graph.add_node(label="isolated")

        result = graph.bellman_ford_shortest_path("A", "isolated", return_list=True)
        print(result)  # None (no path exists)
        ```

        Notes
        -----
        **Algorithm complexity:**

        - time: `O(V × E)` where `V` is vertices, `E` is edges
        - space: `O(V)` for distance and predecessor arrays
        - slower than Dijkstra but handles negative weights

        **When to use:**

        - graph has negative edge weights (debt, cost reductions)
        - graph is sparse (few edges relative to nodes)
        - you need to detect negative cycles (arbitrage opportunities)

        **Applications:**

        - financial modeling (debt, arbitrage detection)
        - network routing with cost penalties
        - game theory (negative payoffs)
        - resource allocation with penalties

        %family Distance and Paths
        %seealso shortest_path, all_pairs_shortest_path, k_shortest_paths
        """
        source_id = self._resolve_node_id(source, auto_create=False)
        target_id = self._resolve_node_id(target, auto_create=False)

        result = rust.bellman_ford_shortest_path(self._graph, source_id, target_id)

        # If no path exists, return None regardless of return_list
        if result is None:
            return None

        path_ids, _distance = result

        if return_list:
            return path_ids

        # Store path positions as node attributes
        if attr_name is None:
            attr_name = "bellman_ford_path_order"

        # Create dict mapping node_id -> position in path
        positions = {node_id: idx for idx, node_id in enumerate(path_ids)}
        self.set_node_attrs(attr_name, positions)

        # Track computation
        params = {"source": source, "target": target}
        self._computations.append(("bellman_ford_shortest_path", params, attr_name))

        return self

    def shortest_path(
        self,
        source: Union[int, str],
        target: Union[int, str],
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[int], "Graph", None]:
        """
        Find shortest path between two nodes.

        Uses breadth-first search (BFS) to find the shortest path by number of hops. Can return the
        path as a list, or store the position of each node along the path as a node attribute for
        visualization.

        For weighted shortest paths, use Dijkstra's algorithm instead.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        target
            Target node (integer ID or string label).
        attr_name
            Name of the node attribute to store path position. If `None`, uses `shortest_path_order`
            as the attribute name. Only nodes on the path get this attribute set.
        return_list
            If `True`, returns list of node IDs in the path. If `False`, stores positions as node
            attributes and returns self for chaining.

        Returns
        -------
        Union[list[int], Graph, None]
            If `return_list=` is `True`: list of node IDs from source to target (inclusive), or
            `None` if no path exists. If `return_list=` is `False` it returns self for method
            chaining, or `None` if no path exists.

        Examples
        --------
        Find shortest path and get it as a list:

        ```python
        graph = Graph()
        graph.add_path_graph(5)  # Creates nodes 1-2-3-4-5

        # Get path as list (explicit)
        path = graph.shortest_path(1, 5, return_list=True)
        print(path)  # [1, 2, 3, 4, 5]

        # Works with labels too; it returns node IDs
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge(5, 6)
        graph.add_edge(6, 7)
        path = graph.shortest_path("A", "B", return_list=True)
        print(path)  # [6, 7]
        ```

        Store path as node attributes for visualization:

        ```python
        graph = (
            Graph()
            .add_grid_2d(rows=5, cols=5)
            .shortest_path(1, 25, return_list=False)
            .colorize_node_attrs(
                node_attr_from="shortest_path_order",
                palette="plasma"
            )
        )
        # Visualize the shortest path with gradient coloring
        ```

        Handle disconnected components:

        ```python
        graph = Graph()
        graph.add_path_graph(3)
        graph.add_node(label="isolated")
        iso_id = graph.get_id_from_label("isolated")

        # Returns None when no path exists
        path = graph.shortest_path(1, iso_id, return_list=True)
        print(path)  # None

        # With return_list=False, also returns None
        result = graph.shortest_path(1, iso_id)
        print(result)  # None
        ```

        Chain multiple shortest paths:

        ```python
        graph = (
            Graph()
            .add_grid_2d(rows=10, cols=10)
            .shortest_path(1, 50, attr_name="path1")
            .shortest_path(1, 100, attr_name="path2")
        )
        # Store multiple paths as different attributes
        ```

        %family Distance and Paths
        %seealso bellman_ford_shortest_path, all_pairs_shortest_path, k_shortest_paths, bfs
        """
        source_id = self._resolve_node_id(source, auto_create=False)
        target_id = self._resolve_node_id(target, auto_create=False)
        path_ids = rust.shortest_path(self._graph, source_id, target_id)

        # If no path exists, return None regardless of return_list
        if path_ids is None:
            return None

        if return_list:
            return path_ids

        # Store path positions as node attributes
        if attr_name is None:
            attr_name = "shortest_path_order"

        # Create dict mapping node_id -> position in path
        positions = {node_id: idx for idx, node_id in enumerate(path_ids)}
        self.set_node_attrs(attr_name, positions)

        # Track computation
        params = {"source": source, "target": target}
        self._computations.append(("shortest_path", params, attr_name))

        return self

    def ada_star_search(
        self,
        source: Union[int, str],
        target: Union[int, str],
        epsilon: float = 1.0,
        weight_attr: str = "weight",
        heuristic: Optional[Callable[[Union[int, str], Union[int, str]], float]] = None,
        reuse_planner: bool = False,
    ) -> list[int]:
        """
        Find path using ADA* (Anytime Dynamic A*) search algorithm.

        ADA* is an anytime optimal pathfinding algorithm that can return suboptimal paths quickly
        and iteratively improve them. The epsilon parameter controls the trade-off between solution
        quality and speed.

        Parameters
        ----------
        source
            Starting node ID or label.
        target
            Target node ID or label.
        epsilon
            Suboptimality bound. epsilon=1.0 finds optimal paths, epsilon>1.0 trades optimality for
            speed. Higher values explore less but may return longer paths.
        weight_attr
            Edge attribute to use as weight. Falls back to `edge.weight` if attribute not found.
        heuristic
            Custom heuristic function `h(node, target) -> float` that estimates cost from any node
            to target. Parameters can be node IDs (int) or labels (str). Must be admissible (never
            overestimate). If `None`, uses zero heuristic (equivalent to Dijkstra's algorithm).
        reuse_planner
            If `True`, reuses planner state from previous call with same source/target. Enables
            iterative refinement by calling with decreasing epsilon values. If `False`, creates
            fresh planner for each call.

        Returns
        -------
        list[int]
            Path from source to target as list of node IDs.

        Raises
        ------
        ValueError
            If source or target nodes don't exist, or if no path exists.

        Examples
        --------
        Find optimal path (equivalent to Dijkstra):

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", attributes={"weight": 1.0})
            .add_edge("A", "C", attributes={"weight": 4.0})
            .add_edge("B", "C", attributes={"weight": 2.0})
            .add_edge("C", "D", attributes={"weight": 1.0})
            .add_edge("B", "D", attributes={"weight": 5.0})
        )

        path = graph.ada_star_search("A", "D", epsilon=1.0)
        print(path)  # [1, 2, 3, 4] (node IDs)
        ```

        Fast suboptimal path with higher epsilon:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="A", target="C", weight=4.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="C", target="D", weight=1.0)
            .add_edge(source="B", target="D", weight=5.0)
        )

        # epsilon=5.0 explores less, finds path faster
        path = graph.ada_star_search(source="A", target="D", epsilon=5.0)
        # May return suboptimal path but much faster on large graphs
        ```

        Custom heuristic (e.g., Euclidean distance for spatial graphs):

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="A", x=0.0, y=0.0)
            .add_node(label="B", x=1.0, y=0.0)
            .add_node(label="C", x=1.0, y=1.0)
            .add_node(label="D", x=2.0, y=1.0)
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="A", target="C", weight=4.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="C", target="D", weight=1.0)
            .add_edge(source="B", target="D", weight=5.0)
        )

        # Heuristic using labels
        def euclidean_labels(node: str, target: str) -> float:
            n1 = graph.get_node_from_label(node)
            n2 = graph.get_node_from_label(target)
            x1, y1 = n1.attributes["x"], n1.attributes["y"]
            x2, y2 = n2.attributes["x"], n2.attributes["y"]
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        path = graph.ada_star_search(source="A", target="D", heuristic=euclidean_labels)

        # Heuristic using node IDs
        def euclidean_ids(node_id: int, target_id: int) -> float:
            n1 = graph.get_node(node_id)
            n2 = graph.get_node(target_id)
            x1, y1 = n1.attributes["x"], n1.attributes["y"]
            x2, y2 = n2.attributes["x"], n2.attributes["y"]
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        a_id = graph.get_id_from_label("A")
        d_id = graph.get_id_from_label("D")
        path = graph.ada_star_search(source=a_id, target=d_id, heuristic=euclidean_ids)
        ```

        Iterative refinement (anytime behavior):

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="A", target="C", weight=4.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="C", target="D", weight=1.0)
            .add_edge(source="B", target="D", weight=5.0)
        )

        # Quick suboptimal path first
        path1 = graph.ada_star_search(source="A", target="D", epsilon=5.0, reuse_planner=True)

        # Refine to better path (reuses previous work)
        path2 = graph.ada_star_search(source="A", target="D", epsilon=2.0, reuse_planner=True)

        # Get optimal path (reuses previous work)
        path3 = graph.ada_star_search(source="A", target="D", epsilon=1.0, reuse_planner=True)
        ```

        Notes
        -----
        - time complexity: `O(b^d)` where `b` is the branching factor, `d` is depth
        - space complexity: `O(b^d)` for storing search states
        - `epsilon=1.0` guarantees optimal paths (A* behavior)
        - `epsilon>1.0` provides bounded suboptimality with faster search
        - the algorithm performs backward search from target to source
        - set `reuse_planner=True` for iterative refinement with decreasing epsilon
        - planner state is stored in `graph._ada_star_planner` attribute
        - different source/target pairs automatically create separate planners

        %family Distance and Paths
        """
        # Check if we should reuse existing planner
        planner_key = (source, target, weight_attr)

        if reuse_planner and hasattr(self, "_ada_star_planner"):
            stored_planner, stored_key = self._ada_star_planner
            if stored_key == planner_key:
                # Reuse existing planner
                return stored_planner.compute_or_improve_path(epsilon)

        # Create new planner
        planner = AdaStar(
            self,
            source,
            target,
            weight=weight_attr,
            initial_epsilon=1000.0,
            heuristic=heuristic,
        )

        # Store planner if reuse requested
        if reuse_planner:
            self._ada_star_planner = (planner, planner_key)

        return planner.compute_or_improve_path(epsilon)

    def all_pairs_shortest_path_length(self) -> dict[int, dict[int, float]]:
        """
        Compute shortest path lengths for all pairs of nodes.

        Uses Dijkstra's algorithm from each source node to find the shortest path distance to all
        other reachable nodes. For unweighted graphs, all edges are treated as having weight `1.0`.

        Returns
        -------
        dict[int, dict[int, float]]
            Nested dictionary where `result[source][target] = distance`. Only includes reachable
            pairs. Unreachable pairs are omitted.

        Examples
        --------
        Simple path graph:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        distances = graph.all_pairs_shortest_path_length()
        print(distances[1][4])  # 3.0 (three hops)
        print(distances[2][4])  # 2.0 (two hops)
        ```

        Weighted graph:

        ```python
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="C", weight=2.0)
            .add_edge(source="A", target="C", weight=5.0)
        )

        distances = graph.all_pairs_shortest_path_length()
        a_id = graph.get_id_from_label("A")
        c_id = graph.get_id_from_label("C")
        print(distances[a_id][c_id])  # 3.0 (A->B->C is shorter than A->C)
        ```

        Disconnected components:

        ```python
        graph = Graph()
        graph.add_path_graph(3)  # 1-2-3
        graph.add_node(4)  # Isolated node

        distances = graph.all_pairs_shortest_path_length()
        print(1 in distances)  # True
        print(4 in distances[1])  # False (not reachable from 1)
        ```

        Notes
        -----
        Time complexity: `O(n * (m + n) * log(n))` where `n` is nodes and `m` is edges. Space
        complexity: `O(n²)` for storing all pairwise distances.

        For dense graphs or when you need all pairs, this is more efficient than calling
        `shortest_path()` repeatedly. For sparse queries, use `shortest_path()` instead.

        %family Distance and Paths
        """
        from edgecase._core import all_pairs_shortest_path_length

        return all_pairs_shortest_path_length(self._graph)

    def all_pairs_shortest_path(self) -> dict[int, dict[int, list[int]]]:
        """
        Compute shortest paths for all pairs of nodes.

        Uses Dijkstra's algorithm from each source node to find the shortest path to all other
        reachable nodes. Returns the full path as a list of node IDs.

        Returns
        -------
        dict[int, dict[int, list[int]]]
            Nested dictionary where `result[source][target] = path` (list of node IDs from source to
            target, inclusive). Only includes reachable pairs. Unreachable pairs are omitted.

        Examples
        --------
        Simple path graph:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(4)  # Creates 1-2-3-4

        paths = graph.all_pairs_shortest_path()
        print(paths[1][4])  # [1, 2, 3, 4]
        print(paths[2][4])  # [2, 3, 4]
        ```

        Weighted graph with multiple routes:

        ```python
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)
            .add_edge(source="A", target="C", weight=1.0)
            .add_edge(source="C", target="D", weight=5.0)
        )

        paths = graph.all_pairs_shortest_path()
        a_id = graph.get_id_from_label("A")
        d_id = graph.get_id_from_label("D")
        path = paths[a_id][d_id]
        # Returns [a_id, b_id, d_id] (shortest by weight)
        ```

        Distance matrix for centrality calculations:

        ```python
        graph = Graph()
        graph.add_star_graph(5)  # Star with center node 1

        paths = graph.all_pairs_shortest_path()
        node_ids = graph.get_node_ids()
        center = 1

        # Check paths from periphery to periphery go through center
        leaves = [n for n in node_ids if n != center]
        for source in leaves:
            for target in leaves:
                if source < target:
                    path = paths[source][target]
                    assert center in path  # Center node 1 is on all periphery-to-periphery paths
        ```

        Notes
        -----
        Time complexity: `O(n * (m + n) * log(n))` where `n` is nodes and `m` is edges. Space
        complexity: `O(n² * average_path_length)` for storing all paths.

        This method stores full paths, which can be memory intensive for large graphs. If you only
        need distances, use `all_pairs_shortest_path_length()` instead.

        %family Distance and Paths
        """
        from edgecase._core import all_pairs_shortest_path

        return all_pairs_shortest_path(self._graph)

    def k_shortest_paths(
        self,
        source: Union[int, str],
        target: Union[int, str],
        k: int,
    ) -> list[tuple[list[int], float]]:
        """
        Find k shortest paths between source and target using Yen's algorithm.

        Finds multiple alternative paths ranked by total path weight/length. Useful for finding
        backup routes, route diversity analysis, or identifying critical edges.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        target
            Target node (integer ID or string label).
        k
            Number of shortest paths to find. If fewer than `k` paths exist, returns all available
            paths.

        Returns
        -------
        list[tuple[list[int], float]]
            List of `(path, distance)` tuples sorted by distance (shortest first). Each path is a
            list of node IDs from source to target (inclusive). Returns empty list if no path
            exists or `k=0`.

        Examples
        --------
        Find alternative routes:

        ```python
        from edgecase import Graph

        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B", weight=1.0)
            .add_edge(source="B", target="D", weight=1.0)  # Route 1: A->B->D (cost 2.0)
            .add_edge(source="A", target="C", weight=1.5)
            .add_edge(source="C", target="D", weight=1.0)  # Route 2: A->C->D (cost 2.5)
            .add_edge(source="A", target="D", weight=4.0)  # Route 3: A->D (cost 4.0)
        )

        paths = graph.k_shortest_paths("A", "D", k=3)
        for i, (path, distance) in enumerate(paths, 1):
            labels = [graph.get_node(nid).label for nid in path]
            print(f"Route {i}: {' -> '.join(labels)} (distance: {distance})")
        # Route 1: A -> B -> D (distance: 2.0)
        # Route 2: A -> C -> D (distance: 2.5)
        # Route 3: A -> D (distance: 4.0)
        ```

        Unweighted graph (counts hops):

        ```python
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)
        # Grid:
        # 1 - 2 - 3
        # |   |   |
        # 4 - 5 - 6
        # |   |   |
        # 7 - 8 - 9

        paths = graph.k_shortest_paths(1, 9, k=5)
        print(f"Found {len(paths)} paths from corner to corner")
        print(f"Shortest path length: {paths[0][1]}")  # 4.0 hops
        ```

        Identify critical edges:

        ```python
        graph = (
            Graph(weighted=True)
            .add_node(label="start")
            .add_node(label="mid")
            .add_node(label="end")
            .add_edge(source="start", target="mid", weight=1.0)
            .add_edge(source="mid", target="end", weight=1.0)
        )

        paths = graph.k_shortest_paths("start", "end", k=2)
        if len(paths) == 1:
            print("Critical path - no alternative routes!")
        else:
            path1, dist1 = paths[0]
            path2, dist2 = paths[1]
            redundancy = (dist2 - dist1) / dist1 * 100
            print(f"Backup route is {redundancy:.1f}% longer")
        ```

        Notes
        -----
        Time complexity: `O(k * n * (m + n * log(n)))` where `n` is nodes, `m` is edges. Space
        complexity: `O(k * average_path_length)`.

        Uses Yen's algorithm which iteratively finds the next shortest path by systematically
        removing edges from previous paths. More efficient than finding all simple paths when `k`
        is small relative to the total number of paths.

        Paths are simple (no repeated nodes) and loop-free. For graphs with negative weights,
        consider using `bellman_ford_shortest_path()` for single shortest path instead.

        %family Distance and Paths
        """
        from edgecase._core import k_shortest_paths

        source_id = self._resolve_node_id(source, auto_create=False)
        target_id = self._resolve_node_id(target, auto_create=False)

        return k_shortest_paths(self._graph, source_id, target_id, k)

    def bfs(
        self,
        start: Union[int, str],
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[int], "Graph"]:
        """
        Perform breadth-first search (BFS) traversal from a start node.

        BFS explores the graph layer by layer, visiting all neighbors of a node before moving to
        their neighbors. Can return the visit order as a list, or store the position of each node in
        the traversal as a node attribute for visualization.

        Parameters
        ----------
        start
            Starting node (integer ID or string label).
        attr_name
            Name of the node attribute to store traversal position. If `None`, uses `bfs_order` as
            the attribute name. Position is 0-indexed (start node = `0`, first neighbor = `1`,
            etc.). Nodes not visited get no attribute set.
        return_list
            If `True`, returns list of node IDs in visit order. If `False`, stores positions as node
            attributes and returns self for chaining.

        Returns
        -------
        Union[list[int], Graph]
            If `return_list=` is `True`: list of node IDs visited in BFS order. If `return_list=` is
            `False`: returns self for method chaining.

        Examples
        --------
        BFS traversal from root of a tree:

        ```python
        graph = Graph()
        graph.add_balanced_tree(k=2, h=3)

        visited = graph.bfs(start=1)
        print(visited)  # [1, 2, 3, 4, 5, 6, 7, ...] (level by level)
        ```

        Store BFS order as node attributes for visualization:

        ```python
        graph = (
            Graph()
            .add_balanced_tree(k=2, h=3)
            .bfs(start=1, return_list=False)
            .rescale_node_attrs(
                node_attr_from="bfs_order",
                node_attr_to="size",
                range=[0.5, 2.0]
            )
        )
        # Root has position 0, visualize traversal depth
        ```

        BFS on a grid from corner:

        ```python
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)

        # Start from top-left corner
        visited = graph.bfs(start=1)
        # Visits neighbors layer by layer from start

        %family Distance and Paths
        %seealso dfs, shortest_path, random_walk
        ```
        """
        start_id = self._resolve_node_id(start, auto_create=False)
        visited_order = self._graph.bfs(start_id)

        if return_list:
            return visited_order

        # Store traversal positions as node attributes
        if attr_name is None:
            attr_name = "bfs_order"

        # Create dict mapping node_id -> position in traversal
        positions = {node_id: idx for idx, node_id in enumerate(visited_order)}
        self.set_node_attrs(attr_name, positions)

        # Track computation
        params = {"start": start}
        self._computations.append(("bfs", params, attr_name))

        return self

    def dfs(
        self,
        start: Union[int, str],
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[int], "Graph"]:
        """
        Perform depth-first search (DFS) traversal from a start node.

        DFS explores as far as possible along each branch before backtracking. Can return the visit
        order as a list, or store the position of each node in the traversal as a node attribute for
        visualization.

        Parameters
        ----------
        start
            Starting node (integer ID or string label).
        attr_name
            Name of the node attribute to store traversal position. If `None`, uses `dfs_order` as
            the attribute name. Position is 0-indexed (start node = `0`, first child = `1`, etc.).
            Nodes not visited get no attribute set.
        return_list
            If `True`, returns list of node IDs in visit order. If `False`, stores positions as node
            attributes and returns self for chaining.

        Returns
        -------
        Union[list[int], Graph]
            If `return_list=` is `True`: list of node IDs visited in DFS order. If `return_list=` is
            `False`: returns self for method chaining.

        Examples
        --------
        DFS traversal from root of a tree:

        ```python
        graph = Graph()
        graph.add_balanced_tree(k=2, h=3)

        visited = graph.dfs(start=1)
        print(visited)  # [1, 2, 4, 8, ...] (depth-first)
        ```

        Store DFS order as node attributes for visualization:

        ```python
        graph = (
            Graph()
            .add_balanced_tree(k=2, h=3)
            .dfs(start=1, return_list=False)
            .colorize_node_attrs(
                node_attr_from="dfs_order",
                palette="viridis"
            )
        )
        # Visualize depth-first traversal pattern
        ```

        DFS on a graph to explore connected component:

        ```python
        graph = Graph()
        graph.add_path_graph(5)

        visited = graph.dfs(start=1)
        print(visited)  # [1, 2, 3, 4, 5]
        ```

        %family Distance and Paths
        %seealso bfs, get_topological_sort_order, random_walk
        """
        start_id = self._resolve_node_id(start, auto_create=False)
        visited_order = self._graph.dfs(start_id)

        if return_list:
            return visited_order

        # Store traversal positions as node attributes
        if attr_name is None:
            attr_name = "dfs_order"

        # Create dict mapping node_id -> position in traversal
        positions = {node_id: idx for idx, node_id in enumerate(visited_order)}
        self.set_node_attrs(attr_name, positions)

        # Track computation
        params = {"start": start}
        self._computations.append(("dfs", params, attr_name))

        return self

    # ==============================================================================
    # Path and Topological Analysis
    # ==============================================================================

    def get_all_simple_paths(
        self,
        source: Union[int, str],
        target: Union[int, str],
        cutoff: Optional[int] = None,
    ) -> list[list[str]]:
        """
        Find all simple paths between two nodes.

        A simple path is a path with no repeated nodes. This method uses depth-first search with
        backtracking to enumerate all possible simple paths from source to target.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        target
            Target node (integer ID or string label).
        cutoff
            Maximum path length (number of edges) to consider. If `None`, no limit is applied.
            Useful for preventing excessive computation in large graphs.

        Returns
        -------
        list of lists of str
            List of paths, where each path is a list of node labels from source to target. Returns
            an empty list if no paths exist.

        Algorithm
        ---------
        - uses DFS with backtracking
        - time complexity: O(n!) worst case (exponential in path length)
        - space complexity: O(n) for recursion stack
        - best for: Finding alternative routes, analyzing network redundancy, path enumeration

        Examples
        --------
        Find all paths in a simple network:

        ```python
        # Create diamond-shaped graph: A->B->D, A->C->D
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )

        # Find all simple paths from A to D
        paths = graph.get_all_simple_paths("A", "D")
        for path in paths:
            print(" -> ".join(path))
        # Output:
        # A -> B -> D
        # A -> C -> D
        ```

        Limit path length with cutoff:

        ```python
        # Create longer chain with shortcut
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "E")
            .add_edge("A", "E")
        )

        # Find paths with max 2 edges
        short_paths = graph.get_all_simple_paths("A", "E", cutoff=2)
        print(f"Found {len(short_paths)} short paths")  # 1 path: A->E

        # Find all paths (no limit)
        all_paths = graph.get_all_simple_paths("A", "E")
        print(f"Found {len(all_paths)} total paths")  # 2 paths
        ```

        Analyze network redundancy:

        ```python
        # Communication network with redundant paths
        network = (
            Graph()
            .add_node(label="Server1")
            .add_node(label="Router1")
            .add_node(label="Router2")
            .add_node(label="Server2")
            .add_edge("Server1", "Router1")
            .add_edge("Server1", "Router2")
            .add_edge("Router1", "Server2")
            .add_edge("Router2", "Server2")
        )

        # Find all backup routes
        routes = network.get_all_simple_paths("Server1", "Server2")
        print(f"Network has {len(routes)}x redundancy")
        ```

        Use Cases
        ---------
        - **Network Redundancy**: Find backup routes in communication networks
        - **Social Network Analysis**: Discover indirect connections between people
        - **Logistics**: Identify alternative delivery routes
        - **Knowledge Graphs**: Trace reasoning paths between concepts
        - **Biology**: Analyze metabolic pathways with multiple routes

        Notes
        -----
        - for large graphs or long paths, the number of simple paths can be exponential
        - use the `cutoff` parameter to limit search depth and prevent excessive computation
        - for weighted graphs needing k-shortest paths, consider implementing Yen's algorithm
        - this method returns ALL paths, which can be memory-intensive for dense graphs

        %family Path Analysis
        """
        # Resolve node labels to internal IDs
        source_label = self.get_label_from_id(self._resolve_node_id(source, auto_create=False))
        target_label = self.get_label_from_id(self._resolve_node_id(target, auto_create=False))

        # Call Rust implementation
        paths = self._graph.all_simple_paths(source_label, target_label, cutoff)

        return paths

    def is_simple_path(self, nodes: list[Union[int, str]]) -> bool:
        """
        Check if a sequence of nodes forms a simple path in the graph.

        A simple path is a sequence where:

        1. all nodes exist in the graph
        2. each consecutive pair of nodes is connected by an edge
        3. no node appears more than once (no cycles)

        This is a fast `O(n)` validation method useful for checking user input, validating computed
        paths, or testing path properties.

        Parameters
        ----------
        nodes
            Sequence of node IDs or labels to validate.

        Returns
        -------
        bool
            `True` if the sequence forms a valid simple path, `False` otherwise.

        Algorithm
        ---------
        - time complexity: `O(n)` where n is the length of the node sequence
        - space complexity: `O(n)` for tracking visited nodes
        - checks performed: node existence, edge connectivity, no repetitions

        Examples
        --------
        Validate a simple path:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        # Valid simple path
        assert graph.is_simple_path(["A", "B", "C"]) == True

        # Invalid: no edge from A to C
        assert graph.is_simple_path(["A", "C"]) == False

        # Invalid: node D doesn't exist
        assert graph.is_simple_path(["A", "B", "D"]) == False
        ```

        Check for cycles:

        ```python
        # Create triangle
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # Valid simple path
        assert graph.is_simple_path(["A", "B", "C"]) == True

        # Invalid: A appears twice (cycle)
        assert graph.is_simple_path(["A", "B", "C", "A"]) == False
        ```

        Validate computed paths:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
        )

        # Get all paths and validate them
        paths = graph.get_all_simple_paths("A", "D")

        for path in paths:
            # All returned paths should be valid
            assert graph.is_simple_path(path) == True
            print(f"✓ Valid path: {' -> '.join(path)}")
        ```

        Edge cases:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )

        # Empty path is valid
        assert graph.is_simple_path([]) == True

        # Single node is valid (if it exists)
        assert graph.is_simple_path(["A"]) == True

        # Nonexistent single node is invalid
        assert graph.is_simple_path(["Z"]) == False
        ```

        Use Cases
        ---------
        - input validation: check user-provided paths before processing
        - path verification: validate paths returned by algorithms
        - route Checking: verify delivery or travel routes
        - graph Algorithms: test path properties in custom algorithms
        - debugging: verify intermediate results in path computations

        Notes
        -----
        - this method is much faster than calling `get_all_simple_paths()` and checking membership
        - for directed graphs, edges must exist in the specified direction
        - empty path and single-node paths are considered valid simple paths
        - compatible with both integer node IDs and string labels

        %family Path Analysis
        """
        # Convert node IDs to labels for Rust
        node_labels = []
        for node in nodes:
            try:
                node_id = self._resolve_node_id(node, auto_create=False)
                label = self.get_label_from_id(node_id)
                # If no label, use string of node ID
                if label is None:
                    label = str(node_id)
                node_labels.append(label)
            except (ValueError, KeyError):
                # Node doesn't exist
                return False

        # Call Rust implementation
        return self._graph.is_simple_path(node_labels)

    def random_walk(
        self,
        start: Union[int, str],
        steps: int,
        mode: str = "out",
        stuck: str = "return",
        seed: int | None = None,
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[int], "Graph"]:
        """
        Perform a random walk on the graph.

        Takes a random walk starting from a given node, choosing neighbors uniformly at random at
        each step. Can return the visit order as a list of node IDs, or store the position of each
        node in the walk as a node attribute for visualization.

        The walk follows edges according to the `mode` parameter:

        - `"out"`: follow outgoing edges (default for directed graphs)
        - `"in"`: follow incoming edges (reverse direction)
        - `"all"`: ignore edge directions (treat as undirected)

        If the walk reaches a node with no valid neighbors (gets 'stuck'), behavior is controlled by
        the `stuck=` parameter.

        Time complexity: `O(steps)`, plus `O(degree)` per step for neighbor lookup.

        Parameters
        ----------
        start
            Starting node (integer ID or string label).
        steps
            Number of steps to take in the walk.
        mode
            Direction to follow edges: (1) `"out"` to follow outgoing edges, (2) `"in"` to follow
            incoming edges, and (3) `"all"` to ignore edge direction.
        stuck
            What to do if the walk gets stuck. Use `"return"` to return the partial walk and use
            `"error"` to raise an error.
        seed
            Random seed for reproducibility. If `None`, uses system entropy.
        attr_name
            Name of the node attribute to store walk position. If `None`, uses `random_walk_order`
            as the attribute name. Position is 0-indexed (start node = `0`, next node = `1`, etc.).
            Nodes not visited get no attribute set.
        return_list
            If `True`, returns list of node IDs in walk order. If `False`, stores positions as node
            attributes and returns self for chaining.

        Returns
        -------
        Union[list[int], Graph]
            If `return_list=` is `True`: list of node IDs visited in walk order,
            of length `steps + 1` (includes the starting node). May be shorter
            if `stuck="return"` and the walk terminates early.
            If `return_list=` is `False`: returns self for method chaining.

        Raises
        ------
        ValueError
            If start node not found, or invalid mode/stuck parameter.
        RuntimeError
            If walk gets stuck and `stuck="error"`.

        Examples
        --------
        Random walk on a cycle:

        ```python
        graph = Graph()
        for i in range(1, 7):
            graph.add_node(label=str(i))
        for i in range(1, 7):
            graph.add_edge(str(i), str((i % 6) + 1))

        # Take a 10-step walk, returns node IDs
        path = graph.random_walk(1, steps=10, seed=23, return_list=True)
        print(len(path))  # 11 (including start)
        print(path[0])    # 1 (starting node ID)
        ```

        Store walk order as node attributes for visualization:

        ```python
        graph = (
            Graph()
            .add_cycle_graph(20)
            .random_walk(1, steps=100, return_list=False, seed=23)
            .colorize_node_attrs(
                node_attr_from="random_walk_order",
                palette="viridis"
            )
        )
        # Nodes colored by visit order in the walk
        ```

        Random walk on directed graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="A")
        )

        a_id = graph.get_id_from_label("A")

        # Follow edge directions
        path = graph.random_walk(a_id, steps=10, mode="out", seed=23, return_list=True)

        # Reverse directions
        path_reverse = graph.random_walk(a_id, steps=10, mode="in", seed=23, return_list=True)
        ```

        Handle stuck situations:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")  # C has no outgoing edges
        )

        a_id = graph.get_id_from_label("A")

        # Return partial walk when stuck
        path = graph.random_walk(a_id, steps=10, stuck="return", return_list=True)
        print(len(path))  # 3: node IDs for A, B, C

        # Or raise error
        try:
            path = graph.random_walk(a_id, steps=10, stuck="error", return_list=True)
        except RuntimeError as e:
            print("Got stuck!")
        ```

        Notes
        -----
        Random walks are fundamental to many graph algorithms:

        - PageRank and personalized PageRank
        - Node2Vec and DeepWalk embeddings
        - community detection (Walktrap)
        - graph sampling and exploration

        %family Path Analysis
        """
        start_id = self._resolve_node_id(start, auto_create=False)

        # Get node labels for Rust call (Rust implementation uses labels)
        start_label = self.get_label_from_id(start_id)
        walk_labels = self._graph.random_walk(start_label, steps, mode, stuck, seed)

        # Convert labels back to IDs
        walk_ids = [self.get_id_from_label(label) for label in walk_labels]

        if return_list:
            return walk_ids

        # Store walk positions as node attributes
        if attr_name is None:
            attr_name = "random_walk_order"

        # Create dict mapping node_id -> position in walk
        positions = {node_id: idx for idx, node_id in enumerate(walk_ids)}
        self.set_node_attrs(attr_name, positions)

        # Track computation
        params = {"start": start, "steps": steps, "mode": mode, "stuck": stuck, "seed": seed}
        self._computations.append(("random_walk", params, attr_name))

        return self

    def random_edge_walk(
        self,
        start: Union[int, str],
        steps: int,
        mode: str = "out",
        stuck: str = "return",
        seed: int | None = None,
        attr_name: str | None = None,
        return_list: bool = False,
    ) -> Union[list[tuple[int, int, int]], "Graph"]:
        """
        Perform a random walk and track the edges traversed.

        Like `random_walk()`, but operates on edges instead of nodes. Can return the sequence of
        edges as a list, or store the traversal order as an edge attribute for visualization.

        Time complexity: `O(steps)`, plus `O(degree)` per step for edge lookup.

        Parameters
        ----------
        start
            Starting node (integer ID or string label).
        steps
            Number of steps to take in the walk.
        mode
            Direction to follow edges: (1) `"out"` to follow outgoing edges, (2) `"in"` to follow
            incoming edges, and (3) `"all"` to ignore edge direction.
        stuck
            What to do if the walk gets stuck. Use `"return"` to return the partial walk and use
            `"error"` to raise an error.
        seed
            Random seed for reproducibility. If `None`, uses system entropy.
        attr_name
            Name of the edge attribute to store traversal order. If `None`, uses
            `random_edge_walk_order`. Only used when `return_list=False`.
        return_list
            If `True`, return list of edge tuples. If `False`, store traversal order as edge
            attributes and return Graph for chaining.

        Returns
        -------
        list[tuple[int, int, int]] | Graph
            If `return_list=True`: List of (edge_id, source_id, target_id) tuples. If
            `return_list=False`: Self for method chaining, with edge traversal order stored as edge
            attributes.

        Raises
        ------
        ValueError
            If start node not found, or invalid mode/stuck parameter.
        RuntimeError
            If walk gets stuck and `stuck="error"`.

        Examples
        --------
        Return edge list (explicit mode):

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        edges = graph.random_edge_walk("A", steps=5, seed=23, return_list=True)

        # Each edge is (edge_id, source_id, target_id)
        for edge_id, source_id, target_id in edges:
            print(f"Edge {edge_id}: {source_id} -> {target_id}")
        ```

        Store traversal order as edge attributes (default mode):

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        # Store traversal order, returns Graph for chaining
        graph.random_edge_walk("A", steps=100, seed=23)

        # Access traversal order from edge attributes
        orders = graph.get_edge_attrs_by_id("random_edge_walk_order")
        for edge_id, order in orders.items():
            print(f"Edge {edge_id} traversed at position {order}")
        ```

        Method chaining with multiple walks:

        ```python
        graph = (
            Graph()
            .add_cycle_graph(6)
            .random_edge_walk(1, steps=50, seed=23, attr_name="walk1")
            .random_edge_walk(1, steps=50, seed=123, attr_name="walk2")
        )

        # Compare which edges were used in each walk
        walk1_orders = graph.get_edge_attrs_by_id("walk1")
        walk2_orders = graph.get_edge_attrs_by_id("walk2")

        # Find edges present in both walks
        common_edges = set(walk1_orders.keys()) & set(walk2_orders.keys())
        for edge_id in sorted(common_edges)[:3]:  # Show first 3
            print(f"Edge {edge_id}: walk1={walk1_orders[edge_id]}, walk2={walk2_orders[edge_id]}")
        ```

        Analyze edge usage frequency:

        ```python
        from collections import Counter

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        edges = graph.random_edge_walk(start="A", steps=1000, seed=23, return_list=True)

        # Count edge usage (by source and target IDs)
        edge_counts = Counter((src_id, tgt_id) for _, src_id, tgt_id in edges)
        print("Most used edges:", edge_counts.most_common(3))
        ```

        Notes
        -----
        This is useful for:

        - analyzing edge importance via usage frequency
        - edge-centric graph algorithms
        - transition probability estimation
        - graph sketching and sampling
        - visualizing edge usage patterns

        %family Path Analysis
        """
        # Convert start to string for Rust (handles both int and str)
        start_str = str(start)
        edge_tuples = self._graph.random_edge_walk(start_str, steps, mode, stuck, seed)

        # Return list if requested
        if return_list:
            return edge_tuples

        # Store traversal order as edge attributes
        attr = attr_name if attr_name is not None else "random_edge_walk_order"

        # Create dict mapping edge_id -> position in traversal
        positions = {edge_id: idx for idx, (edge_id, _, _) in enumerate(edge_tuples)}

        # Set edge attributes
        self.set_edge_attrs(attr, positions)

        # Track computation
        params = {
            "start": start,
            "steps": steps,
            "mode": mode,
            "stuck": stuck,
            "seed": seed,
        }
        self._computations.append(("random_edge_walk", params, attr))

        return self

    def get_topological_sort_order(self) -> list[int]:
        """
        Sort nodes in topological order (DAG only).

        Returns a linear ordering of nodes such that for every directed edge `u -> v`, node `u`
        appears before `v` in the ordering. This is only possible for directed acyclic graphs
        (DAGs).

        Uses Kahn's algorithm (BFS-based) which has `O(V + E)` time complexity. The algorithm
        processes nodes with zero in-degree first, then removes them and updates in-degrees of
        neighbors.

        Returns
        -------
        list[int]
            node IDs in topological order.

        Raises
        ------
        ValueError
            If the graph is undirected or contains a cycle (not a DAG).

        Examples
        --------
        Sort tasks with dependencies:

        ```python
        # Build system: compile order for files with dependencies
        graph = (
            Graph(directed=True)
            .add_node(label="main.cpp")
            .add_node(label="utils.cpp")
            .add_node(label="utils.h")
            .add_node(label="config.h")
            .add_edge("main.cpp", "utils.h")     # main depends on utils.h
            .add_edge("main.cpp", "config.h")    # main depends on config.h
            .add_edge("utils.cpp", "utils.h")    # utils.cpp depends on utils.h
            .add_edge("utils.h", "config.h")     # utils.h depends on config.h
        )

        order_ids = graph.get_topological_sort_order()
        # Convert to labels for display
        order_labels = [graph.get_label_from_id(nid) for nid in order_ids]
        print("Build order:", " -> ".join(order_labels))
        # Output: config.h -> utils.h -> main.cpp -> utils.cpp (or similar valid order)
        ```

        Course prerequisites:

        ```python
        # University course dependencies
        graph = (
            Graph(directed=True)
            .add_node(label="Intro CS")
            .add_node(label="Data Structures")
            .add_node(label="Algorithms")
            .add_node(label="Machine Learning")
            .add_edge("Intro CS", "Data Structures")
            .add_edge("Data Structures", "Algorithms")
            .add_edge("Algorithms", "Machine Learning")
        )

        curriculum_ids = graph.get_topological_sort_order()
        for i, node_id in enumerate(curriculum_ids, 1):
            course = graph.get_label_from_id(node_id)
            print(f"Semester {i}: {course}")
        ```

        Detect cycles (not a DAG):

        ```python
        # Circular dependency
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Creates cycle
        )

        try:
            order = graph.get_topological_sort_order()
        except ValueError as e:
            print(f"Error: {e}")  # "Graph contains a cycle and not a DAG"
        ```

        Multiple valid orderings:

        ```python
        # Some graphs have multiple valid topological sorts
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "C")
            .add_edge("B", "C")
        )

        # Both [A, B, C] and [B, A, C] are valid
        order = graph.get_topological_sort_order()
        print(order)
        ```

        Notes
        -----
        Common use cases:

        - build systems and dependency resolution
        - task scheduling with prerequisites
        - package installation order
        - Makefile execution order
        - database migration ordering
        - course curriculum planning

        Implementation details:

        - uses Kahn's algorithm (BFS-based approach)
        - maintains in-degree count for each node
        - processes nodes with in-degree 0 first
        - detects cycles by checking if all nodes are processed

        %family Path Analysis
        """
        return self._graph.get_topological_sort_order()

    def transitive_closure(self) -> "Graph":
        """
        Compute the transitive closure of a directed graph.

        The transitive closure is a graph where there is an edge `(u, v)` if and only if there is a
        path from `u` to `v` in the original graph. This is useful for reachability analysis,
        determining which nodes can reach which other nodes.

        For example, if there's a path `A -> B -> C` in the original graph, the transitive closure
        will have edges `A -> B`, `B -> C`, and `A -> C`.

        This method only works on directed graphs. Undirected graphs should first be converted to
        directed or analyzed using connectivity methods.

        Time complexity: `O(V × (V + E))` where `V` is nodes and `E` is edges
        Space complexity: `O(V + E')` where `E'` is edges in the closure

        Returns
        -------
        Graph
            New directed graph representing the transitive closure.

        Raises
        ------
        ValueError
            If called on an undirected graph.

        Examples
        --------
        Basic transitive closure:

        ```python
        # Create a simple chain: A -> B -> C
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        closure = graph.transitive_closure()
        print(f"Original edges: {graph.count_edges()}")  # 2
        print(f"Closure edges: {closure.count_edges()}")  # 3 (adds A -> C)
        ```

        Reachability analysis:

        ```python
        # Dependency graph
        graph = (
            Graph(directed=True)
            .add_node(label="main")
            .add_node(label="utils")
            .add_node(label="config")
            .add_node(label="base")
            .add_edge("main", "utils")
            .add_edge("main", "config")
            .add_edge("utils", "base")
            .add_edge("config", "base")
        )

        closure = graph.transitive_closure()
        # In the closure, we can see all transitive dependencies
        # main can reach: utils, config, base
        print(f"Closure edges: {closure.count_edges()}")
        ```

        Notes
        -----
        Common use cases:

        - determining all reachable nodes from a given node
        - computing transitive dependencies in build systems
        - checking if one node can influence another in a network
        - preprocessing for constant-time reachability queries

        The transitive closure is the inverse operation of transitive reduction:
        ``transitive_closure(transitive_reduction(G)) == transitive_closure(G)``
        (for DAGs)

        %family Path Analysis
        """
        from edgecase._core import transitive_closure

        result = Graph.__new__(Graph)
        result._graph = transitive_closure(self._graph)
        return result

    def transitive_reduction(self) -> "Graph":
        """
        Compute the transitive reduction of a directed acyclic graph (DAG).

        The transitive reduction is the minimal subgraph that maintains the same reachability as the
        original graph. It removes redundant edges while preserving all path relationships. If
        there's a path from `A` to `C` via `B`, the direct edge `A -> C` (if it exists) is redundant
        and will be removed.

        This method only works on directed acyclic graphs (DAGs). Cyclic graphs do not have a unique
        transitive reduction.

        Time complexity: `O(V × E)` where `V` is nodes and `E` is edges
        Space complexity: `O(V + E)`

        Returns
        -------
        Graph
            New directed graph representing the transitive reduction.

        Raises
        ------
        ValueError
            If called on an undirected graph or a graph containing cycles.

        Examples
        --------
        Remove redundant edges:

        ```python
        # Graph with redundant edge A -> C (since A -> B -> C exists)
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("A", "C")  # Redundant: A can reach C via B
        )

        reduction = graph.transitive_reduction()
        print(f"Original edges: {graph.count_edges()}")  # 3
        print(f"Reduced edges: {reduction.count_edges()}")  # 2 (A->C removed)
        ```

        Simplify a dependency graph:

        ```python
        # Build dependency graph with some direct dependencies that
        # are implied by transitive dependencies
        graph = (
            Graph(directed=True)
            .add_node(label="app")
            .add_node(label="lib")
            .add_node(label="utils")
            .add_node(label="core")
            .add_edge("app", "lib")
            .add_edge("lib", "utils")
            .add_edge("utils", "core")
            .add_edge("app", "core")   # Redundant (app -> lib -> utils -> core)
            .add_edge("lib", "core")   # Redundant (lib -> utils -> core)
        )

        reduction = graph.transitive_reduction()
        print(f"Original: {graph.count_edges()} edges")  # 5
        print(f"Reduced: {reduction.count_edges()} edges")  # 3 (minimal)
        ```

        Handling cyclic graphs:

        ```python
        # Cyclic graph
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
            .add_edge("B", "A")  # Creates cycle
        )

        try:
            reduction = graph.transitive_reduction()
        except ValueError as e:
            print(f"Error: {e}")  # Graph contains a cycle
        ```

        Notes
        -----
        Common use cases:

        - simplifying dependency graphs to show only direct dependencies
        - reducing storage for reachability queries
        - visualizing the minimal structure of a DAG
        - cleaning up inferred relationships

        Properties:

        - the transitive reduction of a DAG is unique
        - the reduction preserves all reachability relationships

        %family Path Analysis
        """
        from edgecase._core import transitive_reduction

        result = Graph.__new__(Graph)
        result._graph = transitive_reduction(self._graph)
        return result

    def count_walks(self, walk_length: int) -> dict[int, dict[int, int]]:
        """
        Count walks of a specific length between all pairs of nodes.

        A walk is a sequence of nodes where each adjacent pair is connected by an edge. Unlike
        paths, walks can repeat nodes and edges. The number of walks of length `k` from node `i` to
        node `j` equals the `(i,j)` entry of `A^k` where `A` is the adjacency matrix.

        This method computes the *k*-th power of the adjacency matrix using fast matrix
        exponentiation, which is efficient even for large k values.

        Time complexity: `O(V³ × log k)` where `V` is the number of nodes
        Space complexity: `O(V²)`

        Parameters
        ----------
        walk_length : int
            Non-negative integer representing the number of edges in the walk. A walk of length 0
            means staying at the same node (identity matrix). A walk of length 1 counts direct
            edges.

        Returns
        -------
        dict[int, dict[int, int]]
            Nested dictionary where outer keys are source node IDs, inner keys are target node IDs,
            and inner values are the number of walks of length `walk_length=` from source to target.

        Raises
        ------
        ValueError
            If walk_length is negative.

        Examples
        --------
        Count walks in a simple path:

        ```python
        g = Graph()
        g.add_node(label="0")
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_edge("0", "1")
        g.add_edge("1", "2")

        # Walks of length 2
        walks = g.count_walks(2)
        node_0 = g.get_id_from_label("0")
        node_1 = g.get_id_from_label("1")
        node_2 = g.get_id_from_label("2")
        print(walks[node_0][node_2])  # 1 (path: 0 -> 1 -> 2)
        print(walks[node_1][node_1])  # 2 (paths: 1 -> 0 -> 1 and 1 -> 2 -> 1)
        ```

        Count walks in a triangle:

        ```python
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1")
        g.add_edge("1", "2")
        g.add_edge("2", "0")

        # Walks of length 3 (can return to start)
        walks = g.count_walks(3)
        node_0 = g.get_id_from_label("0")
        print(walks[node_0][node_0])  # 2 (two ways to return: 0->1->2->0 and 0->1->0->0)
        ```

        Total number of walks in the graph:

        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge(source="0", target="1")
        g.add_edge(source="1", target="2")
        g.add_edge(source="2", target="0")

        walks = g.count_walks(walk_length=2)
        total = sum(
            sum(targets.values()) for targets in walks.values()
        )
        print(f"Total walks of length 2: {total}")
        ```

        Walks from a specific source:

        ```python
        from edgecase import Graph

        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge(source="0", target="1")
        g.add_edge(source="1", target="2")
        g.add_edge(source="2", target="0")

        walks = g.count_walks(walk_length=3)
        node_0 = g.get_id_from_label("0")
        source_walks = walks[node_0]
        print(f"Walks from node 0: {sum(source_walks.values())}")
        ```

        Notes
        -----
        **Walk vs. Path**:

        - a *walk* can repeat nodes and edges (counted by this method)
        - a *path* cannot repeat nodes (use shortest path algorithms)
        - a *trail* cannot repeat edges but can repeat nodes

        **Walk length 0**:
        Returns the identity matrix where each node has exactly 1 "walk" to itself
        (staying in place) and 0 walks to other nodes.

        **Walk length 1**:
        Returns the adjacency matrix where counts direct edges. For multigraphs,
        this counts multiple edges between the same node pair.

        **Growth of walk counts**:
        The number of walks can grow very quickly, especially in dense graphs
        and with larger walk lengths. For example, in a complete graph K_n,
        the number of walks grows exponentially with walk_length.

        **Applications**:

        - **Network reachability**: Measure connectivity strength beyond shortest paths
        - **Information diffusion**: Model how information can spread through multiple redundant
        paths
        - **Random walk analysis**: Expected number of random walks between nodes
        - **Graph similarity**: Compare graphs based on walk structure
        - **Spectral analysis**: Related to eigenvalues of the adjacency matrix

        **Relationship to matrix powers**:

        This method computes `A^k` where `A` is the adjacency matrix. The trace of `A^k` (sum of
        diagonal elements) counts closed walks (cycles) of length `k`.

        %family Path Analysis
        """
        if walk_length < 0:
            raise ValueError("walk_length must be non-negative")

        return self._graph.count_walks(walk_length)

    def count_trails(
        self,
        source: Union[int, str],
        target: Union[int, str],
        max_trails: int | None = None,
    ) -> int | None:
        """
        Count all trails (walks with distinct edges) from source to target.

        A *trail* is a walk where all edges are distinct, but nodes may repeat. This is different
        from:

        - *path*: no repeated nodes or edges (most restrictive)
        - *trail*: no repeated edges, but nodes may repeat
        - *walk*: nodes and edges may repeat freely (least restrictive)

        Uses iterative depth-first search with edge tracking. Can grow exponentially for dense
        graphs, so use `max_trails=` parameter to limit computation.

        Parameters
        ----------
        source
            Source node ID (int) or label (str).
        target
            Target node ID (int) or label (str).
        max_trails
            Maximum number of trails to count before stopping. If `None`, counts all trails. If
            exceeded, returns `None`.

        Returns
        -------
        int | None
            Number of distinct trails from source to target. Returns `None` if `max_trails=`
            exceeded. Returns `1` if source == target (trivial empty trail).

        Raises
        ------
        ValueError
            If source or target node not found in graph.

        Examples
        --------
        Count trails using labels:

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

        # Triangle: multiple trails from A to C
        count = graph.count_trails("A", "C")
        print(f"Trails from A to C: {count}")  # 2 trails: A->C and A->B->C
        ```

        Count trails using node IDs:

        ```python
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        # Get node IDs
        a_id = graph.get_id_from_label("A")
        b_id = graph.get_id_from_label("B")

        # Use IDs directly
        count = graph.count_trails(a_id, b_id)
        print(f"Found {count} trails")
        ```

        Count trails with limit to prevent explosion:

        ```python
        from edgecase import karate_club
        G = karate_club()

        # Get actual node IDs (karate_club uses integer IDs)
        node_ids = list(G.get_node_ids())
        first_node = node_ids[0]
        last_node = node_ids[-1]

        # Limit to 10000 trails to avoid long computation
        count = G.count_trails(first_node, last_node, max_trails=10000)
        if count is None:
            print("More than 10,000 trails exist")
        else:
            print(f"Found {count} trails")
        ```

        Compare paths vs trails:

        ```python
        # Create graph with cycle
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("B", "D")  # Shortcut
        )

        # Trails: can use different edges
        trails = graph.count_trails("A", "D")
        print(f"Trails: {trails}")  # 2: A->B->D and A->B->C->D

        # Simple paths would only count non-repeated nodes
        ```

        Notes
        -----
        **Complexity**:

        - Time: O(trails × avg_trail_length) - exponential in worst case
        - Space: O(edges) per trail in exploration stack

        **Growth Rate**:

        - triangle (3 nodes): ~2 trails
        - complete K4 (4 nodes): ~12 trails
        - complete K5 (5 nodes): ~100+ trails
        - can easily exceed millions for dense graphs

        **Implementation**:

        - uses iterative DFS with explicit stack (avoids recursion limits)
        - tracks visited edges (not nodes) to allow node revisiting
        - for undirected graphs, edges are bidirectional

        %family Path Analysis
        """
        # Convert string node IDs to internal IDs
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)

        # Call Rust implementation
        result = self._graph.count_trails(source_id, target_id, max_trails)
        return result

    def enumerate_trails(
        self,
        source: Union[int, str],
        target: Union[int, str],
        max_trails: int | None = 1000,
    ) -> list[list[str]]:
        """
          Enumerate all trails (walks with distinct edges) from source to target.

          Returns the actual sequences of nodes for each trail. A trail is a walk where all edges are
          distinct, though nodes may repeat.

          Parameters
          ----------
          source
              Source node ID (int) or label (str).
          target
              Target node ID (int) or label (str).
          max_trails
              Maximum number of trails to return. Default is `1000`. If `None`, returns all trails
              (dangerous for large graphs). If limit exceeded, returns partial list.

          Returns
          -------
          list[list[str]]
              List of trails, where each trail is a sequence of node IDs. If `max_trails=` exceeded,
              returns first `max_trails=` trails found.

          Raises
          ------
          ValueError
              If source or target node not found in graph.

          Examples
          --------
          Enumerate trails using labels:

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

          trails = graph.enumerate_trails("A", "C")
          for i, trail in enumerate(trails, 1):
              print(f"Trail {i}: {' -> '.join(trail)}")
          # Trail 1: A -> C
          # Trail 2: A -> B -> C
          ```

          Enumerate using node IDs:

          ```python
          graph = Graph()
          graph.add_node(label="start")
          graph.add_node(label="end")
          graph.add_edge("start", "end")

          # Get node IDs
          start_id = graph.get_id_from_label("start")
          end_id = graph.get_id_from_label("end")

          # Use IDs directly
          trails = graph.enumerate_trails(start_id, end_id)
          print(f"Found {len(trails)} trails")
          ```

          Enumerate with limit:

          ```python
          from edgecase import Graph

          graph = (
              Graph()
              .add_node(label="start")
              .add_node(label="middle")
              .add_node(label="end")
              .add_edge(source="start", target="middle")
              .add_edge(source="middle", target="end")
              .add_edge(source="start", target="end")
          )

          # Get first 100 trails only
          trails = graph.enumerate_trails(source="start", target="end", max_trails=100)
          print(f"Found {len(trails)} trails (may be more)")
          ```

          Analyze trail lengths:

          ```python
          from edgecase import karate_club

          G = karate_club()

          # Karate club has nodes 0-33, get any two node IDs
          node_ids = list(G.get_node_ids(limit=None))
          if len(node_ids) >= 2:
              source_id = node_ids[0]
              target_id = node_ids[-1]
              source_label = G.get_label_from_id(node_id=source_id) or str(source_id)
              target_label = G.get_label_from_id(node_id=target_id) or str(target_id)

              # Get sample of trails
              trails = G.enumerate_trails(source=source_label, target=target_label, max_trails=100)

              if trails:
                  # Analyze lengths
                  lengths = [len(trail) - 1 for trail in trails]  # -1 for edge count
                  print(f"Trail lengths: min={min(lengths)}, max={max(lengths)}")
                  print(f"Average edges per trail: {sum(lengths) / len(lengths):.1f}")
          ```

          Notes
          -----
          **Memory Warning**:

          - Each trail stores full node sequence
          - 1000 trails × 50 nodes each = ~50KB minimum
          - Can quickly consume many GBs of memory for large graphs
          - Always set max_trails to reasonable value

          **Use Cases**:

          - Analyzing alternative routes in networks
          - Finding redundant paths for reliability
          - Network vulnerability analysis
          - Understanding connectivity patterns

        %family Path Analysis
        """
        # Convert string node IDs to internal IDs
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)

        # Call Rust implementation
        trails = self._graph.enumerate_trails(source_id, target_id, max_trails)

        # Convert internal IDs back to string labels
        result = []
        for trail in trails:
            result.append([self.get_label_from_id(node_id) or str(node_id) for node_id in trail])

        return result

    # ==============================================================================
    # Eulerian Paths and Circuits
    # ==============================================================================

    def is_eulerian(self) -> bool:
        """
        Check if the graph has an Eulerian circuit.

        An Eulerian circuit is a closed trail that visits every edge exactly once.

        For undirected graphs, an Eulerian circuit exists if and only if:

        - the graph is connected (ignoring isolated nodes)
        - all vertices have even degree

        For directed graphs, an Eulerian circuit exists if and only if:

        - the graph is weakly connected
        - every vertex has equal in-degree and out-degree

        Returns
        -------
        bool
            `True` if the graph has an Eulerian circuit, `False` otherwise.

        Examples
        --------
        Check a cycle graph (has Eulerian circuit):

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(5)

        print(graph.is_eulerian())  # True (all degrees are 2)
        ```

        Check a path graph (no Eulerian circuit):

        ```python
        graph = Graph()
        graph.add_path_graph(5)

        print(graph.is_eulerian())  # False (endpoints have odd degree)
        ```

        Check a directed graph:

        ```python
        graph = Graph(directed=True)
        graph.add_cycle_graph(4)

        print(graph.is_eulerian())  # True (in-degree = out-degree for all)
        ```

        %family Eulerian Paths
        """
        from edgecase._core import is_eulerian

        return is_eulerian(self._graph)

    def is_semieulerian(self) -> bool:
        """
        Check if the graph has an Eulerian path but not an Eulerian circuit.

        An Eulerian path is a trail that visits every edge exactly once. A graph is semi-Eulerian if
        it has an Eulerian path but NOT an Eulerian circuit.

        For undirected graphs, this occurs when:

        - the graph is connected
        - exactly two vertices have odd degree (start and end of path)

        For directed graphs, this occurs when:

        - the graph is weakly connected
        - exactly one vertex has out-degree - in-degree = 1 (start)
        - exactly one vertex has in-degree - out-degree = 1 (end)
        - all other vertices have equal in-degree and out-degree

        Returns
        -------
        bool
            `True` if the graph has an Eulerian path but not circuit, `False` otherwise.

        Examples
        --------
        Check a path graph (semi-Eulerian):

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(5)

        print(graph.is_semieulerian())  # True (2 odd-degree nodes)
        print(graph.is_eulerian())      # False (not a circuit)
        ```

        Check a cycle graph (not semi-Eulerian):

        ```python
        graph = Graph()
        graph.add_cycle_graph(5)

        print(graph.is_semieulerian())  # False (has Eulerian circuit)
        print(graph.is_eulerian())      # True
        ```

        Check a directed path:

        ```python
        graph = Graph(directed=True)
        graph.add_path_graph(5)

        print(graph.is_semieulerian())  # True (one source, one sink)
        ```

        %family Eulerian Paths
        """
        from edgecase._core import is_semieulerian

        return is_semieulerian(self._graph)

    def eulerian_circuit(self) -> Optional[list[int]]:
        """
        Find an Eulerian circuit in the graph if one exists.

        An Eulerian circuit is a closed trail that visits every edge exactly once, starting and
        ending at the same node. Uses Hierholzer's algorithm.

        Returns
        -------
        Optional[list[int]]
            List of node IDs forming the Eulerian circuit, or `None` if no circuit exists. The
            circuit starts and ends at the same node.

        Examples
        --------
        Find a circuit in a cycle graph:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(5)

        circuit = graph.eulerian_circuit()
        print(circuit)  # e.g., [0, 1, 2, 3, 4, 0]
        print(len(circuit))  # 6 (5 edges + return to start)
        ```

        No circuit in a path graph:

        ```python
        graph = Graph()
        graph.add_path_graph(5)

        circuit = graph.eulerian_circuit()
        print(circuit)  # None (path has no circuit)
        ```

        Find circuit in a cycle graph:

        ```python
        graph = Graph()
        graph.add_cycle_graph(6)

        circuit = graph.eulerian_circuit()
        if circuit:
            print(f"Circuit length: {len(circuit)}")
            print(f"Visits {len(circuit) - 1} edges")
        ```

        Directed Eulerian circuit:

        ```python
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        circuit = graph.eulerian_circuit()
        print(len(circuit))  # 6 (visits 5 edges + returns to start)
        ```

        Notes
        -----
        - Uses Hierholzer's algorithm: O(E) time complexity
        - Only returns a circuit if one exists (use is_eulerian() to check first)
        - For semi-Eulerian graphs (path but no circuit), use a pathfinding approach
        - The returned circuit is one of potentially many valid circuits

        %family Eulerian Paths
        """
        from edgecase._core import eulerian_circuit

        return eulerian_circuit(self._graph)

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # CONNECTIVITY
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Bridges and Connectivity
    # ==============================================================================

    def bridges(self) -> list[int]:
        """
        Find all bridge edges in the graph.

        A bridge (also called a cut edge) is an edge whose removal increases the number of connected
        components. Bridges are critical edges that, if removed, would disconnect parts of the
        network.

        Uses Tarjan's algorithm for undirected graphs (`O(V + E)` time complexity) and strongly
        connected component analysis for directed graphs.

        Returns
        -------
        list[int]
            List of edge IDs that are bridges. Empty list if no bridges exist.

        Examples
        --------
        Find bridges in a simple graph:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(4)

        bridges = graph.bridges()
        print(len(bridges))  # 3 (all edges are bridges in a path)
        ```

        Find bridges in a graph with a cycle:

        ```python
        graph = Graph()
        graph.add_cycle_graph(4)  # 0-1-2-3-0
        graph.add_node(label="tail")
        tail_id = graph.get_id_from_label("tail")
        nodes = sorted(graph.get_node_ids())
        graph.add_edge(nodes[-1], tail_id)  # Add a tail

        bridges = graph.bridges()
        print(len(bridges))  # 1 (only edge to tail is a bridge)
        ```

        Check bridge edges in a network:

        ```python
        graph = Graph()
        graph.add_path_graph(3)

        bridge_ids = graph.bridges()
        for edge_id in bridge_ids:
            result = graph.get_node_ids_by_edge_id(edge_id)
            if result:
                u, v = result
                print(f"Bridge: {u} - {v}")
        ```

        Notes
        -----
        - bridges are critical for network vulnerability analysis
        - in a tree, all edges are bridges
        - in a cycle, no edges are bridges
        - related to articulation points (vertices whose removal disconnects the graph)

        %family Bridges and Connectivity
        """
        from edgecase._core import bridges

        return bridges(self._graph)

    def has_bridges(self) -> bool:
        """
        Check if the graph contains any bridge edges.

        A bridge (cut edge) is an edge whose removal increases the number of connected components.
        This method is more efficient than `bridges()` when you only need to know if bridges exist,
        not which edges they are.

        Returns
        -------
        bool
            `True` if at least one bridge exists, `False` otherwise.

        Examples
        --------
        Check if a cycle has bridges:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(5)

        print(graph.has_bridges())  # False (cycles have no bridges)
        ```

        Check if a path has bridges:

        ```python
        graph = Graph()
        graph.add_path_graph(5)

        print(graph.has_bridges())  # True (all tree edges are bridges)
        ```

        Check a mixed graph:

        ```python
        graph = Graph()
        graph.add_cycle_graph(4)  # Dense component
        graph.add_node(label="isolated")
        isolated_id = graph.get_id_from_label("isolated")
        nodes = sorted(graph.get_node_ids())
        graph.add_edge(nodes[-2], isolated_id)  # Bridge to isolated node

        if graph.has_bridges():
            bridges = graph.bridges()
            print(f"Found {len(bridges)} bridge(s)")
        ```

        Notes
        -----
        - more efficient than `len(bridges()) > 0` for large graphs
        - useful for quick connectivity analysis
        - returns `False` for complete graphs and cycles

        %family Bridges and Connectivity
        """
        from edgecase._core import has_bridges

        return has_bridges(self._graph)

    def get_node_vitality(self, node: Union[int, str]) -> float:
        """
        Compute the vitality of a single node.

        Node vitality measures the impact of removing a node on graph connectivity, defined as the
        change in sum of all pairwise shortest path distances:

        vitality(v) = sum_distances(G) - sum_distances(G - v)

        Higher values indicate more important nodes whose removal would significantly increase
        average path lengths or disconnect the graph. Infinite values indicate the node's removal
        disconnects the graph.

        Parameters
        ----------
        node
            Node ID (int) or label (str) to compute vitality for.

        Returns
        -------
        float
            The vitality score for the node. Higher values indicate more vital nodes. Returns `inf`
            if removing the node disconnects the graph.

        Raises
        ------
        ValueError
            If the node does not exist in the graph.

        Examples
        --------
        Compute vitality for nodes in a path graph:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(5)
        nodes = sorted(graph.get_node_ids())

        # Middle nodes are most vital (removal disconnects)
        middle_vitality = graph.get_node_vitality(nodes[2])
        end_vitality = graph.get_node_vitality(nodes[0])

        print(f"Middle node vitality: {middle_vitality}")
        print(f"End node vitality: {end_vitality}")
        # Middle has higher vitality (or infinite if it disconnects)
        ```

        Compare vitality in a star graph:

        ```python
        graph = Graph()
        graph.add_star_graph(4)
        nodes = sorted(graph.get_node_ids())

        # Hub (first node) is most vital
        hub_vitality = graph.get_node_vitality(nodes[0])
        leaf_vitality = graph.get_node_vitality(nodes[1])

        print(f"Hub vitality: {hub_vitality}")
        print(f"Leaf vitality: {leaf_vitality}")
        # Hub has much higher vitality
        ```

        Use with node labels:

        ```python
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # B is the critical connector
        vitality_b = graph.get_node_vitality("B")
        vitality_a = graph.get_node_vitality("A")

        print(f"Node B vitality: {vitality_b}")
        print(f"Node A vitality: {vitality_a}")
        ```

        Notes
        -----
        - computationally expensive: `O(V * E)` where `V` is vertices, `E` is edges
        - returns `inf` if node removal disconnects the graph
        - related to `closeness_vitality()` which computes for all nodes
        - useful for identifying critical nodes in network vulnerability analysis
        - for all nodes, consider using `closeness_vitality()` instead

        %family Bridges and Connectivity
        """
        from edgecase._core import node_vitality

        # Convert label to node ID if necessary
        if isinstance(node, str):
            node_id = self.get_id_from_label(node)
            if node_id is None:
                raise ValueError(f"Node with label '{node}' does not exist")
        else:
            node_id = node

        return node_vitality(self._graph, node_id)

    # ==============================================================================
    # Connectivity and Flow
    # ==============================================================================

    def get_all_connected_nodes(self, node: Union[int, str]) -> list[int]:
        """
        Get all nodes reachable from the specified node.

        Uses breadth-first search (BFS) to find all nodes in the same connected component. For
        directed graphs, treats edges as undirected during traversal.

        Parameters
        ----------
        node
            Starting node ID (int) or label (str) for the search.

        Returns
        -------
        list[int]
            List of all reachable node IDs, excluding the starting node itself.
            Empty list if the node is isolated.

        Raises
        ------
        KeyError
            If `node=` does not exist in the graph.

        Examples
        --------
        ```python
        graph = Graph()

        # Create two separate components
        graph.add_path_graph(3)  # Component 1: nodes 0-1-2
        graph.add_path_graph(2)  # Component 2: nodes 3-4

        # Find all nodes reachable from node 0
        node_ids = list(graph.get_node_ids())
        connected = graph.get_all_connected_nodes(node_ids[0])
        print(len(connected))  # 2 (nodes 1 and 2)

        # Node 3 is in a separate component
        connected_3 = graph.get_all_connected_nodes(node_ids[3])
        print(len(connected_3))  # 1 (node 4)
        ```

        Component size analysis:

        ```python
        from edgecase import Graph

        graph = Graph()
        # Create two separate components
        graph.add_path_graph(n=3)  # Component 1: nodes 0-1-2
        graph.add_path_graph(n=2)  # Component 2: nodes 3-4

        # Find size of each component
        visited = set()
        components = []

        for node_id in graph.get_node_ids(limit=None):
            if node_id not in visited:
                component = graph.get_all_connected_nodes(node=node_id)
                component.append(node_id)  # Include starting node
                components.append(component)
                visited.update(component)

        print(f"Found {len(components)} components")
        ```

        %family Connectivity and Flow
        """
        # Resolve node to integer ID
        node_id = self._resolve_node_id(node)

        if not self.has_node(node_id):
            raise KeyError(f"Node '{node}' does not exist")

        visited = set()
        queue = [node_id]
        visited.add(node_id)

        while queue:
            current = queue.pop(0)

            # Get all neighbors (treating directed as undirected)
            neighbors = self.neighbors(current, mode="all")

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Remove the starting node from results
        visited.discard(node_id)

        return sorted(list(visited))

    def get_articulation_points(self) -> list[int]:
        """
        Find articulation points (cut nodes) in the graph.

        An articulation point is a node whose removal disconnects the graph or increases the number
        of connected components. Uses Tarjan's DFS-based algorithm. Only defined for undirected
        graphs.

        Returns
        -------
        list[int]
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

        # In a path: 1-2-3-4-5
        # Middle nodes 2, 3, 4 are articulation points
        cut_node_ids = graph.get_articulation_points()
        print(len(cut_node_ids))  # 3

        # Convert to labels if needed
        cut_node_labels = [graph.get_label_from_id(nid) for nid in cut_node_ids]
        ```

        Notes
        -----
        Applications:

        - network vulnerability analysis: Critical infrastructure nodes
        - social networks: Key influencers whose removal fragments communities
        - communication networks: Single points of failure

        %family Connectivity and Flow
        """
        return self._graph.get_articulation_points()

    def get_periphery(self) -> list[int]:
        """
        Get periphery nodes (nodes with maximum eccentricity).

        Periphery nodes are farthest from the graph's center. These nodes have the maximum
        eccentricity (longest shortest path to any other node). Returns empty list if graph is
        disconnected.

        Returns
        -------
        list[int]
            List of node IDs in the periphery, empty if disconnected.

        Examples
        --------
        Find periphery nodes in a path graph:

        ```python
        graph = Graph().add_path_graph(5)

        # In path 1-2-3-4-5, ends (1 and 5) are periphery
        periphery_ids = graph.get_periphery()
        print(len(periphery_ids))  # 2

        # Convert to labels if needed
        periphery_labels = sorted([graph.get_label_from_id(nid) for nid in periphery_ids])
        ```

        Notes
        -----
        Characteristics:

        - opposite of center: peripheral vs central nodes
        - in trees: always leaf nodes
        - multiple nodes can have maximum eccentricity

        %family Connectivity and Flow
        """
        return self._graph.get_periphery()

    def get_node_connectivity(self) -> int:
        """
        Compute global node connectivity.

        Returns the minimum number of nodes whose removal disconnects the graph. This is computed as
        the minimum pairwise node connectivity over all pairs of non-adjacent nodes.

        Uses Menger's theorem: node connectivity equals the maximum number of node-disjoint paths.
        Implementation transforms the problem to max-flow.

        Time complexity: `O(V^4)` for complete enumeration of pairs.

        Returns
        -------
        int
            Minimum number of nodes needed to disconnect the graph. Returns `0` if already
            disconnected. Returns `n-1` for complete graph `K_n`.

        Examples
        --------
        Cycle graph has node connectivity 2:

        ```python
        graph = Graph()
        n = 6
        for i in range(1, n + 1):
            graph.add_node(label=str(i))
        for i in range(1, n + 1):
            graph.add_edge(str(i), str((i % n) + 1))

        print(graph.get_node_connectivity())  # 2
        ```

        Complete graph K_5 has node connectivity 4:

        ```python
        graph = Graph()
        for i in range(1, 6):
            graph.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                graph.add_edge(str(i), str(j))

        print(graph.get_node_connectivity())  # 4
        ```

        %family Connectivity and Flow
        """
        return self._graph.node_connectivity()

    def get_node_connectivity_between(
        self, source: Union[int, str], target: Union[int, str]
    ) -> int:
        """
        Compute node connectivity between two nodes.

        Returns the minimum number of nodes that must be removed to disconnect the source from the
        target. Uses Menger's theorem: this equals the maximum number of node-disjoint paths between
        the nodes.

        Implementation transforms the graph by splitting each internal node into in-node and
        out-node with capacity 1, converting node connectivity to edge connectivity, which is
        computed using max-flow.

        Time complexity: `O(V * E^2)` using Ford-Fulkerson with BFS.

        Parameters
        ----------
        source
            Source node ID (int) or label (str).
        target
            Target node ID (int) or label (str).

        Returns
        -------
        int
            Minimum number of nodes whose removal disconnects source from target.

        Raises
        ------
        ValueError
            If source or target doesn't exist, or if source == target.

        Examples
        --------
        Path graph connectivity using labels:

        ```python
        graph = Graph()
        for i in range(1, 6):
            graph.add_node(label=str(i))
        for i in range(1, 5):
            graph.add_edge(str(i), str(i + 1))

        # Endpoints have connectivity 1 (any middle node disconnects them)
        print(graph.get_node_connectivity_between("1", "5"))  # 1
        ```

        Using node IDs:

        ```python
        graph = Graph()
        for i in range(1, 6):
            graph.add_node(label=str(i))
        for i in range(1, 5):
            graph.add_edge(str(i), str(i + 1))

        # Get node IDs
        node1_id = graph.get_id_from_label("1")
        node5_id = graph.get_id_from_label("5")

        # Can use node IDs directly
        print(graph.get_node_connectivity_between(node1_id, node5_id))  # 1
        ```

        Complete graph connectivity:

        ```python
        graph = Graph()
        for i in range(1, 6):
            graph.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                graph.add_edge(str(i), str(j))

        # Any two nodes in K_5 have connectivity 4 (need to remove 4 nodes to disconnect)
        print(graph.get_node_connectivity_between("1", "5"))  # 4
        ```

        %family Connectivity and Flow
        """
        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)
        return self._graph.node_connectivity_between(source_id, target_id)

    def get_edge_connectivity(self) -> int:
        """
        Compute edge connectivity (minimum edge cut).

        Returns the minimum number of edges whose removal disconnects the graph. For undirected
        graphs, this is computed as the minimum over strategic pairs of nodes. For directed graphs,
        respects edge directions.

        Edge connectivity provides a measure of graph robustness: higher values indicate more
        redundant paths and better fault tolerance.

        Time complexity: `O(V * E^2)` using max-flow computations.

        Returns
        -------
        int
            Minimum number of edges needed to disconnect the graph. Returns `0` if already
            disconnected or has isolated nodes.

        Examples
        --------
        Cycle graph has edge connectivity 2:

        ```python
        graph = Graph()
        n = 6
        for i in range(1, n + 1):
            graph.add_node(label=str(i))
        for i in range(1, n + 1):
            graph.add_edge(str(i), str((i % n) + 1))

        print(graph.get_edge_connectivity())  # 2
        ```

        Star graph has edge connectivity 1:

        ```python
        graph = Graph()
        graph.add_node(label="center")
        for i in range(1, 6):
            graph.add_node(label=str(i))
            graph.add_edge("center", str(i))

        print(graph.get_edge_connectivity())  # 1
        ```

        %family Connectivity and Flow
        """
        return self._graph.edge_connectivity()

    def get_adhesion(self) -> int:
        """
        Get adhesion (minimum edge cut size).

        The adhesion is the minimum number of edges whose removal would disconnect the graph. Also
        called edge connectivity. Only defined for undirected graphs. Returns `0` if already
        disconnected.

        Note: Current implementation provides an upper bound (minimum degree) for efficiency. Exact
        computation requires max-flow algorithms.

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

        %family Connectivity and Flow
        """
        return self._graph.get_adhesion()

    def get_cohesion(self) -> int:
        """
        Get cohesion (minimum node cut size).

        The cohesion is the minimum number of nodes whose removal would disconnect the graph. Also
        called node connectivity. Only defined for undirected graphs. Returns `0` if already
        disconnected.

        Note: Current implementation provides an upper bound (minimum degree) for efficiency. Exact
        computation requires sophisticated algorithms.

        Returns
        -------
        int
            Minimum node cut size.

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

        %family Connectivity and Flow
        """
        return self._graph.get_cohesion()

    def get_density(self) -> float:
        """
        Calculate graph density (ratio of actual to possible edges).

        Density measures how complete a graph is. A complete graph has density `1.0`, while a graph
        with no edges has density `0.0`.

        For directed graphs: `density = E / (N × (N-1))`
        For undirected graphs: `density = 2E / (N × (N-1))`

        Returns
        -------
        float
            Graph density between `0.0` and `1.0`.

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
        print(f"{graph.get_density():.3f}")  # 0.667
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

        print(f"{complete.get_density():.3f}")  # 1.000
        ```

        %family Connectivity and Flow
        """
        return self._graph.density()

    def find_cohesive_blocks(self) -> tuple[list[list[int]], list[int], list[int | None]]:
        """
        Find hierarchical cohesive block structure.

        Identifies k-cohesive blocks in the graph, where a k-cohesive block is a maximal subgraph
        that requires removal of at least `k` nodes to disconnect it. Returns a hierarchical
        structure where child blocks have higher cohesion than their parents.

        Time complexity: `O(V^5)` due to node connectivity computations.

        Returns
        -------
        tuple[list[list[int]], list[int], list[int | None]]
            A tuple containing: (1) blocks, a list of node sets (each as list of node IDs), (2)
            cohesion, a list of k-values (cohesion of each block), and (3) parent, a list of parent
            block indices (`None` for root blocks).

        Examples
        --------
        Find cohesive blocks in a complete graph:

        ```python
        # K_5 complete graph has cohesion 4
        graph = Graph()
        for i in range(1, 6):
            graph.add_node(label=str(i))
        for i in range(1, 6):
            for j in range(i + 1, 6):
                graph.add_edge(str(i), str(j))

        blocks, cohesions, parents = graph.find_cohesive_blocks()
        # Returns: ([[1,2,3,4,5]], [4], [None])

        # Convert to labels if needed
        blocks_labels = [[graph.get_label_from_id(nid) for nid in block] for block in blocks]
        ```

        Find hierarchical structure:

        ```python
        # Graph with nested cohesive structure
        graph = Graph()
        # Build K_4 (cohesion 3)
        for i in range(1, 5):
            graph.add_node(label=str(i))
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_edge(str(i), str(j))

        # Add nodes to create larger 2-cohesive block
        graph.add_node(label="5")
        graph.add_node(label="6")
        graph.add_edge("1", "5")
        graph.add_edge("2", "5")
        graph.add_edge("5", "6")
        graph.add_edge("6", "1")
        graph.add_edge("6", "2")

        blocks, cohesions, parents = graph.find_cohesive_blocks()
        # Returns hierarchical structure with root block and K_4 sub-block (as node IDs)
        ```

        %family Connectivity and Flow
        """
        return self._graph.cohesive_blocks()

    def get_min_cut_between(self, source: Union[int, str], sink: Union[int, str]) -> int:
        """
        Get minimum cut between source and sink nodes.

        Computes the minimum number of edges that must be removed to disconnect the source from the
        sink. Uses the Ford-Fulkerson algorithm with BFS (Edmonds-Karp) to find maximum flow, which
        equals the minimum cut by the max-flow min-cut theorem.

        For unweighted graphs, each edge has capacity `1`. For undirected graphs, edges are treated
        as bidirectional.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).

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
            .add_edge("A", "B", auto_create_nodes=True)
            .add_edge("B", "C", auto_create_nodes=True)
            .add_edge("C", "D", auto_create_nodes=True)
        )

        a_id, d_id = graph.get_id_from_label("A"), graph.get_id_from_label("D")
        print(graph.get_min_cut_between(a_id, d_id))  # 1 (bottleneck)
        ```

        %family Connectivity and Flow
        """
        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        return self._graph.get_min_cut_between(source_id, sink_id)

    def minimum_node_cut(self, source: Union[int, str], sink: Union[int, str]) -> set[int]:
        """
        Find minimum node cut between source and sink nodes.

        Returns the set of nodes whose removal disconnects the source from the sink. Uses
        transformation to an edge connectivity problem where each node is split into in/out copies
        connected by unit capacity.

        The source and sink themselves are never included in the returned cut set.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).

        Returns
        -------
        set[int]
            Set of node IDs forming the minimum node cut.

        Raises
        ------
        ValueError
            If source or sink node doesn't exist, or if they're the same.

        Notes
        -----
        The node connectivity between two nodes is the minimum number of nodes that
        must be removed to disconnect them. This is closely related to Menger's theorem.

        Examples
        --------
        Find minimum node cut:

        ```python
        graph = (
            Graph()
            .add_edge("A", "B", auto_create_nodes=True)
            .add_edge("A", "C", auto_create_nodes=True)
            .add_edge("B", "D", auto_create_nodes=True)
            .add_edge("C", "D", auto_create_nodes=True)
            .add_edge("D", "E", auto_create_nodes=True)
        )

        # Find nodes that separate A from E
        cut_nodes = graph.minimum_node_cut("A", "E")
        # Returns {node_id_d} where removing node D disconnects A from E
        ```

        Here's an example on the topic of network vulnerability. We can identify single points of
        failure:

        ```python
        # Communication network
        graph = (
            Graph()
            .add_edge("Client", "Gateway", auto_create_nodes=True)
            .add_edge("Gateway", "Server1", auto_create_nodes=True)
            .add_edge("Gateway", "Server2", auto_create_nodes=True)
            .add_edge("Server1", "Database", auto_create_nodes=True)
            .add_edge("Server2", "Database", auto_create_nodes=True)
        )

        critical_nodes = graph.minimum_node_cut("Client", "Database")
        # Identifies Gateway as single point of failure
        print(f"Critical nodes: {len(critical_nodes)}")
        ```

        Social network bottlenecks:

        ```python
        # Find intermediaries between two people
        graph = (
            Graph()
            .add_edge("Alice", "Bob", auto_create_nodes=True)
            .add_edge("Bob", "Carol", auto_create_nodes=True)
            .add_edge("Alice", "Dave", auto_create_nodes=True)
            .add_edge("Dave", "Carol", auto_create_nodes=True)
        )

        intermediaries = graph.minimum_node_cut("Alice", "Carol")
        # Returns {bob_id, dave_id} - both must be removed to disconnect
        ```

        %family Connectivity and Flow
        """
        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        return self._graph.minimum_node_cut(source_id, sink_id)

    def minimum_edge_cut(
        self, source: Union[int, str], sink: Union[int, str]
    ) -> set[tuple[int, int]]:
        """
        Find minimum edge cut between source and sink nodes.

        Returns the set of edges whose removal disconnects the source from the sink. Uses the
        max-flow min-cut theorem via Ford-Fulkerson algorithm to identify the minimum cut.

        For undirected graphs, edges are treated as bidirectional. For directed graphs, only forward
        edges are considered.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).

        Returns
        -------
        set[tuple[int, int]]
            Set of edge tuples (source_id, target_id) forming the minimum cut.

        Raises
        ------
        ValueError
            If source or sink node doesn't exist, or if they're the same.

        Examples
        --------
        Find minimum edge cut in a simple graph:

        ```python
        graph = (
            Graph()
            .add_edge("A", "B", auto_create_nodes=True)
            .add_edge("B", "C", auto_create_nodes=True)
            .add_edge("C", "D", auto_create_nodes=True)
            .add_edge("A", "D", auto_create_nodes=True)
        )

        # Find edges that separate A from C
        cut_edges = graph.minimum_edge_cut("A", "C")
        # Returns set of edge tuples like {(node_id_b, node_id_c), (node_id_d, node_id_c)}
        ```

        Network vulnerability analysis:

        ```python
        # Analyze which connections are critical
        graph = (
            Graph(directed=True)
            .add_edge("Server1", "Router1", auto_create_nodes=True)
            .add_edge("Router1", "Router2", auto_create_nodes=True)
            .add_edge("Router2", "Server2", auto_create_nodes=True)
            .add_edge("Router1", "Router3", auto_create_nodes=True)
            .add_edge("Router3", "Server2", auto_create_nodes=True)
        )

        critical_links = graph.minimum_edge_cut("Server1", "Server2")
        print(f"Removing {len(critical_links)} link(s) disconnects the servers")
        ```

        %family Connectivity and Flow
        """
        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        return self._graph.minimum_edge_cut(source_id, sink_id)

    def max_flow(
        self,
        source: Union[int, str],
        sink: Union[int, str],
        capacity: str | None = None,
    ) -> dict[int, float]:
        """
        Compute maximum flow from source to sink.

        Finds the maximum amount of flow that can be sent from the source node to the sink node
        through the network. Uses the Edmonds-Karp algorithm (Ford-Fulkerson with BFS), which
        guarantees `O(VE²)` time complexity.

        For weighted graphs, edge weights are used as capacities by default. For unweighted graphs,
        each edge has capacity `1`. You can also specify a custom capacity attribute.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).
        capacity
            Optional name of edge attribute to use as capacity. If `None` (default), uses edge
            weights for weighted graphs, or `1.0` for unweighted graphs.

        Returns
        -------
        dict[int, float]
            Dictionary mapping `edge_id -> flow amount` for edges with positive flow. Use
            `max_flow_value()` if you only need the total flow value.

        Raises
        ------
        ValueError
            If source or sink doesn't exist, they're the same node, or any capacity is negative.

        Examples
        --------
        Basic max flow with edge weights as capacities:

        ```python
        # Create a flow network
        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="S")  # source
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")  # sink
            .add_edge("S", "A", weight=10.0)
            .add_edge("S", "B", weight=5.0)
            .add_edge("A", "B", weight=15.0)
            .add_edge("A", "T", weight=10.0)
            .add_edge("B", "T", weight=10.0)
        )

        flow_dict = graph.max_flow("S", "T")
        total_flow = sum(flow_dict.values())  # Note: count outgoing from source
        print(f"Edges with flow: {len(flow_dict)}")

        # See flow on each edge
        for edge_id, flow in flow_dict.items():
            edge = graph.get_edge(edge_id)
            src = graph.get_label_from_id(edge.source)
            tgt = graph.get_label_from_id(edge.target)
            print(f"  {src} -> {tgt}: {flow}")
        ```

        Using custom capacity attribute:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", bandwidth=100)
            .add_edge("B", "C", bandwidth=50)
            .add_edge("A", "C", bandwidth=30)
        )

        flow_dict = graph.max_flow("A", "C", capacity="bandwidth")
        print(f"Edges carrying flow: {len(flow_dict)}")
        ```

        Unweighted graph (unit capacities):

        ```python
        graph = (
            Graph(directed=True)
            .add_edge("S", "A", auto_create_nodes=True)
            .add_edge("S", "B", auto_create_nodes=True)
            .add_edge("A", "T", auto_create_nodes=True)
            .add_edge("B", "T", auto_create_nodes=True)
        )

        flow_dict = graph.max_flow("S", "T")
        # All 4 edges carry flow of 1.0
        ```

        Notes
        -----
        - for undirected graphs, each edge is treated as two directed edges with equal capacity
        - self-loops are ignored in flow computation
        - the max-flow min-cut theorem states that max_flow equals the minimum cut capacity

        %family Connectivity and Flow
        """
        from edgecase._core import max_flow as _max_flow

        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        return dict(_max_flow(self._graph, source_id, sink_id, capacity))

    def max_flow_value(
        self,
        source: Union[int, str],
        sink: Union[int, str],
        capacity: str | None = None,
    ) -> float:
        """
        Compute only the maximum flow value from source to sink.

        This is a faster variant of `max_flow()` when you only need the flow value and not the flow
        assignment on each edge.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).
        capacity
            Optional name of edge attribute to use as capacity. If `None` (the default), uses edge
            weights for weighted graphs, or `1.0` for unweighted graphs.

        Returns
        -------
        float
            The maximum flow value.

        Examples
        --------
        ```python
        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_node(label="A")
            .add_edge("S", "A", weight=10.0)
            .add_edge("A", "T", weight=5.0)
        )

        flow = graph.max_flow_value("S", "T")
        print(f"Maximum flow: {flow}")  # 5.0
        ```

        %family Connectivity and Flow
        """
        from edgecase._core import max_flow_value as _max_flow_value

        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        return _max_flow_value(self._graph, source_id, sink_id, capacity)

    def min_cost_flow(
        self,
        source: Union[int, str],
        sink: Union[int, str],
        demand: float | None = None,
        capacity: str | None = None,
        cost: str | None = None,
    ) -> tuple[float, float, dict[int, float]]:
        """
        Compute minimum cost flow from source to sink.

        Finds the flow that minimizes total cost. Can either send a specified amount of flow
        (demand) or find the maximum flow at minimum cost.

        Uses the Successive Shortest Path algorithm with Bellman-Ford, which handles negative edge
        costs correctly.

        Parameters
        ----------
        source
            Source node (integer ID or string label).
        sink
            Sink node (integer ID or string label).
        demand
            Required flow amount. If `None` (default), finds maximum flow at minimum cost. If
            specified, the algorithm attempts to send exactly this much flow.
        capacity
            Optional name of edge attribute to use as capacity. If `None` (default), uses edge
            weights for weighted graphs, or `1.0` for unweighted graphs.
        cost
            Optional name of edge attribute to use as cost per unit flow. If `None` (default), looks
            for a `"cost"` attribute; otherwise uses `1.0`.

        Returns
        -------
        tuple[float, float, dict[int, float]]
            A tuple containing: (1) total_flow, the total amount of flow sent, (2) total_cost: the
            total cost of the flow, and (3) flow_dict, a dictionary mapping `edge_id` to flow amount
            for edges with positive flow.

        Raises
        ------
        ValueError
            If the graph is undirected, source/sink doesn't exist, they're the same node, demand is
            negative, or demand cannot be satisfied.

        Examples
        --------
        Basic minimum cost flow with maximum flow:

        ```python
        from edgecase import Graph

        # Create a flow network with costs
        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")
            .add_edge("S", "A", weight=10.0, cost=1.0)   # capacity 10, cost 1
            .add_edge("S", "B", weight=10.0, cost=2.0)   # capacity 10, cost 2
            .add_edge("A", "T", weight=10.0, cost=1.0)   # capacity 10, cost 1
            .add_edge("B", "T", weight=10.0, cost=1.0)   # capacity 10, cost 1
        )

        total_flow, total_cost, flow_dict = graph.min_cost_flow("S", "T")
        print(f"Flow: {total_flow}, Cost: {total_cost}")
        # Prefers S->A->T path (cost 2) over S->B->T (cost 3)
        ```

        Sending a specific amount of flow:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="T")
            .add_edge("S", "A", weight=20.0, cost=5.0)
            .add_edge("A", "T", weight=20.0, cost=3.0)
        )

        # Send exactly 10 units of flow
        total_flow, total_cost, flow_dict = graph.min_cost_flow("S", "T", demand=10.0)
        print(f"Sent {total_flow} units at cost {total_cost}")  # Flow: 10, Cost: 80
        ```

        Using custom capacity and cost attributes:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", bandwidth=100, price=5)
            .add_edge("B", "C", bandwidth=50, price=3)
            .add_edge("A", "C", bandwidth=30, price=10)
        )

        total_flow, total_cost, flow_dict = graph.min_cost_flow(
            "A", "C", capacity="bandwidth", cost="price"
        )
        print(f"Maximum flow: {total_flow}")
        print(f"Minimum cost: {total_cost}")
        ```

        Transportation problem example:

        ```python
        from edgecase import Graph

        # Warehouses to stores with shipping costs
        graph = (
            Graph(directed=True, weighted=True)
            .add_node(label="Source")
            .add_node(label="W1")  # Warehouse 1
            .add_node(label="W2")  # Warehouse 2
            .add_node(label="S1")  # Store 1
            .add_node(label="S2")  # Store 2
            .add_node(label="Sink")
            # Source to warehouses (supply)
            .add_edge("Source", "W1", weight=50.0, cost=0.0)
            .add_edge("Source", "W2", weight=60.0, cost=0.0)
            # Warehouses to stores (shipping costs)
            .add_edge("W1", "S1", weight=100.0, cost=8.0)
            .add_edge("W1", "S2", weight=100.0, cost=5.0)
            .add_edge("W2", "S1", weight=100.0, cost=4.0)
            .add_edge("W2", "S2", weight=100.0, cost=7.0)
            # Stores to sink (demand)
            .add_edge("S1", "Sink", weight=40.0, cost=0.0)
            .add_edge("S2", "Sink", weight=50.0, cost=0.0)
        )

        flow, cost, assignment = graph.min_cost_flow("Source", "Sink", demand=90.0)
        print(f"Total shipping cost: {cost}")
        ```

        Notes
        -----
        - graph must be directed (use `.to_directed()` to convert if needed)
        - self-loops are ignored in flow computation
        - negative costs are allowed (the algorithm handles them correctly)
        - the algorithm uses Successive Shortest Path with Bellman-Ford, with time complexity
        `O(V * E * F)` where `F` is the total flow

        %family Connectivity and Flow
        """
        from edgecase._core import min_cost_flow as _min_cost_flow

        source_id = self._resolve_node_id(source, auto_create=False)
        sink_id = self._resolve_node_id(sink, auto_create=False)
        result = _min_cost_flow(self._graph, source_id, sink_id, demand, capacity, cost)
        return (result[0], result[1], dict(result[2]))

    # ==============================================================================
    # Component Counts
    # ==============================================================================

    def count_weakly_connected_components(self) -> int:
        """
        Count weakly connected components in the graph.

        For undirected graphs, counts regular connected components. For directed graphs, treats all
        edges as undirected (ignoring direction) when finding components.

        A weakly connected component is a maximal set of nodes where every node is reachable from
        every other node when edge directions are ignored.

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

        %family Connected Components
        %seealso weakly_connected_components, count_strongly_connected_components, is_weakly_connected
        """
        return self._graph.count_weakly_connected_components()

    def count_strongly_connected_components(self) -> int:
        """
        Count strongly connected components.

        For directed graphs, a strongly connected component is a maximal set of nodes where every
        node is reachable from every other node following directed edges. Uses Kosaraju's algorithm
        for efficient computation.

        For undirected graphs, returns the same result as `count_weakly_connected_components()`
        since strong connectivity equals regular connectivity when edges are bidirectional.

        Returns
        -------
        int
            Number of strongly connected components.

        Examples
        --------
        Count strongly connected components in a directed graph:

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

        For undirected graphs, equivalent to connected components:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="X")
            .add_edge("A", "B")
        )

        # Both return the same result for undirected graphs
        print(graph.count_strongly_connected_components())  # 2
        print(graph.count_weakly_connected_components())     # 2
        ```

        %family Connected Components
        %seealso strongly_connected_components, count_weakly_connected_components, is_strongly_connected
        """
        return self._graph.count_strongly_connected_components()

    def weakly_connected_components(self) -> list[set[int]]:
        """
        Get all weakly connected components as sets of node IDs.

        For undirected graphs, returns regular connected components. For directed graphs, treats all
        edges as undirected (ignoring direction) when finding components.

        A weakly connected component is a maximal set of nodes where every node is reachable from
        every other node when edge directions are ignored.

        Returns
        -------
        list[set[int]]
            List of sets, where each set contains the node IDs in a component. Components are
            returned in no particular order.

        Examples
        --------
        Get components in a disconnected graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge("A", "B")
            .add_edge("X", "Y")
        )

        components = graph.weakly_connected_components()
        print(len(components))  # 2
        for comp in components:
            print(comp)  # Sets of node IDs
        ```

        Find the largest component:

        ```python
        graph = Graph(directed=False)
        graph.add_path_graph(10)  # Connected chain of 10 nodes
        graph.add_node(label="isolated")

        components = graph.weakly_connected_components()
        largest = max(components, key=len)
        print(f"Largest component has {len(largest)} nodes")
        ```

        %family Connected Components
        %seealso count_weakly_connected_components, strongly_connected_components, is_weakly_connected
        """
        # Convert list of lists to list of sets
        return [set(comp) for comp in self._graph.weakly_connected_components()]

    def strongly_connected_components(self) -> list[set[int]]:
        """
        Get all strongly connected components as sets of node IDs.

        For directed graphs, a strongly connected component (SCC) is a maximal set of nodes where
        every node is reachable from every other node following directed edges. Uses Kosaraju's
        algorithm for efficient computation.

        For undirected graphs, returns the same result as `weakly_connected_components()` since
        strong connectivity equals regular connectivity when edges are bidirectional.

        Returns
        -------
        list[set[int]]
            List of sets, where each set contains the node IDs in a strongly connected component.
            Components are returned in no particular order.

        Examples
        --------
        Find SCCs in a directed graph:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")  # Creates cycle -> SCC {A, B, C}
            .add_edge("C", "D")  # D is reachable but can't reach back
        )

        sccs = graph.strongly_connected_components()
        print(len(sccs))  # 2: {A, B, C} and {D}
        ```

        Analyze web graph structure:

        ```python
        # In a web graph, SCCs represent "communities" of mutually linked pages
        graph = (
            Graph(directed=True)
            .add_node(label="A").add_node(label="B")
            .add_node(label="C").add_node(label="D")
            .add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")  # SCC of 3
            .add_edge("C", "D")  # D is a separate SCC
        )

        sccs = graph.strongly_connected_components()
        largest_scc = max(sccs, key=len)
        print(f"Largest SCC contains {len(largest_scc)} pages")
        ```

        Notes
        -----
        The algorithm used is Kosaraju's algorithm which runs in `O(V + E)` time:

        1. perform DFS on original graph to get finish times
        2. transpose the graph (reverse all edges)
        3. perform DFS on transposed graph in reverse finish order

        %family Connected Components
        %seealso count_strongly_connected_components, weakly_connected_components, is_strongly_connected
        """
        # Convert list of lists to list of sets
        return [set(comp) for comp in self._graph.strongly_connected_components()]

    def is_strongly_connected(self) -> bool:
        """
        Check if the graph is strongly connected.

        A directed graph is strongly connected if there is a directed path from every node to every
        other node. Equivalently, the graph has exactly one strongly connected component containing
        all nodes.

        For undirected graphs, this is equivalent to checking regular connectivity (same as
        `is_connected()`).

        Returns
        -------
        bool
            `True` if the graph is strongly connected, `False` otherwise. Empty graphs are
            considered strongly connected (vacuously true).

        Examples
        --------
        Check if all nodes can reach each other:

        ```python
        # Cycle: every node can reach every other
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        print(graph.is_strongly_connected())  # True

        # Add a node that can't be reached from the cycle
        graph.add_node(label="D")
        graph.add_edge("D", "A")  # D can reach A but nothing can reach D
        print(graph.is_strongly_connected())  # False
        ```

        Notes
        -----
        Time complexity: `O(V + E)` using Kosaraju's algorithm to count SCCs.

        %family Connected Components
        %seealso is_weakly_connected, is_connected, strongly_connected_components
        """
        return self._graph.is_strongly_connected()

    def is_weakly_connected(self) -> bool:
        """
        Check if the graph is weakly connected.

        A directed graph is weakly connected if the underlying undirected graph (ignoring edge
        directions) is connected. Equivalently, there is a path between every pair of nodes when
        edge directions are ignored.

        For undirected graphs, this is equivalent to checking regular connectivity (same as
        `is_connected()`).

        Returns
        -------
        bool
            `True` if the graph is weakly connected, `False` otherwise. Empty graphs are considered
            weakly connected (vacuously true).

        Examples
        --------
        Check weak connectivity:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("C", "B")  # All connected when ignoring direction
        )
        print(graph.is_weakly_connected())  # True
        print(graph.is_strongly_connected())  # False (can't reach C from A)
        ```

        Disconnected components:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="X")
            .add_node(label="Y")
            .add_edge("A", "B")
            .add_edge("X", "Y")
        )
        print(graph.is_weakly_connected())  # False
        ```

        Notes
        -----
        Time complexity: `O(V + E)` using BFS/DFS traversal.

        %family Connected Components
        %seealso is_strongly_connected, is_connected, weakly_connected_components
        """
        return self._graph.is_weakly_connected()

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # COMBINATORIAL OPTIMIZATION
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Graph Coloring
    # ==============================================================================

    def greedy_color(
        self,
        strategy: str = "largest_first",
        interchange: bool = False,
    ) -> dict[int, int]:
        """
        Color the graph using a greedy algorithm.

        Graph coloring assigns colors (represented as integers) to nodes such that no two adjacent
        nodes share the same color. The greedy algorithm processes nodes in a specified order and
        assigns the smallest available color to each node.

        Parameters
        ----------
        strategy
            The strategy for ordering nodes during coloring. Options are: `"largest_first"` (the
            default), `"smallest_last"`, `"saturation_largest_first"`, `"independent_set"`,
            `"random"`, and `"arbitrary"`. See the *Strategy Types* section for more information.

        interchange
            Whether to use color interchange optimization (not yet implemented).

        Returns
        -------
        dict[int, int]
            A dictionary mapping node IDs to their assigned color (0-indexed integers).
            The colors are consecutive integers starting from 0.

        Strategy Types
        --------------
        There are several strategies that can be used:

        - `"largest_first"`: process high-degree nodes first; often produces good results by
        handling constrained nodes early
        - `"smallest_last"`: Process nodes in reverse order of removal by minimum degree; often
        produces optimal or near-optimal colorings
        - `"saturation_largest_first"` or `"dsatur"`: DSATUR algorithm that prioritizes nodes with
        the most differently-colored neighbors; often produces excellent results
        - `"independent_set"`: colors independent sets iteratively
        - `"random"`: random node ordering
        - `"arbitrary"`: process in the order nodes appear in the graph

        Notes
        -----
        - greedy coloring does not guarantee the minimum number of colors (chromatic number)
        - the number of colors used is at most `max_degree + 1`
        - different strategies may produce different numbers of colors
        - for bipartite graphs, optimal coloring uses exactly 2 colors
        - the `"smallest_last"` and `"dsatur"` strategies often produce good results

        Examples
        --------
        Color a simple graph:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
            .add_edge("D", "A")
        )

        coloring = graph.greedy_color()
        print(coloring)  # {1: 0, 2: 1, 3: 0, 4: 1}  (2 colors for a square)
        ```

        Use DSATUR strategy for potentially better results:

        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        coloring = graph.greedy_color(strategy="dsatur")
        num_colors = len(set(coloring.values()))
        print(f"Used {num_colors} colors")  # 3 colors for a triangle
        ```

        %family Graph Coloring
        """
        from edgecase._core import greedy_color as rust_greedy_color

        return dict(rust_greedy_color(self._graph, strategy, interchange))

    def is_valid_coloring(self, coloring: dict[int, int]) -> bool:
        """
        Check if a coloring is valid (no adjacent nodes share a color).

        Parameters
        ----------
        coloring
            A dictionary mapping node IDs to colors (integers).

        Returns
        -------
        bool
            `True` if the coloring is valid, `False` otherwise.

        Examples
        --------
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        node_ids = graph.get_node_ids()

        # Valid coloring (alternating colors)
        valid = {node_ids[0]: 0, node_ids[1]: 1, node_ids[2]: 0}
        print(graph.is_valid_coloring(valid))  # True

        # Invalid coloring (adjacent nodes same color)
        invalid = {node_ids[0]: 0, node_ids[1]: 0, node_ids[2]: 1}
        print(graph.is_valid_coloring(invalid))  # False
        ```

        %family Graph Coloring
        """
        from edgecase._core import is_valid_coloring as rust_is_valid_coloring

        return rust_is_valid_coloring(self._graph, coloring)

    def chromatic_number_upper_bound(self, coloring: dict[int, int] | None = None) -> int:
        """
        Get an upper bound on the chromatic number from a coloring.

        The chromatic number is the minimum number of colors needed to color a graph. This method
        returns the number of distinct colors used in a coloring, which provides an upper bound.

        Parameters
        ----------
        coloring
            A dictionary mapping node IDs to colors. If `None`, computes a greedy coloring first
            using the `"largest_first"` strategy.

        Returns
        -------
        int
            The number of distinct colors used in the coloring.

        Examples
        --------
        ```python
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        # Get upper bound using default coloring
        bound = graph.chromatic_number_upper_bound()
        print(f"Chromatic number <= {bound}")  # Chromatic number <= 3
        ```

        %family Graph Coloring
        """
        if coloring is None:
            coloring = self.greedy_color()

        from edgecase._core import chromatic_number_upper_bound as rust_bound

        return rust_bound(coloring)

    # ==============================================================================
    # Matching
    # ==============================================================================

    def is_matching(self, edges: list[Union[int, str]]) -> bool:
        """
        Check if a set of edges forms a valid matching.

        A matching is a set of edges where no two edges share a common endpoint (node). This is
        useful for validating matchings or checking matching constraints.

        Parameters
        ----------
        edges
            List of edge IDs or edge labels to check.

        Returns
        -------
        bool
            `True` if the edges form a valid matching (no shared endpoints), `False` otherwise.

        Examples
        --------
        Check a valid matching:

        ```python
        from edgecase import Graph

        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("C", "D")
        )

        edge_ids = graph.get_edge_ids()
        is_valid = graph.is_matching(edge_ids)
        print(is_valid)  # True (no shared endpoints)
        ```

        Check an invalid matching:

        ```python
        graph = (
            Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        edge_ids = graph.get_edge_ids()
        is_valid = graph.is_matching(edge_ids)
        print(is_valid)  # False (B is shared)
        ```

        Validate a computed matching:

        ```python
        graph = Graph()
        graph.add_path_graph(6)

        matching = graph.maximal_matching()
        assert graph.is_matching(matching)  # Always True for maximal_matching result
        ```

        %family Matching
        """
        from edgecase._core import is_matching

        # Edge IDs are already integers, no need to resolve
        return is_matching(self._graph, edges)

    def maximal_matching(self) -> list[int]:
        """
        Find a maximal matching in the graph.

        A maximal matching is a matching (set of edges with no shared endpoints) that cannot be
        extended by adding any other edge. Uses a greedy algorithm.

        Note that a maximal matching is not necessarily a maximum matching (largest possible). For
        weighted graphs, use `max_weight_matching()`.

        Returns
        -------
        list[int]
            List of edge IDs forming a maximal matching.

        Examples
        --------
        Find a maximal matching in a path:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_path_graph(6)  # 1-2-3-4-5-6

        matching = graph.maximal_matching()
        print(f"Matched {len(matching)} edges")
        # Will match 3 edges (e.g., 1-2, 3-4, 5-6)

        # Verify it's a valid matching
        assert graph.is_matching(matching)
        ```

        Visualize the matching:

        ```python
        graph = Graph()
        graph.add_grid_2d(rows=4, cols=4)

        matching = graph.maximal_matching()

        # Mark matched edges
        matched_dict = {edge_id: True for edge_id in matching}
        graph.set_edge_attrs("matched", matched_dict)

        # Color matched edges
        colors = {edge_id: "red" for edge_id in matching}
        graph.set_edge_attrs("color", colors)
        ```

        Compare with complete graph:

        ```python
        graph = Graph()
        graph.add_full_graph(10)

        matching = graph.maximal_matching()
        print(f"Nodes: {graph.count_nodes()}")  # 10
        print(f"Matched edges: {len(matching)}")  # 5 (perfect matching)
        ```

        Notes
        -----
        - time complexity: `O(E)` where `E` is the number of edges
        - space complexity: `O(V)` for tracking matched nodes
        - greedy algorithm: processes edges in arbitrary order
        - result may vary between calls due to iteration order
        - for maximum cardinality matching, use specialized algorithms
        - for weighted matching, use `max_weight_matching()`

        %family Matching
        """
        from edgecase._core import maximal_matching

        return maximal_matching(self._graph)

    def max_weight_matching(self) -> list[int]:
        """
        Find a maximum weight matching in the graph.

        Finds a matching that maximizes the sum of edge weights. Uses a greedy algorithm that
        selects edges by weight in descending order. For unweighted graphs, all edges have weight
        `1.0`.

        Note that this is a greedy approximation, not guaranteed to be optimal for general graphs.
        For optimal solutions on bipartite graphs, use specialized bipartite matching algorithms.

        Returns
        -------
        list[int]
            List of edge IDs forming a maximum weight matching.

        Examples
        --------
        Find maximum weight matching:

        ```python
        from edgecase import Graph

        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=10.0)
            .add_edge("B", "C", weight=1.0)
            .add_edge("C", "D", weight=1.0)
            .add_edge("A", "C", weight=2.0)
        )

        matching = graph.max_weight_matching()

        # Calculate total weight
        total_weight = sum(
            graph.get_edge(eid).weight for eid in matching
        )
        print(f"Total weight: {total_weight}")
        ```

        Unweighted graph (maximizes cardinality):

        ```python
        graph = Graph()
        graph.add_full_graph(8)

        matching = graph.max_weight_matching()
        print(f"Matched {len(matching)} edges")  # 4 (perfect matching)

        # Verify it's valid
        assert graph.is_matching(matching)
        ```

        Compare greedy with optimal:

        ```python
        from edgecase import Graph

        # Triangle with one heavy edge
        graph = (
            Graph(weighted=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=5.0)
            .add_edge("B", "C", weight=3.0)
            .add_edge("A", "C", weight=2.0)
        )

        matching = graph.max_weight_matching()
        # Greedy picks A-B (5.0), blocks B-C and A-C
        # Optimal would be A-C (2.0) + B-? if more nodes existed

        matched_edges = [graph.get_edge(eid) for eid in matching]
        for edge in matched_edges:
            print(f"{edge.source} - {edge.target}: {edge.weight}")
        ```

        Visualize the matching:

        ```python
        import random
        from edgecase import Graph

        graph = Graph(weighted=True)
        graph.add_grid_2d(rows=5, cols=5)

        # Add random weights
        weights = {eid: random.uniform(0.5, 2.0) for eid in graph.get_edge_ids()}
        graph.set_edge_attrs("weight", weights)

        matching = graph.max_weight_matching()

        # Highlight matched edges
        colors = {edge_id: "red" for edge_id in matching}
        widths = {edge_id: 3.0 for edge_id in matching}
        graph.set_edge_attrs("color", colors)
        graph.set_edge_attrs("width", widths)
        ```

        Notes
        -----
        - time complexity: `O(E log E)` for sorting edges
        - space complexity: `O(V + E)` for tracking matched nodes and edges
        - greedy approximation: selects heaviest edges first
        - not guaranteed to be optimal for general graphs
        - optimal for trees and forests
        - for bipartite graphs, consider Hungarian algorithm
        - for unweighted graphs, equivalent to maximal matching

        %family Matching
        """
        from edgecase._core import max_weight_matching

        return max_weight_matching(self._graph)

    # ==============================================================================
    # Independent Sets
    # ==============================================================================

    def is_independent_set(self, nodes: list[int]) -> bool:
        """
        Check if a set of nodes forms an independent set.

        An independent set (also called a stable set) is a set of vertices in a graph where no two
        vertices are adjacent. This method validates whether the given nodes satisfy this property.

        Parameters
        ----------
        nodes
            List of node IDs to check.

        Returns
        -------
        bool
            `True` if the nodes form a valid independent set (no edges between them), `False`
            otherwise.

        Examples
        --------
        Check a valid independent set:

        ```python
        from edgecase import Graph

        # Create a path: A-B-C-D
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

        # A and C are not adjacent
        a_id = graph.get_id_from_label("A")
        c_id = graph.get_id_from_label("C")
        print(graph.is_independent_set([a_id, c_id]))  # True

        # A, C, D: C and D are adjacent
        d_id = graph.get_id_from_label("D")
        print(graph.is_independent_set([a_id, c_id, d_id]))  # False
        ```

        Validate a computed independent set:

        ```python
        graph = Graph()
        graph.add_path_graph(10)

        mis = graph.maximal_independent_set()
        assert graph.is_independent_set(mis)  # Always True
        ```

        Notes
        -----
        - time complexity: `O(V + E)` where `V` is nodes in the set and `E` is edges
        - empty set is trivially an independent set
        - single node is trivially an independent set

        %family Independent Sets
        """
        from edgecase._core import is_independent_set

        return is_independent_set(self._graph, nodes)

    def maximal_independent_set(
        self,
        nodes: list[int] | None = None,
        seed: int | None = None,
    ) -> list[int]:
        """
        Find a maximal independent set in the graph.

        A maximal independent set is an independent set that cannot be extended by adding any other
        vertex. In other words, every vertex not in the set is adjacent to at least one vertex in
        the set.

        This uses a randomized greedy algorithm that picks nodes randomly and adds them if they
        don't conflict with already-chosen nodes.

        Parameters
        ----------
        nodes
            Optional starting set of node IDs. These nodes must already form an independent set. The
            algorithm will extend this set to be maximal. If `None`, starts with an empty set.
        seed
            Optional random seed for reproducibility. If `None`, uses system entropy.

        Returns
        -------
        list[int]
            List of node IDs forming a maximal independent set.

        Raises
        ------
        ValueError
            If the starting nodes don't form an independent set (have adjacent nodes).
        ValueError
            If any starting node doesn't exist in the graph.

        Examples
        --------
        Find a maximal independent set:

        ```python
        from edgecase import Graph

        # Create a path graph: 1-2-3-4-5
        graph = Graph()
        graph.add_path_graph(5)

        mis = graph.maximal_independent_set(seed=23)
        print(f"Independent set size: {len(mis)}")  # 2 or 3

        # Verify validity
        assert graph.is_independent_set(mis)
        ```

        Start with specific nodes:

        ```python
        graph = Graph()
        graph.add_cycle_graph(6)

        # Start with a specific node
        first_node = graph.get_node_ids()[0]
        mis = graph.maximal_independent_set(nodes=[first_node], seed=23)

        assert first_node in mis
        ```

        Find multiple maximal independent sets:

        ```python
        graph = Graph()
        graph.add_full_graph(6)

        # Different seeds give different results
        sets = [graph.maximal_independent_set(seed=i) for i in range(5)]
        for i, s in enumerate(sets):
            print(f"Seed {i}: {s}")
        ```

        Scheduling application:

        ```python
        # Jobs are nodes, edges connect conflicting jobs
        graph = (
            Graph()
            .add_node(label="Job_A")
            .add_node(label="Job_B")
            .add_node(label="Job_C")
            .add_node(label="Job_D")
            .add_edge("Job_A", "Job_B")  # A conflicts with B
            .add_edge("Job_B", "Job_C")  # B conflicts with C
            .add_edge("Job_C", "Job_D")  # C conflicts with D
        )

        # Find jobs that can run simultaneously
        parallel_jobs = graph.maximal_independent_set()
        labels = [graph.get_label_from_id(nid) for nid in parallel_jobs]
        print(f"Can run together: {labels}")
        ```

        Notes
        -----
        - time complexity: `O(V + E)` where `V` is vertices and `E` is edges
        - space complexity: `O(V)` for tracking available nodes
        - randomized algorithm: results may vary without a seed
        - maximal ≠ maximum: this is not guaranteed to find the largest possible set
        - for the largest independent set, use `maximum_independent_set()`

        %family Independent Sets
        %seealso maximum_independent_set, find_maximal_independent_sets, is_independent_set, find_cliques
        """
        from edgecase._core import maximal_independent_set

        return maximal_independent_set(self._graph, nodes, seed)

    def maximum_independent_set(self) -> list[int]:
        """
        Find an approximate maximum independent set in the graph.

        The maximum independent set problem is NP-hard, so this method uses a greedy heuristic that
        prioritizes nodes with minimum degree. This provides a good approximation but is not
        guaranteed to find the true maximum.

        The algorithm works by iteratively selecting the node with the minimum number of neighbors
        (among remaining candidates), adding it to the set, and removing it and its neighbors from
        further consideration.

        Returns
        -------
        list[int]
            List of node IDs forming an approximate maximum independent set.

        Examples
        --------
        Find maximum independent set:

        ```python
        from edgecase import Graph

        # Create a path graph: 1-2-3-4-5
        graph = Graph()
        graph.add_path_graph(5)

        mis = graph.maximum_independent_set()
        print(f"Maximum independent set size: {len(mis)}")  # 3 for path of 5

        # Verify it's valid
        assert graph.is_independent_set(mis)
        ```

        Compare with maximal independent set:

        ```python
        graph = Graph()
        graph.add_star_graph(10)  # Hub with 10 leaves

        maximal = graph.maximal_independent_set(seed=23)
        maximum = graph.maximum_independent_set()

        print(f"Maximal size: {len(maximal)}")
        print(f"Maximum size: {len(maximum)}")  # Should be larger or equal
        ```

        Vertex cover complement:

        ```python
        # Maximum independent set + minimum vertex cover = all nodes
        graph = Graph()
        graph.add_cycle_graph(8)

        mis = graph.maximum_independent_set()
        vertex_cover = set(graph.get_node_ids()) - set(mis)

        print(f"Independent set: {len(mis)}")
        print(f"Vertex cover: {len(vertex_cover)}")
        print(f"Total: {len(mis) + len(vertex_cover)} = {graph.count_nodes()}")
        ```

        Optimization problem:

        ```python
        # Find maximum set of non-conflicting items
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
            .add_edge("E", "A")  # Cycle
        )

        mis = graph.maximum_independent_set()
        labels = [graph.get_label_from_id(nid) for nid in mis]
        print(f"Best selection: {labels}")
        ```

        Notes
        -----
        - time complexity: `O(V² + VE)` due to repeated neighbor lookups
        - space complexity: `O(V)` for tracking available nodes
        - this is a greedy approximation, not an exact algorithm
        - for exact solutions on small graphs, exhaustive search is needed
        - the complement of a maximum independent set is a minimum vertex cover

        %family Independent Sets
        %seealso maximal_independent_set, find_maximal_independent_sets, is_independent_set, max_clique
        """
        from edgecase._core import maximum_independent_set

        return maximum_independent_set(self._graph)

    def find_maximal_independent_sets(
        self,
        count: int = 10,
        seed: int | None = None,
    ) -> list[list[int]]:
        """
        Find multiple distinct maximal independent sets.

        Runs the maximal independent set algorithm multiple times with different random seeds to
        discover diverse solutions. This is useful for exploring the space of possible independent
        sets or for ensemble methods.

        Parameters
        ----------
        count
            Number of maximal independent sets to find. Default is `10`.
        seed
            Optional base seed for reproducibility. Each iteration uses seed + iteration_number.

        Returns
        -------
        list[list[int]]
            List of maximal independent sets, each represented as a list of node IDs. May return
            fewer than `count` if there aren't enough unique sets.

        Examples
        --------
        Find multiple independent sets:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_cycle_graph(8)

        sets = graph.find_maximal_independent_sets(count=5, seed=23)
        for i, s in enumerate(sets):
            print(f"Set {i}: size={len(s)}, nodes={s}")
        ```

        Find the best among multiple:

        ```python
        graph = Graph()
        graph.add_gnm_graph(n=50, m=100, seed=23)

        sets = graph.find_maximal_independent_sets(count=20, seed=0)
        best = max(sets, key=len)
        print(f"Best size: {len(best)}")
        ```

        Analyze diversity:

        ```python
        graph = Graph()
        graph.add_full_graph(8)

        sets = graph.find_maximal_independent_sets(count=10)
        unique_sizes = set(len(s) for s in sets)
        print(f"Found {len(sets)} unique sets with sizes: {unique_sizes}")
        ```

        Notes
        -----
        - Time complexity: `O(count × (V + E))`
        - Attempts to find unique sets; may return fewer than requested
        - With a seed, results are reproducible
        - Without a seed, each call may produce different sets

        %family Independent Sets
        %seealso maximal_independent_set, maximum_independent_set, is_independent_set
        """
        from edgecase._core import find_maximal_independent_sets

        return find_maximal_independent_sets(self._graph, count, seed)

    # ==============================================================================
    # Dominating Sets
    # ==============================================================================

    def is_dominating_set(self, nodes: list[int]) -> bool:
        """
        Check if a set of nodes forms a dominating set.

        A dominating set `D` of a graph `G = (V, E)` is a subset of V such that every vertex not in
        `D` is adjacent to at least one member of `D`. In other words, every node is either in `D`
        or has a neighbor in `D`.

        Parameters
        ----------
        nodes
            List of node IDs to check.

        Returns
        -------
        bool
            `True` if the nodes form a valid dominating set, `False` otherwise.

        Examples
        --------
        Check a valid dominating set:

        ```python
        from edgecase import Graph

        # Create a path: A-B-C-D
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

        # {B, D} dominates: B covers A,C and D covers C
        b_id = graph.get_id_from_label("B")
        d_id = graph.get_id_from_label("D")
        print(graph.is_dominating_set([b_id, d_id]))  # True

        # {B} alone doesn't dominate D
        print(graph.is_dominating_set([b_id]))  # False
        ```

        Validate a computed dominating set:

        ```python
        graph = Graph()
        graph.add_gnm_graph(n=50, m=100, seed=23)

        dom = graph.dominating_set()
        assert graph.is_dominating_set(dom)  # Always True
        ```

        Notes
        -----
        - time complexity: `O(V + E)` where `V` is nodes and `E` is edges
        - empty set is NOT a dominating set (unless graph is empty)
        - the entire node set is always a dominating set
        - a single node dominates only itself and its neighbors

        %family Dominating Sets
        """
        from edgecase._core import is_dominating_set

        return is_dominating_set(self._graph, nodes)

    def dominating_set(self, start_with: int | None = None) -> list[int]:
        """
        Find a dominating set for the graph.

        A dominating set `D` of a graph `G = (V, E)` is a subset of V such that every vertex not in
        `D` is adjacent to at least one member of `D`. This method uses a greedy algorithm that
        finds some dominating set, but not necessarily the minimum one.

        Parameters
        ----------
        start_with
            Optional node ID to start the algorithm from. If `None`, an arbitrary node is chosen.

        Returns
        -------
        list[int]
            List of node IDs forming a dominating set.

        Raises
        ------
        ValueError
            If the `start_with` node doesn't exist in the graph.

        Examples
        --------
        Find a dominating set:

        ```python
        from edgecase import Graph

        graph = Graph()
        graph.add_gnm_graph(n=100, m=200, seed=23)

        dom = graph.dominating_set()
        print(f"Dominating set size: {len(dom)}")

        # Verify it's valid
        assert graph.is_dominating_set(dom)
        ```

        Start from a specific node:

        ```python
        graph = Graph()
        graph.add_star_graph(10)  # Hub with 9 leaves

        # Start from the hub (node ID 1)
        dom = graph.dominating_set(start_with=1)
        print(f"Size: {len(dom)}")  # 1 (just the hub)
        ```

        Compare with independent sets:

        ```python
        graph = Graph()
        graph.add_cycle_graph(8)

        dom = graph.dominating_set()
        ind = graph.maximal_independent_set(seed=23)

        print(f"Dominating set: {len(dom)}")
        print(f"Independent set: {len(ind)}")
        # Note: Every maximal independent set is also a dominating set!
        ```

        Notes
        -----
        - time complexity: `O(V + E)` greedy algorithm
        - space complexity: `O(V)` for tracking dominated nodes
        - not guaranteed to find the minimum dominating set (NP-hard problem)
        - every maximal independent set is a dominating set
        - related to but different from `domirank_centrality()`

        %family Dominating Sets
        """
        from edgecase._core import dominating_set

        return dominating_set(self._graph, start_with)

    def min_weighted_dominating_set(self, weight_attr: str | None = None) -> list[int]:
        """
        Find an approximate minimum weighted dominating set.

        Uses a greedy algorithm that iteratively adds the most cost-effective node (lowest weight
        per number of newly dominated nodes) to the dominating set. This provides an `O(log n)`
        approximation to the optimal solution.

        Parameters
        ----------
        weight_attr
            Optional node attribute name to use as weights. If `None` or the attribute doesn't
            exist on a node, the node is assumed to have weight `1`.

        Returns
        -------
        list[int]
            List of node IDs forming an approximate minimum weighted dominating set.

        Examples
        --------
        Find minimum weighted dominating set:

        ```python
        from edgecase import Graph

        # Create graph with node weights
        graph = (
            Graph()
            .add_node(label="hub", attributes={"cost": 10})
            .add_node(label="leaf1", attributes={"cost": 1})
            .add_node(label="leaf2", attributes={"cost": 1})
            .add_node(label="leaf3", attributes={"cost": 1})
            .add_edge("hub", "leaf1")
            .add_edge("hub", "leaf2")
            .add_edge("hub", "leaf3")
        )

        # Without weights: any dominating set
        dom1 = graph.dominating_set()

        # With weights: prefer lower cost nodes
        dom2 = graph.min_weighted_dominating_set(weight_attr="cost")

        labels = [graph.get_label_from_id(n) for n in dom2]
        print(f"Weighted dominating set: {labels}")
        ```

        Facility location problem:

        ```python
        # Place facilities to cover all nodes with minimum cost
        graph = Graph()
        graph.add_grid_2d(rows=5, cols=5)

        # Assign random costs to each node
        import random
        random.seed(42)
        node_ids = graph.get_node_ids()
        costs = {nid: random.uniform(1, 10) for nid in node_ids}
        graph.set_node_attrs("facility_cost", costs)

        facilities = graph.min_weighted_dominating_set(weight_attr="facility_cost")
        print(f"Need {len(facilities)} facilities to cover all nodes")
        ```

        Notes
        -----
        - time complexity: `O(V × E)` for the greedy algorithm
        - approximation guarantee: `O(log(sum of weights)) × optimal`
        - for unweighted case (all weights = `1`), equivalent to `dominating_set()`
        - NP-hard to find the exact minimum; this is a greedy approximation

        %family Dominating Sets
        """
        from edgecase._core import min_weighted_dominating_set

        # Build weights dictionary from node attributes if specified
        weights = None
        if weight_attr is not None:
            weights = {}
            for node_id in self.get_node_ids(limit=None):
                try:
                    node = self._graph.get_node(node_id)
                    if node and node.attributes:
                        val = node.attributes.get(weight_attr)
                        if val is not None:
                            weights[node_id] = float(val)
                        else:
                            weights[node_id] = 1.0
                    else:
                        weights[node_id] = 1.0
                except (KeyError, TypeError, ValueError):
                    weights[node_id] = 1.0

        return min_weighted_dominating_set(self._graph, weights)

    # ==============================================================================
    # Minimum/Maximum Spanning Trees
    # ==============================================================================

    def minimum_spanning_tree(
        self,
        weight_attr: str | None = None,
        return_subgraph: bool = True,
    ) -> "Graph | list[str]":
        """
        Find the minimum spanning tree using Kruskal's algorithm.

        A minimum spanning tree (MST) is a subset of edges that connects all nodes with the minimum
        total edge weight. Uses the optimized union-find data structure with path-halving for
        efficient cycle detection.

        Parameters
        ----------
        weight_attr
            Name of edge attribute to use for weights. If `None` (the default), uses the edge's
            weight field. If the attribute doesn't exist on an edge, defaults to `1.0`.
        return_subgraph
            If `True`, returns a new Graph containing only the MST edges. If `False`, returns a list
            of edge IDs as strings.

        Returns
        -------
        Graph or list[str]
            Either a new Graph with MST edges or a list of edge ID strings, depending on the
            `return_subgraph=` parameter.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_node("D")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "D", weight=3.0)
        g.add_edge("D", "A", weight=4.0)
        g.add_edge("A", "C", weight=5.0)

        mst = g.minimum_spanning_tree()
        print(mst.count_edges())  # 3

        # Calculate total weight
        total = sum(e.weight for e in [mst.get_edge(eid) for eid, _, _ in mst.get_edges_as_tuples()]
                    if e and e.weight is not None)
        print(total)  # 6.0
        ```

        Notes
        -----
        - time complexity: `O(E log E)` for sorting edges
        - space complexity: `O(V)` for union-find structure
        - for disconnected graphs, finds a minimum spanning forest
        - if graph has < 2 nodes, returns empty graph or list

        %family Spanning Trees
        """
        edge_ids = self._graph.kruskal_mst_edges(minimum=True, weight_attr=weight_attr)

        if return_subgraph:
            # Create a new graph with the MST edges
            mst = Graph(directed=self.is_directed(), weighted=True)

            # Add all nodes; nodes() returns node IDs, use label_from_id() to get labels
            for node_id in self.get_node_ids(limit=None):
                label = self.get_label_from_id(node_id)
                # Try to get node attributes if they exist
                try:
                    node_obj = self._graph.get_node(label)
                    mst.add_node(
                        label,
                        attributes=node_obj.attributes if hasattr(node_obj, "attributes") else None,
                    )
                except Exception:
                    mst.add_node(label)

            # Add MST edges
            for edge_id in edge_ids:
                edge = self.get_edge(edge_id)
                if edge:
                    mst.add_edge(
                        self.get_label_from_id(edge.source),
                        self.get_label_from_id(edge.target),
                        weight=edge.weight,
                        attributes=edge.attributes,
                    )

            return mst
        else:
            return [str(eid) for eid in edge_ids]

    def maximum_spanning_tree(
        self,
        weight_attr: str | None = None,
        return_subgraph: bool = True,
    ) -> "Graph | list[str]":
        """
        Find the maximum spanning tree using Kruskal's algorithm.

        A maximum spanning tree is a subset of edges that connects all nodes with the maximum total
        edge weight. Essentially finds the MST of the negated edge weights.

        Parameters
        ----------
        weight_attr
            Name of edge attribute to use for weights. If `None` (default), uses the edge's weight
            field. If the attribute doesn't exist on an edge, defaults to `1.0`.
        return_subgraph
            If `True`, returns a new Graph containing only the spanning tree edges. If `False`,
            returns a list of edge IDs as strings.

        Returns
        -------
        Graph or list[str]
            Either a new Graph with spanning tree edges or a list of edge ID strings, depending on
            the `return_subgraph=` parameter.

        Examples
        --------
        ```python
        from edgecase import Graph

        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=3.0)
        g.add_edge("C", "A", weight=2.0)

        max_st = g.maximum_spanning_tree()

        # Calculate total weight
        total = sum(e.weight for e in [max_st.get_edge(eid) for eid, _, _ in max_st.get_edges_as_tuples()]
                    if e and e.weight is not None)
        print(total)  # 5.0 (chooses edges with weights 3.0 and 2.0)
        ```

        Notes
        -----
        - time complexity: `O(E log E)` for sorting edges
        - space complexity: `O(V)` for union-find structure
        - useful for finding most important connections in a network

        %family Spanning Trees
        """
        edge_ids = self._graph.kruskal_mst_edges(minimum=False, weight_attr=weight_attr)

        if return_subgraph:
            # Create a new graph with the spanning tree edges
            st = Graph(directed=self.is_directed(), weighted=True)

            # Add all nodes; nodes() returns node IDs, use label_from_id() to get labels
            for node_id in self.get_node_ids(limit=None):
                label = self.get_label_from_id(node_id)
                # Try to get node attributes if they exist
                try:
                    node_obj = self._graph.get_node(label)
                    st.add_node(
                        label,
                        attributes=node_obj.attributes if hasattr(node_obj, "attributes") else None,
                    )
                except Exception:
                    st.add_node(label)

            # Add spanning tree edges
            for edge_id in edge_ids:
                edge = self.get_edge(edge_id)
                if edge:
                    st.add_edge(
                        self.get_label_from_id(edge.source),
                        self.get_label_from_id(edge.target),
                        weight=edge.weight,
                        attributes=edge.attributes,
                    )

            return st
        else:
            return [str(eid) for eid in edge_ids]

    def steiner_tree(
        self,
        terminal_nodes: list[str],
        weight_attr: str | None = None,
        return_subgraph: bool = True,
    ) -> "Graph | list[tuple[str, str, float]]":
        """
        Find an approximate Steiner tree connecting the given terminal nodes.

        A Steiner tree is a minimum weight tree that connects a specified set of terminal nodes. It
        may include additional non-terminal nodes (called Steiner nodes) if they help reduce the
        total weight. This implementation uses Mehlhorn's algorithm which provides a
        2-approximation.

        Parameters
        ----------
        terminal_nodes
            List of node labels that must be connected by the tree. Must contain at least 2 nodes.
        weight_attr
            Name of edge attribute to use for weights. If `None` (default), uses the edge's weight
            field. If the attribute doesn't exist on an edge, defaults to `1.0`.
        return_subgraph
            If `True`, returns a new Graph containing only the Steiner tree edges. If `False`,
            returns a list of (source_label, target_label, weight) tuples.

        Returns
        -------
        Graph or list[tuple[str, str, float]]
            Either a new Graph with Steiner tree edges or a list of edge tuples, depending on the
            `return_subgraph=` parameter.

        Raises
        ------
        ValueError
            If fewer than 2 terminal nodes are provided, if any terminal node doesn't exist in the
            graph, or if the terminal nodes are not all connected.

        Examples
        --------
        **Network Design: Connecting Data Centers**

        Find the minimum cost network to connect specific data centers:

        ```python
        from edgecase import Graph

        # Create a network topology with connection costs
        g = Graph(weighted=True)
        for node in ["DC1", "DC2", "DC3", "Router1", "Router2", "Router3"]:
            g.add_node(node)

        # Add network links with costs
        g.add_edge("DC1", "Router1", weight=10.0)
        g.add_edge("DC2", "Router1", weight=15.0)
        g.add_edge("DC3", "Router2", weight=12.0)
        g.add_edge("Router1", "Router2", weight=5.0)
        g.add_edge("Router2", "Router3", weight=8.0)
        g.add_edge("Router1", "Router3", weight=20.0)

        # Find minimum cost tree connecting the data centers
        terminals = ["DC1", "DC2", "DC3"]
        steiner = g.steiner_tree(terminals)

        print(f"Steiner tree has {steiner.count_edges()} edges")
        # May include Router1 and Router2 as Steiner nodes
        ```

        **Phylogenetics: Connecting Key Species**

        Connect a subset of species with minimum evolutionary distance:

        ```python
        from edgecase import Graph

        g = Graph(weighted=True)
        species = ["Human", "Chimp", "Gorilla", "Mouse", "CommonAncestor1", "CommonAncestor2"]
        for s in species:
            g.add_node(s)

        g.add_edge("Human", "CommonAncestor1", weight=1.0)
        g.add_edge("Chimp", "CommonAncestor1", weight=1.2)
        g.add_edge("CommonAncestor1", "CommonAncestor2", weight=3.0)
        g.add_edge("Gorilla", "CommonAncestor2", weight=2.5)
        g.add_edge("Mouse", "CommonAncestor2", weight=10.0)

        # Find tree connecting just the primates
        primates = ["Human", "Chimp", "Gorilla"]
        tree = g.steiner_tree(primates)
        ```

        Notes
        -----
        - time complexity: `O(|T| * (E log V))` where `T` is the number of terminal nodes
        - space complexity: `O(V + E)`
        - uses Mehlhorn's 2-approximation algorithm
        - the approximation factor of 2 means the result is at most twice the optimal weight
        - for graphs with only 2 terminal nodes, returns the shortest path between them

        %family Spanning Trees
        """
        from edgecase._core import steiner_tree as _steiner_tree

        edges = _steiner_tree(self._graph, terminal_nodes, weight_attr)

        if return_subgraph:
            # Create a new graph with the Steiner tree edges
            st = Graph(directed=self.is_directed(), weighted=True)

            # Collect all nodes that appear in the tree
            tree_nodes = set()
            for source, target, _ in edges:
                tree_nodes.add(source)
                tree_nodes.add(target)

            # Add the tree nodes
            for label in tree_nodes:
                try:
                    node_obj = self._graph.get_node(label)
                    st.add_node(
                        label,
                        attributes=node_obj.attributes if hasattr(node_obj, "attributes") else None,
                    )
                except Exception:
                    st.add_node(label)

            # Add the Steiner tree edges
            for source, target, weight in edges:
                st.add_edge(source, target, weight=weight)

            return st
        else:
            return edges

    def all_spanning_trees(
        self,
        max_trees: int | None = None,
        weight_attr: str | None = None,
        return_subgraphs: bool = True,
    ) -> "list[Graph] | list[list[tuple[str, str, float]]]":
        """
        Enumerate all spanning trees of the graph.

        A spanning tree is a subgraph that connects all vertices with the minimum number of edges
        (`V-1` edges for `V` vertices) and contains no cycles. This method enumerates all possible
        spanning trees, which is useful for reliability analysis and network design.

        Parameters
        ----------
        max_trees
            Maximum number of spanning trees to return. If `None`, defaults to `1000` to prevent
            excessive memory usage. Set to a specific value for different limits.
        weight_attr
            Name of edge attribute to use for weights in the returned edges. If `None` (default),
            uses the edge's weight field. If the attribute doesn't exist, defaults to `1.0`.
        return_subgraphs
            If `True`, returns a list of Graph objects, each representing a spanning tree. If
            `False`, returns a list of edge lists (each edge is a (source, target, weight) tuple).

        Returns
        -------
        list[Graph] or list[list[tuple[str, str, float]]]
            Either a list of Graph objects or a list of edge lists, depending on the
            `return_subgraphs=` parameter.

        Raises
        ------
        ValueError
            If the graph is disconnected (no spanning tree exists) or empty.

        Examples
        --------
        **Reliability Analysis: Finding Backup Paths**

        Enumerate all ways to connect a network:

        ```python
        from edgecase import Graph

        # Create a simple network
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)

        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "A")
        g.add_edge("A", "C")

        # Get all spanning trees
        trees = g.all_spanning_trees()
        print(f"Found {len(trees)} spanning trees")

        # Each tree connects all 4 nodes with exactly 3 edges
        for i, tree in enumerate(trees):
            print(f"Tree {i+1}: {tree.count_edges()} edges")
        ```

        **Counting Spanning Trees**

        Count the number of spanning trees (related to the matrix tree theorem):

        ```python
        from edgecase import Graph

        g = Graph()
        # Create a complete graph K4
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            g.add_node(n)

        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                g.add_edge(n1, n2)

        # K4 should have 16 spanning trees
        trees = g.all_spanning_trees()
        print(f"K4 has {len(trees)} spanning trees")  # 16
        ```

        **Getting Edge Lists Instead of Graphs**

        For more efficient processing of many trees:

        ```python
        from edgecase import Graph

        g = Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=2.0)
        g.add_edge("C", "A", weight=3.0)

        # Get edge lists for efficiency
        tree_edges = g.all_spanning_trees(return_subgraphs=False)
        for edges in tree_edges:
            total_weight = sum(w for _, _, w in edges)
            print(f"Tree weight: {total_weight}")
        ```

        Notes
        -----
        - time complexity: `O(N * E)` where `N` is the number of spanning trees
        - space complexity: `O(N * V)` for storing all trees
        - the number of spanning trees can grow exponentially with graph size
        - for dense graphs, the default `max_trees=1000` prevents memory issues
        - uses a systematic edge enumeration algorithm based on Kruskal's approach

        Warnings
        --------
        The number of spanning trees can be extremely large. For a complete graph `Kn`, there are
        `n^(n-2)` spanning trees (Cayley's formula). Use `max_trees=` to limit results.

        %family Spanning Trees
        """
        from edgecase._core import all_spanning_trees as _all_spanning_trees

        # Default max_trees to 1000 if not specified
        effective_max = max_trees if max_trees is not None else 1000

        tree_edge_lists = _all_spanning_trees(self._graph, effective_max, weight_attr)

        if return_subgraphs:
            result = []
            for edges in tree_edge_lists:
                # Create a new graph for this spanning tree
                st = Graph(directed=self.is_directed(), weighted=True)

                # Add all nodes from the original graph
                for node_id in self.get_node_ids(limit=None):
                    label = self.get_label_from_id(node_id)
                    try:
                        node_obj = self._graph.get_node(label)
                        st.add_node(
                            label,
                            attributes=node_obj.attributes
                            if hasattr(node_obj, "attributes")
                            else None,
                        )
                    except Exception:
                        st.add_node(label)

                # Add the spanning tree edges
                for source, target, weight in edges:
                    st.add_edge(source, target, weight=weight)

                result.append(st)

            return result
        else:
            return tree_edge_lists

    # ==============================================================================
    # ------------------------------------------------------------------------------
    # EDGE & NETWORK ANALYSIS
    # ------------------------------------------------------------------------------
    # ==============================================================================

    # ==============================================================================
    # Edge Type Analysis
    # ==============================================================================

    def which_multiple(self) -> list[int]:
        """
        Find edge IDs of multiple edges (duplicate edges between same node pairs).

        Returns a list of edge IDs that represent multiple/duplicate edges. These edges share the
        same source and target nodes as another edge in the graph. This is useful for detecting and
        cleaning multigraphs.

        For undirected graphs, edges `(u,v)` and `(v,u)` are considered the same pair. When multiple
        edges exist between a pair, all edge IDs (including the first) are returned.

        Returns
        -------
        list[int]
            List of edge IDs representing multiple edges. Empty if no duplicates exist.

        Examples
        --------
        **Data Cleaning: Finding Duplicate Relationships**

        ```python
        from edgecase import Graph

        g = Graph(directed=True)
        g.add_node(label="Alice")
        g.add_node(label="Bob")
        g.add_node(label="Carol")
        g.add_edge(source="Alice", target="Bob", weight=1.0)
        g.add_edge(source="Alice", target="Bob", weight=2.0)  # Duplicate connection
        g.add_edge(source="Alice", target="Bob", weight=3.0)  # Another duplicate
        g.add_edge(source="Bob", target="Carol", weight=1.0)  # Single edge

        multiples = g.which_multiple()
        len(multiples)  # All 3 Alice->Bob edges
        # 3
        ```

        **Multigraph Validation**

        ```python
        # Check if graph is a simple graph (no multiple edges)
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="A")
        )

        is_simple = len(g.which_multiple()) == 0
        is_simple
        # True
        ```

        **Network Simplification: Identifying Edges to Merge**

        ```python
        # Social network with multiple interactions
        social = (
            Graph(directed=True)
            .add_node(label="User1")
            .add_node(label="User2")
            .add_node(label="User3")
            .add_edge(source="User1", target="User2", interaction="like")
            .add_edge(source="User1", target="User2", interaction="comment")
            .add_edge(source="User1", target="User2", interaction="share")
            .add_edge(source="User2", target="User3", interaction="like")
        )

        # Find which edges represent multiple interactions
        multiple_edges = social.which_multiple()
        # These could be merged into single weighted edges
        len(multiple_edges)
        # 3
        ```

        **Undirected Graph: Bidirectional Edges**

        ```python
        g = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="B")  # Duplicate in undirected graph
            .add_edge(source="B", target="C")
        )

        multiples = g.which_multiple()
        len(multiples)
        # 2
        ```

        Notes
        -----
        - self-loops are not considered multiple edges (use `which_loops()` instead)
        - in undirected graphs, (u,v) and (v,u) are treated as the same pair
        - returns all edges involved in duplication, including the first occurrence
        - useful for data quality checks before analysis

        %family Edge Analysis
        """
        return self._graph.which_multiple()

    def count_multiple(self) -> dict[tuple[int, int], int]:
        """
        Count multiple edges grouped by node pair.

        Returns a dictionary showing how many edges exist between each pair of nodes
        that has multiple edges. Only includes pairs with 2 or more edges.

        For undirected graphs, returns unordered pairs (smaller node ID first).

        Returns
        -------
        dict[tuple[int, int], int]
            Dictionary mapping (source_id, target_id) to edge count.
            Only includes pairs with count >= 2.

        Examples
        --------
        **Data Quality Report**

        ```python
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B")
        g.add_edge(source="A", target="B")  # Duplicate
        g.add_edge(source="A", target="B")  # Another duplicate
        g.add_edge(source="B", target="C")
        g.add_edge(source="B", target="C")  # Duplicate

        counts = g.count_multiple()
        # Returns node IDs: {(node_id_a, node_id_b): 3, (node_id_b, node_id_c): 2}

        # Convert to labels if needed
        counts_by_label = {
            (g.get_label_from_id(src), g.get_label_from_id(tgt)): count
            for (src, tgt), count in counts.items()
        }
        counts_by_label
        # {('A', 'B'): 3, ('B', 'C'): 2}
        ```

        **Communication Network Analysis**

        ```python
        # Email network with multiple messages
        email = Graph(directed=True)
        email.add_node(label="Alice")
        email.add_node(label="Bob")
        email.add_node(label="Carol")
        email.add_edge(source="Alice", target="Bob")    # Email 1
        email.add_edge(source="Alice", target="Bob")    # Email 2
        email.add_edge(source="Alice", target="Bob")    # Email 3
        email.add_edge(source="Bob", target="Carol")    # Email 1
        email.add_edge(source="Bob", target="Carol")    # Email 2

        multi_count = email.count_multiple()
        # Returns node IDs: {(alice_id, bob_id): 3, (bob_id, carol_id): 2}

        # Convert to labels for display
        for (sender_id, receiver_id), count in multi_count.items():
            sender = email.get_label_from_id(sender_id)
            receiver = email.get_label_from_id(receiver_id)
            print(f"{sender} -> {receiver}: {count} emails")
        # Alice -> Bob: 3 emails
        # Bob -> Carol: 2 emails
        ```

        **Network Simplification Planning**

        ```python
        # Identify which edges need aggregation
        transport = Graph(directed=False)
        transport.add_node(label="CityA")
        transport.add_node(label="CityB")
        transport.add_node(label="CityC")
        transport.add_edge(source="CityA", target="CityB", route="highway")
        transport.add_edge(source="CityA", target="CityB", route="railway")
        transport.add_edge(source="CityA", target="CityB", route="airline")
        transport.add_edge(source="CityB", target="CityC", route="highway")

        multi = transport.count_multiple()
        # Returns node IDs: {(cityA_id, cityB_id): 3}

        # Convert to labels
        multi_labels = {
            (transport.get_label_from_id(id1), transport.get_label_from_id(id2)): count
            for (id1, id2), count in multi.items()
        }
        # CityA-CityB has 3 routes to consolidate
        multi_labels
        # {('CityA', 'CityB'): 3}
        ```

        Notes
        -----
        - only returns pairs with 2+ edges (single edges are excluded)
        - for undirected graphs, pairs are normalized (smaller node ID first)
        - self-loops are counted if there are 2+ self-loop edges for a node
        - useful for understanding graph complexity before simplification

        %family Edge Analysis
        """
        return self._graph.count_multiple()

    def which_loops(self) -> list[int]:
        """
        Find edge IDs of self-loops (edges from a node to itself).

        Returns a list of edge IDs for all edges where the source node equals the target node.
        Self-loops are common in certain network types like state machines, chemical reactions, or
        social networks (self-mentions).

        Returns
        -------
        list[int]
            List of edge IDs representing self-loops. Empty if no self-loops exist.

        Examples
        --------
        Data cleaning and finding self-references:

        ```python
        from edgecase import Graph

        g = Graph(directed=True)
        g.add_node(label="Alice")
        g.add_node(label="Bob")
        g.add_node(label="Carol")
        g.add_edge(source="Alice", target="Bob")
        g.add_edge(source="Alice", target="Alice")  # Self-loop
        g.add_edge(source="Bob", target="Carol")
        g.add_edge(source="Carol", target="Carol")  # Self-loop

        loops = g.which_loops()
        print(len(loops))  # 2
        ```

        State Machine Validation:

        ```python
        from edgecase import Graph

        # Check for states that transition to themselves
        fsm = Graph(directed=True)
        fsm.add_node(label="Idle")
        fsm.add_node(label="Processing")
        fsm.add_node(label="Complete")
        fsm.add_edge(source="Idle", target="Processing")
        fsm.add_edge(source="Processing", target="Processing")  # Wait state
        fsm.add_edge(source="Processing", target="Complete")
        fsm.add_edge(source="Complete", target="Idle")

        self_transitions = fsm.which_loops()
        print(len(self_transitions))  # 1
        ```

        **Social Network Analysis**

        ```python
        from edgecase import Graph

        # Find self-mentions or self-citations
        social = Graph(directed=True)
        social.add_node(label="@alice")
        social.add_node(label="@bob")
        social.add_node(label="@carol")
        social.add_edge(source="@alice", target="@bob")
        social.add_edge(source="@alice", target="@alice")  # Self-mention
        social.add_edge(source="@bob", target="@carol")

        self_mentions = social.which_loops()
        # These might be removed for certain analyses
        len(self_mentions)
        # 1
        ```

        **Network Simplification**

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="B")  # Self-loop
        g.add_edge(source="C", target="C")  # Self-loop

        # Identify edges to remove for simple graph
        loops_to_remove = g.which_loops()
        len(loops_to_remove)
        # 2
        ```

        Notes
        -----
        - works for both directed and undirected graphs
        - a node can have multiple self-loops (use `count_loops()` to count per node)
        - self-loops affect certain graph metrics (degree, centrality, etc.)
        - common preprocessing step is to remove self-loops for cleaner analysis

        %family Edge Analysis
        """
        return self._graph.which_loops()

    def count_loops(self) -> dict[int, int]:
        """
        Count self-loops grouped by node.

        Returns a dictionary mapping node IDs to the count of self-loops for each node.
        Only includes nodes that have at least one self-loop.

        Returns
        -------
        dict[int, int]
            Dictionary mapping node IDs to self-loop counts.
            Only includes nodes with count >= 1.

        Examples
        --------
        **Data Quality Analysis**

        ```python
        from edgecase import Graph

        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge(source="A", target="B")
        g.add_edge(source="A", target="A")  # Self-loop
        g.add_edge(source="A", target="A")  # Another self-loop
        g.add_edge(source="B", target="B")  # Self-loop
        g.add_edge(source="C", target="D")  # No self-loop

        loop_counts = g.count_loops()
        # Returns node IDs: {a_id: 2, b_id: 1}

        # Convert to labels if needed
        loop_counts_labels = {g.get_label_from_id(nid): count for nid, count in loop_counts.items()}
        loop_counts_labels
        # {'A': 2, 'B': 1}
        ```

        **State Machine Analysis**

        ```python
        from edgecase import Graph

        # Identify states with self-transitions
        fsm = Graph(directed=True)
        fsm.add_node(label="Loading")
        fsm.add_node(label="Ready")
        fsm.add_node(label="Processing")
        fsm.add_node(label="Complete")
        fsm.add_edge(source="Loading", target="Loading")  # Retry loop
        fsm.add_edge(source="Loading", target="Loading")  # Another retry path
        fsm.add_edge(source="Loading", target="Ready")
        fsm.add_edge(source="Processing", target="Processing")  # Busy wait
        fsm.add_edge(source="Processing", target="Complete")

        wait_states = fsm.count_loops()
        # Returns node IDs with self-transitions: {loading_id: 2, processing_id: 1}

        # Convert to labels
        wait_states_labels = {fsm.get_label_from_id(nid): count for nid, count in wait_states.items()}
        wait_states_labels
        # {'Loading': 2, 'Processing': 1}
        ```

        **Chemical Reaction Network**

        ```python
        # Autocatalytic reactions (self-referencing)
        rxn = Graph(directed=True)
        rxn.add_node(label="A")
        rxn.add_node(label="B")
        rxn.add_node(label="C")
        rxn.add_edge(source="A", target="B")
        rxn.add_edge(source="B", target="B")  # B catalyzes itself
        rxn.add_edge(source="B", target="B")  # Multiple pathways
        rxn.add_edge(source="C", target="C")  # C is autocatalytic

        autocatalytic = rxn.count_loops()
        # Returns node IDs: {b_id: 2, c_id: 1}

        # Convert to labels
        autocatalytic_labels = {rxn.get_label_from_id(nid): count for nid, count in autocatalytic.items()}
        autocatalytic_labels
        # {'B': 2, 'C': 1}
        ```

        **social network: self-interactions**

        ```python
        social = Graph(directed=True)
        social.add_node(label="User1")
        social.add_node(label="User2")
        social.add_edge(source="User1", target="User1", action="self-like")
        social.add_edge(source="User2", target="User1", action="like")
        social.add_edge(source="User2", target="User2", action="self-like")
        social.add_edge(source="User2", target="User2", action="self-comment")

        self_interactions = social.count_loops()
        # Returns node IDs: {user1_id: 1, user2_id: 2}

        # Convert to labels
        self_interactions_labels = {social.get_label_from_id(nid): count for nid, count in self_interactions.items()}
        self_interactions_labels
        # {'User1': 1, 'User2': 2}
        ```

        Notes
        -----
        - only returns nodes with at least one self-loop
        - a node can have multiple self-loops (all counted)
        - useful for identifying unusual patterns in network data
        - common in multigraphs where multiple self-edges may exist

        %family Edge Analysis
        """
        return self._graph.count_loops()

    # ==============================================================================
    # Reciprocity and Mutuality
    # ==============================================================================

    def which_mutual(self) -> list[tuple[int, int]]:
        """
        Find pairs of mutual (reciprocal) edges in directed graph.

        Identifies edge pairs where both A->B and B->A exist, indicating bidirectional
        connections. Only meaningful for directed graphs. Each pair is returned once
        with edge IDs in ascending order.

        This is useful for analyzing reciprocity patterns in social networks,
        communication graphs, and other directed relationships.

        Returns
        -------
        list[tuple[int, int]]
            List of (edge_id1, edge_id2) tuples representing mutual edge pairs.
            Empty list if no mutual edges exist.

        Raises
        ------
        ValueError
            If called on an undirected graph.

        Examples
        --------
        Find mutual edges in a social network:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Carol")
            .add_edge("Alice", "Bob")   # Edge 1
            .add_edge("Bob", "Alice")   # Edge 2: mutual with 1
            .add_edge("Alice", "Carol") # Edge 3: no reciprocal
        )

        mutual = graph.which_mutual()
        print(mutual)  # [(1, 2)]
        ```

        Analyze communication patterns:

        ```python
        from edgecase import Graph

        graph = (
            Graph(directed=True)
            .add_node(label="Alice")
            .add_node(label="Bob")
            .add_node(label="Carol")
            .add_edge(source="Alice", target="Bob")   # Edge 1
            .add_edge(source="Bob", target="Alice")   # Edge 2: mutual with 1
            .add_edge(source="Alice", target="Carol") # Edge 3: no reciprocal
            .add_edge(source="Bob", target="Carol")   # Edge 4
            .add_edge(source="Carol", target="Bob")   # Edge 5: mutual with 4
        )

        # Find bidirectional communication
        mutual_pairs = graph.which_mutual()
        print(f"Bidirectional connections: {len(mutual_pairs)}")

        # Get details of mutual relationships
        for edge1_id, edge2_id in mutual_pairs:
            e1 = graph.get_node_ids_by_edge_id(edge_id=edge1_id)
            e2 = graph.get_node_ids_by_edge_id(edge_id=edge2_id)
            if e1:
                print(f"Mutual: {e1[0]} ↔ {e1[1]}")
        ```

        %family Reciprocity
        """
        return self._graph.which_mutual()

    def get_reciprocity(self, mode: str = "default") -> float:
        """
        Calculate reciprocity (fraction of edges with reciprocal counterpart).

        Reciprocity measures the proportion of edges in a directed graph that are bidirectional
        (mutual). Returns a value between `0.0` (no reciprocal edges) and `1.0` (all edges are
        reciprocal).

        This metric is fundamental for understanding relationship symmetry in social networks,
        mutual influence in citation networks, and bidirectional flows in transportation or
        communication networks.

        Parameters
        ----------
        mode
            Calculation mode for reciprocity:

            - `"default"` or `"ratio"`: (# mutual edges) / (total # edges)

        Returns
        -------
        float
            Reciprocity value between `0.0` and `1.0`, or `0.0` for empty graphs.

        Raises
        ------
        ValueError
            If called on an undirected graph or with invalid mode.

        Examples
        --------
        Measure reciprocity in a social network:

        ```python
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "A")  # Mutual with above
            .add_edge("A", "C")  # No reciprocal
        )

        recip = graph.get_reciprocity()
        print(f"Reciprocity: {recip:.3f}")  # 0.667 (2 mutual out of 3 edges)
        ```

        Compare different networks:

        ```python
        from edgecase import Graph

        # Highly reciprocal network (e.g., friendships)
        friendship = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="B")
            .add_edge(source="A", target="C")
            .add_edge(source="C", target="A")
        )
        print(f"Friendship reciprocity: {friendship.get_reciprocity():.2f}")  # 1.00

        # Low reciprocity network (e.g., followers)
        followers = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
            .add_edge(source="B", target="C")
            .add_edge(source="D", target="A")
        )
        print(f"Follower reciprocity: {followers.get_reciprocity():.2f}")  # 0.00
        ```

        Analyze communication patterns:

        ```python
        # Email network with some mutual communication
        email = Graph(directed=True)
        # ... add edges ...

        recip = email.get_reciprocity()
        if recip > 0.5:
            print("High mutual communication")
        elif recip > 0.2:
            print("Moderate mutual communication")
        else:
            print("Mostly one-way communication")
        ```

        %family Reciprocity
        """
        return self._graph.reciprocity(mode)

    # ==============================================================================
    # Assortativity Measures
    # ==============================================================================

    def get_assortativity_degree(self) -> float:
        """
        Calculate degree assortativity coefficient.

        Measures the tendency of nodes to connect to other nodes with similar degrees. Returns the
        Pearson correlation coefficient of node degrees across edges, quantifying whether
        high-degree nodes preferentially connect to other high-degree nodes (assortative) or to
        low-degree nodes (disassortative).

        For directed graphs, uses out-degree for source nodes and in-degree for target nodes. For
        undirected graphs, uses total degree.

        Returns
        -------
        float
            Assortativity coefficient in range [-1, 1]:

            - r > 0: assortative mixing (similar degrees connect)
            - r < 0: disassortative mixing (different degrees connect)
            - r ≈ 0: no degree correlation

        Raises
        ------
        ValueError
            If graph has no edges or insufficient degree variance.

        Examples
        --------
        **Social Network: Assortative Mixing**

        ```python
        # High-status users connect to high-status users
        g = Graph(directed=False)
        for label in ["Influencer1", "Influencer2", "Influencer3", "User1", "User2", "User3", "User4"]:
            g.add_node(label=label)
        g.add_edge(source="Influencer1", target="Influencer2")
        g.add_edge(source="Influencer1", target="Influencer3")
        g.add_edge(source="Influencer2", target="Influencer3")
        g.add_edge(source="User1", target="User2")
        g.add_edge(source="User3", target="User4")

        r = g.get_assortativity_degree()
        # r > 0: hubs connect to hubs, periphery to periphery
        ```

        **Technology Network: Disassortative Mixing**

        ```python
        # Client-server architecture: high-degree servers connect to low-degree clients
        g = Graph(directed=False)
        for label in ["Server1", "Server2", "Client1", "Client2", "Client3", "Client4", "Client5"]:
            g.add_node(label=label)
        g.add_edge(source="Server1", target="Client1")
        g.add_edge(source="Server1", target="Client2")
        g.add_edge(source="Server1", target="Client3")
        g.add_edge(source="Server2", target="Client4")
        g.add_edge(source="Server2", target="Client5")

        r = g.get_assortativity_degree()
        # r < 0: high-degree nodes connect to low-degree nodes
        ```

        **Random Network**

        ```python
        from edgecase import Graph
        import random

        # Erdős-Rényi random graph typically shows no correlation
        random.seed(42)

        g = Graph(directed=False)
        nodes = [f"N{i}" for i in range(20)]
        for node in nodes:
            g.add_node(label=node)
        for i in range(30):
            u, v = random.sample(nodes, 2)
            g.add_edge(source=u, target=v)

        r = g.get_assortativity_degree()
        # r ≈ 0: no systematic degree correlation
        ```

        Notes
        -----
        - uses Newman's formula for assortativity coefficient
        - most real-world social networks are assortative (`r > 0`)
        - many technological/biological networks are disassortative (`r < 0`)
        - perfect assortativity (`r = 1`) means only equal-degree nodes connect
        - requires at least one edge and variance in node degrees

        %family Assortativity
        """
        return self._graph.assortativity_degree()

    def get_assortativity_nominal(self, attribute: str) -> float:
        """
        Calculate assortativity for categorical (nominal) node attributes.

        Measures the tendency of nodes with the same attribute value to connect, also known as
        homophily. Uses Newman's formula for categorical mixing patterns.

        Parameters
        ----------
        attribute
            Name of the categorical node attribute to analyze. Must exist on all nodes as a node
            attribute.

        Returns
        -------
        float
            Assortativity coefficient in range `[-1, 1]`:

            - `r = 1`: perfect assortativity (only same-type connections)
            - `r > 0`: assortative (same types prefer to connect)
            - `r = 0`: random mixing
            - `r < 0`: disassortative (different types prefer to connect)

        Raises
        ------
        ValueError
            If attribute doesn't exist on all nodes, graph has no edges, or there are fewer than two
            different attribute values.

        Examples
        --------
        **Homophily in Social Networks**

        ```python
        # People tend to befriend similar people
        g = Graph(directed=False)
        g.add_node("Alice", group="students")
        g.add_node("Bob", group="students")
        g.add_node("Carol", group="students")
        g.add_node("Prof1", group="faculty")
        g.add_node("Prof2", group="faculty")

        # Mostly within-group friendships
        g.add_edge("Alice", "Bob")
        g.add_edge("Bob", "Carol")
        g.add_edge("Prof1", "Prof2")
        g.add_edge("Alice", "Prof1")  # One cross-group edge

        r = g.get_assortativity_nominal("group")
        # r > 0: since students connect to students, faculty to faculty
        ```

        **Mixing Patterns by Gender**

        ```python
        # Collaboration network
        g = Graph(directed=False)
        g.add_node("Alice", gender="F")
        g.add_node("Bob", gender="M")
        g.add_node("Carol", gender="F")
        g.add_node("Dave", gender="M")

        g.add_edge("Alice", "Bob")
        g.add_edge("Bob", "Dave")
        g.add_edge("Carol", "Dave")
        g.add_edge("Alice", "Carol")

        r = g.get_assortativity_nominal("gender")
        # Measures gender homophily in collaboration
        ```

        **Biological Networks: Species Interactions**

        ```python
        # Predator-prey relationships are disassortative
        g = Graph(directed=True)
        g.add_node("Wolf", type="predator")
        g.add_node("Lion", type="predator")
        g.add_node("Deer", type="prey")
        g.add_node("Rabbit", type="prey")

        g.add_edge("Wolf", "Deer")
        g.add_edge("Wolf", "Rabbit")
        g.add_edge("Lion", "Deer")

        r = g.get_assortativity_nominal("type")
        # r < 0: predators connect to prey, not to each other
        ```

        **Network Segregation Analysis**

        ```python
        # Measure segregation in communication network
        g = Graph(directed=False)
        g.add_node("A", dept="Engineering")
        g.add_node("B", dept="Engineering")
        g.add_node("C", dept="Marketing")
        g.add_node("D", dept="Marketing")

        # All edges within departments
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        r = g.get_assortativity_nominal("dept")
        # r = 1.0: perfect segregation by department
        ```

        Notes
        -----
        - also called "modularity by attribute" or "homophily coefficient"
        - `r = 1` means complete segregation (no cross-type edges)
        - common in social networks where "birds of a feather flock together"
        - can be used to detect and measure community structure by attribute
        - for undirected graphs, the mixing matrix is symmetric

        %family Assortativity
        """
        return self._graph.get_assortativity_nominal(attribute)

    def get_assortativity(self, attribute: str) -> float:
        """
        Calculate assortativity for numeric node attributes.

        Measures the tendency of nodes with similar numeric attribute values to connect. Returns the
        Pearson correlation coefficient of attribute values across edges, indicating whether similar
        values tend to be connected.

        Parameters
        ----------
        attribute
            Name of the numeric node attribute to analyze. Must exist on all nodes and be
            convertible to float.

        Returns
        -------
        float
            Assortativity coefficient in range `[-1, 1]`:

            - `r = 1`: perfect positive correlation (identical values connect)
            - `r > 0`: assortative (similar values prefer to connect)
            - `r = 0`: no correlation
            - `r < 0`: disassortative (dissimilar values prefer to connect)
            - `r = -1`: perfect negative correlation

        Raises
        ------
        ValueError
            If attribute doesn't exist on all nodes, isn't numeric, graph has no edges, or attribute
            has no variance.

        Examples
        --------
        **Age-Based Friendship Networks**

        ```python
        # People befriend others of similar age
        g = Graph(directed=False)
        g.add_node("Alice", age=25)
        g.add_node("Bob", age=27)
        g.add_node("Carol", age=26)
        g.add_node("Dave", age=55)
        g.add_node("Eve", age=53)

        g.add_edge("Alice", "Bob")  # Similar ages (25-27)
        g.add_edge("Bob", "Carol")  # Similar ages (27-26)
        g.add_edge("Dave", "Eve")   # Similar ages (55-53)
        g.add_edge("Alice", "Carol")  # Similar ages (25-26)

        r = g.get_assortativity("age")
        # r > 0: people tend to befriend similar-aged people
        ```

        **Academic Collaboration by Experience**

        ```python
        # Citation network by publication count
        g = Graph(directed=False)
        g.add_node("Junior1", papers=5)
        g.add_node("Junior2", papers=7)
        g.add_node("Senior1", papers=50)
        g.add_node("Senior2", papers=45)

        # Cross-level mentorship
        g.add_edge("Senior1", "Junior1")
        g.add_edge("Senior1", "Junior2")
        g.add_edge("Senior2", "Junior1")

        r = g.get_assortativity("papers")
        # r < 0: experienced researchers often collaborate with juniors
        ```

        **Wealth Inequality in Trade Networks**

        ```python
        # Economic network with GDP
        g = Graph(directed=False)
        g.add_node("USA", gdp=21.0)
        g.add_node("China", gdp=14.7)
        g.add_node("Germany", gdp=3.8)
        g.add_node("India", gdp=2.9)
        g.add_node("Brazil", gdp=1.8)

        g.add_edge("USA", "China")
        g.add_edge("Germany", "India")
        g.add_edge("USA", "Germany")

        r = g.get_assortativity("gdp")
        # Measures if similar-wealth countries trade more
        ```

        **Protein Interaction Networks**

        ```python
        # Proteins by molecular weight
        g = Graph(directed=False)
        g.add_node("P1", weight=25.5)
        g.add_node("P2", weight=27.3)
        g.add_node("P3", weight=52.1)
        g.add_node("P4", weight=51.8)

        g.add_edge("P1", "P2")  # Similar weights
        g.add_edge("P3", "P4")  # Similar weights

        r = g.get_assortativity("weight")
        # r > 0: similar-sized proteins tend to interact
        ```

        Notes
        -----
        - uses Pearson correlation coefficient formula
        - common in social networks (homophily by age, income, education)
        - can reveal hierarchical structures when negative
        - sensitive to outliers: considers data normalization
        - requires variance in attribute values (all same value will error)

        %family Assortativity
        """
        return self._graph.assortativity(attribute)

    # ==============================================================================
    # Spectral Analysis
    # ==============================================================================

    def get_spectrum(
        self,
        which: str = "largest",
        k: int | None = None,
    ) -> dict[str, list[float] | list[list[float]] | float | None]:
        """
        Compute eigenvalues and eigenvectors of the adjacency matrix.

        Returns the spectrum (eigenvalues and eigenvectors) of the graph's adjacency matrix. The
        spectrum reveals important structural properties including connectivity, bipartiteness, and
        graph regularity.

        Parameters
        ----------
        which
            Which eigenvalues to compute:

            - `"all"`: All `n` eigenvalues (`O(n³)`, recommended for `n < 100`)
            - `"largest"`: Largest k eigenvalues (default `k=6`)
            - `"smallest"`: Smallest k eigenvalues
        k
            Number of eigenvalues to compute for "largest"/"smallest".
            Default is `min(6, n-1)` where `n` is the number of nodes.

        Returns
        -------
        dict
            Dictionary with keys:
            - `"values"`: List of eigenvalues (sorted descending)
            - `"vectors"`: List of eigenvectors (each as list of floats)
            - `"spectral_gap"`: λ₁ - λ₂ (difference between two largest eigenvalues)

        Raises
        ------
        ValueError
            If graph is empty or eigenvalue computation fails.

        Examples
        --------
        **Spectral Analysis of Complete Graph**

        ```python
        # Complete graph K_5
        g = Graph(directed=False)

        for i in range(5):
            g.add_node(f"v{i}")
        for i in range(5):
            for j in range(i+1, 5):
                g.add_edge(f"v{i}", f"v{j}")

        spec = g.get_spectrum(which="all")
        print(f"Eigenvalues: {spec['values']}")

        # Complete graph: λ₁ = n-1, other λᵢ = -1
        print(f"Spectral gap: {spec['spectral_gap']}")
        # K_n has spectral gap = n
        ```

        **Detecting Bipartite Structure**

        ```python
        # Bipartite graph has symmetric spectrum
        g = Graph(directed=False)

        # Left partition: A, B
        # Right partition: X, Y, Z
        for node in ["A", "B", "X", "Y", "Z"]:
            g.add_node(node)
        g.add_edge("A", "X")
        g.add_edge("A", "Y")
        g.add_edge("B", "Y")
        g.add_edge("B", "Z")

        spec = g.get_spectrum(which="all")

        # Bipartite: λ and -λ appear in pairs
        # Check if λ_max ≈ -λ_min
        values = spec['values']
        is_bipartite = abs(values[0] + values[-1]) < 0.01
        ```

        **Spectral Clustering Using Leading Eigenvector**

        ```python
        from edgecase import Graph

        # Use second eigenvector (Fiedler vector) for graph clustering
        g = Graph(directed=False)

        # Add two communities with a weak bridge
        # Community 1: nodes 0-4
        for i in range(5):
            g.add_node(label=f"n{i}")
        for i in range(5):
            for j in range(i+1, 5):
                g.add_edge(source=f"n{i}", target=f"n{j}")

        # Community 2: nodes 5-9
        for i in range(5, 10):
            g.add_node(label=f"n{i}")
        for i in range(5, 10):
            for j in range(i+1, 10):
                g.add_edge(source=f"n{i}", target=f"n{j}")

        # Bridge
        g.add_edge(source="n4", target="n5")

        spec = g.get_spectrum(which="largest", k=2)
        # Use eigenvector to partition graph
        fiedler_vector = spec['vectors'][1]  # Second eigenvector
        partition = [1 if v > 0 else 0 for v in fiedler_vector]
        ```

        **Computing Spectral Radius**

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        for i in range(5):
            g.add_node(label=f"v{i}")
        for i in range(5):
            for j in range(i+1, 5):
                g.add_edge(source=f"v{i}", target=f"v{j}")

        # Spectral radius = max absolute eigenvalue
        spec = g.get_spectrum(which="all")
        spectral_radius = max(abs(v) for v in spec['values'])

        # For regular graphs: spectral radius = degree
        # Bound: spectral_radius ≤ max_degree
        ```

        Notes
        -----
        **Spectral Properties**:

        - **Spectral radius**: `ρ(A) = max|λᵢ|`
          - always equals largest eigenvalue for undirected graphs
          - bounded by max degree: `ρ(A) ≤ Δ(G)`
          - for regular graphs: `ρ(A) = degree`

        - **Spectral gap**: `λ₁ - λ₂`
          - measures connectivity and expansion
          - large gap -> well-connected graph
          - small gap -> bottlenecks or clusters
          - zero gap -> disconnected graph

        - **Trace**: `Σλᵢ = 0` (sum of eigenvalues equals zero for undirected)

        - **Number of spanning trees**: Related to Laplacian eigenvalues

        **Special Graph Spectra**:

        - complete graph K_n: {n-1, -1, -1, ..., -1}
        - cycle C_n: {2cos(2πk/n) | k=0,...,n-1}
        - path P_n: {2cos(πk/(n+1)) | k=1,...,n}
        - star S_n: {√(n-1), 0, ..., 0, -√(n-1)}

        **Computational Complexity**:

        - `which="all"`: `O(n³)` using dense eigendecomposition
        - `which="largest"/"smallest"`: `O(kn²)` using iterative methods
        - Recommended: Use `k < 10` for large graphs (`n > 1000`)

        **Use Cases**:

        - spectral clustering: community detection via eigenvectors
        - graph embedding: represent nodes using eigenvector coordinates
        - bipartite detection: check if spectrum is symmetric about `0`
        - random walk analysis: eigenvalues determine mixing time
        - graph comparison: compare spectral distributions

        %family Spectral Analysis
        """
        return self._graph.spectrum(which, k)

    def get_laplacian_spectrum(
        self,
        which: str = "smallest",
        k: int | None = None,
        normalized: bool = False,
    ) -> dict[str, list[float] | list[list[float]] | float | None]:
        """
        Compute eigenvalues and eigenvectors of the graph Laplacian.

        Returns the spectrum of the Laplacian matrix `L = D - A` (or normalized variant). The
        Laplacian spectrum provides insights into graph connectivity, cuts, and diffusion processes.

        Parameters
        ----------
        which
            Which eigenvalues to compute. With `"all"`, this is all `n` eigenvalues (`O(n³)`,
            recommended for `n < 100`). Using `"smallest"` works with the smallest `k` eigenvalues
            (default `k=6`). The `"largest"` options works with the largest `k` eigenvalues.
        k
            Number of eigenvalues to compute for `"largest"`/`"smallest"`. Default is `min(6, n-1)`
            where `n` is the number of nodes.
        normalized
            If `True`, use normalized Laplacian: `L_norm = D^(-1/2) L D^(-1/2)`. Normalized version
            has eigenvalues in `[0, 2]` for undirected graphs.

        Returns
        -------
        dict
            Dictionary with keys:

            - `"values"`: list of eigenvalues (sorted ascending)
            - `"vectors"`: list of eigenvectors (each as list of floats)
            - `"algebraic_connectivity"`: `λ₂` (second-smallest eigenvalue, measures how
            well-connected the graph is; `0` if disconnected)

        Raises
        ------
        ValueError
            If graph is empty or eigenvalue computation fails.

        Examples
        --------
        **Detecting Disconnected Components**

        ```python
        from edgecase import Graph

        # Graph with 2 components
        g = Graph(directed=False)
        # Component 1
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="C")
        # Component 2
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_edge(source="X", target="Y")

        spec = g.get_laplacian_spectrum(which="smallest", k=3)
        zero_eigenvalues = sum(1 for v in spec['values'] if abs(v) < 1e-6)
        print(f"Number of components: {zero_eigenvalues}")
        # λ₁ = 0 has multiplicity = number of connected components

        connectivity = spec['algebraic_connectivity']
        print(f"Algebraic connectivity: {connectivity}")
        # = 0 means disconnected
        ```

        **Spectral Clustering with Fiedler Vector**

        ```python
        # Two-way spectral clustering
        g = Graph(directed=False)
        # Create graph with two communities...
        for i in range(6):
            g.add_node(label=f"n{i}")

        # Dense intra-community edges
        g.add_edge(source="n0", target="n1")
        g.add_edge(source="n1", target="n2")
        g.add_edge(source="n3", target="n4")
        g.add_edge(source="n4", target="n5")
        # Sparse inter-community edge
        g.add_edge(source="n2", target="n3")

        spec = g.get_laplacian_spectrum(which="smallest", k=3)
        fiedler_vector = spec['vectors'][1]  # Second eigenvector (λ₂)

        # Partition based on sign of Fiedler vector
        cluster_1 = [i for i, v in enumerate(fiedler_vector) if v < 0]
        cluster_2 = [i for i, v in enumerate(fiedler_vector) if v >= 0]
        ```

        **Measuring Graph Expansion**

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        for i in range(6):
            g.add_node(label=f"n{i}")
        g.add_edge(source="n0", target="n1")
        g.add_edge(source="n1", target="n2")
        g.add_edge(source="n3", target="n4")
        g.add_edge(source="n4", target="n5")
        g.add_edge(source="n2", target="n3")

        # Algebraic connectivity measures expansion
        spec = g.get_laplacian_spectrum(which="smallest", k=2)
        algebraic_connectivity = spec['algebraic_connectivity']

        # Cheeger inequality: h²/(2Δ) ≤ λ₂ ≤ 2h
        # where h = edge expansion, Δ = max degree
        # Large λ₂ -> expander graph (hard to partition)
        # Small λ₂ -> bottleneck (easy to cut)
        ```

        **Normalized Laplacian for Weighted Graphs**

        ```python
        from edgecase import Graph

        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="B", target="C", weight=2.0)
        g.add_edge(source="C", target="A", weight=0.5)

        # Normalized Laplacian accounts for degree variations
        spec = g.get_laplacian_spectrum(which="all", normalized=True)
        # Eigenvalues in [0, 2] for undirected graphs
        # λ = 0 ⟺ connected component
        # λ = 2 ⟺ bipartite structure
        ```

        **Random Walk and Diffusion Analysis**

        ```python
        g = Graph(directed=False)
        for i in range(6):
            g.add_node(label=f"n{i}")
        g.add_edge(source="n0", target="n1")
        g.add_edge(source="n1", target="n2")
        g.add_edge(source="n3", target="n4")
        g.add_edge(source="n4", target="n5")
        g.add_edge(source="n2", target="n3")

        # Laplacian eigenvalues determine diffusion speed
        spec = g.get_laplacian_spectrum(which="smallest", k=5)

        # Mixing time ≈ 1 / λ₂ (algebraic connectivity)
        # Small λ₂ -> slow mixing (long random walks to reach steady state)
        # Large λ₂ -> fast mixing (rapid convergence)
        ```

        Notes
        -----
        **Laplacian Properties**:

        - **Always has 0 eigenvalue**: Corresponding eigenvector is constant
        - **Number of 0 eigenvalues**: Equals number of connected components
        - **Algebraic connectivity** (λ₂, Fiedler value):
          - λ₂ > 0 ⟺ graph is connected
          - λ₂ = 0 ⟺ disconnected
          - measures robustness to edge removal
          - lower bound on edge/node connectivity

        - **Fiedler vector**: Eigenvector for λ₂
          - sign pattern reveals natural graph partition
          - used in spectral bisection algorithms
          - minimizes cut weight among all bisections

        **Normalized vs Unnormalized**:

        - **Unnormalized L = D - A**:
          - Eigenvalues: 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ
          - Sensitive to degree distribution
          - Natural for unweighted graphs

        - **Normalized L_norm = D^(-1/2) L D^(-1/2)**:
          - Eigenvalues in [0, 2] for undirected
          - Better for weighted/degree-imbalanced graphs
          - Used in normalized cut clustering

        **Special Graph Laplacians**:

        - Path P_n: {2 - 2cos(πk/n) | k=1,...,n}, λ₂ = 2(1 - cos(π/n))
        - Cycle C_n: {2 - 2cos(2πk/n) | k=0,...,n-1}, λ₂ = 2(1 - cos(2π/n))
        - Complete K_n: {0, n, n, ..., n}, λ₂ = n
        - Star S_n: {0, 1, 1, ..., 1, n}, λ₂ = 1

        **Computational Complexity**:

        - which="all": O(n³) for dense eigendecomposition
        - which="smallest": O(kn²) for k smallest eigenvalues
        - Normalized version: Additional O(n²) for degree computation

        **Use Cases**:

        - spectral clustering: use Fiedler vector for graph partitioning
        - community detection: k-way clustering using k eigenvectors
        - graph cuts: min-cut via spectral relaxation
        - synchronization: analyze coupled oscillators on networks
        - diffusion processes: heat kernel and random walks
        - network robustness: algebraic connectivity as connectivity measure

        %family Spectral Analysis
        """
        return self._graph.get_laplacian_spectrum(which, k, normalized)

    # ==============================================================================
    # Network Efficiency
    # ==============================================================================

    def global_efficiency(self) -> float:
        """
        Calculate the global efficiency of the graph.

        Global efficiency is the average efficiency over all pairs of nodes, where efficiency is the
        multiplicative inverse of the shortest path distance. It measures how efficiently
        information flows through the network.

        Returns
        -------
        float
            The global efficiency (between `0` and `1`).

        Examples
        --------
        Complete graph has efficiency `1`:

        ```python
        from edgecase import Graph

        # Complete graph K_5
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        eff = graph.global_efficiency()
        print(f"Efficiency: {eff:.4f}")  # Close to 1.0
        ```

        Compare different topologies:

        ```python
        # Path graph
        path = Graph()
        path.add_path_graph(10)
        path_eff = path.global_efficiency()

        # Cycle graph
        cycle = Graph()
        cycle.add_cycle_graph(10)
        cycle_eff = cycle.global_efficiency()

        # Star graph
        star = Graph()
        star.add_star_graph(10)
        star_eff = star.global_efficiency()

        print(f"Path: {path_eff:.4f}")
        print(f"Cycle: {cycle_eff:.4f}")
        print(f"Star: {star_eff:.4f}")
        ```

        Disconnected graph:

        ```python
        graph = Graph()
        # Two separate triangles
        for i in range(1, 7):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(6, 4)

        eff = graph.global_efficiency()
        print(f"Efficiency: {eff:.4f}")  # Lower due to disconnection
        ```

        Notes
        -----
        - returns `0` for graphs with `0` or `1` nodes
        - returns `1` for complete graphs
        - unweighted shortest paths are used (all edge weights = 1)
        - for disconnected node pairs, efficiency is `0`
        - related to the average inverse distance

        %family Network Efficiency
        """
        from edgecase._core import global_efficiency

        return global_efficiency(self._graph)

    def local_efficiency(self) -> float:
        """
        Calculate the local efficiency of the graph.

        Local efficiency is the average global efficiency of each node's neighborhood subgraph.
        It measures how efficiently the neighbors of a node can communicate when that node
        is removed. High local efficiency indicates good redundancy and fault tolerance.

        Returns
        -------
        float
            The average local efficiency (between 0 and 1).

        Examples
        --------
        Complete graph has local efficiency 1:

        ```python
        from edgecase import Graph

        # Complete graph K_5
        graph = Graph()
        n = 5
        for i in range(n):
            graph.add_node(label=f"n{i}")

        nodes = sorted(graph.get_node_ids())
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(nodes[i], nodes[j])

        local_eff = graph.local_efficiency()
        print(f"Local efficiency: {local_eff:.4f}")  # 1.0
        ```

        Compare different topologies:

        ```python
        # Cycle graph
        cycle = Graph()
        cycle.add_cycle_graph(10)
        cycle_eff = cycle.local_efficiency()

        # Star graph
        star = Graph()
        star.add_star_graph(10)
        star_eff = star.local_efficiency()

        # Path graph
        path = Graph()
        path.add_path_graph(10)
        path_eff = path.local_efficiency()

        print(f"Cycle: {cycle_eff:.4f}")
        print(f"Star: {star_eff:.4f}")
        print(f"Path: {path_eff:.4f}")
        ```

        Fault tolerance analysis:

        ```python
        graph = Graph()
        graph.add_cycle_graph(20)

        local_eff = graph.local_efficiency()
        global_eff = graph.global_efficiency()

        print(f"Local efficiency: {local_eff:.4f}")
        print(f"Global efficiency: {global_eff:.4f}")

        # High local efficiency suggests good fault tolerance
        if local_eff > 0.5:
            print("Network has good redundancy!")
        ```

        Notes
        -----
        - returns `0` for empty graphs
        - nodes with fewer than 2 neighbors contribute `0` to the average
        - useful for measuring network resilience
        - related to clustering but considers path lengths
        - unweighted shortest paths are used

        %family Network Efficiency
        """
        from edgecase._core import local_efficiency

        return local_efficiency(self._graph)

    # ==============================================================================
    # Distance Metrics
    # ==============================================================================

    def get_wiener_index(self, weight: str | None = None) -> float:
        """
        Calculate the Wiener index of the graph.

        The Wiener index is the sum of all pairwise shortest path distances in the graph. It is a
        fundamental topological descriptor used in chemistry and network analysis to measure graph
        compactness.

        For undirected graphs: `W(G) = Σ_{u < v} d(u, v)`. For directed graphs:
        `W(G) = Σ_{u ≠ v} d(u, v)`.

        Parameters
        ----------
        weight : str, optional
            Edge attribute to use as weight. If `None`, uses unweighted distances (all edges have
            distance `1`).

        Returns
        -------
        float
            The Wiener index. Returns 0.0 for empty graphs or single-node graphs. Returns infinity
            (`float('inf')`) for disconnected graphs.

        Examples
        --------
        Wiener index of a path graph:

        ```python
        from edgecase import Graph

        # Path P4: 1-2-3-4
        # Distances: d(1,2)=1, d(1,3)=2, d(1,4)=3, d(2,3)=1, d(2,4)=2, d(3,4)=1
        # Wiener index = 1+2+3+1+2+1 = 10
        graph = Graph()
        graph.add_path_graph(4)
        w = graph.get_wiener_index()
        print(f"Wiener index: {w}")  # 10.0
        ```

        Compare different topologies:

        ```python
        # Complete graph K_n has Wiener index n*(n-1)/2 (all pairs at distance 1)
        complete = Graph()
        complete.add_full_graph(5)
        print(f"K5 Wiener index: {complete.get_wiener_index()}")  # 10.0

        # Cycle graph
        cycle = Graph()
        cycle.add_cycle_graph(6)
        print(f"C6 Wiener index: {cycle.get_wiener_index()}")  # 18.0

        # Star graph
        star = Graph()
        star.add_star_graph(5)
        print(f"Star(5) Wiener index: {star.get_wiener_index()}")  # 16.0
        ```

        Disconnected graph returns infinity:

        ```python
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        # No edge between A and B
        w = graph.get_wiener_index()
        print(f"Disconnected: {w}")  # inf
        ```

        Notes
        -----
        - time complexity: `O(V × (V + E))` for unweighted graphs
        - time complexity: `O(V × (V + E) log V)` for weighted graphs
        - a lower Wiener index indicates a more compact graph
        - for trees: `W = Σ_{edges e} n₁(e) × n₂(e)` where `n₁`, `n₂` are subtree sizes

        %family Distance Metrics
        """
        from edgecase._core import wiener_index

        return wiener_index(self._graph, weight)

    def get_estrada_index(self) -> float:
        """
        Calculate the Estrada index of the graph.

        The Estrada index is the sum of exponentials of the eigenvalues of the adjacency matrix:

        `EE(G) = Σᵢ exp(λᵢ)`

        where `λᵢ` are the eigenvalues of the adjacency matrix. This index captures the degree of
        folding or compactness of a network, and is related to subgraph centrality and the number of
        closed walks.

        Returns
        -------
        float
            The Estrada index. Returns 0.0 for empty graphs, 1.0 for single-node graphs.

        Raises
        ------
        ValueError
            If the graph is directed (Estrada index is only defined for undirected graphs).

        Examples
        --------
        Estrada index of basic graphs:

        ```python
        from edgecase import Graph
        import math

        # Single edge (K2): eigenvalues are 1 and -1
        # EE = exp(1) + exp(-1) ≈ 3.086
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        ee = g.get_estrada_index()
        print(f"K2 Estrada index: {ee:.3f}")  # ≈ 3.086
        ```

        Compare different topologies:

        ```python
        # Path graph P3: eigenvalues are -√2, 0, √2
        path = Graph()
        path.add_path_graph(3)
        print(f"P3 Estrada: {path.get_estrada_index():.3f}")

        # Complete graph K3: eigenvalues are 2, -1, -1
        complete = Graph()
        complete.add_full_graph(3)
        print(f"K3 Estrada: {complete.get_estrada_index():.3f}")

        # Cycle graph C4
        cycle = Graph()
        cycle.add_cycle_graph(4)
        print(f"C4 Estrada: {cycle.get_estrada_index():.3f}")
        ```

        Empty graph (no edges) vs isolated nodes:

        ```python
        # n isolated nodes: all eigenvalues are 0, so EE = n
        isolated = Graph()
        for i in range(5):
            isolated.add_node()
        print(f"5 isolated nodes: {isolated.get_estrada_index():.1f}")  # 5.0
        ```

        Notes
        -----
        - time complexity: `O(n³)` for eigendecomposition
        - the Estrada index is always `≥ n` for any graph with `n` nodes
        - for graphs with no edges: `EE = n` (all eigenvalues are `0`)
        - higher Estrada index indicates more closed walks (more "folded" structure)
        - related to subgraph centrality: `SC(i) = Σⱼ φⱼ(i)² exp(λⱼ)`
        - EE = trace(exp(A)) where A is the adjacency matrix

        %family Distance Metrics
        """
        from edgecase._core import estrada_index

        return estrada_index(self._graph)

    def get_resistance_distance(self, source: int | str, target: int | str) -> float:
        """
        Calculate the resistance distance between two nodes.

        The resistance distance (also called effective resistance) is the electrical resistance
        between two nodes if each edge is replaced by a 1-ohm resistor. It is a metric on graphs
        that satisfies the triangle inequality and provides insight into network connectivity and
        robustness.

        Parameters
        ----------
        source
            Source node identifier (ID or label).
        target
            Target node identifier (ID or label).

        Returns
        -------
        float
            The resistance distance between the two nodes. Returns `0.0` if source equals target.

        Raises
        ------
        ValueError
            If the graph is directed (resistance distance is only defined for undirected graphs).
        ValueError
            If either source or target node doesn't exist in the graph.

        Examples
        --------
        Resistance distance between adjacent nodes:

        ```python
        from edgecase import Graph

        # Two adjacent nodes have resistance distance 1.0
        g = Graph()
        g.add_full_graph(2)
        print(g.get_resistance_distance(1, 2))  # 1.0
        ```

        Resistance in a path graph:

        ```python
        # In a path, resistance = number of edges (resistors in series)
        path = Graph()
        path.add_path_graph(4)  # 1-2-3-4
        print(path.get_resistance_distance(1, 2))  # 1.0
        print(path.get_resistance_distance(1, 3))  # 2.0
        print(path.get_resistance_distance(1, 4))  # 3.0
        ```

        Resistance in a complete graph:

        ```python
        # In Kn, resistance between any two nodes is 2/n
        complete = Graph()
        complete.add_full_graph(4)
        print(f"K4 resistance: {complete.get_resistance_distance(1, 2):.3f}")  # 0.5
        ```

        Resistance in a cycle (parallel paths):

        ```python
        # In a cycle, resistance is reduced by parallel paths
        cycle = Graph()
        cycle.add_cycle_graph(4)  # 1-2-3-4-1
        # From 1 to 2: path of length 1 || path of length 3
        # R = 1*3/(1+3) = 0.75
        print(f"C4 resistance: {cycle.get_resistance_distance(1, 2):.3f}")  # 0.75
        ```

        Notes
        -----
        - time complexity: `O(n³)` for matrix operations
        - for adjacent nodes: `Ω(u,v) ≤ 1` (equality for bridges)
        - resistance distance satisfies the triangle inequality
        - related to random walk commute time: `C(u,v) = 2m × Ω(u,v)`
        - the sum of all resistance distances is the Kirchhoff index

        %family Distance Metrics
        """
        from edgecase._core import resistance_distance

        source_id = self._resolve_node_id(source)
        target_id = self._resolve_node_id(target)
        return resistance_distance(self._graph, source_id, target_id)

    def resistance_distance_matrix(self) -> dict[tuple[int, int], float]:
        """
        Calculate the resistance distance matrix for all node pairs.

        Returns a dictionary mapping (source_id, target_id) tuples to their resistance distances.
        The matrix is symmetric with zeros on the diagonal.

        Returns
        -------
        dict[tuple[int, int], float]
            Dictionary mapping node ID pairs to resistance distances.

        Raises
        ------
        ValueError
            If the graph is directed.

        Examples
        --------
        Get all pairwise resistance distances:

        ```python
        from edgecase import Graph

        g = Graph()
        g.add_path_graph(3)
        matrix = g.resistance_distance_matrix()

        # Access specific pair
        print(matrix[(1, 3)])  # 2.0

        # The matrix is symmetric
        print(matrix[(1, 2)] == matrix[(2, 1)])  # True
        ```

        Notes
        -----
        - Time complexity: `O(n³)` for matrix operations
        - Returns all `n²` pairs (including both `(i,j)` and `(j,i)`)

        %family Distance Metrics
        """
        from edgecase._core import resistance_distance_matrix

        return resistance_distance_matrix(self._graph)

    def get_kirchhoff_index(self) -> float:
        """
        Calculate the Kirchhoff index (total resistance) of the graph.

        The Kirchhoff index is the sum of all pairwise resistance distances:

        `Kf(G) = Σᵢ<ⱼ Ω(i,j)`

        It can also be computed from the Laplacian eigenvalues:

        `Kf(G) = n × Σₖ (1/λₖ)`

        where `λₖ` are the non-zero eigenvalues of the Laplacian matrix.

        Returns
        -------
        float
            The Kirchhoff index. Returns `0.0` for single-node graphs. Returns infinity for
            disconnected graphs.

        Raises
        ------
        ValueError
            If the graph is directed.

        Examples
        --------
        Kirchhoff index of common graphs:

        ```python
        from edgecase import Graph

        # Path graph Pn: Kf = n(n²-1)/6
        path = Graph()
        path.add_path_graph(4)
        print(f"P4 Kirchhoff: {path.get_kirchhoff_index():.1f}")  # 10.0

        # Complete graph Kn: Kf = n-1
        complete = Graph()
        complete.add_full_graph(4)
        print(f"K4 Kirchhoff: {complete.get_kirchhoff_index():.1f}")  # 3.0

        # Cycle graph Cn: Kf = n(n²-1)/12
        cycle = Graph()
        cycle.add_cycle_graph(4)
        print(f"C4 Kirchhoff: {cycle.get_kirchhoff_index():.1f}")  # 5.0
        ```

        Disconnected graph returns infinity:

        ```python
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        # No edge between A and B
        print(g.get_kirchhoff_index())  # inf
        ```

        Notes
        -----
        - time complexity: O(n³) for eigendecomposition
        - the Kirchhoff index equals the quasi-Wiener index (Wiener index using
          resistance distance instead of shortest path distance)
        - lower Kirchhoff index indicates better overall connectivity
        - for connected graphs: Kf ≥ n-1 with equality only for complete graphs

        %family Distance Metrics
        """
        from edgecase._core import kirchhoff_index

        return kirchhoff_index(self._graph)


# ============================================================================
# AdaStar Pathfinding Classes
# ============================================================================


class AdaStar:
    """
    Anytime Dynamic A* (ADA*) path planner.

    An advanced pathfinding algorithm that combines:

    - **Anytime property**: returns suboptimal paths quickly, iteratively improves
    - **Dynamic property**: efficiently handles graph changes, reuses computation

    This is a stateful search algorithm that maintains its state between calls, allowing it to:

    1. find an initial path quickly with high epsilon (suboptimal but fast)
    2. iteratively improve the path by lowering epsilon
    3. handle dynamic edge weight changes efficiently

    The algorithm is based on the ARA* (Anytime Repairing A*) and D* Lite algorithms.

    Parameters
    ----------
    graph
        The graph to search. Must remain unchanged between calls unless explicitly updated via
        `update_graph()`.
    source
        Label of the source node.
    target
        Label of the target node.
    weight
        Name of the edge attribute to use as weight (default: "weight").
    initial_epsilon
        Initial epsilon value for suboptimality bound (default: 1000.0). Lower values give better
        paths but take longer. Using `epsilon = 1.0` is optimal (equivalent to A*).
    heuristic
        Heuristic function `h(u, v)` that estimates distance from u to v. Should be admissible
        (never overestimate). If `None`, uses `h=0` (Dijkstra). Function should accept two node IDs
        (integers) and return float.

    Attributes
    ----------
    graph : Graph
        The graph being searched.
    source : str
        Source node label.
    target : str
        Target node label.
    source_id : int
        Internal node ID for source.
    target_id : int
        Internal node ID for target.
    ada_star : _core.AdaStar
        Rust backend AdaStar instance.

    Examples
    --------
    Basic usage with anytime refinement:

    ```python
    import edgecase as ec

    # Create a weighted graph
    g = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B", weight=1.0)
        .add_edge(source="B", target="C", weight=1.0)
        .add_edge(source="A", target="D", weight=2.0)
        .add_edge(source="D", target="C", weight=0.5)
    )

    # Create ADA* planner
    from edgecase.graph import AdaStar
    planner = AdaStar(g, source="A", target="C", initial_epsilon=2.0)

    # Get initial suboptimal path quickly
    path = planner.compute_or_improve_path(epsilon=2.0)
    print(f"Initial path: {path}")

    # Improve path with lower epsilon
    path = planner.compute_or_improve_path(epsilon=1.2)
    print(f"Improved path: {path}")

    # Get optimal path
    path = planner.compute_or_improve_path(epsilon=1.0)
    print(f"Optimal path: {path}")
    ```

    Dynamic graph updates:

    ```python
    import edgecase as ec
    from edgecase.graph import AdaStar

    g = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B", weight=1.0)
        .add_edge(source="B", target="C", weight=1.0)
        .add_edge(source="A", target="D", weight=5.0)
        .add_edge(source="D", target="C", weight=5.0)
    )

    planner = AdaStar(g, source="A", target="C")
    path = planner.compute_or_improve_path(epsilon=1.0)
    print(f"Initial path: {path}")  # Should be A -> B -> C

    # Edge weight changes (e.g., traffic, road closure)
    b_id = g.get_id_from_label("B")
    c_id = g.get_id_from_label("C")

    # Update planner with changes
    planner.update_graph([(b_id, c_id, 10.0)])

    # Compute new path (reuses previous computation)
    path = planner.compute_or_improve_path(epsilon=1.0)
    print(f"Updated path: {path}")  # May now use A -> D -> C
    ```

    With custom heuristic (Euclidean distance):

    ```python
    import edgecase as ec
    from edgecase.graph import AdaStar
    import math

    g = (
        ec.Graph()
        .add_node(label="A", x=0.0, y=0.0)
        .add_node(label="B", x=1.0, y=0.0)
        .add_node(label="C", x=1.0, y=1.0)
        .add_edge(source="A", target="B", weight=1.0)
        .add_edge(source="B", target="C", weight=1.0)
        .add_edge(source="A", target="C", weight=2.0)
    )

    # Define Euclidean distance heuristic
    def euclidean_heuristic(u_id, v_id):
        u = g.get_node(u_id)
        v = g.get_node(v_id)
        dx = u.attributes.get("x", 0.0) - v.attributes.get("x", 0.0)
        dy = u.attributes.get("y", 0.0) - v.attributes.get("y", 0.0)
        return math.sqrt(dx*dx + dy*dy)

    planner = AdaStar(g, source="A", target="C", heuristic=euclidean_heuristic)
    path = planner.compute_or_improve_path(epsilon=1.0)
    print(f"Path: {path}")
    ```

    Notes
    -----
    - the heuristic function receives internal node IDs (integers), not labels
    - for epsilon > 1.0, the path cost is guaranteed to be at most epsilon times optimal
    - lower epsilon values require more computation but give better paths
    - the `update_graph()` method efficiently reuses previous computation
    - graph must not be modified between calls except via `update_graph()`
    """

    def __init__(
        self,
        graph,
        source: Union[int, str],
        target: Union[int, str],
        weight: str = "weight",
        initial_epsilon: float = 1000.0,
        heuristic=None,
    ):
        """
        Initialize ADA* planner.

        Parameters
        ----------
        graph
            The graph to search.
        source
            Source node ID (int) or label (str).
        target
            Target node ID (int) or label (str).
        weight
            Edge attribute name for weights (default: `"weight"`).
        initial_epsilon
            Initial epsilon for suboptimality bound (default: `1000.0`).
        heuristic
            Heuristic function `h(u, v) -> float` where `u` and `v` can be node IDs or labels. If
            `None`, uses `h=0`.
        """
        from ._core import AdaStar as RustAdaStar

        self.graph = graph
        self.source = source
        self.target = target

        # Get internal node IDs and handle both ID and label inputs
        self.source_id = graph._resolve_node_id(source)
        self.target_id = graph._resolve_node_id(target)

        if self.source_id is None:
            raise ValueError(f"Source node '{source}' not found in graph")
        if self.target_id is None:
            raise ValueError(f"Target node '{target}' not found in graph")

        # Create Rust backend
        self.ada_star = RustAdaStar(
            self.source_id,
            self.target_id,
            weight=weight,
            initial_epsilon=initial_epsilon,
            heuristic=heuristic,
        )

    def compute_or_improve_path(self, epsilon: float = 1.0) -> list[int]:
        """
        Compute or improve the path with given epsilon.

        On first call, initializes the search and finds an initial path.
        On subsequent calls, improves the existing path if epsilon is lower.

        Parameters
        ----------
        epsilon
            Suboptimality bound (default: `1.0`).

            - epsilon = 1.0: optimal path (A*)
            - epsilon > 1.0: path cost ≤ epsilon × optimal cost
            Higher epsilon = faster but less optimal

        Returns
        -------
        list[int]
            Path from source to target as list of node IDs.

        Raises
        ------
        ValueError
            If no path exists from source to target.

        Examples
        --------
        ```python
        import edgecase as ec
        from edgecase.graph import AdaStar
        graph = ec.Graph()
        graph.add_node(label="A")
        graph.add_node(label="Z")
        graph.add_edge(source="A", target="Z", weight=1.0)
        planner = AdaStar(graph, "A", "Z", initial_epsilon=2.0)
        path = planner.compute_or_improve_path(epsilon=2.0)  # Fast, suboptimal
        path = planner.compute_or_improve_path(epsilon=1.5)  # Better
        path = planner.compute_or_improve_path(epsilon=1.0)  # Optimal
        ```
        """
        # Run search in Rust
        self.ada_star.compute_or_improve_path(self.graph._graph, epsilon)

        # Extract and return path as node IDs
        return self.ada_star.extract_path(self.graph._graph)

    def update_graph(self, changes: list[tuple[int, int, float]]) -> None:
        """
        Update the planner with edge weight changes.

        This efficiently reuses previous computation rather than restarting
        the search from scratch.

        Parameters
        ----------
        changes
            List of edge weight changes as (source_id, target_id, new_weight).
            Use `graph.get_node_id(label)` to convert labels to IDs.

        Examples
        --------
        ```python
        import edgecase as ec
        from edgecase.graph import AdaStar
        graph = ec.Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="Z")
        graph.add_edge(source="A", target="B", weight=1.0)
        graph.add_edge(source="B", target="C", weight=1.0)
        graph.add_edge(source="C", target="Z", weight=1.0)
        planner = AdaStar(graph, "A", "Z")
        path = planner.compute_or_improve_path(epsilon=1.0)
        # Notify planner of edge weight changes
        b_id = graph.get_id_from_label("B")
        c_id = graph.get_id_from_label("C")
        planner.update_graph([(b_id, c_id, 10.0)])
        # Recompute path (reuses previous work)
        new_path = planner.compute_or_improve_path(epsilon=1.0)
        ```
        """
        self.ada_star.update_graph(self.graph._graph, changes)

    def extract_path(self) -> list[int]:
        """
        Extract the current path without running search.

        Returns the path based on current g-values. Useful for inspecting
        the path without triggering additional computation.

        Returns
        -------
        list[int]
            Current path from source to target as list of node IDs.

        Raises
        ------
        ValueError
            If no path exists or search hasn't been initialized.
        """
        return self.ada_star.extract_path(self.graph._graph)
