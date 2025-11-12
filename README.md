# EdgeCase

EdgeCase is a Python library for working with graphs and networks. It provides a comprehensive set of tools for graph creation, analysis, and visualization, with a focus on performance and ease of use.

## Features

EdgeCase offers a wide range of graph operations and algorithms:

**Graph Construction**

- Create directed and undirected graphs
- Add nodes and edges with custom attributes
- Build graphs from edge lists and adjacency data
- Generate random graphs (Erdős-Rényi, Barabási-Albert, Watts-Strogatz)

**Path and Traversal**

- Shortest path algorithms (Dijkstra, Bellman-Ford, BFS)
- All simple paths enumeration
- Path validation and checking
- Depth-first and breadth-first search

**Centrality Measures**

- Betweenness centrality
- Closeness centrality
- Harmonic centrality
- Eigenvector centrality
- Degree centrality
- PageRank

**Community Detection**

- Louvain method
- Label propagation
- Fast greedy modularity optimization
- Leading eigenvector method
- Walktrap algorithm
- Edge betweenness

**Graph Properties**

- Connected components
- Clustering coefficients
- Graph diameter and radius
- Degree distributions
- Transitivity and density

**Visualization**

- Interactive graph widgets for Jupyter notebooks
- Customizable node and edge styling
- Support for large graphs

## Quick Start

```python
import polars as pl
from edgecase import Graph

# Create node and edge DataFrames
nodes_df = pl.DataFrame({
    "id": [1, 2, 3],
    "label": ["Alice", "Bob", "Charlie"],
    "type": ["person", "person", "person"]
})

edges_df = pl.DataFrame({
    "from": [1, 2, 3],
    "to": [2, 3, 1],
    "rel": ["friend", "colleague", "friend"],
    "weight": [1.0, 2.0, 1.5]
})

# Create graph from DataFrames
graph = Graph(nodes_df=nodes_df, edges_df=edges_df, directed=False, weighted=True)

# Analyze the graph
paths = graph.get_all_simple_paths("Alice", "Charlie")
centrality = graph.get_betweenness_centrality()
communities = graph.detect_communities(method="louvain")

# Visualize (in Jupyter)
graph.show()
```

## Integration with Polars

EdgeCase is designed to work seamlessly with Polars DataFrames:

```python
import polars as pl
from edgecase import Graph

# Node DataFrame with attributes
nodes_df = pl.DataFrame({
    "id": [1, 2, 3],
    "label": ["Alice", "Bob", "Charlie"],
    "type": ["admin", "user", "user"],
    "active": [True, True, False]
})

# Edge DataFrame with relationships
edges_df = pl.DataFrame({
    "from": [1, 2],
    "to": [2, 3],
    "rel": ["manages", "collaborates"],
    "weight": [1.0, 0.8]
})

# Create graph directly from DataFrames
graph = Graph(nodes_df=nodes_df, edges_df=edges_df, directed=True, weighted=True)

graph.show()
```

## Documentation

For detailed documentation, examples, and API reference, visit the [documentation site](https://rich-iannone.github.io/edgecase/).

## License

EdgeCase is licensed under the MIT License. See the LICENSE file for details.
