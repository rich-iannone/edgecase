from typing import Optional, Union

from edgecase.graph import Graph


class MultiplexGraph:
    """
    Container for multiple graph layers on the same node set.

    A multiplex (multi-layer) network holds several `Graph` objects that share the same set of node
    labels but carry different edge types. For example, a protein-protein interaction layer, a
    co-expression layer, and a genetic interaction layer can be combined so that cross-layer
    algorithms (PageRank, community detection) integrate information from all three.

    Parameters
    ----------
    layers
        A dictionary mapping layer names to `Graph` objects. Every graph must contain exactly the
        same set of node labels.

    Raises
    ------
    ValueError
        If `layers` is empty or if the node-label sets of the graphs differ.

    Examples
    --------
    ```{python}
    import edgecase as ec

    ppi = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
    )

    coex = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="B", target="C")
    )

    mpx = ec.MultiplexGraph({"ppi": ppi, "coex": coex})
    mpx
    ```
    """

    def __init__(self, layers: dict[str, Graph]) -> None:
        if not layers:
            raise ValueError("At least one layer is required")

        self._layers: dict[str, Graph] = dict(layers)

        ref_name, ref_graph = next(iter(self._layers.items()))
        ref_labels = self._node_labels(ref_graph)

        for name, g in self._layers.items():
            if name == ref_name:
                continue
            other_labels = self._node_labels(g)
            if other_labels != ref_labels:
                raise ValueError(f"Node labels in layer '{name}' do not match layer '{ref_name}'")

    @staticmethod
    def _node_labels(graph: Graph) -> set[str]:
        node_ids = graph.get_node_ids()
        labels = set()
        for nid in node_ids:
            label = graph.get_label_from_id(nid)
            labels.add(label if label is not None else str(nid))
        return labels

    def __repr__(self) -> str:
        n_nodes = self.count_nodes()
        layer_parts = []
        for name, g in self._layers.items():
            layer_parts.append(f"{name}({g.count_edges()} edges)")
        layers_str = ", ".join(layer_parts)
        return f"MultiplexGraph({n_nodes} nodes, layers=[{layers_str}])"

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def layers(self) -> dict[str, Graph]:
        """Return the dictionary of layer name to Graph."""
        return dict(self._layers)

    @property
    def layer_names(self) -> list[str]:
        """Return the list of layer names."""
        return list(self._layers.keys())

    def get_layer(self, name: str) -> Graph:
        """
        Return a single layer by name.

        Parameters
        ----------
        name
            Layer name.

        Raises
        ------
        KeyError
            If the layer does not exist.
        """
        if name not in self._layers:
            raise KeyError(f"No layer named '{name}'")
        return self._layers[name]

    def count_nodes(self) -> int:
        """Return the number of nodes (shared across all layers)."""
        ref_graph = next(iter(self._layers.values()))
        return ref_graph.count_nodes()

    def count_layers(self) -> int:
        """Return the number of layers."""
        return len(self._layers)

    def node_labels(self) -> set[str]:
        """Return the set of node labels shared across all layers."""
        return self._node_labels(next(iter(self._layers.values())))

    # ------------------------------------------------------------------
    # Supra-adjacency helpers
    # ------------------------------------------------------------------

    def _layer_edges(
        self,
        layer_weights: Optional[dict[str, float]] = None,
    ):
        """
        Yield `(source_label, target_label, weight, directed)` for every edge of every layer.

        The weight is the layer weight times the edge's own weight (`1.0` if it has none).
        """
        if layer_weights is None:
            layer_weights = {}

        for name, g in self._layers.items():
            w = layer_weights.get(name, 1.0)
            directed = g.is_directed()
            for _, s, t, edge_weight in g._graph._weighted_edges():
                s_label = g.get_label_from_id(s)
                t_label = g.get_label_from_id(t)
                if s_label is None:
                    s_label = str(s)
                if t_label is None:
                    t_label = str(t)
                yield s_label, t_label, w * (1.0 if edge_weight is None else edge_weight), directed

    def _build_supra_adjacency(
        self,
        layer_weights: Optional[dict[str, float]] = None,
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        """
        Build a weighted supra-adjacency structure from all layers.

        Returns (labels_list, adj) where adj[u][v] = aggregated weight. Undirected edges count in
        both directions, except self-loops, which count once.
        """
        labels = sorted(self.node_labels())
        adj: dict[str, dict[str, float]] = {lbl: {} for lbl in labels}

        for s_label, t_label, w, directed in self._layer_edges(layer_weights):
            adj[s_label][t_label] = adj[s_label].get(t_label, 0.0) + w
            if not directed and s_label != t_label:
                adj[t_label][s_label] = adj[t_label].get(s_label, 0.0) + w

        return labels, adj

    # ------------------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------------------

    def multiplex_pagerank(
        self,
        layer_weights: Optional[dict[str, float]] = None,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        return_dict: bool = False,
    ) -> Union[dict[str, float], "MultiplexGraph"]:
        """
        Compute PageRank across all layers with configurable layer weights.

        Builds a supra-adjacency representation that aggregates edges from every layer, then runs
        power-iteration PageRank on the aggregate. Each edge counts with its own weight (`1.0` if
        it has none) times its layer's weight, and an undirected edge counts in both directions.
        The rank of a node with no out-going weight is spread evenly over all nodes, as in
        `Graph.pagerank()`, so the result matches standard PageRank on the aggregate graph.

        Parameters
        ----------
        layer_weights
            A `{layer_name: weight}` dictionary controlling each layer's contribution. Defaults to
            equal weight (`1.0`) for every layer.
        damping
            Damping factor (probability of following an edge vs. random jump).
        max_iter
            Maximum number of power iterations.
        tol
            Convergence tolerance.
        return_dict
            If `True`, return a `{node_label: score}` dictionary. Otherwise return `self` for method
            chaining.

        Returns
        -------
        dict or MultiplexGraph
            Scores keyed by node label, or `self`.

        Examples
        --------
        ```{python}
        import edgecase as ec

        ppi = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
        )
        coex = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="B", target="C")
        )

        mpx = ec.MultiplexGraph({"ppi": ppi, "coex": coex})
        mpx.multiplex_pagerank(return_dict=True)
        ```
        """
        labels, adj = self._build_supra_adjacency(layer_weights)
        n = len(labels)
        if n == 0:
            if return_dict:
                return {}
            return self

        out_weight = {src: sum(adj[src].values()) for src in labels}
        scores = {lbl: 1.0 / n for lbl in labels}

        for _ in range(max_iter):
            # Rank at nodes without out-going weight is spread over all nodes
            dangling = sum(scores[src] for src in labels if out_weight[src] <= 0)
            base = (1.0 - damping) / n + damping * dangling / n
            new_scores = {lbl: base for lbl in labels}
            for src in labels:
                if out_weight[src] > 0:
                    share = damping * scores[src] / out_weight[src]
                    for tgt, weight in adj[src].items():
                        new_scores[tgt] += share * weight

            # Normalize
            total = sum(new_scores.values())
            if total > 0:
                new_scores = {k: v / total for k, v in new_scores.items()}

            diff = sum(abs(new_scores[k] - scores[k]) for k in labels)
            scores = new_scores
            if diff < tol:
                break

        if return_dict:
            return scores
        self._last_pagerank = scores
        return self

    def multiplex_community_detection(
        self,
        layer_weights: Optional[dict[str, float]] = None,
        method: str = "louvain",
        resolution: float = 1.0,
        return_dict: bool = False,
    ) -> Union[dict[str, int], "MultiplexGraph"]:
        """
        Detect communities considering all layers simultaneously.

        Builds an aggregate weighted graph from all layers, then runs a community detection
        algorithm on it. The aggregation sums edge weights (`1.0` for an edge without one) across
        layers, each scaled by its layer's weight. The aggregate is undirected: the edges between
        two nodes, in either direction, add up to one edge.

        Parameters
        ----------
        layer_weights
            A `{layer_name: weight}` dictionary controlling each layer's contribution. Defaults to
            equal weight (`1.0`) for every layer.
        method
            Community detection method: `"louvain"` (default) or `"walktrap"`.
        resolution
            Resolution parameter (Louvain only).  Higher values produce more communities.
        return_dict
            If `True`, return a `{node_label: community_id}` dictionary. Otherwise return `self` for
            method chaining.

        Returns
        -------
        dict or MultiplexGraph
            Community assignments keyed by node label, or `self`.

        Raises
        ------
        ValueError
            If `method` is not `"louvain"` or `"walktrap"`.

        Examples
        --------
        ```{python}
        import edgecase as ec

        layer_a = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="C", target="D")
        )
        layer_b = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="C", target="D")
        )

        mpx = ec.MultiplexGraph({"a": layer_a, "b": layer_b})
        mpx.multiplex_community_detection(return_dict=True)
        ```
        """
        if method not in ("louvain", "walktrap"):
            raise ValueError(f"Unknown method '{method}'; use 'louvain' or 'walktrap'")

        labels = sorted(self.node_labels())
        if not labels:
            if return_dict:
                return {}
            return self

        # Sum each pair's weights, in both directions, into one undirected edge
        pair_weights: dict[tuple[str, str], float] = {}
        for s_label, t_label, w, _ in self._layer_edges(layer_weights):
            key = (min(s_label, t_label), max(s_label, t_label))
            pair_weights[key] = pair_weights.get(key, 0.0) + w

        # Build an aggregate Graph with weighted edges
        agg = Graph(weighted=True)
        for lbl in labels:
            agg.add_node(label=lbl)
        for (src, tgt), weight in sorted(pair_weights.items()):
            agg.add_edge(source=src, target=tgt, weight=weight)

        if method == "louvain":
            communities_by_id = agg.cmty_louvain(resolution=resolution, return_dict=True)
        else:
            communities_by_id = agg.cmty_walktrap(return_dict=True)

        # Map node IDs back to labels
        communities: dict[str, int] = {}
        for nid, cmty in communities_by_id.items():
            label = agg.get_label_from_id(nid)
            if label is not None:
                communities[label] = cmty
            else:
                communities[str(nid)] = cmty

        if return_dict:
            return communities
        self._last_communities = communities
        return self
