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

    def _build_supra_adjacency(
        self,
        layer_weights: Optional[dict[str, float]] = None,
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        """
        Build a weighted supra-adjacency structure from all layers.

        Returns (labels_list, adj) where adj[u][v] = aggregated weight.
        """
        labels = sorted(self.node_labels())

        if layer_weights is None:
            layer_weights = {name: 1.0 for name in self._layers}

        adj: dict[str, dict[str, float]] = {lbl: {} for lbl in labels}

        for name, g in self._layers.items():
            w = layer_weights.get(name, 1.0)
            edges = g._graph.edges()
            for _, s, t in edges:
                s_label = g.get_label_from_id(s)
                t_label = g.get_label_from_id(t)
                if s_label is None:
                    s_label = str(s)
                if t_label is None:
                    t_label = str(t)

                adj[s_label][t_label] = adj[s_label].get(t_label, 0.0) + w
                if not g.is_directed():
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

        Builds a supra-adjacency representation that aggregates edges from every layer (each
        weighted by its layer weight), then runs power-iteration PageRank on the aggregate.

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

        scores = {lbl: 1.0 / n for lbl in labels}

        for _ in range(max_iter):
            new_scores: dict[str, float] = {}
            for lbl in labels:
                rank = (1.0 - damping) / n
                for src in labels:
                    if lbl in adj.get(src, {}):
                        out_weight = sum(adj[src].values())
                        if out_weight > 0:
                            rank += damping * scores[src] * adj[src][lbl] / out_weight
                new_scores[lbl] = rank

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
        algorithm on it. The aggregation sums edge weights across layers (scaled by
        `layer_weights`).

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

        labels, adj = self._build_supra_adjacency(layer_weights)
        if not labels:
            if return_dict:
                return {}
            return self

        # Build an aggregate Graph with weighted edges
        agg = Graph(weighted=True)
        for lbl in labels:
            agg.add_node(label=lbl)

        seen_edges: set[tuple[str, str]] = set()
        for src, targets in adj.items():
            for tgt, weight in targets.items():
                edge_key = (min(src, tgt), max(src, tgt))
                if edge_key not in seen_edges:
                    agg.add_edge(source=src, target=tgt, weight=weight)
                    seen_edges.add(edge_key)

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
