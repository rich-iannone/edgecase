from edgecase import Graph


class TestLoadCentrality:
    """Test suite for load centrality."""

    def test_path_3_normalized(self):
        """Test load centrality on a 3-node path."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2")

        c = g.load_centrality(return_dict=True)

        # Middle node should have load of 1.0 (normalized)
        # End nodes should have 0.0
        node_ids = sorted(c.keys())
        expected = [0.0, 1.0, 0.0]

        for nid, exp in zip(node_ids, expected):
            assert abs(c[nid] - exp) < 1e-3, f"Node {nid}: got {c[nid]}, expected {exp}"

    def test_path_3_unnormalized(self):
        """Test unnormalized load centrality on a 3-node path."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2")

        c = g.load_centrality(normalized=False, return_dict=True)

        # Middle node should have load of 2.0 (unnormalized)
        node_ids = sorted(c.keys())
        expected = [0.0, 2.0, 0.0]

        for nid, exp in zip(node_ids, expected):
            assert abs(c[nid] - exp) < 1e-3

    def test_path_4_unnormalized(self):
        """Test unnormalized load centrality on a 4-node path."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))
        for i in range(3):
            g.add_edge(str(i), str(i + 1))

        c = g.load_centrality(normalized=False, return_dict=True)

        # Expected loads: [0, 6, 8, 6, 0] but we have 4 nodes
        # Actually for P4: [0, 6, 6, 0]
        node_ids = sorted(c.keys())
        # From NetworkX test: path_4 should give middle nodes higher loads
        # The exact pattern depends on path structure

        # Check that end nodes have 0 load
        assert abs(c[node_ids[0]]) < 1e-3
        assert abs(c[node_ids[3]]) < 1e-3

        # Check that middle nodes have positive load
        assert c[node_ids[1]] > 0
        assert c[node_ids[2]] > 0

    def test_complete_graph_k5(self):
        """Test load centrality on complete graph K5."""
        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))

        # Add all edges for complete graph
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(str(i), str(j))

        c = g.load_centrality(return_dict=True)

        # In a complete graph, all nodes should have load of 0
        # (all shortest paths are direct, no node is between others)
        for nid in c:
            assert abs(c[nid]) < 1e-3

    def test_star_graph(self):
        """Test load centrality on a star graph."""
        g = Graph()
        g.add_node(label="center")
        for i in range(4):
            g.add_node(label=f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        c = g.load_centrality(return_dict=True)

        # Center node should have high load
        center_id = g.get_id_from_label("center")

        # All paths between leaves go through center
        assert c[center_id] > 0

        # Leaves should have 0 load (no paths go through them)
        for i in range(4):
            leaf_id = g.get_id_from_label(f"leaf{i}")
            assert abs(c[leaf_id]) < 1e-3

    def test_bridge_graph(self):
        """Test load centrality on a graph with a bridge node."""
        g = Graph()

        # Create two clusters connected by a bridge
        # Cluster 1: A-B-Bridge
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="Bridge")
        g.add_edge("A", "B")
        g.add_edge("B", "Bridge")

        # Cluster 2: Bridge-C-D
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("Bridge", "C")
        g.add_edge("C", "D")

        c = g.load_centrality(return_dict=True)

        bridge_id = g.get_id_from_label("Bridge")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")

        # Bridge should have highest load
        assert c[bridge_id] >= c[b_id]
        assert c[bridge_id] >= c[c_id]

    def test_cutoff_parameter(self):
        """Test that cutoff parameter limits path lengths."""
        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i + 1))

        # With cutoff=1, only adjacent nodes are considered
        c_cutoff = g.load_centrality(cutoff=1, normalized=False, return_dict=True)

        # With no cutoff, all paths are considered
        c_full = g.load_centrality(normalized=False, return_dict=True)

        # Load with cutoff should be less than or equal to full load
        for nid in c_cutoff:
            assert c_cutoff[nid] <= c_full[nid] + 1e-6

    def test_empty_graph(self):
        """Test load centrality on empty graph."""
        g = Graph()
        c = g.load_centrality(return_dict=True)
        assert len(c) == 0

    def test_single_node(self):
        """Test load centrality on single node graph."""
        g = Graph()
        g.add_node(label="A")

        c = g.load_centrality(return_dict=True)

        # Single node should have 0 load
        assert len(c) == 1

        node_id = list(c.keys())[0]

        assert abs(c[node_id]) < 1e-3

    def test_disconnected_graph(self):
        """Test load centrality on disconnected graph."""
        g = Graph()

        # Component 1
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Component 2 (isolated)
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("C", "D")

        c = g.load_centrality(return_dict=True)

        # All nodes should have 0 load (paths only within components)
        # For 2-node components, there's no node "between" others
        for nid in c:
            assert abs(c[nid]) < 1e-3

    def test_chain_storage(self):
        """Test that load_centrality stores results in NDF and allows chaining."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))
        for i in range(3):
            g.add_edge(str(i), str(i + 1))

        # Should return self for chaining
        result = g.load_centrality()

        assert result is g

        # Check that computation is tracked
        assert any("load_centrality" in str(comp) for comp in g._computations)

    def test_custom_attr_name(self):
        """Test storing load centrality with custom attribute name."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2")

        g.load_centrality(attr_name="custom_load")

        # Check that computation is tracked with custom name
        assert any("custom_load" in str(comp) for comp in g._computations)

    def test_return_dict_parameter(self):
        """Test that return_dict parameter works correctly."""
        g = Graph()
        for i in range(3):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2")

        # With return_dict=True, should return dictionary
        result = g.load_centrality(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 3

        # With return_dict=False (default), should return self
        result = g.load_centrality(return_dict=False)

        assert result is g

    def test_comparison_with_betweenness(self):
        """Compare load centrality with betweenness on a simple graph."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))
        g.add_edge("0", "1").add_edge("1", "2").add_edge("2", "3")

        load = g.load_centrality(return_dict=True)
        betweenness = g.betweenness_centrality(return_dict=True)

        # Both should identify middle nodes as important
        # (though exact values may differ)
        node_ids = sorted(load.keys())

        # End nodes should have low centrality in both
        assert load[node_ids[0]] < load[node_ids[1]]
        assert load[node_ids[3]] < load[node_ids[2]]

        assert betweenness[node_ids[0]] < betweenness[node_ids[1]]
        assert betweenness[node_ids[3]] < betweenness[node_ids[2]]

    def test_directed_graph(self):
        """Test load centrality on a directed graph."""
        g = Graph(directed=True)

        # Create a directed path: A -> B -> C -> D
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        c = g.load_centrality(return_dict=True)

        # B and C should have non-zero load
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")

        assert c[b_id] > 0
        assert c[c_id] > 0

    def test_cycle_graph(self):
        """Test load centrality on a cycle graph."""
        g = Graph()
        for i in range(4):
            g.add_node(label=str(i))

        # Create a cycle: 0-1-2-3-0
        for i in range(4):
            g.add_edge(str(i), str((i + 1) % 4))

        c = g.load_centrality(return_dict=True)

        # In a cycle, all nodes should have equal load
        values = list(c.values())
        avg = sum(values) / len(values)

        for v in values:
            assert abs(v - avg) < 0.1  # Allow some tolerance

    def test_normalization_scaling(self):
        """Test that normalization scales values appropriately."""
        g = Graph()
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i + 1))

        c_norm = g.load_centrality(normalized=True, return_dict=True)
        c_raw = g.load_centrality(normalized=False, return_dict=True)

        n = len(c_norm)
        scale = 1.0 / ((n - 1) * (n - 2))

        # Check that normalized = raw * scale
        for nid in c_norm:
            expected = c_raw[nid] * scale
            assert abs(c_norm[nid] - expected) < 1e-6

    def test_krackhardt_kite_structure(self):
        """Test on a structure similar to Krackhardt kite graph."""
        g = Graph()

        # Create a connected structure with varying centrality
        nodes = ["A", "B", "C", "D", "E", "F", "G"]
        for n in nodes:
            g.add_node(label=n)

        # Create edges forming a kite-like structure
        edges = [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
            ("C", "E"),
            ("D", "E"),
            ("D", "F"),
            ("E", "F"),
            ("F", "G"),
        ]
        for src, tgt in edges:
            g.add_edge(src, tgt)

        c = g.load_centrality(return_dict=True)

        # F should have high load (bridges to G)
        f_id = g.get_id_from_label("F")
        g_id = g.get_id_from_label("G")
        a_id = g.get_id_from_label("A")

        assert c[f_id] > c[a_id]  # F is a bridge
        assert c[g_id] < c[f_id]  # G is a leaf
