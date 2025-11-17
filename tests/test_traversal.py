"""
Test suite for graph traversal operations.

Tests the 7 core traversal methods (trav_out, trav_in, trav_both,
trav_out_edge, trav_in_edge, trav_reverse_edge).
"""

import pytest
import edgecase as ec


class TestBasicTraversal:
    """Tests for trav_out, trav_in, trav_both."""

    def test_trav_out_directed(self):
        """Test outward traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "D", auto_create_nodes=True)

        # Traverse from A to its neighbors
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == [2, 3]  # B=2, C=3

    def test_trav_out_multi_hop(self):
        """Test chaining trav_out for multi-hop traversal."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse 2 hops from A
        G.select_nodes(lambda n: n.label == "A").trav_out().trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == [3]  # C=3

    def test_trav_in_directed(self):
        """Test inward traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse from C to predecessors
        G.select_nodes(lambda n: n.label == "C").trav_in()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == [1, 3]  # A=1, B=3 (C=2, D=4)

    def test_trav_both_directed(self):
        """Test both-direction traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("C", "B", auto_create_nodes=True)
        G.add_edge("B", "D", auto_create_nodes=True)

        # Traverse from B to all neighbors
        G.select_nodes(lambda n: n.label == "B").trav_both()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == [1, 3, 4]  # A=1, C=3, D=4

    def test_trav_out_undirected(self):
        """Test that undirected graphs work like trav_both."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)

        # In undirected, trav_out should get all neighbors
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == [2, 3]  # B=2, C=3

    def test_trav_empty_selection(self):
        """Test traversal with no selection does nothing."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)

        # No selection
        G.trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []

    def test_trav_no_neighbors(self):
        """Test traversal from node with no neighbors."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")

        # Node A has no outgoing edges
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []


class TestEdgeTraversal:
    """Tests for trav_out_edge, trav_in_edge."""

    def test_trav_out_edge_directed(self):
        """Test outward edge traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "D", auto_create_nodes=True)

        # Traverse from A to its outgoing edges
        G.select_nodes(lambda n: n.label == "A").trav_out_edge()

        # Should select 2 edges (A->B, A->C)
        edges = G.get_selected_edges()
        assert len(edges) == 2

        # Node selection should be cleared
        nodes = G.get_selected_nodes()
        assert nodes == []

    def test_trav_in_edge_directed(self):
        """Test inward edge traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse from C to its incoming edges
        G.select_nodes(lambda n: n.label == "C").trav_in_edge()

        # Should select 2 edges (A->C, B->C)
        edges = G.get_selected_edges()
        assert len(edges) == 2

        # Node selection should be cleared
        nodes = G.get_selected_nodes()
        assert nodes == []

    def test_trav_out_edge_undirected(self):
        """Test edge traversal in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)

        # In undirected, trav_out_edge should get all incident edges
        G.select_nodes(lambda n: n.label == "A").trav_out_edge()
        edges = G.get_selected_edges()
        assert len(edges) == 2

    def test_trav_edge_multiple_nodes(self):
        """Test edge traversal from multiple selected nodes."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Select both A and B
        G.select_nodes(lambda n: n.label in ["A", "B"]).trav_out_edge()
        edges = G.get_selected_edges()
        assert len(edges) == 2


class TestReverseEdgeTraversal:
    """Tests for trav_reverse_edge."""

    def test_trav_reverse_edge_basic(self):
        """Test basic reverse edge traversal."""
        G = ec.Graph(directed=True)
        # Add bidirectional edges
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)

        edge_ids = G.edge_ids(limit=None)
        edge_id_1, edge_id_2 = edge_ids[0], edge_ids[1]

        # Select A->B edge
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge()

        # Should now select B->A edge
        selected = G.get_selected_edges()
        assert edge_id_2 in selected
        assert edge_id_1 not in selected

    def test_trav_reverse_edge_add_to_selection(self):
        """Test reverse edge with add_to_selection=True."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)

        edge_ids = G.edge_ids(limit=None)
        edge_id_1, edge_id_2 = edge_ids[0], edge_ids[1]

        # Select A->B and add its reverse
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge(add_to_selection=True)

        # Should have both edges selected
        selected = G.get_selected_edges()
        assert edge_id_1 in selected
        assert edge_id_2 in selected
        assert len(selected) == 2

    def test_trav_reverse_edge_no_reverse(self):
        """Test reverse edge when no reverse exists."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)

        edge_ids = G.edge_ids(limit=None)
        edge_id = edge_ids[0]

        # Select A->B (no B->A exists)
        G.select_edges_by_id([edge_id]).trav_reverse_edge()

        # Selection should be empty
        selected = G.get_selected_edges()
        assert selected == []

    def test_trav_reverse_edge_multiple(self):
        """Test reverse edge with multiple selected edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "C", auto_create_nodes=True)

        edge_ids = G.edge_ids(limit=None)
        edge_id_1, edge_id_2, edge_id_3, edge_id_4 = (
            edge_ids[0],
            edge_ids[1],
            edge_ids[2],
            edge_ids[3],
        )

        # Select A->B and C->D
        G.select_edges_by_id([edge_id_1, edge_id_3]).trav_reverse_edge()

        # Should select B->A and D->C
        selected = G.get_selected_edges()
        assert edge_id_2 in selected
        assert edge_id_4 in selected
        assert len(selected) == 2


class TestTraversalChaining:
    """Tests for chaining multiple traversal operations."""

    def test_chain_out_out(self):
        """Test chaining trav_out twice."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)  # Creates chain: 1->2->3->4->5

        # Traverse 2 hops from node 1
        G.select_nodes(lambda n: n.label == "1").trav_out().trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == [3]  # Node ID 3

    def test_chain_out_in(self):
        """Test chaining trav_out then trav_in."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Traverse out from A (to B), then back in (to A)
        G.select_nodes(lambda n: n.label == "A").trav_out().trav_in()
        nodes = G.get_selected_nodes()
        assert nodes == [1]  # A=1

    def test_chain_node_to_edge_to_node(self):
        """Test chaining node->edge->node traversals."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Select A, get its outgoing edges, then traverse back to A
        G.select_nodes(lambda n: n.label == "A").trav_out_edge()

        # Edge selection active now
        edges = G.get_selected_edges()
        assert len(edges) == 1


class TestTraversalEdgeCases:
    """Edge cases and error conditions."""

    def test_trav_single_node_graph(self):
        """Test traversal in single-node graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")

        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []

    def test_trav_self_loop(self):
        """Test traversal with self-loop."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A", auto_create_nodes=True)  # Self-loop

        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == [1]  # A=1

    def test_trav_disconnected_graph(self):
        """Test traversal in disconnected graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse from A (should not reach C or D)
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == [2]  # B=2
        assert 3 not in nodes  # C=3
        assert 4 not in nodes  # D=4


class TestConditionalTraversal:
    """Tests for trav_out_until and trav_in_until."""

    def test_trav_out_until_simple_condition(self):
        """Test outward traversal stopping at a specific value."""
        G = ec.Graph(directed=True)
        G.add_path_graph(10)  # 1 -> 2 -> 3 -> ... -> 10

        # Add values 1-10 to nodes
        for i in range(1, 11):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at node 1, traverse until value == 8
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 8
            if G.get_node(nid).attributes
            else False
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["8"]

    def test_trav_in_until_simple_condition(self):
        """Test inward traversal stopping at a specific value."""
        G = ec.Graph(directed=True)
        G.add_path_graph(10)  # 1 -> 2 -> 3 -> ... -> 10

        # Add values 1-10 to nodes
        for i in range(1, 11):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at node 10, traverse backward until value == 1
        G.select_nodes(lambda n: n.label == "10")
        G.trav_in_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 1
            if G.get_node(nid).attributes
            else False
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["1"]

    def test_trav_out_until_multiple_values(self):
        """Test condition checking multiple values."""
        G = ec.Graph(directed=True)

        # Create two cycles
        # Cycle 1: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1
        for i in range(1, 7):
            next_node = (i % 6) + 1
            G.add_edge(str(i), str(next_node), auto_create_nodes=True)
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Cycle 2: 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 7
        for i in range(7, 13):
            next_node = ((i - 7 + 1) % 6) + 7
            G.add_edge(str(i), str(next_node), auto_create_nodes=True)
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at 1 and 7, traverse until value in [5, 6, 9]
        G.select_nodes(lambda n: n.label in ["1", "7"])
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") in [5, 6, 9]
            if G.get_node(nid).attributes
            else False,
            exclude_unmatched=True,
        )

        node_ids = G.get_selected_nodes()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        assert set(nodes) == {"5", "9"}  # Labels for matched nodes

    def test_trav_in_until_multiple_values(self):
        """Test inward traversal with multiple target values."""
        G = ec.Graph(directed=True)

        # Create two cycles
        # Cycle 1: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1
        for i in range(1, 7):
            next_node = (i % 6) + 1
            G.add_edge(str(i), str(next_node), auto_create_nodes=True)
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Cycle 2: 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 7
        for i in range(7, 13):
            next_node = ((i - 7 + 1) % 6) + 7
            G.add_edge(str(i), str(next_node), auto_create_nodes=True)
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at 6 and 12, traverse backward until value in [1, 2, 10]
        G.select_nodes(lambda n: n.label in ["6", "12"])
        G.trav_in_until(
            lambda nid: G.get_node(nid).attributes.get("value") in [1, 2, 10]
            if G.get_node(nid).attributes
            else False,
            exclude_unmatched=True,
        )

        node_ids = G.get_selected_nodes()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        assert set(nodes) == {"2", "10"}  # Labels for matched nodes

    def test_trav_out_until_max_steps(self):
        """Test that max_steps limits traversal."""
        G = ec.Graph(directed=True)
        G.add_path_graph(20)  # Long path

        for i in range(1, 21):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at 1, try to reach 20 but limit to 5 steps
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 20
            if G.get_node(nid).attributes
            else False,
            max_steps=5,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        # Should stop at node 6 (5 steps from 1)
        assert "6" in nodes
        assert "20" not in nodes

    def test_trav_out_until_add_to_selection(self):
        """Test add_to_selection includes all traversed nodes."""
        G = ec.Graph(directed=True)
        G.add_path_graph(10)

        for i in range(1, 11):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Traverse from 1 to 5, keeping all visited nodes
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 5
            if G.get_node(nid).attributes
            else False,
            add_to_selection=True,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_nodes()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        # Should include 1, 2, 3, 4, 5 (all visited)
        assert set(nodes) == {"1", "2", "3", "4", "5"}

    def test_trav_in_until_add_to_selection(self):
        """Test add_to_selection with inward traversal."""
        G = ec.Graph(directed=True)
        G.add_path_graph(10)

        for i in range(1, 11):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Traverse from 10 to 6, keeping all visited nodes
        G.select_nodes(lambda n: n.label == "10")
        G.trav_in_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 6
            if G.get_node(nid).attributes
            else False,
            add_to_selection=True,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_nodes()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        # Should include 10, 9, 8, 7, 6 (all visited)
        assert set(nodes) == {"6", "7", "8", "9", "10"}

    def test_trav_out_until_exclude_unmatched_false(self):
        """Test keeping unmatched nodes when exclude_unmatched=False."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)

        for i in range(1, 6):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Traverse but look for non-existent value
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 999
            if G.get_node(nid).attributes
            else False,
            max_steps=2,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        # Should get node 3 (2 steps from 1), even though it doesn't match
        assert "3" in nodes

    def test_trav_out_until_no_match_exclude(self):
        """Test that no nodes selected when exclude_unmatched=True and no match."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)

        for i in range(1, 6):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Traverse but look for non-existent value
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 999
            if G.get_node(nid).attributes
            else False,
            max_steps=2,
            exclude_unmatched=True,
        )

        nodes = G.get_selected_nodes()
        # Should have no selection (no matches and exclude_unmatched=True)
        assert len(nodes) == 0

    def test_trav_out_until_immediate_match(self):
        """Test when starting node already satisfies condition."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)

        for i in range(1, 6):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})

        # Start at node that already matches
        G.select_nodes(lambda n: n.label == "3")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value") == 3
            if G.get_node(nid).attributes
            else False
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["3"]

    def test_trav_out_until_branching_graph(self):
        """Test conditional traversal in branching graph."""
        G = ec.Graph(directed=True)

        # Create diamond pattern: 1 -> 2,3 -> 4
        G.add_edge("1", "2", auto_create_nodes=True)
        G.add_edge("1", "3", auto_create_nodes=True)
        G.add_edge("2", "4", auto_create_nodes=True)
        G.add_edge("3", "4", auto_create_nodes=True)

        # Set attributes using node IDs
        node_id_1 = G.get_id_from_label("1")
        node_id_2 = G.get_id_from_label("2")
        node_id_3 = G.get_id_from_label("3")
        node_id_4 = G.get_id_from_label("4")
        G.set_node_attrs("value", {node_id_1: 1, node_id_2: 2, node_id_3: 3, node_id_4: 4})

        # Traverse from 1 until value >= 3
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: G.get_node(nid).attributes.get("value", 0) >= 3
            if G.get_node(nid).attributes
            else False
        )

        node_ids = G.get_selected_nodes()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        # Should find node 3 (value=3) in first hop
        assert "3" in nodes

    def test_trav_in_until_empty_selection(self):
        """Test that empty selection returns empty."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)

        # No selection
        G.trav_in_until(lambda nid: True)

        nodes = G.get_selected_nodes()
        assert nodes == []

    def test_trav_out_until_complex_condition(self):
        """Test with complex lambda condition."""
        G = ec.Graph(directed=True)
        G.add_path_graph(10)

        for i in range(1, 11):
            node_id = G.get_id_from_label(str(i))
            G.set_node_attrs("value", {node_id: i})
            G.set_node_attrs("category", {node_id: "even" if i % 2 == 0 else "odd"})

        # Traverse until we find an even number >= 6
        G.select_nodes(lambda n: n.label == "1")
        G.trav_out_until(
            lambda nid: (
                G.get_node(nid).attributes.get("value", 0) >= 6
                and G.get_node(nid).attributes.get("category") == "even"
            )
            if G.get_node(nid).attributes
            else False
        )

        node_ids = G.get_selected_nodes()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["6"]


class TestBFS:
    """Test breadth-first search traversal."""

    def test_bfs_path_graph(self):
        """Test BFS on simple path graph."""
        g = ec.Graph()
        g.add_path_graph(5)

        result = g.bfs(1)
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5

    def test_bfs_karate_graph(self):
        """Test BFS on karate dataset - should visit all connected nodes."""
        g = ec.load_dataset("karate")

        result = g.bfs(1)
        # Should visit all 34 nodes starting from node 1
        assert result[0] == 1
        assert len(result) == 34
        # All nodes should be unique
        assert len(set(result)) == 34

    def test_bfs_star_graph(self):
        """Test BFS on star graph."""
        g = ec.Graph()
        g.add_star_graph(5)

        result = g.bfs(1)  # Start from hub
        # Hub first, then all leaves
        assert result[0] == 1
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}

    def test_bfs_cycle_graph(self):
        """Test BFS on cycle graph."""
        g = ec.Graph()
        g.add_cycle_graph(6)

        result = g.bfs(1)
        assert len(result) == 6
        assert set(result) == {1, 2, 3, 4, 5, 6}
        # First node should be start
        assert result[0] == 1

    def test_bfs_disconnected_graph(self):
        """Test BFS on disconnected graph - only visits connected component."""
        g = ec.Graph()
        g.add_path_graph(3)
        g.add_node(10)
        g.add_node(11)
        g.add_edge(10, 11)

        result = g.bfs(1)
        # Should only visit component containing node 1
        assert set(result) == {1, 2, 3}

    def test_bfs_single_node(self):
        """Test BFS on single isolated node."""
        g = ec.Graph()
        g.add_node(42)

        result = g.bfs(42)
        assert result == [42]

    def test_bfs_complete_graph(self):
        """Test BFS on complete graph."""
        g = ec.Graph()
        g.add_full_graph(5)

        result = g.bfs(1)
        # All nodes reachable from any node
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}
        assert result[0] == 1

    def test_bfs_with_label(self):
        """Test BFS using node label instead of ID."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.bfs("A")
        # Returns node IDs
        assert len(result) == 3

    def test_bfs_nonexistent_node(self):
        """Test BFS with non-existent start node."""
        g = ec.Graph()
        g.add_path_graph(5)

        with pytest.raises(ValueError):
            g.bfs(999)


class TestDFS:
    """Test depth-first search traversal."""

    def test_dfs_path_graph(self):
        """Test DFS on simple path graph."""
        g = ec.Graph()
        g.add_path_graph(5)

        result = g.dfs(1)
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5

    def test_dfs_karate_graph(self):
        """Test DFS on karate dataset - should explore depth first."""
        g = ec.load_dataset("karate")

        result = g.dfs(1)
        # Should visit all 34 nodes starting from node 1
        assert result[0] == 1
        assert len(result) == 34
        # All nodes should be unique
        assert len(set(result)) == 34

    def test_dfs_star_graph(self):
        """Test DFS on star graph."""
        g = ec.Graph()
        g.add_star_graph(5)

        result = g.dfs(1)  # Start from hub
        # Hub first, then leaves
        assert result[0] == 1
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}

    def test_dfs_cycle_graph(self):
        """Test DFS on cycle graph."""
        g = ec.Graph()
        g.add_cycle_graph(6)

        result = g.dfs(1)
        assert len(result) == 6
        assert set(result) == {1, 2, 3, 4, 5, 6}
        assert result[0] == 1

    def test_dfs_disconnected_graph(self):
        """Test DFS on disconnected graph - only visits connected component."""
        g = ec.Graph()
        g.add_path_graph(3)
        g.add_node(10)
        g.add_node(11)
        g.add_edge(10, 11)

        result = g.dfs(1)
        # Should only visit component containing node 1
        assert set(result) == {1, 2, 3}

    def test_dfs_single_node(self):
        """Test DFS on single isolated node."""
        g = ec.Graph()
        g.add_node(42)

        result = g.dfs(42)
        assert result == [42]

    def test_dfs_complete_graph(self):
        """Test DFS on complete graph."""
        g = ec.Graph()
        g.add_full_graph(5)

        result = g.dfs(1)
        # All nodes reachable from any node
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}
        assert result[0] == 1

    def test_dfs_with_label(self):
        """Test DFS using node label instead of ID."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.dfs("A")
        # Returns node IDs
        assert len(result) == 3

    def test_dfs_nonexistent_node(self):
        """Test DFS with non-existent start node."""
        g = ec.Graph()
        g.add_path_graph(5)

        with pytest.raises(ValueError):
            g.dfs(999)


class TestBFSvsDFS:
    """Compare BFS and DFS behavior."""

    def test_same_nodes_visited(self):
        """BFS and DFS should visit the same set of nodes."""
        g = ec.Graph()
        g.add_balanced_tree(k=3, h=4)

        start = 1
        bfs_result = g.bfs(start)
        dfs_result = g.dfs(start)

        # Same nodes visited
        assert set(bfs_result) == set(dfs_result)
        # Same count
        assert len(bfs_result) == len(dfs_result)

    def test_different_order_on_tree(self):
        """BFS and DFS should visit nodes in different order on tree."""
        g = ec.Graph()
        g.add_balanced_tree(k=2, h=3)

        bfs_result = g.bfs(1)
        dfs_result = g.dfs(1)

        # Same nodes
        assert set(bfs_result) == set(dfs_result)
        # Different order (except for root)
        assert bfs_result[0] == dfs_result[0]  # Both start at root
        # At least one difference in order
        assert bfs_result != dfs_result

    def test_same_order_on_path(self):
        """BFS and DFS should have same order on path graph."""
        g = ec.Graph()
        g.add_path_graph(10)

        bfs_result = g.bfs(1)
        dfs_result = g.dfs(1)

        # Same order for path
        assert bfs_result == dfs_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
