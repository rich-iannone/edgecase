import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_TRAVERSE


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
        nodes = sorted(G.get_selected_node_ids())
        assert nodes == [2, 3]  # B=2, C=3

    def test_trav_out_multi_hop(self):
        """Test chaining trav_out for multi-hop traversal."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse 2 hops from A
        G.select_nodes(lambda n: n.label == "A").trav_out().trav_out()
        nodes = G.get_selected_node_ids()
        assert nodes == [3]  # C=3

    def test_trav_in_directed(self):
        """Test inward traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse from C to predecessors
        G.select_nodes(lambda n: n.label == "C").trav_in()
        nodes = sorted(G.get_selected_node_ids())
        assert nodes == [1, 3]  # A=1, B=3 (C=2, D=4)

    def test_trav_both_directed(self):
        """Test both-direction traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("C", "B", auto_create_nodes=True)
        G.add_edge("B", "D", auto_create_nodes=True)

        # Traverse from B to all neighbors
        G.select_nodes(lambda n: n.label == "B").trav_both()
        nodes = sorted(G.get_selected_node_ids())
        assert nodes == [1, 3, 4]  # A=1, C=3, D=4

    def test_trav_out_undirected(self):
        """Test that undirected graphs work like trav_both."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)

        # In undirected, trav_out should get all neighbors
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = sorted(G.get_selected_node_ids())
        assert nodes == [2, 3]  # B=2, C=3

    def test_trav_empty_selection(self):
        """Test traversal with no selection does nothing."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)

        # No selection
        G.trav_out()
        nodes = G.get_selected_node_ids()
        assert nodes == []

    def test_trav_no_neighbors(self):
        """Test traversal from node with no neighbors."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")

        # Node A has no outgoing edges
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_node_ids()
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
        edges = G.get_selected_edge_ids()
        assert len(edges) == 2

        # Node selection should be cleared
        nodes = G.get_selected_node_ids()
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
        edges = G.get_selected_edge_ids()
        assert len(edges) == 2

        # Node selection should be cleared
        nodes = G.get_selected_node_ids()
        assert nodes == []

    def test_trav_out_edge_undirected(self):
        """Test edge traversal in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)

        # In undirected, trav_out_edge should get all incident edges
        G.select_nodes(lambda n: n.label == "A").trav_out_edge()
        edges = G.get_selected_edge_ids()
        assert len(edges) == 2

    def test_trav_edge_multiple_nodes(self):
        """Test edge traversal from multiple selected nodes."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Select both A and B
        G.select_nodes(lambda n: n.label in ["A", "B"]).trav_out_edge()
        edges = G.get_selected_edge_ids()
        assert len(edges) == 2


class TestReverseEdgeTraversal:
    """Tests for trav_reverse_edge."""

    def test_trav_reverse_edge_basic(self):
        """Test basic reverse edge traversal."""
        G = ec.Graph(directed=True)
        # Add bidirectional edges
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)

        edge_ids = G.get_edge_ids(limit=None)
        edge_id_1, edge_id_2 = edge_ids[0], edge_ids[1]

        # Select A->B edge
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge()

        # Should now select B->A edge
        selected = G.get_selected_edge_ids()
        assert edge_id_2 in selected
        assert edge_id_1 not in selected

    def test_trav_reverse_edge_add_to_selection(self):
        """Test reverse edge with add_to_selection=True."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)

        edge_ids = G.get_edge_ids(limit=None)
        edge_id_1, edge_id_2 = edge_ids[0], edge_ids[1]

        # Select A->B and add its reverse
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge(add_to_selection=True)

        # Should have both edges selected
        selected = G.get_selected_edge_ids()
        assert edge_id_1 in selected
        assert edge_id_2 in selected
        assert len(selected) == 2

    def test_trav_reverse_edge_no_reverse(self):
        """Test reverse edge when no reverse exists."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)

        edge_ids = G.get_edge_ids(limit=None)
        edge_id = edge_ids[0]

        # Select A->B (no B->A exists)
        G.select_edges_by_id([edge_id]).trav_reverse_edge()

        # Selection should be empty
        selected = G.get_selected_edge_ids()
        assert selected == []

    def test_trav_reverse_edge_multiple(self):
        """Test reverse edge with multiple selected edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "C", auto_create_nodes=True)

        edge_ids = G.get_edge_ids(limit=None)
        edge_id_1, edge_id_2, edge_id_3, edge_id_4 = (
            edge_ids[0],
            edge_ids[1],
            edge_ids[2],
            edge_ids[3],
        )

        # Select A->B and C->D
        G.select_edges_by_id([edge_id_1, edge_id_3]).trav_reverse_edge()

        # Should select B->A and D->C
        selected = G.get_selected_edge_ids()
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
        nodes = G.get_selected_node_ids()
        assert nodes == [3]  # Node ID 3

    def test_chain_out_in(self):
        """Test chaining trav_out then trav_in."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Traverse out from A (to B), then back in (to A)
        G.select_nodes(lambda n: n.label == "A").trav_out().trav_in()
        nodes = G.get_selected_node_ids()
        assert nodes == [1]  # A=1

    def test_chain_node_to_edge_to_node(self):
        """Test chaining node->edge->node traversals."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Select A, get its outgoing edges, then traverse back to A
        G.select_nodes(lambda n: n.label == "A").trav_out_edge()

        # Edge selection active now
        edges = G.get_selected_edge_ids()
        assert len(edges) == 1


class TestTraversalEdgeCases:
    """Edge cases and error conditions."""

    def test_trav_single_node_graph(self):
        """Test traversal in single-node graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")

        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_node_ids()
        assert nodes == []

    def test_trav_self_loop(self):
        """Test traversal with self-loop."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A", auto_create_nodes=True)  # Self-loop

        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_node_ids()
        assert nodes == [1]  # A=1

    def test_trav_disconnected_graph(self):
        """Test traversal in disconnected graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)

        # Traverse from A (should not reach C or D)
        G.select_nodes(lambda n: n.label == "A").trav_out()
        nodes = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 8
                if G.get_node(nid).attributes
                else False
            )
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 1
                if G.get_node(nid).attributes
                else False
            )
        )

        node_ids = G.get_selected_node_ids()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["1"]

    @pytest.mark.parametrize("method", ["trav_out_until", "trav_in_until"])
    def test_add_to_selection_keeps_existing_selection(self, method):
        """Regression: add_to_selection=True still cleared the selection first."""
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C", "X"])
        g.add_edges(
            [("A", "B"), ("B", "C")] if method == "trav_out_until" else [("C", "B"), ("B", "A")]
        )
        a, c, x = (g.get_id_from_label(n) for n in "ACX")
        g.select_nodes_by_id([a, x])
        getattr(g, method)(lambda nid: nid == c, add_to_selection=True)
        # A and X (unmatched starting nodes) stay; the match C is added
        assert sorted(g.get_selected_node_ids()) == sorted([a, c, x])
        g.clear_selection().select_nodes_by_id([a, x])
        getattr(g, method)(lambda nid: nid == c)
        assert g.get_selected_node_ids() == [c]

    def test_trav_in_until_undirected_follows_every_edge(self):
        """Regression: on undirected graphs the walk only followed edges stored toward the node."""
        # Path stored as A-B, B-C, C-D; walking "in" from A used to find nothing
        G = ec.Graph().add_nodes(["A", "B", "C", "D"])
        G.add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        d = G.get_id_from_label("D")
        G.select_nodes(lambda n: n.label == "A")
        G.trav_in_until(lambda nid: nid == d)
        assert G.get_selected_node_ids() == [d]

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
            lambda nid: (
                G.get_node(nid).attributes.get("value") in [5, 6, 9]
                if G.get_node(nid).attributes
                else False
            ),
            exclude_unmatched=True,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") in [1, 2, 10]
                if G.get_node(nid).attributes
                else False
            ),
            exclude_unmatched=True,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 20
                if G.get_node(nid).attributes
                else False
            ),
            max_steps=5,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 5
                if G.get_node(nid).attributes
                else False
            ),
            add_to_selection=True,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 6
                if G.get_node(nid).attributes
                else False
            ),
            add_to_selection=True,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 999
                if G.get_node(nid).attributes
                else False
            ),
            max_steps=2,
            exclude_unmatched=False,
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 999
                if G.get_node(nid).attributes
                else False
            ),
            max_steps=2,
            exclude_unmatched=True,
        )

        nodes = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value") == 3
                if G.get_node(nid).attributes
                else False
            )
        )

        node_ids = G.get_selected_node_ids()
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
            lambda nid: (
                G.get_node(nid).attributes.get("value", 0) >= 3
                if G.get_node(nid).attributes
                else False
            )
        )

        node_ids = G.get_selected_node_ids()
        nodes = sorted([G.get_label_from_id(nid) for nid in node_ids])
        # Should find node 3 (value=3) in first hop
        assert "3" in nodes

    def test_trav_in_until_empty_selection(self):
        """Test that empty selection returns empty."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)

        # No selection
        G.trav_in_until(lambda nid: True)

        nodes = G.get_selected_node_ids()
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
                (
                    G.get_node(nid).attributes.get("value", 0) >= 6
                    and G.get_node(nid).attributes.get("category") == "even"
                )
                if G.get_node(nid).attributes
                else False
            )
        )

        node_ids = G.get_selected_node_ids()
        nodes = [G.get_label_from_id(nid) for nid in node_ids]
        assert nodes == ["6"]


class TestBFS:
    """Test breadth-first search traversal."""

    def test_bfs_path_graph(self):
        """Test BFS on simple path graph."""
        g = ec.Graph()
        g.add_path_graph(5)

        result = g.bfs(1, return_list=True)
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5

    def test_bfs_karate_graph(self):
        """Test BFS on karate dataset - should visit all connected nodes."""
        g = ec.load_dataset("karate")

        result = g.bfs(1, return_list=True)
        # Should visit all 34 nodes starting from node 1
        assert result[0] == 1
        assert len(result) == 34
        # All nodes should be unique
        assert len(set(result)) == 34

    def test_bfs_star_graph(self):
        """Test BFS on star graph."""
        g = ec.Graph()
        g.add_star_graph(5)

        result = g.bfs(1, return_list=True)  # Start from hub
        # Hub first, then all leaves
        assert result[0] == 1
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}

    def test_bfs_cycle_graph(self):
        """Test BFS on cycle graph."""
        g = ec.Graph()
        g.add_cycle_graph(6)

        result = g.bfs(1, return_list=True)
        assert len(result) == 6
        assert set(result) == {1, 2, 3, 4, 5, 6}
        # First node should be start
        assert result[0] == 1

    def test_bfs_disconnected_graph(self):
        """Test BFS on disconnected graph - only visits connected component."""
        g = ec.Graph()
        g.add_path_graph(3)
        g.add_node(label="10")
        g.add_node(label="11")
        g.add_edge("10", "11")

        result = g.bfs(1, return_list=True)
        # Should only visit component containing node 1
        assert set(result) == {1, 2, 3}

    def test_bfs_single_node(self):
        """Test BFS on single isolated node."""
        g = ec.Graph()
        g.add_node(label="23")

        result = g.bfs("23", return_list=True)
        # Returns node IDs, but we search by label
        assert len(result) == 1

    def test_bfs_complete_graph(self):
        """Test BFS on complete graph."""
        g = ec.Graph()
        g.add_full_graph(5)

        result = g.bfs("1", return_list=True)
        # All nodes reachable from any node
        assert len(result) == 5

    def test_bfs_with_label(self):
        """Test BFS using node label instead of ID."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.bfs("A", return_list=True)
        # Returns node IDs
        assert len(result) == 3

    def test_bfs_nonexistent_node(self):
        """Test BFS with non-existent start node."""
        g = ec.Graph()
        g.add_path_graph(5)

        with pytest.raises(ValueError):
            g.bfs(999, return_list=True)


class TestDFS:
    """Test depth-first search traversal."""

    def test_dfs_path_graph(self):
        """Test DFS on simple path graph."""
        g = ec.Graph()
        g.add_path_graph(5)

        result = g.dfs(1, return_list=True)
        assert result == [1, 2, 3, 4, 5]
        assert len(result) == 5

    def test_dfs_karate_graph(self):
        """Test DFS on karate dataset - should explore depth first."""
        g = ec.load_dataset("karate")

        result = g.dfs(1, return_list=True)
        # Should visit all 34 nodes starting from node 1
        assert result[0] == 1
        assert len(result) == 34
        # All nodes should be unique
        assert len(set(result)) == 34

    def test_dfs_star_graph(self):
        """Test DFS on star graph."""
        g = ec.Graph()
        g.add_star_graph(5)

        result = g.dfs(1, return_list=True)  # Start from hub
        # Hub first, then leaves
        assert result[0] == 1
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}

    def test_dfs_cycle_graph(self):
        """Test DFS on cycle graph."""
        g = ec.Graph()
        g.add_cycle_graph(6)

        result = g.dfs(1, return_list=True)
        assert len(result) == 6
        assert set(result) == {1, 2, 3, 4, 5, 6}
        assert result[0] == 1

    def test_dfs_disconnected_graph(self):
        """Test DFS on disconnected graph - only visits connected component."""
        g = ec.Graph()
        g.add_path_graph(3)
        g.add_node(label="10")
        g.add_node(label="11")
        g.add_edge("10", "11")

        result = g.dfs(1, return_list=True)
        # Should only visit component containing node 1
        assert set(result) == {1, 2, 3}

    def test_dfs_single_node(self):
        """Test DFS on single isolated node."""
        g = ec.Graph()
        g.add_node(label="23")

        result = g.dfs("23", return_list=True)
        # Returns node IDs, but we search by label
        assert len(result) == 1

    def test_dfs_complete_graph(self):
        """Test DFS on complete graph."""
        g = ec.Graph()
        g.add_full_graph(5)

        result = g.dfs(1, return_list=True)
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

        result = g.dfs("A", return_list=True)
        # Returns node IDs
        assert len(result) == 3

    def test_dfs_nonexistent_node(self):
        """Test DFS with non-existent start node."""
        g = ec.Graph()
        g.add_path_graph(5)

        with pytest.raises(ValueError):
            g.dfs(999, return_list=True)


class TestBFSvsDFS:
    """Compare BFS and DFS behavior."""

    def test_same_nodes_visited(self):
        """BFS and DFS should visit the same set of nodes."""
        g = ec.Graph()
        g.add_balanced_tree(k=3, h=4)

        start = 1
        bfs_result = g.bfs(start, return_list=True)
        dfs_result = g.dfs(start, return_list=True)

        # Same nodes visited
        assert set(bfs_result) == set(dfs_result)
        # Same count
        assert len(bfs_result) == len(dfs_result)

    def test_different_order_on_tree(self):
        """BFS and DFS should visit nodes in different order on tree."""
        g = ec.Graph()
        g.add_balanced_tree(k=2, h=3)

        bfs_result = g.bfs(1, return_list=True)
        dfs_result = g.dfs(1, return_list=True)

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

        bfs_result = g.bfs(1, return_list=True)
        dfs_result = g.dfs(1, return_list=True)

        # Same order for path
        assert bfs_result == dfs_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# BFS/DFS edges, layers, trees and orders, validated against NetworkX.


def _triangle_tail() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(["A", "B", "C", "D"])
        .add_edges([("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")])
    )


def _pair(seed: int, directed: bool, n: int = 16):
    """A NetworkX graph and an EdgeCase copy with identical node and edge insertion order."""
    rnd = random.Random(seed)
    G0 = nx.gnp_random_graph(n, rnd.choice([0.1, 0.18, 0.3]), seed=seed, directed=directed)
    G = (nx.DiGraph if directed else nx.Graph)()
    G.add_nodes_from(range(n))
    G.add_edges_from(G0.edges)
    g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(n)])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
    return G, g


def _edge_labs(g, edges):
    return [(int(g.get_label_from_id(p)), int(g.get_label_from_id(c))) for p, c in edges]


def _labs(g, ids):
    return [int(g.get_label_from_id(i)) for i in ids]


class TestBfsEdgesLayersTrees:
    @pytest.mark.parametrize("directed", [False, True])
    def test_edges_layers_and_depth_match_networkx(self, directed):
        for seed in range(25):
            G, g = _pair(seed, directed)
            for s in (0, 5):
                sid = g.get_id_from_label(str(s))
                assert _edge_labs(g, g.bfs_edges(sid)) == list(nx.bfs_edges(G, s))
                assert _edge_labs(g, g.bfs_edges(sid, depth_limit=2)) == list(
                    nx.bfs_edges(G, s, depth_limit=2)
                )
                ours = [sorted(_labs(g, layer)) for layer in g.bfs_layers(sid)]
                assert ours == [sorted(layer) for layer in nx.bfs_layers(G, [s])]
                if directed:
                    assert _edge_labs(g, g.bfs_edges(sid, mode="in")) == list(
                        nx.bfs_edges(G, s, reverse=True)
                    )
            srcs = [g.get_id_from_label(str(x)) for x in (0, 7, 11)]
            ours = [sorted(_labs(g, layer)) for layer in g.bfs_layers(srcs)]
            assert ours == [sorted(layer) for layer in nx.bfs_layers(G, [0, 7, 11])]

    def test_consistent_with_bfs_order(self):
        for seed in range(10):
            _, g = _pair(seed, False)
            order = g.bfs(1, return_list=True)
            layered = [n for layer in g.bfs_layers(1) for n in layer]
            assert layered == order
            assert [c for _, c in g.bfs_edges(1)] == order[1:]

    def test_tree_graph(self):
        g = _triangle_tail()
        g.set_node_attrs("x", {g.get_id_from_label("C"): 5})
        tree = g.bfs_tree("A")
        assert tree.is_directed() and tree.is_arborescence()
        assert sorted(tree.get_edges_str_repr(use_labels=True)) == ["A->B", "A->C", "C->D"]
        assert tree.get_node_attrs("x") == {g.get_id_from_label("C"): 5}  # IDs and attrs kept
        assert g.count_edges() == 4  # original untouched
        grid = ec.Graph().add_grid_2d(rows=3, cols=3)
        assert len(grid.bfs_tree(1).dag_longest_path()) - 1 == 4
        assert grid.bfs_tree(1, depth_limit=1).count_nodes() == 3

    def test_errors(self):
        g = _triangle_tail()
        with pytest.raises(ValueError, match="depth_limit"):
            g.bfs_edges("A", depth_limit=-1)
        with pytest.raises(ValueError, match="mode"):
            g.bfs_layers("A", mode="sideways")
        with pytest.raises(ValueError):
            g.bfs_tree("nope")

    def test_bfs_records_real_tree_edges(self):
        g = _triangle_tail()
        g.bfs("A", return_list=True)
        step = g.steps[-1]
        assert step.kind == KIND_TRAVERSE
        assert [s.code for s in step.sub_steps] == ["visit A", "A → B", "A → C", "C → D"]

    def test_bfs_layers_step(self):
        path = ec.Graph().add_path_graph(4)
        path.bfs_layers(1)
        assert path.steps[-1].delta.node_groups == {"1": 0, "2": 1, "3": 2, "4": 3}


class TestDfsEdgesAndOrders:
    @pytest.mark.parametrize("directed", [False, True])
    def test_orders_and_edges_match_networkx(self, directed):
        for seed in range(25):
            G, g = _pair(seed, directed)
            for s in (0, 5):
                sid = g.get_id_from_label(str(s))
                assert _edge_labs(g, g.dfs_edges(sid)) == list(nx.dfs_edges(G, s))
                assert _labs(g, g.dfs_postorder_nodes(sid)) == list(nx.dfs_postorder_nodes(G, s))
                assert _edge_labs(g, g.dfs_edges(sid, depth_limit=2)) == list(
                    nx.dfs_edges(G, s, depth_limit=2)
                )
            assert _edge_labs(g, g.dfs_edges()) == list(nx.dfs_edges(G))
            assert _labs(g, g.dfs_postorder_nodes()) == list(nx.dfs_postorder_nodes(G))

    def test_preorder_consistent_with_dfs(self):
        for seed in range(15):
            for directed in (False, True):
                _, g = _pair(seed, directed)
                pre = g.dfs(1, return_list=True)
                assert [1] + [c for _, c in g.dfs_edges(1)] == pre

    def test_postorder_with_depth_limit_keeps_nodes_at_the_limit(self):
        path = ec.Graph().add_path_graph(5)
        assert path.dfs_postorder_nodes(1, depth_limit=2) == [3, 2, 1]  # NetworkX omits 3

    def test_bfs_and_dfs_differ_on_the_triangle(self):
        g = _triangle_tail()
        lab = g.get_label_from_id
        assert [(lab(p), lab(c)) for p, c in g.bfs_edges("A")] == [
            ("A", "B"), ("A", "C"), ("C", "D"),
        ]  # fmt: skip
        assert [(lab(p), lab(c)) for p, c in g.dfs_edges("A")] == [
            ("A", "B"), ("B", "C"), ("C", "D"),
        ]  # fmt: skip

    def test_forest_and_postorder_reverse_is_topological(self):
        dag = ec.Graph(directed=True).add_nodes(list("ABCDEF"))
        dag.add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("E", "F")])
        forest = dag.dfs_tree()
        assert forest.count_nodes() == 6 and forest.count_edges() == 4
        order = list(reversed(dag.dfs_postorder_nodes()))
        pos = {n: i for i, n in enumerate(order)}
        assert all(pos[s] < pos[t] for _, s, t in dag.get_edges_as_tuples())
        grid = ec.Graph().add_grid_2d(rows=3, cols=3)
        assert len(grid.dfs_tree(1).dag_longest_path()) - 1 == 8

    def test_traversal_methods_record_playback(self):
        g = _triangle_tail()
        g.dfs_edges("A")
        step = g.steps[-1]
        assert step.kind == KIND_TRAVERSE and step.method == "dfs_edges"
        assert [s.code for s in step.sub_steps] == ["visit A", "A → B", "B → C", "C → D"]
        assert step.code == "G.dfs_edges('A')"
        g.add_nodes(["E"]).dfs_edges()
        assert g.steps[-1].sub_steps[-1].narration == "Start a new tree at E"


class TestTraversalMaps:
    """BFS/DFS predecessor and successor maps, descendants at a distance, and edge BFS/DFS on
    multigraphs, validated against NetworkX."""

    def test_maps_against_networkx(self):
        for seed in range(100):
            rnd = random.Random(seed)
            directed = seed % 2 == 0
            G = nx.gnp_random_graph(rnd.randint(1, 10), 0.3, seed=seed, directed=directed)
            g = ec.Graph(directed=directed).add_nodes([str(n) for n in G])
            for u, v in G.edges:
                g.add_edge(str(u), str(v))
            lab = lambda x: int(g.get_label_from_id(x))  # noqa: E731
            assert {lab(k): lab(v) for k, v in g.bfs_predecessors("0").items()} == dict(
                nx.bfs_predecessors(G, 0)
            )
            assert {lab(k): [lab(x) for x in v] for k, v in g.bfs_successors("0").items()} == dict(
                nx.bfs_successors(G, 0)
            )
            assert {
                lab(k): lab(v) for k, v in g.dfs_predecessors("0").items()
            } == nx.dfs_predecessors(G, 0)
            assert {
                lab(k): [lab(x) for x in v] for k, v in g.dfs_successors().items()
            } == nx.dfs_successors(G)
            for d in range(4):
                assert {
                    lab(x) for x in g.descendants_at_distance("0", d)
                } == nx.descendants_at_distance(G, 0, d)

    def test_edge_traversals_on_multigraphs(self):
        for seed in range(100):
            rnd = random.Random(seed)
            directed = seed % 2 == 0
            G = (nx.MultiDiGraph if directed else nx.MultiGraph)()
            n = rnd.randint(1, 7)
            G.add_nodes_from(range(n))
            g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(n)])
            for _ in range(rnd.randint(0, 12)):
                u, v = rnd.randrange(n), rnd.randrange(n)
                G.add_edge(u, v)
                g.add_edge(str(u), str(v))
            lab = lambda x: int(g.get_label_from_id(x))  # noqa: E731
            for orientation in (
                [None, "original", "reverse", "ignore"] if directed else [None, "original"]
            ):
                for src in (None, 0):
                    for ours_fn, ref_fn in ((g.edge_bfs, nx.edge_bfs), (g.edge_dfs, nx.edge_dfs)):
                        ours = ours_fn(None if src is None else "0", orientation=orientation)
                        ref = list(ref_fn(G, src, orientation=orientation))
                        assert [(lab(e[0]), lab(e[1])) + tuple(e[3:]) for e in ours] == [
                            (r[0], r[1]) + tuple(r[3:]) for r in ref
                        ]

    def test_examples_and_errors(self):
        g = (
            ec.Graph(directed=True)
            .add_nodes(["A", "B", "C"])
            .add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        )
        assert len(g.edge_bfs("A")) == 3 and len(g.bfs_edges("A")) == 2
        a, b, c = (g.get_id_from_label(x) for x in "ABC")
        assert g.bfs_successors("C", mode="in") == {c: [b], b: [a]}
        lonely = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("B", "A")
        assert lonely.bfs_successors("A") == {lonely.get_id_from_label("A"): []}
        assert g.descendants_at_distance("A", 9) == set()
        with pytest.raises(ValueError, match="orientation"):
            g.edge_bfs(orientation="sideways")
