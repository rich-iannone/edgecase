import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ids(g, *labels):
    return [g.get_id_from_label(lbl) for lbl in labels]


def make_triangle():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="A", target="C")
        .add_edge(source="B", target="C")
    )


def make_star():
    g = ec.Graph()
    g.add_node(label="H")
    for lbl in "ABCD":
        g.add_node(label=lbl)
        g.add_edge(source="H", target=lbl)
    return g


def make_chain():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="C", target="D")
    )


# ===========================================================================
# topological_overlap — basic values
# ===========================================================================


class TestTOMBasic:
    def test_triangle_all_one(self):
        g = make_triangle()
        tom = g.topological_overlap(return_dict=True)
        a, b, c = ids(g, "A", "B", "C")
        # In a triangle, every pair shares all neighbors and is connected
        assert abs(tom[a][b] - 1.0) < 1e-10
        assert abs(tom[a][c] - 1.0) < 1e-10
        assert abs(tom[b][c] - 1.0) < 1e-10

    def test_diagonal_is_one(self):
        g = make_triangle()
        tom = g.topological_overlap(return_dict=True)
        for nid in g.get_node_ids():
            assert tom[nid][nid] == 1.0

    def test_chain_endpoints_low(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        a, d = ids(g, "A", "D")
        # A and D share no neighbors and aren't connected
        assert tom[a][d] == 0.0

    def test_chain_adjacent_nonzero(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        a, b = ids(g, "A", "B")
        # A-B connected, share no common neighbors
        # TOM(A,B) = (0 + 1) / (min(1,2) + 1 - 1) = 1/1 = 1.0
        assert abs(tom[a][b] - 1.0) < 1e-10

    def test_chain_middle_pair(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        b, c = ids(g, "B", "C")
        # B-C connected, share no common neighbors (A is B's only other, D is C's)
        # TOM(B,C) = (0 + 1) / (min(2,2) + 1 - 1) = 1/2 = 0.5
        assert abs(tom[b][c] - 0.5) < 1e-10


# ===========================================================================
# topological_overlap — star graph
# ===========================================================================


class TestTOMStar:
    def test_leaves_connected_through_hub(self):
        g = make_star()
        tom = g.topological_overlap(return_dict=True)
        a, b = ids(g, "A", "B")
        # A and B are not connected, share hub H as a neighbor
        # |N(A)∩N(B)| = 1 (hub), a_AB = 0
        # TOM = (1 + 0) / (min(1,1) + 1 - 0) = 1/2 = 0.5
        assert abs(tom[a][b] - 0.5) < 1e-10

    def test_hub_to_leaf(self):
        g = make_star()
        tom = g.topological_overlap(return_dict=True)
        h, a = ids(g, "H", "A")
        # H(deg 4) and A(deg 1): connected, no shared neighbors
        # TOM = (0 + 1) / (min(4,1) + 1 - 1) = 1/1 = 1.0
        assert abs(tom[h][a] - 1.0) < 1e-10

    def test_symmetry(self):
        g = make_star()
        tom = g.topological_overlap(return_dict=True)
        a, b = ids(g, "A", "B")
        assert abs(tom[a][b] - tom[b][a]) < 1e-10


# ===========================================================================
# topological_overlap — properties
# ===========================================================================


class TestTOMProperties:
    def test_values_between_zero_and_one(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        for ni, row in tom.items():
            for nj, val in row.items():
                assert 0.0 <= val <= 1.0 + 1e-10

    def test_symmetric(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        node_ids = g.get_node_ids()
        for i in node_ids:
            for j in node_ids:
                assert abs(tom[i][j] - tom[j][i]) < 1e-10

    def test_all_nodes_present(self):
        g = make_chain()
        tom = g.topological_overlap(return_dict=True)
        node_ids = g.get_node_ids()
        assert len(tom) == len(node_ids)
        for nid in node_ids:
            assert nid in tom
            assert len(tom[nid]) == len(node_ids)


# ===========================================================================
# topological_overlap — chaining
# ===========================================================================


class TestTOMChaining:
    def test_return_self(self):
        g = make_triangle()
        result = g.topological_overlap()
        assert result is g

    def test_return_dict(self):
        g = make_triangle()
        result = g.topological_overlap(return_dict=True)
        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_triangle()
        g.topological_overlap()
        found = any(c[0] == "topological_overlap" for c in g._computations)
        assert found


# ===========================================================================
# topological_overlap — edge cases
# ===========================================================================


class TestTOMEdgeCases:
    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        tom = g.topological_overlap(return_dict=True)
        a = g.get_id_from_label("A")
        assert tom[a][a] == 1.0

    def test_two_connected_nodes(self):
        g = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
        tom = g.topological_overlap(return_dict=True)
        a, b = ids(g, "A", "B")
        # No shared neighbors, connected: (0+1)/(min(1,1)+1-1) = 1
        assert abs(tom[a][b] - 1.0) < 1e-10

    def test_two_disconnected_nodes(self):
        g = ec.Graph().add_node(label="A").add_node(label="B")
        tom = g.topological_overlap(return_dict=True)
        a, b = ids(g, "A", "B")
        assert tom[a][b] == 0.0

    def test_empty_graph(self):
        g = ec.Graph()
        tom = g.topological_overlap(return_dict=True)
        assert tom == {}

    def test_complete_graph(self):
        g = ec.Graph()
        for lbl in "ABCD":
            g.add_node(label=lbl)
        for i, a in enumerate("ABCD"):
            for b in "ABCD"[i + 1 :]:
                g.add_edge(source=a, target=b)
        tom = g.topological_overlap(return_dict=True)
        # Complete graph: all pairs have TOM = 1.0
        for ni, row in tom.items():
            for nj, val in row.items():
                assert abs(val - 1.0) < 1e-10

    def test_large_graph(self):
        g = ec.Graph()
        for i in range(30):
            g.add_node(label=str(i))
        for i in range(29):
            g.add_edge(source=str(i), target=str(i + 1))
        tom = g.topological_overlap(return_dict=True)
        assert len(tom) == 30
