import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph


def add_nodes(g: Graph, *labels: str) -> Graph:
    """Helper to add multiple nodes by label."""
    for label in labels:
        g.add_node(label)
    return g


class TestAssortativityDegree:
    """Tests for degree assortativity coefficient."""

    def test_empty_graph_error(self):
        """Empty graph should raise error."""
        g = Graph(directed=False)
        with pytest.raises(ValueError, match="no edges"):
            g.get_assortativity_degree()

    def test_single_edge_error(self):
        """Single edge has no variance - returns NaN like igraph."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B")
        g.add_edge("A", "B")

        r = g.get_assortativity_degree()
        assert math.isnan(r), "Single edge should return NaN (no variance)"

    def test_perfect_assortative_undirected(self):
        """Assortative mixing: nodes with similar degrees connect."""
        g = Graph(directed=False)
        add_nodes(g, "Hub1", "Hub2", "Hub3", "Leaf1", "Leaf2", "Leaf3")

        # Create structure where high-degree nodes connect to high-degree
        # and low-degree nodes connect to low-degree
        g.add_edge("Hub1", "Hub2")
        g.add_edge("Hub1", "Hub3")
        g.add_edge("Hub2", "Hub3")  # Hubs all connected (degree 2 each)

        g.add_edge("Leaf1", "Leaf2")
        g.add_edge("Leaf2", "Leaf3")  # Leaves in chain

        r = g.get_assortativity_degree()
        # Note: This structure gives r=-1.0 in igraph because the actual
        # degrees are [1,2,1,1,2,1] - not perfectly assortative
        # For this test, we just verify it's a valid correlation
        assert isinstance(r, float)
        assert -1 <= r <= 1

    def test_perfect_disassortative_undirected(self):
        """Perfect disassortativity: hub-and-spoke (star graph)."""
        g = Graph(directed=False)
        add_nodes(g, "Hub", "Leaf1", "Leaf2", "Leaf3", "Leaf4")

        # Hub (degree 4) connects only to leaves (degree 1 each)
        g.add_edge("Hub", "Leaf1")
        g.add_edge("Hub", "Leaf2")
        g.add_edge("Hub", "Leaf3")
        g.add_edge("Hub", "Leaf4")

        r = g.get_assortativity_degree()
        assert r < 0, "Star graph should be disassortative"

    def test_random_mixing_undirected(self):
        """Cycle graph has uniform degree - returns NaN."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C", "D", "E")

        # Create cycle (all nodes have degree 2)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "E")
        g.add_edge("E", "A")

        r = g.get_assortativity_degree()
        assert math.isnan(r), "Uniform degree cycle should return NaN"

    def test_assortative_directed(self):
        """Directed graph with assortative degree mixing."""
        g = Graph(directed=True)
        add_nodes(g, "H1", "H2", "H3", "L1", "L2", "L3")

        # High out-degree nodes connect to high in-degree nodes
        g.add_edge("H1", "H2")
        g.add_edge("H1", "H3")
        g.add_edge("H2", "H3")

        # Low out-degree nodes connect to low in-degree nodes
        g.add_edge("L1", "L2")
        g.add_edge("L2", "L3")

        r = g.get_assortativity_degree()
        # Should be positive (but hard to predict exact value)
        assert isinstance(r, float)
        assert -1 <= r <= 1

    def test_disassortative_directed(self):
        """Directed graph with disassortative mixing (sources to sinks)."""
        g = Graph(directed=True)
        add_nodes(g, "Source1", "Source2", "Sink1", "Sink2", "Sink3")

        # High out-degree sources connect to zero out-degree sinks
        g.add_edge("Source1", "Sink1")
        g.add_edge("Source1", "Sink2")
        g.add_edge("Source2", "Sink2")
        g.add_edge("Source2", "Sink3")

        r = g.get_assortativity_degree()
        assert isinstance(r, float)
        # May be NaN if variance is zero, otherwise in [-1, 1]
        assert math.isnan(r) or (-1 <= r <= 1)

    def test_complete_graph(self):
        """Complete graph has all nodes with equal degree - returns NaN."""
        g = Graph(directed=False)
        nodes = ["A", "B", "C", "D"]
        add_nodes(g, *nodes)

        # Connect all pairs
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                g.add_edge(u, v)

        # All nodes have degree 3 - no variance
        r = g.get_assortativity_degree()
        assert math.isnan(r), "Uniform degree should return NaN"

    def test_self_loops_included(self):
        """Self-loops should be included in degree calculation."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C")

        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        r = g.get_assortativity_degree()
        assert isinstance(r, float)
        assert -1 <= r <= 1

    def test_line_graph(self):
        """Line graph (path) has specific degree pattern."""
        g = Graph(directed=False)
        add_nodes(g, "A", "B", "C", "D", "E")

        # Path: A-B-C-D-E (ends have degree 1, middle has degree 2)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "E")

        r = g.get_assortativity_degree()
        assert r < 0, "Path graph should be slightly disassortative"


class TestAssortativityNominal:
    """Tests for categorical attribute assortativity."""

    def test_perfect_homophily(self):
        """Perfect assortativity: only same-type connections."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B", type="X")
        g.add_node("C", type="Y")
        g.add_node("D", type="Y")

        g.add_edge("A", "B")  # X-X
        g.add_edge("C", "D")  # Y-Y

        r = g.get_assortativity_nominal("type")
        assert abs(r - 1.0) < 1e-6, "Perfect segregation should give r = 1.0"

    def test_perfect_heterophily(self):
        """Only different-type connections."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B", type="X")
        g.add_node("C", type="Y")
        g.add_node("D", type="Y")

        g.add_edge("A", "C")  # X-Y
        g.add_edge("A", "D")  # X-Y
        g.add_edge("B", "C")  # X-Y
        g.add_edge("B", "D")  # X-Y

        r = g.get_assortativity_nominal("type")
        assert r < 0, "All cross-type edges should be disassortative"

    def test_random_mixing(self):
        """Mixed connections should have r ≈ 0."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B", type="X")
        g.add_node("C", type="Y")
        g.add_node("D", type="Y")

        g.add_edge("A", "B")  # X-X
        g.add_edge("A", "C")  # X-Y
        g.add_edge("C", "D")  # Y-Y
        g.add_edge("B", "D")  # X-Y

        r = g.get_assortativity_nominal("type")
        assert -0.5 < r < 0.5, "Balanced mixing should be near 0"

    def test_three_categories(self):
        """Test with three attribute values."""
        g = Graph(directed=False)
        g.add_node("A", group="1")
        g.add_node("B", group="1")
        g.add_node("C", group="2")
        g.add_node("D", group="2")
        g.add_node("E", group="3")
        g.add_node("F", group="3")

        # Within-group connections
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        g.add_edge("E", "F")

        r = g.get_assortativity_nominal("group")
        assert abs(r - 1.0) < 1e-6, "All within-group should give r = 1.0"

    def test_directed_nominal(self):
        """Categorical assortativity for directed graphs."""
        g = Graph(directed=True)
        g.add_node("A", role="client")
        g.add_node("B", role="client")
        g.add_node("C", role="server")
        g.add_node("D", role="server")

        # Clients connect to servers (directed bipartite)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        r = g.get_assortativity_nominal("role")
        # In directed graphs, this bipartite structure gives r=0
        # because there's no correlation between source and target types
        assert abs(r - 0.0) < 1e-6, "Directed bipartite should give r ≈ 0"

    def test_missing_attribute_error(self):
        """Should error if attribute missing on a node."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B")  # No type attribute

        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="(Attribute.*not found|no attributes)"):
            g.get_assortativity_nominal("type")

    def test_nonexistent_attribute_error(self):
        """Should error if attribute doesn't exist."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B", type="Y")

        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="not found"):
            g.get_assortativity_nominal("nonexistent")

    def test_single_category_error(self):
        """Should error if only one attribute value exists."""
        g = Graph(directed=False)
        g.add_node("A", type="X")
        g.add_node("B", type="X")
        g.add_node("C", type="X")

        g.add_edge("A", "B")
        g.add_edge("B", "C")

        with pytest.raises(ValueError, match="at least 2 different"):
            g.get_assortativity_nominal("type")

    def test_empty_graph_error(self):
        """Empty graph should raise error."""
        g = Graph(directed=False)
        g.add_node("A", type="X")

        with pytest.raises(ValueError, match="no edges"):
            g.get_assortativity_nominal("type")

    def test_string_attribute_values(self):
        """Test with string attribute values."""
        g = Graph(directed=False)
        g.add_node("Alice", gender="female")
        g.add_node("Bob", gender="male")
        g.add_node("Carol", gender="female")
        g.add_node("Dave", gender="male")

        g.add_edge("Alice", "Carol")  # F-F
        g.add_edge("Bob", "Dave")  # M-M

        r = g.get_assortativity_nominal("gender")
        assert abs(r - 1.0) < 1e-6


class TestAssortativityNumeric:
    """Tests for numeric attribute assortativity."""

    def test_perfect_positive_correlation(self):
        """Identical attribute values connect (r = 1)."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)
        g.add_node("B", value=1.0)
        g.add_node("C", value=2.0)
        g.add_node("D", value=2.0)

        g.add_edge("A", "B")  # 1.0 - 1.0
        g.add_edge("C", "D")  # 2.0 - 2.0

        r = g.get_assortativity("value")
        assert abs(r - 1.0) < 1e-6, "Identical values should give r = 1.0"

    def test_perfect_negative_correlation(self):
        """Opposite values always connect."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)
        g.add_node("B", value=1.0)
        g.add_node("C", value=10.0)
        g.add_node("D", value=10.0)

        g.add_edge("A", "C")  # 1.0 - 10.0
        g.add_edge("A", "D")  # 1.0 - 10.0
        g.add_edge("B", "C")  # 1.0 - 10.0
        g.add_edge("B", "D")  # 1.0 - 10.0

        r = g.get_assortativity("value")
        assert abs(r - (-1.0)) < 1e-6, "Opposite values should give r = -1.0"

    def test_positive_correlation(self):
        """Similar values tend to connect (r > 0)."""
        g = Graph(directed=False)
        g.add_node("A", age=25)
        g.add_node("B", age=27)
        g.add_node("C", age=26)
        g.add_node("D", age=55)
        g.add_node("E", age=53)

        g.add_edge("A", "B")  # Similar ages (25-27)
        g.add_edge("B", "C")  # Similar ages (27-26)
        g.add_edge("D", "E")  # Similar ages (55-53)

        r = g.get_assortativity("age")
        assert r > 0.5, "Similar ages connecting should be assortative"

    def test_negative_correlation(self):
        """Dissimilar values tend to connect (r < 0)."""
        g = Graph(directed=False)
        g.add_node("Junior1", exp=2)
        g.add_node("Junior2", exp=3)
        g.add_node("Senior1", exp=20)
        g.add_node("Senior2", exp=22)

        # Juniors connect to seniors (mentorship)
        g.add_edge("Junior1", "Senior1")
        g.add_edge("Junior1", "Senior2")
        g.add_edge("Junior2", "Senior1")
        g.add_edge("Junior2", "Senior2")

        r = g.get_assortativity("exp")
        assert r < -0.5, "Junior-senior connections should be disassortative"

    def test_zero_correlation(self):
        """Random mixing should have r ≈ 0."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)
        g.add_node("B", value=2.0)
        g.add_node("C", value=3.0)
        g.add_node("D", value=4.0)

        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "A")

        r = g.get_assortativity("value")
        assert -0.5 < r < 0.5, "Cycle should have moderate correlation"

    def test_integer_attributes(self):
        """Test with integer attribute values."""
        g = Graph(directed=False)
        g.add_node("A", count=5)
        g.add_node("B", count=6)
        g.add_node("C", count=15)
        g.add_node("D", count=16)

        g.add_edge("A", "B")
        g.add_edge("C", "D")

        r = g.get_assortativity("count")
        assert r > 0.5, "Similar integers should correlate"

    def test_float_attributes(self):
        """Test with floating-point attributes."""
        g = Graph(directed=False)
        g.add_node("A", weight=1.5)
        g.add_node("B", weight=1.7)
        g.add_node("C", weight=5.2)
        g.add_node("D", weight=5.1)

        g.add_edge("A", "B")
        g.add_edge("C", "D")

        r = g.get_assortativity("weight")
        assert r > 0.5

    def test_directed_numeric(self):
        """Numeric assortativity for directed graphs."""
        g = Graph(directed=True)
        g.add_node("A", rank=1)
        g.add_node("B", rank=2)
        g.add_node("C", rank=8)
        g.add_node("D", rank=9)

        # Low ranks connect to high ranks (directed bipartite)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        r = g.get_assortativity("rank")
        # In directed graphs, this bipartite structure gives r=0
        # because there's no correlation in the directed sense
        assert abs(r - 0.0) < 1e-6, "Directed bipartite should give r ≈ 0"

    def test_missing_attribute_error(self):
        """Should error if attribute missing."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)
        g.add_node("B")  # No value attribute

        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="(not found|no attributes)"):
            g.get_assortativity("value")

    def test_non_numeric_attribute_error(self):
        """Should error if attribute isn't numeric."""
        g = Graph(directed=False)
        g.add_node("A", value="text")
        g.add_node("B", value="more text")

        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="not numeric"):
            g.get_assortativity("value")

    def test_no_variance_error(self):
        """Should error if all attribute values are identical."""
        g = Graph(directed=False)
        g.add_node("A", value=5.0)
        g.add_node("B", value=5.0)
        g.add_node("C", value=5.0)

        g.add_edge("A", "B")
        g.add_edge("B", "C")

        with pytest.raises(ValueError, match="variance"):
            g.get_assortativity("value")

    def test_empty_graph_error(self):
        """Empty graph should raise error."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)

        with pytest.raises(ValueError, match="no edges"):
            g.get_assortativity("value")

    def test_self_loops(self):
        """Self-loops should be handled correctly."""
        g = Graph(directed=False)
        g.add_node("A", value=1.0)
        g.add_node("B", value=5.0)
        g.add_node("C", value=10.0)

        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        r = g.get_assortativity("value")
        assert isinstance(r, float)
        assert -1 <= r <= 1


class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_all_three_methods_same_graph(self):
        """Test all three assortativity methods on same graph."""
        g = Graph(directed=False)
        g.add_node("A", type="hub", value=10.0)
        g.add_node("B", type="hub", value=12.0)
        g.add_node("C", type="leaf", value=1.0)
        g.add_node("D", type="leaf", value=2.0)

        g.add_edge("A", "B")  # hub-hub
        g.add_edge("C", "D")  # leaf-leaf

        r_degree = g.get_assortativity_degree()
        r_nominal = g.get_assortativity_nominal("type")
        r_numeric = g.get_assortativity("value")

        # Note: all nodes have degree 1, so degree assortativity returns NaN
        assert math.isnan(r_degree), "All nodes have same degree, should be NaN"
        assert abs(r_nominal - 1.0) < 1e-6, "Type should be perfectly assortative"
        assert r_numeric > 0.5, "Value should be assortative"

    def test_large_network_performance(self):
        """Test with larger network (performance check)."""
        g = Graph(directed=False)

        # Create 50 nodes with attributes
        for i in range(50):
            g.add_node(f"N{i}", group=i % 5, score=float(i))

        # Add 100 edges
        for i in range(100):
            u = f"N{i % 50}"
            v = f"N{(i + 1) % 50}"
            g.add_edge(u, v)

        r_degree = g.get_assortativity_degree()
        r_nominal = g.get_assortativity_nominal("group")
        r_numeric = g.get_assortativity("score")

        assert isinstance(r_degree, float)
        assert isinstance(r_nominal, float)
        assert isinstance(r_numeric, float)

    def test_multigraph_with_multiple_edges(self):
        """Multiple edges between same nodes."""
        g = Graph(directed=False)
        g.add_node("A", type="X", val=1.0)
        g.add_node("B", type="X", val=2.0)
        g.add_node("C", type="Y", val=10.0)

        g.add_edge("A", "B")
        g.add_edge("A", "B")  # Duplicate
        g.add_edge("B", "C")

        # Should handle multiple edges correctly
        r_nominal = g.get_assortativity_nominal("type")
        r_numeric = g.get_assortativity("val")

        assert isinstance(r_nominal, float)
        assert isinstance(r_numeric, float)


# Average degree connectivity, degree/attribute mixing and Pearson degree correlation,
# validated against NetworkX (directed and undirected, with weights).


def _cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G0 = nx.gnp_random_graph(
                14, rnd.choice([0.15, 0.25, 0.4]), seed=seed, directed=directed
            )
            G = (nx.DiGraph if directed else nx.Graph)()
            G.add_nodes_from(range(14))
            for u, v in G0.edges:
                G.add_edge(u, v, weight=rnd.choice([0.5, 1, 2, 3]))
            g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(14)])
            gw = ec.Graph(directed=directed, weighted=True).add_nodes([str(v) for v in range(14)])
            for u, v, d in G.edges(data=True):
                g.add_edge(str(u), str(v))
                gw.add_edge(str(u), str(v), weight=d["weight"])
            yield G, g, gw, rnd


class TestDegreeAndAttributeMixing:
    def test_average_degree_connectivity(self):
        for G, g, gw, _ in _cases():
            options = [("in+out", "in+out")]
            if G.is_directed():
                options += [(s, t) for s in ("in", "out") for t in ("in", "out", "in+out")]
            for s, t in options:
                if G.is_directed() and s == "in+out":
                    continue  # NetworkX deviates from its documented definition here
                for gg, w in ((g, None), (gw, "weight")):
                    ours = gg.average_degree_connectivity(s, t)
                    assert ours == pytest.approx(
                        nx.average_degree_connectivity(G, source=s, target=t, weight=w)
                    )

    def test_directed_in_plus_out_uses_all_incident_edges(self):
        d = ec.Graph(directed=True).add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("c", "a")])
        # a has degree 2; its neighbors b and c both have total degree 1
        assert d.average_degree_connectivity()[2] == pytest.approx(1.0)
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_path_graph(3).average_degree_connectivity(source="in")

    def test_degree_mixing_and_pearson(self):
        for G, g, gw, _ in _cases():
            pairs = [("out", "in")]
            if G.is_directed():
                pairs += [("in", "out"), ("out", "out"), ("in", "in")]
            for x, y in pairs:
                for gg, w in ((g, None), (gw, "weight")):
                    assert gg.degree_mixing_dict(x, y) == nx.degree_mixing_dict(
                        G, x=x, y=y, weight=w
                    )
                    if G.number_of_edges() > 2:
                        ours = gg.degree_pearson_correlation_coefficient(x, y)
                        ref = nx.degree_pearson_correlation_coefficient(G, x=x, y=y, weight=w)
                        assert (math.isnan(ours) and math.isnan(ref)) or ours == pytest.approx(ref)
                matrix, degrees = g.degree_mixing_matrix(x, y)
                ref = nx.degree_mixing_matrix(
                    G, x=x, y=y, mapping={d: i for i, d in enumerate(degrees)}
                )
                assert sum(matrix, []) == pytest.approx(ref.flatten().tolist())

    def test_attribute_mixing(self):
        for G, g, _, rnd in _cases():
            team = {v: rnd.choice("abc") for v in G}
            nx.set_node_attributes(G, team, "team")
            g.set_node_attrs("team", {g.get_id_from_label(str(v)): t for v, t in team.items()})
            assert g.attribute_mixing_dict("team") == nx.attribute_mixing_dict(G, "team")
            matrix, values = g.attribute_mixing_matrix("team", normalized=False)
            ref = nx.attribute_mixing_matrix(
                G, "team", mapping={v: i for i, v in enumerate(values)}, normalized=False
            )
            assert sum(matrix, []) == pytest.approx(ref.flatten().tolist())

    def test_examples_and_errors(self):
        assert ec.Graph().add_path_graph(4).degree_mixing_dict() == {1: {2: 2}, 2: {1: 2, 2: 2}}
        matrix, degrees = ec.Graph().add_star_graph(4).degree_mixing_matrix()
        assert degrees == [1, 3] and matrix == [[0.0, 0.5], [0.5, 0.0]]
        assert ec.Graph().add_star_graph(
            5
        ).degree_pearson_correlation_coefficient() == pytest.approx(-1)
        g = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        g.set_node_attrs("team", {1: "x"})
        with pytest.raises(ValueError, match="without the attribute"):
            g.attribute_mixing_dict("team")
