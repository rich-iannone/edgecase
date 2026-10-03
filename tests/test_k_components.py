"""Tests for k-components (directed and undirected)."""

import pytest
import edgecase as ec
import networkx as nx


@pytest.fixture
def triangle_undirected():
    """Undirected triangle: A-B, B-C, C-A."""
    g = ec.Graph(directed=False)
    g.add_node(label="A").add_node(label="B").add_node(label="C")
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    return g


@pytest.fixture
def k4_undirected():
    """Complete undirected graph on 4 nodes."""
    g = ec.Graph(directed=False)
    nodes = ["A", "B", "C", "D"]
    for n in nodes:
        g.add_node(label=n)
    for i, s in enumerate(nodes):
        for t in nodes[i + 1 :]:
            g.add_edge(s, t)
    return g


@pytest.fixture
def barbell_undirected():
    """Two triangles connected by a bridge edge (C-D)."""
    g = ec.Graph(directed=False)
    for n in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(label=n)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    g.add_edge("C", "D")
    g.add_edge("D", "E").add_edge("E", "F").add_edge("F", "D")
    return g


@pytest.fixture
def directed_triangle():
    """Directed cycle: A -> B -> C -> A."""
    g = ec.Graph(directed=True)
    g.add_node(label="A").add_node(label="B").add_node(label="C")
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    return g


@pytest.fixture
def k3_directed():
    """Complete directed graph on 3 nodes (6 edges)."""
    g = ec.Graph(directed=True)
    nodes = ["A", "B", "C"]
    for n in nodes:
        g.add_node(label=n)
    for s in nodes:
        for t in nodes:
            if s != t:
                g.add_edge(s, t)
    return g


@pytest.fixture
def k4_directed():
    """Complete directed graph on 4 nodes (12 edges)."""
    g = ec.Graph(directed=True)
    nodes = ["A", "B", "C", "D"]
    for n in nodes:
        g.add_node(label=n)
    for s in nodes:
        for t in nodes:
            if s != t:
                g.add_edge(s, t)
    return g


class TestKComponentsReturnsDict:
    def test_returns_dict(self, triangle_undirected):
        result = triangle_undirected.k_components()
        assert isinstance(result, dict)

    def test_keys_are_positive_ints(self, k4_undirected):
        result = k4_undirected.k_components()
        for k in result.keys():
            assert isinstance(k, int)
            assert k >= 1

    def test_values_are_lists_of_lists(self, k4_undirected):
        result = k4_undirected.k_components()
        for k, comps in result.items():
            assert isinstance(comps, list)
            for comp in comps:
                assert isinstance(comp, list)


class TestKComponentsUndirected:
    def test_triangle_k1_and_k2(self, triangle_undirected):
        """Triangle is 2-connected: appears at k=1 and k=2."""
        result = triangle_undirected.k_components()
        assert 1 in result
        assert 2 in result
        assert 3 not in result

    def test_k4_up_to_k3(self, k4_undirected):
        """K4 is 3-connected: appears at k=1, 2, 3."""
        result = k4_undirected.k_components()
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert 4 not in result

    def test_k4_single_component_per_level(self, k4_undirected):
        result = k4_undirected.k_components()
        for k, comps in result.items():
            assert len(comps) == 1
            assert len(comps[0]) == 4

    def test_barbell_k1_two_components(self, barbell_undirected):
        """Barbell: at k=1 one big component, at k=2 the two triangles."""
        result = barbell_undirected.k_components()
        assert 1 in result
        assert len(result[1]) == 1
        assert len(result[1][0]) == 6

    def test_path_only_k1(self):
        """Path graph has connectivity 1 everywhere."""
        g = ec.Graph(directed=False)
        for i in range(5):
            g.add_node(label=str(i))
        for i in range(4):
            g.add_edge(str(i), str(i + 1))
        result = g.k_components()
        assert 1 in result
        assert 2 not in result


class TestKComponentsDirected:
    def test_directed_triangle_only_k1(self, directed_triangle):
        """Directed 3-cycle: removing any node disconnects it. Only k=1."""
        result = directed_triangle.k_components()
        assert 1 in result
        assert 2 not in result

    def test_k3_directed_up_to_k2(self, k3_directed):
        """Complete directed K3: 2-connected."""
        result = k3_directed.k_components()
        assert 1 in result
        assert 2 in result
        assert 3 not in result

    def test_k4_directed_up_to_k3(self, k4_directed):
        """Complete directed K4: 3-connected."""
        result = k4_directed.k_components()
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert 4 not in result

    def test_two_directed_cycles_separate_sccs(self):
        """Two disjoint directed cycles: two separate 1-components."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D", "E", "F"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        g.add_edge("D", "E").add_edge("E", "F").add_edge("F", "D")
        result = g.k_components()
        assert 1 in result
        assert len(result[1]) == 2

    def test_dag_no_components(self):
        """A DAG has no non-trivial SCCs, so no k-components."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C")
        result = g.k_components()
        assert result == {}


class TestKComponentsEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph(directed=False)
        assert g.k_components() == {}

    def test_single_node(self):
        g = ec.Graph(directed=False)
        g.add_node(label="A")
        assert g.k_components() == {}

    def test_single_edge_undirected(self):
        g = ec.Graph(directed=False)
        g.add_node(label="A").add_node(label="B").add_edge("A", "B")
        result = g.k_components()
        assert 1 in result
        assert len(result[1]) == 1

    def test_isolated_nodes_excluded(self):
        """Isolated nodes don't form k-components."""
        g = ec.Graph(directed=False)
        for n in ["A", "B", "C", "X", "Y"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        result = g.k_components()
        assert 1 in result
        for comp in result[1]:
            assert len(comp) >= 2

    def test_does_not_mutate_graph(self, k4_undirected):
        n = k4_undirected.count_nodes()
        e = k4_undirected.count_edges()
        k4_undirected.k_components()
        assert k4_undirected.count_nodes() == n
        assert k4_undirected.count_edges() == e

    def test_directed_does_not_mutate(self, k4_directed):
        n = k4_directed.count_nodes()
        e = k4_directed.count_edges()
        k4_directed.k_components()
        assert k4_directed.count_nodes() == n
        assert k4_directed.count_edges() == e


class TestKComponentsHierarchy:
    def test_higher_k_subset_of_lower(self, k4_undirected):
        """Every (k+1)-component should be a subset of some k-component."""
        result = k4_undirected.k_components()
        keys = sorted(result.keys())
        for i in range(len(keys) - 1):
            k_low = keys[i]
            k_high = keys[i + 1]
            for high_comp in result[k_high]:
                high_set = set(high_comp)
                found_parent = False
                for low_comp in result[k_low]:
                    if high_set.issubset(set(low_comp)):
                        found_parent = True
                        break
                assert found_parent, (
                    f"k={k_high} component {high_comp} not subset of any k={k_low} component"
                )

    def test_hierarchy_directed(self, k4_directed):
        result = k4_directed.k_components()
        keys = sorted(result.keys())
        for i in range(len(keys) - 1):
            k_low = keys[i]
            k_high = keys[i + 1]
            for high_comp in result[k_high]:
                high_set = set(high_comp)
                found_parent = False
                for low_comp in result[k_low]:
                    if high_set.issubset(set(low_comp)):
                        found_parent = True
                        break
                assert found_parent

    @staticmethod
    def _labels(g):
        return {
            k: sorted(sorted(g.get_label_from_id(n) for n in comp) for comp in comps)
            for k, comps in g.k_components().items()
        }

    def test_recurses_past_cut_vertex(self):
        """A K4 with a pendant is only 1-connected, but the K4 is still a 3-component."""
        g = ec.Graph()
        for label in "ABCDE":
            g.add_node(label=label)
        for u, v in [(a, b) for a in "ABCD" for b in "ABCD" if a < b] + [("D", "E")]:
            g.add_edge(u, v)
        got = self._labels(g)
        assert got[1] == [["A", "B", "C", "D", "E"]]
        assert got[3] == [["A", "B", "C", "D"]]

    def test_barbell_matches_networkx(self):
        """Barbell graph: k-components match NetworkX at every level."""
        ref = nx.barbell_graph(4, 1)
        g = ec.Graph()
        for n in ref.nodes:
            g.add_node(label=str(n))
        for u, v in ref.edges:
            g.add_edge(str(u), str(v))
        want = {
            k: sorted(sorted(str(n) for n in comp) for comp in comps)
            for k, comps in nx.k_components(ref).items()
        }
        assert self._labels(g) == want
