"""Round trips, NetworkX interop and edge cases for the file formats in `edgecase.io`."""

import json
import math
import random
import warnings

import edgecase as ec
import networkx as nx
import pytest
from edgecase import _core as rust
from edgecase.steps import KIND_BUILD_NODES

# =============================================================================
# Helpers
# =============================================================================


def nodes_of(g):
    """`(label, attributes)` per node, in ID order."""
    return [
        (g.get_node(n).label, g.get_node(n).attributes or {})
        for n in sorted(g.get_node_ids(limit=None))
    ]


def edges_of(g):
    """`(source label, target label, weight, attributes)` per edge, in ID order."""
    out = []
    for eid, _, _ in sorted(g._graph.edges()):
        e = g.get_edge(eid)
        out.append(
            (
                g.get_label_from_id(e.source),
                g.get_label_from_id(e.target),
                e.weight,
                e.attributes or {},
            )
        )
    return out


def pairs(g):
    """The edge multiset by label (endpoint order ignored for undirected graphs)."""
    out = []
    for s, t, _, _ in edges_of(g):
        out.append((s, t) if g.is_directed() else tuple(sorted((s, t))))
    return sorted(out)


def nx_pairs(G):
    out = [
        (str(u), str(v)) if G.is_directed() else tuple(sorted((str(u), str(v))))
        for u, v in G.edges()
    ]
    return sorted(out)


def assert_same(a, b):
    assert nodes_of(a) == nodes_of(b)
    assert edges_of(a) == edges_of(b)
    assert a.is_directed() == b.is_directed()
    assert a.is_weighted() == b.is_weighted()
    assert a.get_graph_name() == b.get_graph_name()


def rich_graph(directed=False):
    """Typed attributes, parallel edges with distinct weights, a self-loop and an isolated node."""
    g = ec.Graph(directed=directed, weighted=True, graph_name="Rich graph")
    g.add_node(label="A", type="person", age=30, score=1.5, active=True, tags=["x", "y"])
    g.add_node(label="B", type="org", age=2, meta={"k": 1, "nested": [1, 2]})
    g.add_node(label='C & "quoted" é')
    g.add_node(label="Lonely")
    g.add_edge("A", "B", weight=2.5, rel="knows")
    g.add_edge("A", "B", weight=3.5, rel="likes")
    g.add_edge("B", 'C & "quoted" é', weight=1.0, since=2020)
    g.add_edge("A", "A", weight=0.5)
    return g.layout_circular()


def big_graph(n=1500):
    """More nodes than `get_node_ids()` returns by default."""
    return ec.Graph().add_path_graph(n)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


# =============================================================================
# Round trips that keep everything
# =============================================================================


class TestLosslessRoundTrips:
    @pytest.mark.parametrize("directed", [False, True])
    def test_graphml(self, tmp, directed):
        g = rich_graph(directed)
        ec.to_graphml(g, tmp / "g.graphml")
        assert_same(ec.from_graphml(tmp / "g.graphml"), g)

    @pytest.mark.parametrize("version", ["1.2draft", "1.3"])
    @pytest.mark.parametrize("directed", [False, True])
    def test_gexf(self, tmp, directed, version):
        g = rich_graph(directed)
        ec.to_gexf(g, tmp / "g.gexf", version=version)
        assert_same(ec.from_gexf(tmp / "g.gexf"), g)

    @pytest.mark.parametrize("directed", [False, True])
    def test_node_link(self, tmp, directed):
        g = rich_graph(directed)
        data = ec.to_node_link(g, tmp / "g.json")
        assert_same(ec.from_node_link(data), g)
        assert_same(ec.from_node_link(tmp / "g.json"), g)
        assert_same(ec.from_node_link(str(tmp / "g.json")), g)
        assert_same(ec.from_node_link(json.dumps(data)), g)

    def test_node_link_links_key(self):
        g = rich_graph()
        data = ec.to_node_link(g, edges="links")
        assert "links" in data and "edges" not in data
        assert_same(ec.from_node_link(data), g)

    @pytest.mark.parametrize("directed", [False, True])
    def test_cytoscape(self, tmp, directed):
        g = rich_graph(directed)
        data = ec.to_cytoscape(g, tmp / "g.cyjs")
        assert_same(ec.from_cytoscape(data), g)
        assert_same(ec.from_cytoscape(tmp / "g.cyjs"), g)

    def test_gml_keeps_all_but_booleans(self, tmp):
        g = rich_graph()
        ec.to_gml(g, tmp / "g.gml")
        h = ec.from_gml(tmp / "g.gml")
        expected = []
        for label, attrs in nodes_of(g):
            # GML has no booleans: they come back as 1 and 0
            expected.append(
                (label, {k: int(v) if isinstance(v, bool) else v for k, v in attrs.items()})
            )
        assert nodes_of(h) == expected
        assert edges_of(h) == edges_of(g)
        assert h.get_graph_name() == "Rich graph"

    def test_unlabeled_nodes(self, tmp):
        g = ec.Graph()
        g.add_node()
        g.add_node(label="1")
        g.add_node()
        g.add_edge(1, 3)
        for write, read, name in [
            (ec.to_graphml, ec.from_graphml, "g.graphml"),
            (ec.to_gexf, ec.from_gexf, "g.gexf"),
            (ec.to_gml, ec.from_gml, "g.gml"),
            (ec.to_pajek, ec.from_pajek, "g.net"),
        ]:
            write(g, tmp / name)
            h = read(tmp / name)
            labels = [label for label, _ in nodes_of(h)]
            assert len(set(labels)) == 3 and "1" in labels, name
            assert h.count_edges() == 1

    def test_empty_graph(self, tmp):
        g = ec.Graph()
        for write, read, name in [
            (ec.to_graphml, ec.from_graphml, "g.graphml"),
            (ec.to_gexf, ec.from_gexf, "g.gexf"),
            (ec.to_gml, ec.from_gml, "g.gml"),
            (ec.to_pajek, ec.from_pajek, "g.net"),
        ]:
            write(g, tmp / name)
            assert read(tmp / name).count_nodes() == 0
        assert ec.from_node_link(ec.to_node_link(g)).count_nodes() == 0
        assert ec.from_cytoscape(ec.to_cytoscape(g)).count_nodes() == 0
        assert ec.from_graph6(ec.to_graph6(g)).count_nodes() == 0
        assert ec.from_sparse6(ec.to_sparse6(g)).count_nodes() == 0


# =============================================================================
# Regressions in the original readers and writers
# =============================================================================


class TestRegressions:
    def parallel(self):
        g = ec.Graph(weighted=True)
        g.add_nodes(["A", "B"])
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("A", "B", weight=2.0)
        g.add_edge("B", "A", weight=3.0)
        return g

    @pytest.mark.parametrize(
        "write,read,name",
        [
            (ec.to_graphml, ec.from_graphml, "g.graphml"),
            (ec.to_gml, ec.from_gml, "g.gml"),
            (ec.to_gexf, ec.from_gexf, "g.gexf"),
            (ec.to_pajek, ec.from_pajek, "g.net"),
        ],
    )
    def test_parallel_edge_weights(self, tmp, write, read, name):
        write(self.parallel(), tmp / name)
        assert [w for _, _, w, _ in edges_of(read(tmp / name))] == [1.0, 2.0, 3.0]

    def test_parallel_edge_weights_tsv(self, tmp):
        ec.to_tsv_edge_list(self.parallel(), tmp / "g.tsv")
        h = ec.from_tsv_edge_list(tmp / "g.tsv", has_header=True, weight_col="weight")
        assert [w for _, _, w, _ in edges_of(h)] == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        "write,read,name",
        [
            (ec.to_graphml, ec.from_graphml, "g.graphml"),
            (ec.to_gml, ec.from_gml, "g.gml"),
            (ec.to_gexf, ec.from_gexf, "g.gexf"),
            (ec.to_pajek, ec.from_pajek, "g.net"),
            (ec.to_sif, ec.from_sif, "g.sif"),
        ],
    )
    def test_more_than_1000_nodes(self, tmp, write, read, name):
        g = big_graph()
        write(g, tmp / name)
        h = read(tmp / name)
        assert (h.count_nodes(), h.count_edges()) == (1500, 1499)
        assert [label for label, _ in nodes_of(h)] == [str(i) for i in range(1, 1501)]

    def test_more_than_1000_nodes_tsv_and_json(self, tmp):
        g = big_graph()
        ec.to_tsv_edge_list(g, tmp / "g.tsv")
        assert ec.from_tsv_edge_list(tmp / "g.tsv", has_header=True).count_nodes() == 1500
        assert ec.from_node_link(ec.to_node_link(g)).count_nodes() == 1500
        assert ec.from_cytoscape(ec.to_cytoscape(g)).count_nodes() == 1500
        assert ec.from_graph6(ec.to_graph6(g)).count_edges() == 1499
        assert ec.from_sparse6(ec.to_sparse6(g)).count_edges() == 1499

    def test_node_order_follows_file(self, tmp):
        g = ec.Graph()
        g.add_nodes([f"n{i}" for i in range(40, 0, -1)])
        ec.to_graphml(g, tmp / "g.graphml")
        assert [lb for lb, _ in nodes_of(ec.from_graphml(tmp / "g.graphml"))] == [
            f"n{i}" for i in range(40, 0, -1)
        ]

    def test_graphml_namespace_is_the_real_one(self, tmp):
        ec.to_graphml(rich_graph(), tmp / "g.graphml")
        text = (tmp / "g.graphml").read_text()
        assert 'xmlns="http://graphml.graphdrawing.org/xmlns"' in text
        # NetworkX finds nothing in a file with the wrong namespace
        assert nx.read_graphml(tmp / "g.graphml").number_of_nodes() == 4

    def test_sif_keeps_interaction_types(self, tmp):
        g = rich_graph()
        ec.to_sif(g, tmp / "g.sif")
        lines = (tmp / "g.sif").read_text().splitlines()
        assert "A\tknows\tB" in lines and "A\tlikes\tB" in lines
        assert "A\tinteracts_with\tA" in lines and "Lonely" in lines
        h = ec.from_sif(tmp / "g.sif")
        assert sorted(a.get("rel") for *_, a in edges_of(h)) == [
            "interacts_with",
            "interacts_with",
            "knows",
            "likes",
        ]

    def test_sif_groups_targets(self, tmp):
        g = ec.Graph().add_star_graph(4)
        ec.to_sif(g, tmp / "g.sif", interaction_type="pp")
        assert (tmp / "g.sif").read_text() == "1\tpp\t2\t3\t4\n"

    @pytest.mark.parametrize(
        "write,read,name",
        [
            (ec.to_graphml, ec.from_graphml, "g.graphml"),
            (ec.to_gml, ec.from_gml, "g.gml"),
            (ec.to_gexf, ec.from_gexf, "g.gexf"),
            (ec.to_pajek, ec.from_pajek, "g.net"),
            (ec.to_sif, ec.from_sif, "g.sif"),
            (ec.to_tsv_edge_list, ec.from_tsv_edge_list, "g.tsv"),
        ],
    )
    def test_read_is_a_single_step(self, tmp, monkeypatch, write, read, name):
        monkeypatch.chdir(tmp)
        write(ec.karate_club(), name)
        h = read(name)
        assert len(h.steps) == 1
        step = h.steps[-1]
        assert step.kind == KIND_BUILD_NODES
        assert step.method == read.__name__
        assert step.code == f"{read.__name__}({name!r})"
        assert len(step.delta.nodes_added) == h.count_nodes()

    def test_json_and_six_bit_reads_are_single_steps(self):
        g = ec.karate_club()
        for h, method in [
            (ec.from_node_link(ec.to_node_link(g)), "from_node_link"),
            (ec.from_cytoscape(ec.to_cytoscape(g)), "from_cytoscape"),
            (ec.from_graph6(ec.to_graph6(g)), "from_graph6"),
            (ec.from_sparse6(ec.to_sparse6(g)), "from_sparse6"),
        ]:
            assert [s.method for s in h.steps] == [method]
        assert ec.from_node_link(ec.to_node_link(g)).steps[-1].code == "from_node_link(<dict>)"


# =============================================================================
# NetworkX reads our files
# =============================================================================


class TestNetworkXReadsOurFiles:
    def test_graphml(self, tmp):
        g = rich_graph()
        ec.to_graphml(g, tmp / "g.graphml")
        G = nx.read_graphml(tmp / "g.graphml")
        assert isinstance(G, nx.MultiGraph)
        assert nx_pairs(G) == pairs(g)
        assert G.nodes["A"]["age"] == 30 and G.nodes["A"]["active"] is True
        assert sorted(d["weight"] for *_, d in G.edges(data=True)) == [0.5, 1.0, 2.5, 3.5]
        assert G.graph["name"] == "Rich graph"

    @pytest.mark.parametrize("directed", [False, True])
    def test_gexf(self, tmp, directed):
        g = rich_graph(directed)
        ec.to_gexf(g, tmp / "g.gexf")
        G = nx.read_gexf(tmp / "g.gexf", relabel=True)
        assert G.is_directed() == directed
        assert nx_pairs(G) == pairs(g)
        assert G.nodes["A"]["type"] == "person" and G.nodes["A"]["score"] == 1.5
        x, y = g.get_node("B").attributes["pos"]
        assert G.nodes["B"]["viz"]["position"]["x"] == pytest.approx(x)
        assert G.nodes["B"]["viz"]["position"]["y"] == pytest.approx(y)

    def test_gml(self, tmp):
        g = rich_graph()
        ec.to_gml(g, tmp / "g.gml")
        G = nx.read_gml(tmp / "g.gml")
        assert nx_pairs(G) == pairs(g)
        assert G.nodes['C & "quoted" é'] == {"pos": g.get_node(3).attributes["pos"]}
        assert G.nodes["A"]["tags"] == ["x", "y"]
        assert G.nodes["B"]["meta"] == {"k": 1, "nested": [1, 2]}

    def test_node_link(self):
        g = rich_graph(directed=True)
        G = nx.node_link_graph(ec.to_node_link(g), edges="edges")
        assert G.is_directed() and G.is_multigraph()
        assert nx_pairs(G) == pairs(g)
        assert G.nodes["B"]["meta"] == {"k": 1, "nested": [1, 2]}

    def test_cytoscape(self):
        g = rich_graph()
        G = nx.cytoscape_graph(ec.to_cytoscape(g))
        assert nx_pairs(G) == pairs(g)
        assert G.nodes["A"]["age"] == 30

    def test_pajek(self, tmp):
        g = ec.karate_club()
        ec.to_pajek(g, tmp / "g.net")
        G = nx.read_pajek(tmp / "g.net")
        assert nx_pairs(G) == pairs(g)

    def test_graph6_sparse6_bytes_match(self):
        rng = random.Random(0)
        for n in [0, 1, 2, 4, 8, 16, 62, 63, 64, 100, 300]:
            for p in [0.0, 0.1, 0.5, 1.0]:
                G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**6))
                g = ec.Graph()
                g.add_nodes([str(i) for i in range(n)])
                g.add_edges([(str(u), str(v)) for u, v in G.edges()])
                assert ec.to_graph6(g) == nx.to_graph6_bytes(G, header=False).decode().strip()
                assert ec.to_sparse6(g) == nx.to_sparse6_bytes(G, header=False).decode().strip()

    def test_sparse6_loops_and_parallel_edges(self):
        g = ec.Graph()
        g.add_nodes([str(i) for i in range(8)])
        g.add_edges([("0", "0"), ("0", "1"), ("0", "1"), ("3", "3"), ("2", "3")])
        G = nx.from_sparse6_bytes(ec.to_sparse6(g).encode())
        assert sorted(G.edges()) == [(0, 0), (0, 1), (0, 1), (2, 3), (3, 3)]
        h = ec.from_sparse6(ec.to_sparse6(g))
        assert pairs(h) == [("1", "1"), ("1", "2"), ("1", "2"), ("3", "4"), ("4", "4")]

    def test_sparse6_padding_special_case(self):
        # n = 2^k with small k: the 1s padding the last character must not read as an edge
        for n in (2, 4, 8, 16, 32):
            for edges in ([(0, 1)], [(0, 1), (0, 2)] if n > 2 else [(0, 0)]):
                g = ec.Graph()
                g.add_nodes([str(i) for i in range(n)])
                g.add_edges([(str(u), str(v)) for u, v in edges])
                G = nx.MultiGraph()
                G.add_nodes_from(range(n))
                G.add_edges_from(edges)
                text = ec.to_sparse6(g)
                assert text == nx.to_sparse6_bytes(G, header=False).decode().strip()
                assert ec.from_sparse6(text).count_edges() == len(edges)


# =============================================================================
# We read NetworkX's files
# =============================================================================


class TestReadingNetworkXFiles:
    graphs = {
        "karate": nx.karate_club_graph(),
        "lesmis": nx.les_miserables_graph(),
        "digraph": nx.gn_graph(30, seed=1),
        "multigraph": nx.MultiGraph([(0, 1), (0, 1), (1, 2)]),
    }

    @pytest.mark.parametrize("name", list(graphs))
    @pytest.mark.parametrize("fmt", ["graphml", "gexf", "gml", "node_link", "cytoscape", "pajek"])
    def test_structure(self, tmp, name, fmt):
        G = self.graphs[name]
        if fmt == "pajek" and G.is_multigraph():
            pytest.skip("write_pajek collapses parallel edges")
        path = tmp / f"g.{fmt}"
        if fmt == "graphml":
            nx.write_graphml(G, path)
            g = ec.from_graphml(path)
        elif fmt == "gexf":
            nx.write_gexf(G, path)
            g = ec.from_gexf(path)
        elif fmt == "gml":
            nx.write_gml(G, path, stringizer=str)
            g = ec.from_gml(path)
        elif fmt == "node_link":
            g = ec.from_node_link(nx.node_link_data(G, edges="edges"))
        elif fmt == "cytoscape":
            g = ec.from_cytoscape(nx.cytoscape_data(G))
        else:
            nx.write_pajek(G, path)
            g = ec.from_pajek(path)
        assert g.count_nodes() == G.number_of_nodes()
        assert g.is_directed() == G.is_directed()
        assert pairs(g) == nx_pairs(G)

    def test_karate_attributes(self, tmp):
        G = nx.karate_club_graph()
        nx.write_graphml(G, tmp / "k.graphml")
        nx.write_gexf(G, tmp / "k.gexf")
        nx.write_gml(G, tmp / "k.gml")
        for g in (
            ec.from_graphml(tmp / "k.graphml"),
            ec.from_gexf(tmp / "k.gexf"),
            ec.from_gml(tmp / "k.gml"),
            ec.from_node_link(nx.node_link_data(G, edges="edges")),
        ):
            assert g.get_node("0").attributes["club"] == "Mr. Hi"
            assert g.is_weighted()
            assert sorted(w for _, _, w, _ in edges_of(g)) == sorted(
                float(d["weight"]) for *_, d in G.edges(data=True)
            )
        assert ec.from_graphml(tmp / "k.graphml").get_graph_name() == "Zachary's Karate Club"
        assert ec.from_gml(tmp / "k.gml").get_graph_name() == "Zachary's Karate Club"

    def test_graph6_and_sparse6(self):
        G = nx.petersen_graph()
        g = ec.from_graph6(nx.to_graph6_bytes(G))
        assert pairs(g) == sorted(tuple(sorted((str(u + 1), str(v + 1)))) for u, v in G.edges())
        g = ec.from_sparse6(nx.to_sparse6_bytes(G))
        assert g.count_edges() == 15 and g.is_isomorphic(ec.from_graph6(nx.to_graph6_bytes(G)))

    def test_tuple_node_ids(self):
        G = nx.grid_2d_graph(2, 2)
        g = ec.from_node_link(json.loads(json.dumps(nx.node_link_data(G, edges="edges"))))
        assert sorted(lb for lb, _ in nodes_of(g)) == ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
        assert g.count_edges() == 4


# =============================================================================
# Labels
# =============================================================================


class TestLabels:
    GEXF_DUPES = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph defaultedgetype="undirected">
    <nodes>
      <node id="a" label="Same"/>
      <node id="b" label="Same"/>
      <node id="c" label="Other"/>
    </nodes>
    <edges>
      <edge id="0" source="a" target="b"/>
    </edges>
  </graph>
</gexf>"""

    def test_duplicate_labels_fall_back_to_ids(self, tmp):
        (tmp / "d.gexf").write_text(self.GEXF_DUPES)
        with pytest.warns(UserWarning, match="'Same' is used by more than one node"):
            g = ec.from_gexf(tmp / "d.gexf")
        assert nodes_of(g) == [
            ("a", {"name": "Same"}),
            ("b", {"name": "Same"}),
            ("c", {"name": "Other"}),
        ]

    def test_duplicate_labels_with_name_taken(self, tmp):
        data = {
            "nodes": [{"id": 1, "label": "x"}, {"id": 2}],
            "edges": [],
        }
        # node-link labels are the ids, so they're unique; Cytoscape names needn't be
        cy = {
            "elements": {
                "nodes": [
                    {"data": {"id": "1", "name": "Gene"}},
                    {"data": {"id": "2", "name": "Gene"}},
                ],
                "edges": [{"data": {"source": "1", "target": "2"}}],
            }
        }
        assert [lb for lb, _ in nodes_of(ec.from_node_link(data))] == ["1", "2"]
        with pytest.warns(UserWarning):
            g = ec.from_cytoscape(cy)
        assert nodes_of(g) == [("1", {"name": "Gene"}), ("2", {"name": "Gene"})]

    def test_duplicate_node_ids_raise(self, tmp):
        (tmp / "d.graphml").write_text(
            '<graphml><graph><node id="a"/><node id="a"/></graph></graphml>'
        )
        with pytest.raises(ValueError, match="'a' appears more than once"):
            ec.from_graphml(tmp / "d.graphml")

    def test_undeclared_edge_endpoints_are_created(self, tmp):
        (tmp / "e.gml").write_text('graph [ node [ id 1 label "one" ] edge [ source 1 target 9 ] ]')
        g = ec.from_gml(tmp / "e.gml")
        assert [lb for lb, _ in nodes_of(g)] == ["one", "9"]


# =============================================================================
# GraphML details
# =============================================================================


class TestGraphMLDetails:
    def test_typed_keys_and_defaults(self, tmp):
        (tmp / "t.graphml").write_text(
            """<?xml version="1.0"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="k0" for="node" attr.name="n" attr.type="int"/>
  <key id="k1" for="node" attr.name="f" attr.type="float"/>
  <key id="k2" for="node" attr.name="b" attr.type="boolean"><default>false</default></key>
  <key id="k3" for="all" attr.name="note" attr.type="string"/>
  <key id="k4" for="edge" attr.name="weight" attr.type="double"><default>1.0</default></key>
  <key id="k5" for="node" attr.name="label" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="a"><data key="k0">7</data><data key="k1">0.25</data><data key="k5">Alpha</data></node>
    <node id="b"><data key="k2">True</data><data key="k3">hello</data></node>
    <edge source="a" target="b"><data key="k3">edge note</data></edge>
    <edge source="b" target="a"><data key="k4">4</data></edge>
  </graph>
</graphml>"""
        )
        g = ec.from_graphml(tmp / "t.graphml")
        assert nodes_of(g) == [
            ("Alpha", {"n": 7, "f": 0.25, "b": False}),
            ("b", {"b": True, "note": "hello"}),
        ]
        assert edges_of(g) == [("Alpha", "b", 1.0, {"note": "edge note"}), ("b", "Alpha", 4.0, {})]
        assert g.is_directed() and g.is_weighted() and g.get_graph_name() == "G"

    def test_structured_data_is_skipped(self, tmp):
        (tmp / "y.graphml").write_text(
            """<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
  xmlns:y="http://www.yworks.com/xml/graphml">
  <key id="d0" for="node" yfiles.type="nodegraphics"/>
  <graph edgedefault="undirected">
    <node id="n0"><data key="d0"><y:ShapeNode><y:NodeLabel>x</y:NodeLabel></y:ShapeNode></data></node>
  </graph>
</graphml>"""
        )
        assert nodes_of(ec.from_graphml(tmp / "y.graphml")) == [("n0", {})]

    def test_json_values_are_marked(self, tmp):
        g = ec.Graph().add_node(label="a", xs=[1, 2], mixed=1).add_node(label="b", mixed="one")
        ec.to_graphml(g, tmp / "j.graphml")
        text = (tmp / "j.graphml").read_text()
        assert 'edgecase:json="true"' in text
        assert nodes_of(ec.from_graphml(tmp / "j.graphml")) == nodes_of(g)


# =============================================================================
# GML details
# =============================================================================


class TestGMLDetails:
    def test_values_entities_lists_and_blocks(self, tmp):
        (tmp / "t.gml").write_text(
            """Creator "someone"
graph [
  directed 0
  label "Net"
  node [ id 0 label "caf&#233; &amp; &quot;bar&quot;" x 1 x 2 graphics [ w 1.5 type "oval" ] ]
  node [ id 1 big INF tiny -INF odd NAN ]
  edge [ source 0 target 1 value 2.5 ]
]"""
        )
        g = ec.from_gml(tmp / "t.gml")
        (label, attrs), (_, attrs1) = nodes_of(g)
        assert label == 'café & "bar"'
        assert attrs == {"x": [1, 2], "graphics": {"w": 1.5, "type": "oval"}}
        assert attrs1["big"] == math.inf and attrs1["tiny"] == -math.inf
        assert math.isnan(attrs1["odd"])
        assert edges_of(g)[0][2] == 2.5
        assert g.get_graph_name() == "Net"

    def test_value_is_an_attribute_when_weight_is_present(self, tmp):
        (tmp / "t.gml").write_text(
            "graph [ node [ id 0 ] node [ id 1 ] edge [ source 0 target 1 weight 3 value 9 ] ]"
        )
        assert edges_of(ec.from_gml(tmp / "t.gml")) == [("0", "1", 3.0, {"value": 9})]

    @pytest.mark.parametrize("bad", ["two words", "1st", "id", "label"])
    def test_bad_node_attribute_names(self, tmp, bad):
        g = ec.Graph()
        g._graph.add_node(rust.Node(id=0, label="a", attributes={bad: 1}))
        with pytest.raises(ValueError, match="can't be written to GML"):
            ec.to_gml(g, tmp / "bad.gml")

    def test_bad_edge_attribute_names(self, tmp):
        g = ec.Graph().add_nodes(["a", "b"])
        g._graph.add_edge(rust.Edge(id=0, source=1, target=2, attributes={"source": 1}))
        with pytest.raises(ValueError, match="source, target are reserved"):
            ec.to_gml(g, tmp / "bad.gml")

    def test_float_specials_written(self, tmp):
        g = ec.Graph().add_node(label="a", hi=math.inf, lo=-math.inf, nan=math.nan)
        ec.to_gml(g, tmp / "s.gml")
        text = (tmp / "s.gml").read_text()
        assert "hi INF" in text and "lo -INF" in text and "nan NAN" in text
        G = nx.read_gml(tmp / "s.gml")
        assert G.nodes["a"]["hi"] == math.inf

    def test_parallel_edges_marked_multigraph(self, tmp):
        g = ec.Graph().add_nodes(["a", "b"])
        g.add_edge("a", "b").add_edge("a", "b")
        ec.to_gml(g, tmp / "m.gml")
        assert "multigraph 1" in (tmp / "m.gml").read_text()
        assert nx.read_gml(tmp / "m.gml").number_of_edges() == 2

    def test_unbalanced_brackets(self, tmp):
        (tmp / "u.gml").write_text("graph [ node [ id 0 ]")
        with pytest.raises(ValueError, match="Unclosed"):
            ec.from_gml(tmp / "u.gml")


# =============================================================================
# GEXF details
# =============================================================================


class TestGEXFDetails:
    GEPHI = """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz"
      version="1.2">
  <graph defaultedgetype="undirected" mode="static">
    <attributes class="node" mode="static">
      <attribute id="modularity_class" title="Modularity Class" type="integer"/>
      <attribute id="1" title="flag" type="boolean"><default>true</default></attribute>
    </attributes>
    <attributes class="edge">
      <attribute id="0" title="since" type="long"/>
    </attributes>
    <nodes>
      <node id="0" label="Alice">
        <attvalues><attvalue for="modularity_class" value="3"/></attvalues>
        <viz:size value="12.5"/>
        <viz:position x="10.0" y="-20.5" z="0.0"/>
        <viz:color r="255" g="0" b="16"/>
      </node>
      <node id="1" label="Bob">
        <attvalues><attvalue for="1" value="false"/></attvalues>
        <viz:color r="0" g="0" b="0" a="0.5"/>
      </node>
    </nodes>
    <edges>
      <edge id="0" source="0" target="1" weight="2.0" label="ties" type="directed">
        <attvalues><attvalue for="0" value="1999"/></attvalues>
      </edge>
    </edges>
  </graph>
</gexf>"""

    def test_gephi_file(self, tmp):
        (tmp / "g.gexf").write_text(self.GEPHI)
        g = ec.from_gexf(tmp / "g.gexf")
        assert nodes_of(g) == [
            (
                "Alice",
                {
                    "Modularity Class": 3,
                    "flag": True,
                    "size": 12.5,
                    "pos": [10.0, -20.5],
                    "color": "#ff0010",
                },
            ),
            ("Bob", {"flag": False, "color": "#00000080"}),
        ]
        assert edges_of(g) == [("Alice", "Bob", 2.0, {"since": 1999, "label": "ties"})]
        # one directed edge makes the graph directed
        assert g.is_directed() and g.is_weighted()

    def test_hex_color_in_1_3(self, tmp):
        (tmp / "h.gexf").write_text(
            '<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">'
            '<graph><nodes><node id="0"><viz:color hex="#AABBCC"/></node></nodes></graph></gexf>'
        )
        assert nodes_of(ec.from_gexf(tmp / "h.gexf")) == [("0", {"color": "#aabbcc"})]

    def test_viz_written_from_attributes(self, tmp):
        g = ec.Graph().add_node(label="a", color="#102030", size=4, pos=[1.5, -2.0])
        ec.to_gexf(g, tmp / "v.gexf")
        G = nx.read_gexf(tmp / "v.gexf")
        viz = G.nodes["0" if "0" in G else "1"]["viz"]
        assert viz["color"] == {"r": 16, "g": 32, "b": 48, "a": 1.0}
        assert viz["size"] == 4.0
        assert viz["position"] == {"x": 1.5, "y": -2.0, "z": 0.0}
        assert nodes_of(ec.from_gexf(tmp / "v.gexf")) == nodes_of(g)

    def test_not_gexf(self, tmp):
        (tmp / "x.gexf").write_text("<graphml/>")
        with pytest.raises(ValueError, match="Not a GEXF file"):
            ec.from_gexf(tmp / "x.gexf")

    def test_bad_version(self, tmp):
        with pytest.raises(ValueError, match="version must be one of"):
            ec.to_gexf(ec.Graph(), tmp / "x.gexf", version="1.1draft")


# =============================================================================
# JSON details
# =============================================================================


class TestJSONDetails:
    def test_node_link_shape(self):
        g = rich_graph(directed=True)
        data = ec.to_node_link(g)
        assert data["directed"] is True and data["multigraph"] is True
        assert data["graph"] == {"name": "Rich graph"}
        assert data["nodes"][0]["id"] == "A"
        assert data["edges"][0] == {"source": "A", "target": "B", "weight": 2.5, "rel": "knows"}
        json.dumps(data)

    def test_node_link_drops_keys(self):
        data = {
            "directed": False,
            "multigraph": True,
            "graph": {"name": "m"},
            "nodes": [{"id": 0}, {"id": 1}],
            "links": [{"source": 0, "target": 1, "key": 0}, {"source": 0, "target": 1, "key": 1}],
        }
        g = ec.from_node_link(data)
        assert edges_of(g) == [("0", "1", None, {}), ("0", "1", None, {})]
        assert g.get_graph_name() == "m" and not g.is_weighted()

    def test_node_link_errors(self):
        with pytest.raises(ValueError, match="'nodes' list"):
            ec.from_node_link({"edges": []})
        with pytest.raises(ValueError, match="needs an 'id'"):
            ec.from_node_link({"nodes": [{"x": 1}]})
        with pytest.raises(ValueError, match="'edges' or 'links'"):
            ec.to_node_link(ec.Graph(), edges="arcs")

    def test_cytoscape_shape(self):
        g = ec.Graph(graph_name="cy").add_path_graph(2).layout_circular(scale=2)
        data = ec.to_cytoscape(g)
        assert data["data"] == {"name": "cy"} and data["multigraph"] is False
        node = data["elements"]["nodes"][1]
        assert node["data"] == {"id": "2", "value": "2", "name": "2"}
        x, y = g.get_node(2).attributes["pos"]
        assert node["position"] == {"x": x, "y": -y}
        assert data["elements"]["edges"] == [{"data": {"id": "e0", "source": "1", "target": "2"}}]

    def test_cytoscape_edge_ids_avoid_node_ids(self):
        g = ec.Graph().add_nodes(["e0", "x"])
        g.add_edge("e0", "x")
        edge = ec.to_cytoscape(g)["elements"]["edges"][0]["data"]
        assert edge["id"] not in ("e0", "x")

    def test_cytoscape_flat_elements_and_desktop_fields(self):
        data = {
            "data": {"name": "Session", "SUID": 52},
            "elements": [
                {
                    "group": "nodes",
                    "data": {"id": "101", "name": "TP53", "SUID": 101},
                    "position": {"x": 3.0, "y": 4.0},
                },
                {"data": {"id": "102", "name": "MDM2", "SUID": 102}},
                {
                    "group": "edges",
                    "data": {"id": "103", "source": "101", "target": "102", "interaction": "pp"},
                },
            ],
        }
        g = ec.from_cytoscape(data)
        assert nodes_of(g) == [
            ("TP53", {"SUID": 101, "pos": [3.0, -4.0]}),
            ("MDM2", {"SUID": 102}),
        ]
        assert edges_of(g) == [("TP53", "MDM2", None, {"interaction": "pp"})]
        assert g.get_graph_name() == "Session" and not g.is_directed()

    def test_cytoscape_errors(self):
        with pytest.raises(ValueError, match="'elements'"):
            ec.from_cytoscape({"elements": 3})
        with pytest.raises(ValueError, match="with an 'id'"):
            ec.from_cytoscape({"elements": {"nodes": [{"data": {}}]}})


# =============================================================================
# Pajek details
# =============================================================================


class TestPajekDetails:
    def test_reading(self, tmp):
        (tmp / "p.net").write_text(
            """% a comment
*Network Test net
*Vertices 5 2
1 "Alice Smith" 0.1 0.2 0.3 box ic Red size 2
2 Bob
4 "Dee" color "light blue"
*Arcs :1 "advice"
1 2 3.5
2 1 col 7
*Edges
3 4
*Arcslist
5 1 2
"""
        )
        g = ec.from_pajek(tmp / "p.net")
        assert g.get_graph_name() == "Test net"
        assert g.is_directed() and g.is_weighted()
        assert nodes_of(g) == [
            ("Alice Smith", {"pos": [0.1, 0.2], "shape": "box", "ic": "Red", "size": 2, "part": 0}),
            ("Bob", {"part": 0}),
            ("3", {"part": 1}),
            ("Dee", {"color": "light blue", "part": 1}),
            ("5", {"part": 1}),
        ]
        assert edges_of(g) == [
            ("Alice Smith", "Bob", 3.5, {"rel": "advice"}),
            ("Bob", "Alice Smith", None, {"rel": "advice", "col": 7}),
            # an undirected edge in a directed network becomes two arcs
            ("3", "Dee", None, {}),
            ("Dee", "3", None, {}),
            ("5", "Alice Smith", None, {}),
            ("5", "Bob", None, {}),
        ]

    def test_edges_only_is_undirected(self, tmp):
        (tmp / "u.net").write_text("*vertices 3\n*edges\n1 2\n2 3\n*edgeslist\n3 1\n")
        g = ec.from_pajek(tmp / "u.net")
        assert not g.is_directed() and not g.is_weighted()
        assert pairs(g) == [("1", "2"), ("1", "3"), ("2", "3")]

    def test_only_first_network(self, tmp):
        (tmp / "two.paj").write_text(
            "*Network one\n*Vertices 2\n*Edges\n1 2\n*Network two\n*Vertices 3\n*Edges\n1 3\n"
        )
        g = ec.from_pajek(tmp / "two.paj")
        assert (g.get_graph_name(), g.count_nodes(), g.count_edges()) == ("one", 2, 1)

    def test_matrix_unsupported(self, tmp):
        (tmp / "m.net").write_text("*Vertices 2\n*Matrix\n0 1\n1 0\n")
        with pytest.raises(ValueError, match="Matrix"):
            ec.from_pajek(tmp / "m.net")

    def test_bad_line(self, tmp):
        (tmp / "b.net").write_text("*Vertices 2\n*Edges\n1 x\n")
        with pytest.raises(ValueError, match="Line 3"):
            ec.from_pajek(tmp / "b.net")

    def test_writing(self, tmp):
        g = ec.Graph(directed=True, weighted=True, graph_name="My net")
        g.add_node(label='Say "hi"', pos=[1.0, 2.0], group="a b", n=3, flag=True, xs=[1])
        g.add_node(label="Two")
        g.add_edge(1, 2, weight=0.5, kind="x")
        with pytest.warns(UserWarning, match="weren't written: xs"):
            ec.to_pajek(g, tmp / "w.net")
        assert (tmp / "w.net").read_text().splitlines() == [
            "*Network My net",
            "*Vertices 2",
            '1 "Say \'hi\'" 1.0 2.0 group "a b" n 3 flag 1',
            '2 "Two"',
            "*Arcs",
            "1 2 0.5 kind x",
        ]
        h = ec.from_pajek(tmp / "w.net")
        assert nodes_of(h) == [
            ("Say 'hi'", {"pos": [1.0, 2.0], "group": "a b", "n": 3, "flag": 1}),
            ("Two", {}),
        ]
        assert edges_of(h) == [("Say 'hi'", "Two", 0.5, {"kind": "x"})]


# =============================================================================
# graph6 / sparse6 details
# =============================================================================


class TestSixBitFormats:
    def test_header_and_file(self, tmp):
        g = ec.Graph().add_cycle_graph(5)
        assert ec.to_graph6(g, header=True) == ">>graph6<<Dhc"
        assert ec.to_sparse6(g, header=True).startswith(">>sparse6<<:")
        ec.to_graph6(g, tmp / "c.g6")
        assert (tmp / "c.g6").read_text() == "Dhc\n"
        for source in (tmp / "c.g6", str(tmp / "c.g6"), b">>graph6<<Dhc\n", "Dhc"):
            assert ec.from_graph6(source).is_isomorphic(g)

    def test_several_graphs(self, tmp):
        lines = [nx.to_graph6_bytes(G, header=False) for G in nx.graph_atlas_g()[1:20]]
        (tmp / "atlas.g6").write_bytes(b"".join(lines))
        graphs = ec.from_graph6(tmp / "atlas.g6")
        assert isinstance(graphs, list) and len(graphs) == 19
        for g, G in zip(graphs, nx.graph_atlas_g()[1:20]):
            assert (g.count_nodes(), g.count_edges()) == (G.number_of_nodes(), G.number_of_edges())

    def test_labels(self):
        g = ec.from_graph6("Dhc")
        assert [lb for lb, _ in nodes_of(g)] == ["1", "2", "3", "4", "5"]
        assert not g.is_directed() and not g.is_weighted()

    def test_large_node_counts(self):
        for n in (63, 258047, 258048):
            g = ec.Graph()
            g.add_nodes([str(i) for i in range(n)])
            g.add_edge("0", str(n - 1))
            text = ec.to_sparse6(g)
            assert text.encode() + b"\n" == nx.to_sparse6_bytes(
                nx.Graph([(0, n - 1)]) if n == 2 else _sparse_nx(n), header=False
            )
            h = ec.from_sparse6(text)
            assert (h.count_nodes(), h.count_edges()) == (n, 1)

    def test_graph6_restrictions(self):
        g = ec.Graph(directed=True).add_path_graph(3)
        with pytest.raises(ValueError, match="undirected graphs only"):
            ec.to_graph6(g)
        with pytest.raises(ValueError, match="undirected graphs only"):
            ec.to_sparse6(g)
        loop = ec.Graph().add_nodes(["a"])
        loop.add_edge("a", "a")
        with pytest.raises(ValueError, match="self-loops"):
            ec.to_graph6(loop)
        multi = ec.Graph().add_nodes(["a", "b"])
        multi.add_edge("a", "b").add_edge("b", "a")
        with pytest.raises(ValueError, match="parallel edges"):
            ec.to_graph6(multi)
        assert ec.from_sparse6(ec.to_sparse6(multi)).count_edges() == 2

    def test_bad_input(self):
        with pytest.raises(ValueError, match="needs 6 adjacency characters, not 1"):
            ec.from_graph6("H?")
        with pytest.raises(ValueError, match="sparse6 or digraph6"):
            ec.from_graph6(":Dgc")
        with pytest.raises(ValueError, match="starts with ':'"):
            ec.from_sparse6("Dhc")
        with pytest.raises(ValueError, match="Incremental"):
            ec.from_sparse6(";Dgc")
        with pytest.raises(ValueError, match="Not graph6/sparse6"):
            ec.from_graph6("D h")
        with pytest.raises(ValueError, match="No graph6/sparse6 data"):
            ec.from_graph6("\n\n")


def _sparse_nx(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edge(0, n - 1)
    return G


# =============================================================================
# TSV details
# =============================================================================


class TestTSVDetails:
    def test_missing_weight_written_empty(self, tmp):
        g = ec.Graph(weighted=True).add_nodes(["a", "b", "c"])
        g.add_edge("a", "b", weight=2.0).add_edge("b", "c")
        ec.to_tsv_edge_list(g, tmp / "w.tsv")
        assert (tmp / "w.tsv").read_text() == "source\ttarget\tweight\na\tb\t2.0\nb\tc\t\n"
        h = ec.from_tsv_edge_list(tmp / "w.tsv", has_header=True, weight_col="weight")
        assert [w for _, _, w, _ in edges_of(h)] == [2.0, None]

    def test_column_name_needs_header(self, tmp):
        (tmp / "n.tsv").write_text("a\tb\n")
        with pytest.raises(ValueError, match="has_header=True"):
            ec.from_tsv_edge_list(tmp / "n.tsv", source_col="from")

    def test_unknown_column_name(self, tmp):
        (tmp / "n.tsv").write_text("x\ty\na\tb\n")
        with pytest.raises(ValueError, match="isn't in the header row"):
            ec.from_tsv_edge_list(tmp / "n.tsv", source_col="from", has_header=True)


def test_exports():
    import edgecase.io as io

    for name in io.__all__:
        assert getattr(ec, name) is getattr(io, name)
        assert name in ec.__all__


def test_no_warnings_on_clean_round_trip(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = ec.karate_club()
        ec.to_gexf(g, tmp_path / "k.gexf")
        ec.from_gexf(tmp_path / "k.gexf")
