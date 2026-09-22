"""Tests for file format I/O: GraphML, GML, SIF, TSV edge list."""

import os
import tempfile

import pytest
import edgecase as ec
from edgecase.io import (
    from_gml,
    from_graphml,
    from_sif,
    from_tsv_edge_list,
    to_gml,
    to_graphml,
    to_sif,
    to_tsv_edge_list,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d


@pytest.fixture
def weighted_triangle():
    g = ec.Graph(weighted=True)
    g.add_node(label="A").add_node(label="B").add_node(label="C")
    g.add_edge("A", "B", weight=1.5)
    g.add_edge("B", "C", weight=2.0)
    g.add_edge("C", "A", weight=0.5)
    return g


@pytest.fixture
def unweighted_triangle():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
    )


@pytest.fixture
def directed_graph():
    return (
        ec.Graph(directed=True)
        .add_node(label="X")
        .add_node(label="Y")
        .add_node(label="Z")
        .add_edge("X", "Y")
        .add_edge("Y", "Z")
    )


@pytest.fixture
def isolated_node_graph():
    g = ec.Graph()
    g.add_node(label="Connected1").add_node(label="Connected2")
    g.add_node(label="Isolated")
    g.add_edge("Connected1", "Connected2")
    return g


@pytest.fixture
def empty_graph():
    return ec.Graph()


@pytest.fixture
def single_node_graph():
    return ec.Graph().add_node(label="Solo")


@pytest.fixture
def large_graph():
    g = ec.Graph(weighted=True)
    for i in range(50):
        g.add_node(label=f"N{i}")
    for i in range(49):
        g.add_edge(f"N{i}", f"N{i + 1}", weight=float(i + 1))
    g.add_edge("N49", "N0", weight=50.0)
    return g


# =============================================================================
# GraphML — from_graphml
# =============================================================================


class TestFromGraphML:
    def test_roundtrip_unweighted(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(unweighted_triangle, path)
        g = from_graphml(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert not g.is_weighted()

    def test_roundtrip_weighted(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(weighted_triangle, path)
        g = from_graphml(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert g.is_weighted()

    def test_roundtrip_directed(self, tmp_dir, directed_graph):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(directed_graph, path)
        g = from_graphml(path)
        assert g.is_directed()
        assert g.count_nodes() == 3
        assert g.count_edges() == 2

    def test_override_directed(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(unweighted_triangle, path)
        g = from_graphml(path, directed=True)
        assert g.is_directed()

    def test_preserves_labels(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(weighted_triangle, path)
        g = from_graphml(path)
        assert g.has_node("A")
        assert g.has_node("B")
        assert g.has_node("C")

    def test_preserves_weights(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(weighted_triangle, path)
        g = from_graphml(path)
        edf = g.get_edge_df()
        weights = sorted(edf["weight"].to_list())
        assert weights == pytest.approx([0.5, 1.5, 2.0])

    def test_empty_graph(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(empty_graph, path)
        g = from_graphml(path)
        assert g.count_nodes() == 0
        assert g.count_edges() == 0

    def test_single_node(self, tmp_dir, single_node_graph):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(single_node_graph, path)
        g = from_graphml(path)
        assert g.count_nodes() == 1
        assert g.count_edges() == 0
        assert g.has_node("Solo")

    def test_large_graph_roundtrip(self, tmp_dir, large_graph):
        path = os.path.join(tmp_dir, "test.graphml")
        to_graphml(large_graph, path)
        g = from_graphml(path)
        assert g.count_nodes() == 50
        assert g.count_edges() == 50

    def test_hand_crafted_graphml(self, tmp_dir):
        path = os.path.join(tmp_dir, "hand.graphml")
        with open(path, "w") as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphstruct.org/xmlns">
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  <graph edgedefault="undirected">
    <node id="n0"/>
    <node id="n1"/>
    <node id="n2"/>
    <edge source="n0" target="n1">
      <data key="weight">3.14</data>
    </edge>
    <edge source="n1" target="n2"/>
  </graph>
</graphml>""")
        g = from_graphml(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        assert g.is_weighted()
        assert not g.is_directed()

    def test_no_graph_element_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.graphml")
        with open(path, "w") as f:
            f.write('<?xml version="1.0"?><graphml></graphml>')
        with pytest.raises(ValueError, match="No <graph> element"):
            from_graphml(path)

    def test_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(Exception):
            from_graphml(os.path.join(tmp_dir, "nonexistent.graphml"))

    def test_graphml_no_namespace(self, tmp_dir):
        path = os.path.join(tmp_dir, "nons.graphml")
        with open(path, "w") as f:
            f.write("""<?xml version="1.0"?>
<graphml>
  <graph edgedefault="directed">
    <node id="a"/>
    <node id="b"/>
    <edge source="a" target="b"/>
  </graph>
</graphml>""")
        g = from_graphml(path)
        assert g.is_directed()
        assert g.count_nodes() == 2
        assert g.count_edges() == 1


# =============================================================================
# GraphML — to_graphml
# =============================================================================


class TestToGraphML:
    def test_creates_file(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "out.graphml")
        to_graphml(unweighted_triangle, path)
        assert os.path.exists(path)

    def test_file_is_valid_xml(self, tmp_dir, unweighted_triangle):
        import xml.etree.ElementTree as ET

        path = os.path.join(tmp_dir, "out.graphml")
        to_graphml(unweighted_triangle, path)
        tree = ET.parse(path)
        assert tree.getroot() is not None

    def test_empty_graph_writes(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "empty.graphml")
        to_graphml(empty_graph, path)
        assert os.path.exists(path)
        g = from_graphml(path)
        assert g.count_nodes() == 0

    def test_directed_attribute(self, tmp_dir, directed_graph):
        import xml.etree.ElementTree as ET

        path = os.path.join(tmp_dir, "dir.graphml")
        to_graphml(directed_graph, path)
        tree = ET.parse(path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        graph_elem = root.find(f".//{ns}graph")
        assert graph_elem.get("edgedefault") == "directed"


# =============================================================================
# GML — from_gml
# =============================================================================


class TestFromGML:
    def test_roundtrip_unweighted(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(unweighted_triangle, path)
        g = from_gml(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert not g.is_weighted()

    def test_roundtrip_weighted(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(weighted_triangle, path)
        g = from_gml(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert g.is_weighted()

    def test_roundtrip_directed(self, tmp_dir, directed_graph):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(directed_graph, path)
        g = from_gml(path)
        assert g.is_directed()
        assert g.count_edges() == 2

    def test_preserves_labels(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(weighted_triangle, path)
        g = from_gml(path)
        assert g.has_node("A")
        assert g.has_node("B")
        assert g.has_node("C")

    def test_preserves_weights(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(weighted_triangle, path)
        g = from_gml(path)
        edf = g.get_edge_df()
        weights = sorted(edf["weight"].to_list())
        assert weights == pytest.approx([0.5, 1.5, 2.0])

    def test_override_directed(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(unweighted_triangle, path)
        g = from_gml(path, directed=True)
        assert g.is_directed()

    def test_empty_graph(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(empty_graph, path)
        g = from_gml(path)
        assert g.count_nodes() == 0

    def test_single_node(self, tmp_dir, single_node_graph):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(single_node_graph, path)
        g = from_gml(path)
        assert g.count_nodes() == 1
        assert g.has_node("Solo")

    def test_large_graph_roundtrip(self, tmp_dir, large_graph):
        path = os.path.join(tmp_dir, "test.gml")
        to_gml(large_graph, path)
        g = from_gml(path)
        assert g.count_nodes() == 50
        assert g.count_edges() == 50

    def test_hand_crafted_gml(self, tmp_dir):
        path = os.path.join(tmp_dir, "hand.gml")
        with open(path, "w") as f:
            f.write("""graph [
  directed 0
  node [
    id 1
    label "Alpha"
  ]
  node [
    id 2
    label "Beta"
  ]
  edge [
    source 1
    target 2
    value 9.5
  ]
]
""")
        g = from_gml(path)
        assert g.count_nodes() == 2
        assert g.count_edges() == 1
        assert g.is_weighted()
        assert g.has_node("Alpha")
        assert g.has_node("Beta")

    def test_gml_with_comments(self, tmp_dir):
        path = os.path.join(tmp_dir, "comments.gml")
        with open(path, "w") as f:
            f.write("""# This is a comment
graph [
  directed 1
  # Another comment
  node [
    id 0
    label "A"
  ]
  node [
    id 1
    label "B"
  ]
  edge [
    source 0
    target 1
  ]
]
""")
        g = from_gml(path)
        assert g.is_directed()
        assert g.count_nodes() == 2
        assert g.count_edges() == 1

    def test_no_graph_block_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.gml")
        with open(path, "w") as f:
            f.write("node [ id 0 ]")
        with pytest.raises(ValueError, match="No 'graph' block"):
            from_gml(path)

    def test_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(Exception):
            from_gml(os.path.join(tmp_dir, "nonexistent.gml"))

    def test_node_without_label_uses_id(self, tmp_dir):
        path = os.path.join(tmp_dir, "nolabel.gml")
        with open(path, "w") as f:
            f.write("""graph [
  directed 0
  node [ id 10 ]
  node [ id 20 ]
  edge [ source 10 target 20 ]
]
""")
        g = from_gml(path)
        assert g.count_nodes() == 2
        assert g.count_edges() == 1
        assert g.has_node("10")
        assert g.has_node("20")


# =============================================================================
# GML — to_gml
# =============================================================================


class TestToGML:
    def test_creates_file(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "out.gml")
        to_gml(unweighted_triangle, path)
        assert os.path.exists(path)

    def test_directed_field(self, tmp_dir, directed_graph):
        path = os.path.join(tmp_dir, "dir.gml")
        to_gml(directed_graph, path)
        with open(path) as f:
            content = f.read()
        assert "directed 1" in content

    def test_undirected_field(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "undir.gml")
        to_gml(unweighted_triangle, path)
        with open(path) as f:
            content = f.read()
        assert "directed 0" in content

    def test_empty_graph_writes(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "empty.gml")
        to_gml(empty_graph, path)
        assert os.path.exists(path)


# =============================================================================
# SIF — from_sif
# =============================================================================


class TestFromSIF:
    def test_roundtrip_basic(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.sif")
        to_sif(unweighted_triangle, path)
        g = from_sif(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3

    def test_preserves_labels(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.sif")
        to_sif(unweighted_triangle, path)
        g = from_sif(path)
        assert g.has_node("A")
        assert g.has_node("B")
        assert g.has_node("C")

    def test_isolated_node(self, tmp_dir, isolated_node_graph):
        path = os.path.join(tmp_dir, "test.sif")
        to_sif(isolated_node_graph, path)
        g = from_sif(path)
        assert g.has_node("Isolated")
        assert g.count_nodes() == 3

    def test_directed(self, tmp_dir, directed_graph):
        path = os.path.join(tmp_dir, "test.sif")
        to_sif(directed_graph, path)
        g = from_sif(path, directed=True)
        assert g.is_directed()

    def test_hand_crafted_sif(self, tmp_dir):
        path = os.path.join(tmp_dir, "hand.sif")
        with open(path, "w") as f:
            f.write("BRCA1\tpp\tTP53\tATM\n")
            f.write("TP53\tpp\tMDM2\n")
            f.write("CDK2\n")
        g = from_sif(path)
        assert g.count_nodes() == 5
        assert g.has_node("CDK2")
        assert g.count_edges() == 3

    def test_multiple_targets_per_line(self, tmp_dir):
        path = os.path.join(tmp_dir, "multi.sif")
        with open(path, "w") as f:
            f.write("Hub\tinteracts\tA\tB\tC\tD\n")
        g = from_sif(path)
        assert g.count_nodes() == 5
        assert g.count_edges() == 4

    def test_comment_lines_skipped(self, tmp_dir):
        path = os.path.join(tmp_dir, "comments.sif")
        with open(path, "w") as f:
            f.write("# This is a comment\n")
            f.write("A\tpp\tB\n")
            f.write("# Another comment\n")
            f.write("B\tpp\tC\n")
        g = from_sif(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2

    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.sif")
        with open(path, "w") as f:
            f.write("")
        g = from_sif(path)
        assert g.count_nodes() == 0
        assert g.count_edges() == 0

    def test_blank_lines_skipped(self, tmp_dir):
        path = os.path.join(tmp_dir, "blanks.sif")
        with open(path, "w") as f:
            f.write("\n\nA\tpp\tB\n\n\n")
        g = from_sif(path)
        assert g.count_nodes() == 2
        assert g.count_edges() == 1

    def test_space_delimited_fallback(self, tmp_dir):
        path = os.path.join(tmp_dir, "spaces.sif")
        with open(path, "w") as f:
            f.write("Gene1 pp Gene2\n")
            f.write("Gene2 pp Gene3\n")
        g = from_sif(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2

    def test_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(Exception):
            from_sif(os.path.join(tmp_dir, "nonexistent.sif"))

    def test_large_graph_roundtrip(self, tmp_dir, large_graph):
        path = os.path.join(tmp_dir, "test.sif")
        to_sif(large_graph, path)
        g = from_sif(path)
        assert g.count_nodes() == 50
        assert g.count_edges() == 50


# =============================================================================
# SIF — to_sif
# =============================================================================


class TestToSIF:
    def test_creates_file(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "out.sif")
        to_sif(unweighted_triangle, path)
        assert os.path.exists(path)

    def test_custom_interaction_type(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "custom.sif")
        to_sif(unweighted_triangle, path, interaction_type="pp")
        with open(path) as f:
            content = f.read()
        assert "pp" in content
        assert "interacts_with" not in content

    def test_empty_graph_writes(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "empty.sif")
        to_sif(empty_graph, path)
        assert os.path.exists(path)

    def test_isolated_node_appears(self, tmp_dir, isolated_node_graph):
        path = os.path.join(tmp_dir, "iso.sif")
        to_sif(isolated_node_graph, path)
        with open(path) as f:
            content = f.read()
        assert "Isolated" in content


# =============================================================================
# TSV Edge List — from_tsv_edge_list
# =============================================================================


class TestFromTSVEdgeList:
    def test_roundtrip_unweighted(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "test.tsv")
        to_tsv_edge_list(unweighted_triangle, path, weight=False)
        g = from_tsv_edge_list(path, has_header=True)
        assert g.count_nodes() == 3
        assert g.count_edges() == 3

    def test_roundtrip_weighted(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.tsv")
        to_tsv_edge_list(weighted_triangle, path)
        g = from_tsv_edge_list(path, has_header=True, weight_col="weight")
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert g.is_weighted()

    def test_preserves_labels(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.tsv")
        to_tsv_edge_list(weighted_triangle, path)
        g = from_tsv_edge_list(path, has_header=True, weight_col="weight")
        assert g.has_node("A")
        assert g.has_node("B")
        assert g.has_node("C")

    def test_preserves_weights(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "test.tsv")
        to_tsv_edge_list(weighted_triangle, path)
        g = from_tsv_edge_list(path, has_header=True, weight_col="weight")
        edf = g.get_edge_df()
        weights = sorted(edf["weight"].to_list())
        assert weights == pytest.approx([0.5, 1.5, 2.0])

    def test_column_indices(self, tmp_dir):
        path = os.path.join(tmp_dir, "idx.tsv")
        with open(path, "w") as f:
            f.write("A\tB\t1.5\n")
            f.write("B\tC\t2.0\n")
        g = from_tsv_edge_list(path, source_col=0, target_col=1, weight_col=2)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        assert g.is_weighted()

    def test_column_names_with_header(self, tmp_dir):
        path = os.path.join(tmp_dir, "named.tsv")
        with open(path, "w") as f:
            f.write("gene1\tgene2\tscore\n")
            f.write("BRCA1\tTP53\t0.9\n")
            f.write("TP53\tATM\t0.8\n")
        g = from_tsv_edge_list(
            path, source_col="gene1", target_col="gene2", weight_col="score", has_header=True
        )
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        assert g.has_node("BRCA1")

    def test_comment_lines_skipped(self, tmp_dir):
        path = os.path.join(tmp_dir, "comments.tsv")
        with open(path, "w") as f:
            f.write("# This is a header comment\n")
            f.write("# Another comment\n")
            f.write("A\tB\n")
            f.write("B\tC\n")
        g = from_tsv_edge_list(path)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2

    def test_skip_rows(self, tmp_dir):
        path = os.path.join(tmp_dir, "skip.tsv")
        with open(path, "w") as f:
            f.write("metadata line 1\n")
            f.write("metadata line 2\n")
            f.write("A\tB\n")
            f.write("B\tC\n")
        g = from_tsv_edge_list(path, skip_rows=2)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2

    def test_custom_delimiter(self, tmp_dir):
        path = os.path.join(tmp_dir, "csv.txt")
        with open(path, "w") as f:
            f.write("A,B,1.0\n")
            f.write("B,C,2.0\n")
        g = from_tsv_edge_list(path, delimiter=",", weight_col=2)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        assert g.is_weighted()

    def test_directed(self, tmp_dir):
        path = os.path.join(tmp_dir, "dir.tsv")
        with open(path, "w") as f:
            f.write("A\tB\n")
            f.write("B\tC\n")
        g = from_tsv_edge_list(path, directed=True)
        assert g.is_directed()

    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.tsv")
        with open(path, "w") as f:
            f.write("")
        g = from_tsv_edge_list(path)
        assert g.count_nodes() == 0
        assert g.count_edges() == 0

    def test_blank_and_comment_only(self, tmp_dir):
        path = os.path.join(tmp_dir, "noop.tsv")
        with open(path, "w") as f:
            f.write("# comment\n\n# another\n\n")
        g = from_tsv_edge_list(path)
        assert g.count_nodes() == 0
        assert g.count_edges() == 0

    def test_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(Exception):
            from_tsv_edge_list(os.path.join(tmp_dir, "nonexistent.tsv"))

    def test_malformed_lines_skipped(self, tmp_dir):
        path = os.path.join(tmp_dir, "malformed.tsv")
        with open(path, "w") as f:
            f.write("A\tB\n")
            f.write("ONLY_ONE_COL\n")
            f.write("C\tD\n")
        g = from_tsv_edge_list(path)
        assert g.count_nodes() == 4
        assert g.count_edges() == 2

    def test_large_graph_roundtrip(self, tmp_dir, large_graph):
        path = os.path.join(tmp_dir, "test.tsv")
        to_tsv_edge_list(large_graph, path)
        g = from_tsv_edge_list(path, has_header=True, weight_col="weight")
        assert g.count_nodes() == 50
        assert g.count_edges() == 50

    def test_biogrid_style_format(self, tmp_dir):
        """Simulate a BioGRID-style TSV with extra columns."""
        path = os.path.join(tmp_dir, "biogrid.tsv")
        with open(path, "w") as f:
            f.write("ID\tGeneA\tGeneB\tScore\tDatabase\n")
            f.write("1\tBRCA1\tTP53\t0.95\tBioGRID\n")
            f.write("2\tTP53\tMDM2\t0.87\tBioGRID\n")
            f.write("3\tATM\tCHEK2\t0.92\tBioGRID\n")
        g = from_tsv_edge_list(
            path, source_col="GeneA", target_col="GeneB", weight_col="Score", has_header=True
        )
        assert g.count_nodes() == 5
        assert g.count_edges() == 3
        assert g.is_weighted()
        assert g.has_node("BRCA1")
        assert g.has_node("MDM2")

    def test_string_style_format(self, tmp_dir):
        """Simulate a STRING database export."""
        path = os.path.join(tmp_dir, "string.tsv")
        with open(path, "w") as f:
            f.write("protein1\tprotein2\tcombined_score\n")
            f.write("9606.ENSP00000269305\t9606.ENSP00000350283\t999\n")
            f.write("9606.ENSP00000269305\t9606.ENSP00000344818\t990\n")
        g = from_tsv_edge_list(
            path,
            source_col="protein1",
            target_col="protein2",
            weight_col="combined_score",
            has_header=True,
        )
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        assert g.is_weighted()

    def test_weight_parse_error_ignored(self, tmp_dir):
        path = os.path.join(tmp_dir, "badweight.tsv")
        with open(path, "w") as f:
            f.write("A\tB\tNOTANUMBER\n")
            f.write("B\tC\t2.5\n")
        g = from_tsv_edge_list(path, weight_col=2)
        assert g.count_edges() == 2


# =============================================================================
# TSV Edge List — to_tsv_edge_list
# =============================================================================


class TestToTSVEdgeList:
    def test_creates_file(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "out.tsv")
        to_tsv_edge_list(unweighted_triangle, path)
        assert os.path.exists(path)

    def test_header_present(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "hdr.tsv")
        to_tsv_edge_list(unweighted_triangle, path, header=True)
        with open(path) as f:
            first_line = f.readline().strip()
        assert "source" in first_line
        assert "target" in first_line

    def test_no_header(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "nohdr.tsv")
        to_tsv_edge_list(unweighted_triangle, path, header=False)
        with open(path) as f:
            first_line = f.readline().strip()
        assert "source" not in first_line

    def test_weight_column_when_weighted(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "wt.tsv")
        to_tsv_edge_list(weighted_triangle, path)
        with open(path) as f:
            first_line = f.readline().strip()
        assert "weight" in first_line

    def test_no_weight_column_when_unweighted(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "nowt.tsv")
        to_tsv_edge_list(unweighted_triangle, path)
        with open(path) as f:
            first_line = f.readline().strip()
        assert "weight" not in first_line

    def test_custom_delimiter(self, tmp_dir, unweighted_triangle):
        path = os.path.join(tmp_dir, "custom.csv")
        to_tsv_edge_list(unweighted_triangle, path, delimiter=",")
        with open(path) as f:
            first_line = f.readline().strip()
        assert "," in first_line
        assert "\t" not in first_line

    def test_suppress_weight(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "nowt.tsv")
        to_tsv_edge_list(weighted_triangle, path, weight=False)
        with open(path) as f:
            first_line = f.readline().strip()
        assert "weight" not in first_line

    def test_empty_graph_writes(self, tmp_dir, empty_graph):
        path = os.path.join(tmp_dir, "empty.tsv")
        to_tsv_edge_list(empty_graph, path)
        assert os.path.exists(path)

    def test_correct_line_count(self, tmp_dir, weighted_triangle):
        path = os.path.join(tmp_dir, "count.tsv")
        to_tsv_edge_list(weighted_triangle, path, header=True)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 4  # 1 header + 3 edges
