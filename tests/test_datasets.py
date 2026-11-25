import gzip
import tempfile
from pathlib import Path

import pytest

from edgecase import (
    get_dataset_info,
    list_available_datasets,
    list_registered_datasets,
    load_dataset,
)


class TestDatasetRegistry:
    """Tests for dataset registry and metadata."""

    def test_list_registered_datasets(self):
        """Test listing all registered datasets."""
        datasets = list_registered_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 14  # At least our bundled datasets
        assert "karate" in datasets

        # Check for small visualization datasets
        small_datasets = ["dolphins", "polbooks", "football", "jazz", "celegans"]
        for ds in small_datasets:
            assert ds in datasets, f"{ds} should be registered"

        # Check for medium visualization datasets (1k-5k nodes)
        medium_datasets = ["power_grid", "pgp", "internet"]
        for ds in medium_datasets:
            assert ds in datasets, f"{ds} should be registered"

    def test_get_dataset_info_valid(self):
        """Test getting metadata for valid dataset."""
        info = get_dataset_info("karate")
        assert isinstance(info, dict)
        assert info["nodes"] == 34
        assert info["edges"] == 78
        assert info["directed"] is False
        assert "description" in info

    def test_get_dataset_info_invalid(self):
        """Test getting metadata for invalid dataset raises error."""
        with pytest.raises(ValueError, match="Dataset .* not found"):
            get_dataset_info("nonexistent_dataset")

    def test_get_dataset_info_without_extension(self):
        """Test getting info works with or without .mtx extension."""
        info1 = get_dataset_info("karate")
        info2 = get_dataset_info("karate.mtx")
        assert info1["nodes"] == info2["nodes"]

    def test_list_available_datasets(self):
        """Test listing datasets in data directory."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)
        # Should include at least karate
        filenames = [d for d in datasets if "karate" in d.lower()]
        assert len(filenames) > 0


class TestDatasetLoading:
    """Tests for loading datasets."""

    def test_load_karate_club(self):
        """Test loading karate club dataset."""
        g = load_dataset("karate")
        assert g.count_nodes() == 34
        assert g.count_edges() == 78
        assert g.is_directed() is False

    def test_load_with_extension(self):
        """Test loading with .mtx extension specified."""
        g = load_dataset("karate.mtx")
        assert g.count_nodes() == 34
        assert g.count_edges() == 78

    def test_load_without_extension(self):
        """Test loading without .mtx extension."""
        g = load_dataset("karate")
        assert g.count_nodes() == 34
        assert g.count_edges() == 78

    def test_load_nonexistent_dataset(self):
        """Test loading nonexistent dataset raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset("this_dataset_does_not_exist")

    def test_metadata_applied_automatically(self):
        """Test that metadata from registry is applied automatically."""
        g = load_dataset("karate")
        # directed flag from registry should be applied
        assert g.is_directed() is False

    def test_override_directed_flag(self):
        """Test overriding directed flag from registry."""
        # Karate is undirected in registry, but we can override
        g = load_dataset("karate", directed=True)
        assert g.is_directed() is True


class TestCompressedDatasetLoading:
    """Tests for loading compressed .mtx.gz datasets."""

    def test_load_from_compressed_file(self):
        """Test loading from .mtx.gz file when .mtx doesn't exist."""
        # This test assumes karate.mtx.gz exists in data/
        import os
        from edgecase.datasets import DATA_DIR

        mtx_path = DATA_DIR / "karate.mtx"
        gz_path = DATA_DIR / "karate.mtx.gz"

        # Skip if compressed file doesn't exist
        if not gz_path.exists():
            pytest.skip("karate.mtx.gz not found")

        # Temporarily rename .mtx if it exists
        mtx_existed = mtx_path.exists()
        if mtx_existed:
            backup_path = mtx_path.with_suffix(".mtx.backup")
            os.rename(mtx_path, backup_path)

        try:
            # Should load from .gz file
            g = load_dataset("karate")
            assert g.count_nodes() == 34
            assert g.count_edges() == 78
        finally:
            # Restore original
            if mtx_existed:
                os.rename(backup_path, mtx_path)

    def test_compressed_file_produces_same_result(self):
        """Test that compressed and uncompressed files produce same graph."""
        from edgecase.datasets import DATA_DIR

        gz_path = DATA_DIR / "karate.mtx.gz"
        if not gz_path.exists():
            pytest.skip("karate.mtx.gz not found")

        mtx_path = DATA_DIR / "karate.mtx"
        if not mtx_path.exists():
            pytest.skip(
                "Test requires uncompressed karate.mtx file (only compressed version available)"
            )

        # Load from uncompressed
        g1 = load_dataset("karate")

        # Create temp compressed version and test
        import os

        backup_path = mtx_path.with_suffix(".mtx.backup")
        os.rename(mtx_path, backup_path)

        try:
            g2 = load_dataset("karate")
            assert g1.count_nodes() == g2.count_nodes()
            assert g1.count_edges() == g2.count_edges()
            assert g1.is_directed() == g2.is_directed()
        finally:
            os.rename(backup_path, mtx_path)


class TestCustomMatrixMarketFiles:
    """Tests for loading custom .mtx files."""

    def test_load_custom_mtx_file(self):
        """Test loading a custom .mtx file from data directory."""
        # Create a temporary simple graph in MTX format
        from edgecase.datasets import DATA_DIR

        test_file = DATA_DIR / "test_custom.mtx"
        test_content = """%%MatrixMarket matrix coordinate pattern symmetric
3 3 3
1 2
2 3
1 3
"""
        test_file.write_text(test_content)

        try:
            g = load_dataset("test_custom")
            assert g.count_nodes() == 3
            assert g.count_edges() == 3
            assert g.is_directed() is False
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_load_custom_compressed_mtx(self):
        """Test loading custom compressed .mtx.gz file."""
        from edgecase.datasets import DATA_DIR

        test_content = b"""%%MatrixMarket matrix coordinate pattern symmetric
4 4 4
1 2
2 3
3 4
1 4
"""
        test_file_gz = DATA_DIR / "test_custom_compressed.mtx.gz"

        with gzip.open(test_file_gz, "wb") as f:
            f.write(test_content)

        try:
            g = load_dataset("test_custom_compressed")
            assert g.count_nodes() == 4
            assert g.count_edges() == 4
        finally:
            if test_file_gz.exists():
                test_file_gz.unlink()


class TestRegistryMetadata:
    """Tests for dataset registry metadata accuracy."""

    def test_bundled_datasets_marked_correctly(self):
        """Test that bundled datasets are marked with correct location."""
        info = get_dataset_info("karate")
        assert info.get("location") == "bundled"

    def test_remote_datasets_marked_correctly(self):
        """Test that remote datasets are marked with correct location."""
        # roadnet_pa should be marked as remote
        info = get_dataset_info("roadnet_pa")
        assert info.get("location") == "remote"
        assert "compressed" in info  # Should have compressed filename

    def test_compressed_bundled_metadata(self):
        """Test metadata for compressed bundled datasets."""
        # Check medium-sized datasets that should be bundled compressed
        for name in ["ca_grqc", "ca_condmat", "ca_hepph"]:
            info = get_dataset_info(name)
            assert info.get("location") == "bundled"
            # Should have compressed_bundled or compressed_kb
            assert "compressed_bundled" in info or "compressed_kb" in info, (
                f"{name} missing compression metadata"
            )

    def test_all_registered_have_basic_metadata(self):
        """Test that all registered datasets have required metadata."""
        required_fields = ["filename", "name", "nodes", "edges", "directed"]

        for name in list_registered_datasets():
            info = get_dataset_info(name)
            for field in required_fields:
                assert field in info, f"{name} missing required field: {field}"


class TestDatasetLoadingPerformance:
    """Performance-related tests for dataset loading."""

    def test_compressed_load_is_fast(self):
        """Test that loading compressed files is reasonably fast."""
        import time

        from edgecase.datasets import DATA_DIR

        gz_path = DATA_DIR / "karate.mtx.gz"
        if not gz_path.exists():
            pytest.skip("karate.mtx.gz not found")

        # Load and time it
        start = time.perf_counter()
        g = load_dataset("karate")
        elapsed = time.perf_counter() - start

        # Should load in under 100ms even from compressed
        assert elapsed < 0.1, f"Loading took {elapsed:.3f}s, too slow"
        assert g.count_nodes() == 34

    def test_multiple_loads_dont_fail(self):
        """Test that loading the same dataset multiple times works."""
        for _ in range(3):
            g = load_dataset("karate")
            assert g.count_nodes() == 34


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_case_sensitive_names(self):
        """Test that dataset names work consistently."""
        # Should work
        g = load_dataset("karate")
        assert g is not None

        # Test that registry lookup is case-sensitive
        # but file system might not be (macOS is case-insensitive by default)
        # So we just test that lowercase works
        assert g.count_nodes() == 34

    def test_empty_filename(self):
        """Test that empty filename raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_dataset("")

    def test_special_characters_in_name(self):
        """Test handling of special characters."""
        # Should handle underscores (ca_grqc)
        if "ca_grqc" in list_registered_datasets():
            info = get_dataset_info("ca_grqc")
            assert info is not None

    def test_weighted_parameter(self):
        """Test weighted parameter is passed through."""
        g_unweighted = load_dataset("karate", weighted=False)
        g_weighted = load_dataset("karate", weighted=True)

        # Both should load successfully
        assert g_unweighted.count_nodes() == 34
        assert g_weighted.count_nodes() == 34


class TestLabeledClassics:
    """The classic networks built with their published labels and attributes."""

    @pytest.mark.parametrize(
        "loader, registry_name, size, directed, weighted",
        [
            ("dolphins", "dolphins", (62, 159), False, False),
            ("football", "football", (115, 613), False, False),
            ("polbooks", "polbooks", (105, 441), False, False),
            ("celegans_neural", "celegans_neural", (297, 2345), True, True),
        ],
    )
    def test_matches_bundled_file(self, loader, registry_name, size, directed, weighted):
        import edgecase as ec
        from edgecase.steps import KIND_BUILD_NODES

        g = getattr(ec, loader)()
        assert (g.count_nodes(), g.count_edges()) == size
        assert g.is_directed() is directed
        assert g.is_weighted() is weighted
        assert len(g.steps) == 1
        assert g.steps[0].kind == KIND_BUILD_NODES
        assert g.steps[0].code == f"{loader}()"
        info = get_dataset_info(registry_name)
        assert (info["nodes"], info["edges"], info["directed"]) == (*size, directed)

        # Node i of the file is the i-th node, so the edges are the file's edges
        plain = load_dataset(registry_name, weighted=weighted)
        assert sorted(g.get_edge_df().select("source", "target").rows()) == sorted(
            plain.get_edge_df().select("source", "target").rows()
        )
        labels = g.get_node_df()["label"].to_list()
        assert len(set(labels)) == len(labels)

    def test_football_conferences(self):
        from collections import Counter

        import edgecase as ec

        g = ec.football()
        conference = g.get_node_attrs("conference")
        assert conference[g.get_id_from_label("Michigan")] == "Big Ten"
        assert conference[g.get_id_from_label("FloridaState")] == "Atlantic Coast"
        counts = Counter(conference.values())
        assert len(counts) == 12
        assert counts["Independents"] == 5
        assert counts["Southeastern"] == 12
        # Most games are within a conference
        within = sum(conference[s] == conference[t] for _, s, t in g.get_edges_as_tuples())
        assert within / g.count_edges() > 0.6

    def test_polbooks_leanings(self):
        from collections import Counter

        import edgecase as ec

        g = ec.polbooks()
        leaning = g.get_node_attrs("leaning")
        assert Counter(leaning.values()) == {"conservative": 49, "liberal": 43, "neutral": 13}
        assert leaning[g.get_id_from_label("Stupid White Men")] == "liberal"
        assert g.assortativity_nominal("leaning") == pytest.approx(0.723, abs=1e-3)

    def test_dolphins_names(self):
        import edgecase as ec

        g = ec.dolphins()
        assert g.get_label_from_id(1) == "Beak"
        assert g.get_degree("Grin") == 12

    def test_celegans_neural_weights(self):
        import edgecase as ec

        g = ec.celegans_neural()
        weights = g.get_edge_df()["weight"]
        assert (weights.min(), weights.max()) == (1, 70)
        assert g.get_label_from_id(2) == "51"
        assert g.count_weakly_connected_components() == 1

    def test_florentine_ties(self):
        from collections import Counter

        import edgecase as ec

        marriage = ec.florentine_families()
        business = ec.florentine_families(ties="business")
        both = ec.florentine_families(ties="all")
        assert [g.count_edges() for g in (marriage, business, both)] == [20, 15, 27]
        labels = marriage.get_node_df()["label"].to_list()
        for g in (business, both):
            assert g.get_node_df()["label"].to_list() == labels
        assert Counter(marriage.get_edge_df()["rel"]) == {"marriage": 20}
        assert Counter(business.get_edge_df()["rel"]) == {"business": 15}
        assert Counter(both.get_edge_df()["rel"]) == {"marriage": 12, "business": 7, "both": 8}
        isolates = sorted(business.get_label_from_id(n) for n in business.isolates())
        assert isolates == ["Acciaiuoli", "Albizzi", "Ridolfi", "Strozzi"]
        assert business.steps[0].code == "florentine_families(ties='business')"
        with pytest.raises(ValueError, match="ties must be"):
            ec.florentine_families(ties="kinship")

    def test_karate_leaders_contracted(self):
        import edgecase as ec

        karate = ec.karate_club()
        leaders = karate.contract_nodes("0", "33")
        assert (karate.get_degree("0"), karate.get_degree("33")) == (16, 17)
        assert leaders.get_degree("0") == 29
        merged = leaders.get_id_from_label("0")
        reach = set(leaders.get_neighbors(merged)) | {merged}
        outside = [leaders.get_label_from_id(n) for n in leaders.get_node_ids() if n not in reach]
        assert sorted(outside) == ["16", "24", "25"]

    def test_karate_communities_contracted(self):
        from collections import Counter

        import edgecase as ec

        karate = ec.karate_club()
        faction = karate.get_node_attrs("faction")
        assert sum(faction[u] != faction[v] for _, u, v in karate.get_edges_as_tuples()) == 10

        membership = karate.cmty_louvain(return_dict=True)
        partition = [
            [node for node, c in membership.items() if c == community]
            for community in sorted(set(membership.values()))
        ]
        assert [len(block) for block in partition] == [12, 5, 11, 6]
        assert karate.get_id_from_label("0") in partition[0]
        assert karate.get_id_from_label("33") in partition[2]

        # The administrator's community as one node
        club = karate.contract_nodes_multi(partition[2], self_loops=False)
        merged = sorted(partition[2])[0]
        assert club.count_nodes() == 24
        assert (club.get_label_from_id(merged), club.get_degree(merged)) == ("8", 10)

        # Friendships between communities
        block = {node: i for i, members in enumerate(partition) for node in members}
        crossing = Counter(
            frozenset((block[u], block[v]))
            for _, u, v in karate.get_edges_as_tuples()
            if block[u] != block[v]
        )
        assert crossing == {
            frozenset((0, 1)): 4,
            frozenset((0, 2)): 7,
            frozenset((0, 3)): 3,
            frozenset((2, 3)): 7,
        }
        assert karate.quotient_graph(partition).count_edges() == 4

    def test_karate_degree_summaries(self):
        import edgecase as ec

        karate = ec.karate_club()
        assert karate.aggregate_degree("mean") == pytest.approx(78 * 2 / 34)
        others = karate.aggregate_degree("mean", conditions=lambda n: n.label not in {"0", "33"})
        assert others == pytest.approx((78 * 2 - 16 - 17) / 32)
        medians = {
            f: karate.aggregate_degree("median", conditions=lambda n: n.attributes["faction"] == f)
            for f in [1, 2]
        }
        assert medians == {1: 3.5, 2: 3.0}
        assert ec.Graph().add_gnm_graph(20, 35, seed=23).aggregate_degree("mean") == 3.5

    def test_karate_leaders_renamed(self):
        import polars as pl

        import edgecase as ec

        karate = ec.karate_club()
        degrees = karate.degrees(return_dict=True)
        karate.relabel_nodes({"0": "Mr. Hi", "33": "John A."})
        leaders = karate.get_node_df().filter(pl.col("role") != "Member")
        assert leaders["label"].to_list() == ["Mr. Hi", "John A."]
        assert leaders["role"].to_list() == ["Instructor", "Administrator"]
        assert karate.get_label_from_id(max(degrees, key=degrees.get)) == "John A."
        with pytest.raises(ValueError, match="already used"):
            karate.relabel_nodes({"1": "Mr. Hi"})
        assert karate.get_label_from_id(2) == "1"

        n = karate.count_nodes()
        assert (n, n * (n - 1) // 2, karate.count_edges()) == (34, 561, 78)
        assert karate.density() == pytest.approx(78 / 561)

    def test_karate_data_frames(self):
        import polars as pl

        import edgecase as ec

        karate = ec.karate_club()
        sides = karate.get_node_df().select("label", "faction")
        crossing = (
            karate.get_edge_df(labels=True)
            .join(sides.rename({"label": "source", "faction": "from"}), on="source")
            .join(sides.rename({"label": "target", "faction": "to"}), on="target")
            .filter(pl.col("from") != pl.col("to"))
        )
        assert crossing.height == 10 and karate.count_edges() == 78
        per_member = dict(pl.concat([crossing["source"], crossing["target"]]).value_counts().rows())
        assert per_member.pop("2") == 5 and max(per_member.values()) == 2

        karate.betweenness_centrality(attr_name="betweenness")
        karate.pagerank(attr_name="pagerank")
        top = karate.get_node_df().sort("betweenness", descending=True).head(5)
        assert top["label"].to_list() == ["0", "33", "32", "2", "31"]
        member_31 = top.row(4, named=True)
        assert member_31["degree"] == 6
        assert member_31["pagerank"] < 0.4 * top["pagerank"].head(2).min()

        karate.cmty_louvain(attr_name="community")
        karate.colorize_node_attrs("community", palette="Set2")
        nodes = karate.get_node_df()
        assert (
            nodes.group_by("community").agg(pl.col("color").n_unique())["color"].to_list()
            == [1] * 4
        )
        assert sorted(nodes.group_by("community").len()["len"].to_list()) == [5, 6, 11, 12]
        mixed = nodes.group_by("community").agg(pl.col("faction").n_unique() > 1)
        assert mixed["faction"].sum() == 1
        mixed_id = mixed.filter(pl.col("faction"))["community"].item()
        mixed_members = nodes.filter(pl.col("community") == mixed_id)
        assert mixed_members.filter(pl.col("faction") == 2)["label"].to_list() == ["9"]
        assert (mixed_members["faction"] == 1).sum() == 11

    def test_karate_combined_score(self):
        import polars as pl

        import edgecase as ec

        karate = ec.karate_club()
        ties = karate.degrees(return_dict=True)
        between = karate.betweenness_centrality(return_dict=True)
        assert max(ties.values()) == 17 and round(max(between.values()), 2) == 0.44
        karate.rescale_node_attrs(values=ties, node_attr_to="ties", range=[0, 1])
        karate.rescale_node_attrs(values=between, node_attr_to="between", range=[0, 1])
        nodes = karate.get_node_df().with_columns(score=(pl.col("ties") + pl.col("between")) / 2)
        ranked = nodes.sort("score", descending=True)["label"].to_list()
        assert ranked[:6] == ["0", "33", "32", "2", "31", "1"]
        degree = dict(nodes.select("label", "degree").rows())
        assert (degree["1"], degree["31"]) == (9, 6)

    def test_karate_without_leaders(self):
        import edgecase as ec

        members = ec.karate_club().delete_node("0").delete_node("33")
        assert not members.is_connected()
        pieces = sorted(members.weakly_connected_components(), key=len)
        assert [sorted(members.get_label_from_id(n) for n in c) for c in pieces[:2]] == [
            ["11"],
            ["10", "16", "4", "5", "6"],
        ]
        assert len(pieces[2]) == 26

    def test_les_miserables_degrees(self):
        import polars as pl

        import edgecase as ec

        novel = ec.les_miserables()
        met = novel.degrees(return_dict=True)
        top = sorted(met, key=met.get, reverse=True)[:5]
        assert [(novel.get_label_from_id(c), met[c]) for c in top] == [
            ("Valjean", 36),
            ("Gavroche", 22),
            ("Marius", 19),
            ("Javert", 17),
            ("Thenardier", 16),
        ]

        # The stored snapshot keeps the old degrees; the live column follows the edit
        novel.degrees(attr_name="met")
        novel.delete_node("Valjean")
        alone = novel.get_node_df().filter(pl.col("degree") == 0)
        assert sorted(alone["label"]) == ["Gervais", "Isabeau", "Labarre", "MmeDeR", "Scaufflaire"]
        assert alone["met"].to_list() == [1] * 5

    def test_les_miserables_degree_distribution(self):
        import edgecase as ec

        novel = ec.les_miserables()
        chance = ec.Graph().add_gnm_graph(77, 254, seed=2)
        counts = novel.degree_distribution()
        assert (counts[1], max(counts), counts[36]) == (17, 36, 1)
        assert sum(n for d, n in counts.items() if d > 13) == 7
        random_counts = chance.degree_distribution()
        assert (min(random_counts), max(random_counts)) == (2, 13)
        assert novel.average_degree() == pytest.approx(chance.average_degree())

        students = [
            "Enjolras", "Combeferre", "Prouvaire", "Feuilly", "Courfeyrac",
            "Bahorel", "Bossuet", "Joly", "Grantaire",
        ]  # fmt: skip
        society = novel.subgraph([novel.get_id_from_label(s) for s in students])
        assert novel.average_degree() == pytest.approx(6.597, abs=1e-3)
        assert society.average_degree() == 8.0


class TestMatrixMarketSizeLine:
    """The size line is the first line after the comments, whatever its shape."""

    def _read(self, text, **kwargs):
        from edgecase.io import from_matrix_market

        with tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False) as fh:
            fh.write(text)
        try:
            return from_matrix_market(fh.name, **kwargs)
        finally:
            Path(fh.name).unlink()

    def test_integer_entries_are_edges(self):
        g = self._read("%%MatrixMarket matrix coordinate integer general\n3 3 2\n1 2 4\n2 3 1\n")
        assert g.count_edges() == 2
        assert sorted(g.get_edge_df()["weight"].to_list()) == [1.0, 4.0]

    def test_size_line_without_count_is_not_an_edge(self):
        g = self._read("%%MatrixMarket matrix coordinate pattern symmetric\n% c\n3 3\n2 1\n3 2\n")
        assert g.count_edges() == 2
        assert g.is_simple()

    def test_jazz_and_celegans_have_no_phantom_loop(self):
        assert load_dataset("jazz").count_edges() == 2742
        assert load_dataset("celegans").count_edges() == 2040
