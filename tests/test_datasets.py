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
