import math

import pytest
from edgecase import Graph


class TestSpectrumBasics:
    """Basic spectrum computation tests."""

    def test_spectrum_complete_graph(self):
        """Complete graph K_n has known spectrum."""
        n = 5
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(f"v{i}", f"v{j}")

        spec = g.get_spectrum(which="all")
        values = sorted(spec["values"], reverse=True)

        # K_n has eigenvalues: n-1, -1, -1, ..., -1
        assert values[0] == pytest.approx(n - 1)
        for i in range(1, n):
            assert values[i] == pytest.approx(-1.0)

        # Spectral gap = (n-1) - (-1) = n
        assert spec["spectral_gap"] == pytest.approx(n)

    def test_spectrum_path_graph(self):
        """Path graph P_n has known spectrum."""
        n = 5
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n - 1):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_spectrum(which="all")
        values = spec["values"]

        # Path P_n: eigenvalues are 2*cos(π*k/(n+1)) for k=1,...,n
        expected = [2 * math.cos(math.pi * k / (n + 1)) for k in range(1, n + 1)]
        expected_sorted = sorted(expected, reverse=True)
        values_sorted = sorted(values, reverse=True)

        for i in range(n):
            assert values_sorted[i] == pytest.approx(expected_sorted[i], abs=1e-6)

    def test_spectrum_cycle_graph(self):
        """Cycle graph C_n has known spectrum."""
        n = 6
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n):
            g.add_edge(f"v{i}", f"v{(i + 1) % n}")

        spec = g.get_spectrum(which="all")
        values = spec["values"]

        # Cycle C_n: eigenvalues are 2*cos(2π*k/n) for k=0,...,n-1
        expected = [2 * math.cos(2 * math.pi * k / n) for k in range(n)]
        expected_sorted = sorted(expected, reverse=True)
        values_sorted = sorted(values, reverse=True)

        for i in range(n):
            assert values_sorted[i] == pytest.approx(expected_sorted[i], abs=1e-6)

    def test_spectrum_star_graph(self):
        """Star graph S_n has known spectrum."""
        n = 5  # Central node + 4 leaves
        g = Graph(directed=False)
        g.add_node("center")
        for i in range(n - 1):
            g.add_node(f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        spec = g.get_spectrum(which="all")
        values = sorted(spec["values"], reverse=True)

        # Star S_n: {√(n-1), 0, ..., 0, -√(n-1)}
        sqrt_n_minus_1 = math.sqrt(n - 1)
        assert values[0] == pytest.approx(sqrt_n_minus_1)
        assert values[-1] == pytest.approx(-sqrt_n_minus_1)
        # Middle values should be close to 0
        for i in range(1, n - 1):
            assert abs(values[i]) < 1e-6

    def test_spectrum_single_edge(self):
        """Graph with single edge has spectrum {1, -1}."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        spec = g.get_spectrum(which="all")
        values = sorted(spec["values"], reverse=True)

        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(-1.0)
        assert spec["spectral_gap"] == pytest.approx(2.0)


class TestSpectrumLaplacian:
    """Test Laplacian spectrum computation."""

    def test_laplacian_complete_graph(self):
        """Complete graph K_n Laplacian has known spectrum."""
        n = 5
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(f"v{i}", f"v{j}")

        spec = g.get_laplacian_spectrum(which="all")
        values = sorted(spec["values"])

        # K_n Laplacian: {0, n, n, ..., n}
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        for i in range(1, n):
            assert values[i] == pytest.approx(n, abs=1e-4)

        # Algebraic connectivity = n
        assert spec["algebraic_connectivity"] == pytest.approx(n, abs=1e-4)

    def test_laplacian_path_graph(self):
        """Path graph Laplacian spectrum."""
        n = 5
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n - 1):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="all")
        values = sorted(spec["values"])

        # Path P_n: 2 - 2*cos(π*k/n) for k=0,...,n-1
        expected = [2 - 2 * math.cos(math.pi * k / n) for k in range(n)]
        expected_sorted = sorted(expected)

        for i in range(n):
            assert values[i] == pytest.approx(expected_sorted[i], abs=1e-5)

        # Algebraic connectivity λ₂ = 2(1 - cos(π/n))
        expected_lambda2 = 2 * (1 - math.cos(math.pi / n))
        assert spec["algebraic_connectivity"] == pytest.approx(expected_lambda2, abs=1e-6)

    def test_laplacian_disconnected_graph(self):
        """Disconnected graph has multiple zero eigenvalues."""
        g = Graph(directed=False)
        # Two disconnected triangles
        for i in range(3):
            g.add_node(f"a{i}")
        for i in range(3):
            g.add_node(f"b{i}")

        g.add_edge("a0", "a1")
        g.add_edge("a1", "a2")
        g.add_edge("a2", "a0")

        g.add_edge("b0", "b1")
        g.add_edge("b1", "b2")
        g.add_edge("b2", "b0")

        spec = g.get_laplacian_spectrum(which="all")
        values = sorted(spec["values"])

        # Two components → at least two zero eigenvalues
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        assert values[1] == pytest.approx(0.0, abs=1e-6)

    def test_laplacian_normalized(self):
        """Normalized Laplacian eigenvalues in [0, 2]."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(f"v{i}")
        for i in range(4):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="all", normalized=True)
        values = spec["values"]

        # All eigenvalues should be in [0, 2]
        for v in values:
            assert -1e-9 <= v <= 2.0 + 1e-6

        # First eigenvalue should be 0
        assert min(values) == pytest.approx(0.0, abs=1e-6)


class TestSpectrumEdgeCases:
    """Test edge cases and special scenarios."""

    def test_spectrum_single_node(self):
        """Single node has eigenvalue 0."""
        g = Graph(directed=False)
        g.add_node("A")

        spec = g.get_spectrum(which="all")
        assert len(spec["values"]) == 1
        assert spec["values"][0] == pytest.approx(0.0)
        assert spec["spectral_gap"] is None  # Only one eigenvalue

    def test_spectrum_empty_graph_error(self):
        """Empty graph should raise error."""
        g = Graph(directed=False)

        with pytest.raises(ValueError, match="empty graph"):
            g.get_spectrum(which="all")

    def test_laplacian_empty_graph_error(self):
        """Empty graph should raise error."""
        g = Graph(directed=False)

        with pytest.raises(ValueError, match="empty graph"):
            g.get_laplacian_spectrum(which="all")

    def test_spectrum_isolated_nodes(self):
        """Isolated nodes have eigenvalue 0."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")

        spec = g.get_spectrum(which="all")
        # All isolated nodes → all eigenvalues = 0
        for v in spec["values"]:
            assert v == pytest.approx(0.0)

    def test_spectrum_bipartite_symmetric(self):
        """Bipartite graph has symmetric spectrum."""
        g = Graph(directed=False)
        # Create bipartite graph K_{2,3}
        for i in range(2):
            g.add_node(f"left{i}")
        for i in range(3):
            g.add_node(f"right{i}")
        for i in range(2):
            for j in range(3):
                g.add_edge(f"left{i}", f"right{j}")

        spec = g.get_spectrum(which="all")
        values = sorted(spec["values"], reverse=True)

        # Bipartite: if λ is eigenvalue, so is -λ
        # Check λ_max ≈ -λ_min
        assert abs(values[0] + values[-1]) < 1e-6


class TestSpectrumPartialComputation:
    """Test computing subset of eigenvalues."""

    def test_spectrum_largest_k(self):
        """Compute only largest k eigenvalues."""
        n = 10
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n - 1):
            g.add_edge(f"v{i}", f"v{i + 1}")

        k = 3
        spec = g.get_spectrum(which="largest", k=k)

        assert len(spec["values"]) == k
        # Largest eigenvalues should be in descending order
        values = spec["values"]
        for i in range(k - 1):
            assert values[i] >= values[i + 1]

    def test_spectrum_smallest_k(self):
        """Compute only smallest k eigenvalues."""
        n = 10
        g = Graph(directed=False)
        for i in range(n):
            g.add_node(f"v{i}")
        for i in range(n - 1):
            g.add_edge(f"v{i}", f"v{i + 1}")

        k = 3
        spec = g.get_spectrum(which="smallest", k=k)

        assert len(spec["values"]) == k
        # Should still be in descending order
        values = spec["values"]
        for i in range(k - 1):
            assert values[i] >= values[i + 1]

    def test_laplacian_smallest_default(self):
        """Default k for Laplacian should work."""
        g = Graph(directed=False)
        for i in range(10):
            g.add_node(f"v{i}")
        for i in range(9):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="smallest")
        # Default k = min(6, n-1) = 6
        assert len(spec["values"]) == 6


class TestSpectrumProperties:
    """Test mathematical properties of spectra."""

    def test_spectrum_trace_zero(self):
        """Sum of adjacency eigenvalues equals trace (should be 0 for simple graphs)."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(f"v{i}")
        g.add_edge("v0", "v1")
        g.add_edge("v1", "v2")
        g.add_edge("v2", "v3")
        g.add_edge("v3", "v4")

        spec = g.get_spectrum(which="all")
        trace = sum(spec["values"])

        # Trace of adjacency matrix = 0 (no self-loops)
        assert trace == pytest.approx(0.0, abs=1e-6)

    def test_laplacian_first_eigenvalue_zero(self):
        """First Laplacian eigenvalue is always 0 for connected graph."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(f"v{i}")
        for i in range(4):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="all")
        values = sorted(spec["values"])

        assert values[0] == pytest.approx(0.0, abs=1e-6)

    def test_spectral_gap_positive_connected(self):
        """Connected graph has positive spectral gap."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(f"v{i}")
        for i in range(4):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_spectrum(which="all")

        # Connected graph → spectral gap > 0
        assert spec["spectral_gap"] > 0

    def test_algebraic_connectivity_positive_connected(self):
        """Connected graph has positive algebraic connectivity."""
        g = Graph(directed=False)
        for i in range(5):
            g.add_node(f"v{i}")
        for i in range(4):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="smallest", k=2)

        # Connected graph → λ₂ > 0
        assert spec["algebraic_connectivity"] > 0


class TestSpectrumDirectedGraphs:
    """Test spectrum on directed graphs."""

    def test_spectrum_directed_asymmetric(self):
        """Directed graph can have complex eigenvalues (real part returned)."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Directed cycle

        spec = g.get_spectrum(which="all")

        # Directed cycle has eigenvalues at the cube roots of unity (complex)
        values = sorted(spec["values"], key=lambda v: (round(v.real, 9), v.imag))
        expected = [complex(-0.5, -math.sqrt(3) / 2), complex(-0.5, math.sqrt(3) / 2), 1 + 0j]
        for v, e in zip(values, expected):
            assert abs(v - e) < 1e-9
        assert spec["vectors"] is None
        assert spec["spectral_gap"] is None

    def test_laplacian_directed(self):
        """Laplacian works on directed graphs."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        # Out-degree Laplacian of A -> B -> C is upper triangular with diagonal (1, 1, 0)
        spec = g.get_laplacian_spectrum(which="all")
        assert [v.real for v in spec["values"]] == pytest.approx([0.0, 1.0, 1.0])
        assert all(abs(v.imag) < 1e-12 for v in spec["values"])
        assert spec["algebraic_connectivity"] is None


class TestSpectrumWeighted:
    """Test spectrum on weighted graphs."""

    def test_spectrum_weighted_edges(self):
        """Weighted edges affect spectrum."""
        g1 = Graph(directed=False)
        g1.add_node("A")
        g1.add_node("B")
        g1.add_node("C")
        g1.add_edge("A", "B", weight=1.0)
        g1.add_edge("B", "C", weight=1.0)

        g2 = Graph(directed=False)
        g2.add_node("A")
        g2.add_node("B")
        g2.add_node("C")
        g2.add_edge("A", "B", weight=2.0)
        g2.add_edge("B", "C", weight=2.0)

        spec1 = g1.get_spectrum(which="all")
        spec2 = g2.get_spectrum(which="all")

        # Weighted graph has different eigenvalues
        assert spec1["values"] != spec2["values"]

    def test_laplacian_weighted_normalized(self):
        """Normalized Laplacian handles weighted graphs."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=1.0)
        g.add_edge("B", "C", weight=3.0)

        spec = g.get_laplacian_spectrum(which="all", normalized=True)

        # Normalized eigenvalues in [0, 2]
        for v in spec["values"]:
            assert -1e-9 <= v <= 2.0 + 1e-6


class TestSpectrumEigenvectors:
    """Test eigenvector computation."""

    def test_spectrum_returns_eigenvectors(self):
        """Eigenvectors should be returned."""
        g = Graph(directed=False)
        for i in range(3):
            g.add_node(f"v{i}")
        g.add_edge("v0", "v1")
        g.add_edge("v1", "v2")

        spec = g.get_spectrum(which="all")

        assert "vectors" in spec
        assert len(spec["vectors"]) == 3
        # Each eigenvector should have length = number of nodes
        for vec in spec["vectors"]:
            assert len(vec) == 3

    def test_laplacian_returns_eigenvectors(self):
        """Laplacian eigenvectors should be returned."""
        g = Graph(directed=False)
        for i in range(3):
            g.add_node(f"v{i}")
        g.add_edge("v0", "v1")
        g.add_edge("v1", "v2")

        spec = g.get_laplacian_spectrum(which="all")

        assert "vectors" in spec
        assert len(spec["vectors"]) == 3
        for vec in spec["vectors"]:
            assert len(vec) == 3

    def test_eigenvector_constant_for_zero_laplacian(self):
        """Eigenvector for λ=0 in Laplacian is constant vector."""
        g = Graph(directed=False)
        for i in range(4):
            g.add_node(f"v{i}")
        for i in range(3):
            g.add_edge(f"v{i}", f"v{i + 1}")

        spec = g.get_laplacian_spectrum(which="all")

        # Find eigenvector corresponding to eigenvalue ≈ 0
        values = spec["values"]
        vectors = spec["vectors"]

        for i, val in enumerate(values):
            if abs(val) < 1e-6:
                # This eigenvector should be approximately constant
                vec = vectors[i]
                # Check all entries have same absolute value (up to sign)
                abs_values = [abs(v) for v in vec]
                avg = sum(abs_values) / len(abs_values)
                for av in abs_values:
                    # All should be close to average
                    assert abs(av - avg) < 0.1


class TestSpectrumInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_spectrum_invalid_which(self):
        """Invalid 'which' parameter should raise error."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="which"):
            g.get_spectrum(which="invalid")

    def test_laplacian_invalid_which(self):
        """Invalid 'which' parameter should raise error."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="which"):
            g.get_laplacian_spectrum(which="invalid")

    def test_spectrum_k_larger_than_n(self):
        """k larger than n should return all eigenvalues."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        spec = g.get_spectrum(which="largest", k=10)

        # Should return all 3 eigenvalues
        assert len(spec["values"]) == 3
