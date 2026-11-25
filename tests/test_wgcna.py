import pytest
import random
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_weighted_random(n=30, p=0.3, seed=23):
    g = ec.Graph(weighted=True)
    for i in range(n):
        g.add_node(label=str(i))
    rng = random.Random(seed)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(source=str(i), target=str(j), weight=rng.random())
    return g


def make_unweighted(n=20):
    g = ec.Graph()
    for i in range(n):
        g.add_node(label=str(i))
    rng = random.Random(99)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.3:
                g.add_edge(source=str(i), target=str(j))
    return g


# ===========================================================================
# soft_threshold_power — basic functionality
# ===========================================================================


class TestSoftThresholdBasic:
    def test_returns_all_keys(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(return_dict=True)

        assert "powers" in result
        assert "signed_r_squared" in result
        assert "slope" in result
        assert "mean_connectivity" in result
        assert "r_squared_cutoff" in result
        assert "recommended_power" in result

    def test_default_powers_length(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(return_dict=True)

        assert len(result["powers"]) == 15  # 1-10 + 12,14,16,18,20

    def test_custom_powers(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(powers=[2.0, 4.0, 6.0], return_dict=True)

        assert len(result["powers"]) == 3
        assert result["powers"] == [2.0, 4.0, 6.0]

    def test_mean_connectivity_decreases(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(return_dict=True)
        mk = result["mean_connectivity"]

        # Higher power reduces connectivity (weights < 1 raised to higher power → smaller)
        assert mk[0] >= mk[-1]

    def test_r_squared_cutoff_stored(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(r_squared_cutoff=0.9, return_dict=True)

        assert result["r_squared_cutoff"] == 0.9


# ===========================================================================
# soft_threshold_power — recommendation
# ===========================================================================


class TestSoftThresholdRecommendation:
    def test_recommendation_is_lowest_passing(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(r_squared_cutoff=0.0, return_dict=True)

        # With cutoff 0, the first power with non-negative signed R² is recommended
        rec = result["recommended_power"]

        assert rec is not None

        # Verify it's actually the lowest passing
        for p, sr2 in zip(result["powers"], result["signed_r_squared"]):
            if sr2 >= 0.0:
                assert rec == p
                break

    def test_impossible_cutoff_returns_none(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(r_squared_cutoff=1.1, return_dict=True)

        assert result["recommended_power"] is None


# ===========================================================================
# soft_threshold_power — unweighted graph
# ===========================================================================


class TestSoftThresholdUnweighted:
    def test_works_on_unweighted(self):
        g = make_unweighted()
        result = g.soft_threshold_power(return_dict=True)

        assert len(result["powers"]) == 15
        assert all(isinstance(v, float) for v in result["signed_r_squared"])


# ===========================================================================
# soft_threshold_power — chaining
# ===========================================================================


class TestSoftThresholdChaining:
    def test_return_self(self):
        g = make_weighted_random()
        result = g.soft_threshold_power()

        assert result is g

    def test_return_dict(self):
        g = make_weighted_random()
        result = g.soft_threshold_power(return_dict=True)

        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_weighted_random()
        g.soft_threshold_power()
        found = any(c[0] == "soft_threshold_power" for c in g._computations)

        assert found


# ===========================================================================
# soft_threshold_power — edge cases
# ===========================================================================


class TestSoftThresholdEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph(weighted=True)
        result = g.soft_threshold_power(return_dict=True)

        assert result["recommended_power"] is None

    def test_single_node(self):
        g = ec.Graph(weighted=True).add_node(label="A")
        result = g.soft_threshold_power(return_dict=True)

        assert len(result["powers"]) == 15

    def test_two_nodes_one_edge(self):
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B", weight=0.5)
        result = g.soft_threshold_power(return_dict=True)

        assert "powers" in result


# ---------------------------------------------------------------------------
# Values quoted in the docstring
# ---------------------------------------------------------------------------


class TestSoftThresholdDocstring:
    def test_eight_genes(self):
        g = ec.Graph.from_edges(
            [("G1", "G2", 0.4), ("G1", "G3", 0.2), ("G2", "G3", 0.8), ("G2", "G4", 0.5)]
            + [("G3", "G4", 0.8), ("G4", "G5", 0.5), ("G5", "G6", 0.2), ("G5", "G7", 0.3)]
            + [("G6", "G7", 0.3), ("G6", "G8", 0.2), ("G7", "G8", 0.2)]
        )
        fit = g.soft_threshold_power(powers=[1, 2, 3, 4, 5, 6], return_dict=True)
        assert [round(r, 2) for r in fit["signed_r_squared"]] == [
            -0.35,
            0.1,
            0.67,
            0.9,
            0.93,
            0.93,
        ]
        assert fit["slope"][0] > 0
        assert fit["recommended_power"] == 4.0

    def test_simulated_coexpression(self):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(1)
        factors = rng.normal(size=(3, 40))
        module = [0] * 20 + [1] * 15 + [2] * 10 + [-1] * 15
        expression = np.array(
            [
                (factors[m] * rng.uniform(0.6, 1.5) if m >= 0 else 0) + rng.normal(size=40)
                for m in module
            ]
        )
        correlation = np.abs(np.corrcoef(expression))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 0)
        fit = ec.from_numpy_array(correlation).soft_threshold_power(return_dict=True)
        assert fit["recommended_power"] == 4.0
        assert round(fit["mean_connectivity"][0]) == 11
        assert round(fit["mean_connectivity"][3], 1) == 0.7

        unrelated = np.array(module) == -1

        def ratio(weights):
            connectivity = weights.sum(axis=1)
            return round(float(connectivity[unrelated].mean() / connectivity[~unrelated].mean()), 2)

        assert ratio(correlation) == 0.63
        assert ratio(correlation**4) == 0.1

        weights = correlation**4
        links = (weights > 0.01).sum(axis=1)
        assert links[unrelated].max() == 5
        assert round(links[~unrelated].mean()) == 12
