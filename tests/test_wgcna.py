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
