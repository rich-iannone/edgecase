import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chain(n=10):
    g = ec.Graph()
    for i in range(n):
        g.add_node(label=str(i))
    for i in range(n - 1):
        g.add_edge(source=str(i), target=str(i + 1))
    return g


def make_star(n=6):
    g = ec.Graph()
    g.add_node(label="hub")
    for i in range(n):
        g.add_node(label=str(i))
        g.add_edge(source="hub", target=str(i))
    return g


def make_complete(n=5):
    g = ec.Graph()
    labels = [str(i) for i in range(n)]
    for lbl in labels:
        g.add_node(label=lbl)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(source=labels[i], target=labels[j])
    return g


# ===========================================================================
# network_robustness — degree strategy
# ===========================================================================


class TestRobustnessDegree:
    def test_starts_at_full_component(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        assert curve[0] == (0.0, 1.0)

    def test_ends_at_zero(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        assert curve[-1][1] == 0.0

    def test_monotonically_decreasing(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        fracs = [c[1] for c in curve]
        for i in range(1, len(fracs)):
            assert fracs[i] <= fracs[i - 1]

    def test_star_hub_removal_shatters(self):
        g = make_star()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        # After removing hub (highest degree), component drops to 1/7
        assert curve[1][1] <= 1.0 / 7 + 1e-10

    def test_curve_length(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        # n+1 points: (0, initial) + n removal steps
        assert len(curve) == 11

    def test_fraction_removed_increases(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        removed = [c[0] for c in curve]
        for i in range(1, len(removed)):
            assert removed[i] > removed[i - 1]


# ===========================================================================
# network_robustness — betweenness strategy
# ===========================================================================


class TestRobustnessBetweenness:
    def test_starts_at_full_component(self):
        g = make_chain()
        curve = g.network_robustness(strategy="betweenness", return_dict=True)
        assert curve[0] == (0.0, 1.0)

    def test_chain_center_removed_first(self):
        g = make_chain()
        curve = g.network_robustness(strategy="betweenness", return_dict=True)
        # Center node has highest betweenness; removing it splits chain
        assert curve[1][1] < 1.0

    def test_complete_graph_resilient(self):
        g = make_complete(5)
        curve = g.network_robustness(strategy="betweenness", return_dict=True)
        # After removing 1 node from K5, remaining 4 still connected
        assert curve[1][1] == 4.0 / 5.0


# ===========================================================================
# network_robustness — random strategy
# ===========================================================================


class TestRobustnessRandom:
    def test_reproducible_with_seed(self):
        g = make_chain()
        c1 = g.network_robustness(strategy="random", seed=23, return_dict=True)
        c2 = g.network_robustness(strategy="random", seed=23, return_dict=True)
        assert c1 == c2

    def test_different_seeds_differ(self):
        g = make_chain(20)
        c1 = g.network_robustness(strategy="random", seed=1, return_dict=True)
        c2 = g.network_robustness(strategy="random", seed=2, return_dict=True)
        fracs1 = [c[1] for c in c1]
        fracs2 = [c[1] for c in c2]
        assert fracs1 != fracs2

    def test_starts_full_ends_zero(self):
        g = make_chain()
        curve = g.network_robustness(strategy="random", seed=0, return_dict=True)
        assert curve[0][1] == 1.0
        assert curve[-1][1] == 0.0


# ===========================================================================
# network_robustness — steps parameter
# ===========================================================================


class TestRobustnessSteps:
    def test_partial_removal(self):
        g = make_chain(10)
        curve = g.network_robustness(strategy="degree", steps=3, return_dict=True)
        assert len(curve) == 4  # initial + 3 steps

    def test_steps_one(self):
        g = make_chain()
        curve = g.network_robustness(strategy="degree", steps=1, return_dict=True)
        assert len(curve) == 2


# ===========================================================================
# network_robustness — error handling
# ===========================================================================


class TestRobustnessErrors:
    def test_invalid_strategy(self):
        g = make_chain()
        with pytest.raises(ValueError, match="Unknown strategy"):
            g.network_robustness(strategy="invalid", return_dict=True)


# ===========================================================================
# network_robustness — chaining
# ===========================================================================


class TestRobustnessChaining:
    def test_return_self(self):
        g = make_chain()
        result = g.network_robustness(strategy="degree")
        assert result is g

    def test_return_dict(self):
        g = make_chain()
        result = g.network_robustness(strategy="degree", return_dict=True)
        assert isinstance(result, list)

    def test_computations_tracked(self):
        g = make_chain()
        g.network_robustness(strategy="degree")
        found = any(c[0] == "network_robustness" for c in g._computations)
        assert found


# ===========================================================================
# network_robustness — edge cases
# ===========================================================================


class TestRobustnessEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph()
        curve = g.network_robustness(strategy="degree", return_dict=True)
        assert curve == [(0.0, 0.0)]

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        curve = g.network_robustness(strategy="degree", return_dict=True)
        assert curve[0] == (0.0, 1.0)
        assert curve[1] == (1.0, 0.0)

    def test_disconnected_graph(self):
        g = ec.Graph()
        for i in range(4):
            g.add_node(label=str(i))
        g.add_edge(source="0", target="1")
        g.add_edge(source="2", target="3")
        curve = g.network_robustness(strategy="degree", return_dict=True)
        # Initially largest component is 2/4 = 0.5
        assert curve[0][1] == 0.5
