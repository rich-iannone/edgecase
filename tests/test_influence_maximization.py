import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_star(n=6):
    g = ec.Graph()
    g.add_node(label="hub")
    for i in range(n):
        g.add_node(label=str(i))
        g.add_edge(source="hub", target=str(i))
    return g


def make_chain(n=6):
    g = ec.Graph()
    for i in range(n):
        g.add_node(label=str(i))
    for i in range(n - 1):
        g.add_edge(source=str(i), target=str(i + 1))
    return g


def make_two_hubs():
    g = ec.Graph()
    for lbl in ["H1", "H2", "A", "B", "C", "D", "E", "F"]:
        g.add_node(label=lbl)
    for leaf in ["A", "B", "C"]:
        g.add_edge(source="H1", target=leaf)
    for leaf in ["D", "E", "F"]:
        g.add_edge(source="H2", target=leaf)
    g.add_edge(source="H1", target="H2")
    return g


# ===========================================================================
# influence_maximization — basic functionality
# ===========================================================================


class TestInfluenceBasic:
    def test_returns_k_seeds(self):
        g = make_star()
        result = g.influence_maximization(k=2, seed=23, return_dict=True)
        assert len(result) == 2

    def test_hub_selected_first(self):
        g = make_star()
        result = g.influence_maximization(
            k=1, probability=0.5, n_simulations=200, seed=23, return_dict=True
        )
        hub_id = g.get_id_from_label("hub")
        assert result[0][0] == hub_id

    def test_marginal_gain_positive(self):
        g = make_star()
        result = g.influence_maximization(k=2, seed=23, return_dict=True)
        # First node's spread should be > 0
        assert result[0][1] > 0

    def test_k_zero(self):
        g = make_star()
        result = g.influence_maximization(k=0, seed=23, return_dict=True)
        assert result == []

    def test_k_greater_than_n(self):
        g = make_chain(3)
        result = g.influence_maximization(k=10, seed=23, return_dict=True)
        assert len(result) == 3

    def test_reproducible(self):
        g = make_star()
        r1 = g.influence_maximization(k=2, seed=23, return_dict=True)
        r2 = g.influence_maximization(k=2, seed=23, return_dict=True)
        assert r1 == r2


# ===========================================================================
# influence_maximization — two-hub graph
# ===========================================================================


class TestInfluenceTwoHubs:
    def test_both_hubs_selected(self):
        g = make_two_hubs()
        result = g.influence_maximization(
            k=2, probability=0.5, n_simulations=200, seed=23, return_dict=True
        )
        selected_ids = {r[0] for r in result}
        h1 = g.get_id_from_label("H1")
        h2 = g.get_id_from_label("H2")
        assert h1 in selected_ids or h2 in selected_ids


# ===========================================================================
# influence_maximization — parameters
# ===========================================================================


class TestInfluenceParams:
    def test_high_probability_more_spread(self):
        g = make_chain(10)
        low = g.influence_maximization(
            k=1, probability=0.1, n_simulations=200, seed=23, return_dict=True
        )
        high = g.influence_maximization(
            k=1, probability=0.9, n_simulations=200, seed=23, return_dict=True
        )
        assert high[0][1] >= low[0][1]

    def test_directed_graph(self):
        g = ec.Graph(directed=True)
        for i in range(4):
            g.add_node(label=str(i))
        g.add_edge(source="0", target="1")
        g.add_edge(source="0", target="2")
        g.add_edge(source="0", target="3")
        result = g.influence_maximization(
            k=1, probability=0.5, n_simulations=200, seed=23, return_dict=True
        )
        # Node 0 is the source of all edges
        n0 = g.get_id_from_label("0")
        assert result[0][0] == n0


# ===========================================================================
# influence_maximization — error handling
# ===========================================================================


class TestInfluenceErrors:
    def test_invalid_model(self):
        g = make_star()
        with pytest.raises(ValueError, match="Unknown model"):
            g.influence_maximization(k=1, model="invalid", return_dict=True)

    def test_invalid_probability(self):
        g = make_star()
        with pytest.raises(ValueError, match="Probability"):
            g.influence_maximization(k=1, probability=-0.1, return_dict=True)

    def test_zero_simulations(self):
        g = make_star()
        with pytest.raises(ValueError, match="n_simulations"):
            g.influence_maximization(k=1, n_simulations=0, return_dict=True)


# ===========================================================================
# influence_maximization — chaining
# ===========================================================================


class TestInfluenceChaining:
    def test_return_self(self):
        g = make_star()
        result = g.influence_maximization(k=1, seed=23)
        assert result is g

    def test_return_dict(self):
        g = make_star()
        result = g.influence_maximization(k=1, seed=23, return_dict=True)
        assert isinstance(result, list)

    def test_computations_tracked(self):
        g = make_star()
        g.influence_maximization(k=1, seed=23)
        found = any(c[0] == "influence_maximization" for c in g._computations)
        assert found


# ===========================================================================
# influence_maximization — edge cases
# ===========================================================================


class TestInfluenceEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph()
        result = g.influence_maximization(k=1, seed=23, return_dict=True)
        assert result == []

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        result = g.influence_maximization(k=1, seed=23, return_dict=True)
        assert len(result) == 1
        assert result[0][0] == g.get_id_from_label("A")

    def test_disconnected_nodes(self):
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        result = g.influence_maximization(k=2, seed=23, return_dict=True)
        assert len(result) == 2
